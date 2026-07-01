#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_calibrate.py
=================
Calibracao automatica SIMPLES guiada por importancia SHAP do IDS.

Para cada iteracao:
  1. executa o gerador (synth_generator.build_schedule -> pcap);
  2. extrai as features (pcap_to_csv -> CSV);
  3. compara com o CICIoT2023 (stat_compare);
  4. calcula a loss ponderada por SHAP:
         Loss = sum_f ( SHAP_f * W1_norm_f )
     priorizando primeiro as features mais importantes para o IDS;
  5. perturba automaticamente os parametros do gerador (busca aleatoria
     com refinamento em torno do melhor = "random + greedy");
  6. repete por N iteracoes e salva os melhores parametros.

Reaproveita a estrutura existente (synth_generator, pcap_to_csv, stat_compare),
sem refatoracoes grandes.

Uso:
  python auto_calibrate.py --variant udpplain --iters 40 \
      --real "../CiCIot2023 - Variantes de Interesse/CIC_IoT_Dataset_Unificado_resumido.csv" \
      --base-params params_sprint2.json --out params_auto_udpplain.json

  python auto_calibrate.py --variant greeth --iters 40 \
      --real "..." --base-params params.json --out params_auto_greeth.json

  # Fase 1: payload (Min, Covariance) -> Fase 2: temporal (flow_duration)
  # Fase 3: covariance (dynamic_streams + dispersao de tamanho)
  python auto_calibrate.py --variant greeth --phase payload --iters 30 \
      --base-params params_statistical_v2.json --out params_greeth_payload.json
  python auto_calibrate.py --variant greeth --phase temporal --iters 40 \
      --base-params params_greeth_payload.json --out params_auto_greeth.json
  python auto_calibrate.py --variant greeth --phase covariance --iters 30 \
      --base-params params_auto_greeth.json --out params_auto_greeth.json

  # Fase 4: refinamento global (L = 1.38*W1(flow_duration) + 0.73*W1(Cov) + 0.65*W1(Min))
  python auto_calibrate.py --variant greeth --phase global --iters 25 \
      --refine-step 0.08 --base-params params_auto_greeth_v2.json --out params_auto_greeth_v2.json

  # Fase 5: HMM emissao conjunta — loss hibrida (marginais + correlacao + MMD)
  python auto_calibrate.py --variant greeth --phase latent --iters 40 \
      --base-params params_statistical_v2.json --cal-windows 1500 \
      --real "..." --out params_auto_greeth_latent.json
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import tempfile

import numpy as np

# stat_compare local deve ser importado ANTES de pcap_to_csv, que insere
# ../pcap2csv no sys.path e sombrearia synthgen/stat_compare.py.
_SYNTHGEN_DIR = os.path.dirname(os.path.abspath(__file__))
if _SYNTHGEN_DIR not in sys.path:
    sys.path.insert(0, _SYNTHGEN_DIR)

import pandas as pd

from stat_compare import clean_pair, load_and_align, pick_features, calibration_loss, mmd_rbf, _standardize

import synth_generator as gen
import pcap_to_csv


# --------------------------------------------------------------------------- #
# Importancia SHAP por variante (fornecida pelo IDS).                          #
# A loss prioriza estas features; pesos = SHAP.                                #
# --------------------------------------------------------------------------- #
SHAP_WEIGHTS = {
    "greeth": {"Min": 1.59, "flow_duration": 1.38, "Covariance": 0.73},
    "greip":  {"Min": 1.09, "flow_duration": 1.00, "UDP": 0.98},
    "udpplain": {"UDP": 4.56, "flow_duration": 2.39, "rst_count": 0.83},
    "benign": {
        "rst_count": 3.92,
        "Variance": 2.22,
        "Min": 1.13,
        "flow_duration": 0.50,
        "Covariance": 0.45,
        "Std": 0.35,
    },
}

# Objetivo conjunto GRE-ETH (W1 por feature) — calibracao structure/global.
# L = 0.35*W1(flow) + 0.25*W1(Cov) + 0.20*W1(Min) + 0.10*W1(Std) + 0.10*W1(Tot sum)
GREETH_JOINT_LOSS_WEIGHTS = {
    "flow_duration": 0.35,
    "Covariance": 0.25,
    "Min": 0.20,
    "Std": 0.10,
    "Tot sum": 0.10,
}
# Penaliza diferenca de std(Covariance) além do W1 (structure/global greeth).
GREETH_COVARIANCE_STD_PENALTY = 0.15
# Nenhuma feature pode concentrar mais que esta fracao da loss joint (greeth).
GREETH_MAX_FEATURE_LOSS_SHARE = 0.80
# Escalas empiricas de W1_norm (structure greeth) — normalizam magnitudes antes dos pesos SHAP.
GREETH_LOSS_REF_SCALE = {
    "flow_duration": 10.0,
    "Covariance": 1.0,
    "Min": 0.45,
    "Std": 1.0,
    "Tot sum": 3.5,
}

# Refinamento global final (pos payload/temporal/covariance/structure).
GLOBAL_SHAP_WEIGHTS = {
    "greeth": dict(GREETH_JOINT_LOSS_WEIGHTS),
}

# Fase structure: matriz Markov + objetivo conjunto (inclui flow_duration leve).
STRUCTURE_SHAP_WEIGHTS = {
    "greeth": dict(GREETH_JOINT_LOSS_WEIGHTS),
    "greip": {
        "Number": 1.0,
        "Header_Length": 0.85,
        "Tot sum": 0.75,
        "Min": 0.65,
    },
}

# Desacoplamento payload vs temporal (divide-and-conquer).
PAYLOAD_FEATURES = {
    "Min", "Max", "AVG", "Std", "Tot size", "Tot sum",
    "Covariance", "Variance", "UDP", "rst_count", "Header_Length",
}
TEMPORAL_FEATURES = {
    "flow_duration", "Rate", "Duration", "Number",
}
COVARIANCE_FEATURES = {
    "Covariance", "Variance",
}
STRUCTURE_FEATURES = {
    "Number", "Header_Length", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
}

# Fase latent: marginais + estrutura multivariada (HMM emissao conjunta).
LATENT_CORR_FEATURES = [
    "flow_duration", "Min", "Std", "Covariance", "Number",
]
# Tot sum omitido: derivado de Min, Std e packet_count.
LATENT_MARGINAL_WEIGHTS = {
    "flow_duration": 0.38,
    "Covariance": 0.28,
    "Min": 0.22,
    "Std": 0.12,
    "Number": 0.05,
}
GREIP_LATENT_MARGINAL_WEIGHTS = {
    "flow_duration": 0.35,
    "Min": 0.25,
    "Covariance": 0.20,
    "Std": 0.10,
    "Number": 0.05,
}
BENIGN_LATENT_MARGINAL_WEIGHTS = {
    "rst_count": 0.40,
    "Variance": 0.22,
    "Min": 0.18,
    "flow_duration": 0.08,
    "Covariance": 0.07,
    "Std": 0.05,
}
LATENT_VARIANTS = ("greeth", "greip")  # benign: preparado via manifest + benign_latent.py; activar quando TUNABLES_LATENT["benign"] existir
LATENT_MARGINAL_BY_VARIANT = {
    "greeth": LATENT_MARGINAL_WEIGHTS,
    "greip": GREIP_LATENT_MARGINAL_WEIGHTS,
    "benign": BENIGN_LATENT_MARGINAL_WEIGHTS,
}
LATENT_CORR_BY_VARIANT = {
    "greeth": list(LATENT_CORR_FEATURES),
    "greip": list(LATENT_CORR_FEATURES),
    "benign": ["Min", "Variance", "Std", "Covariance", "flow_duration", "rst_count"],
}
LATENT_HYBRID_WEIGHTS = {
    "marginal": 0.25,
    "spearman": 0.45,
    "pearson": 0.05,
    "mmd": 0.15,
    "degeneracy": 0.10,
}
LATENT_CORR_BLEND = {"spearman_global": 0.35, "spearman_state": 0.35, "pearson_global": 0.30}
LATENT_MMD_REF_SCALE = 0.35
LATENT_MARGINAL_MAX_SHARE = 0.80
# Escalas para w1_norm ja calculado por calibration_loss (NAO dividir de novo por IQR bruto).
LATENT_LOSS_REF_SCALE = {
    **GREETH_LOSS_REF_SCALE,
    "Number": 15.0,
}
LATENT_LOSS_REF_BY_VARIANT = {
    "greeth": dict(LATENT_LOSS_REF_SCALE),
    "greip": dict(LATENT_LOSS_REF_SCALE),
    "benign": {
        **LATENT_LOSS_REF_SCALE,
        "rst_count": 400.0,
        "Variance": 0.5,
        "Min": 80.0,
    },
}
OM_STATE_NAMES = (
    "plateau", "normal", "transition", "tail", "burst", "outlier",
)
PER_STATE_CORR_STATES = OM_STATE_NAMES
PER_STATE_MIN_SAMPLES = 25

LABELS = {
    "greeth": "Mirai-greeth_flood",
    "greip": "Mirai-greip_flood",
    "udpplain": "Mirai-udpplain",
    "benign": "BenignTraffic",
}

# Espacos de busca por variante: caminho_no_params -> (min, max, tipo, escala_log)
# O calibrador aplica estes paths no JSON legado ou em variants.<variant> quando existir.
TUNABLES_COMMON_TEMPORAL = {
    "iat.on_rate_pps":              (200.0, 8000.0, "float", True),
    "flow.duration_lognorm_mu":     (0.3, 2.2, "float", False),
    "flow.duration_lognorm_sigma":  (0.2, 1.2, "float", False),
    "flow.packets_mean":            (200, 12000, "int", True),
    "botnet.size":                  (5, 40, "int", False),
}

TUNABLES_GMM_TEMPORAL = {
    "iat.on_rate_pps":              (200.0, 8000.0, "float", True),
    "flow.packets_mean":            (200, 12000, "int", True),
    "botnet.size":                  (5, 60, "int", False),
}

TUNABLES_BY_VARIANT = {
    "udpplain": {
        # UDPPlain ja tem UDP/rst_count/AVG/Std bem alinhados.
        # Para duration_model=gmm_log10, duration_lognorm_* nao afeta a duracao.
        # Mantemos foco em temporalidade, cardinalidade e pesos do GMM.
        **TUNABLES_GMM_TEMPORAL,
        "flow.duration_components.0.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.1.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.2.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.3.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.4.weight": (0.05, 4.0, "float", True),
        # componente 4 representa a cauda longa do UDPPlain.
        "flow.duration_components.4.std_log10": (0.25, 1.2, "float", False),
    },
    "greip": {
        # GRE-IP usa zero_inflated_gmm_log10; duration_lognorm_* nao afeta a duracao.
        # Foco: zero inflation, pesos/cauda do GMM, temporalidade, cardinalidade e header GRE.
        **TUNABLES_GMM_TEMPORAL,
        "flow.duration_zero_fraction": (0.0, 0.9, "float", False),
        "flow.duration_components.0.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.1.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.2.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.3.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.3.std_log10": (0.4, 1.5, "float", False),
        "gre_flow.flows_per_bot": (10, 120, "int", True),
        "gre_flow.single_packet_fraction": (0.3, 0.95, "float", False),
        "gre_flow.packets_mean_long": (2, 300, "int", True),
        "gre_flow.packets_std_long": (1, 120, "int", True),
        "gre_flow.packets_poisson_lambda": (2.0, 80.0, "float", False),
        "size.mu":                  (6.05, 6.55, "float", False),
        "size.sigma":               (0.01, 0.06, "float", False),
        "size.sigma_std":           (0.0, 0.04, "float", False),
        "size.mu_std":              (0.0, 0.06, "float", False),
        "gre.flag_weight_4":        (0.05, 0.85, "float", False),
        "gre.flag_weight_8":        (0.05, 0.85, "float", False),
    },
    "greeth": {
        # GRE-ETH tambem usa zero_inflated_gmm_log10; calibrar componentes e GRE.
        **TUNABLES_GMM_TEMPORAL,
        "capture_overhead":             (14, 50, "int", False),
        "size.small_fraction":          (0.0, 0.18, "float", False),
        "size.small_mu":                (3.7, 4.3, "float", False),
        "size.small_sigma":             (0.05, 0.3, "float", False),
        "size.jitter_std":              (0.0, 50.0, "float", False),
        "size.noisy_fraction":          (0.0, 0.12, "float", False),
        "dynamic_streams.size_std":     (10.0, 75.0, "float", False),
        "dynamic_streams.size_correlation": (0.5, 0.99, "float", False),
        "flow.duration_zero_fraction": (0.0, 0.9, "float", False),
        "flow.duration_components.0.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.1.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.2.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.3.weight": (0.05, 4.0, "float", True),
        "flow.duration_components.3.std_log10": (0.4, 1.5, "float", False),
        "gre_flow.flows_per_bot": (10, 120, "int", True),
        "gre_flow.single_packet_fraction": (0.3, 0.98, "float", False),
        "gre_flow.packets_mean_long": (2, 300, "int", True),
        "gre_flow.packets_std_long": (1, 120, "int", True),
        "gre_flow.packets_poisson_lambda": (2.0, 80.0, "float", False),
        "size.mu":                  (6.05, 6.55, "float", False),
        "size.sigma":               (0.01, 0.12, "float", False),
        "size.sigma_std":           (0.0, 0.05, "float", False),
        "size.mu_std":              (0.0, 0.06, "float", False),
        "size.small_sigma_std":     (0.0, 0.10, "float", False),
        "gre.flag_weight_4":        (0.05, 0.85, "float", False),
        "gre.flag_weight_8":        (0.05, 0.85, "float", False),
    },
    "benign": {
        "benign.num_devices":       (100, 2000, "int", True),
        "benign.mean_interval":     (0.1, 5.0, "float", True),
        "benign.interval_spread":   (0.05, 1.2, "float", False),
        "benign.jitter":            (0.01, 0.8, "float", False),
        "benign.payload_mean":      (40.0, 600.0, "float", True),
        "benign.payload_std":       (1.0, 150.0, "float", True),
        "benign.burst_p":           (0.0, 0.2, "float", False),
        "benign.burst_size":        (1, 8, "int", False),
        "benign.profiles.0.weight": (0.2, 0.8, "float", False),
        "benign.profiles.0.mean_interval": (0.2, 3.0, "float", True),
        "benign.profiles.0.payload_mean": (40.0, 180.0, "float", True),
        "benign.profiles.1.weight": (0.1, 0.6, "float", False),
        "benign.profiles.1.mean_interval": (0.1, 1.5, "float", True),
        "benign.profiles.1.payload_mean": (60.0, 240.0, "float", True),
        "benign.profiles.2.weight": (0.01, 0.2, "float", False),
        "benign.profiles.2.payload_mean": (200.0, 900.0, "float", True),
        "benign.profiles.3.weight": (0.01, 0.2, "float", False),
        "benign.profiles.3.burst_p": (0.0, 0.2, "float", False),
    },
}

TUNABLES_LATENT = {
    "greeth": {
        "om.inter_window_gap_sec": (0.0015, 0.005, "float", False),
        "om.mode.plateau.weight": (0.35, 0.55, "float", False),
        "om.mode.normal.weight": (0.25, 0.42, "float", False),
        "om.mode.transition.weight": (0.06, 0.14, "float", False),
        "om.mode.tail.weight": (0.03, 0.12, "float", False),
        "om.mode.burst.weight": (0.01, 0.06, "float", False),
        "om.mode.outlier.weight": (0.005, 0.04, "float", False),
        "om.mode.plateau.template_jitter_std": (0.0, 2.0, "float", False),
        "om.mode.normal.template_jitter_std": (1.0, 6.0, "float", False),
        "om.mode.tail.template_jitter_std": (3.0, 14.0, "float", False),
        "om.mode.normal.flow_duration.low": (0.02, 0.06, "float", False),
        "om.mode.normal.flow_duration.high": (0.05, 0.12, "float", False),
        "om.mode.tail.flow_duration.low": (0.15, 0.32, "float", False),
        "om.mode.tail.flow_duration.high": (0.28, 0.55, "float", False),
        "om.mode.normal.iat_jitter_frac": (0.03, 0.12, "float", False),
        "om.mode.tail.iat_jitter_frac": (0.06, 0.18, "float", False),
        "om.mode.normal.packet_count.lo": (17, 20, "int", False),
        "om.mode.normal.packet_count.hi": (19, 23, "int", False),
        "om.mode.tail.packet_count.lo": (16, 20, "int", False),
        "om.mode.tail.packet_count.hi": (18, 22, "int", False),
        "om.trans.plateau.stay": (0.88, 0.97, "float", False),
        "om.trans.normal.stay": (0.78, 0.92, "float", False),
        "om.trans.tail.stay": (0.45, 0.72, "float", False),
    },
    "greip": {
        "om.inter_window_gap_sec": (0.0015, 0.005, "float", False),
        "om.mode.plateau.weight": (0.30, 0.58, "float", False),
        "om.mode.normal.weight": (0.22, 0.45, "float", False),
        "om.mode.transition.weight": (0.05, 0.16, "float", False),
        "om.mode.tail.weight": (0.03, 0.14, "float", False),
        "om.mode.burst.weight": (0.01, 0.08, "float", False),
        "om.mode.outlier.weight": (0.005, 0.05, "float", False),
        "om.mode.plateau.template_jitter_std": (0.0, 2.0, "float", False),
        "om.mode.normal.template_jitter_std": (1.0, 6.0, "float", False),
        "om.mode.tail.template_jitter_std": (3.0, 14.0, "float", False),
        "om.mode.normal.flow_duration.low": (0.02, 0.06, "float", False),
        "om.mode.normal.flow_duration.high": (0.05, 0.12, "float", False),
        "om.mode.tail.flow_duration.low": (0.12, 0.30, "float", False),
        "om.mode.tail.flow_duration.high": (0.25, 0.50, "float", False),
        "om.mode.normal.iat_jitter_frac": (0.03, 0.12, "float", False),
        "om.mode.tail.iat_jitter_frac": (0.06, 0.18, "float", False),
        "om.mode.normal.packet_count.lo": (17, 20, "int", False),
        "om.mode.normal.packet_count.hi": (19, 23, "int", False),
        "om.mode.tail.packet_count.lo": (16, 20, "int", False),
        "om.mode.tail.packet_count.hi": (18, 22, "int", False),
        "om.trans.plateau.stay": (0.86, 0.97, "float", False),
        "om.trans.normal.stay": (0.75, 0.92, "float", False),
        "om.trans.tail.stay": (0.40, 0.75, "float", False),
        "gre.flag_weight_4": (0.05, 0.85, "float", False),
        "gre.flag_weight_8": (0.05, 0.85, "float", False),
    },
}

TUNABLES_STRUCTURE = {
    "greeth": {
        "capture_overhead":             (14, 50, "int", False),
        "dynamic_streams.window_hetero.markov_gmm.plateau_prob": (0.20, 0.70, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.normal_prob": (0.30, 0.80, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.tail_prob": (0.01, 0.12, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.tail_stay_prob": (0.0, 0.30, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.tail_scale": (0.5, 2.0, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.transition.0.1": (0.03, 0.12, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.transition.0.2": (0.01, 0.06, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.transition.1.0": (0.04, 0.15, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.transition.1.2": (0.02, 0.10, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.transition.2.0": (0.08, 0.30, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.transition.2.1": (0.50, 0.85, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.ar1_rho": (0.55, 0.88, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.spike_prob": (0.0, 0.06, "float", False),
        "dynamic_streams.window_hetero.markov_gmm.jitter_std": (0.5, 20.0, "float", False),
        "dynamic_streams.window_hetero.long_flow_fraction": (0.0, 0.42, "float", False),
        "dynamic_streams.window_hetero.long_flow_reset_after_packets": (200, 700, "int", True),
        "gre_flow.single_packet_fraction": (0.3, 0.98, "float", False),
        "gre.flag_weight_4":            (0.05, 0.85, "float", False),
        "gre.flag_weight_8":            (0.05, 0.85, "float", False),
    },
    "greip": {
        "gre_flow.single_packet_fraction": (0.3, 0.95, "float", False),
        "gre_flow.packets_mean_long":   (2, 300, "int", True),
        "gre_flow.packets_std_long":    (1, 120, "int", True),
        "gre_flow.packets_poisson_lambda": (2.0, 80.0, "float", False),
        "size.mu":                      (6.05, 6.55, "float", False),
        "size.sigma":                   (0.01, 0.06, "float", False),
        "size.sigma_std":               (0.0, 0.04, "float", False),
        "size.mu_std":                  (0.0, 0.06, "float", False),
        "gre.flag_weight_4":            (0.05, 0.85, "float", False),
        "gre.flag_weight_8":            (0.05, 0.85, "float", False),
    },
}

# Parametros globais (nao ficam em variants.*).
ROOT_TUNABLES = {"capture_overhead"}

# Alias legado: scripts externos que importarem TUNABLES continuam funcionando.
TUNABLES = TUNABLES_BY_VARIANT["udpplain"]


# --------------------------------------------------------------------------- #
# Utilidades de acesso a params aninhados                                      #
# --------------------------------------------------------------------------- #
def _get(params, path):
    cur = params
    for k in path.split("."):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur


def _set(params, path, value):
    keys = path.split(".")
    cur = params
    for k in keys[:-1]:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur.setdefault(k, {})
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def _classify_tunable(path):
    """Classifica um parametro tunavel como payload, temporal, covariance ou both (None)."""
    if path in ROOT_TUNABLES:
        return "payload"
    if path.startswith("dynamic_streams."):
        return "covariance"
    if path in (
        "size.small_fraction", "size.small_mu", "size.small_sigma",
        "size.jitter_std", "size.sigma",
    ):
        return None
    if path.startswith(("size.", "gre.flag_weight")):
        return "payload"
    if path.startswith(("iat.", "flow.", "botnet.", "gre_flow.")):
        return "temporal"
    if path.startswith("benign."):
        if any(x in path for x in ("mean_interval", "interval_spread", "jitter", "num_devices", "burst_p")):
            return "temporal"
        if any(x in path for x in ("payload", "weight", "burst_size")):
            return "payload"
    return None


def filter_tunables_by_phase(tunables, phase, variant=None):
    if phase == "latent" and variant in TUNABLES_LATENT:
        return dict(TUNABLES_LATENT[variant])
    if phase == "structure" and variant in TUNABLES_STRUCTURE:
        return dict(TUNABLES_STRUCTURE[variant])
    if phase in ("all", "global"):
        return dict(tunables)
    out = {}
    for path, spec in tunables.items():
        cls = _classify_tunable(path)
        if cls == phase or cls is None:
            out[path] = spec
    return out


def filter_shap_by_phase(shap, phase, variant=None):
    if phase == "all":
        return dict(shap)
    if phase == "latent":
        if variant in LATENT_MARGINAL_BY_VARIANT:
            return dict(LATENT_MARGINAL_BY_VARIANT[variant])
        return dict(LATENT_MARGINAL_WEIGHTS)
    if phase == "global":
        if variant and variant in GLOBAL_SHAP_WEIGHTS:
            return dict(GLOBAL_SHAP_WEIGHTS[variant])
        keys = (PAYLOAD_FEATURES | COVARIANCE_FEATURES | STRUCTURE_FEATURES) - {"Variance"}
        return {k: v for k, v in shap.items() if k in keys}
    if phase == "structure":
        if variant and variant in STRUCTURE_SHAP_WEIGHTS:
            return dict(STRUCTURE_SHAP_WEIGHTS[variant])
        return {k: v for k, v in shap.items() if k in STRUCTURE_FEATURES}
    if phase == "covariance":
        filtered = {k: v for k, v in shap.items() if k in COVARIANCE_FEATURES}
        return filtered if filtered else {"Covariance": float(shap.get("Covariance", 1.0))}
    if phase == "payload":
        # Covariance/Variance tem fase propria (dynamic_streams); payload = geometria/tamanho.
        allowed = PAYLOAD_FEATURES - COVARIANCE_FEATURES
    else:
        allowed = TEMPORAL_FEATURES
    filtered = {k: v for k, v in shap.items() if k in allowed}
    return filtered if filtered else dict(shap)


def _path_exists(params, path):
    try:
        _get(params, path)
        return True
    except (KeyError, IndexError, ValueError, TypeError):
        return False


def _set_variant_aware(params, variant, path, value):
    """
    Aplica parametro no JSON legado ou, se existir JSON v2, em variants.<variant>.
    Isso evita que overrides de variants.* escondam valores ajustados no topo.
    """
    if path in ROOT_TUNABLES:
        _set(params, path, value)
        return
    if variant != "benign" and variant in params.get("variants", {}) and not path.startswith("benign."):
        _set(params["variants"][variant], path, value)
    else:
        _set(params, path, value)


def _sample(rng, lo, hi, typ, log):
    if log:
        v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        v = rng.uniform(lo, hi)
    return int(round(v)) if typ == "int" else v


def _perturb(rng, cur, lo, hi, typ, log, frac):
    """Perturba um valor em torno de 'cur' por +-frac da faixa (refino greedy)."""
    span = (math.log(hi) - math.log(lo)) if log else (hi - lo)
    if log:
        step = rng.uniform(-frac, frac) * span
        v = math.exp(min(max(math.log(cur), math.log(lo)), math.log(hi)) + step)
    else:
        step = rng.uniform(-frac, frac) * span
        v = cur + step
    v = min(max(v, lo), hi)
    return int(round(v)) if typ == "int" else v


def min_percentile_penalty(a, b):
    """Penaliza desvio de P50/P25 de Min (alinhado ao notebook exploratorio)."""
    iqr = np.subtract(*np.percentile(b, [75, 25]))
    scale = iqr if iqr > 1e-9 else (float(b.std()) if b.std() > 1e-9 else 1.0)
    p50 = abs(float(np.percentile(a, 50)) - float(np.percentile(b, 50))) / scale
    p25 = abs(float(np.percentile(a, 25)) - float(np.percentile(b, 25))) / scale
    return p50 + 0.5 * p25


def greeth_balanced_joint_loss(per_feat, weights, max_share=None, ref_scale=None):
    """
    Loss joint GRE-ETH com componentes normalizadas por escala de referencia e
    teto de contribuicao por feature (default 80%).

    L = sum_f w_f * (W1_f / ref_f), com iteracao que limita max_f contrib <= max_share.
    """
    max_share = GREETH_MAX_FEATURE_LOSS_SHARE if max_share is None else max_share
    ref_scale = ref_scale or GREETH_LOSS_REF_SCALE
    skip = {"covariance_std_penalty", "error", "trial_dir"}
    terms = {}
    for f, raw in per_feat.items():
        if f in skip or f not in weights:
            continue
        scale = float(ref_scale.get(f, 1.0))
        terms[f] = float(weights[f]) * (float(raw) / max(scale, 1e-9))

    if not terms:
        return 0.0, {}

    cap_ratio = max_share / max(1e-9, 1.0 - max_share)
    for _ in range(len(terms) + 2):
        total = sum(terms.values())
        if total <= 0:
            return 0.0, terms
        dominant = max(terms, key=terms.get)
        share = terms[dominant] / total
        if share <= max_share:
            break
        other = total - terms[dominant]
        terms[dominant] = min(terms[dominant], cap_ratio * other)

    return sum(terms.values()), terms


def greeth_joint_objective_description(weights, max_share=None):
    max_share = GREETH_MAX_FEATURE_LOSS_SHARE if max_share is None else max_share
    terms = " + ".join(
        f"{w:.2f}*(W1({f})/ref_{f})" for f, w in weights.items()
    )
    return (
        f"L = [{terms}] capped @ {max_share:.0%}/feature "
        f"+ penalidades (Min percentis, std Covariance)"
    )


def latent_hybrid_objective_description():
    w = LATENT_HYBRID_WEIGHTS
    b = LATENT_CORR_BLEND
    return (
        f"L = {w['marginal']:.0%}*W1_norm(SHAP) + {w['spearman']+w['pearson']:.0%}*corr "
        f"({b['spearman_global']:.0%} global Spearman + {b['spearman_state']:.0%} per-state + "
        f"{b['pearson_global']:.0%} Pearson) + {w['mmd']:.0%}*MMD + {w['degeneracy']:.0%}*degeneracy "
        f"(Tot sum fora das marginais; w1_norm de calibration_loss, sem 2a divisao IQR)"
    )


def _real_marginal_scales(real_df, features, mode="iqr"):
    """Escala por feature a partir do real: IQR ou P95-P5 (diagnostico/auditoria)."""
    scales = {}
    for f in features:
        if f not in real_df.columns:
            scales[f] = 1.0
            continue
        b = pd.to_numeric(real_df[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(b) < 10:
            scales[f] = 1.0
            continue
        if mode == "p95p5":
            scale = float(np.percentile(b, 95) - np.percentile(b, 5))
        else:
            scale = float(np.subtract(*np.percentile(b, [75, 25])))
        scales[f] = max(scale, 1e-6)
    return scales


def latent_marginal_weights(variant):
    return LATENT_MARGINAL_BY_VARIANT.get(variant, LATENT_MARGINAL_WEIGHTS)


def latent_corr_features(variant):
    return LATENT_CORR_BY_VARIANT.get(variant, LATENT_CORR_FEATURES)


def latent_loss_ref_scale(variant):
    return LATENT_LOSS_REF_BY_VARIANT.get(variant, LATENT_LOSS_REF_SCALE)


def latent_marginal_loss(per_feat_norm, weights=None, max_share=None, ref_scale=None):
    """
    Combina w1_norm (de calibration_loss / calibration_w1_norm) com pesos SHAP.

    IMPORTANTE: per_feat_norm ja vem normalizado (modo hybrid/balanced). Nao dividir
    novamente por IQR bruto do real — isso explode flow_duration e Number quando o IQR
    real e ~0 (plateau / Number quase constante).

    L_marg = sum_f w_f * (w1_norm_f / ref_f), com cap @ max_share por feature.
    Escala tipica: O(1) quando w1_norm ~ ref_f.
    """
    return greeth_balanced_joint_loss(
        per_feat_norm,
        weights or LATENT_MARGINAL_WEIGHTS,
        max_share=max_share,
        ref_scale=ref_scale or LATENT_LOSS_REF_SCALE,
    )


def _map_synth_state_for_corr(name):
    return str(name).lower()


def _proxy_real_om_state(row):
    """Proxy de estado operacional (6 modos) para linhas reais AGG."""
    fd = float(row.get("flow_duration", 0) or 0)
    mn = float(row.get("Min", 500) or 500)
    std = float(row.get("Std", 0) or 0)
    cov = float(row.get("Covariance", 0) or 0)
    if fd <= 1e-6 and mn >= 550:
        return "plateau"
    if mn <= 450:
        return "tail"
    if cov >= 20000 or (mn < 500 and std > 35):
        return "outlier"
    if fd >= 0.12 and std >= 20:
        return "burst"
    if 460 <= mn < 550 and fd >= 0.01:
        return "transition"
    return "normal"


def _resolve_operational_modes(params, variant):
    """Lê operational_modes do GRE (window_hetero) ou do benign latent_hmm."""
    materialized = gen.materialize_variant_params(params, variant)
    if variant == "benign":
        b = materialized.get("benign") or {}
        return (b.get("latent_hmm") or {}).get("operational_modes") or {}
    wh = (materialized.get("dynamic_streams") or {}).get("window_hetero") or {}
    return wh.get("operational_modes") or {}


def _om_state_corr_weights(params, variant):
    """Pesos w_s por estado (prior do HMM) para sum_s w_s ||Delta rho_s||_F."""
    om = _resolve_operational_modes(params, variant)
    modes = om.get("modes") or []
    weights = {}
    for m in modes:
        name = str(m.get("name", "")).lower()
        if variant == "benign":
            if name:
                weights[name] = max(0.0, float(m.get("weight", 0.0)))
        elif name in PER_STATE_CORR_STATES:
            weights[name] = max(0.0, float(m.get("weight", 0.0)))
    if not weights or sum(weights.values()) <= 0:
        if variant == "benign":
            return {}
        n = len(PER_STATE_CORR_STATES)
        return {s: 1.0 / n for s in PER_STATE_CORR_STATES}
    total = sum(weights.values())
    if variant == "benign":
        return {s: weights[s] / total for s in weights}
    return {s: weights.get(s, 0.0) / total for s in PER_STATE_CORR_STATES}


def _per_state_corr_frobenius(
    synth_df, real_df, features, method="spearman", min_n=None, state_weights=None,
    states=None,
):
    """sum_s w_s ||rho_s_real - rho_s_synth||_F sobre estados com amostras suficientes."""
    min_n = PER_STATE_MIN_SAMPLES if min_n is None else min_n
    state_weights = state_weights or {}
    state_list = list(states) if states else list(PER_STATE_CORR_STATES)
    if not state_weights:
        state_weights = {s: 1.0 / max(1, len(state_list)) for s in state_list}
    synth = synth_df.copy()
    real = real_df.copy()
    state_col = "state" if "state" in synth.columns else "om_state"
    synth["state"] = synth[state_col].astype(str).str.lower().map(_map_synth_state_for_corr)
    if "state" not in real.columns:
        real["state"] = real.apply(_proxy_real_om_state, axis=1)

    per_state = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for st in state_list:
        w = float(state_weights.get(st, 0.0))
        if w <= 0:
            per_state[st] = float("nan")
            continue
        sr = real[real["state"] == st]
        ss = synth[synth["state"] == st]
        if len(sr) < min_n or len(ss) < min_n:
            per_state[st] = float("nan")
            continue
        frob = _corr_frobenius_offdiag(
            ss, sr, features, method=method,
            sample=min(800, len(ss), len(sr)),
        )
        if not np.isfinite(frob):
            per_state[st] = float("nan")
            continue
        per_state[st] = frob
        weighted_sum += w * frob
        weight_total += w
    if weight_total <= 0:
        return None, per_state
    return weighted_sum / weight_total, per_state


def _corr_frobenius_offdiag(synth, real, features, method="spearman", sample=3000, seed=42):
    cols = [f for f in features if f in synth.columns and f in real.columns]
    if len(cols) < 3:
        return 1.0
    s = synth[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    r = real[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = min(len(s), len(r), sample)
    if n < 30:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(s) > n:
        s = s.iloc[rng.choice(len(s), size=n, replace=False)]
    if len(r) > n:
        r = r.iloc[rng.choice(len(r), size=n, replace=False)]
    n = min(len(s), len(r))
    s = s.iloc[:n].reset_index(drop=True)
    r = r.iloc[:n].reset_index(drop=True)
    if method == "pearson":
        rm, sm = r.corr(method="pearson"), s.corr(method="pearson")
    else:
        rm, sm = r.corr(method="spearman"), s.corr(method="spearman")
    rm = rm.fillna(0.0)
    sm = sm.fillna(0.0)
    m = rm.values - sm.values
    np.fill_diagonal(m, 0.0)
    return float(np.sqrt(np.sum(m ** 2)))


def _latent_mmd_norm(synth, real, features, sample=2000, seed=42):
    cols = [f for f in features if f in synth.columns and f in real.columns]
    if len(cols) < 2:
        return 1.0
    s = synth[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    r = real[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = min(len(s), len(r), sample)
    if n < 30:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(s) > n:
        s = s.iloc[rng.choice(len(s), size=n, replace=False)]
    if len(r) > n:
        r = r.iloc[rng.choice(len(r), size=n, replace=False)]
    n = min(len(s), len(r))
    xs = s.iloc[:n].values.astype(float)
    xr = r.iloc[:n].values.astype(float)
    xs, xr = _standardize(xs, xr)
    raw = float(mmd_rbf(xs, xr, max_n=n))
    return raw / max(LATENT_MMD_REF_SCALE, 1e-9)


def operational_mode_degeneracy_penalty(params, variant):
    """Penaliza pesos HMM degenerados (um estado domina ou entropia baixa)."""
    materialized = gen.materialize_variant_params(params, variant)
    om = ((materialized.get("dynamic_streams") or {}).get("window_hetero") or {}).get("operational_modes") or {}
    modes = om.get("modes") or []
    if not modes:
        return 0.0
    w = np.array([max(0.0, float(m.get("weight", 0.0))) for m in modes], dtype=float)
    if w.sum() <= 0:
        return 1.0
    w = w / w.sum()
    entropy = float(-np.sum(w * np.log(w + 1e-12)))
    max_ent = math.log(len(w))
    ent_pen = max(0.0, 0.55 - entropy / max(max_ent, 1e-9))
    share_pen = max(0.0, float(w.max()) - 0.50) * 2.0
    tail_out = 0.0
    by_name = {m.get("name"): float(m.get("weight", 0.0)) for m in modes}
    tail_out = max(
        0.0,
        by_name.get("tail", 0.0) + by_name.get("outlier", 0.0) - 0.18,
    ) * 1.5
    return ent_pen + share_pen + tail_out


def _get_operational_modes_block(params, variant):
    materialized = gen.materialize_variant_params(params, variant)
    ds = materialized.get("dynamic_streams") or {}
    wh = ds.get("window_hetero") or {}
    om = wh.get("operational_modes")
    if not om or not om.get("modes"):
        om = gen._default_operational_modes_greeth()
    return copy.deepcopy(om)


def _apply_operational_mode_tunables(params, variant, values):
    """Aplica tunables om.* ao bloco operational_modes (HMM emissao conjunta)."""
    om = _get_operational_modes_block(params, variant)
    om["enabled"] = True
    om["variable_window_packets"] = True

    if "om.inter_window_gap_sec" in values:
        om["inter_window_gap_sec"] = float(values["om.inter_window_gap_sec"])

    by_name = {m["name"]: m for m in om.get("modes", [])}
    for name in OM_STATE_NAMES:
        if name not in by_name:
            continue
        mode = by_name[name]
        em = mode.setdefault("emission", {})
        em.setdefault("emission_mode", "vector")
        em.setdefault("joint_emission", True)
        em.setdefault("window_template", name)

        wkey = f"om.mode.{name}.weight"
        if wkey in values:
            mode["weight"] = float(values[wkey])

        tkey = f"om.mode.{name}.template_jitter_std"
        if tkey in values:
            em["template_jitter_std"] = float(values[tkey])

        ikey = f"om.mode.{name}.iat_jitter_frac"
        if ikey in values:
            em["iat_jitter_frac"] = float(values[ikey])

        fd_lo = f"om.mode.{name}.flow_duration.low"
        fd_hi = f"om.mode.{name}.flow_duration.high"
        if fd_lo in values or fd_hi in values:
            fd = em.get("flow_duration")
            if isinstance(fd, dict):
                fd = dict(fd)
            elif fd is None:
                fd = {"model": "uniform", "low": 0.0, "high": 0.0}
            else:
                v = float(fd)
                fd = {"model": "uniform", "low": v, "high": v}
            if fd_lo in values:
                fd["low"] = float(values[fd_lo])
            if fd_hi in values:
                fd["high"] = float(values[fd_hi])
            if float(fd.get("high", 0)) < float(fd.get("low", 0)):
                fd["high"] = fd["low"]
            em["flow_duration"] = fd

        pclo = f"om.mode.{name}.packet_count.lo"
        pchi = f"om.mode.{name}.packet_count.hi"
        if pclo in values or pchi in values:
            cur = em.get("packet_count_range") or [18, 22]
            lo = int(values.get(pclo, cur[0]))
            hi = int(values.get(pchi, cur[-1]))
            em["packet_count_range"] = [min(lo, hi), max(lo, hi)]

    modes = om.get("modes") or []
    total_w = sum(float(m.get("weight", 0.0)) for m in modes)
    if total_w <= 0:
        total_w = 1.0
    for m in modes:
        m["weight"] = float(m.get("weight", 1.0)) / total_w

    trans = om.get("transition")
    if trans and len(trans) == len(OM_STATE_NAMES):
        for name in OM_STATE_NAMES:
            skey = f"om.trans.{name}.stay"
            if skey not in values:
                continue
            idx = OM_STATE_NAMES.index(name)
            stay = float(min(max(float(values[skey]), 0.01), 0.99))
            n = len(trans[idx])
            off = (1.0 - stay) / max(1, n - 1)
            row = [off] * n
            row[idx] = stay
            trans[idx] = row
        om["transition"] = trans

    _set_variant_aware(params, variant, "dynamic_streams.window_hetero.operational_modes", om)
    _set_variant_aware(params, variant, "dynamic_streams.enabled", True)
    return params


def latent_hybrid_loss(synth, real, per_feat_w1, params, variant, manifest=None, marginal_weights=None):
    mw = marginal_weights or latent_marginal_weights(variant)
    corr_feats = latent_corr_features(variant)
    ref_scale = latent_loss_ref_scale(variant)
    l_marg, contrib = latent_marginal_loss(per_feat_w1, mw, ref_scale=ref_scale)
    degen = operational_mode_degeneracy_penalty(params, variant)

    l_sp_g = _corr_frobenius_offdiag(synth, real, corr_feats, method="spearman")
    l_pe_g = _corr_frobenius_offdiag(synth, real, corr_feats, method="pearson")
    per_state = {}
    state_weights = _om_state_corr_weights(params, variant)
    state_list = None
    if variant == "benign" and manifest:
        state_list = sorted({
            str(r.get("state", "")).lower()
            for r in manifest if r.get("state")
        })
    if manifest:
        l_sp_s, per_state = _per_state_corr_frobenius(
            pd.DataFrame(manifest), real, corr_feats,
            state_weights=state_weights,
            states=state_list,
        )
        if l_sp_s is None:
            l_sp_s = l_sp_g
    else:
        l_sp_s = l_sp_g

    b = LATENT_CORR_BLEND
    l_corr = (
        b["spearman_global"] * l_sp_g
        + b["spearman_state"] * l_sp_s
        + b["pearson_global"] * l_pe_g
    )
    l_mmd = _latent_mmd_norm(synth, real, corr_feats)

    w = LATENT_HYBRID_WEIGHTS
    total = (
        w["marginal"] * l_marg
        + (w["spearman"] + w["pearson"]) * l_corr
        + w["mmd"] * l_mmd
        + w["degeneracy"] * degen
    )
    breakdown = {
        "loss_marginal": l_marg,
        "loss_spearman_global": l_sp_g,
        "loss_spearman_state": l_sp_s,
        "loss_pearson_global": l_pe_g,
        "loss_correlation": l_corr,
        "loss_mmd_norm": l_mmd,
        "loss_degeneracy": degen,
    }
    for st, val in per_state.items():
        breakdown[f"loss_state_{st}"] = val
    for k, v in contrib.items():
        breakdown[f"loss_contrib_{k}"] = v
    return total, breakdown


def apply_candidate(base, variant, values, phase="all"):
    """Aplica um vetor de valores aos params e mantem coerencia (packets_std etc.)."""
    p = copy.deepcopy(base)
    p["variant"] = variant
    for path, val in values.items():
        if phase == "latent" and path.startswith("om."):
            continue
        _set_variant_aware(p, variant, path, val)
    if any(path.startswith("dynamic_streams.") for path in values):
        _set_variant_aware(p, variant, "dynamic_streams.enabled", True)
    elif phase == "payload":
        _set_variant_aware(p, variant, "dynamic_streams.enabled", False)
    elif phase in ("covariance", "global", "structure"):
        _set_variant_aware(p, variant, "dynamic_streams.enabled", True)
    elif phase == "latent":
        _apply_operational_mode_tunables(p, variant, values)
    # coerencia derivada
    if variant != "benign":
        materialized = gen.materialize_variant_params(p, variant)
        pm = int(_get(materialized, "flow.packets_mean"))
        _set_variant_aware(p, variant, "flow.packets_std", max(5, pm // 4))
        _set_variant_aware(p, variant, "flow.packets_min", max(21, pm // 10))
        _set_variant_aware(p, variant, "iat.model", materialized.get("iat", {}).get("model", "onoff"))
        _normalize_gre_weights(p, variant)
    if any(path.startswith("dynamic_streams.window_hetero.") for path in values):
        _normalize_window_hetero_fractions(p, variant)
        _normalize_markov_gmm_params(p, variant)
    return p


def _normalize_markov_gmm_params(params, variant):
    """
    Coerencia Markov-GMM:
      - plateau/normal/tail_prob normalizados para somar 1;
      - linhas da matriz de transicao somam 1 (diagonal = resto);
      - tail_stay_prob sobrescreve P(t->t) na linha da cauda.
    """
    materialized = gen.materialize_variant_params(params, variant)
    wh = (materialized.get("dynamic_streams") or {}).get("window_hetero") or {}
    mg = wh.get("markov_gmm") or {}
    if not mg:
        return

    base = "dynamic_streams.window_hetero.markov_gmm"
    role_keys = ("plateau_prob", "normal_prob", "tail_prob")
    if any(mg.get(k) is not None for k in role_keys):
        pl_f = float(mg.get("plateau_prob", 0.46))
        no_f = float(mg.get("normal_prob", 0.49))
        ta_f = float(mg.get("tail_prob", 0.05))
        total = pl_f + no_f + ta_f
        if total <= 0:
            pl_f, no_f, ta_f = 0.46, 0.49, 0.05
            total = 1.0
        pl_f, no_f, ta_f = pl_f / total, no_f / total, ta_f / total
        _set_variant_aware(params, variant, f"{base}.plateau_prob", pl_f)
        _set_variant_aware(params, variant, f"{base}.normal_prob", no_f)
        _set_variant_aware(params, variant, f"{base}.tail_prob", ta_f)

    trans = mg.get("transition")
    if not trans or len(trans) != 3:
        return
    new_trans = []
    for i in range(3):
        row = [float(trans[i][j]) for j in range(3)]
        off = sum(row[j] for j in range(3) if j != i)
        off = min(max(off, 0.01), 0.95)
        row[i] = max(0.01, 1.0 - off)
        rs = sum(row)
        if rs <= 0:
            row = list(gen.DEFAULT_MARKOV_TRANSITION_3[i])
            rs = sum(row)
        row = [x / rs for x in row]
        new_trans.append(row)

    tail_stay = mg.get("tail_stay_prob")
    if tail_stay is not None:
        ts = float(min(max(float(tail_stay), 0.0), 0.95))
        row = new_trans[2]
        off_pt, off_pn = float(row[0]), float(row[1])
        off_sum = off_pt + off_pn
        budget = 1.0 - ts
        if off_sum <= 0:
            row = [0.20, 0.75, ts]
        else:
            row = [budget * off_pt / off_sum, budget * off_pn / off_sum, ts]
        new_trans[2] = row

    for i in range(3):
        for j in range(3):
            _set_variant_aware(
                params, variant, f"{base}.transition.{i}.{j}", new_trans[i][j],
            )
    stay = [new_trans[i][i] for i in range(3)]
    _set_variant_aware(params, variant, f"{base}.stay_probs", stay)


def _normalize_window_hetero_fractions(params, variant):
    """Legado mixed: low_std + ramp + outlier = 1. Markov-GMM ignora este passo."""
    materialized = gen.materialize_variant_params(params, variant)
    ds = materialized.get("dynamic_streams") or {}
    wh = ds.get("window_hetero")
    if not wh:
        return
    profile = str(wh.get("tot_sum_profile", "markov_gmm")).lower()
    if profile in ("markov_gmm", "markov", "gmm_markov"):
        _set_variant_aware(params, variant, "dynamic_streams.window_hetero.tot_sum_profile", "markov_gmm")
        return
    f_low = max(0.0, float(wh.get("low_std_fraction", 0.5)))
    f_ramp = max(0.0, float(wh.get("ramp_fraction", 0.4)))
    f_out = max(0.0, float(wh.get("outlier_fraction", 0.05)))
    total = f_low + f_ramp + f_out
    if total <= 0:
        f_low, f_ramp, f_out = 0.5, 0.45, 0.05
        total = 1.0
    f_low, f_ramp, f_out = f_low / total, f_ramp / total, f_out / total
    _set_variant_aware(params, variant, "dynamic_streams.window_hetero.low_std_fraction", f_low)
    _set_variant_aware(params, variant, "dynamic_streams.window_hetero.ramp_fraction", f_ramp)
    _set_variant_aware(params, variant, "dynamic_streams.window_hetero.outlier_fraction", f_out)
    _set_variant_aware(params, variant, "dynamic_streams.window_hetero.tot_sum_profile", "mixed")


def _set_calibration_windows(params, variant, n_windows):
    if n_windows is None:
        return
    n = max(100, int(n_windows))
    _set_variant_aware(params, variant, "gre_flow.target_windows", n)


def _normalize_gre_weights(params, variant):
    """Mantem flag_weight_4 + flag_weight_8 <= 0.95; restante vira flag_weight_12."""
    materialized = gen.materialize_variant_params(params, variant)
    gre = materialized.get("gre")
    if not gre:
        return
    w4 = max(0.0, min(0.95, float(gre.get("flag_weight_4", 0.25))))
    w8 = max(0.0, min(0.95, float(gre.get("flag_weight_8", 0.25))))
    if w4 + w8 > 0.95:
        scale = 0.95 / (w4 + w8)
        w4 *= scale
        w8 *= scale
    _set_variant_aware(params, variant, "gre.flag_weight_4", w4)
    _set_variant_aware(params, variant, "gre.flag_weight_8", w8)
    _set_variant_aware(params, variant, "gre.flag_weight_12", max(0.0, 1.0 - w4 - w8))


# --------------------------------------------------------------------------- #
# Avaliacao de uma candidata                                                   #
# --------------------------------------------------------------------------- #
def evaluate(params, real_path, label, shap, workdir, idx, keep_failed=False,
             loss_mode="balanced", phase="all", min_percentile_weight=0.5):
    trial_dir = os.path.join(workdir, f"it{idx}")
    os.makedirs(trial_dir, exist_ok=True)
    pcap_path = os.path.join(trial_dir, "cap.pcap")
    csv_path = os.path.join(trial_dir, "cap.csv")
    failed = False
    try:
        events = gen.build_schedule(params)
        if not events:
            failed = True
            return float("inf"), {"error": "nenhum evento gerado", "trial_dir": trial_dir}
        gen.write_pcap(events, pcap_path)
        pcap_to_csv.run(pcap_path, csv_path, tmp_dir=os.path.join(trial_dir, "tmp"), quiet=True)

        synth, real = load_and_align(csv_path, real_path, label)
        marginal_keys = list(latent_marginal_weights(params.get("variant"))) if phase == "latent" else list(shap.keys())
        log_feats = marginal_keys + (["Tot sum"] if phase == "latent" else [])
        feats = pick_features(synth, real, log_feats if phase == "latent" else list(shap.keys()))
        if not feats:
            failed = True
            return float("inf"), {
                "error": "nenhuma feature SHAP encontrada nos CSVs",
                "requested": list(shap.keys()),
                "synth_columns": list(synth.columns),
                "real_columns": list(real.columns),
                "trial_dir": trial_dir,
            }

        loss, per_feat = calibration_loss(synth, real, feats, weights=shap, mode=loss_mode)
        loss_contrib = {}
        variant = params.get("variant")
        if variant in LATENT_VARIANTS and phase == "latent":
            manifest = params.get("_window_manifest") or []
            loss, hybrid = latent_hybrid_loss(
                synth, real, per_feat, params, variant, manifest=manifest,
            )
            per_feat = dict(per_feat or {})
            per_feat.update(hybrid)
        elif (
            variant == "greeth"
            and phase in ("structure", "global")
        ):
            loss, loss_contrib = greeth_balanced_joint_loss(per_feat, shap)
            per_feat = dict(per_feat or {})
            for f, c in loss_contrib.items():
                per_feat[f"loss_contrib_{f}"] = c
        if phase == "payload" and "Min" in feats and "Min" in synth.columns and "Min" in real.columns:
            a, b = clean_pair(synth["Min"], real["Min"])
            if len(a) >= 2 and len(b) >= 2:
                pen = min_percentile_penalty(a, b)
                loss += float(shap.get("Min", 1.0)) * min_percentile_weight * pen
        if phase == "structure" and "Min" in feats and "Min" in synth.columns and "Min" in real.columns:
            a, b = clean_pair(synth["Min"], real["Min"])
            if len(a) >= 2 and len(b) >= 2:
                pen = min_percentile_penalty(a, b)
                loss += float(shap.get("Min", 1.0)) * 0.35 * min_percentile_weight * pen
        if (
            phase in ("structure", "global")
            and params.get("variant") == "greeth"
            and "Covariance" in synth.columns
            and "Covariance" in real.columns
        ):
            a, b = clean_pair(synth["Covariance"], real["Covariance"])
            if len(a) >= 2 and len(b) >= 2:
                rs = float(b.std())
                if rs > 0:
                    std_penalty = abs(float(a.std()) - rs) / rs
                    loss += GREETH_COVARIANCE_STD_PENALTY * std_penalty
                    per_feat = dict(per_feat or {})
                    per_feat["covariance_std_penalty"] = std_penalty
        if not per_feat:
            failed = True
            return float("inf"), {
                "error": "metricas vazias apos limpeza de NaN/inf",
                "features": feats,
                "trial_dir": trial_dir,
            }
        return loss, per_feat
    except Exception as e:
        failed = True
        return float("inf"), {
            "error": f"{type(e).__name__}: {e}",
            "trial_dir": trial_dir,
        }
    finally:
        if not (keep_failed and failed):
            import shutil
            shutil.rmtree(trial_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Calibracao automatica ponderada por SHAP.")
    ap.add_argument("--variant", choices=list(TUNABLES_BY_VARIANT.keys()))
    ap.add_argument("--real", help="CSV de referencia CICIoT2023.")
    ap.add_argument("--label", default=None, help="Label no CICIoT2023 (default: mapeado da variante).")
    ap.add_argument("--base-params", default="params.json")
    ap.add_argument("--features", default=None,
                    help="Lista separada por virgula para calibracao customizada (ex: Rate,AVG,Std).")
    ap.add_argument("--weights-json", default=None,
                    help="Arquivo JSON com pesos por feature (ou string JSON inline).")
    ap.add_argument("--iters", type=int, default=40, help="Numero de iteracoes.")
    ap.add_argument("--phase", choices=("all", "payload", "temporal", "covariance", "structure", "global", "latent"), default="all",
                    help="Desacopla calibracao: payload (Min), temporal (flow_duration), "
                         "covariance (dynamic_streams), structure (Number/Header/Tot sum), "
                         "global (refino SHAP conjunto), latent (HMM emissao conjunta + loss hibrida).")
    ap.add_argument("--loss-mode", choices=("balanced", "iqr", "percentile", "hybrid"),
                    default="hybrid",
                    help="hybrid=W1+percentis (recomendado); percentile=só ECDF; balanced=W1 transformado.")
    ap.add_argument("--calibration-seed", type=int, default=42,
                    help="Semente FIXA do gerador durante calibracao (reduz ruido estocastico).")
    ap.add_argument("--vary-seed-per-iter", action="store_true",
                    help="Usa seed diferente por iteracao (comportamento antigo: 1000+it).")
    ap.add_argument("--explore-frac", type=float, default=0.4,
                    help="Fracao de iteracoes de exploracao (random); resto refina o melhor.")
    ap.add_argument("--refine-step", type=float, default=0.15,
                    help="Passo de perturbacao no refino (fracao da faixa).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="CSV de relatorio (default: report_auto_<variant>.csv).")
    ap.add_argument("--out", default=None, help="JSON dos melhores params (default: params_auto_<variant>.json).")
    ap.add_argument("--keep-failed", action="store_true",
                    help="Mantem pasta temporaria de iteracoes com erro para diagnostico.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Nao gerar graficos de convergencia ao final.")
    ap.add_argument("--plot-dir", default=None,
                    help="Diretorio dos graficos de convergencia (default: pasta do CSV).")
    ap.add_argument("--cal-windows", type=int, default=None,
                    help="Reduz gre_flow.target_windows durante calibracao (ex.: 1500).")
    ap.add_argument("--plot-report", default=None,
                    help="Apenas gera graficos a partir de um report CSV existente e sai.")
    args = ap.parse_args()

    if args.plot_report:
        plot_calibration_report(args.plot_report, plot_dir=args.plot_dir)
        return

    if not args.variant or not args.real:
        ap.error("--variant e --real sao obrigatorios (exceto com --plot-report).")

    label = args.label or LABELS[args.variant]
    shap = dict(SHAP_WEIGHTS[args.variant])
    if args.features:
        selected = [f.strip() for f in args.features.split(",") if f.strip()]
        shap = {f: shap.get(f, 1.0) for f in selected}
    if args.weights_json:
        if os.path.exists(args.weights_json):
            with open(args.weights_json, "r", encoding="utf-8") as f:
                shap.update(json.load(f))
        else:
            shap.update(json.loads(args.weights_json))
    shap = filter_shap_by_phase(shap, args.phase, variant=args.variant)
    report_path = args.report or f"report_auto_{args.variant}{'_' + args.phase if args.phase != 'all' else ''}.csv"
    out_path = args.out or f"params_auto_{args.variant}.json"
    base = gen.load_params(args.base_params)
    if args.phase == "latent" and args.variant in LATENT_VARIANTS:
        _set_variant_aware(
            base, args.variant,
            "dynamic_streams.window_hetero.operational_modes.enabled", True,
        )
        _set_variant_aware(base, args.variant, "dynamic_streams.enabled", True)
        if args.variant == "greip":
            _set_variant_aware(base, args.variant, "size.model", "hybrid_window_hetero")
            _set_variant_aware(base, args.variant, "gre_flow.enabled", True)
    if args.phase in ("covariance", "global", "structure", "latent") and args.variant in ("greeth", "greip"):
        _set_variant_aware(base, args.variant, "dynamic_streams.enabled", True)
        if not _path_exists(gen.materialize_variant_params(base, args.variant), "dynamic_streams.block_size"):
            _set_variant_aware(base, args.variant, "dynamic_streams.block_size", 20)
    materialized_base = gen.materialize_variant_params(base, args.variant)
    if args.phase == "latent" and args.variant in TUNABLES_LATENT:
        tunables = dict(TUNABLES_LATENT[args.variant])
    else:
        tunables_all = {
            path: spec for path, spec in TUNABLES_BY_VARIANT[args.variant].items()
            if _path_exists(materialized_base, path)
        }
        tunables = filter_tunables_by_phase(tunables_all, args.phase, variant=args.variant)
    rng = random.Random(args.seed)
    workdir = tempfile.mkdtemp(prefix="autocal_")

    print(f"[auto] variante={args.variant} | label={label}")
    print(f"[auto] phase={args.phase} | loss_mode={args.loss_mode}")
    print(f"[auto] calibration_seed={'1000+it' if args.vary_seed_per_iter else args.calibration_seed}")
    print(f"[auto] SHAP features: {shap}")
    if args.phase == "payload" and "Min" in shap:
        print("[auto] payload: loss inclui W1(Min) + penalizacao P50/P25; "
              "meta tipica 0.8-1.5 (nao comparar com temporal ~0.03)")
    if args.phase == "structure":
        print(f"[auto] structure: {greeth_joint_objective_description(shap)}")
    if args.phase == "latent" and args.variant in LATENT_VARIANTS:
        print(f"[auto] latent: {latent_hybrid_objective_description()}")
    if args.phase == "global" and args.variant == "greeth":
        print(f"[auto] global: {greeth_joint_objective_description(shap)}")
    elif args.phase == "global":
        terms = " + ".join(f"{w:.2f}*W1({f})" for f, w in shap.items())
        print(f"[auto] global: L = {terms}")
    print(f"[auto] tunables ativos ({len(tunables)}): {list(tunables.keys())}")
    if args.cal_windows:
        print(f"[auto] cal-windows={args.cal_windows} (gre_flow.target_windows reduzido)")
    print(f"[auto] iteracoes={args.iters} | workdir={workdir}")

    n_explore = max(1, int(args.iters * args.explore_frac))
    best_values, best_loss, best_feat = None, float("inf"), {}

    rows = []
    feat_names = list(shap.keys())
    if args.phase == "latent" and args.variant in LATENT_VARIANTS:
        feat_names = list(dict.fromkeys(
            list(latent_marginal_weights(args.variant).keys()) + ["Tot sum"],
        ))
    for it in range(args.iters):
        if best_values is None or it < n_explore:
            values = {p: _sample(rng, lo, hi, typ, log)
                      for p, (lo, hi, typ, log) in tunables.items()}
            phase = "explore"
        else:
            values = {}
            for p, (lo, hi, typ, log) in tunables.items():
                values[p] = _perturb(rng, best_values[p], lo, hi, typ, log, args.refine_step)
            phase = "refine"

        params = apply_candidate(base, args.variant, values, phase=args.phase)
        _set_calibration_windows(params, args.variant, args.cal_windows)
        params["seed"] = (1000 + it) if args.vary_seed_per_iter else args.calibration_seed
        loss, per_feat = evaluate(
            params, args.real, label, shap, workdir, it,
            keep_failed=args.keep_failed, loss_mode=args.loss_mode,
            phase=args.phase,
        )

        improved = loss < best_loss
        if improved:
            best_values, best_loss, best_feat = values, loss, per_feat

        row = {
            "iter": it,
            "phase": phase,
            "loss": round(loss, 5),
            "best_loss": round(best_loss, 5),
            "error": per_feat.get("error", ""),
            "trial_dir": per_feat.get("trial_dir", ""),
        }
        for p in tunables:
            row[p] = round(values[p], 4) if isinstance(values[p], float) else values[p]
        for f in feat_names:
            row[f"w1_{f}"] = round(per_feat.get(f, float("nan")), 4)
        if args.phase == "latent":
            for k in (
                "loss_marginal", "loss_spearman_global", "loss_spearman_state",
                "loss_pearson_global", "loss_correlation", "loss_mmd_norm", "loss_degeneracy",
            ):
                row[k] = round(per_feat.get(k, float("nan")), 4)
            for st in PER_STATE_CORR_STATES:
                row[f"loss_state_{st}"] = round(
                    per_feat.get(f"loss_state_{st}", float("nan")), 4,
                )
        rows.append(row)
        tag = "*" if improved else " "
        print(f"  [{phase[:3]}] it={it:02d} loss={loss:8.4f} best={best_loss:8.4f} {tag}")
        if "error" in per_feat:
            print(f"       erro: {per_feat['error']}")
            if args.keep_failed and per_feat.get("trial_dir"):
                print(f"       arquivos preservados em: {per_feat['trial_dir']}")

    # ---- relatorio CSV ----
    if rows:
        cols = list(rows[0].keys())
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[auto] relatorio -> {report_path}")
        if not args.no_plot:
            plot_dir = args.plot_dir or os.path.dirname(os.path.abspath(report_path)) or "."
            plot_calibration_report(report_path, plot_dir=plot_dir)

    # ---- melhores params ----
    if best_values is not None:
        best_params = apply_candidate(base, args.variant, best_values, phase=args.phase)
        best_params["_auto_calibration"] = {
            "variant": args.variant,
            "label": label,
            "phase": args.phase,
            "loss_mode": args.loss_mode,
            "calibration_seed": None if args.vary_seed_per_iter else args.calibration_seed,
            "loss_shap_weighted": round(best_loss, 5),
            "shap_weights": shap,
            "global_objective": (
                latent_hybrid_objective_description()
                if args.phase == "latent" and args.variant in LATENT_VARIANTS
                else (
                    greeth_joint_objective_description(shap)
                    if args.phase in ("global", "structure") and args.variant == "greeth"
                    else (
                        " + ".join(f"{w}*W1({f})" for f, w in shap.items())
                        if args.phase in ("global", "structure") else None
                    )
                )
            ),
            "tunables": list(tunables.keys()),
            "best_w1_norm_per_feature": {
                k: round(v, 4) for k, v in best_feat.items()
                if not str(k).startswith(("loss_contrib_", "covariance_std"))
            },
            "best_loss_contributions": {
                k.replace("loss_contrib_", ""): round(v, 4)
                for k, v in best_feat.items() if str(k).startswith("loss_contrib_")
            },
            "iters": args.iters,
            "objective": (
                latent_hybrid_objective_description()
                if args.variant in LATENT_VARIANTS and args.phase == "latent"
                else (
                    greeth_joint_objective_description(shap)
                    if args.variant == "greeth" and args.phase in ("structure", "global")
                    else "Loss = sum_f (SHAP_f * calibration_w1_norm_f)"
                )
            ),
            "loss_ref_scale": (
                dict(latent_loss_ref_scale(args.variant))
                if args.variant in LATENT_VARIANTS and args.phase == "latent"
                else (
                    dict(GREETH_LOSS_REF_SCALE)
                    if args.variant == "greeth" and args.phase in ("structure", "global")
                    else None
                )
            ),
            "hybrid_weights": (
                dict(LATENT_HYBRID_WEIGHTS)
                if args.variant in LATENT_VARIANTS and args.phase == "latent"
                else None
            ),
            "max_feature_loss_share": (
                GREETH_MAX_FEATURE_LOSS_SHARE
                if args.variant == "greeth" and args.phase in ("structure", "global")
                else None
            ),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"[auto] melhores params -> {out_path}")
        print(f"[auto] melhor loss = {best_loss:.4f}")
        print(f"[auto] W1_norm por feature: {best_feat}")


# --------------------------------------------------------------------------- #
# Graficos de convergencia (pos-calibracao ou modo --plot-report)              #
# --------------------------------------------------------------------------- #
def plot_calibration_report(report_path, plot_dir=None, dpi=300):
    """
    Gera graficos de analise da busca a partir do CSV de relatorio:
      - calibration_loss.png/svg  : loss por iteracao + melhor loss cumulativo
      - calibration_w1.png/svg    : evolucao de w1_<feature> por iteracao
      - calibration_summary.png/svg : figura 2x1 combinada para dissertacao
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(report_path)
    if "iter" not in df.columns or "loss" not in df.columns:
        raise ValueError(f"CSV invalido (faltam colunas iter/loss): {report_path}")

    plot_dir = plot_dir or os.path.dirname(os.path.abspath(report_path)) or "."
    os.makedirs(plot_dir, exist_ok=True)

    iters = df["iter"]
    w1_cols = [c for c in df.columns if c.startswith("w1_")]
    n_explore = int((df["phase"] == "explore").sum()) if "phase" in df.columns else 0

    def _phase_vline(ax):
        if 0 < n_explore < len(df):
            ax.axvline(n_explore - 0.5, color="0.45", ls="--", lw=1.2,
                       label="Fim exploracao")

    def _mark_improvements(ax, y_col="loss"):
        if "best_loss" not in df.columns:
            return
        prev_best = df["best_loss"].shift(1)
        improved = df["best_loss"] < prev_best.fillna(float("inf"))
        if improved.any():
            ax.scatter(
                df.loc[improved, "iter"],
                df.loc[improved, y_col],
                s=120, marker="*", color="gold", edgecolors="0.2",
                linewidths=0.8, zorder=5, label="Novo melhor",
            )

    def _save(fig, stem):
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(plot_dir, f"{stem}.{ext}"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # (a) Curva de convergencia
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(iters, df["loss"], "o-", alpha=0.45, color="0.55", markersize=5,
            label="Loss (iteracao)")
    if "best_loss" in df.columns:
        ax.plot(iters, df["best_loss"], "-", lw=2.2, color="tab:blue",
                label="Melhor loss (cumulativo)")
        _mark_improvements(ax, "loss")
    _phase_vline(ax)
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Loss SHAP ponderada")
    ax.set_title("Curva de convergencia da calibracao")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "calibration_loss")

    # (b) Evolucao individual W1 por feature
    if w1_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10.colors
        for i, col in enumerate(w1_cols):
            feat = col.replace("w1_", "", 1)
            ax.plot(
                iters, df[col], "o-", lw=2, markersize=5,
                color=colors[i % len(colors)], label=f"W1 {feat}",
            )
        _phase_vline(ax)
        ax.set_xlabel("Iteracao")
        ax.set_ylabel("W1_norm_balanced")
        ax.set_title("Evolucao das metricas por feature")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "calibration_w1")

        # Figura combinada 2x1
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(iters, df["loss"], "o-", alpha=0.45, color="0.55", markersize=5,
                       label="Loss (iteracao)")
        if "best_loss" in df.columns:
            axes[0].plot(iters, df["best_loss"], "-", lw=2.2, color="tab:blue",
                           label="Melhor loss (cumulativo)")
            _mark_improvements(axes[0], "loss")
        if 0 < n_explore < len(df):
            axes[0].axvline(n_explore - 0.5, color="0.45", ls="--", lw=1.2)
        axes[0].set_ylabel("Loss SHAP ponderada")
        axes[0].set_title("Convergencia da calibracao")
        axes[0].legend(loc="best", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        for i, col in enumerate(w1_cols):
            feat = col.replace("w1_", "", 1)
            axes[1].plot(
                iters, df[col], "o-", lw=2, markersize=5,
                color=colors[i % len(colors)], label=f"W1 {feat}",
            )
        if 0 < n_explore < len(df):
            axes[1].axvline(n_explore - 0.5, color="0.45", ls="--", lw=1.2,
                            label="Fim exploracao")
        axes[1].set_xlabel("Iteracao")
        axes[1].set_ylabel("W1_norm_balanced")
        axes[1].set_title("Evolucao por feature")
        axes[1].legend(loc="best", fontsize=9)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "calibration_summary")

    print(f"[auto] graficos de convergencia -> {plot_dir}/")
    return plot_dir


if __name__ == "__main__":
    main()
