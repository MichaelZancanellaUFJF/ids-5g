#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_generator.py
==================
Gerador de trafego sintetico de variantes Mirai (GRE-IP / GRE-ETH) para
ambiente Open5GS + UERANSIM, com modelo estocastico calibravel.

Substitui attack_GREIP.py e attack_GRE_ETH.py, corrigindo as limitacoes
identificadas na analise:

  * MULTIPLICIDADE DE FLUXOS: emula uma "botnet" com N IPs de origem
    (parametro botnet.size). Cada IP de origem vira um fluxo independente
    no extrator -> reproduz a distribuicao de pacotes/fluxo e duracao/fluxo
    do CICIoT2023, que tem muitos bots.

  * IAT ESTOCASTICO (ON/OFF + Poisson + Weibull + AR(1)): tempos entre
    pacotes modelados, com rajadas (burst), silencios (OFF) e engasgos
    (hiccup) -> reproduz bursts, periodos de silencio e autocorrelacao.

  * TAMANHO ESTOCASTICO: Log-Normal/Gaussiana com fracao "noisy" para spikes.

  * PACING PRECISO (modo live): usa um unico socket L3 reaproveitado +
    agendador com perf_counter (sleep + busy-wait hibrido). Isto resolve o
    problema de time.sleep()+send() por pacote nao realizar o rate programado.

Dois modos:
  --mode pcap : escreve um .pcap em nivel IP (como uma captura do ogstun),
                pronto para o converter.py / pipeline de extracao. NAO precisa
                do 5G; ideal para calibracao offline.
  --mode live : injeta o trafego ao vivo (rode na VM do UERANSIM).

Uso:
  python3 synth_generator.py --mode pcap --params params.json --out pcap_origin/capture.pcap
  sudo python3 synth_generator.py --mode live --params params.json --iface uesimtun0
"""

import argparse
import copy
import json
import os
import random
import socket
import time
from typing import Optional

import numpy as np
from scapy.all import IP, UDP, TCP, GRE, Ether, Raw, wrpcap, conf

GRE_CSUM = 0x8000  # +4 bytes
GRE_KEY = 0x2000   # +4 bytes


def _read_iface_mtu(iface: Optional[str]) -> Optional[int]:
    if not iface:
        return None
    name = str(iface).split("@", 1)[0]
    try:
        with open(f"/sys/class/net/{name}/mtu", encoding="ascii") as f:
            return int(f.read().strip())
    except OSError:
        return None


def _effective_live_mtu(iface: Optional[str] = None) -> int:
    """MTU efectivo: min(env, sysfs iface) - margem."""
    try:
        cap = int(os.environ.get("SYNTH_LIVE_MTU", "1500") or "1500")
    except ValueError:
        cap = 1500
    imtu = _read_iface_mtu(iface)
    mtu = min(cap, imtu) if imtu and imtu > 0 else cap
    try:
        margin = int(os.environ.get("SYNTH_LIVE_MTU_MARGIN", "8") or "8")
    except ValueError:
        margin = 8
    return max(576, mtu - margin)


def _live_mtu_limit() -> int:
    """MTU IP no inject live (uesimtun0). Default 1500 se env ausente."""
    return _effective_live_mtu(None)


def _l3_send_ip_bytes(sock, data: bytes, iface: Optional[str] = None) -> None:
    """
    Envia datagrama IP em bytes crus no TUN/raw.

    Scapy L3socket.send(IP) usa raw(ll(p)) e causa EMSGSIZE no uesimtun0.
    No TUN o sendto deve usar (nome_iface, 0), nao (dst_ip, 0) — senao ENODEV.
    """
    if len(data) < 20:
        sock.send(IP(data))
        return
    ifname = iface or getattr(sock, "iface", None) or conf.iface
    if ifname:
        sock.outs.sendto(data, (ifname, 0))
        return
    dst = socket.inet_ntoa(data[16:20])
    sock.outs.sendto(data, (dst, 0))


def _live_mtu_max_payload(l3_header_bytes: int) -> Optional[int]:
    """Cap payload no inject L3. Com SYNTH_LIVE_MTU=0 desactiva (nao recomendado)."""
    try:
        mtu = int(os.environ.get("SYNTH_LIVE_MTU", "1500") or "1500")
    except ValueError:
        mtu = 1500
    if mtu <= 0:
        return None
    return max(0, mtu - int(l3_header_bytes))


def _clamp_ip_wire_to_mtu(data: bytes, mtu: Optional[int] = None) -> bytes:
    """Reduz payload TCP/UDP/Raw para len(IP) <= mtu (ultima defesa no send_live)."""
    mtu = mtu or _live_mtu_limit()
    if len(data) <= mtu:
        return data
    pkt = IP(data)
    if len(pkt) <= mtu:
        return bytes(pkt)
    while len(pkt) > mtu:
        trimmed = False
        if TCP in pkt and pkt[TCP].payload is not None:
            pl = bytes(pkt[TCP].payload)
            if not pl:
                break
            pkt[TCP].remove_payload()
            pkt = pkt / Raw(load=pl[: max(0, len(pl) - 1)])
            trimmed = True
        elif UDP in pkt and pkt[UDP].payload is not None:
            pl = bytes(pkt[UDP].payload)
            if not pl:
                break
            pkt[UDP].remove_payload()
            pkt = pkt / Raw(load=pl[: max(0, len(pl) - 1)])
            trimmed = True
        elif pkt.payload is not None and hasattr(pkt.payload, "load"):
            ld = pkt.payload.load
            if not ld:
                break
            pkt.payload.load = ld[: max(0, len(ld) - 1)]
            trimmed = True
        else:
            break
        if IP in pkt:
            del pkt[IP].len
            del pkt[IP].chksum
        if TCP in pkt:
            del pkt[TCP].chksum
        if UDP in pkt:
            del pkt[UDP].len
            del pkt[UDP].chksum
        if not trimmed:
            break
    out = bytes(pkt)
    return out if len(out) <= mtu else out[:mtu]


# --------------------------------------------------------------------------- #
# Carregamento / utilidades                                                   #
# --------------------------------------------------------------------------- #
def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base, override):
    """Merge recursivo preservando compatibilidade com o JSON legado."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def materialize_variant_params(params, variant=None):
    """Materializa params da variante (variants.* sobrescreve base)."""
    p = copy.deepcopy(params)
    variant = variant or p.get("variant")
    if not variant:
        return p
    p["variant"] = variant
    vcfg = (p.get("variants") or {}).get(variant)
    if not vcfg:
        return p
    merge_keys = (
        "botnet", "flow", "iat", "size", "gre", "ttl", "gre_flow",
        "dynamic_streams", "capture_overhead", "inter_flow_delay", "benign",
    )
    for key in merge_keys:
        if key not in vcfg:
            continue
        if isinstance(vcfg[key], dict) and isinstance(p.get(key), dict):
            p[key] = _deep_merge(p.get(key, {}), vcfg[key])
        else:
            p[key] = copy.deepcopy(vcfg[key])
    if "botnet" in vcfg:
        p["botnet"] = {**p.get("botnet", {}), **vcfg["botnet"]}
    _enrich_markov_role_probs(p)
    return p



def make_ip_options(nbytes):
    """Opcoes IP (NOP repetido) para inflar o cabecalho IP externo."""
    if nbytes <= 0:
        return b""
    return b"\x01" * nbytes


def pick_gre_variant(rng, gre_cfg):
    """Sorteia tamanho/flags do cabecalho GRE. Retorna (flags, key_or_None, gre_len)."""
    w4 = gre_cfg["flag_weight_4"]
    w8 = gre_cfg["flag_weight_8"]
    r = rng.random()
    if r < w4:
        return 0x0000, None, 4
    elif r < w4 + w8:
        return GRE_KEY, rng.getrandbits(32), 8
    else:
        return GRE_KEY | GRE_CSUM, rng.getrandbits(32), 12


# --------------------------------------------------------------------------- #
# Amostragem estocastica — tamanhos de pacote (multimodal + correlacao)       #
# --------------------------------------------------------------------------- #
def _component_log_params(comp):
    """Normaliza mean_log/std_log (natural) ou mean_log10/std_log10 do JSON."""
    if "mean_log" in comp:
        mean_log = float(comp["mean_log"])
    elif "mean_log10" in comp:
        mean_log = float(comp["mean_log10"]) * np.log(10.0)
    elif "mean" in comp:
        mean_log = float(np.log(max(float(comp["mean"]), 1.0)))
    else:
        raise ValueError("componente GMM sem mean_log, mean_log10 ou mean")
    if "std_log" in comp:
        std_log = float(comp["std_log"])
    elif "std_log10" in comp:
        std_log = float(comp["std_log10"]) * np.log(10.0)
    elif "std" in comp:
        std_log = float(comp["std"]) / max(float(comp.get("mean", np.exp(mean_log))), 1.0)
    else:
        std_log = 0.05
    return mean_log, max(1e-6, std_log)


def _is_hybrid_size_components(raw):
    """True se components usam kind/type (platô/normal), nao GMM puro."""
    return any(
        str(c.get("kind", c.get("type", ""))).lower()
        in ("constant", "fixed", "plateau", "normal", "gaussian", "lognormal", "lognorm")
        for c in (raw or [])
    )


def _normalize_hybrid_components(size_cfg):
    """
    Componentes do modelo hibrido (platô + contínuas).
    kind: constant | normal | lognormal
    """
    raw = size_cfg.get("components") or []
    if not raw:
        raise ValueError("size.model=hybrid requer size.components")
    out = []
    for comp in raw:
        kind = str(comp.get("kind", comp.get("type", "lognormal"))).lower()
        entry = {"weight": float(comp.get("weight", 1.0)), "kind": kind}
        if kind in ("constant", "fixed", "plateau"):
            entry["value"] = int(comp.get("value", comp.get("constant", 592)))
        elif kind in ("normal", "gaussian"):
            entry["mean"] = float(comp["mean"])
            entry["std"] = max(1e-6, float(comp.get("std", 1.0)))
        elif kind in ("lognormal", "lognorm"):
            mean_log, std_log = _component_log_params(comp)
            entry["mean_log"] = mean_log
            entry["std_log"] = std_log
        else:
            raise ValueError(f"kind de componente hibrido desconhecido: {kind}")
        out.append(entry)
    total = sum(c["weight"] for c in out)
    if total <= 0:
        raise ValueError("componentes hibridos com peso total zero")
    for c in out:
        c["weight"] /= total
    return out


def _normalize_size_components(size_cfg):
    raw = size_cfg.get("components") or size_cfg.get("mixture_components") or []
    if not raw or _is_hybrid_size_components(raw):
        return []
    out = []
    for comp in raw:
        mean_log, std_log = _component_log_params(comp)
        out.append({
            "weight": float(comp.get("weight", 1.0)),
            "mean_log": mean_log,
            "std_log": std_log,
        })
    total = sum(c["weight"] for c in out)
    if total <= 0:
        raise ValueError("componentes de tamanho com peso total zero")
    for c in out:
        c["weight"] /= total
    return out


def sample_packet_size(rng, np_rng, size_cfg):
    """Tamanho-alvo legado (lognorm + cauda pequena + noisy). Mantido para retrocompatibilidade."""
    if rng.random() < float(size_cfg.get("small_fraction", 0.0)):
        smu = float(size_cfg.get("small_mu", 4.05))
        ssig = float(size_cfg.get("small_sigma", 0.12))
        val = np_rng.lognormal(smu, ssig)
    elif rng.random() < size_cfg["noisy_fraction"]:
        val = np_rng.normal(size_cfg["noisy_mu"], size_cfg["noisy_sigma"])
    else:
        if size_cfg["dist"] == "lognorm":
            val = np_rng.lognormal(size_cfg["mu"], size_cfg["sigma"])
        else:
            val = np_rng.normal(size_cfg["mu"], size_cfg["sigma"])
    jitter = float(size_cfg.get("jitter_std", 0.0))
    if jitter > 0:
        val += np_rng.normal(0.0, jitter)
    return int(max(size_cfg["min"], min(val, size_cfg["max"])))


class FlowSizeSampler:
    """
    Mistura multimodal de tamanhos (GMM log-normal, quantis empiricos ou bootstrap)
    com correlacao intra-fluxo via AR(1) na inovacao e persistencia de componente.

    Blocos de `block_size` pacotes (default 20, alinhado ao extrator CIC) preservam
    a estrutura de Covariance/Variance entre streams incoming/outgoing.
    """

    def __init__(self, rng, np_rng, size_cfg, ds_cfg=None):
        self.rng = rng
        self.np_rng = np_rng
        self.size_cfg = size_cfg
        self.ds_cfg = ds_cfg or {}
        self.model = str(size_cfg.get("model", "legacy")).lower()
        self.min_sz = int(size_cfg.get("min", 42))
        self.max_sz = int(size_cfg.get("max", 1500))
        self.jitter = float(size_cfg.get("jitter_std", 0.0))

        ds_on = bool(self.ds_cfg.get("enabled", False))
        self.ar1_rho = float(size_cfg.get(
            "ar1_rho",
            self.ds_cfg.get("size_correlation", 0.0) if ds_on else size_cfg.get("correlation", 0.0),
        ))
        self.ar1_rho = min(max(self.ar1_rho, 0.0), 0.999)
        self.comp_rho = float(size_cfg.get("component_rho", 0.92))
        self.extra_std = float(self.ds_cfg.get("size_std", 0.0)) if ds_on else 0.0

        self.block_size = max(1, int(self.ds_cfg.get("block_size", 20))) if ds_on else 1
        self.block = []
        self.pos = 0
        self.z = 0.0
        self.u_prev = 0.5
        self.comp_idx = None
        self.flow_component_idx = None

        self.components = []
        self.quantiles = None
        self.bootstrap_values = None
        if self.model in ("hybrid", "hybrid_plateau", "plateau"):
            self.components = _normalize_hybrid_components(size_cfg)
        elif self.model in ("gmm", "gmm_log", "mixture"):
            self.components = _normalize_size_components(size_cfg)
            if not self.components:
                raise ValueError("size.model=gmm requer size.components")
        elif self.model == "empirical_quantile":
            q = size_cfg.get("quantiles", [])
            if not q:
                raise ValueError("size.model=empirical_quantile requer size.quantiles")
            self.quantiles = (
                np.array([float(row[0]) for row in q], dtype=float),
                np.array([float(row[1]) for row in q], dtype=float),
            )
        elif self.model == "bootstrap":
            vals = size_cfg.get("values", [])
            if not vals:
                raise ValueError("size.model=bootstrap requer size.values")
            self.bootstrap_values = np.asarray(vals, dtype=float)

        if size_cfg.get("assign_component_per_flow", True) and self.components:
            self.flow_component_idx = int(_weighted_choice_index(rng, self.components))
            self.comp_idx = self.flow_component_idx

    def next_size(self):
        if self.pos >= len(self.block):
            self._refill_block()
            self.pos = 0
        val = self.block[self.pos]
        self.pos += 1
        return val

    def _refill_block(self):
        self.block = [self._sample_one() for _ in range(self.block_size)]

    def _ar1_innovation(self):
        innov = self.np_rng.normal(0.0, 1.0) * np.sqrt(max(1e-9, 1.0 - self.ar1_rho ** 2))
        self.z = self.ar1_rho * self.z + innov
        return self.z

    def _pick_component(self):
        if self.flow_component_idx is not None:
            return self.components[self.flow_component_idx]
        if self.comp_idx is None or self.rng.random() > self.comp_rho:
            self.comp_idx = int(_weighted_choice_index(self.rng, self.components))
        return self.components[self.comp_idx]

    def _sample_gmm(self):
        comp = self._pick_component()
        z = self._ar1_innovation()
        std_log = comp["std_log"]
        if self.extra_std > 0:
            mean_bytes = np.exp(comp["mean_log"])
            std_log = float(np.hypot(std_log, self.extra_std / max(mean_bytes, 1.0)))
        log_val = comp["mean_log"] + std_log * z
        return float(np.exp(log_val))

    def _sample_hybrid(self):
        comp = self._pick_component()
        kind = comp["kind"]
        if kind in ("constant", "fixed", "plateau"):
            return float(comp["value"])
        z = self._ar1_innovation()
        if kind in ("normal", "gaussian"):
            val = float(comp["mean"] + comp["std"] * z)
        else:
            val = float(np.exp(comp["mean_log"] + comp["std_log"] * z))
        if self.jitter > 0:
            val += float(self.np_rng.normal(0.0, self.jitter))
        return val

    def _sample_empirical_correlated(self):
        from scipy.stats import norm as sp_norm
        eps = self.np_rng.normal(0.0, 1.0)
        u = sp_norm.cdf(self.ar1_rho * sp_norm.ppf(self.u_prev) + np.sqrt(max(1e-9, 1.0 - self.ar1_rho ** 2)) * eps)
        self.u_prev = float(np.clip(u, 1e-6, 1.0 - 1e-6))
        qs, vals = self.quantiles
        return float(np.interp(self.u_prev, qs, vals))

    def _sample_one(self):
        if self.model == "legacy":
            val = float(sample_packet_size(self.rng, self.np_rng, self.size_cfg))
        elif self.model in ("gmm", "gmm_log", "mixture"):
            val = self._sample_gmm()
        elif self.model in ("hybrid", "hybrid_plateau", "plateau"):
            val = self._sample_hybrid()
        elif self.model == "empirical_quantile":
            val = self._sample_empirical_correlated()
        elif self.model == "bootstrap":
            val = float(self.bootstrap_values[int(self.np_rng.integers(0, len(self.bootstrap_values)))])
        else:
            raise ValueError(f"size.model nao suportado: {self.model}")
        if self.jitter > 0:
            val += float(self.np_rng.normal(0.0, self.jitter))
        return int(max(self.min_sz, min(val, self.max_sz)))


def _weighted_choice_index(rng, components):
    total = sum(float(c.get("weight", 1.0)) for c in components)
    r = rng.random() * total
    acc = 0.0
    for i, comp in enumerate(components):
        acc += float(comp.get("weight", 1.0))
        if r <= acc:
            return i
    return len(components) - 1


# Alias legado — BlockSizeSampler era global; preferir FlowSizeSampler por fluxo.
BlockSizeSampler = FlowSizeSampler


def make_flow_size_sampler(rng, np_rng, size_cfg, ds_cfg=None):
    """Instancia um sampler de tamanho correlacionado para um unico fluxo/bot."""
    ds_cfg = ds_cfg or {}
    model = str(size_cfg.get("model", "legacy")).lower()
    if model in ("gmm", "gmm_log", "mixture", "hybrid", "hybrid_plateau", "plateau",
                 "empirical_quantile", "bootstrap"):
        return FlowSizeSampler(rng, np_rng, size_cfg, ds_cfg)
    if ds_cfg.get("enabled", False):
        legacy_cfg = dict(size_cfg, model="legacy")
        return FlowSizeSampler(rng, np_rng, legacy_cfg, ds_cfg)
    return None


def make_size_sampler(rng, np_rng, params):
    return make_flow_size_sampler(rng, np_rng, params["size"], params.get("dynamic_streams"))


def next_packet_target_size(rng, np_rng, size_cfg, size_sampler=None):
    if size_sampler is not None:
        return size_sampler.next_size()
    return sample_packet_size(rng, np_rng, size_cfg)


def sample_flow_duration(rng, flow_cfg, np_rng=None):
    """
    Amostra duracao alvo de um fluxo a partir de flow.duration_model.

    Modelos: lognorm (legado), zero_inflated_gmm_log10, gmm_log10, mixture,
    uniform, constant.
    """
    if np_rng is None:
        np_rng = np.random.default_rng(int(rng.random() * 1e9))

    model = str(flow_cfg.get("duration_model", "lognorm")).lower()
    d_min = float(flow_cfg.get("duration_min", 1e-9))
    d_max = float(flow_cfg.get("duration_max", 300.0))
    eps = float(flow_cfg.get("duration_epsilon", 1e-9))

    if model == "constant":
        d = float(flow_cfg.get("duration_value", flow_cfg.get("duration_min", 0.0)))
    elif model == "uniform":
        lo = float(flow_cfg.get("duration_uniform_low", d_min))
        hi = float(flow_cfg.get("duration_uniform_high", d_max))
        d = rng.uniform(lo, hi)
    elif model in ("zero_inflated_gmm_log10", "zero_inflated_gmm"):
        if rng.random() < float(flow_cfg.get("duration_zero_fraction", 0.0)):
            return eps
        d = _sample_duration_gmm_log10(rng, np_rng, flow_cfg)
        return max(d_min, min(float(d), d_max))
    elif model in ("gmm_log10", "gmm"):
        d = _sample_duration_gmm_log10(rng, np_rng, flow_cfg)
    elif model == "mixture":
        comps = flow_cfg.get("duration_mixture") or flow_cfg.get("duration_components") or []
        if not comps:
            d = rng.lognormvariate(
                flow_cfg.get("duration_lognorm_mu", 0.0),
                flow_cfg.get("duration_lognorm_sigma", 1.0),
            )
        else:
            comp = comps[int(_weighted_choice_index(rng, comps))]
            d = _sample_duration_mixture_component(rng, np_rng, comp)
    else:
        d = rng.lognormvariate(
            flow_cfg.get("duration_lognorm_mu", 0.0),
            flow_cfg.get("duration_lognorm_sigma", 1.0),
        )

    return max(d_min, min(float(d), d_max))
    print("duration =", d)




def _sample_duration_gmm_log10(rng, np_rng, flow_cfg):
    comps = flow_cfg.get("duration_components") or []
    if not comps:
        return float(flow_cfg.get("duration_epsilon", 1e-9))
    comp = comps[int(_weighted_choice_index(rng, comps))]
    return float(10 ** np_rng.normal(comp["mean_log10"], comp["std_log10"]))


def _sample_duration_mixture_component(rng, np_rng, comp):
    kind = str(comp.get("model", comp.get("kind", "lognorm10"))).lower()
    if kind == "uniform":
        return rng.uniform(float(comp["low"]), float(comp["high"]))
    if kind in ("constant", "fixed"):
        return float(comp.get("value", 0.0))
    if kind in ("lognorm", "lognormal"):
        return float(np.exp(np_rng.normal(comp["mu"], comp["sigma"])))
    if kind in ("lognorm10", "log10", "gmm_log10"):
        return float(10 ** np_rng.normal(comp["mean_log10"], comp["std_log10"]))
    if kind == "weibull":
        return float(np_rng.weibull(float(comp["k"])) * float(comp.get("scale", 1.0)))
    return float(10 ** np_rng.normal(comp.get("mean_log10", -2.0), comp.get("std_log10", 1.0)))


def _iat_cfg_with_defaults(iat_cfg):
    """Preenche chaves exigidas por sample_iat_sequence."""
    defaults = {
        "ar1_rho": 0.2,
        "jitter": 0.05,
        "hiccup_p": 0.02,
        "hiccup_mult_lo": 2.0,
        "hiccup_mult_hi": 5.0,
        "burst_p": 0.1,
        "burst_mult_lo": 0.01,
        "burst_mult_hi": 0.1,
        "p_off": 0.03,
        "off_weibull_k": 1.2,
        "off_weibull_scale": 0.002,
    }
    return {**defaults, **(iat_cfg or {})}


def sample_iat_sequence(rng, np_rng, iat_cfg, n_packets, flow_duration):
    """
    Gera a sequência de IATs (Inter-Arrival Times).

    Se flow_duration for praticamente zero, todos os pacotes da janela
    recebem o mesmo timestamp, preservando o comportamento do CICIoT.
    """

    if n_packets <= 1:
        return []

    # -------------------------------------------------
    # Fluxo instantâneo (zero-inflated)
    # -------------------------------------------------
    if flow_duration <= 1e-8:
        return [0.0] * (n_packets - 1)

    iat_cfg = _iat_cfg_with_defaults(iat_cfg)

    base_iat = flow_duration / (n_packets - 1)

    rho = iat_cfg["ar1_rho"]
    jitter = iat_cfg["jitter"]

    iats = []
    eps_prev = 0.0

    for _ in range(n_packets - 1):

        # Processo exponencial em torno do IAT base
        x = np_rng.exponential(base_iat)

        # Jitter AR(1)
        eps = rho * eps_prev + (1.0 - rho) * np_rng.normal(0.0, jitter)
        eps_prev = eps

        x *= max(0.05, 1.0 + eps)

        r = rng.random()

        if r < iat_cfg["hiccup_p"]:
            x *= rng.uniform(
                iat_cfg["hiccup_mult_lo"],
                iat_cfg["hiccup_mult_hi"],
            )

        elif r < iat_cfg["hiccup_p"] + iat_cfg["burst_p"]:
            x *= rng.uniform(
                iat_cfg["burst_mult_lo"],
                iat_cfg["burst_mult_hi"],
            )

        if rng.random() < iat_cfg["p_off"]:
            x += (
                np_rng.weibull(iat_cfg["off_weibull_k"])
                * iat_cfg["off_weibull_scale"]
            )

        iats.append(max(0.0, x))

    total = sum(iats)

    if total > 0:
        scale = flow_duration / total
        iats = [v * scale for v in iats]

    return iats




def _sample_numeric_model(rng, np_rng, cfg, model, prefix=""):
    model = str(model or cfg.get(f"{prefix}_model", "normal")).lower()
    if model == "poisson":
        lam = float(cfg.get(f"{prefix}_poisson_lambda", cfg.get("packets_poisson_lambda", 10.0)))
        return float(np_rng.poisson(max(0.0, lam)))
    mean = float(cfg.get(f"{prefix}_mean", cfg.get("packets_mean", 10.0)))
    std = float(cfg.get(f"{prefix}_std", cfg.get("packets_std", 1.0)))
    return float(np_rng.normal(mean, std))

def _is_plateau_component(comp):
    kind = str(comp.get("kind", comp.get("type", ""))).lower()
    return kind in ("constant", "fixed", "plateau")


MARKOV_ROLE_ORDER = ("plateau", "normal", "tail")
DEFAULT_MARKOV_TRANSITION_3 = (
    (0.92, 0.06, 0.02),
    (0.08, 0.88, 0.04),
    (0.20, 0.75, 0.05),
)


class MinTraceCollector:
    """Rastreio opcional da cadeia Min (SYNTH_TRACE_MIN=1 ou window_hetero.min_trace)."""

    def __init__(self, sample_limit=12):
        self.sample_limit = max(1, int(sample_limit))
        self.samples = []
        self.regime_counts = {"plateau": 0, "normal": 0, "tail": 0}
        self.comp_kind_counts = {}
        self.floors = []

    def record_floor(self, window_idx, floor, regime, comp, role_probs, effective_weights):
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1
        kind = str(comp.get("kind", "?"))
        self.comp_kind_counts[kind] = self.comp_kind_counts.get(kind, 0) + 1
        self.floors.append(int(floor))
        if len(self.samples) >= self.sample_limit:
            return
        self.samples.append({
            "wi": int(window_idx),
            "floor": int(floor),
            "regime": regime,
            "comp_kind": kind,
            "comp": {k: comp.get(k) for k in ("kind", "value", "mean", "std", "mean_log", "std_log")},
            "role_probs": dict(role_probs or {}),
            "effective_weights": list(effective_weights or []),
        })

    def record_sizes(self, window_idx, floor, sizes):
        if not self.samples:
            return
        for s in self.samples:
            if s["wi"] == int(window_idx):
                s["min_after_intra"] = int(min(sizes)) if sizes else int(floor)
                s["max_after_intra"] = int(max(sizes)) if sizes else int(floor)
                s["n_sizes"] = len(sizes)
                break

    def summarize(self):
        if not self.floors:
            return
        import numpy as np
        arr = np.array(self.floors)
        print(
            "[min_trace] floors: "
            f"n={len(arr)} med={float(np.median(arr)):.0f} "
            f"p25={float(np.percentile(arr, 25)):.0f} "
            f"p75={float(np.percentile(arr, 75)):.0f} "
            f"@592={float((arr == 592).mean()) * 100:.1f}% "
            f"regime={self.regime_counts} comp_kind={self.comp_kind_counts}",
            flush=True,
        )
        for s in self.samples:
            print(
                f"[min_trace] wi={s['wi']} floor={s['floor']} regime={s['regime']} "
                f"kind={s['comp_kind']} comp={s['comp']} "
                f"min_after_intra={s.get('min_after_intra', '?')} "
                f"role_probs={s.get('role_probs')}",
                flush=True,
            )


def _min_trace_enabled(params):
    import os
    wh = (params.get("dynamic_streams") or {}).get("window_hetero") or {}
    if wh.get("min_trace"):
        return True
    return os.environ.get("SYNTH_TRACE_MIN", "").strip().lower() in ("1", "true", "yes")


def _init_min_trace(params):
    if not _min_trace_enabled(params):
        return None
    wh = (params.get("dynamic_streams") or {}).get("window_hetero") or {}
    limit = int(wh.get("min_trace_samples", 12))
    return MinTraceCollector(sample_limit=limit)


def _log_min_generation_path(params):
    """Documenta qual bloco controla Min no greeth window-hetero."""
    size_cfg = params.get("size") or {}
    wh = (params.get("dynamic_streams") or {}).get("window_hetero") or {}
    rp = size_cfg.get("markov_role_probs") or {}
    print(
        "[min] cadeia greeth: "
        "1) sample_window_floor(size.components + markov_role_probs de markov_gmm) -> floor; "
        "2) expand_intra_window_sizes(markov_gmm jitter/spike) -> sizes, sizes[0]=floor; "
        "3) build_packet(target_size) -> Tot size; Min CSV = min(Tot size) na janela AGG",
        flush=True,
    )
    print(
        f"[min] tot_sum_profile={wh.get('tot_sum_profile', 'markov_gmm')} | "
        f"role_probs plateau/normal/tail="
        f"{rp.get('plateau', '?')}/{rp.get('normal', '?')}/{rp.get('tail', '?')} | "
        f"component weights JSON sao reponderados por role_prob (nao ignorados)",
        flush=True,
    )

def _hybrid_component_role(comp):
    """Classifica componente hibrido: platô (592), normal (~516) ou cauda (364)."""
    kind = str(comp.get("kind", comp.get("type", ""))).lower()
    if kind in ("normal", "gaussian"):
        return "normal"
    if kind in ("constant", "fixed", "plateau"):
        v = int(comp.get("value", comp.get("constant", 592)))
        return "tail" if v <= 420 else "plateau"
    return "tail"


def _resolve_markov_role_probs(mg, hybrid=None):
    """
    Probabilidades de entrada por regime (Min + estado inicial Markov).

    Prioridade: markov_gmm.plateau_prob/normal_prob/tail_prob > pesos hibridos.
    """
    mg = mg or {}
    if any(mg.get(k) is not None for k in ("plateau_prob", "normal_prob", "tail_prob")):
        pl = float(mg.get("plateau_prob", 0.46))
        no = float(mg.get("normal_prob", 0.49))
        ta = float(mg.get("tail_prob", 0.05))
        total = pl + no + ta
        if total <= 0:
            pl, no, ta = 0.46, 0.49, 0.05
            total = 1.0
        return {"plateau": pl / total, "normal": no / total, "tail": ta / total}
    acc = {r: 0.0 for r in MARKOV_ROLE_ORDER}
    if hybrid:
        for comp in hybrid:
            acc[_hybrid_component_role(comp)] += float(comp.get("weight", 0.0))
    total = sum(acc.values())
    if total <= 0:
        return {"plateau": 1 / 3, "normal": 1 / 3, "tail": 1 / 3}
    return {r: acc[r] / total for r in MARKOV_ROLE_ORDER}


def _enrich_markov_role_probs(p):
    """Propaga plateau/normal/tail_prob para size.markov_role_probs (Min + Markov)."""
    ds = p.get("dynamic_streams") or {}
    wh = ds.get("window_hetero") or {}
    mg = wh.get("markov_gmm") or {}
    size = p.get("size") or {}
    raw = size.get("components") or []
    if not mg or not raw or not _is_hybrid_size_components(raw):
        return
    if not any(mg.get(k) is not None for k in ("plateau_prob", "normal_prob", "tail_prob")):
        return
    try:
        hybrid = _normalize_hybrid_components(size)
    except ValueError:
        return
    probs = _resolve_markov_role_probs(mg, hybrid)
    size["markov_role_probs"] = probs
    mg["plateau_prob"] = probs["plateau"]
    mg["normal_prob"] = probs["normal"]
    mg["tail_prob"] = probs["tail"]


def _apply_role_probs_to_hybrid_components(components, role_probs):
    """Repondera componentes: massa por role_prob, split intra-role pelos weight JSON."""
    out = [dict(c) for c in components]
    role_base = {r: 0.0 for r in MARKOV_ROLE_ORDER}
    for comp in out:
        role_base[_hybrid_component_role(comp)] += float(comp.get("weight", 1.0))
    for comp in out:
        role = _hybrid_component_role(comp)
        rp = float(role_probs.get(role, 0.0))
        base = float(comp.get("weight", 1.0))
        denom = role_base.get(role, 0.0) or 1.0
        comp["weight"] = rp * (base / denom)
    total = sum(float(c["weight"]) for c in out)
    if total <= 0:
        return out
    for comp in out:
        comp["weight"] = float(comp["weight"]) / total
    return out


def _expand_ipv4(base, offset):
    parts = [p for p in base.split(".") if p != ""]
    a = int(parts[0])
    b = int(parts[1])
    third_base = int(parts[2]) if len(parts) >= 3 else 0
    combined = third_base * 256 + int(offset)
    o3 = (combined // 256) % 256
    o4 = combined % 256
    return f"{a}.{b}.{o3}.{o4}"


def _live_routing_flow_ips(live_rt, flow_idx=0, *, window_id=None):
    """Par src/dst roteavel no testbed; delega politica live a live_ue_profile."""
    from live_ue_profile import live_routing_flow_ips

    return live_routing_flow_ips(live_rt, flow_idx, window_id=window_id)


def _aggregation_window_size(size_cfg, ds_cfg=None, gre_flow=None):
    gre_flow = gre_flow or {}
    ds_cfg = ds_cfg or {}
    if "aggregation_packets" in gre_flow:
        return max(1, int(gre_flow["aggregation_packets"]))
    if "plateau_min_packets" in size_cfg:
        return max(1, int(size_cfg["plateau_min_packets"]))
    if ds_cfg.get("block_size"):
        return max(1, int(ds_cfg["block_size"]))
    return 20

def sample_flow_packet_count(np_rng, flow_cfg):
    model = flow_cfg.get("packet_count_model", flow_cfg.get("packets_model", "normal"))
    if model == "normal":
        n = int(np_rng.normal(flow_cfg["packets_mean"], flow_cfg["packets_std"]))
    else:
        # Usa RNG python apenas para escolha de mistura; aqui um Random deterministico derivado evita
        # alterar a assinatura da funcao legada.
        local_rng = random.Random(int(np_rng.integers(0, 2**31 - 1)))
        n = int(round(_sample_numeric_model(local_rng, np_rng, flow_cfg, model, "packets")))
    n = max(int(flow_cfg.get("packets_min", 1)), n)
    packets_max = flow_cfg.get("packets_max")
    if packets_max is not None:
        n = min(n, int(packets_max))
    return n


def _merge_gre_flow_packet_cfg(gre_flow, base_flow):
    """Unifica gre_flow.packets_* com flow.* para sample_flow_packet_count."""
    cfg = dict(base_flow)
    key_map = {
        "packets_model": "packet_count_model",
        "packets_mean_long": "packets_mean",
        "packets_std_long": "packets_std",
        "packets_min_long": "packets_min",
        "packets_max_long": "packets_max",
        "packets_poisson_lambda": "packets_poisson_lambda",
        "packets_quantiles": "packets_quantiles",
        "packets_components": "packets_components",
        "packets_values": "packets_values",
    }
    for src, dst in key_map.items():
        if src in gre_flow:
            cfg[dst] = gre_flow[src]
    if "packet_count_model" not in cfg and "packets_model" in gre_flow:
        cfg["packet_count_model"] = gre_flow["packets_model"]
    return cfg


def sample_gre_flow_packet_count(rng, np_rng, gre_flow, base_flow, single_fraction):
    """
    Amostra pacotes por fluxo GRE: massa em single-packet + modelo configuravel
    (normal, poisson, GMM, bootstrap, quantis empiricos) para fluxos longos.
    """
    if rng.random() < single_fraction:
        return 1
    cfg = _merge_gre_flow_packet_cfg(gre_flow, base_flow)
    return sample_flow_packet_count(np_rng, cfg)


def _flow_quantile_spread(n_flows, mean, std, lo=None, hi=None):
    """Distribui valores por fluxo via quantis espacados (heterogeneidade inter-fluxo)."""
    if n_flows <= 0:
        return []
    if std is None or std <= 0:
        return [mean] * n_flows
    from scipy.stats import norm as sp_norm
    quantiles = np.linspace(1.0 / (n_flows + 1), n_flows / (n_flows + 1), n_flows)
    vals = sp_norm.ppf(quantiles, mean, std)
    if lo is not None or hi is not None:
        vals = np.clip(vals, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    return vals.tolist()

def sample_window_sizes(rng, np_rng, size_cfg, agg_n):
    """
    Amostra tamanhos de uma janela AGG inteira (alinhado ao extrator CIC).
    Retorna (lista_de_tamanhos, regime) com regime in plateau|normal|tail.
    """
    floor, regime = sample_window_floor(rng, np_rng, size_cfg)
    return [floor] * agg_n, regime


def _parse_empirical_quantiles(size_cfg):
    """Extrai pares (prob, valor) de quantiles_empirical_min ou quantiles."""
    q = size_cfg.get("quantiles_empirical_min") or size_cfg.get("quantiles") or []
    if not q:
        return None
    ps = np.array([float(row[0]) for row in q], dtype=float)
    vs = np.array([float(row[1]) for row in q], dtype=float)
    order = np.argsort(ps)
    return ps[order], vs[order]


def _floor_regime_from_value(floor):
    """Classifica regime Min para trace/Markov (platô 592, normal, cauda)."""
    floor = int(floor)
    if floor >= 588:
        return "plateau"
    if floor >= 460:
        return "normal"
    return "tail"


def _sample_floor_empirical_quantile(np_rng, size_cfg):
    """
    Amostra Min via inverse-CDF empirica (quantis CICIoT2023).

    Reproduz assimetria, platô em 592 e cauda inferior sem mistura gaussiana simples.
    """
    parsed = _parse_empirical_quantiles(size_cfg)
    if parsed is None:
        raise ValueError("floor_model=empirical_quantile requer quantiles_empirical_min")
    ps, vs = parsed
    q_lo = float(size_cfg.get("floor_quantile_lo", 0.0))
    q_hi = float(size_cfg.get("floor_quantile_hi", 1.0))
    q_lo = float(np.clip(q_lo, 0.0, 0.95))
    q_hi = float(np.clip(q_hi, q_lo + 1e-6, 1.0))
    u = q_lo + float(np_rng.random()) * (q_hi - q_lo)
    val = float(np.interp(u, ps, vs))
    clip_max = size_cfg.get("floor_clip_max")
    if clip_max is not None:
        val = min(val, float(clip_max))
    clip_min = size_cfg.get("floor_clip_min")
    if clip_min is not None:
        val = max(val, float(clip_min))
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 2200))
    jitter = float(size_cfg.get("floor_jitter_std", 0.0))
    if jitter > 0:
        val += float(np_rng.normal(0.0, jitter))
    floor = int(max(min_sz, min(round(val), max_sz)))
    return floor, _floor_regime_from_value(floor)


def sample_window_floor(rng, np_rng, size_cfg, trace=None, window_idx=None):
    """Sorteia o piso de Min para uma janela AGG (1 componente da mistura hibrida)."""
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 1500))
    floor_model = str(size_cfg.get("floor_model", "hybrid")).lower()
    if floor_model in ("empirical_quantile", "empirical", "quantile"):
        floor, regime = _sample_floor_empirical_quantile(np_rng, size_cfg)
        if trace is not None and window_idx is not None:
            trace.record_floor(
                window_idx, floor, regime, {"kind": "empirical_quantile"}, None,
                [("empirical_quantile", 1.0)],
            )
        return floor, regime

    raw = size_cfg.get("components") or []
    if not raw:
        val = float(np.exp(np_rng.normal(
            float(size_cfg.get("mu", 6.3)),
            float(size_cfg.get("sigma", 0.05)),
        )))
        return int(max(min_sz, min(val, max_sz))), "tail"

    components = _normalize_hybrid_components(size_cfg)
    role_probs = size_cfg.get("markov_role_probs")
    effective = components
    if role_probs:
        effective = _apply_role_probs_to_hybrid_components(components, role_probs)
    comp = effective[int(_weighted_choice_index(rng, effective))]

    role = _hybrid_component_role(comp)
    if role == "plateau":
        v = int(comp.get("value", comp.get("constant", 592)))
        floor, regime = v, "plateau"
    elif role == "normal":
        x = float(np_rng.normal(comp["mean"], comp["std"]))
        floor, regime = int(max(min_sz, min(x, max_sz))), "normal"
    elif comp.get("kind") in ("constant", "fixed", "plateau"):
        v = int(comp.get("value", comp.get("constant", 364)))
        floor, regime = v, "tail"
    else:
        mean_log, std_log = _component_log_params(comp)
        x = int(max(min_sz, min(float(np.exp(np_rng.normal(mean_log, std_log))), max_sz)))
        floor, regime = x, "tail"

    if trace is not None and window_idx is not None:
        trace.record_floor(
            window_idx, floor, regime, comp, role_probs,
            [(c.get("kind"), round(float(c.get("weight", 0)), 4)) for c in effective],
        )
    return floor, regime


def _shuffle_intra_window_packets(np_rng, sizes, floor):
    """Reordena pacotes 1..n-1 preservando Min e Std; atenua max_run Cov monotona."""
    if len(sizes) <= 2:
        return list(sizes)
    out = list(sizes)
    rest = out[1:]
    np_rng.shuffle(rest)
    out = [int(floor)] + [int(s) for s in rest]
    out[0] = int(floor)
    return out


def _shift_sizes_to_min(sizes, target_min, min_sz, max_sz):
    """
    Desloca a sequencia para min(sizes)==target_min preservando Std (shift uniforme).

    Permite emissao Markov abaixo do piso amostrado durante a geracao; o Min AGG
    final coincide com sample_window_floor().
    """
    if not sizes:
        return [int(target_min)]
    sizes = [int(s) for s in sizes]
    lo = min(sizes)
    shift = int(target_min) - lo
    out = [max(int(target_min), min(int(max_sz), s + shift)) for s in sizes]
    hi = max(out)
    if hi > int(max_sz) and hi > int(target_min):
        span = hi - int(target_min)
        if span > 0:
            scale = (int(max_sz) - int(target_min)) / span
            out = [
                int(target_min) if s <= int(target_min) else int(target_min + (s - int(target_min)) * scale)
                for s in out
            ]
    out[0] = int(target_min)
    mi = min(out)
    if mi < int(target_min):
        bump = int(target_min) - mi
        out = [max(int(target_min), min(int(max_sz), s + bump)) for s in out]
    j = out.index(min(out))
    if j != 0:
        out[0], out[j] = out[j], out[0]
    out[0] = int(target_min)
    return [max(int(min_sz), min(int(max_sz), s)) for s in out]


def _clamp_sizes(sizes, min_sz, max_sz, floor):
    out = [max(min_sz, min(int(s), max_sz)) for s in sizes]
    out[0] = max(min_sz, min(int(floor), max_sz))
    return out


def _agg_append_two_stream_sizes(size, srcs, dsts, incoming, outgoing):
    """Replica Feature_extraction._append_two_stream_sizes (flood unidirecional)."""
    src_key, dst_key = "src", "dst"
    if src_key in dsts:
        outgoing.append(size)
    else:
        dsts[src_key] = 1
        outgoing.append(size)
    if dst_key in srcs:
        incoming.append(size)
    else:
        srcs[dst_key] = 1
        incoming.append(size)


def _agg_covariance_max_run(sizes):
    """Covariance AGG: max da serie corrida dynamic_two_streams (COVARIANCE_AGG_MODE=max_run)."""
    inc, out, srcs, dsts = [], [], {}, {}
    max_cov = 0.0
    for size in sizes:
        _agg_append_two_stream_sizes(float(size), srcs, dsts, inc, out)
        n_pairs = min(len(inc), len(out))
        if n_pairs <= 0:
            continue
        inco_ave = sum(inc) / len(inc)
        outgo_ave = sum(out) / len(out)
        inco_var = float(np.var(inc)) if len(inc) > 0 else 0.0
        outgo_var = float(np.var(out)) if len(out) > 0 else 0.0
        if n_pairs > 0 and inco_var > 0 and outgo_var > 0:
            covariance = sum(
                (a - inco_ave) * (b - outgo_ave)
                for a, b in zip(inc[:n_pairs], out[:n_pairs])
            ) / n_pairs
        else:
            covariance = 0.0
        max_cov = max(max_cov, float(covariance))
    return max_cov


def _agg_variance_std_over_min_sq(sizes):
    """Variance AGG: (Std/Min)^2 se Std>1 (VARIANCE_AGG_MODE=std_over_min_sq)."""
    arr = np.asarray(sizes, dtype=float)
    if len(arr) < 2:
        return 0.0
    std_v = float(np.std(arr, ddof=1))
    if std_v <= 1.0:
        return 0.0
    min_v = float(np.min(arr))
    return float((std_v / max(min_v, 1e-9)) ** 2)


def _agg_variance_cic_min_std(sizes, wh=None):
    """Variance AGG: Ridge(Min,Std) calibrado ao CSV CIC (treino LGBM)."""
    arr = np.asarray(sizes, dtype=float)
    if len(arr) < 2:
        return 0.0
    std_v = float(np.std(arr, ddof=1))
    if std_v <= 1.0:
        return 0.0
    min_v = float(np.min(arr))
    wh = wh or {}
    vc = wh.get("variance_cic") or {}
    if all(k in vc for k in ("intercept", "min_coef", "std_coef")):
        ic, mc, sc = float(vc["intercept"]), float(vc["min_coef"]), float(vc["std_coef"])
    else:
        ic, mc, sc = _variance_cic_preset(_variance_agg_mode(wh))
    return max(0.0, ic + mc * min_v + sc * std_v)


def _dynamic_features_engine():
    """Import lazy de Dynamic_features (pcap2csv) para var_ratio/cov corrida."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    pcap2csv_dir = os.path.abspath(os.path.join(here, "..", "pcap2csv"))
    if pcap2csv_dir not in sys.path:
        sys.path.insert(0, pcap2csv_dir)
    from Dynamic_features import Dynamic_features  # noqa: WPS433
    return Dynamic_features()


def _agg_running_dynamic_series(sizes):
    """Replica Feature_extraction._running_dynamic_series (cov + var_ratio)."""
    dy = _dynamic_features_engine()
    inc_r, out_r = [], []
    srcs, dsts = {}, {}
    covs, vars_ = [], []
    for size in sizes:
        _agg_append_two_stream_sizes(float(size), srcs, dsts, inc_r, out_r)
        _, _, _, cov, var_ratio, _ = dy.dynamic_two_streams(inc_r, out_r)
        covs.append(float(cov))
        vars_.append(float(var_ratio) if isinstance(var_ratio, (int, float)) else 0.0)
    return covs, vars_


def _agg_variance_mean_run(sizes):
    """Variance AGG: media var_ratio corrido (VARIANCE_AGG_MODE=mean_run, CICIoT2023)."""
    _, vars_ = _agg_running_dynamic_series(sizes)
    if not vars_:
        return 0.0
    return float(np.mean(vars_))


def _variance_cic_preset(mode: str):
    presets = {
        "greeth": (1.0187810555158603, -0.00170413, 0.00040736),
        "greip": (1.025512, -0.00175167, 0.00034770),
        "udpplain": (1.020003, -0.00182406, 0.00048342),
        "benign": (0.833107, -0.00020219, 0.00013237),
    }
    mode_l = str(mode).lower()
    for key, coeffs in presets.items():
        if key in mode_l:
            return coeffs
    return presets["greeth"]


def _variance_agg_mode(wh=None):
    wh = wh or {}
    reg = _cov_regulation_cfg(wh)
    mode = str(
        wh.get("variance_agg_mode")
        or reg.get("variance_agg_mode")
        or os.environ.get("VARIANCE_AGG_MODE", "std_over_min_sq")
    ).strip().lower()
    return mode


def _estimate_window_agg_features(sizes, wh=None):
    """Estima Min/Covariance/Variance como o extrator AGG (greeth calibrado)."""
    arr = np.asarray(sizes, dtype=float)
    wh = wh or {}
    var_mode = _variance_agg_mode(wh)
    if var_mode == "std_over_min_sq":
        variance = _agg_variance_std_over_min_sq(arr.tolist())
    elif var_mode in ("cic_greeth_min_std", "cic_greeth", "lgbm_greeth"):
        variance = _agg_variance_cic_min_std(arr.tolist(), wh)
    elif var_mode.startswith("cic_") and var_mode.endswith("_min_std"):
        variance = _agg_variance_cic_min_std(arr.tolist(), wh)
    elif var_mode in ("mean_run", "mean"):
        variance = _agg_variance_mean_run(arr.tolist())
    elif var_mode in ("max_run", "max"):
        _, vars_ = _agg_running_dynamic_series(arr.tolist())
        variance = float(max(vars_)) if vars_ else 0.0
    else:
        variance = _agg_variance_mean_run(arr.tolist())
    return {
        "Min": float(np.min(arr)) if len(arr) else 0.0,
        "Covariance": _agg_covariance_max_run(arr.tolist()),
        "Variance": variance,
        "Std": float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0,
    }


def _cov_regulation_cfg(wh):
    reg = dict((wh or {}).get("cov_regulation") or {})
    if "enabled" not in reg:
        reg["enabled"] = True
    return reg


def _cov_regime_markov_params(floor, mg, wh):
    """Escala jitter/spike Markov intra-janela consoante o floor (Min preservado)."""
    reg = _cov_regulation_cfg(wh)
    spread = dict((wh or {}).get("intra_spread") or {})
    floor = int(floor)
    jitter = float(mg.get("jitter_std", wh.get("size_std", spread.get("jitter_std", 12.0))))
    spike_p = float(mg.get("spike_prob", wh.get("outlier_fraction", 0.03)))
    spike_d = float(mg.get("spike_delta", wh.get("std_outlier_delta", 320.0)))
    if spread.get("enabled", True):
        jitter = float(spread.get("jitter_std", jitter))
        spike_p = float(spread.get("spike_prob", spike_p))
        spike_d = float(spread.get("spike_delta", spike_d))
        return max(0.0, jitter), max(0.0, spike_p), max(0.0, spike_d)
    plateau_min = int(reg.get("plateau_floor_min", 590))
    low_max = int(reg.get("low_floor_max", 560))
    if floor >= plateau_min:
        jitter = float(reg.get("plateau_jitter_std", max(jitter, 18.0)))
        spike_p *= float(reg.get("plateau_spike_scale", 0.35))
        spike_d = min(spike_d, float(reg.get("plateau_spike_delta_cap", 42.0)))
    elif floor < low_max:
        jitter *= float(reg.get("low_floor_jitter_scale", 0.32))
        spike_p *= float(reg.get("low_floor_spike_scale", 0.15))
        spike_d *= float(reg.get("low_floor_spike_delta_scale", 0.25))
    else:
        jitter *= float(reg.get("mid_floor_jitter_scale", 0.55))
        spike_p *= float(reg.get("mid_floor_spike_scale", 0.30))
        spike_d *= float(reg.get("mid_floor_spike_delta_scale", 0.45))
    return max(0.0, jitter), max(0.0, spike_p), max(0.0, spike_d)


def _boost_multimodal_size_spread(np_rng, sizes, floor, min_sz, max_sz, wh):
    """Alternancia entre niveis acima do Min — eleva Std/Variance sem shift artificial."""
    spread = dict((wh or {}).get("intra_spread") or {})
    tiers = spread.get("tiers") or [0, 40, 85, 145]
    tier_w = spread.get("tier_weights") or [0.28, 0.34, 0.26, 0.12]
    w = np.asarray(tier_w, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(tiers), dtype=float)
    w = w / w.sum()
    switch_p = float(spread.get("tier_switch_prob", 0.42))
    tier_jitter = float(spread.get("tier_jitter", 18.0))
    n = len(sizes)
    if n <= 1:
        return list(sizes)
    tier_idx = int(np_rng.choice(len(tiers), p=w))
    out = [int(floor)]
    for _ in range(1, n):
        if np_rng.random() < switch_p:
            tier_idx = int(np_rng.choice(len(tiers), p=w))
        delta = float(tiers[tier_idx]) + abs(float(np_rng.normal(0.0, tier_jitter)))
        out.append(int(max(floor, min(max_sz, floor + delta))))
    return _clamp_sizes(out, min_sz, max_sz, floor)


def _boost_size_spread(np_rng, sizes, floor, min_sz, max_sz, sigma, floor_frac=0.10):
    """Aumenta heterogeneidade intra-janela (indices >=1) preservando Min=floor."""
    out = [int(floor)]
    sigma = max(0.5, float(sigma))
    floor_frac = float(np.clip(floor_frac, 0.0, 0.45))
    for _ in range(1, len(sizes)):
        if np_rng.random() < floor_frac:
            out.append(int(floor))
        else:
            delta = abs(float(np_rng.normal(0.0, sigma)))
            out.append(int(max(floor, min(max_sz, floor + delta))))
    return _clamp_sizes(out, min_sz, max_sz, floor)


def _compress_size_spread(sizes, floor, min_sz, max_sz, factor):
    """Comprime desvios relativos ao Min real da janela (reduz Cov/Var extremos)."""
    factor = float(np.clip(factor, 0.05, 0.95))
    if not sizes:
        return [int(floor)]
    base = min(int(floor), min(int(s) for s in sizes))
    out = []
    for s in sizes:
        delta = (int(s) - base) * factor
        out.append(int(max(base, min(max_sz, base + delta))))
    out[0] = int(floor)
    return _clamp_sizes(out, min_sz, max_sz, floor)


def _expand_size_spread(sizes, floor, min_sz, max_sz, factor):
    """Expande desvios relativos ao Min (eleva Cov/Std quando abaixo do alvo CIC)."""
    factor = float(np.clip(factor, 1.0, 3.0))
    if not sizes:
        return [int(floor)]
    base = min(int(floor), min(int(s) for s in sizes))
    out = []
    for s in sizes:
        delta = (int(s) - base) * factor
        out.append(int(max(base, min(max_sz, base + delta))))
    out[0] = int(floor)
    return _clamp_sizes(out, min_sz, max_sz, floor)


def _regularize_window_covariance(sizes, floor, wh, rng, np_rng):
    """
    v6-M2: comprime greip_trap (Min baixo + Cov alta) e soft-cap Cov extrema.

    Preserva Std/Variance gerados pelo markov delta_above_floor.
    """
    reg = _cov_regulation_cfg(wh)
    if not reg.get("enabled", True):
        return list(sizes)

    min_sz = int((wh or {}).get("_size_min", 42))
    max_sz = int((wh or {}).get("_size_max", 2200))
    cov_hi = float(reg.get("cov_hi", 2500.0))
    greip_threshold = float(reg.get("greip_cov_threshold", cov_hi))
    cov_extreme = float(reg.get("cov_extreme", 5500.0))
    cov_soft_factor = float(reg.get("cov_soft_compress_factor", 0.72))
    max_iter = int(reg.get("max_iterations", 6))
    low_floor_max = int(reg.get("low_floor_max", 560))

    sizes = _clamp_sizes(list(sizes), min_sz, max_sz, floor)
    floor = int(floor)

    cov_target = float(reg.get("cov_target", 0.0))
    if cov_target > 0:
        cov_now = float(_estimate_window_agg_features(sizes, wh)["Covariance"])
        expand_max = float(reg.get("cov_expand_max", 2.5))
        best_sizes = list(sizes)
        best_diff = abs(cov_now - cov_target)

        if cov_now < cov_target:
            lo, hi = 1.0, expand_max
            for _ in range(12):
                mid = (lo + hi) / 2.0
                trial = _expand_size_spread(sizes, floor, min_sz, max_sz, mid)
                cov = float(_estimate_window_agg_features(trial, wh)["Covariance"])
                diff = abs(cov - cov_target)
                if diff < best_diff:
                    best_diff = diff
                    best_sizes = trial
                if cov < cov_target:
                    lo = mid
                else:
                    hi = mid
        else:
            lo, hi = 0.30, 1.0
            for _ in range(12):
                mid = (lo + hi) / 2.0
                trial = _compress_size_spread(sizes, floor, min_sz, max_sz, mid)
                cov = float(_estimate_window_agg_features(trial, wh)["Covariance"])
                diff = abs(cov - cov_target)
                if diff < best_diff:
                    best_diff = diff
                    best_sizes = trial
                if cov > cov_target:
                    lo = mid
                else:
                    hi = mid
        return _clamp_sizes(best_sizes, min_sz, max_sz, floor)

    if floor < low_floor_max:
        for _ in range(max_iter):
            feats = _estimate_window_agg_features(sizes, wh)
            cov = float(feats["Covariance"])
            if cov <= greip_threshold:
                break
            factor = float(reg.get("low_floor_compress_factor", 0.55))
            sizes = _compress_size_spread(sizes, floor, min_sz, max_sz, factor)

    for _ in range(max_iter):
        feats = _estimate_window_agg_features(sizes, wh)
        cov = float(feats["Covariance"])
        if cov <= cov_extreme:
            break
        sizes = _compress_size_spread(sizes, floor, min_sz, max_sz, cov_soft_factor)

    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _var_model_regulation_cfg(wh):
    reg = dict((wh or {}).get("var_model_regulation") or {})
    cov_reg = _cov_regulation_cfg(wh)
    if "enabled" not in reg:
        reg["enabled"] = bool(cov_reg.get("var_model_enabled", False))
    return reg


def _boost_variance_signed_spread(np_rng, sizes, floor, min_sz, max_sz, sigma, floor_frac=0.06):
    """
    Aumenta Std/Min (Variance) com jitter bidireccional acima do floor.

    Preserva Min=floor; alterna picos moderados para elevar (Std/Min)^2
    sem explosao unilateral de max_run Covariance.
    """
    out = [int(floor)]
    sigma = max(0.5, float(sigma))
    floor_frac = float(np.clip(floor_frac, 0.0, 0.35))
    for i in range(1, len(sizes)):
        if np_rng.random() < floor_frac:
            out.append(int(floor))
            continue
        sign = -1.0 if (i % 2 == 0 and np_rng.random() < 0.35) else 1.0
        delta = sign * abs(float(np_rng.normal(0.0, sigma)))
        out.append(int(max(floor, min(max_sz, floor + delta))))
    return _clamp_sizes(out, min_sz, max_sz, floor)


def _regularize_window_variance_model(sizes, floor, wh, np_rng):
    """
    Fase v6-K: alinha Variance ao envelope LightGBM/CIC greeth (~0.04 mediana).

    Corre apos cov_regulation; nao altera flow_duration. Comprime se Cov estourar.
    """
    reg = _var_model_regulation_cfg(wh)
    if not reg.get("enabled", True):
        return list(sizes)

    min_sz = int((wh or {}).get("_size_min", 42))
    max_sz = int((wh or {}).get("_size_max", 2200))
    var_lo = float(reg.get("var_lo", 0.018))
    var_hi = float(reg.get("var_hi", 0.12))
    var_target = float(reg.get("var_target", 0.040))
    cov_cap = float(reg.get("cov_cap", 2600.0))
    max_iter = int(reg.get("max_iterations", 10))
    floor = int(floor)
    plateau_min = int(reg.get("plateau_floor_min", 590))

    sizes = _clamp_sizes(list(sizes), min_sz, max_sz, floor)
    for _ in range(max_iter):
        feats = _estimate_window_agg_features(sizes, wh)
        var = float(feats["Variance"])
        cov = float(feats["Covariance"])
        if var_lo <= var <= var_hi and cov <= cov_cap:
            break
        if cov > cov_cap:
            factor = float(reg.get("cov_compress_factor", 0.72))
            sizes = _compress_size_spread(sizes, floor, min_sz, max_sz, factor)
        elif var < var_lo:
            if floor >= plateau_min:
                sigma = float(reg.get("plateau_signed_std", 52.0))
            elif floor < int(reg.get("low_floor_max", 560)):
                sigma = float(reg.get("low_floor_signed_std", 38.0))
            else:
                sigma = float(reg.get("mid_signed_std", 45.0))
            deficit = max(0.0, (var_target - var) / max(var_target, 1e-9))
            sigma *= 1.0 + 0.75 * min(1.5, deficit)
            sizes = _boost_variance_signed_spread(
                np_rng, sizes, floor, min_sz, max_sz, sigma,
                floor_frac=float(reg.get("floor_frac", 0.06)),
            )
        elif var > var_hi:
            factor = float(reg.get("compress_factor", 0.68))
            sizes = _compress_size_spread(sizes, floor, min_sz, max_sz, factor)
        else:
            break
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _target_tot_sum(floor, wh):
    target_base = float(wh.get("target_tot_sum_median", 6200.0))
    ref_min = float(wh.get("target_tot_sum_ref_min", 576.0))
    return target_base * (float(floor) / max(1.0, ref_min))


def _sizes_homogeneous(floor, agg_n):
    return [int(floor)] * max(1, int(agg_n))


def _sizes_micro_jitter(np_rng, floor, agg_n, min_sz, max_sz, sigma):
    n = max(1, int(agg_n))
    sigma = max(0.0, float(sigma))
    sizes = [int(floor)]
    for _ in range(1, n):
        delta = float(np_rng.normal(0.0, sigma)) if sigma > 0 else 0.0
        sizes.append(int(max(floor, min(max_sz, floor + delta))))
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _sizes_one_min_rest_plateau(np_rng, floor, agg_n, plateau_size, min_sz, max_sz, jitter_sigma=0.0):
    """
    Um pacote no Min (floor) e restantes num platô alto (~592 CIC).

    Reproduz janelas greeth com Min baixo mas AVG/Std alinhados ao dataset real.
    """
    n = max(1, int(agg_n))
    floor = int(floor)
    plateau_size = int(plateau_size)
    sigma = max(0.0, float(jitter_sigma))
    sizes = [floor]
    for _ in range(1, n):
        val = plateau_size
        if sigma > 0:
            val = int(round(plateau_size + float(np_rng.normal(0.0, sigma))))
        sizes.append(int(max(floor, min(max_sz, val))))
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _low_floor_plateau_jitter(floor, wh):
    """Jitter no platô intra-janela consoante o Min amostrado."""
    floor = int(floor)
    if floor < 500:
        return float(wh.get("low_floor_plateau_jitter_low", 18.0))
    if floor < 540:
        return float(wh.get("low_floor_plateau_jitter_mid", 12.0))
    return float(wh.get("low_floor_plateau_jitter_hi", 5.0))


def _ordered_spike_plateau_for_floor(floor, agg_n, wh):
    """
    Spike + platô ordenados para max_run Covariance (ordem importa no extrator).

    Banda mid (540-575): [floor, spike=628, plateau=592 x (n-2)] -> Cov~1066, AVG~590.
    Banda baixa (<540): spike=640, plateau derivado para AVG~580.
    """
    floor = int(floor)
    n = max(2, int(agg_n))
    mid_lo = int(wh.get("low_floor_mid_min", 540))
    if floor >= mid_lo:
        spike = int(wh.get("low_floor_spike_mid", 628))
        plateau = int(wh.get("low_floor_plateau_mid", 592))
        return spike, plateau
    spike = int(wh.get("low_floor_spike_low", 640))
    target_avg = float(wh.get("low_floor_target_avg_low", 580.0))
    plateau = (target_avg * n - floor - spike) / max(1, n - 2)
    plateau = int(round(max(floor + 1, plateau)))
    return spike, plateau


def _sizes_one_min_ordered_spike(
    np_rng, floor, agg_n, spike, plateau, min_sz, max_sz, jitter_sigma=0.0,
):
    """Min=floor, 2.o pacote spike alto, restantes platô — ordem fixa (Covariance max_run)."""
    n = max(2, int(agg_n))
    floor = int(floor)
    spike = int(spike)
    plateau = int(plateau)
    sigma = max(0.0, float(jitter_sigma))
    sizes = [floor, spike]
    for _ in range(n - 2):
        val = plateau
        if sigma > 0:
            val = int(round(plateau + float(np_rng.normal(0.0, sigma))))
        sizes.append(int(max(floor, min(max_sz, val))))
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _sizes_plateau_spike(floor, agg_n, target_tot_sum, max_sz, min_sz):
    """
    floor no 1.o pacote + corpo constante S nos restantes: Min=floor, Tot sum alto, Std moderada.
    mean(Cumul) = floor + S*(n-1)/2  (n pacotes).
    """
    n = max(1, int(agg_n))
    if n == 1:
        return [int(floor)]
    coef = (n - 1) / 2.0
    s_body = (float(target_tot_sum) - float(floor)) / max(1e-9, coef)
    s_body = int(max(floor, min(max_sz, s_body)))
    return _clamp_sizes([int(floor)] + [s_body] * (n - 1), min_sz, max_sz, floor)


def _sizes_linear_ramp(floor, agg_n, target_tot_sum, max_sz, min_sz):
    n = max(1, int(agg_n))
    d = _linear_ramp_step(floor, n, target_tot_sum, max_sz)
    sizes = [int(floor)]
    for i in range(1, n):
        sizes.append(int(min(max_sz, max(floor, floor + i * d))))
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _apply_std_outlier(np_rng, sizes, floor, max_sz, min_sz, wh):
    n = len(sizes)
    if n <= 1:
        return sizes
    outlier_d = float(wh.get("std_outlier_delta", 280.0))
    j = int(np_rng.integers(1, n))
    sizes = list(sizes)
    sizes[j] = int(min(max_sz, sizes[j] + outlier_d))
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _markov_states_from_size_cfg(size_cfg, mg=None):
    """Estados Markov a partir de size.components (GMM ou hibrido Min)."""
    mg = mg or {}
    raw = size_cfg.get("components") or size_cfg.get("mixture_components") or []
    if _is_hybrid_size_components(raw):
        hybrid = _normalize_hybrid_components(size_cfg)
    else:
        gmm = _normalize_size_components(size_cfg)
        if gmm:
            return gmm
        hybrid = _normalize_hybrid_components(size_cfg)

    role_probs = _resolve_markov_role_probs(mg, hybrid)
    states = []
    for comp in hybrid:
        role = _hybrid_component_role(comp)
        entry = {
            "weight": role_probs.get(role, comp["weight"]),
            "role": role,
            "mean_log": 0.0,
            "std_log": 0.05,
        }
        if comp["kind"] in ("constant", "fixed", "plateau"):
            v = float(comp["value"])
            entry["mean_log"] = float(np.log(max(v, 1.0)))
            entry["std_log"] = float(mg.get("plateau_emit_std_log", 0.045))
        elif comp["kind"] in ("normal", "gaussian"):
            m = float(comp["mean"])
            s = max(1e-6, float(comp["std"]))
            entry["mean_log"] = float(np.log(max(m, 1.0)))
            spread_mult = float(mg.get("normal_emit_std_mult", 1.35))
            entry["std_log"] = max(1e-6, (s / max(m, 1.0)) * spread_mult)
        else:
            mean_log, std_log = comp["mean_log"], comp["std_log"]
            entry["mean_log"] = mean_log
            tail_mult = float(mg.get("tail_emit_std_mult", 1.25))
            entry["std_log"] = max(1e-6, float(std_log) * tail_mult)
        states.append(entry)

    if len(states) == 3 and len({s["role"] for s in states}) == 3:
        order = {r: i for i, r in enumerate(MARKOV_ROLE_ORDER)}
        states.sort(key=lambda s: order.get(s["role"], 99))
    return states


def _markov_initial_state(rng, states, mg):
    """Estado inicial da cadeia Markov (plateau_prob / normal_prob / tail_prob)."""
    role_probs = _resolve_markov_role_probs(mg or {})
    weights = [
        float(role_probs.get(s.get("role", "normal"), s.get("weight", 1.0)))
        for s in states
    ]
    total = sum(weights)
    if total <= 0:
        return int(_weighted_choice_index(rng, states))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return int(i)
    return int(len(states) - 1)


def _markov_transition_matrix(n_states, stay_prob, mg=None):
    """
    Matriz de transicao Markov (linhas somam 1).

    Prioridade: mg.transition explicita > stay_probs por estado > stay_prob
    uniforme > padrao assimétrico 3x3 (persistencia ~0.90).
    """
    mg = mg or {}
    raw = mg.get("transition")
    if raw is not None:
        p = np.asarray(raw, dtype=float)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError("markov_gmm.transition deve ser matriz quadrada")
        rs = p.sum(axis=1, keepdims=True)
        rs[rs <= 0] = 1.0
        p = p / rs
        tail_stay = mg.get("tail_stay_prob")
        if tail_stay is not None and p.shape[0] >= 3:
            ts = float(min(max(float(tail_stay), 0.0), 0.95))
            row = p[2].copy()
            off_sum = float(row[0]) + float(row[1])
            budget = 1.0 - ts
            if off_sum <= 0:
                p[2] = np.array([0.20, 0.75, ts], dtype=float)
            else:
                p[2, 0] = budget * float(row[0]) / off_sum
                p[2, 1] = budget * float(row[1]) / off_sum
                p[2, 2] = ts
        return p

    n = max(1, int(n_states))
    if n == 1:
        return np.array([[1.0]])

    stay_probs = mg.get("stay_probs")
    if stay_probs is not None and len(stay_probs) == n:
        p = np.zeros((n, n), dtype=float)
        for i in range(n):
            si = float(min(max(stay_probs[i], 0.01), 0.999))
            off = (1.0 - si) / max(1, n - 1)
            p[i, :] = off
            p[i, i] = si
        return p

    stay = float(min(max(stay_prob, 0.01), 0.999))
    if n == 3 and mg.get("asymmetric", True):
        # Cadeia heterogenea: cauda transitoria (eleva Std/Cov sem inflar Min).
        return np.array(DEFAULT_MARKOV_TRANSITION_3, dtype=float)

    off = (1.0 - stay) / (n - 1)
    p = np.full((n, n), off, dtype=float)
    np.fill_diagonal(p, stay)
    return p


def _markov_next_state(rng, state, trans):
    """Amostra proximo estado a partir da linha de transicao."""
    row = trans[int(state)]
    r = rng.random()
    acc = 0.0
    for i, prob in enumerate(row):
        acc += float(prob)
        if r <= acc:
            return int(i)
    return int(len(row) - 1)


def _expand_intra_window_ar1(np_rng, floor, agg_n, ds_cfg, size_cfg):
    """Fallback AR(1) crescente (legado)."""
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 1500))
    wh = (ds_cfg or {}).get("window_hetero", {})
    size_std = float(wh.get("size_std", ds_cfg.get("size_std", 36.0) if ds_cfg else 36.0))
    rho = float(wh.get("size_correlation", ds_cfg.get("size_correlation", 0.92) if ds_cfg else 0.92))
    drift = float(wh.get("size_drift", size_std * 0.12))
    z = 0.0
    sizes = [int(floor)]
    for _ in range(1, agg_n):
        innov = float(np_rng.normal(0.0, 1.0)) * np.sqrt(max(1e-9, 1.0 - rho ** 2))
        z = rho * z + innov
        delta = max(0, int(drift + abs(z) * size_std * 0.22))
        nxt = int(max(floor, min(max_sz, sizes[-1] + delta)))
        sizes.append(nxt)
    return [max(min_sz, min(s, max_sz)) for s in sizes]


def _expand_intra_window_markov_gmm(rng, np_rng, floor, agg_n, ds_cfg, size_cfg):
    """
    Sequencia intra-janela AGG=20 via GMM + cadeia de Markov + spikes.

    v6-L: emite tamanhos absolutos dos estados (sem clamp max(floor, val)),
    depois shift uniforme para min==floor — preserva Std e permite multimodal.
    """
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 1500))
    wh = (ds_cfg or {}).get("window_hetero", {})
    mg = wh.get("markov_gmm") or {}
    states = _markov_states_from_size_cfg(size_cfg, mg)
    if not states:
        return _sizes_homogeneous(floor, agg_n)

    stay_prob = float(mg.get("stay_prob", wh.get("markov_stay_prob", 0.90)))
    ar1_rho = float(mg.get(
        "ar1_rho",
        size_cfg.get("ar1_rho", ds_cfg.get("size_correlation", 0.85) if ds_cfg else 0.85),
    ))
    ar1_rho = min(max(ar1_rho, 0.0), 0.999)
    tail_scale = float(mg.get("tail_scale", 1.0))
    jitter, spike_p, spike_delta = _cov_regime_markov_params(floor, mg, wh)

    trans = _markov_transition_matrix(len(states), stay_prob, mg)
    state = _markov_initial_state(rng, states, mg)
    z = 0.0
    sizes = []
    n = max(1, int(agg_n))
    emit_mode = str(mg.get("emit_mode", "delta_above_floor")).lower()
    spread = dict((wh or {}).get("intra_spread") or {})
    delta_tiers = spread.get("role_deltas") or {
        "plateau": [0, 18],
        "normal": [38, 95],
        "tail": [75, 165],
    }

    for i in range(n):
        if i > 0:
            state = _markov_next_state(rng, state, trans)

        role = states[state].get("role", "normal")
        if emit_mode in ("delta_above_floor", "delta", "tiers"):
            lo, hi = delta_tiers.get(role, delta_tiers.get("normal", [30, 80]))
            lo, hi = float(lo), float(hi)
            if floor >= 540:
                boost = float(spread.get("mid_floor_delta_scale", 1.22))
                lo *= boost
                hi *= boost
            elif floor < int(_cov_regulation_cfg(wh).get("low_floor_max", 560)):
                scale = float(spread.get("low_floor_delta_scale", 0.82))
                lo *= scale
                hi *= scale
            delta = float(np_rng.uniform(lo, hi))
            if jitter > 0:
                delta += abs(float(np_rng.normal(0.0, jitter * 0.35)))
            val = float(floor) + delta
        else:
            innov = float(np_rng.normal(0.0, 1.0)) * np.sqrt(max(1e-9, 1.0 - ar1_rho ** 2))
            z = ar1_rho * z + innov
            mean_log = float(states[state]["mean_log"])
            std_log = float(states[state]["std_log"])
            if role == "tail" and tail_scale != 1.0:
                mean_log += float(np.log(max(tail_scale, 0.05)))
                std_log = max(1e-6, std_log * max(0.25, min(3.0, tail_scale)))
            val = float(np.exp(mean_log + std_log * z))
            if jitter > 0:
                val += float(np_rng.normal(0.0, jitter))

        if rng.random() < spike_p:
            val += spike_delta
        if emit_mode in ("delta_above_floor", "delta", "tiers"):
            val = max(float(floor), val)
        else:
            val = max(float(min_sz), val)
        sizes.append(int(min(max_sz, val)))

    if emit_mode not in ("delta_above_floor", "delta", "tiers"):
        sizes = _shift_sizes_to_min(sizes, floor, min_sz, max_sz)
    else:
        sizes[0] = int(floor)
    if mg.get("shuffle_packets", spread.get("shuffle_packets", True)):
        sizes = _shuffle_intra_window_packets(np_rng, sizes, floor)
    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def _apply_burst_model(rng, np_rng, sizes, floor, min_sz, max_sz, wh):
    """
    Injeta 1+ pacotes grandes numa janela homogenea (probabilistico).

    Preserva Min=floor; aumenta Std/Tot sum/Covariance apenas em ~probability
    das janelas — alinhado ao CICIoT2023 (outliers localizados).
    """
    bm = wh.get("burst_model") or {}
    if not bm.get("enabled", False):
        return sizes

    reg = _cov_regulation_cfg(wh)
    floor = int(floor)
    prob = float(bm.get("probability", 0.08))
    if floor >= int(reg.get("plateau_floor_min", 590)):
        prob *= float(reg.get("plateau_burst_scale", 0.40))
    elif floor < int(reg.get("low_floor_max", 560)):
        prob *= float(reg.get("low_floor_burst_scale", 0.20))
    if rng.random() >= prob:
        return sizes

    sizes = list(sizes)
    n = len(sizes)
    if n <= 0:
        return sizes

    k = min(n, max(1, int(bm.get("packets_per_burst", 1))))
    mean = float(bm.get("burst_mean", 1050))
    std = max(1e-6, float(bm.get("burst_std", 120)))
    if floor < int(reg.get("low_floor_max", 560)):
        mean = min(mean, float(reg.get("low_floor_burst_mean_cap", 620)))
    elif floor >= int(reg.get("plateau_floor_min", 590)):
        mean = min(mean, float(floor) + float(reg.get("plateau_burst_mean_delta_cap", 48)))

    if k >= n:
        indices = np.arange(n)
    else:
        indices = np_rng.choice(n, size=k, replace=False)

    for idx in indices:
        val = float(np_rng.normal(mean, std))
        sizes[int(idx)] = int(max(floor, min(max_sz, val)))

    return _clamp_sizes(sizes, min_sz, max_sz, floor)


def expand_intra_window_sizes(rng, np_rng, floor, agg_n, ds_cfg, size_cfg, trace=None, window_idx=None):
    """
    Perfis mistos intra-janela (estilo udpplain + rampa GRE):

      low_std (default 68%): homogeneo ou micro-jitter -> Std baixa, Tot sum ~5.5*floor
      ramp (default 27%): platô+spike ou rampa linear -> Tot sum ~6200
      outlier (default 5%): homogeneo + 1 outlier -> cauda de Std

    Min=floor preservado em todos os perfis.
    burst_model (window_hetero.burst_model) aplica-se no final a qualquer perfil.
    """
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 1500))
    wh = (ds_cfg or {}).get("window_hetero", {})
    profile_mode = str(wh.get("tot_sum_profile", "markov_gmm")).lower()
    plateau_homo_min = int(wh.get("plateau_homogeneous_floor_min", 576))
    low_profile = str(wh.get("low_floor_profile", "")).lower()

    def _finish_light(sizes, apply_regulation=True):
        wh_eff = dict(wh)
        wh_eff["_size_min"] = min_sz
        wh_eff["_size_max"] = max_sz
        out = _clamp_sizes(sizes, min_sz, max_sz, floor)
        if apply_regulation:
            out = _apply_burst_model(rng, np_rng, out, floor, min_sz, max_sz, wh_eff)
            out = _regularize_window_covariance(out, floor, wh_eff, rng, np_rng)
        if trace is not None and window_idx is not None:
            trace.record_sizes(window_idx, floor, out)
        return out

    if floor >= plateau_homo_min and wh.get("plateau_homogeneous", True):
        sigma = float(wh.get("plateau_micro_jitter_sigma", 0.0))
        plat_size = wh.get("plateau_homogeneous_size")
        homo_floor = int(plat_size) if plat_size is not None else int(floor)
        if sigma <= 0:
            return _finish_light(_sizes_homogeneous(homo_floor, agg_n), apply_regulation=False)
        return _finish_light(
            _sizes_micro_jitter(np_rng, homo_floor, agg_n, min_sz, max_sz, sigma),
            apply_regulation=False,
        )

    if low_profile in (
        "one_min_ordered_spike", "ordered_spike", "cic_cov",
        "one_min_rest_plateau", "min_plateau", "cic_low",
    ):
        jitter = _low_floor_plateau_jitter(floor, wh)
        if low_profile in ("one_min_ordered_spike", "ordered_spike", "cic_cov"):
            spike, plateau = _ordered_spike_plateau_for_floor(floor, agg_n, wh)
            sizes = _sizes_one_min_ordered_spike(
                np_rng, floor, agg_n, spike, plateau, min_sz, max_sz, jitter,
            )
        else:
            plateau_size = int(wh.get("low_floor_plateau_size", 592))
            sizes = _sizes_one_min_rest_plateau(
                np_rng, floor, agg_n, plateau_size, min_sz, max_sz, jitter,
            )
        return _finish_light(
            sizes,
            apply_regulation=bool(wh.get("low_floor_apply_cov_regulation", False)),
        )

    def _finish(sizes):
        wh_eff = dict(wh)
        wh_eff["_size_min"] = min_sz
        wh_eff["_size_max"] = max_sz
        out = _apply_burst_model(
            rng, np_rng, _clamp_sizes(sizes, min_sz, max_sz, floor),
            floor, min_sz, max_sz, wh_eff,
        )
        out = _regularize_window_covariance(out, floor, wh_eff, rng, np_rng)
        if _var_model_regulation_cfg(wh_eff).get("enabled", False):
            out = _regularize_window_variance_model(out, floor, wh_eff, np_rng)
        if trace is not None and window_idx is not None:
            trace.record_sizes(window_idx, floor, out)
        return out

    if profile_mode in ("homogeneous", "flat"):
        return _finish(_sizes_homogeneous(floor, agg_n))

    if profile_mode in ("markov_gmm", "markov", "gmm_markov"):
        return _finish(_expand_intra_window_markov_gmm(
            rng, np_rng, floor, agg_n, ds_cfg, size_cfg,
        ))

    if profile_mode == "ar1":
        return _finish(_expand_intra_window_ar1(np_rng, floor, agg_n, ds_cfg, size_cfg))

    if profile_mode == "linear_ramp":
        target = _target_tot_sum(floor, wh)
        sizes = _sizes_linear_ramp(floor, agg_n, target, max_sz, min_sz)
        jitter = float(wh.get("size_jitter", 0.0))
        if jitter > 0:
            sizes = _sizes_micro_jitter(np_rng, floor, agg_n, min_sz, max_sz, jitter)
            sizes[0] = int(floor)
        return _finish(sizes)

    # mixed (legado)
    f_low = float(wh.get("low_std_fraction", 0.68))
    f_ramp = float(wh.get("ramp_fraction", 0.27))
    f_out = float(wh.get("outlier_fraction", max(0.0, 1.0 - f_low - f_ramp)))
    total = f_low + f_ramp + f_out
    if total <= 0:
        f_low, f_ramp, f_out = 0.68, 0.27, 0.05
        total = 1.0
    f_low, f_ramp, f_out = f_low / total, f_ramp / total, f_out / total

    u = rng.random()
    target = _target_tot_sum(floor, wh)
    use_spike = bool(wh.get("ramp_use_plateau_spike", True))

    if u < f_low:
        if rng.random() < float(wh.get("micro_jitter_within_low", 0.40)):
            sigma = float(wh.get("micro_jitter_sigma", 3.5))
            return _finish(_sizes_micro_jitter(np_rng, floor, agg_n, min_sz, max_sz, sigma))
        return _finish(_sizes_homogeneous(floor, agg_n))

    if u < f_low + f_ramp:
        if use_spike:
            return _finish(_sizes_plateau_spike(floor, agg_n, target, max_sz, min_sz))
        return _finish(_sizes_linear_ramp(floor, agg_n, target, max_sz, min_sz))

    sizes = _sizes_homogeneous(floor, agg_n)
    return _finish(_apply_std_outlier(np_rng, sizes, floor, max_sz, min_sz, wh))


# --------------------------------------------------------------------------- #
# HMM operacional (6-8 estados latentes -> vetor conjunto da janela)          #
# Substitui calibracao separada de flow_duration GMM + Markov tamanhos +      #
# jitter/ar1 quando operational_modes.enabled=true.                           #
# --------------------------------------------------------------------------- #
LATENT_STATE_NAMES = (
    "plateau", "normal", "transition", "tail", "burst", "outlier",
)
OPERATIONAL_EMISSION_KEYS = (
    "flow_duration",
    "packet_count",
    "packet_count_jitter",
    "payload_regime",
    "payload_model",
    "payload_profile",
    "payload_floor",
    "payload_normal_mean",
    "payload_normal_std",
    "payload_variance",
    "ar1_rho",
    "jitter",
    "burst_probability",
    "plateau_probability",
    "normal_probability",
    "tail_probability",
    "inter_arrival_scale",
    "long_flow_fraction",
    "emission_mode",
    "window_template",
    "window_sizes",
    "template_jitter_std",
    "iat_profile",
    "iat_jitter_frac",
    "joint_emission",
)


def _coerce_emission_value(key, val):
    if val is None:
        return None
    if key in ("payload_model", "payload_profile", "payload_regime", "emission_mode", "window_template", "iat_profile"):
        return str(val)
    if key == "flow_duration" and isinstance(val, dict):
        return dict(val)
    if key == "packet_count_range" and isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if key == "window_sizes" and isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if key == "joint_emission":
        return bool(val)
    return float(val)


def _role_from_floor_value(floor):
    v = int(floor)
    if v <= 420:
        return "tail"
    if v >= 550:
        return "plateau"
    return "normal"


# Perfis completos de janela (20 pacotes) — Min/Std/Tot sum/Covariance emergem da mesma sequencia.
GREETH_WINDOW_TEMPLATES = {
    "plateau": [592, 592, 592, 592, 592, 592, 592, 592, 592, 592],
    "normal": [510, 520, 515, 530, 521, 527, 519, 514, 522, 518],
    "transition": [520, 540, 510, 480, 450, 520, 560, 510, 490, 530],
    "tail": [240, 370, 510, 592, 592, 470, 360, 592, 592, 310],
    "burst": [592, 592, 880, 592, 592, 950, 592, 592, 592, 920],
    "outlier": [280, 320, 890, 360, 1050, 310, 780, 290, 920, 340],
}

_UDP_TEMPLATE_SCALE = 548.0 / 592.0
UDPPLAIN_WINDOW_TEMPLATES = {
    name: [
        int(max(400, min(700, round(v * _UDP_TEMPLATE_SCALE))))
        for v in tmpl
    ]
    for name, tmpl in GREETH_WINDOW_TEMPLATES.items()
}


def _window_templates_for_variant(variant):
    if str(variant).lower() == "udpplain":
        return UDPPLAIN_WINDOW_TEMPLATES
    return GREETH_WINDOW_TEMPLATES

# Perfil temporal intra-janela por defeito (pesos normalizados -> sum(IAT)=flow_duration).
GREETH_IAT_PROFILE_DEFAULTS = {
    "plateau": "instant",
    "normal": "uniform",
    "transition": "spread",
    "tail": "back_loaded",
    "burst": "micro_burst",
    "outlier": "spread",
}


def _resolve_iat_profile(emission):
    if emission.get("iat_profile"):
        return str(emission["iat_profile"]).lower()
    wt = str(emission.get("window_template", "normal")).lower()
    return GREETH_IAT_PROFILE_DEFAULTS.get(wt, "uniform")


def _iat_weights_from_profile(profile, n_iats, np_rng):
    """Vetor de pesos positivos (ou zeros) para repartir flow_duration em n_iats."""
    if n_iats <= 0:
        return np.array([], dtype=float)
    p = str(profile).lower()
    if p in ("instant", "plateau", "zero"):
        return np.zeros(n_iats, dtype=float)
    if p == "uniform":
        return np.ones(n_iats, dtype=float)
    if p == "spread":
        return np.linspace(0.4, 1.6, n_iats, dtype=float)
    if p in ("back_loaded", "tail"):
        x = np.linspace(0.15, 1.0, n_iats, dtype=float)
        return x ** 2
    if p in ("front_loaded", "burst"):
        x = np.linspace(1.0, 0.15, n_iats, dtype=float)
        return x ** 2
    if p == "micro_burst":
        w = np.ones(n_iats, dtype=float)
        if n_iats > 1:
            w[-1] = 0.05
        return w
    return np.ones(n_iats, dtype=float)


def _distribute_duration_to_iats(duration, weights, min_iat=1e-9):
    """IATs cuja soma = duration (span da janela AGG)."""
    n = len(weights)
    if n == 0:
        return []
    if duration <= min_iat:
        return [0.0] * n
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    wsum = float(w.sum())
    if wsum <= 0.0:
        return [duration / n] * n
    return [max(0.0, float(duration * wi / wsum)) for wi in w]


def _resample_size_template(template, n):
    """Interpola ou repete um template para n pacotes."""
    if n <= 0:
        return []
    tmpl = [int(x) for x in template]
    if len(tmpl) == n:
        return list(tmpl)
    if len(tmpl) == 1:
        return [tmpl[0]] * n
    idx = np.linspace(0, len(tmpl) - 1, n)
    return [int(tmpl[int(round(i))]) for i in idx]


def _emission_uses_window_vector(emission):
    """True quando o estado emite sequencia completa em vez de parametros Markov-GMM."""
    if not emission:
        return False
    if emission.get("window_sizes"):
        return True
    mode = str(emission.get("emission_mode", "")).lower()
    if mode == "vector":
        return True
    if emission.get("window_template"):
        return True
    return False


def _window_sizes_from_emission(emission, win_n, size_cfg, np_rng, variant="greeth"):
    """
    Gera tamanhos intra-janela a partir de template vetorial do estado HMM.
    Retorna (sizes, floor, regime) com floor=min(sizes) alinhado ao extrator CIC.
    """
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 1500))
    templates = _window_templates_for_variant(variant)
    raw = emission.get("window_sizes")
    if raw:
        tmpl = [int(x) for x in raw]
    else:
        name = str(emission.get("window_template", emission.get("payload_regime", "normal"))).lower()
        tmpl = list(templates.get(name, templates["normal"]))
    sizes = _resample_size_template(tmpl, win_n)
    jitter = float(emission.get("template_jitter_std", emission.get("jitter", 0.0)))
    if jitter > 0:
        sizes = [
            int(max(min_sz, min(max_sz, s + np_rng.normal(0, jitter))))
            for s in sizes
        ]
    else:
        sizes = [int(max(min_sz, min(s, max_sz))) for s in sizes]
    floor = int(min(sizes))
    regime = _role_from_floor_value(floor)
    return sizes, floor, regime


def _window_joint_emission(emission, win_n, size_cfg, flow_cfg, rng, np_rng, variant="greeth"):
    """
    Emissao conjunta por estado HMM: tamanhos + flow_duration + IATs semanticos.

    Min, Std, Tot sum e Covariance derivam de sizes[].
    flow_duration e IATs retornados alimentam o manifest (semantica CIC).
    O PCAP usa pcap_intra_iat_sec via resolve_pcap_intra_iat_sec() no schedule.
    """
    sizes, floor, regime = _window_sizes_from_emission(
        emission, win_n, size_cfg, np_rng, variant=variant,
    )
    n_iats = max(0, len(sizes) - 1)
    duration = _flow_duration_from_emission(emission, flow_cfg, rng, np_rng)
    profile = _resolve_iat_profile(emission)

    if profile in ("instant", "plateau", "zero") or duration <= 1e-9:
        duration = 0.0
        iats = [0.0] * n_iats
    else:
        weights = _iat_weights_from_profile(profile, n_iats, np_rng)
        jitter = float(emission.get("iat_jitter_frac", 0.08))
        if jitter > 0 and n_iats > 0:
            noise = 1.0 + np_rng.normal(0.0, jitter, size=n_iats)
            weights = np.maximum(weights * noise, 1e-12)
        iats = _distribute_duration_to_iats(duration, weights)
        scale = float(emission.get("inter_arrival_scale", 1.0))
        if scale != 1.0 and iats:
            scaled = [x * scale for x in iats]
            s = sum(scaled)
            iats = [x * duration / s for x in scaled] if s > 0 else iats

    return sizes, iats, floor, regime, duration


def compute_window_feature_vector(sizes, iats, rst_count=None, flow_duration=None):
    """Features AGG derivadas da mesma sequencia emitida (extrator-compativel)."""
    arr = np.asarray(sizes, dtype=float)
    n = len(arr)
    if n == 0:
        return {}
    cum = np.cumsum(arr)
    if flow_duration is not None:
        fd = float(flow_duration)
    else:
        fd = float(sum(max(0.0, float(x)) for x in (iats or [])))
    std_sample = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    out = {
        "flow_duration": fd,
        "Min": float(np.min(arr)),
        "Std": std_sample,
        "Tot sum": float(np.mean(cum)),
        "Covariance": float(np.var(arr, ddof=0)) if n >= 2 else 0.0,
        "Variance": float(std_sample ** 2),
        "Number": float(n - 1) / 2.0,
        "UDP": 1.0,
        "Rate": float(n) / fd if fd > 1e-9 else 0.0,
    }
    if rst_count is not None:
        out["rst_count"] = float(rst_count)
    return out


def build_window_manifest_record(state, sizes, iats, window_id=None, rst_count=None, flow_duration=None):
    """Um registro por janela AGG para calibracao por estado latente."""
    rec = compute_window_feature_vector(
        sizes, iats, rst_count=rst_count, flow_duration=flow_duration,
    )
    name = str(state).lower()
    rec["state"] = name
    rec["om_state"] = name
    rec["packet_count"] = int(len(sizes))
    if window_id is not None:
        rec["window_id"] = int(window_id)
    return rec


def _default_operational_modes_greeth():
    """
    Seis estados latentes Mirai GRE-ETH (HMM emissao conjunta).

    Cada estado emite vector completo: window_template + flow_duration +
    iat_profile + packet_count. Min/Std/Tot sum/Covariance derivam de sizes[];
    flow_duration = sum(IATs) da janela.
    """
    return {
        "enabled": False,
        "variable_window_packets": True,
        "inter_window_gap_sec": 0.003,
        "pcap_intra_iat_sec": DEFAULT_PCAP_INTRA_IAT_SEC,
        "initial_mode": "plateau",
        "modes": [
            {
                "name": "plateau",
                "weight": 0.42,
                "emission": {
                    "emission_mode": "vector",
                    "joint_emission": True,
                    "window_template": "plateau",
                    "iat_profile": "instant",
                    "flow_duration": 0.0,
                    "packet_count_range": [19, 21],
                    "template_jitter_std": 0.5,
                    "long_flow_fraction": 0.02,
                },
            },
            {
                "name": "normal",
                "weight": 0.35,
                "emission": {
                    "emission_mode": "vector",
                    "joint_emission": True,
                    "window_template": "normal",
                    "iat_profile": "uniform",
                    "flow_duration": {"model": "uniform", "low": 0.04, "high": 0.08},
                    "packet_count_range": [18, 22],
                    "template_jitter_std": 2.5,
                    "iat_jitter_frac": 0.06,
                    "long_flow_fraction": 0.04,
                },
            },
            {
                "name": "transition",
                "weight": 0.10,
                "emission": {
                    "emission_mode": "vector",
                    "joint_emission": True,
                    "window_template": "transition",
                    "iat_profile": "spread",
                    "flow_duration": {"model": "uniform", "low": 0.08, "high": 0.18},
                    "packet_count_range": [17, 23],
                    "template_jitter_std": 4.0,
                    "iat_jitter_frac": 0.10,
                    "long_flow_fraction": 0.08,
                },
            },
            {
                "name": "tail",
                "weight": 0.07,
                "emission": {
                    "emission_mode": "vector",
                    "joint_emission": True,
                    "window_template": "tail",
                    "iat_profile": "back_loaded",
                    "flow_duration": {"model": "uniform", "low": 0.20, "high": 0.40},
                    "packet_count_range": [17, 21],
                    "template_jitter_std": 6.0,
                    "iat_jitter_frac": 0.12,
                    "long_flow_fraction": 0.06,
                },
            },
            {
                "name": "burst",
                "weight": 0.04,
                "emission": {
                    "emission_mode": "vector",
                    "joint_emission": True,
                    "window_template": "burst",
                    "iat_profile": "micro_burst",
                    "flow_duration": {"model": "uniform", "low": 0.01, "high": 0.04},
                    "packet_count_range": [20, 23],
                    "template_jitter_std": 8.0,
                    "iat_jitter_frac": 0.05,
                    "long_flow_fraction": 0.10,
                },
            },
            {
                "name": "outlier",
                "weight": 0.02,
                "emission": {
                    "emission_mode": "vector",
                    "joint_emission": True,
                    "window_template": "outlier",
                    "iat_profile": "spread",
                    "flow_duration": {"model": "uniform", "low": 0.30, "high": 0.70},
                    "packet_count_range": [16, 20],
                    "template_jitter_std": 12.0,
                    "iat_jitter_frac": 0.15,
                    "long_flow_fraction": 0.12,
                },
            },
        ],
        "transition": [
            [0.90, 0.06, 0.02, 0.01, 0.01, 0.00],
            [0.08, 0.82, 0.06, 0.02, 0.02, 0.00],
            [0.06, 0.10, 0.72, 0.06, 0.04, 0.02],
            [0.12, 0.18, 0.08, 0.55, 0.05, 0.02],
            [0.15, 0.25, 0.05, 0.05, 0.45, 0.05],
            [0.20, 0.25, 0.05, 0.20, 0.05, 0.25],
        ],
    }


def _default_operational_modes_udpplain(flow_cfg=None):
    """
    HMM conjunto UDPPlain: emissao vetorial por janela AGG=20.
    flow_duration alinhado ao GMM log10 calibrado (params.flow.duration_components).
    """
    base = _default_operational_modes_greeth()
    base["enabled"] = True
    base["variable_window_packets"] = False
    base["inter_window_gap_sec"] = 0.003
    base["pcap_intra_iat_sec"] = DEFAULT_PCAP_INTRA_IAT_SEC
    duration_specs = {
        "plateau": 0.0,
        "normal": {"model": "uniform", "low": 0.85, "high": 2.5},
        "transition": {"model": "uniform", "low": 1.2, "high": 3.5},
        "tail": {"model": "uniform", "low": 2.5, "high": 6.0},
        "burst": {"model": "uniform", "low": 0.05, "high": 0.35},
        "outlier": {"model": "uniform", "low": 4.0, "high": 12.0},
    }
    for mode in base["modes"]:
        name = str(mode["name"]).lower()
        em = mode["emission"]
        em["align_agg_packets"] = True
        em["packet_count"] = 20
        em.pop("packet_count_range", None)
        em["long_flow_fraction"] = 0.0
        em["pcap_intra_iat_sec"] = DEFAULT_PCAP_INTRA_IAT_SEC
        if name in duration_specs:
            em["flow_duration"] = duration_specs[name]
        if name == "plateau":
            em["iat_profile"] = "instant"
    return base


def _normalize_operational_modes(om_cfg):
    om_cfg = om_cfg or {}
    defaults = _default_operational_modes_greeth()
    modes = om_cfg.get("modes") or defaults["modes"]
    out_modes = []
    for i, m in enumerate(modes):
        name = str(m.get("name", f"mode_{i}"))
        em = dict(m.get("emission") or {})
        if "payload_model" in em and "payload_profile" not in em:
            em["payload_profile"] = em.pop("payload_model")
        clean = {}
        for k, val in em.items():
            if k == "packet_count_range" or k in OPERATIONAL_EMISSION_KEYS:
                coerced = _coerce_emission_value(k, val)
                if coerced is not None:
                    clean[k] = coerced
            elif k.startswith("target_") and isinstance(val, dict):
                clean[k] = dict(val)
            elif k in (
                "tcp_companion", "tcp_primary", "align_agg_packets", "aggregation_packets",
                "tcp_dport", "flow_devices", "pcap_intra_iat_sec", "rst_per_packet_scale", "size_min", "size_max",
            ):
                clean[k] = val
            elif k == "joint_samples" and isinstance(val, list):
                clean[k] = val
            elif k == "rst_count_model" and isinstance(val, dict):
                clean[k] = val
            elif k == "flow_duration" and isinstance(val, dict):
                clean[k] = dict(val)
            elif k == "flow_duration" and val is not None:
                clean[k] = float(val) if not isinstance(val, dict) else dict(val)
        out_modes.append({
            "name": name,
            "weight": float(m.get("weight", 1.0)),
            "emission": clean,
        })
    total_w = sum(m["weight"] for m in out_modes)
    if total_w <= 0:
        total_w = 1.0
    for m in out_modes:
        m["weight"] /= total_w

    n = len(out_modes)
    raw_t = om_cfg.get("transition")
    if raw_t is not None and len(raw_t) == n:
        trans = []
        for row in raw_t:
            r = [float(x) for x in row[:n]]
            rs = sum(r)
            trans.append([x / rs if rs > 0 else 1.0 / n for x in r])
    else:
        stay = float(om_cfg.get("stay_prob", 0.85))
        off = (1.0 - stay) / max(1, n - 1)
        trans = []
        for i in range(n):
            row = [off] * n
            row[i] = stay
            trans.append(row)

    name_to_idx = {m["name"]: i for i, m in enumerate(out_modes)}
    init_name = str(om_cfg.get("initial_mode", out_modes[0]["name"]))
    init_idx = name_to_idx.get(init_name, 0)

    return {
        "enabled": bool(om_cfg.get("enabled", defaults["enabled"])),
        "variable_window_packets": bool(
            om_cfg.get("variable_window_packets", defaults.get("variable_window_packets", False)),
        ),
        "inter_window_gap_sec": float(
            om_cfg.get("inter_window_gap_sec", defaults.get("inter_window_gap_sec", 0.003)),
        ),
        "pcap_intra_iat_sec": float(
            om_cfg.get("pcap_intra_iat_sec", defaults.get("pcap_intra_iat_sec", DEFAULT_PCAP_INTRA_IAT_SEC)),
        ),
        "modes": out_modes,
        "transition": trans,
        "initial_idx": init_idx,
    }


class OperationalModeChain:
    """HMM: cadeia de Markov sobre modos operacionais Mirai (emissao multivariada)."""

    def __init__(self, om_cfg, rng):
        self.cfg = _normalize_operational_modes(om_cfg)
        self.rng = rng
        self.state = int(self.cfg["initial_idx"])

    def emit(self):
        """Retorna (nome_modo, dict emissao) do estado corrente."""
        mode = self.cfg["modes"][self.state]
        return mode["name"], dict(mode["emission"])

    def step(self):
        row = self.cfg["transition"][self.state]
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(row):
            acc += float(p)
            if r <= acc:
                self.state = int(i)
                return
        self.state = len(row) - 1


def _merge_wh_from_emission(wh, emission):
    """Propaga emissao do modo operacional para window_hetero / markov_gmm."""
    out = copy.deepcopy(wh or {})
    mg = dict(out.get("markov_gmm") or {})
    field_map = {
        "ar1_rho": "ar1_rho",
        "jitter": "jitter_std",
        "burst_probability": "spike_prob",
        "plateau_probability": "plateau_prob",
        "normal_probability": "normal_prob",
        "tail_probability": "tail_prob",
    }
    for src, dst in field_map.items():
        if emission.get(src) is not None:
            mg[dst] = float(emission[src])
    pl = mg.get("plateau_prob")
    ta = mg.get("tail_prob")
    no = mg.get("normal_prob")
    if no is None and pl is not None and ta is not None:
        mg["normal_prob"] = max(0.0, 1.0 - float(pl) - float(ta))
    if emission.get("payload_variance") is not None:
        pv = float(emission["payload_variance"])
        mg["jitter_std"] = max(float(mg.get("jitter_std", 2.0)), pv * 0.15)
    out["markov_gmm"] = mg
    profile = emission.get("payload_profile") or emission.get("payload_model")
    if profile:
        out["tot_sum_profile"] = str(profile)
    bm = dict(out.get("burst_model") or {})
    bp = emission.get("burst_probability")
    if bp is not None:
        bm["enabled"] = float(bp) > 0.001
        bm["probability"] = float(bp)
    out["burst_model"] = bm
    return out


def _sample_window_floor_from_emission(rng, np_rng, size_cfg, emission):
    """Min por janela a partir do regime emitido (constant|normal|tail|mixture)."""
    min_sz = int(size_cfg.get("min", 42))
    max_sz = int(size_cfg.get("max", 1500))
    regime = str(emission.get("payload_regime", "mixture")).lower()

    if regime == "constant" and emission.get("payload_floor") is not None:
        base = float(emission["payload_floor"])
        var = float(emission.get("payload_variance", 0.0))
        if var > 0:
            base = float(np_rng.normal(base, var))
        floor = int(max(min_sz, min(base, max_sz)))
        return floor, _role_from_floor_value(floor)

    if regime == "normal":
        mean = float(emission.get("payload_normal_mean", 520.0))
        std = max(1e-6, float(emission.get("payload_normal_std", 40.0)))
        floor = int(max(min_sz, min(float(np_rng.normal(mean, std)), max_sz)))
        return floor, "normal"

    if regime == "tail":
        base = float(emission.get("payload_floor", 380.0))
        var = max(1e-6, float(emission.get("payload_variance", 45.0)))
        floor = int(max(min_sz, min(float(np_rng.normal(base, var)), max_sz)))
        return floor, "tail"

    if emission.get("payload_floor") is not None:
        base = float(emission["payload_floor"])
        var = float(emission.get("payload_variance", 0.0))
        if var > 0:
            base = float(np_rng.normal(base, var))
        floor = int(max(min_sz, min(base, max_sz)))
        return floor, _role_from_floor_value(floor)

    pl = float(emission.get("plateau_probability", 0.46))
    ta = float(emission.get("tail_probability", 0.05))
    no = float(emission.get("normal_probability", max(0.0, 1.0 - pl - ta)))
    total = pl + no + ta
    if total <= 0:
        pl, no, ta = 0.46, 0.49, 0.05
        total = 1.0
    sc = dict(size_cfg)
    sc["markov_role_probs"] = {"plateau": pl / total, "normal": no / total, "tail": ta / total}
    return sample_window_floor(rng, np_rng, sc)


def _sample_window_packet_count(np_rng, emission, default_n, om_cfg=None):
    """
    Pacotes por janela AGG. Com index_mean, Number=(n-1)/2 -> n in [17,23] => Number 8..11.
    """
    om_cfg = om_cfg or {}
    if emission.get("packet_count_range"):
        lo, hi = emission["packet_count_range"][:2]
        return int(np_rng.integers(int(lo), int(hi) + 1))
    base = float(emission.get("packet_count", default_n))
    jitter = float(emission.get("packet_count_jitter", 0.0))
    if jitter > 0:
        base = float(np_rng.normal(base, jitter))
    lo, hi = om_cfg.get("packet_count_bounds", [17, 23])
    return int(max(int(lo), min(int(hi), round(base))))


def _flow_duration_from_emission(emission, flow_cfg, rng, np_rng):
    raw = emission.get("flow_duration")
    if raw is None:
        return sample_flow_duration(rng, flow_cfg, np_rng)
    if isinstance(raw, dict):
        d = _sample_duration_mixture_component(rng, np_rng, raw)
    else:
        d = float(raw)
    d_min = float(flow_cfg.get("duration_min", 1e-9))
    d_max = float(flow_cfg.get("duration_max", 300.0))
    return max(d_min, min(d, d_max))


DEFAULT_PCAP_INTRA_IAT_SEC = 1e-4
DEFAULT_AGG_BURST_IDLE_SEC = 0.0025


def _agg_burst_idle_sec():
    try:
        return float(os.environ.get("AGG_BURST_IDLE_SEC", DEFAULT_AGG_BURST_IDLE_SEC))
    except ValueError:
        return DEFAULT_AGG_BURST_IDLE_SEC


def _pacing_wh(params, emission=None):
    """Config de pacing PCAP (window_hetero + gre_flow + emissao HMM)."""
    em = emission or {}
    gf = params.get("gre_flow") or {}
    wh = (params.get("dynamic_streams") or {}).get("window_hetero") or {}
    om = wh.get("operational_modes") or {}
    return em, gf, wh, om


def resolve_pcap_intra_iat_mode(emission, params):
    em, gf, wh, om = _pacing_wh(params, emission)
    raw = em.get("pcap_intra_iat_mode")
    if raw is None:
        raw = gf.get(
            "pcap_intra_iat_mode",
            wh.get("pcap_intra_iat_mode", "fixed"),
        )
    return str(raw).strip().lower()


def resolve_pcap_intra_iat_bounds(emission, params, n_pkts=20):
    """
    Limites de IAT intra-janela. Modo uniform: [low, high] por pacote.
    Modo fixed: low=high=pcap_intra_iat_sec (apos cap burst opcional).
    """
    em, gf, wh, om = _pacing_wh(params, emission)
    mode = resolve_pcap_intra_iat_mode(emission, params)
    raw = em.get("pcap_intra_iat_sec")
    if raw is None:
        raw = om.get(
            "pcap_intra_iat_sec",
            gf.get("pcap_intra_iat_sec", wh.get("pcap_intra_iat_sec", DEFAULT_PCAP_INTRA_IAT_SEC)),
        )
    center = float(raw)
    lo = float(em.get("pcap_intra_iat_low_sec", gf.get(
        "pcap_intra_iat_low_sec", wh.get("pcap_intra_iat_low_sec", center * 0.67))))
    hi = float(em.get("pcap_intra_iat_high_sec", gf.get(
        "pcap_intra_iat_high_sec", wh.get("pcap_intra_iat_high_sec", center * 1.33))))
    if lo > hi:
        lo, hi = hi, lo

    cap_burst = em.get("pcap_iat_cap_burst")
    if cap_burst is None:
        cap_burst = gf.get("pcap_iat_cap_burst", wh.get("pcap_iat_cap_burst", True))
    if cap_burst is not False and str(cap_burst).lower() not in ("0", "false", "no"):
        burst = _agg_burst_idle_sec()
        max_iat = burst * 0.8 / max(1, int(n_pkts) - 1)
        center = min(center, max_iat)
        lo = min(lo, max_iat)
        hi = min(hi, max_iat)

    if mode != "uniform":
        lo = hi = center
    return mode, lo, hi


def resolve_pcap_intra_iat_sec(emission, params, n_pkts=20):
    """
    IAT fixo de referencia (media no modo uniform). Com pcap_iat_cap_burst=false
    o cap AGG_BURST_IDLE_SEC nao limita o IAT — util no live com extract AGG=0.
    """
    mode, lo, hi = resolve_pcap_intra_iat_bounds(emission, params, n_pkts)
    if mode == "uniform":
        return (lo + hi) / 2.0
    return lo


def make_pcap_intra_iats(n_pkts, pcap_iat_sec, *, rng=None, params=None, emission=None):
    """IATs entre pacotes consecutivos dentro de uma janela AGG."""
    n_iats = max(0, int(n_pkts) - 1)
    if n_iats == 0:
        return []
    if params is not None:
        mode, lo, hi = resolve_pcap_intra_iat_bounds(emission, params, n_pkts)
        if mode == "uniform" and rng is not None:
            return [max(1e-6, float(rng.uniform(lo, hi))) for _ in range(n_iats)]
        step = lo if mode == "uniform" else float(pcap_iat_sec)
        return [max(1e-6, step)] * n_iats
    step = float(pcap_iat_sec)
    return [max(1e-6, step)] * n_iats


def resolve_inter_window_gap_sec(emission, params, om_norm=None, *, n_pkts=20, n_bots=1, pcap_iat_sec=None):
    """
    Pausa entre janelas AGG no PCAP: deve exceder AGG_BURST_IDLE_SEC para
    o extrator nao fundir duas janelas logicas num unico slice.

    UDPPlain: se gre_flow.target_rate_pps estiver definido, dimensiona o gap
    para Rate (modo flow) ≈ n_pkts / (n_bots * ciclo), com ciclo = micro_span + gap.
    Round-robin entre bots reduz a taxa por fluxo; com AGG=20 o maximo sustentavel
    de bots ≈ n_pkts / (target_rate * (micro + gap_min)).
    """
    em = emission or {}
    gf = params.get("gre_flow") or {}
    om_norm = om_norm or {}
    wh = (params.get("dynamic_streams") or {}).get("window_hetero") or {}
    om = wh.get("operational_modes") or {}
    raw = em.get("inter_window_gap_sec")
    if raw is None:
        raw = om_norm.get(
            "inter_window_gap_sec",
            om.get(
                "inter_window_gap_sec",
                gf.get(
                    "inter_window_gap_sec",
                    gf.get("window_gap", params.get("inter_flow_delay", 0.003)),
                ),
            ),
        )
    gap = float(raw)
    cap_burst = em.get("inter_window_gap_cap_burst")
    if cap_burst is None:
        cap_burst = gf.get(
            "inter_window_gap_cap_burst",
            wh.get("inter_window_gap_cap_burst", True),
        )
    if cap_burst is False or str(cap_burst).lower() in ("0", "false", "no"):
        min_gap = gap
    else:
        burst = _agg_burst_idle_sec()
        min_gap = max(gap, burst * 1.25)

    target_rate = float(gf.get("target_rate_pps", 0) or 0)
    if target_rate > 0 and str(params.get("variant", "")).lower() == "udpplain":
        if pcap_iat_sec is None:
            pcap_iat_sec = resolve_pcap_intra_iat_sec(emission, params, n_pkts)
        micro_span = max(0.0, (max(1, int(n_pkts)) - 1) * float(pcap_iat_sec))
        eff_bots = max(1, int(n_bots))
        cycle_target = float(n_pkts) / (target_rate * eff_bots)
        paced_gap = cycle_target - micro_span
        if paced_gap >= min_gap - 1e-12:
            return max(min_gap, paced_gap)

    return min_gap


def summarize_pacing(params, emission=None, *, n_pkts=20, n_bots=1):
    """Resumo para logs inject/audit (modo fixed ou uniform + gap desacoplado)."""
    mode, iat_lo, iat_hi = resolve_pcap_intra_iat_bounds(emission, params, n_pkts)
    pcap_mean = resolve_pcap_intra_iat_sec(emission, params, n_pkts)
    win_gap = resolve_inter_window_gap_sec(
        emission, params, None, n_pkts=n_pkts, n_bots=n_bots, pcap_iat_sec=pcap_mean,
    )
    n_iat = max(0, int(n_pkts) - 1)
    if mode == "uniform":
        intra_lo = n_iat * iat_lo
        intra_hi = n_iat * iat_hi
        intra_mean = n_iat * (iat_lo + iat_hi) / 2.0
    else:
        intra_lo = intra_hi = intra_mean = n_iat * iat_lo
    em, gf, wh, _om = _pacing_wh(params, emission)
    cap_burst = wh.get("pcap_iat_cap_burst", gf.get("pcap_iat_cap_burst", True))
    gap_cap = wh.get("inter_window_gap_cap_burst", gf.get("inter_window_gap_cap_burst", True))
    fd_include_gap = bool(wh.get("flow_duration_include_inter_gap", False))
    window_span_mean = intra_mean + (win_gap if fd_include_gap else 0.0)
    return {
        "mode": mode,
        "iat_lo": iat_lo,
        "iat_hi": iat_hi,
        "intra_span_mean": intra_mean,
        "intra_span_lo": intra_lo,
        "intra_span_hi": intra_hi,
        "inter_gap": win_gap,
        "pcap_iat_mean": pcap_mean,
        "pcap_iat_cap_burst": cap_burst,
        "inter_window_gap_cap_burst": gap_cap,
        "cycle_mean": intra_mean + win_gap,
        "flow_duration_include_inter_gap": fd_include_gap,
        "flow_duration_window_mean": window_span_mean,
    }


def _window_iats_from_emission(rng, np_rng, iat_cfg, flow_cfg, emission, n_pkts):
    if n_pkts <= 1:
        return []
    duration = _flow_duration_from_emission(emission, flow_cfg, rng, np_rng)
    n_iat = max(2, int(emission.get("packet_count", n_pkts)))
    n_iat = min(n_iat, n_pkts)
    iats = sample_iat_sequence(rng, np_rng, iat_cfg, n_iat, duration)
    scale = float(emission.get("inter_arrival_scale", 1.0))
    if scale != 1.0:
        iats = [max(0.0, x * scale) for x in iats]
    if len(iats) < n_pkts - 1:
        iats.extend([iats[-1] if iats else 0.0] * (n_pkts - 1 - len(iats)))
    return iats[: max(0, n_pkts - 1)]


def _uses_semantic_pcap_iats(variant, flow_cfg, use_latent_hmm):
    """GRE-IP/GRE-ETH: PCAP timestamps seguem zero_inflated + sample_iat_sequence."""
    if use_latent_hmm:
        return False
    if str(variant).lower() not in ("greip", "greeth"):
        return False
    model = str(flow_cfg.get("duration_model", "")).lower()
    return model.startswith("zero_inflated") or float(flow_cfg.get("duration_zero_fraction", 0)) > 0


def _greip_uses_semantic_pcap_iats(variant, flow_cfg, use_latent_hmm):
    return _uses_semantic_pcap_iats(variant, flow_cfg, use_latent_hmm)


def _pcap_iats_for_window(
    rng,
    params,
    emission,
    n_pkts,
    win_iats,
    win_gap,
    wh,
    variant,
    flow_cfg,
    use_latent_hmm,
):
    """
    IATs efectivos entre pacotes da janela AGG (alimentam ts no PCAP).

    GRE-IP zero-inflated: reutiliza win_iats de sample_flow_duration +
    sample_iat_sequence (zeros => FLOW_DURATION_MODE=window == 0).
    Demais variantes: pacing pcap_intra_iat uniform/fixo + gap opcional.
    """
    n_iats = max(0, int(n_pkts) - 1)
    if _uses_semantic_pcap_iats(variant, flow_cfg, use_latent_hmm):
        pcap_iats = list(win_iats or [])
        if len(pcap_iats) < n_iats:
            pad = float(pcap_iats[-1]) if pcap_iats else 0.0
            pcap_iats.extend([pad] * (n_iats - len(pcap_iats)))
        return [max(0.0, float(x)) for x in pcap_iats[:n_iats]]

    pcap_iat_sec = resolve_pcap_intra_iat_sec(emission, params, n_pkts)
    pcap_iats = make_pcap_intra_iats(
        n_pkts, pcap_iat_sec, rng=rng, params=params, emission=emission,
    )
    if wh.get("flow_duration_include_inter_gap", False) and pcap_iats:
        pcap_iats = list(pcap_iats)
        pcap_iats[-1] = max(1e-6, float(pcap_iats[-1]) + float(win_gap))
    return pcap_iats


def build_gre_window_hetero_schedule(params):
    """
    Window-hetero AGG=20 (GRE-ETH / GRE-IP / UDPPlain).

    Emissao conjunta por estado HMM; PCAP usa pcap_intra_iat_sec + inter_window_gap.
    """
    rng = random.Random(params["seed"])
    np_rng = np.random.default_rng(params["seed"])
    variant = str(params.get("variant", "greeth")).lower()
    is_udpplain = variant == "udpplain"
    gre_flow = params.get("gre_flow", {})
    size_cfg = params["size"]
    bot = params["botnet"]
    ds_cfg = params.get("dynamic_streams") or {}
    wh = dict(ds_cfg.get("window_hetero", {}))
    if is_udpplain:
        om_raw = dict(wh.get("operational_modes") or {})
        if om_raw.get("enabled") is not False:
            if not om_raw.get("modes"):
                om_raw = _default_operational_modes_udpplain(params.get("flow", {}))
            om_raw["enabled"] = True
        wh["operational_modes"] = om_raw
        ds_cfg = dict(ds_cfg)
        ds_cfg["window_hetero"] = wh
    agg_n = _aggregation_window_size(size_cfg, ds_cfg, gre_flow)

    n_bots = int(bot["size"])
    flows_per_bot = int(gre_flow.get("flows_per_bot", 50))
    n_windows = int(gre_flow.get("target_windows", max(1, n_bots * flows_per_bot)))
    src_base = gre_flow.get("src_base", bot.get("src_base", "10.45.0."))
    dst_base = gre_flow.get("dst_base", "10.45.1.")
    src_first = int(gre_flow.get("src_first_octet", bot.get("src_first_octet", 20)))
    dst_first = int(gre_flow.get("dst_first_octet", 1))

    iat_cfg = params.get("iat", {})
    flow_cfg = params.get("flow", {})
    om_cfg = wh.get("operational_modes") or {}
    use_latent_hmm = bool(om_cfg.get("enabled"))
    om_norm = _normalize_operational_modes(om_cfg) if use_latent_hmm else {}
    window_gap = resolve_inter_window_gap_sec(None, params, om_norm if use_latent_hmm else None)
    long_flow_frac = 0.0 if is_udpplain else float(wh.get(
        "long_flow_fraction",
        1.0 - float(gre_flow.get("single_packet_fraction", 0.72)),
    ))
    long_reset_after = int(wh.get("long_flow_reset_after_packets", 400))
    long_reset_p = float(wh.get("long_flow_reset_prob", 0.35))
    live_rt = params.get("_live_routing")
    min_trace = _init_min_trace(params)
    if variant in ("greeth", "greip") and not use_latent_hmm:
        _log_min_generation_path(params)
    per_window_dst = (
        live_rt
        and live_rt.get("vary_dst")
        and live_rt.get("dst_strategy") == "unique_per_window"
    )
    mode_chain = OperationalModeChain(om_cfg, rng) if use_latent_hmm else None
    mode_counts = {}
    var_window_pkts = bool(om_norm.get("variable_window_packets")) if use_latent_hmm else False
    if use_latent_hmm and var_window_pkts:
        window_gap = max(
            window_gap,
            resolve_inter_window_gap_sec(None, params, om_norm),
        )
    log_tag = "udpplain_latent_hmm" if is_udpplain and use_latent_hmm else (
        "gre_latent_hmm" if use_latent_hmm else "gre_window_hetero"
    )

    events = []
    window_manifest = []
    t = 0.0
    flow_idx = 0
    regime_windows = {"plateau": 0, "normal": 0, "tail": 0}
    long_windows = 0
    long_flow = None
    udp_states = {}
    dst_ip_fixed = bot.get("dst_ip", "10.45.0.1")
    default_dport = int(gre_flow.get("udp_dport", bot.get("dport", 44768)))
    udp_long_cfg = _resolve_udp_long_flow_cfg(params, wh, gre_flow) if is_udpplain else None

    def _gre_for_regime(regime):
        if regime == "plateau":
            return 0x0000, None, 4
        return pick_gre_variant(rng, params["gre"])

    def _flow_ips(window_id=None):
        nonlocal flow_idx
        if live_rt:
            src_ip, dst_ip = _live_routing_flow_ips(
                live_rt, flow_idx, window_id=window_id,
            )
        else:
            idx = flow_idx if window_id is None else window_id
            src_ip = _expand_ipv4(src_base, src_first + idx)
            dst_ip = _expand_ipv4(dst_base, dst_first + idx)
        if window_id is None:
            flow_idx += 1
        return src_ip, dst_ip

    def _new_flow(regime, window_id=None):
        nonlocal long_flow
        src_ip, dst_ip = _flow_ips(window_id)
        gre_f, gre_k, gre_l = _gre_for_regime(regime)
        long_flow = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "gre_flags": gre_f,
            "gre_key": gre_k,
            "gre_len": gre_l,
            "ttl": int(params.get("ttl", {}).get("mean", 64)),
            "pkt_in_flow": 0,
        }

    def _emit(
        src_ip, dst_ip, target_size, iat_after=None,
        *, gre_f=None, gre_k=None, gre_l=None, ttl=None, sport=0, dport=0,
    ):
        nonlocal t

        pkt = build_packet(
            rng, np_rng, params,
            src_ip, dst_ip,
            sport, dport,
            force_gre_flags=gre_f,
            force_gre_key=gre_k,
            force_gre_len=gre_l,
            flow_ttl=ttl,
            target_size=target_size,
        )

        events.append((t, pkt))

        if iat_after is not None:
            t += max(0.0, float(iat_after))

    def _udp_bot_id(window_id):
        return int(window_id) % max(1, n_bots)

    def _ensure_udp_flow(bot_id, duration_target, *, long_lived=False):
        dur = max(0.05, float(duration_target or sample_flow_duration(rng, flow_cfg, np_rng)))
        prev = udp_states.get(bot_id) or {}
        sport_seq = int(prev.get("sport_seq", 0)) + (0 if long_lived and prev else 1)
        if long_lived and prev.get("sport") is not None:
            sport = int(prev["sport"])
            dport = int(prev.get("dport", default_dport))
            pkts = int(prev.get("pkts_in_flow", 0))
        else:
            sport = 1024 + (int(bot_id) * 17 + sport_seq * 31 + rng.randint(0, 99)) % 64000
            dport = default_dport if default_dport > 0 else rng.randint(1024, 65535)
            pkts = 0
        deadline = (t + 1e12) if long_lived else (t + dur)
        udp_states[bot_id] = {
            "sport": sport,
            "dport": dport,
            "deadline_t": deadline,
            "duration_target": dur,
            "sport_seq": sport_seq,
            "episode_t0": t,
            "long_lived": bool(long_lived),
            "pkts_in_flow": pkts,
        }

    def _udp_flow_needs_reset(bot_id):
        st = udp_states.get(bot_id)
        if st is None:
            return True
        if st.get("long_lived"):
            return False
        return t >= float(st.get("deadline_t", 0.0))

    def _window_iats(n_pkts, emission=None):
        if emission is not None:
            return _window_iats_from_emission(
                rng, np_rng, iat_cfg, flow_cfg, emission, n_pkts,
            )
        duration = sample_flow_duration(rng, flow_cfg, np_rng)
        if n_pkts <= 1:
            return []
        return sample_iat_sequence(rng, np_rng, iat_cfg, n_pkts, duration)

    if udp_long_cfg:
        bot_id = 0
        src_ip = _expand_ipv4(src_base, src_first + bot_id)
        dst_ip = dst_ip_fixed
        _ensure_udp_flow(bot_id, None, long_lived=True)
        st = udp_states[bot_id]
        warmup_n = int(udp_long_cfg["warmup_packets"])
        warmup_iat = float(udp_long_cfg["warmup_iat_sec"])
        warmup_size = int(udp_long_cfg["warmup_size"])
        print(
            f"[udp_long_flow] warmup {warmup_n} pkts @ {warmup_iat}s "
            f"(Header_Length alvo med ~{udp_long_cfg['header_length_median']:.0f})",
            flush=True,
        )
        for _ in range(warmup_n):
            _emit(
                src_ip, dst_ip, warmup_size, warmup_iat,
                sport=st["sport"], dport=st["dport"],
            )
            st["pkts_in_flow"] = int(st.get("pkts_in_flow", 0)) + 1

    for wi in range(n_windows):
        if wi == 0:
            est = n_windows * agg_n
            if use_latent_hmm:
                extra = " [HMM conjunto: sizes+duration+IAT por estado]"
            else:
                extra = ""
            print(f"[{log_tag}] a gerar {n_windows} janelas x {agg_n} pkts (~{est} pacotes)...{extra}", flush=True)
        elif wi % 500 == 0:
            print(f"[{log_tag}] janela {wi}/{n_windows} ({len(events)} pkts)...", flush=True)

        emission = None
        mode_name = None
        win_wh = wh
        win_long_frac = long_flow_frac
        if mode_chain is not None:
            mode_name, emission = mode_chain.emit()
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
            win_wh = _merge_wh_from_emission(wh, emission)
            if emission.get("long_flow_fraction") is not None:
                win_long_frac = float(emission["long_flow_fraction"])
            ds_window = dict(ds_cfg)
            ds_window["window_hetero"] = win_wh
        else:
            ds_window = ds_cfg

        win_n = (
            _sample_window_packet_count(np_rng, emission or {}, agg_n, om_norm)
            if use_latent_hmm and var_window_pkts
            and not (emission or {}).get("align_agg_packets", gre_flow.get("align_agg_packets", True))
            else agg_n
        )

        use_vector = emission is not None and _emission_uses_window_vector(emission)
        use_joint = use_vector and bool(emission.get("joint_emission", True))
        manifest_fd = None
        if use_joint:
            sizes, win_iats, floor, regime, manifest_fd = _window_joint_emission(
                emission, win_n, size_cfg, flow_cfg, rng, np_rng, variant=variant,
            )
        elif use_vector:
            sizes, floor, regime = _window_sizes_from_emission(
                emission, win_n, size_cfg, np_rng, variant=variant,
            )
            win_iats = _window_iats_from_emission(
                rng, np_rng, iat_cfg, flow_cfg, emission, len(sizes),
            )
            manifest_fd = _flow_duration_from_emission(emission, flow_cfg, rng, np_rng)
        else:
            if emission is not None:
                floor, regime = _sample_window_floor_from_emission(rng, np_rng, size_cfg, emission)
            else:
                floor, regime = sample_window_floor(
                    rng, np_rng, size_cfg, trace=min_trace, window_idx=wi,
                )
            sizes = expand_intra_window_sizes(
                rng, np_rng, floor, win_n, ds_window, size_cfg,
                trace=min_trace, window_idx=wi,
            )
            win_iats = _window_iats(len(sizes), emission)
            if emission is not None:
                manifest_fd = _flow_duration_from_emission(emission, flow_cfg, rng, np_rng)

        manifest_iats = win_iats
        pcap_iat_sec = resolve_pcap_intra_iat_sec(emission, params, len(sizes))
        win_gap = resolve_inter_window_gap_sec(
            emission,
            params,
            om_norm if use_latent_hmm else None,
            n_pkts=len(sizes),
            n_bots=n_bots,
            pcap_iat_sec=pcap_iat_sec,
        )
        pcap_iats = _pcap_iats_for_window(
            rng, params, emission, len(sizes), win_iats, win_gap, wh,
            variant, flow_cfg, use_latent_hmm,
        )

        regime_windows[regime] = regime_windows.get(regime, 0) + 1
        if sizes:
            if use_latent_hmm and mode_name:
                window_manifest.append(
                    build_window_manifest_record(
                        mode_name, sizes, manifest_iats, flow_duration=manifest_fd,
                    ),
                )
            elif variant in ("greip", "greeth"):
                fd_intended = float(sum(max(0.0, float(x)) for x in (pcap_iats or [])))
                window_manifest.append(
                    build_window_manifest_record(
                        regime, sizes, pcap_iats, window_id=wi,
                        flow_duration=fd_intended,
                    ),
                )
            elif variant == "udpplain":
                fd_intended = float(sum(max(0.0, float(x)) for x in (pcap_iats or [])))
                window_manifest.append(
                    build_window_manifest_record(
                        regime or mode_name or "udpplain",
                        sizes,
                        pcap_iats,
                        window_id=wi,
                        flow_duration=fd_intended,
                    ),
                )
        is_long = rng.random() < win_long_frac
        if mode_chain is not None:
            mode_chain.step()
        if per_window_dst:
            win_src, win_dst = _flow_ips(wi)

        if is_udpplain:
            if per_window_dst:
                src_ip, dst_ip = win_src, win_dst
                bot_id = wi
            else:
                bot_id = _udp_bot_id(wi)
                src_ip = _expand_ipv4(src_base, src_first + bot_id)
                dst_ip = dst_ip_fixed
            if _udp_flow_needs_reset(bot_id):
                _ensure_udp_flow(bot_id, manifest_fd, long_lived=bool(udp_long_cfg))
            st = udp_states[bot_id]
            for pi, target_size in enumerate(sizes):
                iat = pcap_iats[pi] if pi < len(pcap_iats) else None
                _emit(
                    src_ip, dst_ip, target_size, iat,
                    sport=st["sport"], dport=st["dport"],
                )
                st["pkts_in_flow"] = int(st.get("pkts_in_flow", 0)) + 1
        elif is_long:
            long_windows += 1
            if long_flow is None:
                _new_flow(regime, wi if per_window_dst else None)
            elif long_flow["pkt_in_flow"] >= long_reset_after and rng.random() < long_reset_p:
                _new_flow(regime, wi if per_window_dst else None)
            lf = long_flow
            for pi, target_size in enumerate(sizes):
                iat = pcap_iats[pi] if pi < len(pcap_iats) else None
                _emit(
                    lf["src_ip"], lf["dst_ip"], target_size, iat,
                    gre_f=lf["gre_flags"], gre_k=lf["gre_key"], gre_l=lf["gre_len"],
                    ttl=lf["ttl"],
                )
                lf["pkt_in_flow"] += 1
        else:
            gre_f, gre_k, gre_l = _gre_for_regime(regime)
            ttl = int(params.get("ttl", {}).get("mean", 64))
            if per_window_dst:
                src_ip, dst_ip = win_src, win_dst
            else:
                src_ip, dst_ip = _flow_ips(wi)
            for pi, target_size in enumerate(sizes):
                iat = pcap_iats[pi] if pi < len(pcap_iats) else None
                _emit(
                    src_ip, dst_ip, target_size, iat,
                    gre_f=gre_f, gre_k=gre_k, gre_l=gre_l, ttl=ttl,
                )

        t += win_gap

    if window_manifest:
        params["_window_manifest"] = window_manifest

    events.sort(key=lambda e: e[0])
    total = len(events)
    pw = regime_windows["plateau"] / max(1, n_windows)
    lw = long_windows / max(1, n_windows)
    msg = (f"[{log_tag}] {n_windows} janelas x {agg_n} pkts -> {total} pacotes "
           f"| plateau/normal/tail={regime_windows['plateau']}/{regime_windows['normal']}/"
           f"{regime_windows['tail']} ({pw:.1%} platô) | fluxos longos={lw:.1%}")
    if mode_counts:
        modes_str = ",".join(f"{k}={v}" for k, v in sorted(mode_counts.items()))
        msg += f" | modos=[{modes_str}]"
    print(msg)
    if min_trace is not None:
        min_trace.summarize()
    fd_note = wh.get("flow_duration_include_inter_gap", False)
    greip_sem = _uses_semantic_pcap_iats(variant, flow_cfg, use_latent_hmm)
    if greip_sem:
        zf = float(flow_cfg.get("duration_zero_fraction", 0))
        print(
            f"[flow_duration] {variant.upper()} zero_inflated: PCAP IATs = sample_iat_sequence "
            f"(~{zf:.0%} janelas fd=0 com FLOW_DURATION_MODE=window); "
            f"inter_window_gap ({window_gap:.3f}s) entre janelas, fora do span",
            flush=True,
        )
    elif fd_note:
        print(
            "[flow_duration] FLOW_DURATION_MODE=window usa ts.max-ts.min por janela AGG; "
            f"flow_duration_include_inter_gap=ON -> ultimo IAT intra inclui inter_gap ({window_gap:.3f}s)",
            flush=True,
        )
    else:
        print(
            "[flow_duration] FLOW_DURATION_MODE=window: med ~ (n_pkts-1)*IAT intra "
            f"(~{resolve_pcap_intra_iat_sec(None, params, agg_n) * max(0, agg_n - 1):.3f}s); "
            f"inter_window_gap ({window_gap:.3f}s) fica ENTRE fluxos/janelas, fora do span",
            flush=True,
        )
    return events

# --------------------------------------------------------------------------- #
# Construcao dos pacotes                                                       #
# --------------------------------------------------------------------------- #
def build_packet(rng, np_rng, params, src_ip, dst_ip, sport=0, dport=0,
                 force_gre_flags=None, force_gre_key=None, force_gre_len=None,
                 flow_ttl=None, target_size=None, size_sampler=None):
    """Monta um pacote (greip | greeth | udpplain) com tamanho-alvo sorteado."""
    variant = params["variant"]
    size_cfg = params["size"]
    gre_cfg = params["gre"]
    # bytes adicionados APOS a geracao, antes do extrator (Ethernet do converter.py)
    capture_overhead = params.get("capture_overhead", 14)

    if target_size is None:
        if size_sampler is not None:
            target_size = size_sampler.next_size()
        else:
            target_size = sample_packet_size(rng, np_rng, size_cfg)

    if variant == "udpplain":
        return _build_udp_packet_exact(
            src_ip, dst_ip, sport, dport, int(target_size),
        )

    if force_gre_flags is not None:
        flags, keyval, gre_len = force_gre_flags, force_gre_key, force_gre_len
    else:
        flags, keyval, gre_len = pick_gre_variant(rng, gre_cfg)

    if variant == "greip":
        ttl = int(flow_ttl if flow_ttl is not None else params.get("ttl", {}).get("mean", 64))
        return _build_greip_packet_exact(
            src_ip, dst_ip, int(target_size), flags, keyval, gre_len, flow_ttl=ttl,
        )

    else:  # greeth
        ttl = int(flow_ttl if flow_ttl is not None else params.get("ttl", {}).get("mean", 64))
        pkt = _build_greeth_packet_exact(
            src_ip, dst_ip, int(target_size), flags, keyval, gre_len, flow_ttl=ttl,
        )

    return pkt


def _greeth_frame_payload_len(target_frame, gre_len, capture_overhead=14):
    """
    Estimativa inicial de payload para len(Ether) ~= target_frame.
    Use _greeth_packet_exact para ajuste fino com GRE KEY/CSUM.
    """
    del capture_overhead
    return max(0, int(target_frame) - 14 - 20 - int(gre_len) - 14)


def _measure_eth_frame(ip_pkt):
    """Replica pcap_to_csv.add_ethernet para medir len(eth) do extrator."""
    from scapy.all import Ether, IP
    if IP not in ip_pkt:
        return 0
    return len(Ether(src="aa:bb:cc:dd:ee:ff", dst="11:22:33:44:55:66", type=0x0800) / ip_pkt[IP])


def _measure_greeth_eth_frame(ip_pkt):
    return _measure_eth_frame(ip_pkt)


def _build_greeth_ip_packet(src_ip, dst_ip, flow_ttl, flags, keyval, gre_len, payload_len):
    from scapy.all import IP, GRE, Ether, Raw
    GRE_KEY = 0x2000
    outer = IP(src=src_ip, dst=dst_ip, proto=47, ttl=flow_ttl)
    gre = GRE(proto=0x6558, flags=flags)
    if flags & GRE_KEY:
        gre.key = keyval
    inner = Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55") / Raw(b"A" * payload_len)
    return outer / gre / inner


def _build_greeth_packet_exact(src_ip, dst_ip, target_frame, flags, keyval, gre_len, flow_ttl=64):
    """
    GRE-ETH com len(Ether/IP) == target_frame apos pcap_to_csv.add_ethernet.

    O geo_adj legado em build_packet inflava + (capture_overhead-14) bytes (ex.: +29
    com overhead 43 -> Min 592 virava 635 no CSV). Busca binaria monotona no payload.
    """
    target_frame = max(0, int(target_frame))
    if target_frame == 0:
        return _build_greeth_ip_packet(src_ip, dst_ip, flow_ttl, flags, keyval, gre_len, 0)

    est = _greeth_frame_payload_len(target_frame, gre_len)
    lo = max(0, est - 32)
    hi = est + 32 + target_frame
    best_pl, best_diff = est, 10**9
    while lo <= hi:
        mid = (lo + hi) // 2
        measured = _measure_greeth_eth_frame(
            _build_greeth_ip_packet(src_ip, dst_ip, flow_ttl, flags, keyval, gre_len, mid)
        )
        diff = measured - target_frame
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_pl = mid
        if measured < target_frame:
            lo = mid + 1
        elif measured > target_frame:
            hi = mid - 1
        else:
            best_pl = mid
            break
    return _build_greeth_ip_packet(src_ip, dst_ip, flow_ttl, flags, keyval, gre_len, best_pl)


def _build_greip_ip_packet(src_ip, dst_ip, flags, keyval, gre_len, inner_payload_len, flow_ttl=64, ip_opt_len=20):
    """GRE-IP: IP(ext)/GRE/IP(int)/payload."""
    outer = IP(
        src=src_ip, dst=dst_ip, proto=47, ttl=flow_ttl,
        options=make_ip_options(ip_opt_len),
    )
    gre = GRE(proto=0x0800, flags=flags)
    if flags & GRE_KEY:
        gre.key = keyval
    inner = IP(src="1.1.1.1", dst="2.2.2.2") / Raw(b"G" * max(0, int(inner_payload_len)))
    return outer / gre / inner


def _greip_frame_payload_len(target_frame, gre_len, ip_opt_len=20):
    """Estimativa inicial de payload interno para len(Ether) ~= target_frame."""
    return max(0, int(target_frame) - 14 - 20 - int(ip_opt_len) - int(gre_len) - 20)


def _build_greip_packet_exact(src_ip, dst_ip, target_frame, flags, keyval, gre_len, flow_ttl=64, ip_opt_len=20):
    """
    GRE-IP com len(Ether) == target_frame apos pcap_to_csv.add_ethernet.
    Busca binaria no payload interno (como greeth).
    """
    target_frame = max(0, int(target_frame))
    if target_frame == 0:
        return _build_greip_ip_packet(src_ip, dst_ip, flags, keyval, gre_len, 0, flow_ttl, ip_opt_len)

    est = _greip_frame_payload_len(target_frame, gre_len, ip_opt_len)
    lo = max(0, est - 48)
    hi = est + 48 + target_frame
    best_pl, best_diff = est, 10**9
    while lo <= hi:
        mid = (lo + hi) // 2
        measured = _measure_eth_frame(
            _build_greip_ip_packet(src_ip, dst_ip, flags, keyval, gre_len, mid, flow_ttl, ip_opt_len)
        )
        diff = measured - target_frame
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_pl = mid
        if measured < target_frame:
            lo = mid + 1
        elif measured > target_frame:
            hi = mid - 1
        else:
            best_pl = mid
            break
    return _build_greip_ip_packet(src_ip, dst_ip, flags, keyval, gre_len, best_pl, flow_ttl, ip_opt_len)


def _build_udp_ip_packet(src_ip, dst_ip, sport, dport, payload_len):
    return IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / Raw(b"U" * max(0, int(payload_len)))


def _udp_frame_payload_len(target_frame):
    return max(0, int(target_frame) - 14 - 20 - 8)


def _build_udp_packet_exact(src_ip, dst_ip, sport, dport, target_frame):
    """UDP com len(Ether) == target_frame (Tot size no extrator)."""
    target_frame = max(0, int(target_frame))
    if target_frame == 0:
        return _build_udp_ip_packet(src_ip, dst_ip, sport, dport, 0)

    est = _udp_frame_payload_len(target_frame)
    lo = max(0, est - 32)
    hi = est + 32 + target_frame
    best_pl, best_diff = est, 10**9
    while lo <= hi:
        mid = (lo + hi) // 2
        measured = _measure_eth_frame(_build_udp_ip_packet(src_ip, dst_ip, sport, dport, mid))
        diff = measured - target_frame
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_pl = mid
        if measured < target_frame:
            lo = mid + 1
        elif measured > target_frame:
            hi = mid - 1
        else:
            best_pl = mid
            break
    return _build_udp_ip_packet(src_ip, dst_ip, sport, dport, best_pl)


def _resolve_udp_long_flow_cfg(params, wh, gre_flow):
    """Config fluxo UDP longo (Header_Length cumulativo estilo CIC)."""
    raw = (
        (gre_flow or {}).get("udp_long_flow")
        or (wh or {}).get("udp_long_flow")
        or (params.get("variants", {}).get("udpplain") or {}).get("gre_flow", {}).get("udp_long_flow")
        or {}
    )
    if not raw.get("enabled", False):
        return None
    target_hl = float(raw.get("header_length_median", 1189355.745))
    hdr_per_pkt = float(raw.get("header_bytes_per_packet", 28.0))
    agg_n = int((params.get("dynamic_streams") or {}).get("block_size", 20))
    default_warmup = max(0, int(round(target_hl / hdr_per_pkt - agg_n / 2.0)))
    return {
        "enabled": True,
        "warmup_packets": int(raw.get("warmup_packets", default_warmup)),
        "warmup_iat_sec": float(raw.get("warmup_iat_sec", 0.00005)),
        "warmup_size": int(raw.get("warmup_size", 554)),
        "header_length_median": target_hl,
    }


def _benign_device_ip(base_int, offset):
    """Gera IP de origem de um dispositivo a partir de um inteiro base (suporta milhares)."""
    v = base_int + offset
    return f"10.45.{(v >> 8) & 0xFF}.{v & 0xFF}"


def build_benign_legacy_schedule(params):
    """
    Gera trafego BENIGNO mMTC legado: dispositivos independentes com intervalos
    e payloads estocasticos globais (sem HMM por janela AGG).
    """
    rng = random.Random(params["seed"])
    np_rng = np.random.default_rng(params["seed"])

    b = params["benign"]
    n_dev = int(b.get("num_devices", 500))
    dst_ip = b.get("dst_ip", "10.45.0.1")
    sim_duration = float(b.get("sim_duration", 60.0))
    mean_interval = float(b.get("mean_interval", 1.0))
    interval_spread = float(b.get("interval_spread", 0.3))   # heterogeneidade entre devices
    jitter = float(b.get("jitter", 0.1))                     # jitter temporal intra-device
    payload_mean = float(b.get("payload_mean", 80.0))
    payload_std = float(b.get("payload_std", 20.0))
    payload_min = int(b.get("payload_min", 20))
    burst_p = float(b.get("burst_p", 0.02))
    burst_size = int(b.get("burst_size", 4))
    dport = int(b.get("dport", 5683))                        # CoAP por padrao
    base_int = int(b.get("src_base_int", 20))                # offset inicial no /16
    capture_overhead = params.get("capture_overhead", 14)
    udp_overhead = 28  # IP(20) + UDP(8)

    events = []
    for dev in range(n_dev):
        src_ip = _benign_device_ip(base_int, dev)
        sport = 1024 + (dev % 64000)
        # intervalo medio especifico do device (heterogeneidade mMTC)
        dev_interval = max(0.02, np_rng.normal(mean_interval, mean_interval * interval_spread))
        # fase inicial independente
        t = rng.uniform(0.0, dev_interval)
        while t < sim_duration:
            size = int(max(payload_min, np_rng.normal(payload_mean, payload_std)))
            payload_len = max(0, size - udp_overhead - capture_overhead)
            pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / UDP(sport=sport, dport=dport) / Raw(
                b"B" * payload_len
            )
            events.append((t, pkt))

            # pequeno burst ocasional (rajada curta do device)
            if rng.random() < burst_p:
                nb = rng.randint(1, max(1, burst_size))
                for _ in range(nb):
                    t += max(0.0005, dev_interval * 0.02)
                    if t >= sim_duration:
                        break
                    size = int(max(payload_min, np_rng.normal(payload_mean, payload_std)))
                    payload_len = max(0, size - udp_overhead - capture_overhead)
                    events.append((t, IP(src=src_ip, dst=dst_ip, ttl=64) /
                                   UDP(sport=sport, dport=dport) / Raw(b"B" * payload_len)))

            gap = dev_interval * (1.0 + np_rng.normal(0.0, jitter))
            t += max(0.001, gap)

    events.sort(key=lambda e: e[0])
    print(f"[benign] {n_dev} dispositivos -> {len(events)} pacotes (sim={sim_duration:.0f}s)")
    return events


def _benign_ds_window_from_emission(emission, ds_cfg):
    """Constrói dynamic_streams local a partir da emissao do cluster benigno."""
    out = copy.deepcopy(ds_cfg or {})
    wh = dict(out.get("window_hetero") or {})
    mg = dict(wh.get("markov_gmm") or {})
    mg["jitter_std"] = float(min(8.0, max(0.3, emission.get("size_std", mg.get("jitter_std", 5.0)))))
    mg["ar1_rho"] = float(emission.get("ar1_rho", mg.get("ar1_rho", 0.75)))
    wh["markov_gmm"] = mg
    out["window_hetero"] = wh
    out["size_std"] = float(emission.get("size_std", out.get("size_std", 30.0)))
    out["size_correlation"] = float(emission.get("size_correlation", out.get("size_correlation", 0.88)))
    return out


def _benign_emit_window_sizes(np_rng, emission, win_n):
    """
    Janela benigna: Min/Variance/AVG do cluster (CICIoT: Variance~1, std intra-janela baixo).
    """
    from benign_latent import sample_emission_quantile

    tmin = sample_emission_quantile(
        np_rng, emission.get("target_min"), default=float(emission.get("payload_floor", 66.0)),
    )
    tvar = sample_emission_quantile(
        np_rng, emission.get("target_variance"), default=0.9,
    )
    tavg = sample_emission_quantile(
        np_rng, emission.get("target_avg") or emission.get("AVG"),
        default=tmin + 20.0,
    )
    tgt_std = max(0.1, float(np.sqrt(max(1e-6, tvar))))
    return [
        max(42, int(round(tmin + np_rng.normal(0.0, tgt_std))))
        for _ in range(win_n)
    ]


def _adjust_benign_sizes_for_shap_targets(np_rng, sizes, emission, size_cfg):
    """Ajusta sizes[] para alinhar Min e Variance (SHAP LGBM) por janela."""
    from benign_latent import sample_emission_quantile

    target_min = sample_emission_quantile(
        np_rng,
        emission.get("target_min") or emission.get("Min"),
        default=float(emission.get("payload_floor", 66.0)),
    )
    target_var = sample_emission_quantile(
        np_rng, emission.get("target_variance") or emission.get("Variance"), default=-1.0,
    )
    if not sizes:
        return sizes, target_min, target_var
    arr = np.asarray(sizes, dtype=float)
    lo = float(emission.get("size_min", size_cfg.get("min", 42)))
    hi = float(emission.get("size_max", size_cfg.get("max", 1900)))
    if target_min > 0:
        cur_min = float(np.min(arr))
        arr = arr - cur_min + float(target_min)
    if target_var >= 0 and len(arr) >= 2:
        tgt_std = float(np.sqrt(max(1e-6, target_var)))
        mu = float(np.mean(arr))
        if tgt_std <= 1.5:
            arr = mu + np_rng.normal(0.0, tgt_std, size=len(arr))
        else:
            cur_std = float(np.std(arr, ddof=1))
            if cur_std > 1e-6:
                arr = mu + (arr - mu) * (tgt_std / cur_std)
        arr = np.maximum(arr, float(target_min))
    arr = np.clip(arr, lo, hi)
    return [max(int(lo), int(round(x))) for x in arr], target_min, target_var


def _append_benign_tcp_rst_companion(
    events, win_t, src_ip, dst_ip, n_rst, rng, iat_step=0.0005,
):
    """Fluxo TCP companheiro com flag RST (legado UDP+burst; preferir tcp_primary)."""
    if n_rst <= 0:
        return win_t
    sport = 30000 + (rng.randint(0, 30000))
    dport = 443
    t = win_t
    for _ in range(n_rst):
        pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
            sport=sport, dport=dport, flags="R",
        )
        events.append((t, pkt))
        t += iat_step
    return t


def _append_benign_tcp_window(
    events,
    win_t,
    src_ip,
    dst_ip,
    sport,
    dport,
    sizes,
    iats,
    capture_overhead,
    tcp_flags="P",
):
    """Emite janela AGG alinhada: exactamente len(sizes) pacotes TCP com payload (PSH, sem ACK)."""
    ip_tcp_hdr = 40
    t = win_t
    for pi, target_size in enumerate(sizes):
        payload_len = max(0, int(target_size) - ip_tcp_hdr - capture_overhead)
        cap = _live_mtu_max_payload(ip_tcp_hdr)
        if cap is not None:
            payload_len = min(payload_len, cap)
        pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
            sport=sport, dport=dport, flags=tcp_flags,
        ) / Raw(b"B" * payload_len)
        events.append((t, pkt))
        if pi < len(iats):
            t += max(0.0, float(iats[pi]))
    return t


def build_benign_latent_hmm_schedule(params):
    """
    Benigno mMTC com HMM vetorial aprendido (K-Means): emissao conjunta por janela
    AGG (sizes + flow_duration + IATs), manifest para calibracao futura.
    """
    rng = random.Random(params["seed"])
    np_rng = np.random.default_rng(params["seed"])
    b = params.get("benign") or {}
    latent = b.get("latent_hmm") or {}
    om_cfg = latent.get("operational_modes") or b.get("operational_modes") or {}
    if latent.get("model_path") and not om_cfg.get("modes"):
        from benign_latent import load_model
        mp = str(latent["model_path"])
        if not os.path.isabs(mp):
            for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
                cand = os.path.join(base, mp)
                if os.path.isfile(cand):
                    mp = cand
                    break
        if os.path.isfile(mp):
            fitted = load_model(mp)
            om_cfg = fitted.get("operational_modes") or om_cfg
    if not om_cfg.get("enabled"):
        raise ValueError("benign latent_hmm requer operational_modes.enabled=true")

    mode_chain = OperationalModeChain(om_cfg, rng)
    n_windows = int(latent.get("target_windows", b.get("target_windows", 5000)))
    agg_n = int(latent.get("aggregation_packets", b.get("aggregation_packets", 20)))
    window_gap = float(latent.get("inter_window_gap_sec", om_cfg.get("inter_window_gap_sec", 0.05)))
    n_dev = int(b.get("num_devices", 800))
    flow_devices = int(latent.get("flow_devices", b.get("flow_devices", 2)))
    flow_devices = max(1, min(flow_devices, n_dev))
    tcp_primary = bool(latent.get("tcp_primary", True))
    dst_ip = b.get("dst_ip", "10.45.0.1")
    tcp_dport = int(b.get("tcp_dport", latent.get("tcp_dport", 443)))
    base_int = int(b.get("src_base_int", 20))
    capture_overhead = int(params.get("capture_overhead", 14))
    flow_cfg = params.get("flow") or {}
    iat_cfg = params.get("iat") or {}
    size_cfg = params.get("size") or {}
    size_cfg = dict(size_cfg)
    benign_size = (b.get("_benign_size_override") or {})
    if benign_size:
        size_cfg.update(benign_size)
    else:
        size_cfg.setdefault("min", 42)
        size_cfg.setdefault("max", 1900)
    ds_cfg = params.get("dynamic_streams") or {}

    events = []
    window_manifest = []
    mode_counts = {}
    t = 0.0

    print(
        f"[benign_latent_hmm] a gerar {n_windows} janelas x {agg_n} pkts "
        f"(HMM K-Means, tcp_primary={tcp_primary}, flow_devices={flow_devices})...",
        flush=True,
    )

    for wi in range(n_windows):
        if wi > 0 and wi % 500 == 0:
            print(f"[benign_latent_hmm] janela {wi}/{n_windows} ({len(events)} pkts)...", flush=True)

        mode_name, emission = mode_chain.emit()
        mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

        from benign_latent import emit_benign_cluster_window

        win = emit_benign_cluster_window(np_rng, emission)
        sizes = win["sizes"]
        max_frame = int(b.get("_benign_live_max_frame", 0) or 0)
        if max_frame > 0:
            sizes = [min(int(s), max_frame) for s in sizes]
        if len(sizes) != agg_n:
            sizes = sizes[:agg_n] if len(sizes) > agg_n else sizes + [sizes[-1]] * (agg_n - len(sizes))
        win_iats = win["iats"]
        if len(win_iats) < max(0, len(sizes) - 1):
            win_iats = win_iats + [0.0] * (len(sizes) - 1 - len(win_iats))
        manifest_rst = float(win["rst_count"])
        n_rst = int(win["n_rst_packets"])

        window_manifest.append(
            build_window_manifest_record(
                mode_name, sizes, win_iats, window_id=wi,
                rst_count=manifest_rst, flow_duration=win["flow_duration"],
            ),
        )

        dev = wi % flow_devices
        src_ip = _benign_device_ip(base_int, dev)
        sport = 40000 + dev
        win_t = t
        if tcp_primary:
            tcp_flags = str(
                latent.get("tcp_flags")
                or b.get("tcp_flags")
                or "P",
            )
            win_t = _append_benign_tcp_window(
                events,
                win_t,
                src_ip,
                dst_ip,
                sport,
                int(emission.get("tcp_dport", tcp_dport)),
                sizes,
                win_iats,
                capture_overhead,
                tcp_flags=tcp_flags,
            )
        else:
            if emission.get("tcp_companion", False) and n_rst > 0:
                win_t = _append_benign_tcp_rst_companion(
                    events, win_t, src_ip, dst_ip, n_rst, rng,
                )
            udp_overhead = 28
            dport = int(b.get("dport", 5683))
            sport_udp = 1024 + (dev % 64000)
            for pi, target_size in enumerate(sizes):
                payload_len = max(0, int(target_size) - udp_overhead - capture_overhead)
                cap = _live_mtu_max_payload(udp_overhead)
                if cap is not None:
                    payload_len = min(payload_len, cap)
                pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / UDP(sport=sport_udp, dport=dport) / Raw(
                    b"B" * payload_len,
                )
                events.append((win_t, pkt))
                if pi < len(win_iats):
                    win_t += max(0.0, float(win_iats[pi]))

        t = win_t + window_gap
        mode_chain.step()

    params["_window_manifest"] = window_manifest
    events.sort(key=lambda e: e[0])
    modes_str = ",".join(f"{k}={v}" for k, v in sorted(mode_counts.items()))
    print(
        f"[benign_latent_hmm] {n_windows} janelas -> {len(events)} pacotes "
        f"| modos=[{modes_str}] | manifest={len(window_manifest)}",
    )
    return events


def build_benign_schedule(params):
    """
    Dispatcher benigno: HMM latente (K-Means) se latent_hmm.enabled, senao legado.
    """
    b = params.get("benign") or {}
    latent = b.get("latent_hmm") or {}
    om = latent.get("operational_modes") or b.get("operational_modes") or {}
    if latent.get("enabled") or om.get("enabled"):
        return build_benign_latent_hmm_schedule(params)
    return build_benign_legacy_schedule(params)




def build_gre_multiflow_schedule(params):
    """Fallback: delega para hetero se disponivel."""
    return build_gre_window_hetero_schedule(params)


def build_schedule(params):
    """
    Constroi o cronograma completo: lista de (offset_t, pacote) ordenada por tempo.

    Efeitos colaterais (_window_manifest, etc.) sao copiados de volta para o dict
    original passado pelo caller (ex.: auto_calibrate.evaluate).
    """
    caller = params
    params = materialize_variant_params(params, params.get("variant"))
    if params.get("variant") == "benign":
        events = build_benign_schedule(params)
    elif params.get("variant") in ("greip", "greeth"):
        gre_flow = params.get("gre_flow", {})
        if gre_flow.get("enabled"):
            size_model = str(params.get("size", {}).get("model", "")).lower()
            if size_model in ("hybrid_window_hetero", "window_hetero"):
                events = build_gre_window_hetero_schedule(params)
            else:
                events = build_gre_multiflow_schedule(params)
        else:
            events = _build_legacy_botnet_schedule(params)
    elif params.get("variant") == "udpplain":
        gre_flow = params.get("gre_flow", {})
        if gre_flow.get("enabled", True):
            events = build_gre_window_hetero_schedule(params)
        else:
            events = _build_legacy_botnet_schedule(params)
    else:
        events = _build_legacy_botnet_schedule(params)
    if isinstance(caller, dict):
        for key in ("_window_manifest",):
            if key in params:
                caller[key] = params[key]
    return events


def export_schedule_manifest(params, out_path, *, meta=None):
    """
    Grava manifesto por janela AGG (IAT/fd pretendidos no inject live).

    Usar com a mesma seed do inject para correlacionar com captura ogstun.
    """
    manifest = params.get("_window_manifest") or []
    if not manifest:
        raise ValueError("manifesto vazio: gere o cronograma com build_schedule primeiro")

    agg_n = _aggregation_window_size(
        params.get("size", {}),
        params.get("dynamic_streams") or {},
        params.get("gre_flow") or {},
    )
    flow_cfg = params.get("flow", {})
    fds = [float(w.get("flow_duration", 0.0)) for w in manifest]
    zero_frac = float(sum(1 for x in fds if x <= 1e-6) / max(1, len(fds)))

    payload = {
        "variant": params.get("variant"),
        "seed": params.get("seed"),
        "agg_n": agg_n,
        "semantic_pcap_iats": _uses_semantic_pcap_iats(
            params.get("variant", ""), flow_cfg, False,
        ),
        "duration_zero_fraction": float(flow_cfg.get("duration_zero_fraction", 0) or 0),
        "summary": {
            "n_windows": len(manifest),
            "fd_zero_fraction": zero_frac,
            "fd_median": float(np.median(fds)) if fds else 0.0,
            "fd_p90": float(np.percentile(fds, 90)) if fds else 0.0,
        },
        "windows": manifest,
    }
    if meta:
        payload["meta"] = dict(meta)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[schedule] manifesto {len(manifest)} janelas -> {out_path}")
    print(
        f"[schedule] fd pretendido: zero={zero_frac:.1%} med={payload['summary']['fd_median']:.6f}s",
        flush=True,
    )
    return payload


def _build_legacy_botnet_schedule(params):
    rng = random.Random(params["seed"])
    np_rng = np.random.default_rng(params["seed"])
    variant = params["variant"]
    bot = params["botnet"]
    dst_ip = bot["dst_ip"]
    src_ips = [_expand_ipv4(bot["src_base"], bot["src_first_octet"] + i)
               for i in range(bot["size"])]
    inter_flow = params.get("inter_flow_delay", 0.3)
    events = []
    flow_start = 0.0
    for idx, src_ip in enumerate(src_ips):
        n_pkts = sample_flow_packet_count(np_rng, params["flow"])
        duration = sample_flow_duration(rng, params["flow"], np_rng)
        iats = sample_iat_sequence(rng, np_rng, params["iat"], n_pkts, duration)
        if variant == "udpplain":
            sport = 1024 + idx
            dport = rng.randint(1024, 65535)
        else:
            sport = dport = 0
        t = flow_start
        for k in range(n_pkts):
            pkt = build_packet(rng, np_rng, params, src_ip, dst_ip, sport, dport)
            events.append((t, pkt))
            if k < len(iats):
                t += iats[k]
        flow_start = t + inter_flow
    events.sort(key=lambda e: e[0])
    return events


# --------------------------------------------------------------------------- #
# Saida: PCAP                                                                  #
# --------------------------------------------------------------------------- #
def write_pcap(events, out_path):
    """Escreve um .pcap em nivel IP (como captura do ogstun) com timestamps."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base_ts = time.time()
    pkts = []
    for t_off, pkt in events:
        pkt.time = base_ts + t_off
        pkts.append(pkt)
    wrpcap(out_path, pkts)
    print(f"[pcap] {len(pkts)} pacotes -> {out_path}  "
          f"(span={events[-1][0]-events[0][0]:.2f}s)")


# --------------------------------------------------------------------------- #
# Saida: LIVE (pacing preciso)                                                 #
# --------------------------------------------------------------------------- #
def _live_min_send_interval_sec(*, max_pps=None, min_pkt_interval_sec=None):
    """
    Intervalo minimo entre envios consecutivos no live.

    Rajadas zero-inflated (IAT=0) disparam dezenas de pacotes no mesmo
    timestamp; o UPF/ogstun perde ~10% sob picos >> 1 kpps. O cap so afecta
    a entrega live — o export-schedule mantem IAT semanticos.

    Env: SYNTH_LIVE_MAX_PPS, SYNTH_LIVE_MIN_PKT_INTERVAL_SEC
    """
    if max_pps is not None and float(max_pps) > 0:
        return 1.0 / float(max_pps)
    if min_pkt_interval_sec is not None and float(min_pkt_interval_sec) > 0:
        return float(min_pkt_interval_sec)
    try:
        env_pps = float(os.environ.get("SYNTH_LIVE_MAX_PPS", "0") or "0")
    except ValueError:
        env_pps = 0.0
    if env_pps > 0:
        return 1.0 / env_pps
    try:
        env_dt = float(os.environ.get("SYNTH_LIVE_MIN_PKT_INTERVAL_SEC", "0") or "0")
    except ValueError:
        env_dt = 0.0
    return max(0.0, env_dt)


def send_live(events, iface=None, *, max_pps=None, min_pkt_interval_sec=None):
    """
    Injeta os pacotes ao vivo com agendamento de alta precisao.

    Otimizacoes vs. send() por pacote:
      - serializa os pacotes para bytes UMA vez (sem rebuild do scapy no loop);
      - reaproveita UM unico socket L3;
      - agendador hibrido: sleep ate ~1.5ms antes, depois busy-wait.
    """
    if iface:
        conf.iface = iface
    sock = conf.L3socket(iface=iface) if iface else conf.L3socket()

    mtu = _effective_live_mtu(iface)
    imtu = _read_iface_mtu(iface)
    # pre-serializa (com clamp MTU — evita EMSGSIZE no uesimtun0)
    raw = []
    clamped = 0
    for t_off, pkt in events:
        data = bytes(pkt)
        if len(data) > mtu:
            data = _clamp_ip_wire_to_mtu(data, mtu)
            clamped += 1
        raw.append((t_off, data))
    min_interval = _live_min_send_interval_sec(
        max_pps=max_pps, min_pkt_interval_sec=min_pkt_interval_sec,
    )

    print(f"[live] Injetando {len(raw)} pacotes em '{iface or conf.iface}'...")
    if imtu:
        print(f"[live] MTU iface={imtu} efectivo={mtu} (env SYNTH_LIVE_MTU)", flush=True)
    if clamped:
        print(
            f"[live] MTU={mtu}: {clamped} pacotes reduzidos (benign alto_payload / EMSGSIZE)",
            flush=True,
        )
    if min_interval > 0:
        cap_pps = 1.0 / min_interval
        print(
            f"[live] burst cap: intervalo min {min_interval * 1e3:.3f} ms "
            f"(~{cap_pps:.0f} pps max entre envios; schedule JSON inalterado)",
            flush=True,
        )
    t0 = time.perf_counter()
    sent = 0
    last_send_at = 0.0
    try:
        for t_off, data in raw:
            target = t0 + t_off
            if min_interval > 0 and sent > 0:
                target = max(target, last_send_at + min_interval)
            while True:
                now = time.perf_counter()
                remaining = target - now
                if remaining <= 0:
                    break
                if remaining > 0.0015:
                    time.sleep(remaining - 0.0010)
                # else: busy-wait ate o instante exato
            if len(data) > mtu:
                data = _clamp_ip_wire_to_mtu(data, mtu)
            _l3_send_ip_bytes(sock, data, iface=iface)
            last_send_at = time.perf_counter()
            sent += 1
            if sent % 5000 == 0:
                print(f"  [live] {sent}/{len(raw)} enviados...")
    finally:
        sock.close()
    elapsed = time.perf_counter() - t0
    pps = sent / elapsed if elapsed > 0 else 0.0
    print(f"[live] Concluido: {sent} pacotes em {elapsed:.2f}s ({pps:.0f} pps efetivos).")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Gerador de trafego sintetico Mirai (Open5GS/UERANSIM).")
    ap.add_argument("--params", default="params.json", help="Arquivo JSON de parametros.")
    ap.add_argument("--mode", choices=["pcap", "live"], default="pcap")
    ap.add_argument("--out", default="pcap_origin/capture.pcap", help="(modo pcap) arquivo de saida.")
    ap.add_argument("--iface", default=None, help="(modo live) interface de envio (ex: uesimtun0).")
    ap.add_argument("--variant", default=None, help="Sobrescreve params.variant (greip|greeth|udpplain|benign).")
    ap.add_argument("--seed", type=int, default=None, help="Sobrescreve params.seed.")
    args = ap.parse_args()

    params = load_params(args.params)
    if args.variant:
        params["variant"] = args.variant
    if args.seed is not None:
        params["seed"] = args.seed

    if params.get("variant") == "benign":
        ndev = params.get("benign", {}).get("num_devices", "-")
        print(f"[gen] variante=benign | devices={ndev} | seed={params['seed']}")
    else:
        bots = params.get("botnet", {}).get("size", "-")
        print(f"[gen] variante={params['variant']} | bots={bots} | seed={params['seed']}")
    events = build_schedule(params)
    if not events:
        print("[gen] Nenhum evento gerado. Verifique os parametros.")
        return

    if args.mode == "pcap":
        write_pcap(events, args.out)
    else:
        send_live(events, iface=args.iface)


if __name__ == "__main__":
    main()
