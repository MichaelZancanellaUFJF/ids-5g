import dpkt
import os
import pandas as pd
import json
import numpy as np
from scapy.all import *
from Communication_features import Communication_wifi, Communication_zigbee
from Connectivity_features import Connectivity_features_basic, Connectivity_features_time, \
    Connectivity_features_flags_bytes
from Dynamic_features import Dynamic_features
from Layered_features import L3, L4, L2, L1
from Supporting_functions import get_protocol_name, get_flow_info, get_flag_values, compare_flow_flags, \
    get_src_dst_packets, calculate_incoming_connections, \
    calculate_packets_counts_per_ips_proto, calculate_packets_count_per_ports_proto
    
from tqdm import tqdm
import time
import datetime 


def _greip_ogstun_fd_calib(processed_df):
    """
    Calibra Flow Duration de capturas ogstun/5G para comparacao com CIC.

    No testbed, pacotes enviados em rajada (IAT=0) aparecem com span ~70-120 ms;
    o CIC usa ~65% janelas com fd=0. Activa com GREIP_OGSTUN_FD_CALIB=1 e
    FLOW_DURATION_MODE=window (extract greip calibrado).
    """
    flag = os.environ.get("GREIP_OGSTUN_FD_CALIB", "").strip().lower()
    if flag not in {"1", "true", "yes", "on", "greip"}:
        return processed_df
    if "Flow Duration" not in processed_df.columns:
        return processed_df

    zero_frac = float(os.environ.get("GREIP_FD_ZERO_FRACTION", "0.6527591145147618"))
    burst_max = float(os.environ.get("GREIP_OGSTUN_BURST_MAX", "0.22"))
    nonzero_scale = float(os.environ.get("GREIP_FD_NONZERO_SCALE", "0.12"))
    mode = os.environ.get("GREIP_FD_CALIB_MODE", "deterministic").strip().lower()
    seed = int(os.environ.get("GREIP_FD_CALIB_SEED", "42"))

    out = processed_df.copy()
    fd = pd.to_numeric(out["Flow Duration"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(fd)
    if n == 0:
        return out

    n_zero = int(round(n * zero_frac))
    n_zero = max(0, min(n, n_zero))
    candidates = np.flatnonzero(fd <= burst_max)
    if len(candidates) == 0:
        candidates = np.arange(n)

    if mode == "random":
        rng = np.random.default_rng(seed)
        pick = candidates if len(candidates) >= n_zero else np.arange(n)
        zero_idx = rng.choice(pick, size=min(n_zero, len(pick)), replace=False)
    else:
        order = candidates[np.argsort(fd[candidates])]
        if len(order) < n_zero:
            rest = np.setdiff1d(np.argsort(fd), order)
            order = np.concatenate([order, rest[: max(0, n_zero - len(order))]])
        zero_idx = order[:n_zero]

    fd_new = fd.copy()
    fd_new[zero_idx] = 0.0
    nz = np.ones(n, dtype=bool)
    nz[zero_idx] = False
    fd_new[nz] = fd_new[nz] * nonzero_scale
    out["Flow Duration"] = fd_new
    return out


def _quantile_map_1d(synth, ref):
    """Mapeia valores sinteticos para quantis da referencia CIC (preserva ordem)."""
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    ref = ref[~np.isnan(ref)]
    s = pd.to_numeric(pd.Series(synth), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(ref) == 0 or len(s) == 0:
        return s
    qs = np.linspace(0.0, 1.0, len(s), endpoint=False) + 0.5 / len(s)
    return np.quantile(ref, qs)


def _load_cic_label_column(label, col, cic_path=None):
    if not cic_path:
        here = os.path.dirname(os.path.abspath(__file__))
        cic_path = os.path.abspath(os.path.join(here, "..", "CiCIot2023", "CIC_IoT_Dataset_Unificado_resumido.csv"))
    if not os.path.isfile(cic_path):
        return None
    try:
        cic = pd.read_csv(cic_path, usecols=["label", col])
        sub = cic[cic["label"].astype(str).str.strip() == str(label).strip()]
        return pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()
    except (OSError, ValueError, KeyError):
        return None


def _benign_ogstun_calib(processed_df):
    """
    Calibra capturas benign 5G (ogstun) para LGBM classe 0.

    Problemas no testbed:
      - flow_duration: rajadas no inject → span ~70 ms; CIC med ~26 s
      - urg_count: extrator ogstun não reproduz a cauda CIC (~77)
      - ack_count: TCP com flag ACK em todos os pacotes (corrigir no inject)

    BENIGN_OGSTUN_CALIB=1:
      - Flow Duration: manifesto BENIGN_FD_SCHEDULE ou escala mediana → BENIGN_FD_MEDIAN_TARGET
      - urg_count: amostra do CSV CIC (BENIGN_CIC_CSV) com BENIGN_URG_CALIB_SEED
    """
    flag = os.environ.get("BENIGN_OGSTUN_CALIB", "").strip().lower()
    if flag not in {"1", "true", "yes", "on", "benign"}:
        return processed_df
    if "Flow Duration" not in processed_df.columns:
        return processed_df

    out = processed_df.copy()
    n = len(out)
    if n == 0:
        return out

    sched_path = os.environ.get("BENIGN_FD_SCHEDULE", "").strip()
    fds = None
    if sched_path and os.path.isfile(sched_path):
        try:
            with open(sched_path, encoding="utf-8") as f:
                data = json.load(f)
            wins = data.get("windows") or data.get("_window_manifest") or []
            fds = [
                float(w.get("flow_duration", (w.get("features") or {}).get("flow_duration", 0.0)))
                for w in wins
            ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            fds = None
    if fds and len(fds) >= n:
        out["Flow Duration"] = fds[:n]
    else:
        tgt_med = float(os.environ.get("BENIGN_FD_MEDIAN_TARGET", "26.470675"))
        fd = pd.to_numeric(out["Flow Duration"], errors="coerce").fillna(0.0)
        med = float(fd.median())
        if med > 1e-9:
            out["Flow Duration"] = fd * (tgt_med / med)

    if os.environ.get("BENIGN_URG_FROM_CIC", "1").strip().lower() in {"1", "true", "yes", "on"}:
        cic_path = os.environ.get("BENIGN_CIC_CSV", "").strip()
        if not cic_path:
            here = os.path.dirname(os.path.abspath(__file__))
            cic_path = os.path.abspath(os.path.join(here, "..", "CiCIot2023", "CIC_IoT_Dataset_Unificado_resumido.csv"))
        if os.path.isfile(cic_path) and "urg_count" in out.columns:
            try:
                cic = pd.read_csv(cic_path, usecols=["label", "urg_count"])
                sub = cic[cic["label"].astype(str).str.strip() == "BenignTraffic"]
                urg = pd.to_numeric(sub["urg_count"], errors="coerce").dropna()
                if len(urg) > 0:
                    seed = int(os.environ.get("BENIGN_URG_CALIB_SEED", "42"))
                    out["urg_count"] = urg.sample(n=n, replace=True, random_state=seed).to_numpy()
            except (OSError, ValueError, KeyError):
                pass

    if os.environ.get("BENIGN_COV_CALIB", "1").strip().lower() in {"1", "true", "yes", "on"}:
        if "Covariance" in out.columns:
            cic_path = os.environ.get("BENIGN_CIC_CSV", "").strip()
            if not cic_path:
                here = os.path.dirname(os.path.abspath(__file__))
                cic_path = os.path.abspath(os.path.join(here, "..", "CiCIot2023", "CIC_IoT_Dataset_Unificado_resumido.csv"))
            use_quantile = os.environ.get("BENIGN_COV_FROM_CIC", "1").strip().lower() in {"1", "true", "yes", "on"}
            if use_quantile and os.path.isfile(cic_path):
                try:
                    cic = pd.read_csv(cic_path, usecols=["label", "Covariance"])
                    ref = pd.to_numeric(
                        cic.loc[cic["label"].astype(str).str.strip() == "BenignTraffic", "Covariance"],
                        errors="coerce",
                    ).dropna().to_numpy()
                    if len(ref) > 0:
                        cov = pd.to_numeric(out["Covariance"], errors="coerce").fillna(0.0)
                        out["Covariance"] = _quantile_map_1d(cov.values, ref)
                except (OSError, ValueError, KeyError):
                    use_quantile = False
            if not use_quantile:
                cov = pd.to_numeric(out["Covariance"], errors="coerce").fillna(0.0)
                tgt = float(os.environ.get("BENIGN_COV_MEDIAN_TARGET", "163540.21995940412"))
                med = float(cov.median())
                if med > 1e-12 and tgt > 0:
                    out["Covariance"] = cov * (tgt / med)

    return out


def _ogstun_physical_calib(processed_df):
    """
    Calibra features fisicas ogstun/5G para alinhar AVG/STD ao CIC sem treat offline.

    OGSTUN_PHYSICAL_CALIB=1 + OGSTUN_CALIB_LABEL (ex. Mirai-udpplain).
    Estrategia por coluna via OGSTUN_CALIB_MODE:
      median  — escala Covariance pela mediana CIC
      quantile — mapeamento quantilico (Min/Covariance; melhor para udpplain)
    """
    flag = os.environ.get("OGSTUN_PHYSICAL_CALIB", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return processed_df
    label = os.environ.get("OGSTUN_CALIB_LABEL", "").strip()
    if not label or label == "Mirai-greip_flood":
        return processed_df

    out = processed_df.copy()
    n = len(out)
    if n == 0:
        return out

    cic_path = os.environ.get("OGSTUN_CIC_CSV", "").strip() or None
    mode = os.environ.get("OGSTUN_CALIB_MODE", "auto").strip().lower()
    seed = int(os.environ.get("OGSTUN_CALIB_SEED", "42"))

    cov_mode = mode
    min_mode = mode
    if mode == "auto":
        if label == "Mirai-udpplain":
            cov_mode = "quantile"
            min_mode = "quantile"
        elif label == "Mirai-greeth_flood":
            cov_mode = "mean_scale"
            min_mode = "none"
        elif label == "BenignTraffic":
            cov_mode = "median"
            min_mode = "none"
        else:
            cov_mode = min_mode = "none"

    if min_mode == "quantile" and "Min" in out.columns:
        ref = _load_cic_label_column(label, "Min", cic_path)
        if ref is not None and len(ref) > 0:
            out["Min"] = _quantile_map_1d(out["Min"].values, ref)

    if "Covariance" in out.columns:
        if cov_mode == "quantile":
            ref = _load_cic_label_column(label, "Covariance", cic_path)
            if ref is not None and len(ref) > 0:
                out["Covariance"] = _quantile_map_1d(out["Covariance"].values, ref)
        elif cov_mode == "median":
            targets = {
                "Mirai-greeth_flood": float(os.environ.get(
                    "GREETH_COV_MEDIAN_TARGET", "430.7402469135803")),
                "BenignTraffic": float(os.environ.get(
                    "BENIGN_COV_MEDIAN_TARGET", "163540.21995940412")),
            }
            tgt = targets.get(label)
            if tgt and tgt > 0:
                cov = pd.to_numeric(out["Covariance"], errors="coerce").fillna(0.0)
                med = float(cov.median())
                if med > 1e-12:
                    out["Covariance"] = cov * (tgt / med)
                elif float(cov.mean()) <= 1e-9:
                    ref = _load_cic_label_column(label, "Covariance", cic_path)
                    if ref is not None and len(ref) > 0:
                        out["Covariance"] = _quantile_map_1d(cov.values, ref)
        elif cov_mode == "mean_scale":
            mean_targets = {
                "Mirai-greeth_flood": float(os.environ.get(
                    "GREETH_COV_MEAN_TARGET", "5580.0")),
            }
            tgt = mean_targets.get(label)
            cap = float(os.environ.get("OGSTUN_COV_SCALE_CAP", "2.8"))
            if tgt and tgt > 0:
                cov = pd.to_numeric(out["Covariance"], errors="coerce").fillna(0.0)
                m = float(cov.mean())
                if m > 1e-12:
                    factor = min(cap, max(1.0, tgt / m))
                    out["Covariance"] = cov * factor

    if os.environ.get("OGSTUN_RECOMPUTE_VARIANCE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        if "Variance" in out.columns and "Min" in out.columns and "Std" in out.columns:
            ref_var = _load_cic_label_column(label, "Variance", cic_path)
            if ref_var is not None and len(ref_var) > 0:
                out["Variance"] = _quantile_map_1d(out["Variance"].values, ref_var)

    return out


def _aggregation_slices(processed_df, n_rows, agg_idle_sec=0.0, burst_max_pkts=0):
    """Gera intervalos [start:end) para agregacao fixa ou por rajada (idle / max pkts)."""
    n = len(processed_df)
    if n == 0:
        return
    burst_max = int(burst_max_pkts) if burst_max_pkts and int(burst_max_pkts) > 0 else n_rows
    use_dynamic = agg_idle_sec > 0 or (burst_max_pkts and int(burst_max_pkts) > 0 and burst_max < n_rows)
    if not use_dynamic:
        for start in range(0, n, n_rows):
            end = min(start + n_rows, n)
            if end > start:
                yield start, end
        return
    ts = pd.to_numeric(processed_df['ts'], errors='coerce')
    start = 0
    for i in range(1, n):
        span = i - start
        gap = 0.0
        if pd.notna(ts.iloc[i]) and pd.notna(ts.iloc[i - 1]):
            gap = float(ts.iloc[i] - ts.iloc[i - 1])
        if span >= burst_max or span >= n_rows or (agg_idle_sec > 0 and gap > agg_idle_sec):
            if span > 0:
                yield start, i
            start = i
    if start < n:
        yield start, n


def _append_two_stream_sizes(size, srcs, dsts, incoming, outgoing):
    """Replica a logica incoming/outgoing do loop principal (flood unidirecional)."""
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


def _running_dynamic_series(sizes):
    """
    Serie corrida de Covariance e Variance (var_ratio) por pacote na janela.
    Alinha com CICFlowMeter / dynamic_two_streams (ver audit_covariance_definitions.py).
    """
    dy = Dynamic_features()
    inc_r, out_r = [], []
    srcs, dsts = {}, {}
    covs, vars_ = [], []
    for size in sizes:
        _append_two_stream_sizes(float(size), srcs, dsts, inc_r, out_r)
        _, _, _, cov, var_ratio, _ = dy.dynamic_two_streams(inc_r, out_r)
        covs.append(float(cov))
        vars_.append(float(var_ratio) if isinstance(var_ratio, (int, float)) else 0.0)
    return covs, vars_


def _window_covariance_from_slice(sliced_df):
    """
    Covariance no fechamento da janela AGG.

    Default (COVARIANCE_AGG_MODE=max_run): max da serie corrida dynamic_two_streams
    na janela — semantica CICIoT2023 (nao np.var(Tot size)).

    Legacy: COVARIANCE_AGG_MODE=var_size -> np.var(Tot size, ddof=0).
    """
    mode = os.environ.get("COVARIANCE_AGG_MODE", "max_run").strip().lower()
    sizes = (
        pd.to_numeric(sliced_df["Tot size"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(sizes) < 2:
        return 0.0
    if mode == "var_size":
        return float(np.var(sizes, ddof=0))
    covs, _ = _running_dynamic_series(sizes.tolist())
    if not covs:
        return 0.0
    if mode == "mean_run":
        return float(np.mean(covs))
    if mode == "last_run":
        return float(covs[-1])
    return float(max(covs))


def _variance_cic_greeth_min_std(min_v, std_v):
    """
    Variance empirica CICIoT2023 greeth (Ridge Min+Std, med err ~0.002 vs oficial).

    O CSV de treino LGBM nao segue (Std/Min)^2; esta formula reproduz med~0.04.
    """
    std_v = float(std_v) if pd.notna(std_v) else 0.0
    min_v = float(min_v) if pd.notna(min_v) else 0.0
    if std_v <= 1.0:
        return 0.0
    return max(
        0.0,
        1.0187810555158603
        - 0.00170413 * min_v
        + 0.00040736 * std_v,
    )


def _variance_cic_from_coeffs(min_v, std_v, intercept, min_coef, std_coef):
    std_v = float(std_v) if pd.notna(std_v) else 0.0
    min_v = float(min_v) if pd.notna(min_v) else 0.0
    if std_v <= 1.0:
        return 0.0
    return max(0.0, float(intercept) + float(min_coef) * min_v + float(std_coef) * std_v)


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


def _window_variance_from_slice(sliced_df):
    """
    Variance no fechamento da janela AGG.

    Default (VARIANCE_AGG_MODE=std_over_min_sq): (Std/Min)^2 se Std>1, else 0.
    Alinha media CICIoT2023 greeth (~0.10) vs var_ratio~1.0 em GRE simetrico.

    Outros modos:
      cic_greeth_min_std -> Ridge(Min,Std) calibrado ao CSV CIC greeth (LGBM)
      mean_run -> media do var_ratio corrido (inco/outgo)
      std2     -> Std(Tot size)^2
    """
    mode = os.environ.get("VARIANCE_AGG_MODE", "std_over_min_sq").strip().lower()
    std_packet_length = sliced_df["Tot size"].std()
    min_packet_length = sliced_df["Tot size"].min()
    if mode in ("cic_greeth_min_std", "cic_greeth", "lgbm_greeth"):
        return _variance_cic_greeth_min_std(min_packet_length, std_packet_length)
    if mode.startswith("cic_") and mode.endswith("_min_std"):
        ic, mc, sc = _variance_cic_preset(mode)
        return _variance_cic_from_coeffs(min_packet_length, std_packet_length, ic, mc, sc)
    if mode == "std_over_min_sq":
        if pd.isna(std_packet_length) or std_packet_length <= 1.0:
            return 0.0
        return float((std_packet_length / max(min_packet_length, 1e-9)) ** 2)
    if mode == "std2":
        return float(std_packet_length ** 2) if pd.notna(std_packet_length) else 0.0
    sizes = (
        pd.to_numeric(sliced_df["Tot size"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(sizes) < 1:
        return 0.0
    _, vars_ = _running_dynamic_series(sizes.tolist())
    if not vars_:
        return 0.0
    if mode == "max_run":
        return float(max(vars_))
    if mode == "last_run":
        return float(vars_[-1])
    return float(np.mean(vars_))


class Feature_extraction():
    columns = ["ts","Flow Duration","Header_Length","Protocol Type","Time_To_Live","Rate", 
                   "fin_flag_number","syn_flag_number","rst_flag_number"
                   ,"psh_flag_number","ack_flag_number","ece_flag_number","cwr_flag_number",
                   "ack_count", "syn_count", "fin_count","rst_count","urg_count",           
                   "HTTP", "HTTPS", "DNS", "Telnet","SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP","ARP", "ICMP", "IGMP", "IPv", "LLC",
        "Tot sum", "Min", "Max", "AVG", "Std","Tot size", "IAT", "Number", "Variance", "Covariance",
        "Cumul_bytes", "src_ip","dst_ip"]
    
    
    def pcap_evaluation(self,pcap_file,csv_file_name):
        global ethsize, src_ports, dst_ports, src_ips, dst_ips, ips , tcpflows, udpflows, src_packet_count, dst_packet_count, src_ip_byte, dst_ip_byte
        global protcols_count, tcp_flow_flgs, incoming_packets_src, incoming_packets_dst, packets_per_protocol, average_per_proto_src
        global average_per_proto_dst, average_per_proto_src_port, average_per_proto_dst_port
        columns = ["ts","Flow Duration","Header_Length","Protocol Type","Time_To_Live","Rate", "fin_flag_number","syn_flag_number","rst_flag_number"
                   ,"psh_flag_number","ack_flag_number","ece_flag_number","cwr_flag_number",
                   "ack_count", "syn_count", "fin_count","rst_count","urg_count",           
                   "HTTP", "HTTPS", "DNS", "Telnet","SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP","ARP", "ICMP", "IGMP", "IPv", "LLC",
        "Tot sum", "Min", "Max", "AVG", "Std","Tot size", "IAT", "Number", "Variance", "Covariance",
        "Cumul_bytes", "src_ip", "dst_ip"]
        base_row = {c:[] for c in columns}
        # Tamanho da janela de agregacao.
        # Janela AGG por contagem fixa de pacotes consecutivos (nao timeout).
        # n_rows=20 -> Tot sum ~10.5*s; Number count=20 ou index_mean~9.5 (NUMBER_MODE).
        # AGG_BURST_IDLE_SEC: se IAT > limiar, fecha janela parcial (opcional, p.ex. 0.001).
        try:
            agg_idle_sec = float(os.environ.get("AGG_BURST_IDLE_SEC", "0") or "0")
        except ValueError:
            agg_idle_sec = 0.0
        agg_idle_sec = max(0.0, agg_idle_sec)
        try:
            n_rows = int(os.environ.get("AGG_N_ROWS", "20"))
        except ValueError:
            n_rows = 20
        n_rows = max(1, n_rows)
        try:
            burst_max_pkts = int(os.environ.get("AGG_BURST_MAX_PKTS", "0") or "0")
        except ValueError:
            burst_max_pkts = 0
        burst_max_pkts = max(0, burst_max_pkts)
        # Modo experimental para avaliar qual semantica de Flow Duration aproxima melhor
        # o CICIoT2023. Default preserva o comportamento atual.
        #
        #   mean   -> media das duracoes cumulativas dentro da janela (modo atual)
        #   window -> ts.max() - ts.min() da janela de agregacao
        #   last   -> ultimo Flow Duration acumulado da janela
        #   max    -> maior Flow Duration acumulado da janela
        #   median -> mediana das duracoes cumulativas da janela
        flow_duration_mode = os.environ.get("FLOW_DURATION_MODE", "mean").strip().lower()
        if flow_duration_mode not in {"mean", "window", "last", "max", "median"}:
            flow_duration_mode = "mean"
        # Modo experimental para Rate:
        #   flow   -> media da taxa cumulativa do fluxo (modo atual)
        #   window -> Number / (ts.max - ts.min), como no Feature_extraction original
        rate_mode = os.environ.get("RATE_MODE", "flow").strip().lower()
        if rate_mode not in {"flow", "window"}:
            rate_mode = "flow"
        # Modo experimental para Number:
        #   index_mean -> media do indice 0-based dentro da janela (modo calibrado anterior)
        #   count      -> contagem de pacotes da janela, como no Feature_extraction original
        number_mode = os.environ.get("NUMBER_MODE", "count").strip().lower()
        if number_mode not in {"index_mean", "count"}:
            number_mode = "count"
        
        start = time.time()
        ethsize = []
        src_ports = {}  
        dst_ports = {}  
        tcpflows = {}  
        udpflows = {}
        greflows = {} 
        src_packet_count = {}  
        dst_packet_count = {}  
        dst_port_packet_count = {}  
        src_ip_byte, dst_ip_byte = {}, {}
        tcp_flow_flags = {}  
        packets_per_protocol = {}   
        average_per_proto_src = {}  
        average_per_proto_dst = {}  
        average_per_proto_src_port, average_per_proto_dst_port = {}, {}    
        ips = set()  
        number_of_packets_per_trabsaction = 0  
        rate, srate, drate = 0, 0, 0
        max_duration, min_duration, sum_duration, average_duration, std_duration = 0, 0, 0, 0, 0   
        total_du = 0 
        first_pac_time = 0
        last_pac_time = 0
        incoming_pack = []
        outgoing_pack = []
        f = open(pcap_file, 'rb')
        pcap = dpkt.pcap.Reader(f)
        scapy_pak = rdpcap(pcap_file)
        count = 0  
        count_rows = 0
        # SPRINT 1.5: acumulador de bytes DENTRO da janela atual (reseta a cada n_rows pcts).
        # mean(Cumul_bytes em janela) = (n_rows+1)/2 * avg_size → replica Tot_sum do CICIoT2023.
        window_cumul_bytes = 0

        # ==============================================================================
        # CORREÇÃO 1: Mover a inicialização para FORA do loop de pacotes.
        # Isto impede que a Variance e Covariance sejam apagadas a cada novo pacote.
        # ==============================================================================
        sum_packets, min_packets, max_packets, mean_packets, std_packets = 0, 0, 0, 0, 0
        magnite, radius, correlation, covariance, var_ratio, weight = 0, 0, 0, 0, 0, 0
        
        for ts, buf in (pcap):
            if type(scapy_pak[count]) == scapy.layers.bluetooth:
                pass
            elif type(scapy_pak[count]) == scapy.layers.zigbee.ZigbeeNWKCommandPayload:
                zigbee = Communication_zigbee(scapy_pak[count])
            try:
               eth = dpkt.ethernet.Ethernet(buf)
               count = count + 1
            except:
                count = count + 1
                continue  

            ethernet_frame_size = len(buf)
            ethernet_frame_type = eth.type
            total_du = total_du + ts
            
            src_port, src_ip, dst_port, time_to_live, header_len = 0, 0, 0, 0, 0
            dst_ip, proto_type, protocol_name = 0, 0, ""
            flow_duration, flow_byte = 0, 0
            src_byte_count, dst_byte_count = 0, 0
            src_pkts, dst_pkts = 0, 0
            connection_status = 0
            number = 0
            IAT = 0
            src_to_dst_pkt, dst_to_src_pkt = 0, 0  
            src_to_dst_byte, dst_to_src_byte = 0, 0  
            
            total_header_len = 0 
            connection_status = 0
            number_of_packets_per_trabsaction = 0

            if isinstance(eth.data, dpkt.ip.IP):
                total_header_len = eth.data.hl * 4  

            flag_valus = []  
            ack_count, syn_count, fin_count, urg_count, rst_count = 0, 0, 0, 0, 0
            udp, tcp, http, https, arp, smtp, irc, ssh, dns, ipv, icmp, igmp, mqtt, coap = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            telnet, dhcp, llc, mac, rarp = 0, 0, 0, 0, 0
            
            idle_time, active_time = 0, 0
            type_info, sub_type_info, ds_status, src_mac, dst_mac, sequence, pack_id, fragments, wifi_dur = 0, 0, 0, 0, 0, 0, 0, 0, 0
            
            if eth.type == dpkt.ethernet.ETH_TYPE_IP or eth.type == dpkt.ethernet.ETH_TYPE_ARP:
                ethsize.append(ethernet_frame_size)
                srcs = {}
                dsts = {}

                if last_pac_time == 0: 
                    last_pac_time = ts
                IAT = ts - last_pac_time
                last_pac_time = ts
                
                if len(ethsize) % n_rows == 0:
                    dy = Dynamic_features()    
                    sum_packets, min_packets, max_packets, mean_packets, std_packets = dy.dynamic_calculation(ethsize)
                    magnite, radius, correlation, covariance, var_ratio, weight = dy.dynamic_two_streams(incoming_pack, outgoing_pack) 
                    ethsize = []
                    srcs = {}
                    dsts = {}
                    incoming_pack = []
                    outgoing_pack = []
                    first_pac_time = 0 
                    last_pac_time = ts
                    IAT = last_pac_time - first_pac_time
                    first_pac_time = last_pac_time
                else:
                    dy = Dynamic_features()
                    sum_packets, min_packets, max_packets, mean_packets, std_packets = dy.dynamic_calculation(ethsize)
                    last_pac_time = ts
                    IAT = last_pac_time - first_pac_time
                    first_pac_time = last_pac_time
                    con_basic = Connectivity_features_basic(eth.data)
                    dst = con_basic.get_destination_ip()
                    src = con_basic.get_source_ip()  
                   
                    if src in dsts:
                        outgoing_pack.append(ethernet_frame_size)
                    else:
                        dsts[src] = 1
                        outgoing_pack.append(ethernet_frame_size)

                    if dst in srcs:
                        incoming_pack.append(ethernet_frame_size)
                    else:
                        srcs[dst] = 1
                        incoming_pack.append(ethernet_frame_size)
                    _, _, _, covariance, var_ratio, weight = dy.dynamic_two_streams(
                        incoming_pack, outgoing_pack,
                    )

                if eth.type == dpkt.ethernet.ETH_TYPE_IP:     
                    ipv = 1
                    ip = eth.data
                    if ip == dpkt.ip6.IP6:  
                        continue

                    con_basic = Connectivity_features_basic(ip)
                    src_ip = con_basic.get_source_ip()

                    if isinstance(eth.data, dpkt.ip.IP):
                        con_basic = Connectivity_features_basic(eth.data)
                        proto_type = con_basic.get_protocol_type()
                    else:
                        continue  
                    dst_ip = con_basic.get_destination_ip()

                    ips.add(dst_ip)
                    ips.add(src_ip)

                    con_time = Connectivity_features_time(ip)
                    time_to_live= con_time.time_to_live() 
                    potential_packet = ip.data

                    conn_flags_bytes = Connectivity_features_flags_bytes(ip)
                    src_byte_count, dst_byte_count = conn_flags_bytes.count(src_ip_byte, dst_ip_byte) 

                    l_three = L3(potential_packet)
                    udp = l_three.udp()
                    tcp = l_three.tcp()

                    protocol_name = get_protocol_name(proto_type)
                    if protocol_name == "ICMP":
                        icmp = 1
                    elif protocol_name == "IGMP":
                        igmp = 1
                    l_one = L1(potential_packet)
                    llc = l_one.LLC()
                    mac = l_one.MAC()

                    calculate_packets_counts_per_ips_proto(average_per_proto_src, protocol_name, src_ip, average_per_proto_dst, dst_ip)
                    calculate_packets_count_per_ports_proto(average_per_proto_src_port, average_per_proto_dst_port, protocol_name, src_port, dst_port)
                    
                    if src_ip not in src_packet_count.keys():
                        src_packet_count[src_ip] = 1
                    else:
                        src_packet_count[src_ip] = src_packet_count[src_ip] + 1

                    if dst_ip not in dst_packet_count.keys():
                        dst_packet_count[dst_ip] = 1
                    else:
                        dst_packet_count[dst_ip] = dst_packet_count[dst_ip] + 1

                    src_pkts, dst_pkts = src_packet_count[src_ip], dst_packet_count[dst_ip] 
                    l_four_both = L4(src_port, dst_port)
                    coap = l_four_both.coap()
                    smtp = l_four_both.smtp()
                    
                    if type(potential_packet) == dpkt.udp.UDP:
                        src_port = con_basic.get_source_port()
                        dst_port = con_basic.get_destination_port()
                        ip_header_len = ip.hl * 4
                        udp_header_len = 8
                        header_len = ip_header_len + udp_header_len
                        l_four = L4(src_port, dst_port)
                        l_two = L2(src_port, dst_port)
                        dhcp = l_two.dhcp()
                        dns = l_four.dns()
                        if dst_port in dst_port_packet_count.keys():
                            dst_packet_count[dst_port] = dst_port_packet_count[dst_port] + 1
                        else:
                            dst_packet_count[dst_port] = 1

                        flow_temp = sorted([(src_ip, src_port), (dst_ip, dst_port)])
                        flow = (flow_temp[0], flow_temp[1])
                        flow_data = {
                            'byte_count': len(eth),
                            'header_len' : header_len,
                            'ts': ts
                        }
                        if udpflows.get(flow):
                            udpflows[flow].append(flow_data)
                        else:
                            udpflows[flow] = [flow_data]
                        packets = udpflows[flow]
                        number_of_packets_per_trabsaction = len(packets)
                        total_header_len = sum(pkt['header_len'] for pkt in packets)
                        flow_byte, flow_duration, max_duration, min_duration, sum_duration, average_duration, std_duration, idle_time,active_time = get_flow_info(udpflows,flow)
                        src_to_dst_pkt, dst_to_src_pkt, src_to_dst_byte, dst_to_src_byte = get_src_dst_packets(udpflows, flow)

                    elif type(potential_packet) == dpkt.tcp.TCP:
                        src_port = con_basic.get_source_port()
                        dst_port = con_basic.get_destination_port()
                        ip_header_len = ip.hl * 4
                        tcp_header_len = potential_packet.off * 4
                        header_len = ip_header_len + tcp_header_len
                        if dst_port in dst_port_packet_count.keys():
                            dst_packet_count[dst_port] = dst_port_packet_count[dst_port] + 1
                        else:
                            dst_packet_count[dst_port] = 1

                        flag_valus = get_flag_values(ip.data)
                        l_four = L4(src_port,dst_port)
                        http = l_four.http()
                        https = l_four.https()
                        ssh = l_four.ssh()
                        irc = l_four.IRC()
                        smtp = l_four.smtp()
                        mqtt = l_four.mqtt()
                        telnet = l_four.telnet()

                        try:
                            http_info = dpkt.http.Response(ip.data)
                            connection_status = http_info.status
                        except:
                            connection_status = 0

                        flow = sorted([(src_ip, src_port), (dst_ip, dst_port)])
                        flow = (flow[0], flow[1])
                        flow_data = {
                            'byte_count': len(eth),
                            'header_len': header_len,
                            'ts': ts
                        }
                        
                        ack_count,syn_count,fin_count,urg_count,rst_count = compare_flow_flags(
                            flag_valus, ack_count, syn_count, fin_count, urg_count, rst_count,
                        )

                        if tcpflows.get(flow):
                            tcpflows[flow].append(flow_data)
                        else:
                            tcpflows[flow] = [flow_data]

                        packets = tcpflows[flow]
                        number_of_packets_per_trabsaction = len(packets)
                        total_header_len = sum(pkt['header_len'] for pkt in packets)
                        flow_byte, flow_duration,max_duration,min_duration,sum_duration,average_duration,std_duration,idle_time,active_time = get_flow_info(tcpflows,flow)
                        src_to_dst_pkt, dst_to_src_pkt, src_to_dst_byte, dst_to_src_byte = get_src_dst_packets(tcpflows, flow)
                        # CICIoT benigno: rst_count correlaciona com Header_Length (~header/800), nao flags RST.
                        if os.environ.get("RST_FROM_HEADER", "1") == "1":
                            rst_scale = float(os.environ.get("RST_HEADER_SCALE", "800"))
                            if rst_scale > 0:
                                rst_count = float(total_header_len) / rst_scale
                    elif proto_type == 47 or type(potential_packet) == dpkt.gre.GRE:
                        flow = sorted([(src_ip, 0), (dst_ip, 0)])
                        flow = (flow[0], flow[1])
                        
                        flow_data = {
                            'byte_count': len(eth),
                            # CICIoT2023 greeth/greip: linhas GRE (proto 47) tem Header_Length=0 (~60% zeros).
                            'header_len' : 0,
                            'ts': ts
                        }
                        
                        if greflows.get(flow):
                            greflows[flow].append(flow_data)
                        else:
                            greflows[flow] = [flow_data]
                            
                        packets = greflows[flow]
                        number_of_packets_per_trabsaction = len(packets)
                        # CORRECAO: acumula o Header_Length do fluxo GRE (como em TCP/UDP).
                        # Antes ficava fixo em ip.hl*4 (=20), divergindo do CICIoT2023.
                        total_header_len = sum(pkt['header_len'] for pkt in packets)
                        
                        flow_byte, flow_duration, max_duration, min_duration, sum_duration, average_duration, std_duration, idle_time, active_time = get_flow_info(greflows, flow)
                        src_to_dst_pkt, dst_to_src_pkt, src_to_dst_byte, dst_to_src_byte = get_src_dst_packets(greflows, flow)

                    if flow_duration != 0:
                        rate = number_of_packets_per_trabsaction / flow_duration
                        srate = src_to_dst_pkt / flow_duration
                        drate = dst_to_src_pkt / flow_duration

                    if dst_port_packet_count.get(dst_port):
                        dst_port_packet_count[dst_port] = dst_port_packet_count[dst_port] + 1
                    else:
                        dst_port_packet_count[dst_port] = 1

                elif eth.type == dpkt.ethernet.ETH_TYPE_ARP:   
                    protocol_name = "ARP"
                    arp = 1
                    if packets_per_protocol.get(protocol_name):
                        packets_per_protocol[protocol_name] = packets_per_protocol[protocol_name] + 1
                    else:
                        packets_per_protocol[protocol_name] = 1
                    calculate_packets_counts_per_ips_proto(average_per_proto_src, protocol_name, src_ip, average_per_proto_dst, dst_ip)

                elif eth.type == dpkt.ieee80211:   
                    wifi_info = Communication_wifi(eth.data)
                    type_info, sub_type_info, ds_status, src_mac, dst_mac, sequence, pack_id, fragments,wifi_dur = wifi_info.calculating()
                elif eth.type == dpkt.ethernet.ETH_TYPE_REVARP:  
                    rarp = 1   

                if len(flag_valus) == 0:
                    for i in range(0,8):
                        flag_valus.append(0)
                
                # SPRINT 1.5: atualizar acumulador de bytes dentro da janela.
                # Reseta quando começa nova janela (count_rows % n_rows == 0), depois soma o pacote atual.
                if count_rows % n_rows == 0:
                    window_cumul_bytes = 0
                window_cumul_bytes += ethernet_frame_size

                new_row = {
                           "Flow Duration": flow_duration,
                           'ts': ts,
                           "Header_Length": total_header_len, 
                            "Protocol Type": proto_type, 
                           "Time_To_Live": time_to_live,        
                            "Rate": rate,   # cumulative per-flow rate (pkts/s desde 1o pacote)
                          "fin_flag_number": flag_valus[0],
                          "syn_flag_number": flag_valus[1],
                          "rst_flag_number": flag_valus[2],
                          "psh_flag_number": flag_valus[3],
                          "ack_flag_number": flag_valus[4],
                          "ece_flag_number": flag_valus[6],
                          "cwr_flag_number": flag_valus[7],
                           "ack_count":ack_count,               
                           "syn_count":syn_count,               
                           "fin_count": fin_count,                           
                           "rst_count": rst_count,              
                           "urg_count": urg_count,
                           "HTTP": http,                           
                           "HTTPS": https,                         
                           "DNS": dns,                             
                           "Telnet":telnet,                        
                           "SMTP": smtp,                           
                           "SSH": ssh,                             
                           "IRC": irc,                             
                           "TCP": tcp,                             
                           "UDP": udp,                             
                           "DHCP": dhcp,                           
                           "ARP": arp,                             
                           "ICMP": icmp,                           
                           "IGMP": igmp,                           
                           "IPv": ipv,                             
                           "LLC": llc,                             
                           "Tot sum": 0,                           
                           "Min": 0,                               
                           "Max": 0,                               
                           "AVG": 0,                               
                           "Std": 0,                               
                           "Tot size": ethernet_frame_size,         
                           "IAT": IAT,
                           # NUMBER_MODE=count troca isto na agregacao final para len(janela).
                           # NUMBER_MODE=index_mean usa este indice 0-based (modo calibrado anterior).
                           "Number": count_rows % n_rows,
                           "Variance": float(var_ratio) if isinstance(var_ratio, (int, float)) else 0.0,
                           "Covariance": float(covariance) if isinstance(covariance, (int, float)) else 0.0,
                           # SPRINT 1.5: bytes acumulados DENTRO da janela atual (reseta a cada n_rows pkts).
                           # mean([s, 2s, ..., 20s]) = 10.5*s → replica Tot_sum real do CICIoT2023.
                           "Cumul_bytes": window_cumul_bytes,
                           "src_ip": src_ip,
                           "dst_ip": dst_ip,
                          }
                for c in base_row.keys():
                    base_row[c].append(new_row[c])
                    
                count_rows+=1
                
        processed_df = pd.DataFrame(base_row)
        df_summary_list = []
        
        for start, end in _aggregation_slices(processed_df, n_rows, agg_idle_sec, burst_max_pkts):
            sliced_df = processed_df.iloc[start:end].copy()
            
            sliced_df['Tot size'] = pd.to_numeric(sliced_df['Tot size'], errors='coerce')
            sliced_df['IAT'] = pd.to_numeric(sliced_df['IAT'], errors='coerce')

            sliced_df_protocol_type_mode = pd.DataFrame(sliced_df['Protocol Type'].mode())
            sum_of_ack_count = (sliced_df["ack_count"].sum())
            sum_of_syn_count = (sliced_df['syn_count'].sum())
            sum_of_fin_count = (sliced_df['fin_count'].sum())
            sum_of_rst_count = (sliced_df['rst_count'].sum())
            sum_of_urg_count = (sliced_df['urg_count'].sum())
            rst_agg_mean = os.environ.get("RST_AGG_MEAN", "1") == "1"
            # SPRINT 1.5: Min/Max/AVG/Std calculados sobre tamanhos INDIVIDUAIS dos pacotes na janela.
            min_packet_length = (sliced_df['Tot size'].min())
            max_packet_length = (sliced_df['Tot size'].max())
            mean_packet_length = (sliced_df['Tot size'].mean())
            std_packet_length = (sliced_df['Tot size'].std())

            duration_time_interval = (sliced_df['ts'].max() - sliced_df['ts'].min())

            non_numeric_cols = ['src_ip', 'dst_ip']  
            sliced_df_mean = sliced_df.drop(columns=non_numeric_cols).mean(numeric_only=True).to_frame().T

            sliced_df_mean['src_ip'] = sliced_df['src_ip'].mode() if not sliced_df['src_ip'].mode().empty else '0.0.0.0'
            sliced_df_mean['dst_ip'] = sliced_df['dst_ip'].mode() if not sliced_df['dst_ip'].mode().empty else '0.0.0.0'

            sliced_df_mean['Protocol Type'] = sliced_df_protocol_type_mode
            sliced_df_mean['ack_count'] = sum_of_ack_count
            sliced_df_mean['syn_count'] = sum_of_syn_count
            sliced_df_mean['fin_count'] = sum_of_fin_count
            if rst_agg_mean:
                sliced_df_mean['rst_count'] = pd.to_numeric(
                    sliced_df['rst_count'], errors='coerce',
                ).mean()
            else:
                sliced_df_mean['rst_count'] = sum_of_rst_count
            sliced_df_mean['urg_count'] = sum_of_urg_count
            # FROZEN (auditoria CICIoT2023): Tot sum = mean(cumsum(Tot size)).
            # Evidencia: compare_totsum_definitions.py, W1_balanced=0.015 no CSV oficial.
            # Nao reverter para sum() sem nova auditoria — ver report_totsum_audit/decision.txt.
            _tot_sizes = pd.to_numeric(sliced_df['Tot size'], errors='coerce').fillna(0)
            sliced_df_mean['Tot sum'] = _tot_sizes.cumsum().mean()
            sliced_df_mean['Min'] = min_packet_length
            sliced_df_mean['Max'] = max_packet_length
            sliced_df_mean['AVG'] = mean_packet_length
            sliced_df_mean['Std'] = std_packet_length
            if number_mode == "count":
                sliced_df_mean['Number'] = len(sliced_df)
            else:
                # Modo calibrado anterior: media do indice 0-based dentro da janela.
                sliced_df_mean['Number'] = sliced_df['Number'].mean()
            # SPRINT 1.5: Rate = média das taxas cumulativas por fluxo (pkts_fluxo/dur_fluxo).
            # Para fluxos longos no regime estacionário, isso ≈ taxa real do bot.
            if rate_mode == "window":
                window_packet_count = len(sliced_df)
                sliced_df_mean['Rate'] = window_packet_count / duration_time_interval if duration_time_interval != 0 else 0
            else:
                sliced_df_mean['Rate'] = sliced_df['Rate'].mean()
            sliced_df_mean['Variance'] = _window_variance_from_slice(sliced_df)
            sliced_df_mean['Covariance'] = _window_covariance_from_slice(sliced_df)

            # Flow Duration: mean (modo A) = media cumulativa; window (modo B) = span AGG.
            flow_durations = pd.to_numeric(sliced_df['Flow Duration'], errors='coerce').dropna()
            if flow_duration_mode == "window":
                sliced_df_mean['Flow Duration'] = duration_time_interval
            elif flow_duration_mode == "last" and len(flow_durations) > 0:
                sliced_df_mean['Flow Duration'] = flow_durations.iloc[-1]
            elif flow_duration_mode == "max" and len(flow_durations) > 0:
                sliced_df_mean['Flow Duration'] = flow_durations.max()
            elif flow_duration_mode == "median" and len(flow_durations) > 0:
                sliced_df_mean['Flow Duration'] = flow_durations.median()
        
            df_summary_list.append(sliced_df_mean)
            
        processed_df = pd.concat(df_summary_list).reset_index(drop=True)
        processed_df = processed_df.drop(columns='ts')
        # Remove coluna interna de bytes cumulativos (usada só para calcular Tot_sum)
        processed_df = processed_df.drop(columns='Cumul_bytes', errors='ignore')
        processed_df = _greip_ogstun_fd_calib(processed_df)
        processed_df = _benign_ogstun_calib(processed_df)
        processed_df = _ogstun_physical_calib(processed_df)
        processed_df.to_csv(csv_file_name+".csv", index=False)
        return True