#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pcap_to_csv.py
==============
Wrapper offline do seu pipeline de extracao. Faz, em um unico comando:

  pcap (nivel IP, ex. ogstun)  ->  [converter: adiciona Ethernet]  ->
  capture.pcap (Ethernet)      ->  [Feature_extraction]            ->  CSV

Reaproveita os modulos existentes em ../pcap2csv (Feature_extraction.py etc.),
processando o pcap em UMA passada (sem o split por tcpdump do Generating_dataset),
o que e muito mais rapido para o laco de calibracao.

NAO precisa do 5G: opera sobre um arquivo .pcap ja capturado/gerado.

GRE-ETH calibrado (defaults se env nao definido):
  AGG_N_ROWS=20, NUMBER_MODE=index_mean  -> Number~9.5, Tot sum~10.5*s
  COVARIANCE_AGG_MODE=max_run, VARIANCE_AGG_MODE=std_over_min_sq  -> semantica CIC greeth

Uso direto:
  python3 pcap_to_csv.py --in pcap_origin/capture.pcap --out-csv capture.pcap.csv
  python3 pcap_to_csv.py --in capture.pcap --out-csv out.csv --agg-n-rows 20 --number-mode index_mean
"""

import argparse
import os
import sys

from scapy.all import rdpcap, wrpcap, Ether, IP

# Torna os modulos do extrator existente importaveis.
HERE = os.path.dirname(os.path.abspath(__file__))
PCAP2CSV_DIR = os.path.abspath(os.path.join(HERE, "..", "pcap2csv"))
if PCAP2CSV_DIR not in sys.path:
    sys.path.insert(0, PCAP2CSV_DIR)


def add_ethernet(in_pcap, out_pcap,
                 src_mac="aa:bb:cc:dd:ee:ff", dst_mac="11:22:33:44:55:66"):
    """Replica o converter.py: encapsula cada pacote IP em um quadro Ethernet."""
    packets = rdpcap(in_pcap)
    converted = []
    for pkt in packets:
        if IP in pkt:
            eth = Ether(src=src_mac, dst=dst_mac, type=0x0800) / pkt[IP]
            # preserva timestamp original (essencial para IAT/duracao)
            if hasattr(pkt, "time"):
                eth.time = pkt.time
            converted.append(eth)
    wrpcap(out_pcap, converted)
    return len(converted)


def extract_features(eth_pcap, out_csv):
    """Roda o Feature_extraction existente em uma passada e gera o CSV."""
    if not os.environ.get("FLOW_DURATION_MODE"):
        os.environ["FLOW_DURATION_MODE"] = "window"
    from Feature_extraction import Feature_extraction
    fe = Feature_extraction()
    prefix = out_csv[:-4] if out_csv.endswith(".csv") else out_csv
    fe.pcap_evaluation(eth_pcap, prefix)  # gera prefix + ".csv"
    produced = prefix + ".csv"
    if produced != out_csv and os.path.exists(produced):
        os.replace(produced, out_csv)
    return out_csv


def _apply_extractor_env(agg_n_rows=None, number_mode=None, burst_idle_sec=None):
    """Defaults GRE-ETH calibrados; respeita env ja definido pelo utilizador."""
    if agg_n_rows is not None:
        os.environ["AGG_N_ROWS"] = str(int(agg_n_rows))
    elif not os.environ.get("AGG_N_ROWS"):
        os.environ["AGG_N_ROWS"] = "20"
    if number_mode is not None:
        os.environ["NUMBER_MODE"] = str(number_mode)
    elif not os.environ.get("NUMBER_MODE"):
        os.environ["NUMBER_MODE"] = "index_mean"
    if burst_idle_sec is not None:
        os.environ["AGG_BURST_IDLE_SEC"] = str(float(burst_idle_sec))
    elif not os.environ.get("AGG_BURST_IDLE_SEC"):
        # Fecha janela AGG nos gaps inter-janela do HMM (~3 ms); evita fragmentacao excessiva.
        os.environ["AGG_BURST_IDLE_SEC"] = "0.0025"
    # CICIoT benigno: rst_count ~ Header_Length/800 (media na janela AGG).
    os.environ.setdefault("RST_FROM_HEADER", "1")
    os.environ.setdefault("RST_HEADER_SCALE", "800")
    os.environ.setdefault("RST_AGG_MEAN", "1")
    return os.environ.get("AGG_N_ROWS"), os.environ.get("NUMBER_MODE")


def run(in_pcap, out_csv, tmp_dir="/tmp/synthgen", quiet=False, agg_n_rows=None, number_mode=None):
    agg, num_mode = _apply_extractor_env(agg_n_rows, number_mode)
    if not quiet:
        print(f"[pcap_to_csv] AGG_N_ROWS={agg} NUMBER_MODE={num_mode}")
    os.makedirs(tmp_dir, exist_ok=True)
    eth_pcap = os.path.join(tmp_dir, "capture_eth.pcap")
    n = add_ethernet(in_pcap, eth_pcap)
    if not quiet:
        print(f"[pcap_to_csv] {n} pacotes encapsulados -> {eth_pcap}")
    extract_features(eth_pcap, out_csv)
    if not quiet:
        print(f"[pcap_to_csv] CSV gerado -> {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="pcap (IP) -> CSV de features (offline).")
    ap.add_argument("--in", dest="in_pcap", required=True, help="pcap de entrada (nivel IP).")
    ap.add_argument("--out-csv", required=True, help="CSV de saida.")
    ap.add_argument("--tmp", default="/tmp/synthgen", help="diretorio temporario.")
    ap.add_argument("--agg-n-rows", type=int, default=None, help="Janela AGG (default 20).")
    ap.add_argument(
        "--number-mode",
        choices=("index_mean", "count"),
        default=None,
        help="Semantica Number (default index_mean -> ~9.5 com n=20).",
    )
    args = ap.parse_args()
    run(
        args.in_pcap,
        args.out_csv,
        tmp_dir=args.tmp,
        agg_n_rows=args.agg_n_rows,
        number_mode=args.number_mode,
    )


if __name__ == "__main__":
    main()
