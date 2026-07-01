#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inject.py
=========
Injetor padronizado para reproduzir e injetar trafego sintetico de cada variante
no testbed Open5GS/UERANSIM (ou gravar PCAP offline).

Perfis:
  calibrated  — params calibrados (botnet multi-IP / gre_flow; fidelidade estatistica)
  live        — 10.45.0.2 -> 10.45.0.1 (IP real da UE; testbed Open5GS)

Exemplos (VM UERANSIM):

  sudo python3 inject.py --variant udpplain --profile live --iface uesimtun0
  sudo python3 inject.py --variant greeth  --profile calibrated --iface uesimtun0
  python3 inject.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import synth_generator as gen
from live_ue_profile import (
    CALIBRATED_BASE,
    LIVE_DEFAULT_MAX_PPS,
    LIVE_DEFAULT_WINDOWS,
    LIVE_UE_DST,
    LIVE_UE_DST_BASE,
    LIVE_UE_DST_FIRST_OCTET,
    LIVE_UE_DST_POOL_SIZE,
    LIVE_UE_SRC,
    apply_live_routing_profile,
    apply_live_testbed_profile,
    LIVE_UE_DST_BLOCK_START_THIRD,
    LIVE_UE_DST_NET,
    LIVE_UE_DST_STRATEGY,
    live_routing_dst_ip_from_global_id,
    apply_live_ue_profile,
)

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANT_META = {
    "udpplain": {"label": "Mirai-udpplain", "description": "Mirai UDP Plain"},
    "greip": {"label": "Mirai-greip_flood", "description": "Mirai GRE-IP"},
    "greeth": {"label": "Mirai-greeth_flood", "description": "Mirai GRE-ETH"},
    "benign": {"label": "BenignTraffic", "description": "Trafego IoT benigno mMTC"},
}

PROFILE_INFO = {
    "calibrated": "Params calibrados (multi-bot / gre_flow). Ideal para PCAP e stat_compare.",
    "live": f"Testbed LIVE: src={LIVE_UE_SRC} dst={LIVE_UE_DST} (1 UE, gre_flow OFF).",
}


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(HERE, path)


def _params_for_profile(variant: str, profile: str) -> str:
    if profile == "live":
        return f"params_live_ue_{variant}.json"
    return CALIBRATED_BASE[variant]


def _print_presets():
    print("Variantes e perfis:\n")
    for profile, info in PROFILE_INFO.items():
        print(f"  [{profile}] {info}\n")
    for name in sorted(VARIANT_META.keys()):
        meta = VARIANT_META[name]
        print(f"  {name:10}  label={meta['label']}")
        for profile in ("calibrated", "live"):
            p = _resolve(_params_for_profile(name, profile))
            ok = "OK" if os.path.isfile(p) else "MISSING"
            print(f"             {profile:11} -> {os.path.basename(p)}  [{ok}]")
        print()


def _preflight(mode: str, iface: str | None, params_path: str, variant: str, profile: str):
    print("=== synthgen inject ===")
    print(f"  variante : {variant}")
    print(f"  perfil   : {profile}")
    print(f"  params   : {os.path.abspath(params_path)}")
    print(f"  modo     : {mode}")
    if profile == "live":
        print(f"  rota     : {LIVE_UE_SRC} -> {LIVE_UE_DST}")
    if mode == "live":
        print(f"  iface    : {iface or '(scapy default)'}")
        print()
        print("  Pre-requisitos LIVE:")
        print("    1. UE registrada (uesimtun0 UP, IP 10.45.0.2).")
        print("    2. Captura no UPF: sudo tcpdump -i ogstun -w capture.pcap")
        print("    3. Rode com sudo na VM UERANSIM.")
        print()


def _apply_target_windows(params: dict, variant: str, n_windows: int) -> None:
    """Override gre_flow.target_windows antes de materialize/build_schedule."""
    if "variants" in params and variant in params.get("variants", {}):
        params["variants"][variant].setdefault("gre_flow", {})["target_windows"] = int(n_windows)
    params.setdefault("gre_flow", {})["target_windows"] = int(n_windows)


def main():
    ap = argparse.ArgumentParser(
        description="Injetor padronizado synthgen (udpplain | greip | greeth | benign)."
    )
    ap.add_argument("--variant", choices=sorted(VARIANT_META.keys()), help="Variante de trafego.")
    ap.add_argument(
        "--profile",
        choices=("calibrated", "live"),
        default="calibrated",
        help="calibrated=multi-bot/gre_flow; live=10.45.0.2->10.45.0.1 (testbed).",
    )
    ap.add_argument("--params", default=None, help="JSON customizado (ignora --profile).")
    ap.add_argument("--mode", choices=("live", "pcap"), default="live")
    ap.add_argument(
        "--iface",
        default=os.environ.get("SYNTH_IFACE", "uesimtun0"),
        help="Interface L3 (modo live). Env SYNTH_IFACE.",
    )
    ap.add_argument("--out", default=None, help="PCAP de saida (modo pcap).")
    ap.add_argument("--seed", type=int, default=None, help="Seed fixa (default: timestamp).")
    ap.add_argument(
        "--target-windows",
        type=int,
        default=None,
        metavar="N",
        help="Override gre_flow.target_windows (default live: 1500 via SYNTH_LIVE_WINDOWS).",
    )
    ap.add_argument("--list", action="store_true", help="Lista variantes e perfis.")
    ap.add_argument(
        "--write-live-params",
        action="store_true",
        help="Regenera params_live_ue_*.json a partir das bases calibradas.",
    )
    ap.add_argument(
        "--agg-burst-idle-sec",
        type=float,
        default=None,
        metavar="SEC",
        help="Override AGG_BURST_IDLE_SEC (necessario com sudo, que limpa env). Default: env ou 0.0025.",
    )
    ap.add_argument(
        "--allow-default-burst",
        action="store_true",
        help="Permite inject greeth/greip live com AGG_BURST_IDLE_SEC no default (nao recomendado).",
    )
    ap.add_argument(
        "--export-schedule",
        metavar="PATH",
        default=None,
        help="Grava JSON com IAT/fd pretendidos por janela (dissertacao 5G; mesma seed na captura).",
    )
    ap.add_argument(
        "--live-max-pps",
        type=float,
        default=None,
        metavar="PPS",
        help="Limite de pps entre envios consecutivos (rajadas zero-IAT no live). "
        "Env SYNTH_LIVE_MAX_PPS. Recomendado testbed: 600-800.",
    )
    args = ap.parse_args()

    if args.agg_burst_idle_sec is not None:
        os.environ["AGG_BURST_IDLE_SEC"] = str(args.agg_burst_idle_sec)

    if args.write_live_params:
        for variant, base_name in CALIBRATED_BASE.items():
            base_path = _resolve(base_name)
            if not os.path.isfile(base_path):
                print(f"[skip] base ausente: {base_name}", file=sys.stderr)
                continue
            out_path = _resolve(f"params_live_ue_{variant}.json")
            params = apply_live_ue_profile(gen.load_params(base_path), variant)
            params["variant"] = variant
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            print(f"[write] {out_path}")
        return 0

    if args.list:
        _print_presets()
        return 0

    if not args.variant:
        ap.error("Informe --variant ou use --list.")

    if args.params:
        params_path = _resolve(args.params)
    else:
        params_path = _resolve(_params_for_profile(args.variant, args.profile))

    if not os.path.isfile(params_path):
        if args.profile == "live" and not args.params:
            print(f"[inject] Gerando {os.path.basename(params_path)} ...")
            base_path = _resolve(CALIBRATED_BASE[args.variant])
            params = apply_live_ue_profile(gen.load_params(base_path), args.variant)
            params["variant"] = args.variant
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
        else:
            print(f"[erro] Params nao encontrado: {params_path}", file=sys.stderr)
            return 1

    out_path = _resolve(
        args.out or f"pcap_origin/{args.variant}_{args.profile}.pcap"
    )
    seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000

    _preflight(args.mode, args.iface, params_path, args.variant, args.profile)

    if args.mode == "live":
        os.environ["SYNTH_LIVE_MTU"] = os.environ.get("SYNTH_LIVE_MTU", "1500")

    params = gen.load_params(params_path)
    params["variant"] = args.variant
    params["seed"] = seed

    target_windows = args.target_windows
    if target_windows is None and args.mode == "live":
        try:
            target_windows = int(os.environ.get("SYNTH_LIVE_WINDOWS", str(LIVE_DEFAULT_WINDOWS)))
        except ValueError:
            target_windows = LIVE_DEFAULT_WINDOWS

    if target_windows is not None and args.variant in ("greeth", "greip", "udpplain"):
        _apply_target_windows(params, args.variant, target_windows)

    use_5g_testbed = (
        args.mode == "live"
        and args.profile == "calibrated"
        and args.variant in ("greeth", "greip", "udpplain", "benign")
    )
    if use_5g_testbed:
        if os.environ.get("SYNTH_LIVE_SINGLE_DST", "").lower() in ("1", "true", "yes"):
            vary_dst = False
        elif os.environ.get("SYNTH_LIVE_SINGLE_DST", "").lower() in ("0", "false", "no"):
            vary_dst = True
        else:
            vary_dst = None
        params = apply_live_testbed_profile(
            params, args.variant, target_windows=target_windows, vary_dst=vary_dst,
        )
        if args.variant in ("greeth", "greip", "udpplain") and vary_dst is not False:
            rt = {
                "dst_net": LIVE_UE_DST_NET,
                "dst_first_octet": LIVE_UE_DST_FIRST_OCTET,
                "dst_per_block": LIVE_UE_DST_POOL_SIZE,
            }
            tw = int(target_windows or LIVE_DEFAULT_WINDOWS)
            first_dst = live_routing_dst_ip_from_global_id(rt, 0)
            last_dst = live_routing_dst_ip_from_global_id(rt, max(0, tw - 1))
            print(
                f"[inject] live_routing: {LIVE_UE_SRC} -> {first_dst} .. {last_dst} "
                f"({LIVE_UE_DST_STRATEGY}, {tw} janelas, sem reuse src,dst)",
                flush=True,
            )
        elif args.variant in ("greeth", "greip", "udpplain"):
            print(
                f"[inject] live_routing: {LIVE_UE_SRC} -> {LIVE_UE_DST} (dst fixo; mega-fluxo)",
                flush=True,
            )
            if args.variant == "udpplain":
                print(
                    "[inject] udpplain: mega-fluxo UDP + warmup udp_long_flow (Header_Length LGBM)",
                    flush=True,
                )
        elif args.variant == "benign":
            tw = int(target_windows or LIVE_DEFAULT_WINDOWS)
            print(
                f"[inject] benign testbed: {LIVE_UE_SRC} -> {LIVE_UE_DST} | "
                f"{tw} janelas latent_hmm | 1 dispositivo",
                flush=True,
            )

    materialized = gen.materialize_variant_params(params, args.variant)
    if args.variant == "benign":
        ndev = materialized.get("benign", {}).get("num_devices", "-")
        print(f"[inject] benign devices={ndev} | seed={seed}")
    else:
        bots = materialized.get("botnet", {}).get("size", "-")
        src = materialized.get("botnet", {}).get("src_base", "")
        sfo = materialized.get("botnet", {}).get("src_first_octet", "")
        dst = materialized.get("botnet", {}).get("dst_ip", "")
        gre_flow = materialized.get("gre_flow", {})
        if gre_flow.get("enabled"):
            gs = gre_flow.get("src_base", "?")
            gd = gre_flow.get("dst_base", "?")
            nw = gre_flow.get("target_windows", "?")
            print(f"[inject] {args.variant} bots={bots} | gre_flow src={gs}* dst={gd}* | seed={seed}")
            print(f"[inject] gre_flow=ON | target_windows={nw}")
        else:
            print(f"[inject] {args.variant} bots={bots} | src={src}{sfo} -> dst={dst} | seed={seed}")
            if args.variant in ("greip", "greeth"):
                print(f"[inject] gre_flow=OFF (modo live/testbed)")

    agg_n = gen._aggregation_window_size(
        materialized.get("size", {}),
        materialized.get("dynamic_streams") or {},
        materialized.get("gre_flow") or {},
    )
    burst_idle = gen._agg_burst_idle_sec()
    n_bots = int(materialized.get("botnet", {}).get("size", 1))
    pace = gen.summarize_pacing(materialized, n_pkts=agg_n, n_bots=n_bots)
    if args.variant == "greeth":
        size = materialized.get("size", {})
        comps = size.get("components") or []
        mg = (materialized.get("dynamic_streams") or {}).get("window_hetero", {}).get("markov_gmm") or {}
        wh = (materialized.get("dynamic_streams") or {}).get("window_hetero") or {}
        normal = next((c for c in comps if str(c.get("kind", "")).lower() in ("normal", "gaussian")), {})
        print(
            f"[inject] greeth Min: plateau_prob={mg.get('plateau_prob')} "
            f"normal_mean={normal.get('mean')} jitter_std={mg.get('jitter_std')} "
            f"tail_scale={mg.get('tail_scale')} stay_prob={mg.get('stay_prob')}",
            flush=True,
        )
        fd_inc = wh.get("flow_duration_include_inter_gap", False)
        if fd_inc:
            print(
                f"[inject] flow_duration: window span ~{pace.get('intra_span_mean', 0):.3f}s "
                f"+ inter_gap -> ~{pace.get('flow_duration_window_mean', 0):.3f}s (FLOW_DURATION_MODE=window)",
                flush=True,
            )
        else:
            print(
                f"[inject] flow_duration: ~{pace.get('intra_span_mean', 0):.3f}s intra only; "
                f"inter_gap={pace.get('inter_gap', 0):.3f}s entre fluxos",
                flush=True,
            )
    n_windows = int(materialized.get("gre_flow", {}).get("target_windows", 0) or 0)
    decoupled = pace["inter_window_gap_cap_burst"] is False or pace["pcap_iat_cap_burst"] is False
    if pace["mode"] == "uniform":
        print(
            f"[inject] pacing: intra=uniform({pace['iat_lo']:.3f}-{pace['iat_hi']:.3f}s) "
            f"span/janela~{pace['intra_span_lo']:.2f}-{pace['intra_span_hi']:.2f}s | "
            f"inter_gap={pace['inter_gap']:.3f}s | "
            f"ciclo~{pace['cycle_mean']:.2f}s x {n_windows or '?'} janelas",
            flush=True,
        )
    else:
        print(
            f"[inject] pacing AGG: AGG_BURST_IDLE_SEC={burst_idle:.4f} | "
            f"pcap_iat={pace['pcap_iat_mean']:.6f}s | intra/janela~{pace['intra_span_mean']:.4f}s | "
            f"inter_gap>={pace['inter_gap']:.4f}s ({n_windows or '?'} janelas)",
            flush=True,
        )
    if decoupled:
        print(
            f"[inject] pacing desacoplado de AGG_BURST_IDLE (cap_burst off); "
            f"env AGG_BURST_IDLE_SEC={burst_idle:.4f} so afecta extract opcional",
            flush=True,
        )
    if (
        args.mode == "live"
        and args.variant in ("greeth", "greip")
        and args.profile == "calibrated"
        and not decoupled
        and burst_idle < 0.05
        and not args.allow_default_burst
    ):
        print(
            "\n[erro] AGG_BURST_IDLE_SEC no default (0.0025) com cap_burst activo. "
            "sudo NAO herda env — inject ficara ~18s.\n"
            "  Use run_greeth_v6_inject.sh ou --agg-burst-idle-sec 0.38\n"
            "  Ou v6-E: pcap_iat_cap_burst=false + inter_window_gap_sec=0.04 no JSON",
            file=sys.stderr,
        )
        return 1

    print("[inject] a construir cronograma (1-10 min com 10k janelas; live usa 1500 por defeito)...", flush=True)
    events = gen.build_schedule(params)
    if not events:
        print("[erro] Nenhum evento gerado.", file=sys.stderr)
        return 1

    span = events[-1][0] - events[0][0]
    n_win = len(events) // max(1, agg_n)
    print(f"[inject] cronograma: {len(events)} pacotes, span={span:.2f}s (~{n_win} janelas x {agg_n} pkts)")
    flow_cfg = materialized.get("flow", {})
    semantic_iats = gen._uses_semantic_pcap_iats(args.variant, flow_cfg, False)
    if (
        args.mode == "live"
        and args.variant in ("greeth", "greip")
        and args.profile == "calibrated"
        and n_win >= 100
        and not args.allow_default_burst
        and not semantic_iats
    ):
        min_cycle = pace["cycle_mean"] * 0.65 if decoupled else n_win * 0.15
        min_span = min_cycle * n_win if decoupled else n_win * 0.15
        if span < min_span:
            print(
                f"\n[erro] span={span:.1f}s curto demais para {n_win} janelas "
                f"(min ~{min_span:.0f}s). Verifique pacing no JSON / sudo env.",
                file=sys.stderr,
            )
            return 1
    elif semantic_iats and n_win >= 100:
        zf = float(flow_cfg.get("duration_zero_fraction", 0) or 0)
        print(
            f"[inject] span={span:.1f}s OK (zero_inflated ~{zf:.0%} janelas fd=0; "
            "span curto e esperado — check legacy ignorado)",
            flush=True,
        )

    if args.export_schedule:
        sched_path = _resolve(args.export_schedule)
        gen.export_schedule_manifest(
            params,
            sched_path,
            meta={
                "profile": args.profile,
                "mode": args.mode,
                "params": os.path.basename(params_path),
                "n_packets": len(events),
                "span_sec": span,
            },
        )

    if args.mode == "pcap":
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        gen.write_pcap(events, out_path)
        print(f"[inject] PCAP -> {out_path}")
    else:
        if args.live_max_pps is not None and args.live_max_pps <= 0:
            print("[erro] --live-max-pps deve ser > 0.", file=sys.stderr)
            return 1
        if args.variant in ("greeth", "greip", "udpplain") and args.live_max_pps is None:
            try:
                env_cap = float(os.environ.get("SYNTH_LIVE_MAX_PPS", "0") or "0")
            except ValueError:
                env_cap = 0.0
            if env_cap <= 0 and (semantic_iats or args.variant == "udpplain"):
                print(
                    "[inject] AVISO: rajadas intra-janela podem saturar ogstun (~10%% perda). "
                    f"Use --live-max-pps {LIVE_DEFAULT_MAX_PPS} ou "
                    f"export SYNTH_LIVE_MAX_PPS={LIVE_DEFAULT_MAX_PPS}.",
                    flush=True,
                )
        if args.variant == "benign" and args.live_max_pps is None:
            try:
                env_cap = float(os.environ.get("SYNTH_LIVE_MAX_PPS", "0") or "0")
            except ValueError:
                env_cap = 0.0
            if env_cap <= 0:
                print(
                    f"[inject] AVISO: benign live recomenda "
                    f"SYNTH_LIVE_MAX_PPS={LIVE_DEFAULT_MAX_PPS} no testbed.",
                    flush=True,
                )
        gen.send_live(
            events,
            iface=args.iface or None,
            max_pps=args.live_max_pps,
        )
        print("[inject] Injecao concluida.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
