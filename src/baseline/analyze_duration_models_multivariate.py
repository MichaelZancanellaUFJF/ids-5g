#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp, wasserstein_distance

PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]

def read_feature(csv_path, label, feature):
    usecols = ["label", feature]
    df = pd.read_csv(csv_path, usecols=usecols)
    if label is not None:
        df = df[df["label"] == label]
    values = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if len(values) == 0:
        raise ValueError(f"Nenhum valor encontrado para {feature} em label={label!r}")
    return values

def fit_lognorm(pos):
    # loc fixo em 0: interpretacao mais estavel para duracoes positivas.
    shape, loc, scale = stats.lognorm.fit(pos, floc=0)
    ll = np.sum(stats.lognorm.logpdf(pos, shape, loc=loc, scale=scale))
    return {
        "shape_sigma": float(shape),
        "loc": float(loc),
        "scale": float(scale),
        "mu": float(np.log(scale)),
        "log_likelihood": float(ll),
    }


def fit_weibull(pos):
    c, loc, scale = stats.weibull_min.fit(pos, floc=0)
    ll = np.sum(stats.weibull_min.logpdf(pos, c, loc=loc, scale=scale))
    return {
        "shape_k": float(c),
        "loc": float(loc),
        "scale": float(scale),
        "log_likelihood": float(ll),
    }




def fit_gmms(log_pos, max_k, seed):
    x = log_pos.reshape(-1, 1)
    rows = []
    best = None
    best_model = None  # <--- Adicione esta variável
    
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5)
        gmm.fit(x)
        bic = float(gmm.bic(x))
        aic = float(gmm.aic(x))
        
        item = {
            "k": k,
            "bic": bic,
            "aic": aic,
            "weights": [float(v) for v in gmm.weights_],
            "means_log10": [float(v[0]) for v in gmm.means_],
            "stds_log10": [float(np.sqrt(v[0][0])) for v in gmm.covariances_],
        }
        
        order = np.argsort(item["means_log10"])
        item["weights"] = [item["weights"][i] for i in order]
        item["means_log10"] = [item["means_log10"][i] for i in order]
        item["stds_log10"] = [item["stds_log10"][i] for i in order]
        
        rows.append(item)
        
        if best is None or bic < best["bic"]:
            best = item
            best_model = gmm  # <--- Armazena o objeto do modelo
            
    return rows, best, best_model  # <--- Retorna o objeto junto


def choose_recommendation(values, pos, gmms, best, force_k=None):
    zero_fraction = float(np.mean(values == 0))
    if force_k is not None:
        chosen = next((g for g in gmms if g["k"] == force_k), best)
    else:
        # Escolha conservadora: se muitos zeros, usar zero-inflated + GMM.
        # Se nao, GMM somente quando melhora BIC de forma relevante vs k=1.
        k1 = gmms[0]["bic"]
        rel_gain = (k1 - best["bic"]) / max(abs(k1), 1.0)
        if zero_fraction > 0.05 or rel_gain > 0.01:
            chosen = best
        else:
            chosen = gmms[0]

    return {
        "duration_model": "zero_inflated_gmm_log10" if zero_fraction > 0.01 else "gmm_log10",
        "zero_fraction": zero_fraction,
        "epsilon": 1e-9,
        "components": [
            {
                "weight": float(w),
                "mean_log10": float(m),
                "std_log10": float(s),
            }
            for w, m, s in zip(chosen["weights"], chosen["means_log10"], chosen["stds_log10"])
        ],
        "chosen_k": chosen["k"],
        "positive_min": float(np.min(pos)) if len(pos) else 0.0,
        "positive_max": float(np.max(pos)) if len(pos) else 0.0,
    }

def make_plots(values, pos, log_pos, gmms, lognorm, weibull, outdir, label, feature):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # Hist + KDE
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(log_pos, bins=80, density=True, alpha=0.45, label=f"Hist log10({feature} > 0)")
    if len(log_pos) > 10:
        kde = stats.gaussian_kde(log_pos)
        xs = np.linspace(np.min(log_pos), np.max(log_pos), 500)
        ax.plot(xs, kde(xs), label="KDE")
    ax.set_title(f"{label} - {feature} (log10)")
    ax.legend()
    fig.savefig(os.path.join(outdir, f"{feature}_hist_kde.png"), dpi=160)
    plt.close(fig)

    # BIC/AIC
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = [g["k"] for g in gmms]
    ax.plot(ks, [g["bic"] for g in gmms], marker="o", label="BIC")
    ax.plot(ks, [g["aic"] for g in gmms], marker="s", label="AIC")
    ax.set_title(f"{label} - {feature} GMM")
    ax.legend()
    fig.savefig(os.path.join(outdir, f"{feature}_bic_aic.png"), dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--features", nargs="+", required=True, help="Lista de features: Std Rate Min ...")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    final_results = {}

    for feature in args.features:
        print(f"\n--- Processando: {feature} ---")
        values = read_feature(args.csv, args.label, feature)
        pos = values[values > 0]
        if len(pos) < 10:
            print(f"Pulo: {feature} tem poucos valores.")
            continue
            
        log_pos = np.log10(pos)

        summary = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "percentiles": {str(p): float(np.percentile(values, p)) for p in PERCENTILES}
        }

        lognorm = fit_lognorm(pos)
        weibull = fit_weibull(pos)
        
        limite = np.percentile(pos, 99.9)
        pos_fit = pos[pos <= limite]
        log_pos_fit = np.log10(pos_fit)
        
        gmms, best_params, best_gmm_obj = fit_gmms(log_pos_fit, args.max_k, args.seed)
        
        X_synth, _ = best_gmm_obj.sample(len(log_pos_fit))
        w1 = wasserstein_distance(log_pos_fit, X_synth.flatten())

        make_plots(values, pos_fit, log_pos_fit, gmms, lognorm, weibull, outdir, args.label, feature)

        final_results[feature] = {
            "summary": summary,
            "wasserstein_log10": float(w1),
            "recommendation": choose_recommendation(values, pos_fit, gmms, best_params)
        }

    with open(outdir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n[OK] Analise salva em {outdir}/analysis_summary.json")

if __name__ == "__main__":
    main()