#!/usr/bin/env python3
"""Analysis of AgentForget benchmark results.

Metrics:
- retention AUC: trapezoidal area under the recall-vs-day curve
  (normalized: AUC ∈ [0,1], higher = slower forgetting)
- decay slope: linear fit of recall over days (more negative = faster decay)
- paired bootstrap: per-fact AUC differences (C2−C1, C3−C2) with 95% CI —
  answers "does review help?" and "does consolidation add value?"

Usage: python3 -m benchmark.analyze [results.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITIONS = ["C0", "C1", "C2", "C3"]
COND_LABEL = {"C0": "no-memory", "C1": "store-only", "C2": "memreview (SRS)",
              "C3": "memreview-CLS (SRS+consolidation)"}
TIERS = ["verbatim", "gist", "confusable"]


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def auc(days: list[float], recall: np.ndarray) -> float:
    """Normalized trapezoidal AUC over days ∈ [d0, d_last]."""
    d = np.array(days, dtype=float)
    total = d[-1] - d[0]
    if total <= 0:
        return float(recall[0])
    return float(np.trapezoid(recall, d) / total)


def slope(days: list[float], recall: np.ndarray) -> float:
    return float(np.polyfit(np.array(days, dtype=float), recall, 1)[0])


def per_fact_auc_matrix(data: dict, cond: str, tier: str | None = None):
    """Facts × days outcome matrix for one condition (across seeds)."""
    facts_mats = []
    for seed_run in data["results"]:
        days = seed_run["days"]
        curve = seed_run["conditions"][cond]
        # first seed defines fact order; verify others match
        mats = []
        for d in days:
            outcomes = np.array(curve[str(d)]["facts"], dtype=bool)
            mats.append(outcomes)
        facts_mats.append(np.vstack(mats).T)  # facts × days
    return days, np.mean(facts_mats, axis=0)  # average across seeds


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "results", "results.json")
    data = load(path)
    days = data["results"][0]["days"]
    cfg = data["config"]
    print(f"config: {cfg}")

    # ── 1. retention curves (overall + per tier) ───────────────────────────
    print("\n=== mean surface rate by day (overall, ±std across seeds) ===")
    print("day  " + "".join(f"{c:<28}" for c in CONDITIONS))
    for d in days:
        row = []
        for cond in CONDITIONS:
            vals = []
            for r in data["results"]:
                rates = r["conditions"][cond][str(d)]["rates"]
                vals.append(sum(rates.values()) / len(rates))
            row.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
        print(f"{d:<4}" + "".join(f"{v:<28}" for v in row))

    # ── 2. AUC + slope tables ──────────────────────────────────────────────
    print("\n=== retention AUC (0=instant forgetting, 1=perfect retention) ===")
    print(f"{'condition':<24}{'overall':>10}{'verbatim':>10}{'gist':>10}{'confusable':>12}")
    aucs = {}
    for cond in CONDITIONS:
        aucs[cond] = {}
        for tier in ["overall"] + TIERS:
            vals = []
            for r in data["results"]:
                curve = r["conditions"][cond]
                rec = [curve[str(d)]["rates"][tier] if tier != "overall"
                       else sum(curve[str(d)]["rates"].values()) / 3
                       for d in days]
                vals.append(auc(days, np.array(rec)))
            aucs[cond][tier] = (np.mean(vals), np.std(vals))
        print(f"{COND_LABEL[cond]:<24}"
              + "".join(f"{aucs[cond][t][0]:>8.3f}±{aucs[cond][t][1]:.3f}"
                        for t in ["overall", "verbatim", "gist", "confusable"]))

    print("\n=== decay slope (recall/day; more negative = faster forgetting) ===")
    for cond in CONDITIONS:
        vals = []
        for r in data["results"]:
            curve = r["conditions"][cond]
            rec = [sum(curve[str(d)]["rates"].values()) / 3 for d in days]
            vals.append(slope(days, np.array(rec)))
        print(f"{COND_LABEL[cond]:<24}{np.mean(vals):>10.4f}±{np.std(vals):.4f}")

    # ── 3. paired bootstrap (per-fact AUC differences) ─────────────────────
    print("\n=== paired bootstrap (per-fact AUC, 10k resamples, 95% CI) ===")
    rng = np.random.default_rng(0)
    for a, b, question in [("C1", "C2", "review vs store-only"),
                           ("C2", "C3", "consolidation vs review-only")]:
        _, ma = per_fact_auc_matrix(data, a)
        _, mb = per_fact_auc_matrix(data, b)
        # per-fact AUC
        auc_a = np.array([auc(days, ma[i]) for i in range(ma.shape[0])])
        auc_b = np.array([auc(days, mb[i]) for i in range(mb.shape[0])])
        d = auc_b - auc_a
        boot = np.array([
            np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(10000)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        eff = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        print(f"  {question:<32} ΔAUC = {np.mean(d):+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'✅ significant' if eff else '— not significant'}")

    # ── 4. plots ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"C0": "#888", "C1": "#e6a23c", "C2": "#1e9e5a", "C3": "#0e7c66"}
    for ax, tier in zip(axes, ["overall", "gist"]):
        for cond in CONDITIONS:
            vals = []
            for r in data["results"]:
                curve = r["conditions"][cond]
                rec = [curve[str(d)]["rates"][tier] if tier != "overall"
                       else sum(curve[str(d)]["rates"].values()) / 3
                       for d in days]
                vals.append(rec)
            m, s = np.mean(vals, axis=0), np.std(vals, axis=0)
            ax.plot(days, m, "-o", color=colors[cond], label=COND_LABEL[cond],
                    markersize=4)
            ax.fill_between(days, m - s, m + s, color=colors[cond], alpha=0.12)
        ax.set_title(f"retention — {tier}" if tier != "overall"
                     else "retention — overall")
        ax.set_xlabel("days since ingestion")
        ax.set_ylabel("surface rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(HERE, "results", "retention_curves.png")
    fig.savefig(out_png, dpi=150)
    print(f"\n📈 plot saved → {out_png}")


if __name__ == "__main__":
    main()
