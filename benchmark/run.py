#!/usr/bin/env python3
"""AgentForget retention benchmark — main driver.

Protocol (accelerated, deterministic):
  day 0: ingest all facts (Store A; C3 also distills Store B)
  days  1,2,3,5,7,10,12,14,20,26,29,30:
     1. decay all traces (Ebbinghaus)
     2. run due SRS reviews (C2/C3) — the real memreview scheduler
     3. query every fact (per-tier surface rate)

Usage:
  python3 -m benchmark.run --seeds 42 43 44 --checkpoints 1 2 3 5 7 10 12 14 20 26 29 30
  # optional LLM distillation (recommended for C3):
  LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... python3 -m benchmark.run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark import corpus as corpus_mod  # noqa: E402
from benchmark.conditions import MemoryLayer  # noqa: E402

from memreview.indexer import embed  # noqa: E402  (Ollama local embeddings)

CONDITIONS = ["C0", "C1", "C2", "C3"]
DEFAULT_CHECKPOINTS = [1, 2, 3, 5, 7, 10, 12, 14, 20, 26, 29, 30]


def _embed_cached(texts: list[str], cache_path: str) -> np.ndarray:
    """Embed with an on-disk cache keyed by text — the corpus is identical
    across seeds, so 240 Ollama calls total instead of 240×conditions×seeds."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            cache = json.load(fh)
    vecs, missing = [], []
    for t in texts:
        if t in cache:
            vecs.append(cache[t])
        else:
            missing.append(t)
    if missing:
        print(f"    embedding {len(missing)} new texts…")
        for t in missing:
            cache[t] = embed(t)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh)
    return np.array([cache[t] for t in texts])


def run_seed(seed: int, corpus_path: str, checkpoints: list[int],
             use_llm: bool, tau: float, theta: float, scale: float,
             growth: float, tau_neo: float, emb_cache_path: str) -> dict:
    facts = corpus_mod.load(corpus_path)["facts"]
    print(f"  [seed {seed}] {len(facts)} facts, {len(checkpoints)} checkpoints")
    fact_emb = _embed_cached([f["text"] for f in facts], emb_cache_path)

    result = {"seed": seed, "days": [0] + checkpoints, "conditions": {}}

    for cond in CONDITIONS:
        layer = MemoryLayer(cond, facts, embed, seed=seed,
                            fact_emb=fact_emb, tau=tau,
                            theta=theta, scale=scale, growth=growth,
                            tau_neo=tau_neo, use_llm=use_llm)
        layer.ingest(0)
        # baseline at day 0
        day0 = _query_all(layer, facts)
        curve = {0: day0}
        for d in checkpoints:
            layer.daily(d)
            curve[d] = _query_all(layer, facts)
        result["conditions"][cond] = curve
        last = curve[checkpoints[-1]]["rates"]
        overall = sum(last.values()) / len(last)
        print(f"    {cond}: day30 overall = {overall:.3f}  "
              f"(verbatim {last['verbatim']:.3f})")
    return result


def _query_all(layer, facts) -> dict:
    """Per-fact outcomes + per-tier surface rates for the current state.
    Per-fact booleans enable paired statistics across facts."""
    tiers = ["verbatim", "gist", "confusable"]
    hits = {t: 0 for t in tiers}
    total = {t: 0 for t in tiers}
    outcomes = []
    for f in facts:
        ok = layer.query(f)
        total[f["tier"]] += 1
        if ok:
            hits[f["tier"]] += 1
        outcomes.append(bool(ok))
    rates = {t: (hits[t] / total[t] if total[t] else 0.0) for t in tiers}
    return {"rates": rates, "facts": outcomes}


def main():
    ap = argparse.ArgumentParser(description="AgentForget retention benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.json"))
    ap.add_argument("--tau", type=float, default=7.0, help="hippocampal decay τ (days)")
    ap.add_argument("--theta", type=float, default=0.45, help="retrievability threshold")
    ap.add_argument("--scale", type=float, default=0.06, help="retrievability noise scale")
    ap.add_argument("--growth", type=float, default=2.0, help="FSRS stability growth")
    ap.add_argument("--tau-neo", type=float, default=21.0, help="neocortical decay τ (days)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM distillation (C3 falls back to concatenation)")
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        print(f"generating corpus → {args.corpus}")
        c = corpus_mod.generate(seed=42)
        corpus_mod.save(c, args.corpus)

    use_llm = not args.no_llm
    if use_llm and not (os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY")):
        print("⚠️  no LLM_BASE_URL/LLM_API_KEY → C3 falls back to concatenation")
        use_llm = False

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    emb_cache_path = os.path.join(HERE, "embed_cache.json")
    t0 = time.time()
    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args.corpus, args.checkpoints,
                                use_llm, args.tau, args.theta, args.scale,
                                args.growth, args.tau_neo, emb_cache_path))
    out = {
        "config": {"seeds": args.seeds, "checkpoints": args.checkpoints,
                   "tau": args.tau, "theta": args.theta, "scale": args.scale,
                   "growth": args.growth, "tau_neo": args.tau_neo,
                   "llm": use_llm},
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"✅ done in {time.time()-t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
