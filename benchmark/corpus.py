#!/usr/bin/env python3
"""Synthetic corpus for the AgentForget retention benchmark.

Deterministic (fixed seed) so anyone can reproduce the exact corpus:
240 facts across 8 topics, three difficulty tiers:

- verbatim (100): low-level details (IDs, dates, exact numbers) that a
  distilled summary intentionally drops — tests episodic (Store A) recall
  and forgetting-by-decay.
- gist (100): conceptual facts preserved by distillation — tests semantic
  (Store B) recall.
- confusable (40 = 20 pairs): two facts with near-identical embeddings but
  different values — the pattern-separation stress test (Yassa & Stark
  2011): naive merging or decayed discrimination must fail here.

Every fact has a unique `value` token so grading is deterministic string
matching — no LLM judge needed for the core metrics.
"""
from __future__ import annotations

import json
import random

TOPICS = [
    "network", "database", "storage", "deployment",
    "security", "finance", "health", "energy",
]

# verbatim templates: (template, value_generator)
_VERBATIM_TPL = [
    "The {topic} component {name} was configured with {value}",
    "In the {topic} subsystem, {name} has the setting {value}",
    "The {topic} module {name} reports {value}",
    "Record {name} in the {topic} log shows {value}",
]
_GIST_TPL = [
    "The {topic} team prefers {value} over alternatives",
    "For {topic} workloads, the standard practice is {value}",
    "The {topic} policy requires {value}",
    "Reliability in {topic} depends on {value}",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _value(rng: random.Random, kind: str) -> str:
    if kind == "verbatim":
        return f"V{rng.randint(100000, 999999)}"
    if kind == "gist":
        return f"G-{rng.choice(['alpha', 'beta', 'gamma', 'delta', 'epsilon'])}-{rng.randint(1, 99)}"
    return f"C{rng.randint(10000, 99999)}"


def generate(seed: int = 42) -> dict:
    rng = _rng(seed)
    facts = []

    # verbatim tier
    for i in range(100):
        topic = TOPICS[i % len(TOPICS)]
        facts.append({
            "id": f"vb-{i:03d}",
            "tier": "verbatim",
            "topic": topic,
            "text": rng.choice(_VERBATIM_TPL).format(
                topic=topic, name=f"c{i:03d}", value=_value(rng, "verbatim")),
            "value": None,  # set below
            "query": None,  # set below
        })

    # gist tier
    for i in range(100):
        topic = TOPICS[i % len(TOPICS)]
        facts.append({
            "id": f"gs-{i:03d}",
            "tier": "gist",
            "topic": topic,
            "text": rng.choice(_GIST_TPL).format(
                topic=topic, value=_value(rng, "gist")),
            "value": None,
            "query": None,
        })

    # confusable tier: 20 pairs, near-identical wording, different values
    # values are *semantic* capacities (e.g. "320 GiB") — the kind of detail
    # a summary must keep BOTH of, distinctly (pattern separation)
    for i in range(20):
        topic = TOPICS[i % len(TOPICS)]
        base = (f"The {topic} node {chr(65 + i)}-{rng.randint(1, 9)} was assigned "
                f"the capacity of ")
        v1 = f"{rng.randint(100, 900)} GiB"
        v2 = f"{rng.randint(100, 900)} GiB"
        while v2 == v1:
            v2 = f"{rng.randint(100, 900)} GiB"
        facts.append({
            "id": f"cf-{i:03d}a", "tier": "confusable", "topic": topic,
            "text": base + v1, "value": v1, "query": None,
        })
        facts.append({
            "id": f"cf-{i:03d}b", "tier": "confusable", "topic": topic,
            "text": base + v2, "value": v2, "query": None,
        })

    # values & queries: extract value token from the fact text (keeps grading
    # deterministic and independent of template bookkeeping)
    for f in facts:
        if f["value"] is None:
            # value is the trailing token of the generated text
            f["value"] = f["text"].split()[-1]
        f["query"] = _make_query(f, rng)

    return {"seed": seed, "facts": facts}


def _make_query(f: dict, rng: random.Random) -> str:
    """A question whose answer is the fact's unique value token."""
    t = f["text"]
    # strip the value token and ask for it
    stem = t.rsplit(" ", 1)[0].rstrip(".")
    return f"What is the missing detail in: '{stem}' ? Answer with the value only."


def save(corpus: dict, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=1)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/corpus.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    c = generate(args.seed)
    save(c, args.out)
    from collections import Counter
    print(Counter(f["tier"] for f in c["facts"]))
    print(f"saved → {args.out}")
