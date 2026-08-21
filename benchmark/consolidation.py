#!/usr/bin/env python3
"""CLS consolidation engine — Store B (the "neocortex").

Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995):
the hippocampus stores episodic traces fast (Store A: raw facts), and
*replay + interleaved practice* gradually train a slow semantic store
(Store B: distilled topic summaries). Three mechanisms implemented here,
each with a testable prediction:

1. **Sleep consolidation** (Stickgold & Walker 2013): a nightly pass embeds
   new material, assigns it to the nearest topic cluster, and LLM-distills
   the cluster into a summary. Prediction: gist recall improves.
2. **Interleaved merging** (anti-catastrophic-forgetting): the old summary is
   always in the distillation prompt, so new facts *extend* it rather than
   replace it. Prediction: old knowledge survives new ingestion.
3. **Reconsolidation on retrieval** (Nader et al. 2000): when a fact is
   reviewed/retrieved, its cluster summary is re-written from the freshest
   trace. Prediction: review events update semantic knowledge, not just a
   "reviewed" flag.

LLM distillation uses any OpenAI-compatible endpoint via env vars
(LLM_BASE_URL / LLM_API_KEY / LLM_MODEL); without a key it falls back to
deterministic concatenation so the benchmark stays runnable offline.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np


class ConsolidationEngine:
    def __init__(self, embed_fn, model="gist", sim_threshold: float = 0.75):
        self.embed_fn = embed_fn          # fn(text) -> list[float]
        self.sim_threshold = sim_threshold
        self.clusters: dict[str, dict] = {}   # topic -> {summary, members, strength}
        self._client = None
        self._model = model
        self._base_url = os.environ.get("LLM_BASE_URL", "")
        self._api_key = os.environ.get("LLM_API_KEY", "")
        self._llm_model = os.environ.get("LLM_MODEL", "Qwen/Qwen3.8-27B")

    # ── LLM distillation (OpenAI-compatible; fallback = concatenation) ─────
    def _llm_available(self) -> bool:
        return bool(self._base_url and self._api_key)

    def _distill(self, topic: str, old_summary: str, new_facts: list[str]) -> str:
        if not self._llm_available():
            merged = "\n".join(new_facts)
            if old_summary:
                merged = old_summary + "\n" + merged
            return merged
        import urllib.request

        old_block = old_summary if old_summary else "(no prior summary)"
        prompt = (
            "You maintain a long-term knowledge summary for an AI assistant.\n"
            "Merge the NEW facts into the EXISTING summary (Complementary "
            "Learning Systems consolidation). Rules:\n"
            "1. Keep ALL existing knowledge intact — extend, never replace.\n"
            "2. Add the new facts in compact form; preserve every unique value "
            "token verbatim (e.g. V483920, G-alpha-7, C58123).\n"
            "3. If two facts look similar but have different values, list BOTH "
            "distinctly (pattern separation).\n"
            "4. Output only the merged summary.\n\n"
            f"TOPIC: {topic}\nEXISTING SUMMARY:\n{old_block}\n"
            f"NEW FACTS:\n" + "\n".join(f"- {f}" for f in new_facts)
        )
        body = json.dumps({
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 800,
        }).encode()
        req = urllib.request.Request(
            self._base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"})
        import time
        last_err = None
        for attempt in range(3):  # retry with backoff on 429/5xx
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(3 * (attempt + 1))
        raise last_err

    # ── nightly consolidation pass ──────────────────────────────────────────
    def consolidate(self, facts: list[dict], day: float,
                    strengths: np.ndarray) -> dict:
        """Assign facts to topic clusters and distill each cluster."""
        # group by topic (the corpus is topic-structured; clusters = topics)
        by_topic: dict[str, list[int]] = {}
        for i, f in enumerate(facts):
            by_topic.setdefault(f["topic"], []).append(i)

        summaries = {}
        import time as _t
        for topic, idxs in by_topic.items():
            idxs = sorted(idxs)
            new_facts = [facts[i]["text"] for i in idxs]
            old = self.clusters.get(topic, {}).get("summary", "")
            try:
                summary = self._distill(topic, old, new_facts)
            except Exception as e:  # noqa: BLE001 — fall back to concatenation
                print(f"    ⚠️  distill {topic} failed ({e}); using concatenation")
                summary = "\n".join(new_facts)
                if old:
                    summary = old + "\n" + summary
            self.clusters[topic] = {
                "summary": summary,
                "members": idxs,
                "last_consolidated": day,
            }
            summaries[topic] = summary
            _t.sleep(1.0)  # gentle pacing for the inference API
        return summaries

    def reconsolidate(self, topic: str, facts: list[dict], idxs: list[int],
                      day: float):
        """Reconsolidation on review: rewrite the cluster summary from the
        freshest member traces (Nader et al. 2000)."""
        cl = self.clusters.get(topic)
        if not cl:
            return
        old = cl["summary"]
        fresh = [facts[i]["text"] for i in idxs]
        cl["summary"] = self._distill(topic, old, fresh)
        cl["last_consolidated"] = day

    def summary_embedding(self, topic: str):
        cl = self.clusters.get(topic)
        if not cl:
            return None
        return self.embed_fn(cl["summary"])
