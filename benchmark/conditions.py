#!/usr/bin/env python3
"""The four memory-layer conditions.

C0 no-memory      — the agent has no store: nothing is ever retrievable.
C1 store-only     — raw facts (Store A) + strength decay, no intervention
                    (≈ behaviour of a plain vector store / mem0-style layer).
C2 memreview      — Store A + the real memreview SRS scheduler: due items are
                    actively reviewed (retrieval attempt → grade → advance or
                    reschedule). The *testing effect* (Roediger & Karpicke
                    2006): successful retrieval multiplies trace stability.
C3 memreview-CLS  — C2 + the consolidation engine (Store B): summaries are
                    distilled, reconsolidated on review, and decay more
                    slowly (neocortical durability, τ_neo ≫ τ_hippo).
                    Gist/confusable facts become answerable from summaries.

Retrievability model (substrate): a trace is surfaceable with probability
P = sigmoid((strength − θ) / scale). Strength decays exponentially since the
last rehearsal (Ebbinghaus) and grows on successful review (FSRS-style
stability × growth).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.substrate import TraceStore  # noqa: E402
from benchmark.consolidation import ConsolidationEngine  # noqa: E402

import memreview.srs as srs  # noqa: E402  (import AFTER config rewiring)


class _SimDatetime:
    """Injected clock: memreview.srs schedules on simulated days instead of
    the wall clock, so the 30-day protocol runs in seconds. The real SRS
    logic (intervals, due, review) is untouched — only the time source is
    replaced."""
    base = None
    day = 0.0

    @classmethod
    def now(cls, tz=None):
        from datetime import timedelta
        return cls.base + timedelta(days=cls.day)


_ORIG_DATETIME = srs.datetime
from datetime import datetime as _real_dt

_SimDatetime.base = _real_dt(2026, 1, 1, 12, 0)
srs.datetime = _SimDatetime  # patch module-level `datetime` used by srs


class MemoryLayer:
    def __init__(self, condition: str, facts: list[dict], embed_fn,
                 seed: int, fact_emb: np.ndarray | None = None,
                 tau: float = 7.0, theta: float = 0.45,
                 scale: float = 0.06, growth: float = 2.0,
                 tau_neo: float = 21.0, k: int = 5,
                 use_llm: bool = True):
        self.condition = condition
        self.facts = facts
        self.n = len(facts)
        self.embed_fn = embed_fn
        self.rng = np.random.default_rng(seed)
        self.tau = tau
        self.theta = theta
        self.scale = scale
        self.growth = growth
        self.tau_neo = tau_neo
        self.k = k
        self.use_llm = use_llm

        self.store = TraceStore(self.n, tau=tau, growth=growth)
        self.consolidation = None
        self.summary_strength = {}   # topic -> current strength
        self.summary_last = {}       # topic -> last rehearsal day
        self.summaries = {}          # topic -> {"text": str, "embedding": np.ndarray}
        self.reconsolidated = set()  # topics already LLM-reconsolidated
        self.fact_emb = fact_emb if fact_emb is not None else np.zeros((self.n, 0))
        self.topic_of = [f["topic"] for f in facts]

        # real memreview SRS store, isolated per condition/seed: rewire the
        # config attributes (srs reads them at call time, not import time)
        self._tmp = tempfile.mkdtemp(prefix="memreview-bench-")
        import memreview.config as config

        _SimDatetime.day = 0.0  # reset the injected clock for this layer

        config.HOME = self._tmp
        config.NOTES_DIR = os.path.join(self._tmp, "notes")
        config.CONTEXT_DIR = os.path.join(self._tmp, "contexts")
        config.SRS_FILE = os.path.join(self._tmp, "srs", "items.json")
        config.INDEX_FILE = os.path.join(self._tmp, "index.json")
        config.ensure_dirs()
        self.srs_id_to_fact = {}
        for fi, f in enumerate(facts):
            item = srs.add(f["topic"], f["text"], f["value"])
            self.srs_id_to_fact[item["id"]] = fi

        if condition == "C3":
            self.consolidation = ConsolidationEngine(embed_fn,
                                                     sim_threshold=0.75)

    # ── strength helpers ────────────────────────────────────────────────────
    def _p_retrieve(self, strength: float) -> float:
        return 1.0 / (1.0 + np.exp(-(strength - self.theta) / self.scale))

    def _summary_retrievable(self, topic: str) -> bool:
        if topic not in self.summary_strength:
            return False
        s = self.summary_strength[topic]
        return self.rng.random() < self._p_retrieve(s)

    # ── protocol ────────────────────────────────────────────────────────────
    def ingest(self, day: float):
        """Day 0: everything is fresh. C3 additionally builds Store B
        (LLM distillation if configured, deterministic concatenation
        otherwise — the engine handles the fallback)."""
        if self.condition == "C3":
            self._nightly_consolidate(day)

    def daily(self, day: float):
        """Decay everything, then run due reviews (C2/C3)."""
        _SimDatetime.day = day
        # decay facts (hippocampal τ)
        age = np.maximum(0.0, day - self.store.last_rehearsal)
        self.store.strength = np.clip(np.exp(-age / self.tau), 1e-6, None)
        # decay summaries (neocortical τ_neo — slow forgetting)
        for topic in self.summary_strength:
            age = max(0.0, day - self.summary_last.get(topic, day))
            self.summary_strength[topic] = np.exp(-age / self.tau_neo)

        if self.condition in ("C2", "C3"):
            self._run_reviews(day)

    def _run_reviews(self, day: float):
        due = srs.due()
        if not due:
            return
        reconsolidated_today = set()  # one LLM call per topic per day
        for item in due:
            fid = item["id"]
            i = self.srs_id_to_fact.get(fid)
            if i is None:
                continue
            s = float(self.store.strength[i])
            success = self.rng.random() < self._p_retrieve(s)
            if success:
                self.store.rehearse(np.array([i]), day)  # stability × growth
                srs.review(fid, correct=True)
                topic = self.topic_of[i]
                # LLM reconsolidation at most once per topic per day *and*
                # once per topic per seed (content is static after day 0 —
                # later reviews refresh summary *strength* only, which is the
                # functional part of reconsolidation for retrievability)
                if (self.condition == "C3" and self.consolidation
                        and topic not in reconsolidated_today
                        and topic not in self.reconsolidated):
                    reconsolidated_today.add(topic)
                    self.reconsolidated.add(topic)
                    idxs = [j for j in range(self.n) if self.topic_of[j] == topic]
                    try:
                        self.consolidation.reconsolidate(topic, self.facts,
                                                         idxs, day)
                    except Exception as e:  # noqa: BLE001 — keep the run alive
                        print(f"    ⚠️  reconsolidate failed ({e}); keeping old summary")
                    self._refresh_summary(topic, idxs, day)
                elif (self.condition == "C3" and self.consolidation
                      and topic in self.summary_strength):
                    # strength-only refresh (no LLM)
                    idxs = [j for j in range(self.n) if self.topic_of[j] == topic]
                    self._refresh_summary(topic, idxs, day)
            else:
                srs.review(fid, correct=False)  # reschedule tomorrow

    def _fact_index(self, fid: str):
        for i, f in enumerate(self.facts):
            if f["id"] == fid:
                return i
        return None

    # ── consolidation (C3) ──────────────────────────────────────────────────
    def _nightly_consolidate(self, day: float):
        summaries = self.consolidation.consolidate(self.facts, day,
                                                   self.store.strength)
        for topic, text in summaries.items():
            idxs = [i for i in range(self.n) if self.topic_of[i] == topic]
            self.summaries[topic] = {
                "text": text,
                "embedding": np.array(self.embed_fn(text)),
            }
            self._refresh_summary(topic, idxs, day)

    def _refresh_summary(self, topic: str, idxs: list[int], day: float):
        self.summary_strength[topic] = float(self.store.strength[idxs].max())
        self.summary_last[topic] = day

    # ── queries ─────────────────────────────────────────────────────────────
    def query(self, fact: dict) -> bool:
        """Can the system surface this fact at the current state?"""
        if self.condition == "C0":
            return False
        i = self._fact_index(fact["id"])
        s = float(self.store.strength[i])
        p = self._p_retrieve(s)
        hit = self.rng.random() < p

        if hit:
            return True
        # Store B path (C3): summary contains the value verbatim?
        if self.condition == "C3" and self.consolidation:
            topic = fact["topic"]
            if topic in self.summaries and self._summary_retrievable(topic):
                return fact["value"] in self.summaries[topic]["text"]
        return False
