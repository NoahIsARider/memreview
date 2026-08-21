#!/usr/bin/env python3
"""Trace-strength forgetting substrate.

The benchmark's controlled "brain": each fact has a latent trace strength
s(t) that decays exponentially since its last rehearsal (Ebbinghaus 1885):

    s(t) = s0 · exp(−(t − t_last) / τ)

Retrievability is probabilistic (memory is noisy): a trace is surfaceable
with probability P = sigmoid((s − θ) / scale). Interventions modulate s:

- review (spaced repetition / the testing effect, Roediger & Karpicke 2006):
  a successful retrieval *multiplies* stability (FSRS-style): s ← min(s·g, s_max)
- consolidation (Store B): summaries carry the max member strength and decay
  with a slower neocortical τ (slow learning → slow forgetting).
"""
from __future__ import annotations

import numpy as np


class TraceStore:
    """Strength bookkeeping for one condition run."""

    def __init__(self, n_facts: int, tau: float = 7.0, s0: float = 1.0,
                 growth: float = 2.0, s_max: float = 3.0):
        self.n = n_facts
        self.tau = tau
        self.s0 = s0
        self.growth = growth
        self.s_max = s_max
        self.strength = np.full(n_facts, s0, dtype=np.float64)
        self.last_rehearsal = np.zeros(n_facts, dtype=float)

    def decay_to(self, day: float):
        """Apply exponential decay up to `day` (Ebbinghaus curve)."""
        age = np.maximum(0.0, day - self.last_rehearsal)
        self.strength = np.clip(self.s0 * np.exp(-age / self.tau), 1e-6, None)

    def rehearse(self, idx: np.ndarray, day: float):
        """A successful review / replay event — stability grows (testing
        effect / FSRS stability growth), not merely resets."""
        if len(idx) == 0:
            return
        self.strength[idx] = np.minimum(self.strength[idx] * self.growth,
                                        self.s_max)
        self.last_rehearsal[idx] = day

    def strength_at(self, idx: np.ndarray, day: float) -> np.ndarray:
        """Strength of items at `day` without mutating state."""
        age = np.maximum(0.0, day - self.last_rehearsal[idx])
        return np.clip(self.s0 * np.exp(-age / self.tau), 1e-6, None)
