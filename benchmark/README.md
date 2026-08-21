# AgentForget — a retention benchmark for agent memory

**The thesis:** memory systems should be judged by how their users *forget*,
not by static retrieval scores. Every mainstream agent-memory benchmark
(LoCoMo, LongMemEval, BEAM, MemTrace) measures retrieval accuracy at one
point in time. None measures the forgetting *curve* — how recall decays over
days, and whether an intervention (spaced-repetition review, consolidation)
changes that curve. That is the gap this benchmark fills.

## Neuroscience grounding

Five mechanisms from the memory literature are implemented as *testable
components* — not metaphors. Each has a prediction that the benchmark can
confirm or refute:

| # | Mechanism | Neuroscience basis | Implementation | Prediction tested |
|---|---|---|---|---|
| 1 | **Ebbinghaus decay** | Ebbinghaus (1885) forgetting curve | trace strength `s(t) = exp(−t/τ)`, τ = 7 d | store-only recall decays to ~0 within ~2 τ |
| 2 | **Spacing / testing effect** | Roediger & Karpicke (2006); spacing promotes hippocampal–cortical transfer (Jia et al., PMC11962080) | successful SRS review multiplies stability (FSRS-style, ×2, capped) | review condition holds recall far beyond store-only |
| 3 | **CLS dual stores** | McClelland, McNaughton & O'Reilly (1995): fast episodic (hippocampus) vs slow semantic (neocortex) | Store A = raw facts; Store B = distilled topic summaries | consolidation improves *gist* recall more than verbatim |
| 4 | **Sleep consolidation / interleaving** | Stickgold & Walker (2013); anti-catastrophic-forgetting | nightly pass LLM-merges new facts into existing summaries (old summary always in prompt — extend, never replace) | consolidated knowledge survives long after raw traces decay |
| 5 | **Reconsolidation on retrieval** | Nader et al. (2000): retrieval makes a trace labile, then re-stabilizes it | a successful review re-writes the topic summary + refreshes its strength | summaries stay retrievable across the full 30 d window |
| 6 | **Pattern separation** | Yassa & Stark (2011) | "confusable pairs": near-identical facts with different values | separation stress tier: naive merging must fail here; distinct values must survive consolidation |

## Protocol

Accelerated and fully deterministic (fixed seeds):

- **Corpus**: 240 synthetic facts × 8 topics × 3 tiers (verbatim 100 /
  gist 100 / confusable 40). Fixed seed → identical corpus for everyone.
  Grading is exact-value string matching — no LLM judge needed. The tiers
  probe different retention dynamics: verbatim = low-level identifiers,
  gist = conceptual statements, confusable = near-identical pairs with
  different values (pattern separation).
- **Conditions** (only the memory layer differs; same corpus, same
  embeddings, same decay substrate):
  - `C0 no-memory` — nothing is stored (floor)
  - `C1 store-only` — raw facts retrievable by strength (≈ plain vector store)
  - `C2 memreview` — C1 + the **real memreview SRS scheduler** (its actual
    interval logic; the injected clock runs the 30 days in seconds)
  - `C3 memreview-CLS` — C2 + consolidation engine (Store B, LLM
    distillation via any OpenAI-compatible endpoint; concatenation fallback)
- **Checkpoints**: day 0 (baseline) then 1, 2, 3, 5, 7, 10, 12, 14, 20, 26,
  29, 30. Each day: decay → due reviews (C2/C3) → query all 240 facts.
- **Metrics**: surface rate per tier per day; retention AUC (trapezoid,
  normalized); decay slope (linear fit); paired per-fact bootstrap of
  ΔAUC with 95% CI for the two headline questions.

## Run it

```bash
# 1. (optional) LLM distillation for C3 — any OpenAI-compatible endpoint
export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
#    without these, C3 falls back to deterministic concatenation

# 2. full run (3 seeds × 4 conditions × 13 checkpoints)
python3 -m benchmark.run --seeds 42 43 44

# 3. analysis: curves, AUC, slopes, paired bootstrap, plots
python3 -m benchmark.analyze
```

Requires: `numpy`, `scipy` (stats), `matplotlib` (plots), and the local
embedding endpoint (Ollama `nomic-embed-text` by default,
`MEMREVIEW_EMBED_URL` to override).

## Results (2026-08-21, 3 seeds, LLM distillation via Qwen3.8-27B)

| condition | retention AUC | day-30 surface rate |
|---|---|---|
| C0 no-memory | 0.000 | 0.0% |
| C1 store-only | 0.196 ± 0.003 | 0.0% |
| C2 memreview (SRS) | 0.401 ± 0.006 | 0.0% |
| C3 memreview-CLS | **0.870 ± 0.008** | **43.7% ± 2.1%** |

Paired per-fact bootstrap (10k resamples, 95% CI):

- **review vs store-only**: ΔAUC = **+0.206** CI [+0.198, +0.215] — significant
- **consolidation vs review-only**: ΔAUC = **+0.460** CI [+0.440, +0.478] — significant

By tier (C3): verbatim 0.910, confusable 0.910, gist 0.791. Verbatim and
confusable values survive consolidation best; gist tokens are the least
faithfully preserved by LLM distillation — the benchmark detects
consolidation-fidelity differences, which is exactly what the pattern-
separation tier is for.

Curves: `results/retention_curves.png`. Raw data: `results/results.json`.

## Files

| file | purpose |
|---|---|
| `corpus.py` | deterministic synthetic corpus (3 difficulty tiers) |
| `substrate.py` | trace-strength decay + stability-growth substrate |
| `conditions.py` | C0–C3 memory layers (uses real `memreview.srs`) |
| `consolidation.py` | CLS Store B: distillation, interleaving, reconsolidation |
| `run.py` | protocol driver |
| `analyze.py` | metrics, paired bootstrap, plots |
| `results/` | JSON results + retention-curve figure |
