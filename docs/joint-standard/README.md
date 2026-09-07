# `docs/joint-standard/` — emitted analyses

Everything here was produced by `bin/joint-standard` from the per-window means in
`registry/protocol/per-window/`, offline, on this laptop. No GPU, no
re-measurement, no new number: each file is a deterministic function of data this
repository already publishes.

Every file carries `not_submittable: true` and the frozen protocol's two hashes.
These are **analysis receipts**, not measurements — a measurement row still has to
come through `registry/tools/registry_add.py`.

```
analysis/<series>.panel.json        the full 25-window sealed panel
analysis/<series>.selected.json     the 17-window calibration-clean scope
analysis/paired.<A>-vs-<B>.<scope>.json    historical paired contrasts and qualifications
```

Each analysis carries: the window-clustered SE, the naive SE and the design
effect where the receipt has per-window `std`; percentile and BCa bootstrap
intervals (B=5000, seed 20260829); the per-domain table, whose intervals are
Student-t on `log(mean)` rather than BCa following a fitted-lognormal
simulation (BCa 81.3%, delta-t-log 92.0% in those simulated cells). These are
**simulation coverage rates, not real-domain population confidence**; the
historical domain has one source document. See `docs/PUBLISHED-CORRECTIONS.md` §3;
`sigma_run` and the quadrature; the percentile-exceedance guard; and the refusal
that pooled token percentiles are not derivable from per-window summaries.

## The independent unit is the source document, not the window (P1-15/P1-16)

Added 2026-08-31; clarified 2026-09-07. Brandon final25 derives from **four
source documents** (7/6/6/6 windows), clean17 from **three** (7/5/5).
This is panel-specific, not a rule that every 25-window panel has four
documents. Distinct IDs require provenance and independence assumptions.
These selected texts are not a probability sample of deployment. Historical
`paired.*` receipts carry:

* **`document_level`** — the contrast recomputed at the document unit. For
  K6-vs-K8 the four document means are all positive (ordering survives), and
  the exact sign test is **p = 0.125** (full) / **0.25** (clean17), not the
  window-level 0.0041 / 0.049. The window-level mean, BCa interval and sign
  test remain in the receipt as **descriptions of this fixed panel**;
  `window_stats_are` labels that limitation. Document statistics additionally
  require independent-sign/sampling assumptions; their equal-document mean
  changes weighting relative to the raw token mean and is not a replacement.
* **`contract_a` / `contract_b` and `cross_lane`** — what each side declares
  about its own lane. `paired` now **refuses a mixed-lane contrast** unless an
  explicit `--bridge` statement is passed (carried verbatim into the receipt;
  a bridge is context, not a correction). Of the historical pairings, only
  FP8-vs-BF16floor was same-lane; K6-vs-K8 mixed the sealed K6 with the
  streaming K8. The same-lane recompute is published beside it as
  `paired.K6stream-vs-K8` — mean 0.001331 (full) / 0.000847 (clean17), same
  ordering, same document-level p.
* **`--document-map`** — window→document provenance from the selection
  receipt (or per-window document ids where present). Without provenance,
  statistics are descriptive-only; a map does not itself establish independence.

The correction is logged in `docs/PUBLISHED-CORRECTIONS.md` §4 and
`docs/PEER-REVIEW-RESPONSE.md` (P1-15/P1-16).

## What is deliberately absent

**`paired.K6-vs-BF16floor.*`** — K6 is `same_stack` and the cross-stack BF16
replay floor is `cross_stack`. Pairing them is a cross-lane floor subtraction,
which registry rule BIAS-006 refuses, so the file is not published even though
the tool will happily compute it. The same-lane K6 floor is the *streaming* BF16
floor, whose receipt is scalar-only and therefore cannot be re-read on any scope
other than the full panel.

`paired.FP8-vs-BF16floor.*` **is** published as a historical descriptive
excess-over-control contrast (§4.5 of [`../PROTOCOL-ALIGNMENT.md`](../PROTOCOL-ALIGNMENT.md)).
Matching `cross_stack` labels alone does not prove a causal control or floor
eligibility; unknown bias is not `usable_as_floor`.

## Reproducing

```bash
bin/joint-standard analyze \
    --report registry/protocol/per-window/k6-sealed.json \
    --scope-file registry/protocol/window-selection.brandonmusic-final25.json \
    --scope selected --oracle
```

`--oracle` additionally runs the same bootstrap through brandonmusic's own
`kld_eval.analysis.stats` when it is importable, and records the agreement
(observed: 7e-18).
