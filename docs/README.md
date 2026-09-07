# Documentation map

Start with [WHAT-WE-MEASURE](../WHAT-WE-MEASURE.md) for scientific semantics,
[llms.txt](../llms.txt) for reading a result, or [AGENTS.md](../AGENTS.md) for
changing this repository. Documentation includes current contracts and dated
experiments; a historical plan is not evidence that an execution path works.

## Current contracts and entrypoints

| Document | Purpose |
|---|---|
| [FIDELITY-DATASET-SPEC](FIDELITY-DATASET-SPEC.md) | Capture format, identity/seals, hidden/logit form, head policies and comparison refusals. Historical examples retain their original bytes; later rules qualify them. |
| [CARD-ANNOTATION-SPEC](CARD-ANNOTATION-SPEC.md) | Model-card provenance, not a comparability-safe leaderboard by itself. |
| [REGISTRY-INTEGRATION](REGISTRY-INTEGRATION.md) | Comparison/submission boundaries and additive schema integration. |
| [THIRD-PARTY-QUICKSTART](THIRD-PARTY-QUICKSTART.md) | Current prerequisites, safe dry-run, qualification/retrieval and contribution paths. |
| [bin/README](../bin/README.md) | CLI and route reference; verify flags with the actual command's `--help`. |
| [registry/CONTRIBUTING](../registry/CONTRIBUTING.md) | Accepted receipt families and attribution. |
| [CLOUD-RECIPES](CLOUD-RECIPES.md) | Provider admission, credentials and teardown dependencies; no unconditional spending guarantee. |
| [LAYER-OUTER](LAYER-OUTER.md) | Capture scheduling, actual parity evidence and memory limits. |

The portable route already separates three operations:

```text
reference weights + panel -> sealed reference capture A
candidate weights + panel -> sealed candidate capture B
A + B + declared replay/head policy -> comparison receipt
```

Publication is optional for a privately retained qualified root; public reuse
requires a fetchable capture. Comparing retained data avoids re-running weights,
but both reference and candidate datasets are required. Self-comparing one object
is not evidence of independent cold-process repeatability or model correctness.

## Scientific evidence and corrections

- [PROTOCOL-ALIGNMENT](PROTOCOL-ALIGNMENT.md): estimator, masking and sampling
  differences, with bounded experiments rather than universal guarantees.
- [joint-standard](joint-standard/README.md): descriptive window statistics versus
  conditional source-document inference and lane bridges.
- [PUBLISHED-CORRECTIONS](PUBLISHED-CORRECTIONS.md): additive corrections, including
  claims withdrawn without changing historical metric values or sealed receipts.
- [REVIEW-2026-09-07](REVIEW-2026-09-07.md): review coverage, verified repairs, test
  evidence and prioritized remaining work.
- [ARCHITECTURE-DETERMINISM](ARCHITECTURE-DETERMINISM.md): tested devices/shapes and
  limits of cross-device extrapolation.
- [GGUF-MEASUREMENT](GGUF-MEASUREMENT.md): actual scope/rate versus names, and
  reconstruction versus native-serving behavior.
- [reports](../reports/): original experiment logs and derived analyses. Read the
  actual device, sample size, tensor stage and retained artifacts before citing.

## Historical plans and worked examples

[FIDELITY-DATASET-BUILD-PLAN](FIDELITY-DATASET-BUILD-PLAN.md), architecture
feasibility studies and [CAPTURE-SCALING-PLAN](CAPTURE-SCALING-PLAN.md) preserve
historical plans, measurements and projections. [RACE-MODE](RACE-MODE.md) is an
engine experiment, not automatic admission for paid early-model capture.
[REVIEW-DEFERRED](REVIEW-DEFERRED.md) contains original findings and later closures;
read the latest disposition, not only the original failure description.

[examples/SYNTHETIC-DIGESTS](examples/SYNTHETIC-DIGESTS.md) explains the partly
synthetic examples. A valid self-seal on an example is not a measured experiment
or a currently accepted comparison. [cards](cards/README.md) documents local
annotation generation and preserved/corrected model-card bodies; generating a
card does not publish it to Hugging Face.

## Verification

Tests are executable selftests, not pytest discovery. See AGENTS.md for the
required focused checks, full battery, registry regeneration and warning/skip
policy. A passing parser/schema/self-comparison test proves only that boundary;
required native execution that errors or never runs must not be reported as
successful parity. No new GPU, performance, contamination or live-publication
claim should appear without its own exercised evidence.
