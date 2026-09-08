# exllamav3 glm5_next port — design bundle (pre-implementation)

Design + first-draft for running GLM-5.3-Flash natively on exllamav3 (target:
first K6 trellis quant, scored on the fidelity suite in this repo). Produced
2026-08-27 by a 7-agent workflow against exllamav3 v1.4.4, before any
implementation session. Status: **not yet implemented or GPU-tested** — this is
the blueprint the K6 session starts from.

**Updated route, 2026-09-07:** upstream
[exllamav3 v1.4.8](https://github.com/turboderp-org/exllamav3/tree/6ff3a17ea7f3d0026b273d43239398d57f71b788)
now supplies a complete GLM5-next implementation using its released GatedDeltaNet,
MLA and hyperconnection machinery. The draft below is historical; do not overlay
its old class assumptions onto that release.

[`build_glm5_native_fixture.py`](../engines/tools/build_glm5_native_fixture.py)
creates a separately identified KDA128 fixture with all five text blocks and all
eight experts retained, plus independently executed CPU references.
[`glm5_native_forward.py`](tests/glm5_native_forward.py) observes the complete
native text model, full-vocabulary logits, prefill/decode and fresh-state paths.
The published KDA16 tiny fixture is unchanged: the stock native recurrent kernel
does not support its head geometry. A new fixture is not qualification of that
published artifact. Native head padding is retained as raw evidence and excluded
only when scoring the genuine vocabulary; no top-k or heuristic tolerance is used.

**Executed qualification, 2026-09-08:** the separately identified
`glm5-next-native-aligned128-random-bf16-v1` fixture also aligns internal text
linear/MLA widths to 128. Its exact checkpoint and independent CPU references
are retained under
[`glm5-native-aligned128-fixture`](../engines/tools/layer-outer-evidence/glm5-native-aligned128-fixture/fixture-manifest.json).
The stock native runtime completed all text/cache paths but failed exact
decode repeatability; its actual outputs are retained under
[`native-glm5-stock-2026-09-08`](../engines/tools/layer-outer-evidence/native-glm5-stock-2026-09-08/model-cold-repeat.json).

The isolated CUDA experiment observed 64 distinct FP32 final states from 64
identical-input stock invocations in each of three cases, even though the
immediate BF16 outputs matched. An output-only test would miss this defect.
The explicitly changed
[`ordered CUDA backend`](../engines/tools/exl3_ordered_kda.cu) uses fixed-order
partial reductions and single-writer shared normalization buffers; it changes
both ordering and redundant shared writes, so atomics alone are not claimed
as the exclusive cause. Stock wheel files are not modified.

The ordered backend repeated both state and output exactly in the isolation
experiment, then repeated all 32 retained raw/scored complete-model tensors
across fresh processes and fresh-state/reset paths. The
[`complete evidence`](../engines/tools/layer-outer-evidence/native-glm5-ordered-2026-09-08/model-cold-repeat.json)
qualifies determinism **only for this fixture, recorded runtime and observed
paths**. It does not qualify the stock runtime, original published tiny model,
vision, arbitrary history/graph-capture paths or equivalence to the independent
BF16 reference. No numerical tolerance was introduced.

`tests/glm5_native_forward.py --ordered-kda` explicitly selects that changed
runtime. `bin/lambda_native_experiment.py run --fixture-dir <fixture>
--ordered-kda --run-dir <planned-run> --accept-no-provider-deadline` executes the
isolated stock/ordered probe followed by the two full-model processes. CUDA
compiler/header/Ninja prerequisites are checked; no CPU or stock-kernel fallback
stands in for the ordered CUDA implementation.

The design reuses DeepSeek-V4 mHC, `glm_moe_dsa` MLA/indexer/MoE structure
and GDN cache machinery. Source-level similarity is not executed native parity.
Required modules and cache/kernel integration remain unimplemented in this bundle.

**2026-09-07 qualification:** the executable harness now fails native construction,
load/forward errors and missing requested coverage. CPU/reference-only operation
requires `--ref-only` and prints **UNQUALIFIED / NON-NATIVE**. mHC loaded values
must equal independently read checkpoint tensors before numerical comparisons.
The draft probes required installed FLA API parameters instead of a guessed
version floor. Historical embedded source in the design notes is not the current
executable; use `tests/glm5_layer_parity.py` and `glm5_next.py.draft`.
Even all requested layer rows passing would not qualify whole-model serving,
cache rewind, complete long-context behavior or native quant reconstruction.

**Qualification continuation:** `--output` writes atomic layer-only JSON with
required, observed, passed and missing case names. Duplicate or unrelated rows
cannot replace a required comparison; shapes, finite/nonempty tensors, metrics
and tolerances are checked before qualification. `--hc-layer` selects mHC
independently, and unrequested attention/MLP architectures need not be present in
an mHC-only config. Expert truncation and untested sparse/cache regimes remain
explicit limitations.

The synthetic mini-checkpoint now asserts independent loading, causal-prefix and
carried-state behavior, sparse selection and mHC conservation/output values.
Those CPU checks do not supply the missing `KimiDeltaAttention`,
`ContractStreams`, MLA kpool/indexer integration, architecture registration or
native recurrent/cache implementation. Full native qualification still requires
those implementations, supported CUDA dependencies and real checkpoint execution.

| File | What it is |
|---|---|
| [BLUEPRINT.md](BLUEPRINT.md) | Full port blueprint: file-by-file plan, config asserts, tensor-key mapping, what to keep unquantized, hardest-3 risks |
| [glm5_next.py.draft](glm5_next.py.draft) | Syntax-checked first draft of `exllamav3/architecture/glm5_next.py` (PORT-CHECK tags mark unresolved deps) |
| [DRAFT-NOTES.md](DRAFT-NOTES.md) | What the draft assumes and which plan items must land first |
| [tests/glm5_layer_parity.py](tests/glm5_layer_parity.py) | Per-layer torch-oracle parity harness (KDA / NoPE-MLA / noaux_tc MoE / mHC), smoke-tested on a synthetic mini-checkpoint |
| [tests/mini_ckpt_test.py](tests/mini_ckpt_test.py) | Builds the synthetic mini-checkpoint the harness self-checks against |
| [PARITY.md](PARITY.md) | Harness design + measured self-check numbers |
| [REVIEW.md](REVIEW.md) | Historical adversarial review, with dated resolution notes for harness refusal and FLA feature checks; missing modules remain prerequisites |

Context: brandonmusic's 4bpw EXL3 (custom Transformers TP2 adapter +
exllamav3 kernels) proves the quant path works today without this native port;
this bundle is the road to a self-contained exllamav3 architecture and the
K6/K5 variants. See JOURNAL.md at repo root for the campaign log.
