---
title: QFS Explorer
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
license: mit
short_description: Explore quantization evidence, costs and receipt provenance
---

# quant-fidelity-suite — distributional fidelity, with receipts

Tools and a public registry for measuring teacher-forced next-token distribution
changes between exact model artifacts. The primary metric is full-vocabulary
KL(reference || candidate), under a declared capture and replay configuration.
It is a fidelity diagnostic, **not a task benchmark or universal quantizer ranking**.

## Choose your starting point

| Goal | Read |
|---|---|
| Understand or cite a score | [WHAT-WE-MEASURE.md](WHAT-WE-MEASURE.md), then [llms.txt](llms.txt) |
| Inspect measurements | [Generated registry tables](registry/README.md) and [index predicates](registry/index.json) |
| Measure or contribute | [Third-party walkthrough](docs/THIRD-PARTY-QUICKSTART.md) and [contribution contract](registry/CONTRIBUTING.md) |
| Change the code | [AGENTS.md](AGENTS.md) and the relevant executable selftest |
| Reuse captures | [Dataset format](docs/FIDELITY-DATASET-SPEC.md) and [CLI reference](bin/README.md) |
| Audit prior claims | [Published corrections](docs/PUBLISHED-CORRECTIONS.md), [review coverage and next work](docs/REVIEW-2026-09-07.md) |

The registry covers several model families, storage formats and historical
protocols. Read [data/measurements.jsonl](registry/data/measurements.jsonl) for the
current inventory: not every row has fp64 accumulation, pinned provenance,
independent verification, or publicly available captures. Unknown and legacy
fields are scientific disclosures, not defaults to fill in.

## QFS Explorer — no-install evidence and cost planning

**[Open QFS Explorer](https://huggingface.co/spaces/malaiwah/qfs-explorer)**
on Hugging Face CPU Basic. The public app is read-only and requires no tokens.

- **Find & explore:** paste an HF model link, distinguish exact/stale/unpinned
  revisions, then open its comparison groups and original evidence. The full
  registry predicate remains authoritative: filtering never makes an
  incomparable group rankable. Displayed numbers are rounded; original records
  retain full precision.
- **Cost planner:** compare dated, sourced HF Spaces/Jobs, RunPod, Vast.ai,
  JarvisLabs and Lambda offers. Enter your own billable duration; unknown
  marketplace/storage prices stay unknown. Multi-GPU rates cover the whole
  configuration, not one GPU. This is not a runtime prediction or spending cap.
- **Cards:** generate the existing HF `model-index` + QFS `x_fidelity` annotation
  and a readable evidence block from selected published measurements. Exact values,
  recorded estimator precision, scope, disclosures, source receipts and snapshot
  links travel together. Merge into an existing README without changing its body
  or unrelated metadata; ambiguous mixed-metric results are refused rather than
  silently discarded. No model card is uploaded automatically.
- **Contribute & own workspace:** duplicate into your account privately, inspect
  a sealed submission or comparison receipt, and follow the actual HF registry
  discussion workflow. An offline validation preview is not registry acceptance.
  The included Dione Q4 example demonstrates the advisory review path.

The app never rents hardware, executes user model/shell code, accepts credentials,
or automatically submits. A private duplicate is still a CPU Explorer; selecting
paid hardware bills that owner but does not add a runner. HF Jobs is a potential
future execution route, not an implemented QFS backend. Existing paid admission,
scientific qualification, retrieval and budget safeguards are unchanged.

The public registry is fetched at startup as one pinned snapshot. A disclosed
bundled fallback is used if HF is unavailable; integrity/fallback notes disable
ranking and card generation. Restart the Space to load the latest default registry.
Evidence links can load an older **explicit 40-character registry revision**; an
unavailable pin or unknown measurement is refused, never replaced by latest data.
Prices remain an authored snapshot in `explorer/pricing.json`, not live availability.

Share the **Permanent link** beside a measurement, or link directly:

```text
https://malaiwah-qfs-explorer.hf.space/?measurement=measurement--glm53.k6-6bpw.brandonmusic-final25&registry_revision=c8c32709884d8f9521799d19d57ae0ea7b5cb621
```

Add `&tab=cards` to open the generator with that measurement selected. `model`
accepts a registry model ID, `group` a group ID emitted by the app, and `target`
an HF model ID/link for lookup. The native HF dataset viewer link is a **live**
search; the accompanying raw registry URL and Explorer link pin the exact snapshot.
Native Gradio saved-session deep links are disabled because they do not preserve
the registry session-state contract.

The Cards tab follows [CARD-ANNOTATION-SPEC.md](docs/CARD-ANNOTATION-SPEC.md), not a
new citation format. Evaluation links do not reclassify existing training datasets.
Private/undisclosed panels retain their logical QFS panel ID without inventing an HF
dataset repository. Missing model/parent/root-dataset identity causes a clear refusal.
Root dataset links require a manifest seal matching the registry reference, model,
panel and head identities. The generator does not invent arXiv IDs, HF verification
badges or a registered `.eval_results` task. Validation is local unless an operator
explicitly submits a public generated card to HF's YAML validator.

Complete CPU fixtures and reproduction evidence are published separately:

| Architecture | Native tiny model | Reproduction evidence |
|---|---|---|
| GLM MoE DSA | [glm-moe-dsa-tiny-random-bf16](https://huggingface.co/malaiwah/glm-moe-dsa-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/glm-moe-dsa-tiny-cpu-repro-v1) |
| Qwen3.8-27B's Qwen3.5 hybrid | [qwen3-5-tiny-random-bf16](https://huggingface.co/malaiwah/qwen3-5-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/qwen3-5-tiny-cpu-repro-v1) |
| GLM-5.3-Flash's GLM5-Next | [glm5-next-tiny-random-bf16](https://huggingface.co/malaiwah/glm5-next-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/glm5-next-tiny-cpu-repro-v1) |
| Qwen3.8-Flash-Next's Qwen4-Exp | [qwen4-exp-tiny-random-bf16](https://huggingface.co/malaiwah/qwen4-exp-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/qwen4-exp-tiny-cpu-repro-v1) |
| MiniMax M2 / M2.7 | [minimax-m2-tiny-random-bf16](https://huggingface.co/malaiwah/minimax-m2-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/minimax-m2-tiny-cpu-repro-v1) |
| MiniMax M3 autoregressive text path | [minimax-m3-tiny-random-bf16](https://huggingface.co/malaiwah/minimax-m3-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/minimax-m3-tiny-cpu-repro-v1) |
| Kimi K2.5 / K2.7 family | [kimi-k25-tiny-random-bf16](https://huggingface.co/malaiwah/kimi-k25-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/kimi-k25-tiny-cpu-repro-v1) |
| Kimi K3 | [kimi-k3-tiny-random-bf16](https://huggingface.co/malaiwah/kimi-k3-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/kimi-k3-tiny-cpu-repro-v1) |
| DeepSeek V4 text path | [deepseek-v4-tiny-random-bf16](https://huggingface.co/malaiwah/deepseek-v4-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/deepseek-v4-tiny-cpu-repro-v1) |
| Spark-X2.5 | [spark2-5-tiny-random-bf16](https://huggingface.co/malaiwah/spark2-5-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/spark2-5-tiny-cpu-repro-v1) |
| IFM K2-Horizon MoVA | [k2-horizon-tiny-random-bf16](https://huggingface.co/malaiwah/k2-horizon-tiny-random-bf16) | [CPU proof](https://huggingface.co/datasets/malaiwah/k2-horizon-tiny-cpu-repro-v1) |

These are random-init text-path tests, not production model-quality measurements.
They do not add a capture/rental action to this read-only app or create registry rows.

The [machine-readable coverage catalog](engines/coverage.json) separates native
architecture proofs, decoded storage formats, and original-release limitations.
`python bin/fidelity_dataset.py architectures list` discovers the fixtures;
`architectures show NAME` gives immutable pins, licenses and runtime restrictions.
`architectures prepare --help` prepares bound local capture/compare commands without
executing a model, downloading weights, renting hardware or publishing.
See the [local capture workflow](docs/THIRD-PARTY-QUICKSTART.md#local-capture-without-renting-hardware).

The new packed readers cover GPTQ v1/v2 INT4, AWQ GEMM INT4, compressed-tensors
INT4/INT8, MLX affine 4/8-bit, CT MXFP4/NVFP4, ModelOpt NVFP4 and supported explicit
mixed NVFP4/FP8 constituents, and block FP8 with UE8M0 scales. Canonical dense Qwen35
GGUF adds a separately disclosed native text-only view; it is not Qwen4-Exp,
Qwen35-MoE, historical fused-QKVZ, vision, MTP or llama.cpp serving support.
Reconstructed-weight results remain **advisory**, with actual own-head replay and
activation omissions recorded. RTN **format fixtures are not optimizer-quality
benchmarks**. Existing EXL3 and flagship decode contracts remain separate.

The [frozen quantized-repository survey](engines/quant-coverage-audit.json) retains
all 100 sampled repositories and classification rules. Forty-seven had insufficient
format metadata; a family name or `quantized` tag does not prove an executable
architecture/layout combination. Fixture-matrix coverage is not global Hub coverage.

Run locally, without changing the CLI's dependency contract:

```bash
python3.12 -m venv .venv-explorer
.venv-explorer/bin/pip install -r requirements.txt
.venv-explorer/bin/python app.py
```

Publish from a suite checkout using your local HF authentication:

```bash
.venv-explorer/bin/python bin/publish_explorer.py --repo <owner>/qfs-explorer
```

Add `--private` when creating a private deployment. The publisher creates only
CPU Basic, refuses an existing non-CPU-Basic deployment, and uploads an explicit
source/schema/registry-metadata allowlist—no model weights, captures, credentials,
or measurement container. It does not copy secrets or change existing visibility.
The hosted app disables unused upload/local-file/remote-file routes; receipts
are pasted as bounded JSON and validated in a time-limited offline worker.

## Measure a quant from an HF link

Start without execution:

```bash
bin/measure malaiwah/GLM-5.3-Flash-TR3-6bpw --plan-only
bin/measure --help
```

`bin/measure` resolves the target and checks the registry first. It never rents a
machine. An already-measured artifact returns its rows; otherwise it plans a
local route or names the missing prerequisite. The target is positional;
`--model` and `--dry-run` are not flags of this front end. Use `--plan-only` to
avoid execution. Revision drift is a different artifact, not permission to reuse
a prior number silently.

### Browse the registry

```bash
bin/registry-view rows --model glm --lane streaming --registry local
bin/registry-view --help
```

The footer identifies the snapshot. Equal recomputed comparability keys are
necessary, not sufficient: inspect secondary predicates and missing evidence.
A declared incompatible or unknown pair must not acquire a ranking merely
because a renderer puts the rows next to each other.

## Before you rent: what is measurable today

Three independent prerequisites must hold:

1. **Exact inputs you can obtain.** A panel's declared public availability, a
   runner fetch descriptor, and token arrays inside a portable dataset are
   different things. Check the selected [panel](registry/data/panels.jsonl) and
   [reference](registry/data/references.jsonl), including their immutable pins.
   Historical Qwen availability labels and old "no roots published" statements
   do not describe every later dataset route. No live availability is implied
   merely by a local metadata record.
2. **An implemented reader on the chosen route.** The matrix below describes
   `engines.json` runner lanes, not every decoder used by `hf_capture.py` and
   not automatic paid admission. Local inputs may be required even for a reader
   that exists.
3. **Admission for this exact artifact/profile/runtime.** The paid controller
   additionally checks scientific evidence, resources and lifecycle safety.
   Parser choices or historical provider recipes are not authorization to spend.

<!-- BEGIN GENERATED: support-matrix -->
<!-- GENERATED by bin/render_support_matrix.py FROM bin/engines.json and the runners' own argparse declarations -- DO NOT EDIT BY HAND.
     Re-render: python3 bin/render_support_matrix.py --write
     Verified by bin/selftest_support_matrix.py (render drift fails). -->

#### Lane × surface (from `bin/engines.json`)

| surface | `sealed-ep8` | `streaming` | `local-mps` | `local-cuda-budget` | `bf16-floor` |
|---|---|---|---|---|---|
| `packed` | ✓ | ✓ | ✓ | ✓ | — |
| `dione` | ✓ | ✓ | — | — | — |
| `native-bf16` | — | ✓ | ✓ | ✓ | ✓ |
| `exl3hf` | — | ✓ | — | — | — |
| `tr3-published` | — | ✓ | — | — | — |
| `gguf` | — | ✓ | — | — | — |
| `mlx` | — | ✓ | — | — | — |
| `nvfp4` | — | ✓ | — | — | — |

#### What each lane is, and how you reach it

| lane | reachable via | receipt class | rates with a profile |
|---|---|---|---|
| `sealed-ep8` | `bin/measure-cloud` | (not declared) | no bpw→profile map (profile named by the campaign driver) |
| `streaming` | `bin/measure --lane streaming`, `bin/measure-cloud` | submittable | dione: 3.0, 4.0 bpw; exl3hf: 2.0, 2.05, 3.05, 4.05 bpw; gguf: any rate; mlx: any rate; native-bf16: unquantized; nvfp4: any rate; tr3-published: 4.0, 6.0 bpw |
| `local-mps` | `bin/measure --lane local-mps`, `bin/measure-local --lane local-mps` | preview | 6.0→k6, 8.0→k8, native→native-bf16 |
| `local-cuda-budget` | `bin/measure --lane local-cuda-budget`, `bin/measure-local --lane local-cuda-budget` | preview | 6.0→k6, 8.0→k8, native→native-bf16 |
| `bf16-floor` | no runner — campaign lane, driven directly (`engines/tools/`) | (not declared) | fixed: native-bf16 |

Reading the matrix: `bin/measure` never rents, so it plans only the
local lanes and redirects `--lane streaming` to `bin/measure-cloud`;
`packed` needs a payload store that is not published, and `native-bf16`
is an unquantized tree. A streaming-only checkmark means these local
runner lanes cannot execute it, not that the engine requires a rental.
Direct engine execution and `fidelity-dataset capture` are separate
routes on hardware you control. No checkmark means absent from these
runner contracts, even where a decoder exists under
[`engines/tools/`](engines/tools/). Paid admission adds its own exact
artifact/profile/runtime gates. `gguf --path` selects one shelf build;
read its measured scope before comparing it with a routed-only quant
([`docs/GGUF-MEASUREMENT.md`](docs/GGUF-MEASUREMENT.md)).
<!-- END GENERATED: support-matrix -->

For portable hidden capture, `bin/fidelity-dataset capture --engine hf-transformers`
uses `hf_capture.py` / `layer_outer.py`; that is a separate entrypoint from the
local streaming lanes in the matrix. See [LAYER-OUTER](docs/LAYER-OUTER.md) and the
[walkthrough](docs/THIRD-PARTY-QUICKSTART.md) for tested surfaces and current limits.

### Recipe 1 — paid execution

```bash
bin/measure-cloud --help
```

Use the [single maintained paid walkthrough](docs/THIRD-PARTY-QUICKSTART.md),
starting with `--dry-run`, explicit `--max-cost` and `--max-runtime`, and the
required separate credential files. Do not reuse a publication credential for
remote downloads. Fetch, setup, candidate captures, repeat qualification,
comparison, retrieval and teardown all belong in the budget.

Watchdogs, the independent reaper, secure transport and exact-absence checks are
load-bearing. Their availability and provider guarantees must be established;
a declared budget is not an unconditional provider-enforced financial cap.
No machine should be rented just to discover an unsupported profile.

### Recipe 2 — local: your own hardware

```bash
bin/measure-local --help
```

`measure-local` is plan-only by default; `--execute` opts in. It downloads
nothing: execution needs the artifact, teacher panel and pipeline locally
(`--artifact-path`, `--teacher-tree`, `--pipeline-root`). Consult the support
matrix rather than assuming every decoder is available on every local lane.

The layer-outer dataset route is different. Its memory requirement depends on
actual model geometry and surface; a 32 GB or 96 GB card is not a blanket fit
guarantee. Use the local planner and [current capture instructions](bin/README.md)
before downloading a checkpoint. Preserve the declared backend and numerical
policy rather than silently substituting a faster one.

### Recipe 3 — submit it

```bash
bin/registry-submit --help
```

Submission validates locally and publishes nothing. The legacy sealed
submission receipt and a modern candidate comparison receipt have different
filing paths; follow [registry/CONTRIBUTING.md](registry/CONTRIBUTING.md). A root
capture can be qualified and retained locally; publication for public reuse is
a separate permissioned action.

### Which cloud?

[Provider benchmark evidence](reports/provider-bench/README.md) describes dated
host/SKU experiments, not current capacity or prices. A streaming inner-loop
microbenchmark is not an end-to-end capture cost for another model or schedule.
[Cloud recipes](docs/CLOUD-RECIPES.md) distinguish adapter capabilities from
current paid admission and its exact safety requirements.

## Capture once, compare retained data

```text
reference weights + token panel -> sealed reference capture A
candidate weights + token panel -> sealed candidate capture B
A, B + declared head/replay policy -> KL, provenance and comparison receipt
```

Offline comparison needs both captures, not a new model forward. Root reuse saves
reference capture work; it does **not** eliminate the cost of obtaining every
candidate capture. Hidden-form storage is compact, but its fp32 head replay need
not be identical to native bf16 serving logits. Own-head replay includes each
capture's head; a shared reference head can erase candidate head error.

Same-file self-comparison is an arithmetic identity. Independent cold captures,
forced computation, perturbed/nonzero controls and scoped forward parity answer
different validation questions. The dataset format makes these distinguishable;
a self-seal alone cannot prove a correct experiment.

## What a result can honestly establish

- A finite-panel mean is descriptive of those exact token histories. Population
  inference requires source-level provenance and a defensible sampling model.
  The old Flash final25 has four source documents, not 25 independent texts;
  other panels have different designs.
- Checkpoint reconstruction and native serving are distinct estimands. Read
  omitted activation/kernel, scope and head disclosures before comparing.
- Excess over a matched control is a signed contrast, not causal quantization
  error. Runtime effects may cancel or amplify; no universal lower/upper bound
  follows. The old 2.52x residual-ratio interpretation is withdrawn.
- Repeated content hashes establish observed conditional repeatability. They do
  not establish cross-device determinism, accuracy or long-context quality.
- Historical headline measurements and their original receipts remain in the
  [registry](registry/README.md) and [campaign journal](JOURNAL.md), with
  [additive corrections](docs/PUBLISHED-CORRECTIONS.md). This overview does not
  duplicate numerical tables that can diverge from their evidence.

## Repository map

| Path | Purpose |
|---|---|
| `bin/` | Public wrappers, local/cloud controllers, dataset/card tools, selftests and bundle contract |
| `engines/tools/` | Model capture, per-format reconstruction, KL scoring and real-tensor evidence |
| `registry/` | Schemas, frozen receipts/protocols, ingestion, generated data and tables |
| `container/` | Reproducible image/build tooling; distinct from paid admission |
| `suite/`, `calsuite/`, `engines/panels/` | Sampling/calibration manifests and selected token panels |
| `tools/`, `remote/` | Original serving-lane and campaign harnesses |
| `port/` | Unqualified draft native exllamav3 architecture and parity harness |
| `docs/`, `reports/`, `JOURNAL.md` | Contracts, scoped experiments and historical corrections |

## Credits

The methodology builds on the author's Qwen fidelity work and
[brandonmusic's pipeline and teacher datasets](https://huggingface.co/brandonmusic).
Base-model providers and quantizers are credited per artifact; trellis kernels
come from [exllamav3](https://github.com/turboderp-org/exllamav3).
Third-party results remain attributed rather than becoming maintainer measurements.
See [LICENSE](LICENSE) for licensing and third-party notices.
