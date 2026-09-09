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
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - jobs
  - read-repos
  - gated-repos
  - manage-repos
  - write-discussions
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

## QFS Explorer — evidence, plots and caller-funded HF Jobs

**[Open QFS Explorer](https://huggingface.co/spaces/malaiwah/qfs-explorer)**
on Hugging Face CPU Basic. Public browsing and plots require no account or token.

- **Find & explore:** paste an HF model link, distinguish exact/stale/unpinned
  revisions, then open its comparison groups and original evidence. The full
  registry predicate remains authoritative: filtering never makes an
  incomparable group rankable. Displayed numbers are rounded; original records
  retain full precision.
- **Plots:** inspect serialized artifact size in GiB against full-vocabulary KL,
  with linear or zero-preserving symlog scales and adaptive nats units. Select a
  measurement to highlight it without hiding peers. Size basis stays explicit:
  whole-repository bytes, weight-file bytes and tensor payload are not VRAM.
  Mixed bases and uncertified groups have no Pareto ranking guide. Download
  PNG/SVG/CSV/JSON, or embed a pinned snapshot or a clearly labeled live view.
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
- **HF Jobs:** sign in as yourself, choose a tiny fixture, Fruit, the pinned
  Qwen3.8-27B BF16 root, a supported candidate, or immutable capture datasets;
  preview the exact plan or explicitly
  run it. Jobs are billed to your personal namespace, including in private
  duplicates. Native roots and candidates use two fresh captures and numerical
  reproduction controls. Candidate measurement uses each artifact's own head.
- **Durable publication:** results persist in your private bucket without an HF
  token in the worker. Refresh by Job ID after a browser/Space restart. Recovery
  checks the provider invocation, reviewed source, final worker result digest,
  every output byte and scientific receipts. `COMPLETED` alone is not verification.
  Private evidence publication is the default. Public evidence and canonical
  roots require separate redistribution consent; existing repositories are not
  overwritten or made public.
- **Registry review:** post public immutable evidence as a review request, then
  let the registry owner inspect validation, provenance, warnings and exact
  changes. Acceptance requires a separate owner confirmation and unchanged
  registry HEAD. Acceptance and provider metadata checks are **not independent
  model reproduction**.

**Compute boundary:** the app never uses an owner token for a visitor. Launches
need the current caller's HF OAuth identity or explicit Bearer API token; local
mock OAuth cannot spend. The hardware quote includes the selected timeout plus
two startup minutes, but is not an account-wide hard-dollar cap. Storage and
other HF services are separate. HF CPU Basic **Jobs are paid**; CPU Basic Space
hosting is a different service. Cancellation and provider deadlines bound runs.
One active Job is the default. An explicitly reviewed `max_active_jobs: 2`
permits a bounded two-job race; ceilings remain per Job, not a shared campaign
budget. Ambiguous in-flight creations still refuse further launches. New guarded
reservations are never released solely because a provider listing is empty;
genuinely orphaned reservations require reviewed reconciliation.
Explicit deadlines may range from 60 seconds to 24 hours, matching the worker
and bootstrap bound. Every deadline still has to fit the caller's current
quoted compute ceiling; increasing the ceiling alone does not extend a Job.
Preparation and worker execution share the same vetted unexpected-tensor
inventory checks, including exact artifact, model/config/index and name-set
bindings. An inventory that the worker cannot admit is refused before rental.
The check uses the capture CLI's own plain-array/digest loader; provenance
is a separate sidecar, not a wrapper that the capture CLI cannot consume.

**Replay policy:** new CUDA Jobs default to CUDA fp32 head replay and CUDA fp64
normalization/reduction; CPU Jobs use NumPy fp32 replay and CPU fp64 reduction.
The resolved device, dtype and chunk sizes are sealed in `runtime.replay` and
checked through execution, qualification and public reload. There is no CPU
fallback after a CUDA failure. A required CUDA known-answer/zero/nonfinite smoke
runs before expensive capture. Historical plans without this field retain their
original CPU/NumPy semantics; they are not reinterpreted from capture hardware.
Full vocabulary and each capture's own head are preserved. Backend changes can
change last digits and remain distinct measurement identities, not an implicit
claim of CPU/CUDA parity. Recovery verifies tensors and the sealed worker
comparison without recomputing its matrix products on the controller.

**Large captures:** output allowance is explicit per plan (4 GiB by default,
up to 64 GiB); recovery honors the sealed allowance rather than an unrelated
process default. Admission checks verified panel/tokenizer geometry, two cold
captures with their own heads, scratch/durable copies, host RAM and individual
GPU memory. The worker rechecks actual available resources before capture.
These are accounted planning requirements, not measured peak-memory or runtime
guarantees; multi-GPU VRAM is not pooled by this single-device worker.
A 32-GiB output plan needs at least 68 GiB of free disk on the recovery machine
(two approved copies plus 4 GiB headroom). Large results can be recovered by
Job ID from a sufficiently provisioned controller; do not assume the default
CPU Basic Space can hold or qualify them.

The `root:qwen38-27b` preset transports the full historical v5 shard-0 panel
without retokenizing: 512 contexts, 1,048,064 scored positions. Its suggested
A100/7200-second/32-GiB settings do not change the caller's dollar ceiling or
authorize launch. The exact 15 unused MTP draft tensor names are disclosed;
text capture does not establish vision/MTP or native-serving fidelity.
An existing `local-cuda-budget` reference is not relabeled as this worker's
lane: this preset prepares a new root, not an automatic cross-lane comparison.

Worker code and container images are immutable pins. Model code receives
read-only input mounts and a private output volume, never caller credentials.
Only native supported loaders or explicitly reviewed code pins execute; an
arbitrary model `auto_map` is not an execution grant. Failed captures remain
failed, with bounded progress logs and private partial evidence.

Small canonical input metadata is staged through the authenticated private bucket
and verified against its sealed inventory. Large tensor sources use read-only Hub
mounts; the worker builds a canonical dataset view rather than treating Hub
sidecars as captured files. This also avoids the observed truncated JSON prefixes
from some HF dataset-volume files without weakening checksum verification.
Worker computations write to local scratch. Completed-stage evidence is copied
to explicit private-volume paths with byte bounds; final publication verifies
every file by direct readback and checks declared outputs and sealed sidecars
before writing the result manifest last. Bucket directory listings are not the
source of the successful evidence inventory. Scratch, durable output and
recovery storage must all be budgeted; a provider `COMPLETED` status still does
not authorize accepting an incomplete manifest.

The interactive registry loads one pinned public snapshot. A disclosed bundled
fallback disables ranking and card generation. Live plot endpoints refresh the
public registry at most every five minutes and print the resolved revision;
they never expose private Job results. Pinned plot links retain their registry
data snapshot. Restart the Space to refresh its default interactive snapshot.
Evidence links can load an older **explicit 40-character registry revision**; an
unavailable pin or unknown measurement is refused, never replaced by latest data.
Cost-planner prices are an authored snapshot in `explorer/pricing.json`; Jobs
preflight separately fetches current provider hardware and prices.

Share the **Permanent link** beside a measurement, or link directly:

```text
https://malaiwah-qfs-explorer.hf.space/?measurement=measurement--glm53.k6-6bpw.brandonmusic-final25&registry_revision=c8c32709884d8f9521799d19d57ae0ea7b5cb621
```

Add `&tab=cards` for the card generator or `&tab=plots&scale=auto` for a highlighted plot.
`model` accepts a registry model ID, `group` a group ID emitted by the app, and `target`
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

For a private duplicate, retain the README OAuth metadata and requested scopes.
Authorize your own account; do not install an owner `HF_TOKEN` as an application
secret. The canonical registry service uses a stable `QFS_REVIEW_SIGNING_KEY`
for its provider-read attestations. Claims from other workspaces remain reported
unless the caller imports and revalidates their original Job on this service.

Root publication attribution can be completed or corrected after capture, without
rerunning the model. The post-run editor validates the documented author fields,
adds readable evidence navigation, and supersedes any pending request explicitly.
It never changes captured tensors, the sealed execution plan or scientific scope.

The generated `explorer/deployment.json` preserves explicitly reviewed source
revisions so updates need not strand recoverable older Jobs.

The privileged JSON API lives under `/qfs/api/`: `GET account`, `GET jobs`,
`POST jobs/prepare`, `POST jobs/launch`, and per-Job `results`, `publish`,
`request-review`, `metadata`, `cancel` actions. The `metadata` action requires
`confirm_metadata: true` and original attribution facts. Reads of Job status/logs
require the same caller identity. Use an explicit `Authorization: Bearer …` header, never a URL
token. Anonymous calls are refused; general filesystem upload/download proxy
routes are disabled. Public `/plots/` routes generate bounded images/data only.

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
Native fixture bases have qualified public CPU roots and fixture-only registry
references. The HF Jobs presets reproduce these synthetic workflows; quantized
variants are linked artifacts, not invented fine-tunes or new canonical roots.
The registry keeps synthetic reproduction floors separate from trained-model groups.

The [machine-readable coverage catalog](engines/coverage.json) separates native
architecture proofs, decoded storage formats, and original-release limitations.
`python bin/fidelity_dataset.py architectures list` discovers the fixtures;
`architectures show NAME` gives immutable pins, licenses and runtime restrictions.
`architectures prepare --help` prepares bound local capture/compare commands without
executing a model, downloading weights, renting hardware or publishing.
See the [local capture workflow](docs/THIRD-PARTY-QUICKSTART.md#local-capture-without-renting-hardware).

Community collections:
- [Random Architecture Fixtures](https://huggingface.co/collections/malaiwah/qfs-random-architecture-fixtures-6a9f071c3f740e024ce15b72): twelve independently initialized native checkpoints across eleven architecture paths.
- [Matched-Weight Quantization Families](https://huggingface.co/collections/malaiwah/qfs-matched-weight-quantization-families-6a9f071d93dbd3dbf0a1e844): twenty derivatives from three shared-weight sources—eighteen positive-KL lossy cases and two floating-point controls.
- [All fixtures and captures](https://huggingface.co/collections/malaiwah/qfs-test-fixtures-and-reproducible-captures-6a9efbe7e77c03e879c7cde4): the umbrella collection.

The [root capture bundle](https://huggingface.co/datasets/malaiwah/qfs-fixture-root-captures-v1)
retains both cold captures, original token panels, qualification/publication receipts,
and actual measured source. Models are random-initialized, but each quantized
family shares byte-identical pre-quantization weights; RTN format fixtures are not
independently randomized quant candidates or evidence of optimizer calibration.

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

Publish from a suite checkout with an explicit owner-only credential file (never
ambient `$HF_TOKEN` or the HF login cache):

```bash
install -m 600 /dev/stdin ~/.config/qfs/hf-token   # paste the HF token, then Ctrl-D
.venv-explorer/bin/python bin/publish_explorer.py --repo <owner>/qfs-explorer --token-file ~/.config/qfs/hf-token
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
