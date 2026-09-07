# Working on quant-fidelity-suite

This repository produces published scientific claims. A plausible wrong number is
worse than a refusal. Never infer a revision, tokenizer, scope, profile, lane,
dependency, metric, or successful experiment from a name or an old document.

## Start with the right contract

- **Using or citing measurements:** read [llms.txt](llms.txt), then
  [WHAT-WE-MEASURE.md](WHAT-WE-MEASURE.md).
- **Contributing a measurement:** use [registry/CONTRIBUTING.md](registry/CONTRIBUTING.md)
  and [the walkthrough](docs/THIRD-PARTY-QUICKSTART.md). Do not duplicate their recipes.
- **Changing the implementation:** read the relevant code, schema, and actual
  receipt before editing. Probe CLI flags with `--help`; prose is not an API test.
- **Reviewing prior claims:** consult [PUBLISHED-CORRECTIONS](docs/PUBLISHED-CORRECTIONS.md)
  and [REVIEW-DEFERRED](docs/REVIEW-DEFERRED.md). Dated plans and the append-only
  [JOURNAL](JOURNAL.md) are history, not proof of current support or completion.

Current code determines behavior; immutable receipts establish what a past run
recorded. Neither a passing schema nor a hash proves that the model forward was
correct. Preserve the distinction between an implemented mechanism, an exercised
path, an independently reproduced result, and an untested proposal.

## Architecture and ownership

The measurement core is a set of Python/Bash CLIs over explicit filesystem
state, with no `src/` package. The optional read-only Gradio Explorer
(`app.py`, `explorer/`) is a separate web interface, not a paid runner:

1. `bin/measure` / `measure_one.py`: resolve the HF target and revision, check the
   registry before spending, follow lineage, and select a local plan or refusal.
2. `bin/measure_local.py` and `measure_cloud.py`: device/disk/identity planning;
   the latter adds provider admission, budget, leases, watchdogs and teardown.
3. `bin/engines.json`, `fidelity/engines.py`, `fidelity/stages.py`, and
   `stage_measure.sh`: authored engine contracts and execution stages.
4. `engines/tools/`: model capture, weight decoders and scoring. `hf_capture.py`
   with `layer_outer.py` is the portable hidden-capture route; `stream_score.py`
   and `student_capture.py` are the streaming and distributed campaign routes.
5. `bin/fidelity/dscompare.py`: offline comparison of sealed captures;
   `receipt.py` / `seal_receipt.py`: submission sealing.
6. `registry/tools/`: ingestion, validation, receipt-derived rows and rendering.
   `registry-submit` validates locally; it does not publish.

| Path | Boundary |
|---|---|
| `bin/fidelity/` | Shared controller policy, identity, datasets, comparison, provider adapters and receipts; no model forward implementation. |
| `engines/` | Capture/scoring code, per-format surfaces, panels, campaign recipes, upstream patches and parity evidence. |
| `registry/` | Schemas, frozen protocols, receipts, generated data/index/tables and ingestion tools. |
| `container/` | Measurement image and build wrapper; image capability is not paid admission. |
| `app.py`, `explorer/` | Optional Gradio evidence/cost/contribution UI. Keep snapshot predicates, escaped output, bounded offline receipt validation and no-spend/no-token boundaries. `requirements.txt` is web-only. |
| `reports/`, `registry/protocol/` | Experimental evidence. Cite original receipts; do not promote a tiny experiment to universal parity. |
| `suite/`, `calsuite/` | Suite/calibration manifests; large token or tensor payloads may not be present. Some small token panels are committed under `engines/panels/`. |
| `tools/`, `remote/` | Original serving-lane harness and VM campaign pipeline; still relevant to historical receipts. |
| `port/` | Draft native exllamav3 architecture and manual parity harnesses, not a qualified production port. |
| `docs/` | Wire formats, operational instructions, plans and additive corrections. |

State is plans, `job.json`, leases, logs, captures, reports and receipts. A `.done`
marker means the required stage succeeded, not merely that output exists. CLIs
are synchronous; detached remote stages use polling, heartbeats and locked
teardown. Do not introduce an async framework into CLI orchestration without a
demonstrated need; the optional web UI has its own framework lifecycle.

`FIDELITY_ENGINE_ROOT` is the current engine-root variable. The historical
`FIDELITY_K6_ROOT` fallback is read for compatibility, never newly written.

## Scientific invariants

- Full declared vocabulary, fp64 normalization/reduction, direction
  KL(reference || candidate), natural-log units. No top-k substitute. Reject
  non-finite inputs/intermediates; never clamp garbage into plausible output.
- Pin token histories, tokenizer identity, masks and scored positions. Changing
  any of these changes the measurement. A filename is not artifact identity.
- Distinguish **fixed-panel descriptive means**, **run repeatability**, and
  **population inference**. Repeated identical cold runs add no independent text.
  The historical Flash final25 panel has four source documents; clean17 has
  three. Do not transfer those counts to other panels. Use actual document
  provenance; missing provenance cannot authorize inferential p-values.
- Previews and single-window comparisons establish liveness or local behavior,
  not general rate/quality rankings. A nominal confidence level or fitted-model
  coverage simulation is not demonstrated coverage for deployment text.
- Equal recomputed `comparability.key` is necessary, not sufficient. Apply the
  secondary pair/group predicate and inspect lane, pipeline, hardware, replay,
  scope and missing evidence. `unknown` is not permission to rank.
- A same-panel/reference/lane control permits an **excess-over-control** contrast,
  not an automatically causal quantization-error decomposition. It can be
  negative. A candidate below an unquantized control is not invalid for that
  reason alone. Never use a foreign-lane or foreign-scope floor (`BIAS-006`).
- Additional runtime/activation perturbations can amplify or cancel divergence.
  Weights-only KL is not a mathematical lower bound on served KL; a quantized
  proxy reference does not have a guaranteed bias direction.
- Weight reconstruction, native decoding, complete model forward and served
  generation are different validation targets. Real-tensor bitwise parity must
  name the stage and dtype compared. Pre-Hadamard EXL3 equality does not prove
  complete native reconstruction; retain the advisory caveat when that fails.
- Hash tensor **content**, not timestamp-bearing containers, for repeatability.
  Two matching captures prove observed conditional repeatability, not universal
  determinism or correctness. A same-file self-compare is not an independent run.
- Hidden-form capture/replay must state the cut point, each applied head and
  replay precision/backend. A shared head can erase head-quantization error;
  shared head weights do not imply identical logits or padded probability mass.
- Scope/rate/tensor inventory come from artifact bytes. Report actual quantized
  tensor classes, retained tensors and omitted activation/vision/MTP behavior.
  Lower teacher-forced KL alone does not establish task accuracy or long-context
  or native-serving quality.

## Verification and tests

Run the closest executable selftest first, then the full battery and registry:

```bash
python3 bin/selftest_<feature>.py
bash bin/selftest_all.sh
make -C registry check
```

Use an existing suitable interpreter through `FIDELITY_PYTHON` for tensor tests;
do not install into system Python. The battery needs no rental or GPU, but some
rungs access network metadata, optional packages, cached fixtures or read-only
accounts. Report **outer and internal skips** and their prerequisites. Do not
turn a skipped native/GPU/oracle test into a claim of parity.

`make check` permits counted validator warnings for development;
`make -C registry check-release` rejects them. A green development banner is not
release certification. Disposition warnings; do not weaken a gate to get green.

For fixes, retain a behavior-level regression that fails without the fix and
verify that failure in a scratch copy. Tests should exercise observable output,
refusal, identity, numeric boundaries and required coverage—not source substrings,
incidental prose, mock echoes, or a success exit after every real check skipped.
A native constructor/forward exception is a failed requested test, not a SKIP.
A self-comparison zero needs nonzero/perturbed controls to test the computation.
Register new selftests in `bin/SELFTEST-PARTITION.json` and the appropriate battery
or CI tier; explain any intentionally omitted dependency tier.

Additional gates:

- Numeric docs/cards: `python3 bin/check_doc_numbers.py`.
- Names: read [NAMING-SWEEP](docs/NAMING-SWEEP.md), run `selftest_naming_sweep.py`.
- Container/bootstrap/bundle: `selftest_container.py` and `selftest_bundle_complete.py`.
- Engine/surface: matching `engines/tools/selftest_*_offline.py` and scoped real
  tensor evidence. Never manufacture unavailable GPU proof.
- Engine contract edits: `bin/measure-local --probe-engines`.
- Receipt-derived data: `make -C registry reseed-check`; interval changes also
  `make -C registry stat-selftest`.
- Annotations: narrow LSP diagnostics and `selftest_annotations.py`.
  `py_compile`/import alone do not resolve postponed annotations.

There is no installable root package, root build/formatter target, or pytest
collection. `pyproject.toml` configures narrow defect diagnostics only. Read its
comments before changing rules; do not mass-restyle untyped code or remove broad
interrupt-safe exception handlers. Hashed CUDA dependency locks do exist under
`bin/`; they are inputs to the bootstrap, not a root packaging toolchain.

## Dependencies and coding conventions

- Controller startup and all of `registry/` must work on stock Python 3.9.
  Optional tensor/dataset/card modes may lazily import NumPy, Torch, safetensors
  or PyYAML. Keep optional imports off unrelated startup paths.
- Paid CUDA runtime: Python 3.12; `bin/bootstrap_measure.sh` and its hashed lock
  files are the install contract, also used by `container/Dockerfile`.
- MPS has no fp64 support: use CPU for its KL accumulation. CUDA fp64 is a
  separate supported path; record the actual device/backend.
- Match neighboring code: four-space Python, `snake_case`, `pathlib.Path`, small
  dataclasses, `argparse`, `main(...) -> int`; quoted shell arguments and explicit
  error/interrupt traps. A dependency guard must name every dependency it needs.
- Expected invalid inputs use actionable refusals. Preserve exit codes.
  Registry validation accumulates findings rather than hiding later errors.
- Write structured artifacts atomically. Provider/config/console objects and
  scratch directories are existing injection seams; do not add a DI framework.
- Optimize only measured or demonstrably redundant work. A microbenchmark,
  self-comparison, tiny model or one GPU cannot establish an end-to-end speedup
  for all candidates. Numerical-policy changes need separate identity/evidence.

## Money, credentials and publication

- Only execute paid work when explicitly asked. Begin with `--dry-run`, explicit
  `--max-cost` and realistic `--max-runtime`; it must create no provider resource.
  Parser choices, adapter existence and historical recipes are not current paid
  admission. Verify the exact provider/transport/target with the controller.
- Preserve teardown on success, failure, exception and interrupt, with independent
  reaper/watchdog backstops. Verify exact terminal absence using the selected
  provider, not `jl list` for a non-JarvisLabs instance. Unknown state is not gone.
  A leaked instance is a blocker. Never mutate a machine you did not create.
- Budget fetch, setup, all captures/repeats, comparison, retrieval and teardown,
  not compute alone. Hidden-form and logit-form storage costs differ materially.
  Claimed limits depend on functioning backstops; disclose provider limitations.
- Tokens come from protected files, never argv values, logs, receipts, bundles or
  git. No shell tracing around credentials. Keep publication credentials local;
  use the explicit download credential contract for remote fetches. Verify file
  permissions, secure transport, token removal and remote-code exposure.
- GitHub/HF/GHCR publication requires explicit user approval for that destination.
  Before publishing, scan new artifacts for credentials and private host paths.
- Preserve sealed historical receipt bytes and hashed identities. Correct public
  claims additively in [PUBLISHED-CORRECTIONS](docs/PUBLISHED-CORRECTIONS.md), quantify
  any numeric delta, and distinguish third-party attribution. Self-seals are
  tamper-evident against an anchor, not authenticated execution attestations.
- `registry/data/*.jsonl`, `registry/index.json`, registry tables and
  `reports/clean-scope-recompute.*` are generated. Fix the producer and regenerate;
  never hand-edit a scientific value. `CHANGELOG.md` is generated from commits.

## Concurrent work

Other workflows may own these files. Check ownership before shared edits; if a
live campaign holds the runner, record the proposed fix in `docs/REVIEW-DEFERRED.md`
instead. Preserve unrelated changes. Pull with `git pull --rebase origin main`
before every commit, stage only your own explicit paths, and never `git add -A`.
After an on-box fix, reconcile it into the repository the same day.
