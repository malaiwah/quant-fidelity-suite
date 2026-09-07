# The same-lane teacher — a hash-evidenced self-consistency control

**Historical recipe; qualified 2026-09-07.** The original recipe predates newer
same-lane registry rows; it is not a current completion/status ledger or spending
authorization. Consult pinned measurement receipts for completed captures.
The current scorer is `engines/tools/kld_report.py`; `bin/kld-preview` handles previews.

## Why

The streaming lane's measured unquantized control against the original teacher
(EP4 runtime, not the sealed EP8 student capture) is
**0.011505922619330299** nats (2 cold runs with identical reported means and
tokenwise-KL digest; `engines/native-bf16-kld.json`). K6's 0.013715 and K8's
0.012384 have descriptive excesses of 0.002209 and 0.000878 over that control.
Those differences are not causal quantization error or a guaranteed lower bound.

A teacher captured by the same lane has exactly zero self-KL **if complete
logit tensor content is identical**. Equal tokenwise-KL digests against a third
teacher do not prove logit identity. Two captures establish conditional
repeatability, not universal reproducibility or whole-model correctness.
`bin/selftest_zero_floor.py` exercises the identity on synthetic captures.

The `.float()` store defines the recorded reference; it causes no self-KL
residue when both stored tensors match. Same dtype and lane names alone are
not evidence that they match.

## The capture (the ~$6 recipe)

One H200 spot (≈$1.99/h, the class the floor was measured on). Stage the
official BF16 tree + the sealed release inventory + the sealed token panel.
Then, twice:

```bash
FIDELITY_PYTHON stream_score.py \
    --source native --capture-role teacher \
    --inventory <sealed glm-release-inventory.v1> \
    --bf16 <official BF16 tree> \
    --teacher <panel locator> --token-panel <sealed token-panel receipt> \
    --profile native-bf16 --ep-emulate 8 --reduce-order fp32 \
    --cold-run {1,2} --out runs/teacher-r{1,2}
```

~1.5–2 h per run (IO-bound: 609 GB of routed BF16 re-read per window; the
floor runs measured ~8.3 min/window on CephFS at ~1.05 GB/s), so ≈$6–8 for
the pair plus staging. Peak ≈47 GB VRAM, same as every streaming run.

**Determinism evidence requirement:** compare the complete logit tensor-content
SHA-256 sets for all 25 windows, not metadata-bearing safetensors file hashes.
That is `evidence_kind: logits_tensor_sha256`, never receipt or archive hashes.
Only after this check may either capture serve as a hash-evidenced self-control.

What `--capture-role teacher` changes, and only this:

* `capture_role` becomes `bf16_teacher` (the exact predicate
  `kld_report._find_teacher_receipt` discovers teachers by — schema stays
  `quant-pipeline.glm53-logit-capture.v1`, verified by ladder rung L1.g);
* the receipt gains a sealed additive block `teacher_provenance`
  (`schema: malaiwah.glm53-same-lane-teacher-provenance.v1`, carrying
  `teacher_label: native-bf16-streaming-v1`, lane, ep_emulate, reduce_order,
  stream_mode, grouped_mm kernel, device, torch/transformers versions,
  cold_run) — covered by `receipt_sha256`.

Refused by construction: `--capture-role teacher` without `--source native`
(a packed student cannot be a teacher), with `--windows` subsets, or with
`--store-positions` sampling (a subset teacher would silently redefine the
panel every student is scored against).

## The floor ladder (decision rule; tooling enforces it)

* **T1** — a fresh BF16 run's complete per-window logit **tensor-content**
  digests equal the teacher's corresponding content digests → self-KL is 0.0.
  Every tokenwise value is `+0.0`; the run's `tokenwise-kld.npy` is the
  np.save of 51,175 float64 zeros, whose sha256 is the fixed constant
  `3ffddc61af8350782afd24c7a69de1f37c260bf5489c4e0f6e3ad89b0ab9be17`
  (409,528 bytes — asserted by `bin/selftest_zero_floor.py`).
* **T2** — the hashes differ → **NEVER assume small.** A different grouped_mm
  kernel / GPU class / torch build changes the bf16 forward itself, and that
  class of difference is exactly what produced the 0.0115 cross-topology
  floor: it can be 1e-2-class. Measure the residual floor with the native run
  just made.

`bin/fidelity-stats attributable` refuses a zero claimed only by a legacy
scalar summary: an `evidence_kind` string and explanatory prose do not bind
teacher/student tensor operands. Establish zero with two independently captured,
sealed datasets through `fidelity-dataset compare --self-compare --force-compute`
and retain the qualification evidence. T1 is a content comparison, not a label.

**Scope note.** A different CUDA or Apple/MPS lane has an unknown cross-stack
control and unknown bias direction until measured. Unknown direction is a valid
disclosure, not evidence usable as a floor. Subtracting the same control from
two fixed means preserves their difference algebraically; changing the teacher,
lane, panel or arithmetic can change that difference. No cross-lane invariance
follows from the subtraction.

## Recording it in the registry (paste-ready spec — registry/ is not edited here)

One new `references.jsonl` row:

```json
{"id": "reference--glm-5.3-flash--native-bf16--streaming-v1",
 "panel_ref": "panel--glm53.brandonmusic.final25",
 "artifact_ref": "artifact--zai-org.glm-5.3-flash-bf16.a6c167b6",
 "reference_kind": "native_bf16",
 "logits_available": true,
 "capture": {
   "stack": "glm53-fidelity-suite streaming lane (stream_score.py --source native --capture-role teacher, EP8-emulated single device, reduce-order fp32)",
   "stack_version": "<git rev of the capture checkout>",
   "compute_dtype": "bf16",
   "logits_dtype": "float32",
   "capture_receipt_sha256": "<the run's receipt_sha256>"},
 "self_consistency": {"floor_measurement_ref": "<the T1/T2 floor measurement row's id>"}}
```

Consequences, by the registry's own arithmetic:

* `comparability.key_inputs` includes `reference_id`, so rows against the new
  reference form a **new `cmp--` group** and can never be pooled with
  sealed-teacher rows. That is the anti-conflation mechanism working, not a
  limitation.
* Measurements against it use `estimator.stack_relation: "same_stack"` with
  **no bias block** — that is the accuracy win over today's rows.
* A teacher capture is a REFERENCE record, never a measurement:
  `bin/fidelity/receipt.py` refuses any `capture_role: bf16_teacher` input to
  `build_submission`, and `registry_add.py` has no adapter for a capture
  receipt (both demonstrated by selftests).

## Portability rule (already implemented)

A teacher tree moved off its capture box keeps the receipt's absolute logit
paths. `kld_report.py` and `bin/kld-preview` fall back to
`<teacher_root>/logits/<basename>` and, **in the fallback path only**, verify
the file's sha256 against the receipt row before use — hash content, not
containers. The sealed fast path (recorded path exists) is byte-identical to
the pre-change behaviour.

## Open items

* The $6 GPU pair-run itself (T1 verification + publishing the 31.7 GB tree
  the way the sealed teacher is published).
* The local (Apple) lane's floor against any teacher — needs a local native
  pass; until then local runs are previews and say so.
