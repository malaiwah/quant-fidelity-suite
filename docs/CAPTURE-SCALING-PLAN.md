# Plan of record — how we scale captures, and what it costs

**Historical plan of record, 2026-08-30, corrected 2026-09-07.** The measured
single-device schedule and replay experiments below remain evidence at their
recorded scope. Pipeline parallel and production fetch overlap were design
directions, not shipped/proven automatic speedups. The admitted paid route
still refuses race mode; see [`CLOUD-RECIPES.md`](CLOUD-RECIPES.md).

M1.5 measured **10.13x** faster replay on a force-computed **root
self-comparison** (1,754.71 s → 173.27 s, peak 7.13 GB), not arbitrary
nonzero candidate comparisons. A separate 16-window root replay experiment
measured 5.24e-12 nats between backends, not a bound on any candidate row's
change. Receipts: [`reports/m15-replay-backend/`](../reports/m15-replay-backend/).

**Measured 2026-08-30** (`766a7e8`). The layer-outer engine is built and
bit-identical to the window-outer schedule on two architectures and two devices;
`docs/LAYER-OUTER.md` carries the digests. Measured on Fruit on an L4:
peak CUDA allocated **10.409 -> 2.167 GB (4.80x)**, resident weights **9.144 ->
1.471 GB**. The GLM-5.3 projection below is revised accordingly and is now an
extrapolation from measurement rather than from arithmetic alone.

The historical GLM-5.3 projection was peak VRAM **~47–51 GB**,
**0.4–1.6 min/window**, **Stage C $1.08–3.52**. It was an incomplete
projection, not a full-run quote. Later full-model observations are below;
the measured loader peaks are recorded in [`LAYER-OUTER.md`](LAYER-OUTER.md).

The expert-fusion transient was projected from CPU RSS and a smaller model,
not bounded for every full-model/device path. The former **12.5 min/run**
load-overhead term was erroneous: GLM-5.3 has **768** source expert matrices
per sparse layer (256 × 3), not 76,800. Multiplying by shard count was
unjustified. Even 0.13 ms × 768 × 75 = 7.488 s is only a toy-rate
extrapolation, not isolated/measured full-model overhead. See the correction
in [`LAYER-OUTER.md`](LAYER-OUTER.md).

**Measured 2026-09-04, real GLM-5.3 (full, 78L/6144), not projected.** Two
captures (root BF16 resume and FP8 candidate), same panel shape
(`panel--glm53.malaiwah.corpus5x5-v1`, 25 contexts x 2,048), H200, container-
disk. Stage wall times from the FP8 run's `done` markers (`glm53-fp8`):

| stage | wall time | note |
|---|---:|---|
| `fetch_target` (750 GB checkpoint) | **9m 51s** | US-NC-1 pin, ~1.27 GB/s avg (docs/CLOUD-RECIPES.md) |
| `capture` (cold run 1, 78 layers, layer-outer) | **27m 30s** | FP8 candidate, block-dequant-to-bf16 weight source |
| `capture_repeat` (cold run 2) | **27m 27s** | matches cold run 1 to within 3 s |
| `compare_root` (self-compare) | 5m 30s | |
| `compare_reference` (vs BF16 root, numpy) | 5m 09s | see §3.4's M3 row |

The dated full run (setup through publish) was reported as **1h 41m**,
**≈ $6.45** on an H200. These are recorded run/billing observations, not
simply 1h41m × the stated $4.59/h tariff. Each cold run covered the whole
25-window panel in about 27.5 min. The old **$1.08–3.52** projected Stage C
range did **not** cover this full bill, and neither those prices nor
0.4–1.6 min/window should be quoted as a measured universal capture rate.

---

## 1. The parallelism decision and its unproved alternatives

| Form | Bit-identical? | I/O cost | Verdict |
|---|---|---|---|
| **Tensor parallel** (split a layer across GPUs) | not guaranteed; changes reduction order | — | Different lane unless qualified |
| **Window parallel** (each GPU takes some windows) | conditional on device/runtime equivalence | N x weights | Expensive for streamed models |
| **Pipeline parallel** (each GPU owns a layer slice) | proposed acceptance criterion, unproved | projected 1x weights | Design direction, not implemented here |

**Tensor parallel is not a transparent throughput substitution.** Splitting
layers changes reduction/expert-combine order. The byte-identical mirror
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` of
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` had recorded values
`0.025503427634363769` and `0.024554564249958208` on different runtimes.
That demonstrates a runtime-associated gap, not an isolated TP causal
experiment. A new TP capture must identify its lane; equal keys alone do
not permit ranking without `pair_predicate`.

**Window parallel duplicates weight reads.** Each device
needs every layer, so a 1.5 TB model streamed layer-by-layer becomes 12 TB of
reads across 8 devices. Fine for models that fit one device; wrong for exactly
the ones that forced layer-outer to exist.

**Pipeline parallel is a proposal requiring its own acceptance experiment.**
Each device would own a contiguous layer slice, materialize its weights once
per cold run and pass windows onward. Avoiding cross-device reductions helps,
but placement, kernel dispatch and device/runtime differences can still move
bits. Matching single-device content digests is the acceptance criterion,
not a theorem that this unimplemented schedule already satisfies.

### 1.1 Fetch/compute overlap is not automatic

Layer order permits an overlapped fetcher in principle; the engine-level race
experiment is a separate path. Ideal overlap approaches `max(fetch, compute)`,
but buffering, shard placement, contention and startup/drain costs remain.
Plain layer-outer does not make downloads overlap for free, and the paid
controller refuses race mode. The dated 430–600 MB/s rates are observations,
not a guaranteed large-model fetch rate on every provider/host.

## 2. Pricing (JarvisLabs, observed 2026-08-30, spot, IN2)

Per-GPU-hour and linear; 8 free of each at time of writing:

| GPU | VRAM | $/GPU-h | 8x |
|---|---:|---:|---:|
| H200 | 141 GB | 1.99 | 15.92 |
| H100 | 80 GB | 1.19 | 9.52 |
| RTX-PRO6000 (IN1) | 96 GB | 0.99 | 7.92 |
| L4 | 24 GB | 0.29 | 2.32 |

Single-device layer-outer reduces resident weights, but memory includes
embed/head, layer-load/dequant/fusion transients, activations and workspace.
The old "8x H100 beats 1x H200" assertion was an unmeasured projection for an
unimplemented pipeline schedule, not an admission or cost result. Historical
region-local storage and rates above do not establish current stock.

## 3. Cost model — include every capture and comparison

M1 (Qwen3.8-27B, 2026-08-30) and M1.5 (the replay fix, same day, same GPU class)
between them replaced every term in the old model. The old formula priced the
capture, anchored it to a streaming figure, and omitted the comparison entirely
— which was the dominant term. This section is what the measurements say.

```
wall_total = root_fetch + root_decode/cold_captures + root_qualification
           + sum(candidate_fetch + candidate_decode/cold_captures
                 + candidate_qualification + candidate_comparison)
           + setup + verified_retrieval + teardown
cost       = sum(each resource's billed time * its rate) + storage/transfer
```

This is an accounting checklist, not a calibrated predictor. Subtract overlap
only when measured for the actual schedule; include failed attempts and
cleanup reserves. Supplied sealed candidate datasets can remove their fetch/
capture terms, but the table must explicitly state that assumption.

### 3.1 Historical timing anchors and their scope

| term | value | measured on |
|---|---|---|
| `fetch_rate` | **627 MB/s** (52 GiB in 85 s, `HF_HUB_ENABLE_HF_TRANSFER=1`) | 28-vCPU IN1 box, M1 |
| `min_per_window`, model **resident** | **0.0109** (335.1 s / 512 windows, 27B) | 1x RTX PRO 6000, M1 |
| `min_per_window`, model **streamed** | 2.37-3.12 (GLM-5.3-Flash) | the old anchor — streaming lane ONLY |
| `C_replay`, numpy on CPU | **1.317e-12 s** per (position x vocab x hidden), both sides | 1x RTX PRO 6000 + 28 vCPU, M1.5 |
| `C_replay`, cuda fp32 | **1.30e-13 s** per (position x vocab x hidden), both sides | same box, same comparison, M1.5 |

**The capture anchor is regime-dependent and the two regimes are ~250x apart.**
2.37-3.12 min/window describes STREAMING a model that does not fit the device.
When the model fits, the forward pass is the only cost and the capture is nearly
free: 512 windows of a 27B model took 335.1 s. Use the streaming anchor only for
the streaming regime, which for us means GLM-5.3-Flash (642.7 GB) and larger.

### 3.2 The comparison was the whole bill, and it no longer is

M1 measured one 512-window comparison at **60 min 19 s** against a **335 s**
capture of the same panel on the same box — **10.8x the capture it consumes.**
The cause was one line: `dscompare._replay` reconstructed `logits' = hidden @
head.T` in **numpy on the CPU**, while the GPU — already holding that same head
for the fp64 KLD estimator — sat at 0%. (Watched live during the M1.5 baseline
run: `nvidia-smi` reports 0% utilisation for the entire numpy comparison.)

M1.5 moved it. `compare --replay-device cuda` runs the head matmul on the
device the estimator already uses, one position block at a time, so the full
`[positions x vocab]` fp32 logit array is never materialised on the host.

**Measured, same box, same 512-window force-computed root self-comparison:**

| path | wall | GPU | peak device memory |
|---|---:|---:|---:|
| `--replay-device numpy` (default, the published path) | **1,754.71 s (29 min 15 s)** | 0% | — |
| `--replay-device cuda` | **173.27 s** | 88% | **7.13 GB** (6.64 GiB) |

**10.13x** on that self-comparison: 512 contexts, 1,048,064 positions, vocab
248,320, hidden 5,120, identical root bytes on both sides. It saved 26.4 minutes
for this workload. It does not establish the ratio or absolute time for
nonzero candidates, other replay profiles, window sizes or hosts.

**One caveat on the baseline, stated rather than buried.** M1's own numpy
comparison took **3,619 s** on a different rental of the same GPU class; the
numpy path here took 1,754.71 s, 2.1x faster, for the same work. Nothing in the
code differs — the plausible causes are BLAS build and thread count (this box:
numpy 2.2.6 on scipy-openblas; notes say 28 vCPU but the JSON reports 256
logical CPUs without affinity/quota), page-cache state (the 13 GB dataset was
warm from the preceding CUDA run), and M1's note that two
concurrent comparisons contended. That spread is why the comparison above
used one rental and the same input data, with only replay flags changed.
The CLI ran as separate invocations, not one shared comparator process.
Quote 10.13x only with this self-comparison scope, not 20.9x.

Peak device memory is bounded and small: the fp32 head is 5.09 GB
(248,320 x 5,120 x 4 B) and everything else is one position block —
`--chunk-positions 128` gives 127 MB of fp32 logits and 254 MB of fp64 per side.
The 7.13 GB peak is ~7% of a 96 GB RTX PRO 6000, so a comparison fits alongside a
resident model in MEMORY. It no longer fits alongside one in COMPUTE — see §3.5.

### 3.3 It is opt-in, because it changes the last digits — by 5.24e-12 nats

An fp32 GEMM is not one function. BLAS accumulates in fp32 in an order the
implementation's blocking chooses, so `hidden @ head.T` has different last bits
on OpenBLAS, on Accelerate and on cuBLAS. The published Qwen3.8 rows are
16-significant-digit values of a quantity whose inputs carry that noise, so they
are already BLAS-bound and a backend change moves them.

**How much, measured** — `KLD(numpy-fp32-CPU replay || cuda-fp32 replay)` over
16 real root windows, 32,752 positions, the same estimator on both sides:

| quantity | value |
|---|---:|
| mean tokenwise KLD | **5.237e-12 nats** |
| p99 | 3.029e-11 nats |
| max | 1.791e-10 nats |
| **top-1 agreement** | **1.000000** — not one argmax flipped |
| max absolute logit delta | 3.624e-05 |
| max relative logit delta | 1.360e-06 |

This is a measured **replay-backend divergence on root data**. Its size
relative to old means is descriptive, not a bound on a nonzero candidate's
KLD change: KL has no additive triangle/error-budget law. The former
nine-significant-figure promise is withdrawn. Therefore:

- **the numpy path stays the default**, and the published rows stay reproducible
  on the machine and library that produced them;
- every receipt now names `comparator.replay_backend`
  (`numpy:cpu:float32` / `torch:cuda:float32` / `none` for a hash-proof
  short-circuit), because a silent backend swap is precisely the undeclared
  difference this format exists to stop;
- **keep replay policy fixed and disclosed for a comparison set.** Registry
  key equality alone does not bind every numerical axis; `pair_predicate`
  must permit ranking, and missing replay evidence is not inferred equality.

**THE FLOOR IS BACKEND-INDEPENDENT, and that was verified rather than argued.**
A self-compare replays bitwise-equal hidden states through one head on one
backend, so both sides get bitwise-equal logits and the KLD is exactly 0.0
whatever the backend rounds to. Re-run on the published root through the new
path: metric **exactly 0.0**, top-1 **exactly 1.000000**, every percentile 0.0,
and `tokenwise-kld.npy` sha256
**`8be5dccaf885d7dadca697c4d54cff60d1c8c8333b57761c31d882c9f9ec9e5d`** — the
published M1 floor digest, byte for byte, through a matmul that ran on a
different processor. `bin/selftest_replay_device.py` (T12) holds that line
offline as a gate.

`--replay-dtype float64` accumulates the replay in fp64 instead. It is more
accurate AND more reproducible across backends (fp64 reduction-order differences
are ~1e-16 relative rather than ~1e-6), but it is a DIFFERENT measurement from
either fp32 path and is offered as one, not as a better spelling of the same
number.

### 3.4 What this does to M2 and M3

Per-window comparison cost scales as `positions x vocab x hidden`. Against the
Qwen3.8 measurement (2,047 positions, vocab 248,320, hidden 5,120 = 1.0):

| rung | vocab | hidden | per-window vs Qwen3.8 | 512-window compare, cuda | 512-window compare, numpy |
|---|---:|---:|---:|---:|---:|
| M1 Qwen3.8-27B | 248,320 | 5,120 | 1.00 | **173 s** (measured, 512 windows x 2,047 positions cuda) | **1,755 s** (measured, same panel, numpy) |
| M2 GLM-5.3-Flash | 154,880 | 4,096 | 0.499 | ~86 s | ~876 s |
| M3 GLM-5.3 (full) | **154,880** (confirmed) | **6,144** (confirmed) | 0.748 | not measured | **309 s** (measured, numpy) |

**Vocabulary confirmed, not assumed** (2026-09-04): `zai-org/GLM-5.3`
(78L/6144, the full model — not Flash) declares `vocab_size: 154880` in its
own `config.json` at `187fb9ff…`, matching Flash's. The M3 row above is a
real `compare_reference` stage from the GLM-5.3 FP8 candidate measurement
(`glm53-fp8`, numpy:cpu:float32 backend, `shared_reference_head`,
`vocab_chunk=8192`, two-pass): 51,175 scored positions (25 contexts x 2,048,
`panel--glm53.malaiwah.corpus5x5-v1`, every causal position scored —
`windowed: false`), 309 s wall (16:40:12Z-16:45:21Z). That is **not** the
512-window profile the Qwen3.8/Flash rows use, so it is not directly
comparable column-for-column: per scored position it is ~6.04 ms, well above
Qwen3.8's measured numpy rate (~1.67 ms/position) despite GLM-5.3's lower
vocab x hidden product (0.748x). The gap is unexplained by the per-window
scaling model alone — record it as measured, not force-fit to the model; the
`--replay-device cuda` path (§3.3) was not exercised on this run and remains
the one to time next for an apples-to-apples M3 cuda figure.

The old conclusion that "the comparison term stops mattering" is not
established. The projected four-candidate Flash figures (~58 min numpy,
~6 min CUDA) extrapolate a Qwen root self-comparison; they do not measure
nonzero Flash candidate comparisons. Each candidate also needs its own
checkpoint fetch/decode/cold captures unless a qualified dataset is already
supplied. The later M3 observation above demonstrates that a simple
`positions × vocab × hidden` rate does not explain all real run times.

For reproducibility, pick and record the replay profile before comparing.
`comparator.replay_backend` records one part of that profile; it is not
itself one of the seven registry comparability-key fields. Equal keys are
necessary, not sufficient; the pairwise predicate and recorded provenance
must still support the proposed ranking.

**Anchors already paid for:** the Flash 4-rung ladder at $8.02 / ~$11.60 / $6.65
/ $5.41 per artifact; Fruit root+candidate at $0.25 total; the 0.1B fixture
end-to-end at $0.00; **M1 Qwen3.8-27B root (3 cold runs) + 2 candidates + 3
comparisons + publish at $5.12 total**; **M1.5 the replay fix, measured against
the published root on one on-demand RTX PRO 6000, at $1.27.**

### 3.5 Two operational notes the M1.5 run paid for

**A published fidelity dataset does not survive `snapshot_download` unedited.**
Fetching `malaiwah/qwen38-27b-fidelity-root-v1` into a `local_dir` adds
`.cache/huggingface/**` (1,550 files) and a `.gitattributes` the Hub inserts;
neither is in `checksums.txt`, and the SEAL-1(c) unlisted-file gate refuses the
whole comparison. Delete both after fetching, or pass `--allow-partial` and
accept `covers_full_panel: false`. Cost this run two wasted job launches.

**Captures and comparisons no longer trivially pack.** M1 learning 4 said
capture (GPU-bound) and comparison (CPU-bound) could run concurrently for free.
With `--replay-device cuda` the comparison is GPU-bound too, so they now
contend. Pack a comparison against a *fetch*, not against a capture — or leave
the comparison on the numpy path when a capture is running, which is also the
right choice for reproducing an existing group.

## 4. Families, ranked by value per dollar

| Family | Reference | Size | Geometry | Quant children | Note |
|---|---|---:|---|---:|---|
| **GLM-5.2** | `zai-org/GLM-5.2` | 1,506.7 GB | 78L/6144/256e | **100** | natively unquantized; no `-BF16` sibling |
| **GLM-5.3** | `zai-org/GLM-5.3-BF16` | 1,506.7 GB | 78L/6144/256e | 37 | **DONE (M3c, 2026-09-04/05)**: root on two H200 pods bitwise, six candidates measured (FP8, K4, three TR3, one stock EXL3), ≈ $80 including every failed attempt (RunPod balance $191.78 on 09-04 → $152.04 on 09-05 plus the root attempts before it; each figure is in the journal); see docs/HANDOFF-MEASUREMENT-SESSION.md §M3c |
| **GLM-5.3-Flash** | `-Flash-BF16` | 642.7 GB | 45L/4096/288e | 67 | **re-capture**: a NEW group with a 0.0 floor beside the existing eight rows, which it does not upgrade |
| **Qwen3.8-27B** | `Qwen/Qwen3.8-27B` | 55.6 GB | 64L/5120 dense, **hybrid attn + vision + MTP** | — | **DONE (M1, $5.12)**: new same-lane group, 37 old rows NOT upgraded — see below |
| Qwen3.5-397B | `Qwen3.5-397B-A17B` | 806.8 GB | — | GGUF/MLX/REAP | backfill; 1,553 likes on the root |

**GLM-5.2 is the best deal on the board.** Identical size and geometry to
GLM-5.3 — one engine, one panel design, one capture cost — but ~3x the quantized
children (unsloth GGUF 642 likes, nvidia NVFP4 319, lukealonso, 0xSero REAP).
Per dollar of root capture it unlocks the most downstream measurement.

**Re-capturing GLM-5.3-Flash was not redundant.** The older Flash rows use
a teacher from another stack and a control near **0.011506** nats. A qualified
same-stack root enables a new comparison set with a measured zero self-control;
it does not isolate a causal codec term by subtraction or change old rows.
The registry now records a Flash root; this section's proposed re-capture
and family statuses are historical, not a current root availability inventory.

**Qwen3.8-27B was a rounding error** at 55.6 GB — done for **$5.12** — but the
claim that it "retroactively upgrades 37 existing rows" was WRONG and is
withdrawn. A same-lane root does not upgrade a row measured against a different
teacher: the comparability key binds the reference, so the new rows form a NEW
group (`cmp--05e16411a5932713`) beside the old one (`cmp--4a93702ded23e01a`), and
the 37 older rows keep their inferred floors untouched. What a same-lane capture
buys is a *new* group with a measured zero self-control. Reusing the panel
removes one confounder, but the old/new AWQ rows name different artifact
identities; their **3.685e-4** descriptive difference is not an identified
lane term, and cross-stack perturbations have no guaranteed direction.
Flash's lost legacy captures cannot be rescored from their digests. Each
artifact would need a new qualified capture (or supplied surviving capture
bytes), not just the eight ~86-second comparisons previously budgeted here.

Note also that the geometry line above understated the model: Qwen3.8-27B is
`Qwen3_5ForConditionalGeneration` — multimodal, with 48 linear-attention layers
to 16 full-attention, and an MTP block. Dense (zero expert tensors). Check
architecture before sizing a rung from a one-line description.

## 5. Sequencing

1. **Single-device layer-outer** proves bit-identity against the current schedule
   on two small models. The correctness gate. *(in flight)*
2. **Pipeline-parallel** as a pure throughput change whose acceptance test is
   reproducing the single-device digests exactly. Fetch/compute overlap lands here.
3. **Cheapest root first** — Qwen3.8-27B *(**DONE** 2026-08-30, $5.12, floor
   measured at exactly 0.0; `docs/M1-QWEN38-ROOT-LEARNINGS.md`)*, then the
   GLM-5.3-Flash re-capture — to validate the cost model against small bills
   before a 1.5 TB run. M1's verdict on the model: the capture term was ~250x
   cheaper than projected and the comparison term, which the model omitted
   entirely, is the one that cost. *(**M1.5 DONE** 2026-08-30, $1.27: the
   comparison term is now 10.13x smaller and §3 is rewritten around what was
   measured rather than what was projected.)*

   Two things M2 must decide before it starts, both new since M1.5: **which
   replay backend the whole group uses** (§3.3 — it is part of the number), and
   whether the GLM-5.3 MTP layer's `--allow-unexpected-tensors` blocking
   disclosure is acceptable on the sealed root (it is required now; the capture
   refuses otherwise).
4. **GLM-5.2 and GLM-5.3** on the proven engine.
5. Qwen3.5-397B if the backfill still looks worth it.

## 6. What would change this plan

- Layer-outer failing bit-identity on `glm_moe_dsa` — then the schedule is wrong
  and nothing downstream is safe.
- Pipeline fill/drain costing more than the parallelism returns at 78 layers.
- Fetch rate materially below 430-600 MB/s, which would make every large capture
  fetch-bound and shift the answer back toward one device.
- Spot preemption on multi-hour runs; measured behaviour, not assumed.
