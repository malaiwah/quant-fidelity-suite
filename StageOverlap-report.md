# StageOverlap — concurrent stage pairs that leave the science byte-for-byte

Branch `StageOverlap`, worktree `/home/mbelleau/code/worktrees/StageOverlap`
(from `origin/main` @ `6b5236c`). Implements the two highest-ranked efficiency
overlaps from `local://EfficiencyReview-report.md` (§2a fetch_reference ∥
fetch_target, §2c compare_reference ∥ capture_repeat) plus the setup-overlap
analysis (§2g), with per-stage watchdog pgid records so two setsid leaders can
run and be reaped independently.

## 1. Stage dependency graph (candidate route, `bin/fidelity/stages.py` `CANDIDATE_STAGES`)

`stage_sequence(role="root", candidate=True)` returns:
`setup, fetch_target, fetch_reference, capture, verify, capture_repeat,
verify_repeat, compare_root, qualify_root, compare_reference`

Each stage's reads / writes / required marker (`bin/stage_measure.sh`):

| stage | requires (.done) | reads | writes | case lines |
|---|---|---|---|---|
| setup | — | job.json, official BF16 (zai-org/GLM-5.3-Flash-BF16) | `$ROOT/venv` (bootstrap install), `$FS/models/bf16` (config+index, gguf vision shards), `receipts/selftest-*.txt`, `receipts/pip-check.txt`, `receipts/*-build.txt`, `setup.done` | 419–583 |
| fetch_target | setup (venv: `$VENV/bin/hf`) | job.json target download_manifest | `$MODELS/target` (the weights), `receipts/fetch-target-census.json`, `receipts/shard-verification.*`, `receipts/artifact-*.json`, `fetch_target.done` | 584–894 |
| fetch_reference | fetch_target (today) | job.json `capture.candidate.reference`, the HF reference dataset (anonymous) | `$FS/reference-cache`, `$FS/reference` (symlink), `$FS/reference-model`, `receipts/reference-verify.json`, `fetch_reference.done` | 1618–1691 |
| capture | fetch_target | `$MODELS/target`, job.json | `$FS/dataset`, `receipts/dataset-…`, `capture.done` | ~1180–1503 |
| verify | capture | `$FS/dataset` | `receipts/dataset-verify.json`, `verify.done` | 1545–1568 |
| capture_repeat | verify (today, via controller order) | `$MODELS/target`, job.json | `$FS/dataset-repeat`, `receipts/dataset-repeat-…`, `capture_repeat.done` | ~1440–1503 |
| verify_repeat | capture_repeat | `$FS/dataset-repeat` | `receipts/dataset-repeat-verify.json`, `verify_repeat.done` | 1545–1568 |
| compare_root | verify, verify_repeat | `$FS/dataset`, `$FS/dataset-repeat` | `receipts/root-comparison/`, `compare_root.done` | 1556–1594 |
| qualify_root | verify, verify_repeat, compare_root | `$FS/dataset`, `$FS/dataset-repeat`, `receipts/*-verify.json`, `receipts/root-comparison/` | `receipts/root-qualification.json`, `qualify_root.done` | 1596–1616 |
| compare_reference | fetch_reference, qualify_root (today) | **`$FS/reference` + `$FS/dataset` only** | `receipts/reference-comparison/`, `compare_reference.done` | 1693–1755 |

### Why the two overlaps are data-clean

**(a) fetch_reference ∥ fetch_target.** `fetch_reference` (stage_measure.sh:1634–1688)
fetches the published root dataset + a handful of model-class files
**anonymously** (`env -u HF_TOKEN … HF_HUB_DISABLE_IMPLICIT_TOKEN=1`,
HF_HOME=`$FS/hf-anonymous`) and verifies its seal + tensors against the job's
`capture.candidate.reference`. It reads NO target bytes; its only dependency on
`fetch_target` today is the marker gate `require_stage_marker fetch_target`
(:1624), which exists for the token choreography ("the target token is gone by
now"), not for data. Writes: `$FS/reference-cache`, `$FS/reference`,
`$FS/reference-model`, `receipts/reference-verify.json` — disjoint from
fetch_target's `$MODELS/target` + `receipts/fetch-target-census.json`. The 2.4 GB
reference download is <1 % of the 465 GB–1.5 TB target and rides the same link.

**(b) compare_reference ∥ capture_repeat.** `compare_reference` (stage_measure.sh:1740–1747)
reads **`$FS/reference` and `$FS/dataset` only** (the candidate's canonical
capture), via `fidelity_dataset.py compare --reference $FS/reference --candidate
$FS/dataset`. `$FS/reference` is sealed by `fetch_reference`; `$FS/dataset` is
sealed by `verify` (the controller runs `verify` before `capture_repeat`).
`capture_repeat` writes `$FS/dataset-repeat` (a DIFFERENT tree) — no shared
read or write. `qualify_root` (stage_measure.sh:1596–1616) writes only
`receipts/root-qualification.json` + `qualify_root.done`; it reads
`$FS/dataset`, `$FS/dataset-repeat`, `receipts/*-verify.json`,
`receipts/root-comparison/` — **none of which compare_reference writes, and none
of which compare_reference reads**. So `qualify_root` writes nothing
`compare_reference` reads, and `compare_reference` writes nothing
`qualify_root` reads. The `qualify_root` marker today ORDERS ACCEPTANCE of
compare_reference (it `require_stage_marker qualify_root` at :1700), not its
computation — exactly the invariant the overlap preserves.

## 2. Design — pod-side composites, controller loop unchanged

The controller's `_bootstrap_and_run` loop (`bin/measure_cloud.py:8324–8346`)
iterates `stage_sequence(...)` in canonical order and appends each stage to
`stages_done`; `result_archive.py --stages` stores that list in the archive
summary (`resultsink.build_summary` :434) and the validator does NOT enforce
stage-list ORDER for the capture/measure verb (it checks membership-driven
gates like the target census, `resultsink._check_archive_target_census` :1978).
**The loop is left sequential; the concurrency is encapsulated inside
`stage_measure.sh` composite cases**, matching EfficiencyReview §2a ("pod-side
composite … the controller keeps driving one stage, `_runpod_fetch_target_and_remove_token`
is unchanged") — the lowest-risk cutover (the token window, the watchdog arming,
the retrieval tail and `stages_done`'s canonical order are all untouched).

- **fetch_target composite:** after arming the authenticated fetch, launch
  `stage_measure.sh fetch_reference` in the background as its own setsid leader
  (anonymous env, self-records `runtime/stage-fetch_reference.pgid`); run the
  target fetch + census; `wait` for the reference sibling; write `fetch_target.done`
  only after both succeed. `fetch_reference.done` is still written only by the
  reference sub-stage (:1689). The controller's later `fetch_reference` stage
  sees the marker and skips (the existing "already done" path, :221–227) —
  `stages_done` keeps canonical order.
- **capture_repeat composite:** launch `stage_measure.sh compare_reference` in
  the background (setsid, self-records `runtime/stage-compare_reference.pgid`);
  run capture_repeat; `wait` for compare_reference; write `capture_repeat.done`.
  compare_reference computes its receipt to a **pending** dir
  (`receipts/reference-comparison.pending`) and writes NO marker (its
  `write_marker` is conditional on `qualify_root.done`, per the brief) — so a
  run whose `qualify_root` refuses never accepts the comparison.
- **Acceptance ordering:** the `qualify_root` case, after writing
  `qualify_root.done`, promotes the pending comparison
  (`reference-comparison.pending` → `reference-comparison`) and writes
  `compare_reference.done`. If `qualify_root` refuses (exits non-zero before its
  `write_marker`), the pending dir is left in place and the controller discards
  it on the failure path (it is not under `receipts/done/`, so it is not a
  stage marker). The controller's later `compare_reference` stage then skips
  (marker present). `stages_done` lists compare_reference after qualify_root —
  canonical order regardless of wall-clock.

### Per-stage pgid records (watchdog + wrapper + self-record)

Today the pgid record is per-run (`runtime/stage.pgid`), cleared by the wrapper
before each launch (commit `cddcb69`). With two concurrent setsid leaders that
collides. Change to **per-stage** `runtime/stage-<name>.pgid`:
- `stage_measure.sh` self-record passes `runtime/stage-$STAGE.pgid` (the leader
  self-records as its first act, :29–42).
- `record_stage_pgid` derives the receipt name from the record path:
  `stage-<name>.pgid` → `receipts/watchdog-stage-pgid-<name>.json`;
  legacy `stage.pgid` → `receipts/watchdog-stage-pgid.json` (so the existing
  direct-call safety rungs are unchanged).
- the wrapper (`_runpod_stage_command`) clears + waits for
  `runtime/stage-<stage>.pgid`.
- the watchdog's `stop_work` enumerates ALL `runtime/stage-*.pgid` records (plus
  the explicit/legacy one) and signals each independently — so a deadline or
  stale-heartbeat stops every concurrent leader.

## 3. Setup overlap (§2g) — analysis + what moves

`setup` (stage_measure.sh:419–583) runs `bootstrap_measure.sh`, which already
has an INSTALL/CHECK split (`FIDELITY_BOOTSTRAP_INSTALL_ONLY`, bootstrap :437–456):
steps 1–4 install (python3.12, hashed wheels, pipeline clone+patches, exllamav3);
steps 5–6 CHECK (adapter import + the four offline selftests tr3/dione/dione-stream/exl3hf/gguf,
writing `receipts/selftest-*.txt`). The preflight bench is controller-side
(`_preflight_bench`, measure_cloud.py:9380), run after setup before fetch_target.

| part | target-independent? | can overlap fetch_target? | why |
|---|---|---|---|
| venv + hashed wheels (steps 1–2) | yes | **NO** | `fetch_target` calls `$VENV/bin/hf` (:617) — hard prerequisite; must finish first |
| pipeline clone + patches + exl3 build (steps 3–4) | yes | **NO** | same: the surface selftests and the capture import the pipeline; and exl3 build is part of the venv setup fetch_target's census does not need, but the selftests do |
| adapter import + offline selftests (steps 5–6) | yes | **YES (clean)** | writes only `receipts/selftest-*.txt` + `receipts/adapter-import.txt`; disjoint from fetch_target's `$MODELS/target` + `receipts/fetch-target-census.json`; GPU is idle during the network fetch, which is exactly when the gguf CUDA-decode parity rung wants the device |
| preflight bench (controller `_preflight_bench`) | yes | **YES (clean)** | writes `preflight-bench.json`; disjoint from fetch_target; fail-closed before capture (finishes in ~26 s, fetch is 5–23 min, so its result is known before capture) |
| GGUF official vision shards + inventory (setup :525–580) | uses official BF16, not the target | NO | part of setup's install phase; small and before the venv is fully ready |

**What moves:** the bundle selftests (bootstrap step 6) are moved to run
alongside fetch_target — the fetch_target composite launches a check-only
bootstrap (`FIDELITY_BOOTSTRAP_CHECK_ONLY=1`) as a background setsid child
(self-records `runtime/stage-setup_checks.pgid`) and `wait`s for it before
writing `fetch_target.done`, so the selftests gate `fetch_target.done` (and
thus capture) exactly as they gated `setup.done` today. No new stage marker; no
shared writes. The preflight bench is moved to run concurrently with
fetch_target (controller thread, see §4). The venv install (steps 1–4) stays in
setup before fetch_target — it is a hard prerequisite.

## 4. Expected minutes saved per candidate

| overlap | saved/candidate | evidence |
|---|---:|---|
| compare_reference ∥ capture_repeat | ~5.2 min | EfficiencyReview §1.2 row 7 (compare_reference 5.2 min) disappears from the critical path; capture_repeat (4.7–9 min) hides it |
| fetch_reference ∥ fetch_target | ~1–2 min | §1.2 row 3 (fetch_reference 2.2 min) hidden behind fetch_target (5.9–22.9 min) |
| setup selftests + bench ∥ fetch_target | ~1.3 min | §2g: selftests 53 s + bench 26 s, GPU idle during fetch |
