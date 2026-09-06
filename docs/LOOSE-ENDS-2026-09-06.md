# Loose ends, swept 2026-09-06 — what is closed, what is owned, what is a decision

A deliberate sweep of `docs/REVIEW-DEFERRED.md` (40 entries), every `TODO`-class
marker in `bin/`, `engines/tools/` and `container/`, and the open items from
[`SPEND-AUDIT-2026-09-06.md`](SPEND-AUDIT-2026-09-06.md). The point is that a
backlog nobody has enumerated is a backlog nobody can prioritise — and three of
these turned out to be cheap.

Every claim below names the file or the measurement it rests on. Where something
is unverified, it says so.

## Closed today, with evidence

| item | what it was | evidence |
|---|---|---|
| **DESC-01** | `load_panel_descriptor` raised `KeyError: 'repo_id'` on a file that is JSON but not a descriptor — the obvious mistake, since a panel *directory* contains a file literally called `panel.json`. Five keys were indexed directly. | Named `HFError` for a non-dict payload, any missing required key, and a present-but-non-integer count. Rung `DESC-01` in `selftest_shell_guards.sh` covers all four inputs **plus** the negative case that a valid descriptor still loads. Verified failing against the pre-fix loader; 29 passed / 0 failed / 0 skipped. |
| **DECODE-PARITY-01** | Section [2] asserted `torch.equal(cpu, cuda)`: red on every CUDA device ever pointed at it (sm_75 **and** sm_80, identical `max_abs_diff`), vacuously green on the CPU-only boxes that run the battery. Green for the life of the tree, having never tested what it names. | Now bounds the axis it measures: `DEVICE_PARITY_MAX_ABS_DIFF = 5.0e-05`, above the measured reduction-**order** axis (9.537e-06 sm_75, 6.676e-06 sm_80) and below the rounding-**count** axis vs exllamav3 (6.1e-05–2.4e-04), so a regression crossing into the other axis fails rather than being absorbed. Bitwise is reported, never asserted. |
| **The 8th skip format** | `(no accelerator on this machine; parity is vacuous, skipping)` was prose no skip pattern matched, so the battery counted it as *no skip at all*. | Fixed at the **emission** site, not by widening a regex: a canonical `SKIP` marker naming the missing dependency, caught by `SKIP_RE` and pinned as format 8 in `selftest_battery_harness.py`. 8 of 8 formats, 0 false positives on summary lines. |
| **MKL-01** | `selftest_fidelity_reducer.py` died with `rc=132` (SIGILL) on ~15% of runs. Not our code: `mkl_vml_kernel` in `libtorch_cpu.so`, on a Xeon X5570 (Nehalem, SSE4.2, **no AVX**). | Contemporaneous controls: control **9/60**, `MKL_NUM_THREADS=1` **0/60**, `OMP+MKL=1` 0/20, `MKL_ENABLE_INSTRUCTIONS=SSE4_2` 2/20 against a 1–3/20 control. Threading is the discriminator; the ISA cap is not the fix. Scoped `env MKL_NUM_THREADS=1` on the one rung that manifests it. |
| **REAP-1 coverage** | The reaper's sweep loop once reported success when every destroy failed. The code is fixed and confirms each destroy, but the **sweep's own loop** had no rung — only the in-run `Teardown` path did (`selftest_teardown` CLI-01a/c/d, 32/0). | Two **preservation** rungs added to `selftest_reaper.py`: a provider whose `destroy` raises must not be reported as success and must keep the lease, and no absence proof may be sealed for a machine that was never destroyed. Labelled as preservation rungs — they pass on both sides of no diff and are not presented as having caught anything. |

## Checked and deliberately NOT changed

- **`bin/engines.json` `minutes_per_window: 20.0`** for the sealed EP8 lane is
  an unmeasured estimate — and it already says so (`"measured": false`, with a
  provenance string explaining it errs high so the cost estimate errs high).
  No receipt on this machine carries per-window wall clock, so there is nothing
  to replace it *with*. **Replacing an honest labelled placeholder with a guess
  would be strictly worse.** Closing it needs a sealed-lane run receipt.
- **`bin/fidelity/cloudlease.py:1856`** — the paid-admission lock file is still
  named for RunPod, and the comment says it MUST stay that way *for now*. It is
  one **global** lock, not a per-provider one, and a live paid controller holds
  it under that exact name; renaming it mid-flight would let a second paid run
  admit. Correct as it stands; the rename is a maintenance-window job.
- **`bin/fidelity/stackprint.py:562`** — a `JOURNAL 2026-08-29 TODO` about env
  pins the sealed lane never recorded. The fingerprint **says** it does not have
  them rather than inventing them, which is the right failure mode.

## Open, and whose they are

**The registry session's** (evidence complete and in-repo; ingestion is the only
missing step):

1. Three GLM-5.2 quant measurements published, re-verified after publish, and
   with no row: jpsequeira 3.40bpw, willfalco 3.42bpw, brandonmusic
   TR3v4-3.5bpw-MTP78. Beware the naming trap — `glm53` means GLM-5.3-**Flash**
   in registry ids, so grepping a bpw string finds the wrong row. Key on model +
   owner.
2. `flashA-k2-run1`'s K2 number (0.15429493207672532 nats, exact 0.0
   reproduction confirmation). Its dataset is structurally unpublishable
   (`destination_repository: null` sealed inside the qualification) and
   publication is optional for a candidate, so a row is the whole job.
3. `DecoderParity`'s bitwise result retires the `weights_reconstructed` caveat
   those rows carry — ingestion and caveat retirement should be decided
   together.
4. Identity collision needing a ruling: two repos publish `dataset.id`
   `fidelity--fruit.malaiwah.root.bf16` at different `dataset_sha256`, and
   registry ids are hashed into `comparability.key`.
5. One capture digest carries **three** different disclosure verdicts across
   three repos publishing identical bytes.

**The operator's** (each edits state a tool refuses to edit unattended):

6. Release the two `AMBIGUOUS` leases holding a phantom **$8 + $35** of hard-cap
   reservation. Both spent **$0**; no orphan exists; absence confirmed twice by
   read-only listing. Not lost money — but `limit = min(ceiling, settled +
   available)`, so it mis-prices every future ceiling check.
7. `malaiwah/GLM-5.3-Flash-TR3-6bpw`'s card says `fidelity_dataset: null` while
   its dataset was published. `bin/fidelity-card annotate` cross-checks against
   a registry row, so it runs **after** ingestion, not before.
8. Rotate the `mbelleau-buildbox` token. Hygiene, not incident response — see
   below.

**Still deferred, with the reason** (not silently dropped):

9. **MKL-01's general exposure.** Which of the 21 `torch`-tier suites can take
   the same SIGILL, and whether post-AVX hardware (the container, every rented
   CUDA box) is immune by construction, is **unanswered**. The operational
   hazard is the dangerous part: an intermittent SIGILL makes the battery
   intermittently red for a reason unrelated to the code. **Anyone who sees
   `rc=132` should check the CPU before reading the diff.**
10. `STAT-005` — 11 of 95 rows carry `top1_agreement: null`. A scorer change, no
    GPU. Five are `clean17` recomputes whose receipts retain per-window mean KLD
    only; two are author-reported and must keep the warning, because inventing a
    third party's top-1 is not available to us.
11. `FLOOR-003` — 9 ranked groups without a floor. Five are blocked on the Qwen
    `float32_reduce_legacy` estimator question, which is a decision, not a
    measurement.
12. The remaining `REVIEW-DEFERRED` entries — `CLI-21`, `STAT-01/17`, `REAP-2/3`,
    `DEP-01..05`, `PANEL-D6` (three instances), `ROOT-2`, `CC-01`. Each carries
    its own anchor and reasoning in that file.

## The security question, closed

**There is no TLS interceptor on Vast machine 68004.** The cause is on-path
forged UDP DNS injection: one A query for `huggingface.co` returns three
replies, two forged at ~31 ms with a fresh random third-party address each
round, and the genuine CloudFront set losing the race at ~198 ms. The
"certificate hostname mismatch" was **Meta's own valid certificate** on a real
Facebook server our client reached only because DNS lied.

Scope refutes both the benign and the malicious reading: Wikipedia and Google
are poisoned while every HF weight CDN is clean, and `cdn-lfs.huggingface.co` —
which has **NODATA upstream** — was handed a Dropbox address. Nothing that
caches, proxies or forwards can invent records for a name that has none. The
container's CA bundle is byte-identical to the GHCR blobs with all 13 layer
digests verified.

**No credential leaked; the protection worked by refusing.** Both "a middlebox
intercepts TLS on this host" and "this host harvests credentials" are explicitly
refused as unmeasured — a refusal string is where an unmeasured cause does the
most damage, because it is the one place an operator reads a verdict.
