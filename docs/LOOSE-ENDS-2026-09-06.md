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

## Closed 2026-09-07, second pass — and the pattern is the finding

**Four of the six entries picked up on the second pass were ALREADY FIXED in
code and simply had no caller, no rung, or a stale note.** That is now the
most reliable prediction about this backlog: before writing a fix, grep for
the guard. The work has usually been done; what is missing is the wiring or
the evidence that it stayed done.

| item | state found | what was actually needed |
|---|---|---|
| **ROOT-2** | `census.root_fit` existed and was correct, with **zero callers** in `bin/measure_cloud.py` — so the planner that *spends money* still sized every root against a constant **63 GB/GPU**. `measure_local.py` had already fixed it, with a comment naming this exact failure. | Wired it. Measured on the real Fruit config: **8.47 GB/GPU, was 63** — a 7.4× over-requirement that refused a 10.10 GB checkpoint on every card under 63 GB during the GH200 qualification, including an A100 with free capacity. Unknown geometry and an unreadable window count now REFUSE instead of falling back to another model's census. Two of the four rungs assert **the caller exists**, verified failing pre-wiring. |
| **DEP-01** | `vastapi._req` retried 429 and nothing else, so a 502/503/504 or a connection reset raised hard mid-run *after the lease was written* — the same shape as the incident the 429 handling exists to prevent. | Extended **asymmetrically**: transients and connection faults retry on **GET only**; 429 retries on any method. A 5xx on `PUT /asks/{id}/` may mean the instance exists and is billing, so retrying would **double-rent** — that case belongs to the lease store's `LOST_CREATE_RESPONSE` reconciliation. Six rungs, verified failing at `calls=1`; three assert the *refusal* to retry. |
| **REAP-2** | Fixed in code and **stronger than the entry asked** — the plausibility bound applies to lease *and* name deadlines, and the name path can no longer produce a destroy target at all. No rung covered any of it. | Eight rungs driving the real `reaper_sweep`. One corrected my own expectation: the bound fires **first**, so a 1970 name is ignored by name and never reaches the "verify it yourself" branch — both branches are now covered separately, plus the original reproducer. |
| **MKL-01's open question** | Left deferred: "which of the 21 torch-tier suites are exposed". | Answered by isolating the **operator**, not re-running suites (the expensive version was going to take hours). `torch.logsumexp` **5/12**; `log_softmax`, `log`, `exp`, `sum`, `logaddexp` and fp32 matmul **0/6–0/8**; `logsumexp` with `MKL_NUM_THREADS=1` **0/12**. The op appears in **exactly one file** — the guarded one — and the production scorers use `log_softmax`. Now *enforced* by a containment rung, verified able to refuse. |
| **DEP-04** | Already fixed. Liveness comes from the wrapper's own pid, with the reasoning recorded at the code: *"pgrep was tried and does not work here, because the obvious fix does not work either"* — `pgrep -f r_1788…` and `pgrep -f '[r]_1788…'` both self-match. | Nothing. Verified and marked. |
| **STAT-01/17** | Already closed 2026-08-30 by the analytic `delta_t_log` interval, which **retires** the resample stream rather than mitigating it. | Nothing. Verified and marked. |
| **CC-01** (published) | The entry's last line said *"The Hub card still carries the wrong sentence."* | **It does not.** Fetched the live card: the retracted pair `1.73e-3`/`1.22e-3` is absent, all four recomputed values are live, and the methodology paragraph is **byte-identical** to the repo copy (paragraph sha256 `c3f55226d7837c15…` both sides). The 6bpw card carries neither. **No publish was required and none was made.** |
| **DEP-03** | `ControlMaster` reached the JarvisLabs box by hand in `~/.ssh/config` and never reached the shared transport, where it serves RunPod, Vast and Lambda — one full handshake per exec *and per scp*, with a delta uploader that sends one file per scp. | Three flags **appended**. The entry's own drafted patch predates the host-key work and shows `StrictHostKeyChecking=no` with `UserKnownHostsFile=/dev/null`; applying it verbatim would have undone the pinning that makes a measurement attributable to the machine we rented. There is now a rung asserting multiplexing did not weaken authentication — the one that would have caught it. |
| **PANEL-D6** (3rd instance) | Comparing a fresh Fruit capture against the published root was refused over a *name-only* tokenizer difference, with a remedy ("recapture on the reference's panel") that was useless because the panel was already identical. | The refusal now names `--tokenizer-id <the reference's own value>` — `dscompare` holds both sides and can read it. **Not** offered when `vocab_size`/flags/`revision` disagree, because that would turn an honest refusal into a footgun. **The engine-side half stays open on purpose:** it changes the identity string sealed into a dataset, i.e. what compares equal across published rows, so it is a comparability decision for the operator and the registry session. |

## Closed 2026-09-07, third pass — the money ones

Picked by *what can burn a rental*, not by what was cheapest. **Status index in
`REVIEW-DEFERRED.md`: 19 closed, 14 open.**

| item | state found | what was actually needed |
|---|---|---|
| **SH-10** | **Genuinely open, and the expensive one.** The `["*"]` default applies only when the key is ABSENT, so an explicit `"include": []` emitted an empty argv and `hf download` ran **unscoped** — on the panel repo, where the Flash teacher-logits dataset is **1,318 GB**. The comment on the target path already calls include-scoping "the difference between 32 GB and 1.3 TB"; the panel path had no guard in either branch. | Refused in the heredoc, plus a shell-level `-gt 0` guard mirroring `fetch_target`'s. `*` is **not** treated as protective (it fetches the same 1,318 GB) — the stage now warns before paying. **And a defect the entry did not name: a STRING include was iterated per CHARACTER**, so `"windows/*"` became `--include w --include i --include n …`, nine globs matching nothing, i.e. a panel fetch that silently downloads nothing. Six rungs that extract the heredoc from the shell script **verbatim** and execute it; four verified failing pre-fix. |
| **CLI-17** | Code already fixed, and **structurally untestable**: the four launch statements were inline in `main()` behind the whole job-contract validation, so no test could reach them. | Extracted as `spawn_streaming(argv, env)` *so the behaviour could be asserted*. T23 launches a chatty child that sleeps 6 s and reads the log at 2 s — stdout **and** stderr must be on disk before the child exits. **Verified non-vacuous** by substituting a capturing launch and watching both rungs go red. Why it matters: `common.run` hands output back after exit, so a 79-minute capture wrote one line (the argv), and `measure_cloud` tells the operator to `tail -50 <fs>/logs/*.log`. |
| **REAP-3** | Code already fixed — the retirement scan is hoisted out of `if not dry:` and only the `unlink` is guarded. A grep for `WOULD retire` matched **only the implementation**. | Three rungs driving the real `reaper_sweep`: the dry run names the lease it would retire instead of printing "nothing expired", the dry run does not delete it, and the real run retires exactly what the preview named. |
| **SEC-09** | Already fixed. | Verified and marked: `common.write_secret_file` uses `O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW` at 0600 inside a 0700 directory, so the 20.5 µs world-readable window is gone **by construction**. `O_EXCL` is stronger than the trailing `chmod` the entry insisted on, because it never opens an existing inode at all. |

**The pattern held for a third pass: two of four were already fixed and
uncovered.** And a new sub-shape appeared worth naming — *code that is correct
but structurally untestable*. CLI-17 had no rung not because nobody thought of
it but because the property lived in four statements no test could reach.
Extracting them was the fix; the assertion was the easy part.


## Closed 2026-09-07, fourth pass — capability we had already paid to build

**Status index in `REVIEW-DEFERRED.md`: 22 closed, 11 open.**

| item | state found | what was actually needed |
|---|---|---|
| **CC-08** | GGUF and NVFP4 already resolved; **MLX did not**, so a release with a bitwise-verified reader (`mlx_surface.py`, against mlx.core) was refused as *"no recognised surface marker"* — a verdict that sends the operator hunting a missing file when the true answer is "recognised, and no lane declares it yet". | MLX resolves on MLX's **own config shape** — a top-level `quantization` dict with `group_size` and `bits`, which is what `mlx_surface.py` derives per-tensor rates against — not a filename. Checked against all five committed nvfp4 evidence configs: no collision, all four modelopt releases still resolve to `nvfp4`. The block had to move **after** the `quantization_config` block, because an MLX release carries one too and `_apply_quant_config` reset the codec to unknown; measured, not guessed, by a rung that asserted `mlx-affine` and got `unknown`. |
| **CLI-16** | The `KeyError` half went with DESC-01. The arithmetic half was open: `scored_positions` was read verbatim, so `25 x 2047 = 999999` planned happily. | An **upper bound, not an equality** — a shard or subset panel legitimately scores FEWER positions and must not be refused, while scoring MORE than the grid holds is arithmetically impossible. Severity kept low as the entry assessed it: `scored_positions` feeds no cost term and `seal_receipt` already refuses with SCOPE-007, so this is defence in depth where the value *enters* the tree. |
| **SH-05** | Already fixed. | Verified and marked: armed with `nohup setsid` — the 65-minute H200 lesson from `945255b`, which this entry correctly said had reached the *stage* launch and not the watchdog — and `verify-watchdog` runs before `watchdog_armed` is set. |

**Two things this pass declined to do, both recorded rather than silently
skipped.** The NVFP4/RedHat line in CC-08's repro is *imprecise, not
unfixed*: that config declares `format: mixed-precision` with a 4-bit
`tensor_group` group **and** an 8-bit `block` group, so resolving it to
`nvfp4` would claim a uniform rate for a mixed checkpoint — exactly what
AGENTS.md forbids. `unknown` is the correct verdict there. And the half of
CC-08 that actually unlocks the capability — **declaring `mlx`/`nvfp4` on a
lane in `bin/engines.json`** — asserts that the lane's authored entrypoint
really accepts the surface, which needs `--probe-engines` evidence rather
than a guess. That is the operator's call and the next concrete step.

## Closed 2026-09-07, fifth pass — the backlog reads 33 closed / 0 open

`docs/REVIEW-DEFERRED.md`'s status index is now **33 closed, 0 open**. That is
not "the tree has no defects"; it means every entry in that file has a
disposition backed by evidence, and the ones that are decisions rather than
defects say so and name who decides.

| item | state found | what was actually needed |
|---|---|---|
| **CLI-22 / SEC-03** | `_resolve` read a token **unconditionally**, so every compare and verify of a PUBLIC dataset sent a credential to a host that does not need one. | Anonymous first; token only when the anonymous read is refused. Two design points: an anonymous read is **evidence** (it proves a published dataset is publicly readable, the property a third party reproducing a row depends on), and the fallback keys on the **status** — only 401/403, so a 404 or a network fault cannot escalate and a typo in a repo name cannot send the token anywhere. Three rungs, all verified failing pre-fix. |
| **CLI-28** | The HubError branch existed; a v1-format failure and an unreadable path — the two most likely first-run errors — still printed a stack. | Both refuse with their own code. **A real defect still tracebacks**, which is the point: swallowing everything in a catch-all turns a bug into an unexplained refusal. Verified pre-fix, where the suite *dies* with the un-diagnosed `FormatError`. |
| **NUM-16** | `bf16-floor` mapped `decode_cache` and not `decode_cache_dir`, advertising a configuration its own entrypoint refuses. | Mapped, with the flag confirmed by **probing** `stream_score.py --help` rather than read from a doc. Second half left open as a **design question** — whether a `job.json` should steer `ep_emulate`/`device`/`inventory` at all, or whether the authored profile is the authority. |
| **SH-22** | Two halves, and the entry's own analysis refutes one of them. | The resume-on-existence half is **unchanged**, because `capture-receipt.json` is written once after sealing and invalid JSON aborts the stage before `seal`. The swallowed-digest half was real: `\|\| true` left an **empty** `RECEIPT.sha256`, which is worse than a missing one because it looks like evidence. Now refuses. |
| **DEP-05** | Two `GET /instance-types` in one `create`. | Hoisted — and the hazard the entry did not name is the better reason: **two reads can disagree**, so the disk check and the capacity check could pass against different catalogue snapshots while neither describes the instance about to launch. |
| **DEP-02** | The inline comment was already excellent; module-level visibility was missing. | Docstring note, inline comment **unchanged** as the entry instructed. |
| **SEC-01, CLI-01, CLI-11/SEC-08, CC-07, ROOT-1** | All already fixed with real regression suites; markers read "APPLIED" so the index did not count them. | Verified and normalised. `ROOT-1` confirmed closed by reading `resultsink._relevant(include_datasets=…)`. |

**Final tally of the pattern across five passes: of the entries examined,
most were already fixed and missing a caller, a rung, or a marker.** The
backlog's real failure mode was never unfixed defects — it was dispositions
living in commit messages and code comments instead of in the file that exists
to hold them. Hence the status index, and hence the rule at the top of it.

## The three open decisions, resolved by the operator 2026-09-07

All three were held back deliberately because each changes something durable.
The operator took the recommended option in every case. **Status index: 33
closed, 0 open.**

| decision | ruling | what shipped |
|---|---|---|
| **CC-08** — declare `mlx`/`nvfp4` on a lane? | **Probe first, then declare only what the probe confirms.** | `stream_score.py --help` lists both in **`--source` and `--profile`**, and the engine enforces the pairing itself, so `mlx -> mlx` / `nvfp4 -> nvfp4` are the engine's rule and not an authored guess. Declared on `streaming` with wildcard profile maps (the rate is not a bits table, same as `gguf`). Safe on identity: the lane **name** enters a comparability key and did not change, so no published row shifts. New rung scrapes both choice lists from the engine's argparse **by AST** and refuses a surface with no profile map or a profile the engine will not accept. Honest caveat on the record: the readers are bitwise-verified, but **neither surface has been measured end to end on this lane** — the first run of each is new ground. |
| **NUM-16** — who owns the unfilled knobs? | **The authored profile is the authority.** | `engines.json` declares `profile_authoritative_flags` (eleven keys) and `invoke_engine` **refuses** a job whose `runtime` names one. The point of refusing rather than deleting the advertising: the old behaviour was to **silently ignore** such a key, which is the worst of the three options, because the operator believes the value took effect and the receipt cannot show it did not. Narrow and checked — no `job.json` in this tree carries any of the eleven — and the rung asserts **both** directions, since a guard that refused everything would pass a one-sided test while being useless. |
| **PANEL-D6** — whose tokenizer id does a capture record? | **Prefer the panel's own declaration.** | Precedence is `--tokenizer-id` -> panel receipt `tokenizer.id` -> `--weights-repository` -> `--model`, with a verified `--panel-binding-evidence` still above all four because a binding is checked against real bytes. Measured on the committed panel tree, which is what makes it a fix rather than a theory: `panel--fruit.malaiwah.heldout-v1` declares exactly `glm-5.2-siq-fruit`, the id the **published** root records, so a fresh Fruit capture now compares to it **with no flag**. Panels declaring nothing keep the previous default exactly. Recorded additively as `PUBLISHED-CORRECTIONS.md` §5, because it changes what compares equal going forward while rewriting nothing published. |

**One option was explicitly rejected and the reason is worth keeping:** for
PANEL-D6, treating a legacy or path-valued tokenizer id as *unknown* rather
than as a mismatch would have been looser and easier. It was refused because
"unknown" would mean the panel gate can no longer tell you two captures used
different tokenizers — **the one thing the token-id digest cannot see, because
it hashes integers.** Making a gate quieter is not the same as making it
right.

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
