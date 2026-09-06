# Spend audit — did every dollar produce an admitted measurement?

**Status:** audit of record, 2026-09-06. Read-only: nothing was created, destroyed,
published or released to produce it. Four independent lanes reconciled
**spend → receipt → published dataset → registry row** across every lease, run
directory, git worktree, the Hugging Face Hub and `registry/`.

Lane reports: `local://SpendLedger-report.md`, `local://ReceiptInventory-report.md`,
`local://HubPublished-report.md`, `local://RegistryGap-report.md`.

Every figure below names the file it was read from. Cost figures distinguish
**settled** (provider billing evidence) from **quoted** and from **estimated**
(rate × wall clock). A quote is never presented as a settlement.

## The answer

**On RunPod, no dollar is missing.** $106.860975 settled across 115 created
instances, and **every instance that was created has an absence proof** —
`EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING` on 115 of 156 leases, which is exactly
the set that created a resource. **No leaked instance exists.** 39 attempts were
cancelled before create at $0.

Spend that bought **no sealed capture: $0.245958**, three pods:

| pod | target | settled |
|---|---|---:|
| `on4ewsz2pk0d45` | `zai-org/GLM-5.3-Flash-BF16` | $0.219970 |
| `14hfhmvb067vln` | `malaiwah/GLM-5.2-SIQ-Fruit-bf16` | $0.022373 |
| `hb0xbd0g4oetju` | `zai-org/GLM-5.2` | $0.003615 |

The 12 further pods with no workload are the **controller-loss drills**, which are
designed to lose their controller; their $1.136926 bought teardown-under-
controller-loss evidence, not measurements. Source: `~/.fidelity-cloud/leases-v2/`
(156 lease documents) reconciled against 132 auto-ledger + 19 glm53-campaign + 5
`campaign.json` attempts.

Vast: 7 contracts, **~$0.26 estimated** — Vast exposes no settled per-contract
cost after destroy, so every Vast figure is rate × wall clock. Credit moved
$19.481282 → $18.796256 across two lanes jointly; it must not be attributed to
either alone. All contracts destroyed, `list_instances()` empty, each id
individually confirmed absent. JarvisLabs' 137 settled leases are asserted in code
and docs but **their lease store is not on this disk** — reported unknown.

Hub integrity, verified against the served bytes rather than asserted:
**33 of 33** sealed manifests reproduce `dataset_sha256` via
`bin/fidelity/dsformat.recompute_seal`; **2,499 of 2,499** `checksums.txt` entries
match the Hub's own LFS oid; **0** listed-but-absent files across 3,963; 4,089
files / 74.53 GB. **No published number disagrees with its bytes.**

## The one unrecoverable loss

**The MiniMax-M3 root.** Captured **twice, bitwise-identically**,
`capture_content_digest`
`f84237e4b3c4a50a9c3aee9f271573fb11c279f68b16b88e508d398aae822276`, and destroyed
at teardown **before it was published**.

Four independent negative checks: the digest is absent from all 31 published
`capture_content_digest` values under `malaiwah`; no `malaiwah` dataset or model
matches `minimax`; it appears in **none** of the 230 local `fidelity-dataset.json`
manifests; and a grep of the whole repository returns exactly one line —
`docs/HANDOFF-MEASUREMENT-SESSION.md:236`. The money bought a result that now
exists only as 64 hex characters in prose: no manifest, no panel binding, no
comparison receipt, nothing any tool here can consume.

**Why it is worth reading carefully:** teardown was the only step that could
destroy evidence while every other signal read green. Capture passed, verification
passed twice bitwise, and the instance tore down cleanly — which the 115-of-115
absence-proof result shows is enforced hard. **Teardown did its job perfectly and
took the measurement with it, because nothing in the ordering required the upload
first. Not a broken component; a correct component in the wrong sequence.**

The hole is closed: `--publish-root-to` uploads before teardown can run, and
teardown now refuses to destroy a verified-but-unpublished root. The residual
exposure is named precisely — the `qualified-unpublished` status token exists on
two runs and **nothing downstream joins on it**.

## Recoverable today, with no spend

**Three GLM-5.2 quant measurements are captured, compared, sealed, published to the
Hub and re-verified after publish — and have no registry row.** All three were
published between 07:20 and 11:44 UTC on 2026-09-06 against the admitted GLM-5.2
BF16 root (`a544e029a0392c2ae633715b0076ca040821128088fc0025a36de342fa4c0a78`).

| measurement | KLD (nats) | top-1 | dataset |
|---|---:|---:|---|
| jpsequeira 3.40bpw | 0.05989490186712827 | 0.9320371275036639 | `malaiwah/glm52-fidelity-exl3-tr3-3.40bpw-jpsequeira-v1` |
| willfalco 3.42bpw | 0.06285295180178992 | 0.9302589154860772 | `malaiwah/glm52-fidelity-exl3-tr3-3.42bpw-willfalco-v1` |
| brandonmusic TR3v4-3.5bpw-MTP78 | 0.05988025486939588 | 0.9336590131900342 | `malaiwah/glm52-fidelity-exl3-tr3v4-3.5bpw-mtp78-brandonmusic-v1` |

Evidence is in-repo under `registry/protocol/glm-5.2/` as `comparison.*`,
`publish.*`, `dataset.*` and `reproduction.*`. **Ingestion is the only missing
step** — this is the registry session's work, not a re-measurement. Each carries a
`weights_reconstructed` caveat (`affects_comparability: true`) and jpsequeira also
carries `activation_quantization_not_captured`; those are disclosures to carry onto
the rows, not reasons to withhold them.

**A naming trap that will cause a false "already landed" verdict:** in registry ids
`glm53` means GLM-5.3-**Flash**, and there IS an admitted
`measurement--glm-5.3.exl3-tr3-3.42bpw-davidsyoung.corpus5x5-v1`. A filing session
that greps `3.42bpw` gets a hit and wrongly concludes willfalco's GLM-5.2 3.42bpw
is landed. **Key on model + owner, never on the bpw string.**

`DecoderParity`'s bitwise result (`pre_hadamard` identical on all 15 modules, 0 of
115,343,360 elements differing, against exllamav3 1.4.2) retires the
`weights_reconstructed` caveat these three rows carry, so the ingestion and the
caveat retirement should be decided together.

## `flashA-k2-run1` is not a loss — the scary reading was wrong

The first pass classified it as a verified-but-unpublished capture, i.e. exactly
the MiniMax failure repeating. It is not.

The measurement exists and is admissible: **0.15429493207672532 nats**, top-1
**0.8711675622862726**, with an exact **0.0** reproduction confirmation,
`local-verify.json` `structural_status: sealed` and 0 errors, 83/83 checksums
verifying today. What cannot happen is *dataset publication*: the sealed
`root-qualification.json` carries `destination_repository: null`, so **no `--repo`
can ever satisfy the publish gate** — the run was created without
`--publish-root-to`, and the destination is sealed inside the qualification
receipt on purpose. Publication is **optional** for a candidate capture; only a
root requires it. So the gap is a registry row, not a lost number, and the gate
was not routed around.

Both signatures — sealed `destination_repository: null` and `job.json`
`publication_preflight: null` — coincide on exactly the two `qualified-unpublished`
runs and no others.

## A phantom $43 that is not lost money

Two `AMBIGUOUS` leases hold `quote.hard_cap_usd` reservations of **$8** and **$35**
open indefinitely with `billing_reconciliation: null` and `released: false`. **Both
actually spent $0:**

- `a839fcd8…` — RunPod GraphQL `INTERNAL_SERVER_ERROR` on create; no pod of its own
  was ever created. Target `malaiwah/GLM-5.2-SIQ-Fruit-fp8`.
- `f323c9bb…` — a diagnostic that logged *"DEBUG-GGUF: submit stubbed — no pod is
  created in this diagnostic run"*. Target `unsloth/GLM-5.3-GGUF`.

Neither has an orphan: the wrong-name pods each later observed
(`rix7coja1riec8`, `pf8xb0l655ikw8`) are separately reconciled under other leases,
and neither exists in the account inventory, confirmed twice by read-only listing.

**Why it still matters:** ceiling arithmetic is `limit = min(ceiling, settled +
available)`, so a phantom reservation mis-prices every future admission check. It
also turns a battery rung red **on this workstation only** — the reaper exits 90
with `health: not healthy` rather than the expected refusal code 3, because it
cannot settle a lease with no instance id. That rung's colour is a property of the
operator's account, not of the code, and it needs a scratch `--lease-dir` fixture
or a SKIP naming the blocking lease.

Releasing them requires operator confirmation and is deliberately **not** done
here: it edits spend records.

## Value that is being earned

- The **GLM-5.2 BF16 root** capture is the reference for six admitted rows *and*
  for the three unbooked ones above — paid once, earning repeatedly. The GLM-5.3
  root carries 11 rows.
- The **Qwen ladder's** 36 receipts are mirrored in-tree at a commit pin
  (`registry/protocol/qwen38-receipts-public-8558b8c/`), read at seed time and
  tamper-tested, so its 33 rows cite bytes anyone can hash.
- **95 rows** are published, grouped and renderable; 9 of 20 comparability groups
  have a measured floor.
- One `capture_content_digest` (`b417acc22b8aa7f3…`, the Fruit BF16 root) is
  published as **three repos over three transports** with different
  `total_size_bytes` — the transport-parity proof, and the standing reason to match
  on **content, never containers**.

## Open items, and whose they are

1. **Ingest the three GLM-5.2 rows** — registry session. Evidence in-repo.
2. **Four paid publications have no `publish-root.json` anywhere on this machine**
   (~$11.85 of pod time): `glm53-fidelity-root-v1`, `glm53-fidelity-fp8-v1`,
   `glm53-fidelity-exl3-wrld-k4-v1`, `glm52-fidelity-fp8-v1`. The datasets verify
   against the served bytes, so the evidence is sound; the missing thing is the
   record that someone checked *after* upload.
3. **`malaiwah/GLM-5.3-Flash-TR3-6bpw`'s card says `fidelity_dataset: null`** while
   its dataset was published 2026-09-06 11:00:19Z, and its `datasets:` front-matter
   does not list it — so per `docs/CARD-ANNOTATION-SPEC.md:198-201` a paid,
   published root is **invisible from the card**. `bin/fidelity-card annotate`
   cross-checks against a registry row, so it should run after ingestion, not
   before.
4. **Identity collision needing a registry ruling:** `fruit-fidelity-root-v1` and
   `fruit-fidelity-root-runpod-v1` both publish `dataset.id`
   `fidelity--fruit.malaiwah.root.bf16` from two repos at two different
   `dataset_sha256`, and registry ids are hashed into `comparability.key`.
5. **One capture digest, three disclosure verdicts:**
   `fruit-fidelity-root-container-v1` carries a **blocking**
   `unexpected_tensors_overridden` (`affects_comparability: true`) while two other
   repos publishing the identical capture digest carry only a caveat with a pinned
   allowlist. Same bytes; the verdicts should agree or the difference should be
   explained.
6. **11 of 95 rows carry `top1_agreement: null`** (`STAT-005`). Not a lost
   measurement — a scorer that did not emit per-window top-1 counts. Five are
   `clean17` recomputes whose receipts retain per-window mean KLD only; two are
   author-reported and must keep the warning, because inventing a third party's
   top-1 is not available to us.
7. **`docs/CLOUD-RECIPES.md:297`** still records machine 68004 as presenting a
   certificate hostname mismatch / SSL proxy. **Measured false** — see below.
8. **One duplicate filed receipt**, `registry/receipts/malaiwah/turbo-2.05bpw.json`,
   uncited; the same value is admitted from `stream-turbo-2.05bpw-kld.json`.
   Registry hygiene, not spend loss.

## The security question closed in the right direction

**There is no TLS interceptor on Vast machine 68004.** The 2026-09-05 capture
failure was **forged UDP DNS response injection** on that box's path to `1.1.1.1`,
which its Docker-generated `resolv.conf` lists first: one A query for
`huggingface.co` returns **three** replies — two forged at ~31 ms carrying a fresh
random third-party address each round, and the genuine CloudFront set at ~198 ms,
which loses the race. The "certificate hostname mismatch" was **Meta's own valid
DigiCert-issued cert** on a real Facebook server our client reached only because
DNS lied.

Scope refutes both the benign and the malicious reading: `www.google.com`,
`twitter.com` and `www.wikipedia.org` are poisoned, while `pypi.org`,
`github.com`, `api.runpod.io`, `console.vast.ai` and **every** HF weight CDN are
clean with byte-identical leaves. The decisive datum: `cdn-lfs.huggingface.co` has
**NODATA upstream** — zero A/AAAA records — yet the box was handed a Dropbox
address for it, while an unrelated non-existent name correctly returned NXDOMAIN.
**Nothing that caches, proxies or forwards can invent records for a name that has
none.**

The container's CA bundle is byte-identical to the same file extracted from the
GHCR blobs with all 13 layer digests verified (`9481fcd9…3a8bdd`, 182,140 bytes,
121 certs, zero non-public roots). **No credential leaked: the protection worked by
refusing.** Rotation of the `mbelleau-buildbox` token remains prudent hygiene, not
incident response.

Both a "middlebox intercepts TLS on this host" and a "this host harvests Hugging
Face credentials" claim are **explicitly refused as unmeasured**. A refusal string
is where an unmeasured cause does the most damage, because it is the one place an
operator reads a verdict.
