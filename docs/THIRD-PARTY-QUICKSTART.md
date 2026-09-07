# Third-party quickstart — from a fresh clone to a submitted measurement

This is the unaided path for the paid route: one fresh secure on-demand
RunPod pod reached by authenticated SSH, rented for exactly one measurement
and destroyed afterwards. Everything here is $0.00 except the single step
marked **PAID**. No command here publishes externally unless you add
`--publish-root-to`.

What is always enforced, and what strict campaign mode adds on top, is in
[`CLOUD-RECIPES.md`](CLOUD-RECIPES.md). `bin/measure-cloud --help` is the
ground truth for every flag.

## Local capture without renting hardware

This route runs on **your measurement machine**, not inside the read-only Explorer
Space. Listing/preparing is stdlib-only; execution needs the selected fixture's
published, pinned CPU environment (the new fixtures use Python 3.12,
Transformers 5.16.1 and Torch 2.11.0+cpu). For production GPU runs, establish the
appropriate device-specific reference and floor; CPU fixture success is not a
GPU-kernel qualification.

```bash
python bin/fidelity_dataset.py architectures list
python bin/fidelity_dataset.py architectures show minimax-m2
python bin/fidelity_dataset.py architectures prepare --help
```

`engines/coverage.json` names actual public checkpoint/evidence revisions and
original-format limitations. MiniMax M2/M3 and Kimi K2 use native classes. Kimi K3,
DeepSeek V4's corrected runtime, Spark-X2.5 and IFM K2-Horizon require an explicitly
reviewed code pin. IFM K2-Horizon is not Moonshot Kimi despite its name. MiniMax M3's
video-generation task is not measured by next-token text KL.

For example, download the complete public MiniMax M2 fixture and its existing panel
into fresh directories on your machine:

```bash
hf download malaiwah/minimax-m2-tiny-random-bf16 \
  --revision d4825603496f1c636385af403e891aaf20e9afe6 --local-dir "$MODEL_DIR"
hf download malaiwah/minimax-m2-tiny-cpu-repro-v1 --repo-type dataset \
  --revision fb59d421b5fdc9fbd58ed00ac0d740eaa2a5e6b3 \
  --include 'panel/**' --local-dir "$EVIDENCE_DIR"
python bin/fidelity_dataset.py architectures prepare \
  --architecture minimax-m2 --model-dir "$MODEL_DIR" \
  --model-repository malaiwah/minimax-m2-tiny-random-bf16 \
  --model-revision d4825603496f1c636385af403e891aaf20e9afe6 \
  --panel "$EVIDENCE_DIR/panel" --author "$AUTHOR" \
  --dataset-repository "$AUTHOR/my-minimax-capture" \
  --dataset-id "fidelity--minimax-m2.$AUTHOR.cpu-control" \
  --out "$NEW_WORKFLOW_DIR"
```

Set the directory variables and `AUTHOR` to your actual HF handle first. The
dataset repository is an attribution field, not an instruction to create/upload it.
Preparation reads local config, token-panel and license bytes, refuses overwrite,
and writes `workflow.json`, a SHA-bound panel binding, and `run.sh`. Inspect them,
then execute `bash "$NEW_WORKFLOW_DIR/run.sh"` in the pinned environment.
Root mode performs two separate captures, verifies tensor content, and forces
numerical self-comparison. The same generated workflow was exercised end to end.

For **your quantized model**, select `--role quant`, its real immutable model pin,
`--scope-file` describing the actual intervention, `--reference` naming the sealed
reference dataset, and the correct `--codec`/`--bits` if applicable. It captures the
candidate and compares with `--own-heads`. Retain the original config, weights and
tokenizer identity; do not label a quant with the root's model revision. Bind the
panel to the root tokenizer with `--tokenizer-root` where appropriate. The recipe
does not invent common ancestry, a representative evaluation panel, a usable floor,
or a registry submission. Its shell stops on nonzero results, including advisory
warnings: inspect the receipt rather than suppressing the return code.

Custom runtimes additionally require all three flags:
`--trust-remote-code --code-repository OWNER/REPO --code-revision FULL_COMMIT`.
Use the catalog's tested code pin only after reviewing its code and licenses.
Code and transitive Python imports are verified before execution; verification is
**provenance, not a sandbox**. A code pin controls dispatch, never replacement model
dimensions. Original FP4/FP8/MXFP4 exports can still require a different decoder:
the BF16 architecture fixture is not evidence that every original format runs.

The layer-outer path reconstructs admitted packed matrices a module at a time.
Standalone complete safetensors files need no invented shard index; overlapping
downloads require an exhaustive index and an explicit header-admission barrier.
Packed component omissions and unknown constituents refuse rather than passing
integer/FP8 storage through as native weights. Captures retain decoded-module counts,
reader hashes, original checkpoint identity and activation-quantization omissions.

Canonical dense Qwen35 GGUF captures retain the original wrapper config and an
explicit derived **text-only model view** in the receipt. No vision weights are
borrowed or synthesized. A converter/layout attestation is required because a
historical fused-QKVZ export cannot safely be inferred from the architecture label.
Qwen4-Exp and Qwen35-MoE GGUF are explicitly outside this bridge.

The paid route below remains independently gated; none of these additions grants
paid-engine admission or starts a rented instance.

## 1. Prerequisites

- Stock Python 3.9 or newer; `bin/` needs no local install.
- A clean clone: tracked files unmodified. Untracked files are ignored.
- A RunPod API key in an owner-only mode-0600 file, default
  `~/.config/runpod/api_key` (`--runpod-key-file` otherwise). Never argv or
  environment.
- A separate read-only Hugging Face token in an owner-only mode-0600 file,
  explicitly passed as `--hf-download-token-file ~/.hf_read_token`.
  This is the credential transported to the pod for target fetch.
- For optional publication, a different HF write token in
  `--hf-token-file ~/.hf_token`; it stays on the controller. Missing download
  credentials and identical publisher/download token bytes are refused, even
  when copied into different files. File checks do not certify Hub scope:
  create the download token with read-only privileges.
- `~/.ssh/id_ed25519.pub`, accepted by RunPod at create.
- A `systemd --user` session that survives logout:
  `loginctl enable-linger $USER`.

```bash
chmod 600 ~/.config/runpod/api_key ~/.hf_read_token
# Only if publishing:
chmod 600 ~/.hf_token
test -f ~/.ssh/id_ed25519.pub
bin/fidelity-doctor
```

`bin/fidelity-doctor` is offline and prints no secret. Do not proceed on a
failure.

## 2. Install the autonomous reaper (once per machine and account)

```bash
bin/measure-cloud reaper --provider runpod --install
```

This changes only local user-systemd state and performs read-only RunPod
account queries; it creates no provider resource. Every paid run refuses
unless this timer is installed and healthy. A checkout newer than the
installed snapshot is a warning in the plan, not a refusal; re-run the
install to pick it up.

## 3. Dry-run the root recipe — $0.00

The full GLM-5.3 root capture, with every derived value left to the
controller (GPU, storage, host minima, dataset repository and name, tensor
allowlist, download token; the dry-run plan prints each one):

```bash
bin/measure-cloud --provider runpod --role root \
    --model zai-org/GLM-5.3-BF16 --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
    --panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \
    --dataset-id fidelity--glm53.malaiwah.root.bf16 --publish-root-to malaiwah/glm53-fidelity-root-v1 \
    --hf-token-file ~/.hf_token --hf-download-token-file ~/.hf_read_token --measurer malaiwah --runpod-datacenter US-NC-1 \
    --max-cost 65 --max-runtime 7h30m --retrieval-delete-reserve 14400 \
    --out ~/fidelity-runs/glm53-root --dry-run
```

Replace `--dataset-id`, `--publish-root-to`, `--measurer`, both token-file
paths and `--out` with your own identities. Because the hidden-form dataset
redistributes the checkpoint's native output-head weights, the dry-run
validates the exact pinned `LICENSE` bytes anonymously and records
`license: other`.

The two numbers are the tool's own, not estimates. `--max-runtime` must be
at least the **authored bound** for this target on an H200 —
`bin/engines.json` `root_timing_profiles[zai-org/GLM-5.3-BF16@304b8051]`:
`(5520 fetch + 420 setup + 2 x 6600 cold run + 2400 verify/compare/qualify)
x 1.25 = 26925 s`, so `7h30m` (27000 s) is the smallest round value — and
`--max-cost` must cover the all-in maximum the controller computes from it,
`$62.925` at today's $4.59/h for 11.5 h with the 14400 s reserve. Since
2026-09-05 you may **omit `--max-runtime`** for a target with an authored
row: the controller defaults it to the bound and says so in the plan
(`workload bound 26925 s (defaulted to the authored bound; --max-runtime to
override upward)`); it stays required for a target without a row. When the
numbers are below the bound the dry-run reports every finding at once
(observed 2026-09-05 with the older `3h30m` / `40` recipe; four findings
because the maintainer's destination exists and a lease was live):

```text
  ERROR  REFUSE: 4 pre-spend findings; every one must be settled before a pod is created
  ERROR          [1] target-specific timing exceeds --max-runtime: the bound is 26925 s (components_seconds), --max-runtime is 12600.0 s
  ERROR              raise --max-runtime to at least the bound (or omit it: the authored bound is the default); it is the deadline the watchdog enforces, not an estimate
  ERROR              the cost below is priced at the bound, so one edit settles both
  ERROR          [2] all-in maximum $62.81 exceeds --max-cost 40
  ERROR              GPU $4.59/h for 11.48 h (workload 26925 s + retrieval/delete reserve 14400 s) plus storage for that window
  ERROR              raise --max-cost to at least 62.82; the reserve is already the retrieval contract's minimum unless you raised it
  ERROR          [3] local root publication preflight failed: authenticated datasets/malaiwah/glm53-fidelity-root-v1 already exists or collides with the destination
  ERROR          [4] an earlier lease may still hold a pod: ... is ACTIVE (pods hj0h6wjpqwjoj3, ...); a839fcd8c3b33bd8a3bbe517 is AMBIGUOUS (pods none yet, ...)
  ERROR              inspect: measure-cloud reaper --provider runpod --list
```

The plan a passing dry-run prints (2026-09-05, `--max-cost 65`, no
`--max-runtime`, no `--retrieval-delete-reserve`, `--runpod-datacenter
US-NC-1`):

```text
RUNPOD PLAN
  target                 zai-org/GLM-5.3-BF16@304b8051cfb2b260b61ce0cbe330e02a98e73639
  profile timing         root-hf-transformers-bf16 / bound 26925 s = (5520 fetch + 420 setup + 2 x 6600 cold run + 2400 verify/compare/qualify) x 1.25 margin; authored bin/engines.json root_timing_profiles[zai-org/GLM-5.3-BF16@304b8051] on H200
  gpu                    NVIDIA H200 x1 (secure cloud, on-demand) $4.59/h
  datacenter             US-NC-1 (pinned; the create refuses elsewhere)
  all-in hard cap        $65 (calculated $61.94...) -- the BOUND: GPU rate x (workload deadline + retrieval/delete reserve) + storage; not the estimate
  expected spend         ~$27.46 for ~359 min at $4.59/h -- the authored components without the margin (the row's measured run); the cap above is the bound
  storage                container-disk: container disk 1800 GB, pod volume 10 GB, run root under /root
  workload bound         26925 s (defaulted to the authored bound; --max-runtime to override upward); retrieval/delete reserve 13818 s (derived: 1800 build + 3 x (3600 download + 306 verify) + 300 delete; --retrieval-delete-reserve to override upward)
```

The reserve is the retrieval contract's own minimum (archive build, three
bounded download attempts with local verification sized by the archive, the
delete) and is derived by default; pass `--retrieval-delete-reserve` only to
raise it. The expected-spend line is the authored row's measured components
(the container-disk cold run in that row is still a projection until a run
re-measures it — `bin/engines.json` says so); the receipts below are the
observed reality.

What the pod actually did for this root (JOURNAL 2026-09-04): 1.5 TB fetched
in 12 min on the container disk, a cold run ~10 min of forward; the pod that
captured cold run 2 against the imported first run lived 26 min (lease
`v9c25kodqcb26u`: created 12:08:17Z, absence proven 12:34:01Z). The bound is
the watchdog's deadline, not the expected spend; the plan prints the cap as
`all-in hard cap` and the authored components under `profile timing`.

`--dry-run` runs every check — target identity, clean checkout, reaper
health, account inventory and balance, cost quote against `--max-cost` — and
prints the plan without creating anything. A failure is a refusal that names
its reason, exit code 3. Note `clean checkout` means **tracked** files
unmodified; an untracked scope or allowlist file is fine.

The same shape on the suite's 5B CI fixture. Its authored timing row in
`bin/engines.json` names an L4 and a 1-hour conservative bound, so the GPU is
derived and `--max-runtime 1h` is that bound:

```bash
bin/measure-cloud --provider runpod --role root \
    --model malaiwah/GLM-5.2-SIQ-Fruit-bf16 --revision ef68013aa6e16453cf52b5b77647f72fbe258c3c \
    --panel-dir engines/panels/panel--fruit.malaiwah.heldout-v1 \
    --dataset-id fidelity--fruit.<your-hf-handle>.root.bf16 \
    --measurer <your-hf-handle> --hf-download-token-file ~/.hf_read_token \
    --max-cost 5 --max-runtime 1h --out ~/fidelity-runs/fruit-root --dry-run
```

Without `--publish-root-to` the sealed dataset stays under `--out`
(§4 shows how to publish it later).

## 3b. Measure a quant against a published root — the candidate route

This is how every GLM-5.3 quant row in the registry was made (the FP8
release, wrldsuksgo2mars K4, three davidsyoung TR3 builds, drowzeys): the
**root protocol run on a quantized target**. The pod captures the quant twice
in fresh processes under an authored scope, qualifies the two captures
bitwise, and scores the qualified capture against the published root dataset
on the pod; the number comes back in the result archive. Observed on an H200
in `US-NC-1`: ~33–45 min of pod time, ≈ $3–4 per candidate (JOURNAL
2026-09-05: K4 ~33 min; lease `h3nnboclnzu7cs` for the 3.25 bpw TR3 build:
create observed 02:49:29Z, absence proven 03:33:59Z, at $4.59/h).

It is spelled `--role root` plus four candidate flags
(`--candidate-scope`, `--candidate-codec`, `--candidate-bits`,
`--reference-dataset`; all four or none). The legacy `--role quant`
teacher-logits path is **not** this route — `bin/measure <url>` and
`--role quant` both send an EXL3/FP8 quant here instead.

**Inputs you author, $0, seconds each:**

1. *Is it measured already?* `bin/measure <hf-url> --plan-only` answers from
   the public registry in under a second and prints the row if so.
2. *The scope* — which tensor classes are quantized, at what format and bits,
   read from the checkpoint's index, never from its name:

   ```bash
   # two small public files at the pinned revision
   curl -sSLO https://huggingface.co/<owner>/<quant>/resolve/<40-hex>/config.json
   curl -sSLO https://huggingface.co/<owner>/<quant>/resolve/<40-hex>/model.safetensors.index.json
   python3 engines/tools/exl3_scope.py --index model.safetensors.index.json --config config.json \
       --repo <owner>/<quant> --revision <40-hex> --out engines/scopes/scope--my-quant.json
   # block-scaled FP8 release: engines/tools/fp8_scope.py, same flags
   ```

   The committed examples are `engines/scopes/scope--{wrld,dy30,dy325,dy342,drowzeys}-exl3.json`.
   The scope file may live anywhere; the pre-spend gate validates it against
   the registry's scope rules at $0.
3. *The unexpected-tensor allowlist* — the tensor names the checkpoint carries
   beyond what its architecture builds (GLM-5.3's MTP block, `model.layers.78`).
   The capture refuses on the pod unless the observed set equals a bound list.
   **You need not author it**: when no attested row exists for your pin, the
   dry-run derives it from the same two files at plan time
   (`engines/tools/index_census_allowlist.py`, the method behind every
   GLM-5.3 allowlist), binds it into `job.json` by both digests as
   `inputs/allowlist.json`, and says so:

   ```text
     allowlist derived from index           ok  12311 names past layer 78 (GlmMoeDsaForCausalLM); artifact 2d3aed8145884861..., names d567faf9576946cc...; register these in runpodsafety._ALLOWLISTS to attest it
     exact-unexpected-tensor-allowlist      ok
     WARNING  note: no authored unexpected-tensor allowlist for davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw@6d6bd738c0c1: bound the index census instead (12311 names past layer 78, sha256 2d3aed8145884861...). The pod refuses unless the loader's unexpected set equals it exactly. To attest it, register the printed digests in bin/fidelity/runpodsafety.py _ALLOWLISTS and commit the file under engines/tools/layer-outer-evidence/
   ```

   (Observed 2026-09-05 on a scratch checkout with that pin's table row
   removed; the derived digests are exactly the ones the committed row
   attests.) The gate records `provenance: derived_from_index`; a row in
   `bin/fidelity/runpodsafety.py` `_ALLOWLISTS` plus the file under
   `engines/tools/layer-outer-evidence/` upgrades it to `authored`. To author
   one for the record, or to see the digits before spending:

   ```bash
   python3 engines/tools/index_census_allowlist.py --repo <owner>/<quant> --revision <40-hex> \
       --index model.safetensors.index.json --config config.json \
       --out engines/tools/layer-outer-evidence/<name>-unexpected-keys.json
   ```

   It writes the list plus a `.provenance.json` sidecar and prints the three
   digests (`artifact_sha256`, `canonical_sorted_names_sha256`, `count`).
   Reproducibility check: it regenerates the committed
   `dy325-exl3-layer78-unexpected-keys.json` and
   `drowzeys-exl3-layer78-unexpected-keys.json` byte-for-byte
   (`artifact_sha256 2d3aed81…`, `969cee60…`).
4. *The reference* — the published root dataset at its immutable revision:
   `malaiwah/glm53-fidelity-root-v1@9c4a29ee10f393ed2fdbdb9262c1192ddb1507b4`
   for GLM-5.3 (`bin/fidelity-dataset describe hf://…` prints its
   `dataset_sha256 6b8d3a7b…`). Its panel is
   `engines/panels/panel--glm53.malaiwah.corpus5x5-v1`, in this checkout.

**The dry-run.** Verified 2026-09-05 against the K4 quant (already measured,
so the front gate says so and the dry-run continues only through its own
gates; substitute your quant, scope and identities):

```bash
bin/measure-cloud --provider runpod --role root \
    --model wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 --revision 47af23347db743b4666d952e2eb48f2b01c3fede \
    --panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \
    --dataset-id fidelity--glm53.malaiwah.quant.exl3-k4 \
    --candidate-scope engines/scopes/scope--wrld-exl3.json --candidate-codec exl3-mcg --candidate-bits 4 \
    --reference-dataset malaiwah/glm53-fidelity-root-v1@9c4a29ee10f393ed2fdbdb9262c1192ddb1507b4 \
    --gpu H200 --runpod-datacenter US-NC-1 \
    --hf-token-file ~/.hf_token --hf-download-token-file ~/.hf_read_token --measurer malaiwah \
    --max-cost 45 --max-runtime 3h30m --retrieval-delete-reserve 14400 \
    --out ~/fidelity-runs/my-quant --dry-run
```

Observed (29 s, exit 0, `--out` not created; the plan JSON is a byte count
unless you pass `--plan-json FILE` or `--json`):

```text
  candidate is exl3 trellis              ok  quant_method exl3 codebook None declared bits 4; decoded to bf16 per module (exl3-trellis-decode-to-bf16). Per-module codebook and the payload's own bit width are read from the checkpoint on the pod and checked against the declaration
REGISTRY CHECK (before anything is planned or spent)
  1 artifact record(s) for wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1:
    [EXACT] artifact--wrldsuksgo2mars.glm-5.3-exl3-k4-v1
        measured at exactly this revision (47af23347d)
  ...
ALREADY MEASURED: the rows above answer this request. Safe RunPod refuses --force; a separately identified candidate measurement (new --dataset-id) continues only through its own gates.
  candidate panel                        ok  exact for reference root zai-org/GLM-5.3-BF16@304b8051cfb2; tokenizer files byte-identical
  candidate reference                    ok  malaiwah/glm53-fidelity-root-v1@9c4a29ee10f3 dataset_sha256 6b8d3a7bdf934f18, capture 9eba97dddb4ff2e2, panel panel--glm53.malaiwah.corpus5x5-v1
RUNPOD PLAN
  target                 wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1@47af23347db743b4666d952e2eb48f2b01c3fede
  profile timing         root-hf-transformers-bf16 / operator --max-runtime 3h30m (no authored timing row for this target on H200)
  gpu                    NVIDIA H200 x1 (secure cloud, on-demand) $4.59/h
  datacenter             US-NC-1 (pinned; the create refuses elsewhere)
  all-in hard cap        $45 (calculated $36.45...) -- the BOUND: GPU rate x (workload deadline + retrieval/delete reserve) + storage; not the estimate
  expected spend         not stated: no authored components for this target; the cap above is the bound, and the receipts of prior runs of this route are the only estimate (docs/THIRD-PARTY-QUICKSTART.md 3b)
  storage                container-disk: container disk 600 GB, pod volume 10 GB, run root under /root
  workload bound         12600 s (--max-runtime); retrieval/delete reserve 13818 s (derived: 1800 build + 3 x (3600 download + 306 verify) + 300 delete; --retrieval-delete-reserve to override upward)
  ... 18 gates ok (exact-unexpected-tensor-allowlist resolved from the authored table with no flag) ...
  WARNING  note: no authored timing evidence for wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1@47af23347db7 on H200; using your --max-runtime 3h30m as the workload bound.
```

For an unmeasured quant the registry block reads `not yet measured --
proceeding` and the plan is the same. `--gpu H200` is required because no
timing row exists for a quant; the reference root was captured on an H200
and the GPU model moves bits, so use the same class. `$45` is the **hard
cap** (GPU rate x deadline + reserve), not the expected spend — the receipts
say ≈ $3–4. `--retrieval-delete-reserve` defaults to the retrieval
contract's minimum (13818 s for this archive bound; every GLM-5.3 candidate
passed 13818 or 14400 by hand before the default was derived), so the flag
in the command above is optional. Add
`--publish-root-to <you>/<repo>` to publish the sealed candidate dataset from
this machine after teardown (the token never reaches the pod).

A quant that is `ALREADY MEASURED` exits `no_spend` at the front gate when it
is run without the candidate flags (`--role quant`), before any account
access; the front gate is `bin/measure <url> --plan-only`.

## 4. The paid run — the one paid step

**PAID:** repeat the exact dry-run command without `--dry-run`. Leave off
`--yes` for the interactive confirmation:

```text
Create one secure on-demand RunPod (calculated maximum $…; hard cap $…;
--max-cost is the whole budget)? [y/N]
```

Only `y` or `yes` permits the single create POST. From there the controller
authenticates the pod's ED25519 fingerprint through RunPod's container-log
API, runs the capture, retrieves and verifies the archive, deletes the pod
and proves its absence. Retrieval failure still deletes the pod; the lease
deadline and the reaper remain the backstops. Exit codes: 0 ok; 1 the run
failed and the pod is proven gone; 3 refused before anything was created;
90 a pod may remain — run `bin/measure-cloud reaper --provider runpod --list`.

The verified root outputs are:

```text
<out>/result.tar.gz                                   # the retrieved archive (sha256 in <out>/terminal-receipt.json)
<out>/result/receipts/root-qualification.json         # two cold runs bitwise
<out>/result/dataset, <out>/result/dataset-repeat     # the two sealed captures
<out>/result/receipts/publish-root.json               # only with --publish-root-to
<out>/terminal-receipt.json                           # pod id, archive sha256, lease state, billing
```

For the candidate route (§3b) the number is in
`<out>/result/receipts/reference-comparison/comparison-receipt.json` (schema
`malaiwah.fidelity-comparison-receipt.v1`), beside
`root-comparison/comparison-receipt.json` (the self-compare, exactly 0.0)
and the same `root-qualification.json`. **No `measurement-receipt.json` is
written by this route** — that file belongs to the legacy `--role quant`
teacher-logits path. The post is rendered from the receipts:

```bash
bin/fidelity-post render --result <out>/result --out post.md
bin/fidelity-post publish --result <out>/result --token-file ~/.hf_token --receipt <out>/post-receipt.json
```

`render` is $0 and local; `publish` opens a discussion on the **candidate's**
model page (outward-facing — do it only when you mean to).

### Publishing a root later

A root captured without `--publish-root-to` is a sealed dataset under
`--out`, and the same publisher the controller uses can push it afterwards
from this machine. Every value it needs is in the run directory; the two
archive identities are cross-checked against `<out>/result.tar.gz` before
anything is uploaded (`bin/fidelity-dataset publish --help`):

```bash
bin/fidelity-dataset publish <out>/result/dataset \
    --repo <owner>/<repo> --expected-head absent \
    --qualification <out>/result/receipts/root-qualification.json \
    --job <out>/result/job.json \
    --result-archive <out>/result.tar.gz \
    --expected-archive-sha256 "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["result_archive_sha256"])' <out>/terminal-receipt.json)" \
    --expected-archive-bytes "$(stat -c %s <out>/result.tar.gz)" \
    --token-file ~/.hf_token --receipt <out>/result/receipts/publish-root.json
```

`--expected-head absent` is the optimistic authorization: the destination
must not exist (pass its 40-hex HEAD instead to append to a repo you own).
The receipt it writes carries `repository`, `revision`, `dataset_sha256`,
`result_archive_sha256`, `result_archive_bytes` and
`verified_after_publish`; `bin/fidelity-dataset describe hf://<owner>/<repo>@<revision>`
then prints the identity card anyone else will see. `<out>/result/dataset`
is cold run 1, the capture that is published; captures sealed before
2026-09-04 carry their pod path in `validation/` and the publisher refuses
them.

### Optional: strict campaign mode

If the account is dedicated to this suite, several attempts must share one
ceiling, or you want a sealed proof that the reaper destroyed a real pod
after the controller died, add the four `--campaign-*` flags and run the paid
`measure-cloud drill` first. The flags, the drill and the tradeoffs are in
[`CLOUD-RECIPES.md`](CLOUD-RECIPES.md#strict-campaign-mode-opt-in); the
`--help` epilog shows both commands.

## 5. What goes to the registry, and who files it

**Candidate route (§3b):** there is no submission receipt to validate. The
comparison receipt is not a `submission-receipt.v1` (`bin/registry-submit`
on it prints `REJECTED … missing required property 'submission_schema'`,
by design), and every GLM-5.3 row
so far was filed maintainer-side from the receipts under
`registry/protocol/glm-5.3/`. What you deliver is the discussion URL from
`fidelity-post publish`, the `<out>/result/receipts/` directory, and — if you
published the candidate dataset — its repo and revision. The maintainer
validates, files the receipt beside the others and derives the row; the
registry data files are generated, never hand-written.

**Legacy `--role quant` teacher-logits lane:** validate the sealed
`measurement-receipt.json` the way the registry will, offline, $0.00:

```bash
bin/registry-submit <out>/result/receipts/measurement-receipt.json
```

Prints the row your receipt generates, its comparability key and class, and
the rows it may be ranked against — or exactly which check failed. Doing this
before submitting is, in the maintainer's own words, the difference between a
same-day merge and a round trip.

## 6. Submit it — one live destination

**Hugging Face discussion** on the registry dataset — this is the live path:

1. Open <https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
2. New discussion titled `submission: <repo> on <panel>`
3. Candidate route: paste the `fidelity-post publish` discussion URL, the
   published candidate dataset `repo@revision` (if any) and attach
   `<out>/result/receipts/reference-comparison/comparison-receipt.json` and
   `root-qualification.json`. Legacy quant lane: paste the template from
   [CONTRIBUTING §2](../registry/CONTRIBUTING.md) with your
   `measurement-receipt.json` inside the fence.

The GitHub pull-request mirror described in CONTRIBUTING §3 is **not live** —
the URL 404s today; CONTRIBUTING says so and this document will keep saying so
until it exists. Do not wait for it.

## 7. What happens next — what is promised, and what is not

From [CONTRIBUTING §4](../registry/CONTRIBUTING.md), which is the commitment
this project actually makes:

* The maintainer saves your receipt under `receipts/<your-handle>/`, runs the
  same checks CI would, and **replies in your thread** with the generated row
  id, its comparability key, its class, and which rows it sits next to — or,
  if refused, exactly which check failed and what to change. "Either way you
  get a real answer, not silence."
* **No response-time SLA is promised anywhere**, and this document will not
  invent one. The stated fast path is a receipt that already passed
  `bin/registry-submit`: validated submissions are same-day-mergeable; broken
  seals and missing fields cost a round trip.
* Acceptance criteria are mechanical and public: the seal verifies, the
  40-hex revision, `run_count >= 2` with tensor-content determinism evidence,
  a scope census (for a GGUF the registry refuses the row without one), a
  `produced_by` block with `entrypoint_sha256` (HARN-001), your own handle in
  `measurer` — the complete bounce list is
  [CONTRIBUTING §5](../registry/CONTRIBUTING.md).
* Outside measurements enter as class `advisory` (a provenance classification).
  Equal comparability keys are necessary, not sufficient, for ranking:
  `pair_predicate` must also permit the pair. `independently_verified: true`
  means another party reproduced the number on the required panel/reference
  and measurement configuration; it is not conferred by submission alone.

## If your target is not measurable

The refusal identifies the **specific route's** missing prerequisite before
spend: a fetchable/bindable panel, supported architecture/reader, exact
surface/profile or authored admission evidence. MLX/NVFP4/AWQ/GPTQ adapter or
native-loader existence is not blanket paid support, but it is also wrong to
say no such readers exist.

Qwen3.8-27B has a recorded published root, and its shard0 panel was recovered
byte-exactly from the public suite ([M1 learning 14](M1-QWEN38-ROOT-LEARNINGS.md#method)).
Some registry panel entries still say `private` with no URI: that can block
automatic planning without making all underlying bytes private. Public bytes,
current registry routing, verified panel binding and exact paid admission are
distinct requirements; no newly verified live availability is claimed here.
Use the [support matrix](../README.md#before-you-rent-what-is-measurable-today)
and the exact dry-run refusal. [CONTRIBUTING §6](../registry/CONTRIBUTING.md)
describes proposing a panel or model.
