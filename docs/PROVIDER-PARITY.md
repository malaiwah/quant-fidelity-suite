# Provider parity — what a second provider needs before it can publish a number

**Status:** gap analysis and plan of record, 2026-09-06.

RunPod is the only provider a paid measurement runs on today. That is not a
judgement about the other three; it is a statement about twelve methods.
`bin/measure-cloud --provider {jarvislabs,runpod,vast,lambda}` accepts four
names, `jarvislabs` is admitted solely for `reaper` cleanup of historical
leases, and `vast` and `lambda` are refused before any provider mutation.

## Why this is legitimate to fix, scientifically

Provider is **not** a comparability axis. `docs/ARCHITECTURE-DETERMINISM.md`
measured **two A100s in two clouds agreeing bitwise**, while an H200 sat
2.973e-04 nats away from them — an order of magnitude above the effect between
adjacent bit-widths. So what a comparison binds is the **device model** and the
**rebuilt software stack**, not the company that rented you the card.
`stackprint.fingerprint_sha256` hashes the `gpus` block (`props.name`,
`total_memory_mib`, `compute_capability`) and the whole torch/CUDA stack, and
`stack_relation` — derived from that digest — is one of the seven
`COMPARABILITY_KEY_FIELDS`. `panel_id`, `reference_id`, `metric_name`,
`direction`, `accumulation_dtype`, `stack_relation`, `head_policy`. No provider
field appears in it.

Evidence from this campaign: one `stack_fingerprint_sha256` (`e7ddf6b28047…`)
covers the GLM-5.2 root, every GLM-5.2 candidate capture and the whole GLM-5.3
family, and four GLM-5.3-Flash rows landed on one comparability key
(`cmp--c9480e610b94f2aa`) across **three datacenters** — US-NC-1, US-CA-2 and
CA-MTL-3 — because the GPU model and the stack were identical and the site was
not part of what the root asserts.

A Lambda H200 or a Vast H200 running the same `bootstrap_measure.sh` recipe is
therefore the same measurement. The obstacle is not physics. It is that we
cannot yet **prove** what we rented, **prove** it is gone, or **reconcile** what
it cost.

## The twelve methods

`bin/measure_cloud.py` drives a provider through a fixed surface;
`bin/fidelity/runpodapi.py` is the reference implementation. Every non-RunPod
adapter today is missing **the same twelve** — audited, not assumed:

|method|what it buys|
|---|---|
|`prepare_safe_create()`|two-phase create: the request is built and frozen before any mutation|
|`submit_prepared_create(prepared)`|submits the frozen request, so a LOST create RESPONSE is reconcilable instead of ambiguous|
|`validate_safe_resource_binding(id, *, expected_name, gpu_type_id, secure_cloud, gpu_count, volume_gb, container_disk_gb, image_name, terminate_after)`|fail unless the live exact-id resource is the one requested|
|`attest_live_resource(id, *, expected_gpu_model, expected_vram_bytes, min_vcpu, min_ram_gb, volume_gb, container_disk_gb, …)`|**the scientific gate**: read-only SSH proof that the box is the DEVICE the root was captured on, before upload or spend|
|`list_lifecycle_resources()`|complete exact-id rows where every listed status is live|
|`get_lifecycle_resource(provider_id)`|exact-id detail; names are deliberately not accepted as ids|
|`list_network_volumes()`|persistent chargeable volumes, which outlive an instance|
|`chargeable_inventory()`|instances plus volumes with **explicit completeness** — a partial inventory cannot prove no leak|
|`server_time_evidence(*, max_clock_delta_seconds, max_evidence_age_seconds)`|the provider's own clock, so a teardown deadline is encoded against their time and not ours|
|`ssh_host_ed25519_fingerprint(id, *, timeout)`|authenticate the host key from provider logs instead of trusting first contact|
|`billing_history(id, *, start_time, end_time, bucket_size)`|the official per-resource billing response|
|`reconcile_billing(lease, *, now)`|a post-absence, independently stable cost closure|

The split is not arbitrary. The first four are **"is this the thing I asked
for"**, the next four are **"is anything of mine still alive"**, and the last
four are **"what did it cost, and whose clock says so"**. A provider missing any
group can run a capture but cannot publish a receipt anybody should trust.

Adapters already implement the easy surface: `available`, `require`, `status`,
`balance`, `gpus`, `create`, `destroy`, `list_instances`, `get`, `exec`,
`upload`/`download`. JarvisLabs additionally has `run_job`/`run_status`/
`run_logs` from the historical campaign. So the remaining work per provider is
bounded and named.

## Per-provider status and blockers

### Vast (`bin/fidelity/vastapi.py`, 14 public methods)
Container-native execution is implemented and selftested
(`bin/selftest_provider_portability.py`, `bin/fidelity/vastcontract.py`).
Blockers beyond the twelve:
- **No reaper.** `measure-cloud reaper --provider vast` refuses; RunPod and
  historical JarvisLabs are the only supported sweeps. Without an autonomous
  teardown backstop a controller death leaks a billing instance, so no paid
  measurement may run there. **This is the blocking item.**
- **Storage is pod-scoped.** `fs_delete` returns "no separable filesystem on
  vast: the volume went with the pod at destroy". A JarvisLabs filesystem
  outlives its instance, which is what makes a preempted spot box cheap to
  resume; Vast has no equivalent, so `--storage-layout` semantics differ and
  `list_network_volumes` is trivially empty rather than unimplemented.
- **Host quality varies.** A Nevada host's SSL proxy to huggingface.co
  presented a certificate hostname mismatch and `UNEXPECTED_EOF`, failing a
  capture at the setup stage (2026-09-05). Any Vast lane needs a reachability
  probe before it does real work, and a bad host id recorded.

### Lambda (`bin/fidelity/lambdaapi.py`, 15 public methods)
- Instances are **VMs, not containers**: the measurement image cannot be used
  as-is, and `bootstrap_measure.sh` must rebuild the stack on the VM. That is
  the SSH+bundle path RunPod already uses, so it is a fit rather than a rewrite.
- Has `ssh_key_names_available` (no other adapter needs it) because Lambda
  binds keys by name at create.
- Historical note from the GH200 qualification: the fit arithmetic mis-sized a
  root plan at 63 GB/GPU and refused hardware that would have worked — see
  `docs/REVIEW-DEFERRED.md`. Lambda parity should not be declared without that
  fixed, or the first Lambda root will be refused for a phantom requirement.

### JarvisLabs (`bin/fidelity/jlapi.py`, 22 public methods)
Historical. It is the provider the whole harness was originally written
against, and every portability bug found since was a JarvisLabs
*representation* treated as universal truth (int machine ids, `"Running"` vs
`"RUNNING"`, ids comparing as ints in a set). It keeps `reaper` support for
lease cleanup and is the lowest-value target for measurement parity — but it is
the highest-value target for the conformance test, because its surface is the
largest and its assumptions are the ones already known to be wrong.

## Beyond the adapters

Three things are hardcoded to RunPod in the controller and must be generalised
once an adapter conforms:

1. **Execution dispatch.** `main()` routes to `_main_runpod` and the tail
   refuses everything else: "paid measurement execution requires explicit
   --provider runpod".
2. **The reaper.** One provider string, one systemd template instance
   (`fidelity-cloud-reaper@<provider>.timer` already exists — the template unit
   landed 2026-09-06, so the timer side is ready and only the sweep needs a
   per-provider adapter).
3. **The controller-loss drill.** `drill` refuses non-RunPod outright, and
   strict campaign mode binds a sealed safety proof to it. A provider without a
   drill can be measured on, but not in strict campaign mode.

## Definition of done, per provider

1. The twelve methods implemented against the provider's official API, with
   the same refuse-with-advice semantics.
2. A conformance rung in `bin/selftest_provider_portability.py` passing for
   that adapter — offline, no provider contacted.
3. A reaper sweep that settles a lease and proves absence from any machine.
4. One `--dry-run` reaching a cost quote with zero refusals.
5. One paid Fruit-scale capture whose `capture_content_digest` is compared
   against `malaiwah/fruit-fidelity-root-v1`
   (`b417acc22b8aa7f3294b8e62c4b619bc5051aef9fd8a073602572a30af6b3e1c`) — same
   device model reproduces it bitwise, and a different one gives the provider's
   device term as a number rather than a worry.
6. Teardown proven on success, failure, exception and interrupt.

Only after 3 and 6 may a paid measurement run there at all, because a leaked
instance is a blocker-level defect and an unreconciled cost is an unpublishable
receipt.
