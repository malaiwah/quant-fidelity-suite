#!/usr/bin/env python3
"""Which providers may take a paid measurement, and exactly why the rest may not.

One table, read by the controller AND by `bin/selftest_provider_portability.py`,
so a declaration cannot disagree with behaviour.  Before this module the answer
lived in three hardcoded RunPod strings in `bin/measure_cloud.py` (execution
dispatch, the reaper, the drill) plus a hand-maintained level table at the
bottom of the selftest, and the two halves could drift silently: an adapter
could implement everything and still be refused for no stated reason.

The predicate, and it is a PREDICATE rather than a label:

    a provider may take a paid measurement  iff
        it implements all twelve contract methods          (COMPUTED)
    and a paid execution entrypoint exists for it          (COMPUTED)
    and it declares a complete paid execution profile      (DECLARED, CROSS-CHECKED)
    and every enumerated safety property is met            (COMPUTED from both)
    and definition-of-done items 3 and 6 are proven        (COMPUTED, then DECLARED)
    and its blocker tuple is empty                         (DECLARED)

Method conformance is a fact about the code, so it is computed from the adapter
class and never declared.  That is what makes the table safe to leave alone
while three ports land: an adapter reaching twelve-of-twelve needs no edit here
and stays correctly refused for whatever non-method blockers remain.  It also
means a table cannot claim conformance it does not have.

`SAFETY_PROPERTIES` is the same principle applied to the six properties RunPod
paid for.  Each names the methods and the profile claims that ENFORCE it, so a
provider that cannot meet one is refused BY DERIVATION -- never by someone
remembering to write a blocker line, and never by relaxing the property.  A
property RunPod meets that another provider cannot is a blocker for that
provider; it is never a weakening of the path.

`PROVIDER_BLOCKERS` carries only the residue that no offline test can compute --
"no credential on this box", "no paid capture has reproduced the published
root".  Each entry is an explanatory sentence, because a blocker nobody can read
is a blocker nobody will clear, and ENABLING A PROVIDER IS DONE BY DELETING A
BLOCKER LINE.  That deletion is a human judgement about evidence; it is
deliberately not something an adapter can do to itself by growing a method.

Provider is not a comparability axis: `docs/ARCHITECTURE-DETERMINISM.md`
measured two A100s in two clouds agreeing bitwise while an H200 sat 2.973e-04
nats away, and no provider field appears in `COMPARABILITY_KEY_FIELDS`.  What a
comparison binds is the device model and the rebuilt stack.  So parity here is
science rather than convenience, and the twelve exist because we cannot
otherwise PROVE what we rented, PROVE it is gone, or RECONCILE what it cost.
`docs/PROVIDER-PARITY.md` carries the per-provider detail and the definition of
done; this module is the machine-readable half of that document.

Stdlib only, and no adapter is imported at module import time: a controller
startup path must not pay for four provider clients, and importing an adapter
must never be a precondition for reading the table.
"""
from __future__ import annotations
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PARITY_DOC = "docs/PROVIDER-PARITY.md"

#: Every provider name the CLI accepts.  `--provider` has NO default and must
#: keep none: every path that reads it either spends on a provider account or
#: sweeps one wholesale, and a guessed account is how a run bills the wrong
#: person (commit fcb9470 removed a `default="runpod"` after it made two
#: refusals unreachable).
PROVIDERS: Tuple[str, ...] = ("jarvislabs", "lambda", "runpod", "vast")

#: The twelve methods that separate a provider you can RENT from one you can
#: PUBLISH from, read from `bin/fidelity/runpodapi.py` -- the reference
#: implementation.  Four prove the live resource is the one requested (and, via
#: attest_live_resource, that it is the DEVICE the root was captured on), four
#: prove nothing of ours is still alive, four reconcile what it cost against
#: the provider's own clock and billing.
PROVIDER_CONTRACT: Tuple[str, ...] = (
    # is this the thing I asked for?
    "prepare_safe_create", "submit_prepared_create",
    "validate_safe_resource_binding", "attest_live_resource",
    # is anything of mine still alive?
    "list_lifecycle_resources", "get_lifecycle_resource",
    "list_network_volumes", "chargeable_inventory",
    # what did it cost, and whose clock says so?
    "server_time_evidence", "ssh_host_ed25519_fingerprint",
    "billing_history", "reconcile_billing",
)

#: The part of the contract the autonomous reaper sweep depends on, plus the
#: easy-surface methods it needs.  `reap_once` calls `status`,
#: `list_instances`, `destroy`, `chargeable_inventory` and `reconcile_billing`
#: directly; `list_network_volumes` and `billing_history` are what the last
#: two must be BUILT ON, and a sweep whose inventory omits volumes cannot
#: prove absence on a provider whose filesystems outlive their instances
#: (Lambda and JarvisLabs both do; Vast's storage is pod-scoped, so an empty
#: volume family there is correct rather than missing).  A provider with these
#: gets a working per-provider sweep with no further controller change; one
#: without them is refused, because WITHOUT AN AUTONOMOUS TEARDOWN BACKSTOP NO
#: PAID MEASUREMENT MAY RUN ON A PROVIDER AT ALL -- a controller death leaks a
#: billing instance.
SWEEP_CONTRACT: Tuple[str, ...] = (
    "list_network_volumes", "chargeable_inventory",
    "billing_history", "reconcile_billing",
)
SWEEP_BASE: Tuple[str, ...] = ("status", "list_instances", "destroy")

#: JarvisLabs additionally keeps a pre-lease sweep (`measure-cloud reaper
#: --provider jarvislabs --sweep`) for the 137 settled and 1 operator-needing
#: leases written before the lease store existed.  It is a legacy path, kept
#: beside the generic sweep until the generic one is proven against an old
#: lease shape -- not a second way to reap a new lease.
LEGACY_SWEEP_PROVIDERS: Tuple[str, ...] = ("jarvislabs",)

#: provider -> the attribute in `bin/measure_cloud.py` that runs a paid
#: measurement on it.  Resolved against the module object by
#: `execution_entrypoint`, so the table cannot name a function that does not
#: exist and this module never imports the controller.
#:
#: ONE function serves every provider.  It was `_main_runpod`, reachable only
#: for RunPod, which meant a provider could reach twelve-of-twelve with an
#: empty blocker tuple and STILL have nowhere to execute -- a refusal with no
#: remedy.  The lifecycle now takes its provider-specific values from the
#: twelve contract methods and from `PAID_EXECUTION_PROFILES`, and contains no
#: provider name and no per-provider branch, so a second provider's paid path
#: is an adapter plus a profile row and NOT a lifecycle edit.  Every provider
#: is registered here on purpose: a name missing from this table would refuse
#: for the wrong reason (a controller gap) instead of the real one (its own
#: unmet safety properties and blockers).
EXECUTION_ENTRYPOINTS: Dict[str, str] = {
    "jarvislabs": "_main_paid",
    "lambda": "_main_paid",
    "runpod": "_main_paid",
    "vast": "_main_paid",
}

_ADAPTERS: Dict[str, Tuple[str, str]] = {
    "runpod": ("fidelity.runpodapi", "RunPod"),
    "vast": ("fidelity.vastapi", "Vast"),
    "lambda": ("fidelity.lambdaapi", "LambdaCloud"),
    "jarvislabs": ("fidelity.jlapi", "JL"),
}

#: provider -> the environment variable naming its credential FILE.  The
#: secret is always passed as a path, never in argv (AGENTS.md: never put a
#: token in argv, a log, a receipt, a bundle or git).  JarvisLabs is absent on
#: purpose: it is CLI-driven (`jl`, reading JL_API_KEY from the environment)
#: and takes no key file, so a dispatch that assumed every adapter speaks
#: HTTP with a key file would be wrong for it.
KEY_FILE_ENV: Dict[str, str] = {
    "runpod": "RUNPOD_KEY_FILE",
    "vast": "VAST_KEY_FILE",
    "lambda": "LAMBDA_KEY_FILE",
}

#: The ONE credential transport a paid measurement may use.  The token is
#: written to the box as an owner-only file over an SSH channel whose host key
#: was authenticated first, and shredded at teardown.  Anything else is
#: refused by the shared lifecycle BEFORE the credential is read, which is why
#: this is a value in the profile and not a comment: Vast's container mode
#: puts `-e HF_TOKEN=...` into the `PUT /asks/{id}/` body
#: (`vastapi.py:2322-2330`), so the secret reaches the provider's own records
#: and the host's docker environment before an instance -- and therefore
#: before any host key, attestation or TLS check -- exists at all.  That is not
#: fixable by ordering, so the MECHANISM is refused rather than the payload
#: inspected.  The payload half is a separate, complementary guard at the
#: adapter boundary (`create()` refusing a credential-shaped body, rung RP7b);
#: for Vast container mode this profile gate fires FIRST, before any provider
#: call, and the adapter guard is the backstop if a caller bypasses the paid
#: path.
PAID_CREDENTIAL_TRANSPORT = "ssh-0600-file"

#: A host-key fingerprint is only evidence if it identifies THE RESOURCE WE
#: RENTED.  RunPod publishes the key in its own per-pod boot log, keyed by the
#: exact pod id, so the fingerprint is resource-attributable.  Measured on
#: Vast 2026-09-06 (T4Verdict, contracts 50055626 and 50056958 on machine
#: 150014, both the proxy and the direct port, and a different key on machine
#: 146304): the key belongs to the MACHINE, survives destroy-and-create, and
#: the host operator has root and holds it.  So a verified Vast SSH channel
#: today proves "hardware we have seen before", not "the instance we just
#: created" -- and an attacker who is the host operator is inside the
#: attribution, which is a scientific-integrity property and not only a
#: security one.  A paid run requires the stronger value.
PAID_HOST_KEY_ATTRIBUTION = "resource"

#: The facts the ONE shared paid lifecycle needs that no contract method
#: returns.  Every field is a provider FACT that cannot be derived from the
#: adapter class, and each row is cross-checked against the adapter by
#: `paid_execution_profile()`, so a row cannot claim a capability the adapter
#: contradicts.  A provider with no row is REFUSED and never defaulted: every
#: one of these has a wrong value that spends money or leaks a box, so there
#: is no safe default to fall back to.
#:
#: This would ideally be a thirteenth contract method, `paid_execution_profile()`
#: on the adapter, which is where a provider's self-description belongs; it is
#: a table here because the four adapter files are owned by other lanes today.
#: The cross-check is what keeps that stand-in honest.
PAID_EXECUTION_CONTRACT: Tuple[str, ...] = (
    "label",                        # how operator text names the provider
    "resource_family",              # provider-native compute family ("pods")
    "evidence_prefix",              # `fidelity-suite/<prefix>-...` schema stem
    "storage_layouts",              # --storage-layout -> run_base/volume
    "secrets_dir",                  # callable(fs_root) -> 0700 secret dir
    "credential_transport",         # MUST be PAID_CREDENTIAL_TRANSPORT
    "host_key_attribution",         # MUST be PAID_HOST_KEY_ATTRIBUTION
    "safe_create_profile",          # the frozen create shape, field by field
    "provider_enforced_deadline",   # binding kwarg name, or None
    "server_time_origin",           # whose clock the deadline is encoded to
    "prepared_create",              # frozen-request evidence identity
    "balance_source",               # the exact account figure the gate reads
    "cost_model",                   # the price sheet the quote is built from
)

PAID_EXECUTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "runpod": {
        "label": "RunPod",
        "resource_family": "pods",
        "evidence_prefix": "runpod",
        # Measured 2026-09-03 on a secure H200 (us-co-1): the pod VOLUME at
        # /workspace is MooseFS over FUSE, shared, mode-forcing, ~215 MB/s
        # under contention; the CONTAINER disk is host NVMe at 3.5 GB/s.  The
        # run root belongs on the container disk, and a nominal volume is
        # still created so the attestation's /workspace probe keeps meaning.
        "storage_layouts": {
            "container-disk": {"run_base": "/root", "nominal_volume_gb": 10},
            "pod-volume": {"run_base": "/workspace", "nominal_volume_gb": None},
        },
        # NOT under the run root: /workspace accepted `chmod 600` while
        # reporting 0666 (Fruit smoke, 2026-09-03), so the 0700/0600 contract
        # lives on the container disk, keyed by the attempt.
        "secrets_dir": "/root/.fidelity-secrets/%s",
        "credential_transport": PAID_CREDENTIAL_TRANSPORT,
        "host_key_attribution": PAID_HOST_KEY_ATTRIBUTION,
        "safe_create_profile": {
            "secure_cloud": True, "offer": "on-demand", "spot": False,
            "gpu_count": 1, "ports": "22/tcp",
            "volume_mount_path": "/workspace", "network_volume": None,
        },
        "provider_enforced_deadline": "terminate_after",
        "server_time_origin": "https://api.runpod.io",
        "prepared_create": {
            "schema": "fidelity-suite/runpod-prepared-create.v1",
            "body_field_prefix": "graphql_body",
            "body_shape": "graphql-query",
            "body_must_contain": "podFindAndDeployOnDemand",
        },
        "balance_source": "RunPod myself.clientBalance",
        "cost_model": {
            "tariff_flag_prefix": "runpod",
            "storage_month_hours": 672,
            "network_billing_increment_seconds": 3600,
        },
    },
}

#: The properties RunPod's paid path already enforces, each of which was paid
#: for, and what ENFORCES each one.  `unmet_safety_properties()` derives the
#: verdict, so a provider that cannot meet one is refused by computation.  The
#: rule the table encodes: A PROPERTY RUNPOD MEETS THAT ANOTHER PROVIDER
#: CANNOT IS A BLOCKER FOR THAT PROVIDER, NEVER A RELAXATION OF THE PATH.
#:
#: `methods` are contract methods that must exist on the adapter; `claims` are
#: (profile field, required value) pairs; `derived` is a profile field that
#: must merely be present and non-empty.
SAFETY_PROPERTIES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "two-phase-create",
        "statement": "the create request is built and frozen before any "
                     "provider mutation, and submitted separately, so a LOST "
                     "create RESPONSE is reconcilable instead of ambiguous",
        "methods": ("prepare_safe_create", "submit_prepared_create"),
        "claims": (),
        "derived": ("prepared_create",),
    },
    {
        "id": "host-key-before-ssh",
        "statement": "the host key is authenticated out of band, against the "
                     "exact resource id, before any ssh is spawned "
                     "(sshbase._known_hosts_file() RAISES otherwise)",
        "methods": ("ssh_host_ed25519_fingerprint",),
        "claims": (("host_key_attribution", PAID_HOST_KEY_ATTRIBUTION),),
        "derived": (),
    },
    {
        "id": "attestation-before-upload",
        "statement": "the live resource is proven to BE the resource "
                     "requested and the DEVICE the root was captured on, "
                     "before any upload or workload spend",
        "methods": ("validate_safe_resource_binding", "attest_live_resource"),
        "claims": (),
        "derived": (),
    },
    {
        "id": "credential-transport",
        "statement": "the credential reaches the box only as an owner-only "
                     "file over an already-authenticated channel, never in a "
                     "provider create body",
        "methods": ("upload", "exec"),
        "claims": (("credential_transport", PAID_CREDENTIAL_TRANSPORT),),
        "derived": ("secrets_dir",),
    },
    {
        "id": "provider-enforced-deadline",
        "statement": "the teardown deadline is enforced by the PROVIDER and "
                     "encoded against the provider's own clock, so it "
                     "survives simultaneous loss of the controller host and "
                     "of the instance OS -- the one failure the controller, "
                     "the reaper and the on-instance watchdog all share",
        "methods": ("server_time_evidence",),
        "claims": (),
        "derived": ("provider_enforced_deadline", "server_time_origin"),
    },
    {
        "id": "absence-from-authoritative-inventory",
        "statement": "absence is proven from a COMPLETE chargeable "
                     "inventory, instances and volumes, and the cost closed "
                     "against the provider's own billing",
        "methods": ("list_lifecycle_resources", "list_network_volumes",
                    "chargeable_inventory", "billing_history",
                    "reconcile_billing", "get_lifecycle_resource"),
        "claims": (),
        "derived": (),
    },
)

#: Definition-of-done items 3 and 6 (docs/PROVIDER-PARITY.md): "only after 3
#: and 6 may a paid measurement run there at all".  Item 3 is COMPUTED from
#: the lease store the run will itself write to -- a TERMINAL lease for this
#: provider whose terminal proof carries an authoritative absence proof and a
#: reconciled billing closure -- so a provider EARNS it by settling a real
#: lease, with no table edit.  The declaration below is the recorded
#: historical citation, used when this box's store cannot answer (a fresh
#: checkout has no lease history, and the property was proven on the ACCOUNT,
#: not on the box).
#:
#: Item 6 cannot be computed: "teardown proven on success, failure, exception
#: and interrupt" is an operational fact about four code paths having actually
#: run.  All four names are required; a partial record is refused.
PAID_PREREQUISITES: Tuple[Tuple[str, str], ...] = (
    ("sweep_settled_lease", "item 3: a reaper sweep that settled a lease and "
                            "proved absence from a complete chargeable "
                            "inventory"),
    ("teardown_proven", "item 6: teardown proven on success, failure, "
                        "exception and interrupt"),
)
TEARDOWN_PROOF_PATHS: Tuple[str, ...] = (
    "success", "failure", "exception", "interrupt")
PAID_PREREQUISITE_EVIDENCE: Dict[str, Dict[str, Any]] = {
    "runpod": {
        # Counted from ~/.fidelity-cloud/leases-v2 on 2026-09-06: 153 TERMINAL
        # leases, every one carrying a terminal proof with an absence proof
        # and a reconciled billing closure (hour-bucketed pod billing), plus
        # 1 ABSENCE_CONFIRMED and 2 AMBIGUOUS-needs-operator.
        "sweep_settled_lease":
            "153 TERMINAL RunPod leases in the v2 lease store, each with an "
            "absence proof from a complete chargeable inventory and a "
            "reconciled billing closure; bin/selftest_reaper.py drives the "
            "same sweep offline",
        # From the same store's histories: 100 DESTROY_REQUESTED with reason
        # "controller exit" (success), 12 "controller failure before provider
        # POST", 4 "controller failed before intentional loss; immediate
        # cleanup", 7 "absolute reap deadline expired" and 1 "controller
        # process lost" (the controller-loss drill).  The interrupt path is
        # the SIGINT/SIGTERM/SIGHUP handlers installed by `_main_paid` before
        # the first spend, which raise KeyboardInterrupt into the same
        # guaranteed-teardown `finally`.
        "teardown_proven": {
            "success": "100 leases destroyed with reason 'controller exit' "
                       "after WORKLOAD_EXITED",
            "failure": "12 leases closed with 'controller failure before "
                       "provider POST' and 4 with 'controller failed before "
                       "intentional loss; immediate cleanup'",
            "exception": "the execute path's teardown runs from a `finally`, "
                         "and `_main_paid` classifies a BaseException as a "
                         "refusal only when no POST intent was ever fsynced; "
                         "asserted offline by bin/selftest_runpod_safe.py",
            "interrupt": "SIGINT/SIGTERM/SIGHUP are installed before the "
                         "credential is read and raise into the same "
                         "`finally`; the controller-loss drill "
                         "(bin/fidelity/runpoddrill.py) settled 1 lease with "
                         "reason 'controller process lost' and 7 more at the "
                         "absolute reap deadline",
        },
    },
}

#: What is NOT derivable, and only that.  Delete a line when its evidence
#: exists; never when you would like it to.  Numbered items refer to the
#: definition of done in docs/PROVIDER-PARITY.md.
#:
#: Definition-of-done items 3 and 6 used to be hand-written lines here for
#: three providers.  They are COMPUTED now (`missing_paid_prerequisites`:
#: item 3 from a settled absence-proven lease in the store, item 6 from the
#: four named teardown paths), so the lines are gone -- a tuple carrying a
#: stale claim is a tuple a human learns to ignore.  Item 4 (a dry-run
#: reaching a cost quote with zero refusals) is not here either, and not
#: because it is unimportant: it is enforced STRUCTURALLY, because the shared
#: paid lifecycle runs the plan first and cannot reach a create without a
#: refusal-free quote.
PROVIDER_BLOCKERS: Dict[str, Tuple[str, ...]] = {
    "runpod": (),
    "vast": (
        "no --dry-run has reached a Vast cost quote with zero refusals "
        "(item 4), and no paid Vast capture has been compared against "
        "malaiwah/fruit-fidelity-root-v1 (item 5)",
        # Not a flaky host. A certificate hostname mismatch is the signature
        # of a MITM TLS proxy, and we put an HF token on rented boxes; the
        # run failed at setup instead of leaking only because verification
        # was on. Treat such a host as hostile until proven otherwise:
        # destroy, record the id, and never retry a credential-bearing
        # operation against it.
        "host quality varies and a bad host is HOSTILE, not broken: a Nevada "
        "host's SSL proxy to huggingface.co presented a certificate hostname "
        "mismatch and UNEXPECTED_EOF and failed a capture at the setup stage "
        "(2026-09-05). A Vast lane needs peer attestation -- not merely "
        "reachability -- before it does real work, and the host id recorded",
        "container mode is PROHIBITED for credential-bearing runs and this "
        "is not fixable by ordering: vastapi.py:2322-2330 builds `-e "
        "HF_TOKEN=...` into the `PUT /asks/{id}/` body, so the credential "
        "enters Vast's own records and the host's docker environment BEFORE "
        "the instance exists -- there is nothing to attest yet. Vast may "
        "measure PUBLIC artifacts with no token at all (capacity and "
        "rehearsal); a credential-bearing Vast run waits on create() "
        "refusing a credential-shaped payload at the adapter boundary",
        "the contract rate is not the advertised rate: a live T4 contract "
        "billed $0.16667/h against an ask listing $0.13556/h (23% high), so "
        "billing_history/reconcile_billing must read what BILLS, and that has "
        "not been reconciled against a real Vast invoice yet",
    ),
    "lambda": (
        "no Lambda API credential on this box, so no part of the port is "
        "live-verified against the provider",
        "the fit arithmetic mis-sized a root plan at 63 GB/GPU during the "
        "GH200 qualification and refused hardware that would have worked "
        "(docs/REVIEW-DEFERRED.md); a first Lambda root would be refused for "
        "a phantom requirement until that is fixed",
        "no --dry-run has reached a Lambda cost quote with zero refusals "
        "(item 4), and no paid Lambda capture has been compared against "
        "malaiwah/fruit-fidelity-root-v1 (item 5)",
        # Verbatim from LambdaParity via Main, and deliberately not
        # paraphrased: this tuple is what a future operator reads to decide
        # whether to trust a receipt.
        "no provider-enforced termination deadline: Lambda's launch has no "
        "terminateAfter, so the deadline is controller clock plus the "
        "on-instance watchdog",
        "host-key pinning via cloud-init user_data is unverified on a live "
        "Lambda image",
    ),
    "jarvislabs": (
        # The vendor CLI authenticates NO host, ever, and forgets nothing
        # between calls: probed from the installed package, not from docs.
        # Named at file:line so a future reader can re-check the claim.
        "`jl` verifies no host key: jarvislabs/ssh.py:22-30 sets "
        "StrictHostKeyChecking=no AND UserKnownHostsFile=/dev/null, and "
        "cli/instance.py drives exec/upload/download/ssh through "
        "subprocess.call on those options. Two consequences, and the SECOND "
        "decides this: an HF token transits a host we never authenticate; "
        "and the result archive AND its on-pod sha256 both return over that "
        "same channel, so verify_transfer compares attacker-suppliable "
        "against attacker-suppliable and proves internal consistency rather "
        "than provenance. A MEASUREMENT RETRIEVED OVER AN UNAUTHENTICATED "
        "CHANNEL IS NOT ATTRIBUTABLE TO THE MACHINE WE RENTED -- a scientific "
        "integrity property, not only a security one",
        "host-key pinning by construction (fresh ED25519 per create via `jl "
        "scripts add` + --script-id, expected fingerprint and script digest "
        "frozen into the request identity) does not clear the line above on "
        "its own: a pin is worthless while the transport ignores keys, so a "
        "verifying ssh invocation of OUR OWN is the missing half. Until it "
        "exists, JarvisLabs may measure PUBLIC artifacts with no credential "
        "on the box -- useful for capacity and rehearsal -- but such a run is "
        "NOT publishable, for the attribution reason above",
        "`jl` status exposes no account identity, which the generic lease "
        "sweep requires in order to name the account it is acting on; the "
        "legacy JL sweep therefore still owns this provider",
        "137 settled and 1 operator-needing legacy lease predate the lease "
        "store; the legacy JL sweep still owns them and the generic sweep is "
        "unproven against an old lease shape",
        "no paid JarvisLabs capture has been compared against "
        "malaiwah/fruit-fidelity-root-v1 (item 5)",
    ),
}

#: NAMED DEGRADATIONS: true, disclosed, and deliberately NOT blocking.
#:
#: The distinction is a ruling, not a convenience.  JarvisLabs cannot
#: reconcile a cost after destroy -- `jl 0.2.17` has no billing subcommand
#: and no time-windowed query -- and that was first written here as a
#: blocker.  It is not one: the registry publishes no cost field, so cost
#: reconciliation protects the OPERATOR, not the number.  A provider may
#: therefore host an official measurement with a degradation, provided the
#: degradation is stated in these words rather than papered over, and
#: provided nothing invents a number to fill the gap.
#:
#: A degradation is disclosed before spend by `measure-cloud`; it never
#: changes `measurement_refusal`.  Promoting one to a blocker, or retiring
#: one, is the same human judgement as deleting a blocker line.
PROVIDER_DEGRADATIONS: Dict[str, Tuple[str, ...]] = {
    "runpod": (),
    "vast": (
        "billing is day-granular, so a lease seals settled: false and a "
        "next-day sweep closes it -- the same later-sweep settlement RunPod "
        "already uses, with a ~24h window instead of ~1h; a scheduling fact, "
        "not a defect",
        "a status field is a claim about the provider's INTENT, never "
        "evidence of reachability: `actual_status=running` is a contract "
        "state and was observed for ~14 minutes on a healthy host whose "
        "reverse SSH tunnel was dead, so reachability must be probed and "
        "never read from a status field",
    ),
    "lambda": (),
    "jarvislabs": (
        "cost is unreconcilable-after-destroy BY DESIGN: `jl 0.2.17` has no "
        "billing subcommand and no time-windowed query, so the only "
        "per-resource figure is a running total readable while the instance "
        "is still listed. A lease therefore seals settled: false with a "
        "pre-destroy cost_snapshot and unreconcilable_by_provider: true, and "
        "NO local arithmetic is added -- an hourly-billed residual the "
        "provider never prices must stay unpriced, because a computed cost "
        "that looks settled is worse than an honest gap",
        "no provider clock: `server_time_evidence` refuses, so a teardown "
        "deadline rests on the instance clock plus ours rather than a "
        "provider-attested deadline. That is survivable only because the "
        "on-instance watchdog is the real backstop and runs on the box's own "
        "clock; our clock is never passed off as theirs",
    ),
}


class ProviderRefusal(object):
    """A refusal with a reason and actionable advice, framework-free.

    `bin/measure_cloud.py` has its own `Refusal`; this module is imported by
    the controller AND by the selftest, so it must not depend on either.
    """

    __slots__ = ("reason", "advice")

    def __init__(self, reason: str, advice: Optional[List[str]] = None) -> None:
        self.reason = reason
        self.advice = list(advice or [])

    def __str__(self) -> str:
        return "; ".join([self.reason] + self.advice)

    def __repr__(self) -> str:                                # pragma: no cover
        return "ProviderRefusal(%r, %r)" % (self.reason, self.advice)


def _known(name: str) -> str:
    if name not in PROVIDERS:
        raise KeyError(
            "unknown provider %r; the CLI accepts %s"
            % (name, ", ".join(PROVIDERS)))
    return name


def adapter_class(name: str) -> Any:
    """Import the adapter CLASS (never an instance: no credential is read)."""
    module_name, attribute = _ADAPTERS[_known(name)]
    return getattr(importlib.import_module(module_name), attribute)


def key_file_path(name: str, explicit: Optional[str] = None) -> Optional[str]:
    """Absolute credential-file path for a provider, or None if it needs none.

    Order: an explicit flag, then the provider's environment variable, then
    the adapter's own `DEFAULT_KEY_FILE` if it declares one.  Raises when a
    key-file provider has no resolvable path -- guessing a credential is how a
    sweep touches the wrong account.
    """
    if _known(name) not in KEY_FILE_ENV:
        return None
    module_name = _ADAPTERS[name][0]
    module = importlib.import_module(module_name)
    selected = (
        explicit or os.environ.get(KEY_FILE_ENV[name])
        or getattr(module, "DEFAULT_KEY_FILE", None) or "")
    if not selected:
        raise KeyError(
            "no %s credential path: set %s to a 0600 key file"
            % (name, KEY_FILE_ENV[name]))
    return str(Path(str(selected)).expanduser().resolve())


def degradations(name: str) -> Tuple[str, ...]:
    """Named, disclosed, non-blocking losses of guarantee for this provider."""
    return tuple(PROVIDER_DEGRADATIONS[_known(name)])


def _profile_claim(profile: Dict[str, Any], field: str) -> Any:
    return profile.get(field)


def paid_execution_profile(name: str) -> Dict[str, Any]:
    """The provider's declared paid execution profile, cross-checked.

    Raises `KeyError` when no row exists (a provider with no profile is
    REFUSED, never defaulted: every field has a wrong value that spends money
    or leaks a box) and `ValueError` when the row is incomplete or contradicts
    the adapter.  The cross-check is what stops a table from claiming a
    capability the code does not have -- the same rule that makes method
    conformance computed rather than declared.
    """
    row = PAID_EXECUTION_PROFILES.get(_known(name))
    if row is None:
        raise KeyError(
            "no paid execution profile is declared for %s" % name)
    missing = [field for field in PAID_EXECUTION_CONTRACT if field not in row]
    unexpected = [field for field in row
                  if field not in PAID_EXECUTION_CONTRACT]
    if missing or unexpected:
        raise ValueError(
            "%s paid execution profile keys differ: missing=%s unexpected=%s"
            % (name, sorted(missing), sorted(unexpected)))
    deadline = row["provider_enforced_deadline"]
    if deadline is not None:
        # The claim being checked is "this adapter can be CALLED with the
        # deadline it says the provider enforces".  An explicit parameter
        # satisfies that; so does `**kw`, which genuinely accepts it.  A
        # signature that accepts neither is a row claiming a guarantee the
        # code cannot express, and that is what this refuses.
        binding = getattr(
            adapter_class(name), "validate_safe_resource_binding", None)
        try:
            parameters = inspect.signature(binding).parameters
        except (TypeError, ValueError):                   # pragma: no cover
            parameters = {}
        binds = deadline in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values())
        if not binds:
            raise ValueError(
                "%s declares a provider-enforced deadline %r that its "
                "validate_safe_resource_binding cannot be called with"
                % (name, deadline))
    # The campaign ledger projects a lease's resources onto a FIXED family
    # set (`campaign._RESOURCE_FAMILIES` == {"pods", "network_volumes"}), and
    # every paid run is campaign-bound.  A provider whose compute family is
    # named anything else -- Vast calls them instances -- would KeyError deep
    # inside campaign accounting AFTER the create.  Refuse it here instead,
    # before any spend, naming the file that has to change.
    from .campaign import _RESOURCE_FAMILIES
    if row["resource_family"] not in _RESOURCE_FAMILIES:
        raise ValueError(
            "%s declares resource_family %r, and campaign accounting "
            "projects only onto %s: generalising that is a change in "
            "bin/fidelity/campaign.py, not a value this table may pick"
            % (name, row["resource_family"],
               ", ".join(sorted(_RESOURCE_FAMILIES))))
    # The result verifier and the on-box watchdog verifier BOTH pin the
    # RunPod evidence names: `resultsink.RUNPOD_ATTESTATION_PATH` is
    # `receipts/runpod-live-attestation.json` and `runpodsafety.py:759`
    # compares against `fidelity-suite/runpod-ssh-host-key-proof.v2`.  The
    # paid path derives those names from `evidence_prefix`, so a provider
    # with a different prefix would upload evidence the verifier then cannot
    # find -- AFTER the create, mid-run.  Refuse it here, before any spend,
    # naming both files.  This is the honest boundary of the generalisation:
    # neither module is mine to change, and a late failure dressed as
    # readiness is worse than an early refusal that names the work.
    from . import resultsink
    expected_path = "receipts/%s-live-attestation.json" % row["evidence_prefix"]
    if resultsink.RUNPOD_ATTESTATION_PATH != expected_path:
        raise ValueError(
            "%s declares evidence_prefix %r, so the paid path would upload "
            "%s while bin/fidelity/resultsink.py verifies %r and "
            "bin/fidelity/runpodsafety.py pins the runpod host-key-proof "
            "schema: both must be parameterised by provider before a "
            "non-RunPod paid run can verify its own result"
            % (name, row["evidence_prefix"], expected_path,
               resultsink.RUNPOD_ATTESTATION_PATH))
    return dict(row)


def profile_refusal_reason(name: str) -> Optional[str]:
    """Why this provider has no usable paid execution profile, or None."""
    try:
        paid_execution_profile(name)
    except (KeyError, ValueError) as exc:
        return str(exc)
    return None


def unmet_safety_properties(name: str) -> Tuple[Tuple[str, str], ...]:
    """Which enumerated safety properties this provider cannot meet, and why.

    Derived from the adapter class and the profile row, so a provider that
    cannot meet one is refused by computation.  A property RunPod meets that
    another provider cannot is a BLOCKER for that provider; nothing here ever
    relaxes a property to admit one.

    A provider with no profile row is reported by `profile_refusal_reason`
    once, not repeated under every property that reads the profile: the
    method half is still evaluated, because a missing method is a different
    and equally real gap.
    """
    _known(name)
    try:
        profile: Dict[str, Any] = paid_execution_profile(name)
        have_profile = True
    except (KeyError, ValueError):
        profile, have_profile = {}, False
    unmet: List[Tuple[str, str]] = []
    for prop in SAFETY_PROPERTIES:
        absent = _missing(name, tuple(prop["methods"]))
        if absent:
            unmet.append((prop["id"], "the adapter is missing %s"
                          % ", ".join(absent)))
            continue
        if not have_profile:
            continue
        broken = None
        for field, required in prop["claims"]:
            observed = _profile_claim(profile, field)
            if observed != required:
                broken = ("its profile declares %s=%r, and a paid run "
                          "requires %r" % (field, observed, required))
                break
        for field in () if broken else prop["derived"]:
            if not _profile_claim(profile, field):
                broken = "its profile declares no %s" % field
                break
        if broken:
            unmet.append((prop["id"], broken))
    return tuple(unmet)


def _settled_lease_in_store(name: str, lease_dir: Any) -> bool:
    """True iff this store already holds a settled, absence-proven lease.

    Item 3 EARNED rather than declared: a TERMINAL lease for this provider
    whose terminal proof carries an authoritative absence proof and a
    reconciled billing closure.  Read-only, and a store we cannot read is
    simply no evidence -- never an error, because this is one of two
    admissible sources.
    """
    try:
        root = Path(str(lease_dir)).expanduser()
        entries = sorted(root.glob("*.json"))
    except OSError:                                       # pragma: no cover
        return False
    for path in entries:
        try:
            document = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if not isinstance(document, dict) or document.get("state") != "TERMINAL":
            continue
        if ((document.get("create") or {}).get("provider")) != name:
            continue
        proof = document.get("terminal_proof") or {}
        if not isinstance(proof, dict):
            continue
        # The exact key names a sealed lease carries: `provider_absence`
        # holds the authoritative inventory the absence was proven from, and
        # `billing_reconciliation.reconciled` is the stabilized cost closure.
        absence = proof.get("provider_absence") or {}
        billing = proof.get("billing_reconciliation")
        if (isinstance(absence, dict)
                and absence.get("authoritative_inventory")
                and absence.get("complete_listing")
                and isinstance(billing, dict) and billing.get("reconciled")):
            return True
    return False


def missing_paid_prerequisites(
        name: str, lease_dir: Any = None) -> Tuple[Tuple[str, str], ...]:
    """Definition-of-done items 3 and 6 that this provider has not proven.

    Item 3 is computed from the lease store the run will itself write to, and
    falls back to the recorded historical citation (the property was proven on
    the ACCOUNT, not on the box, so a fresh checkout must not un-prove it).
    Item 6 is declared and needs all four named teardown paths; a partial
    record is refused rather than rounded up.
    """
    evidence = PAID_PREREQUISITE_EVIDENCE.get(_known(name)) or {}
    unproven: List[Tuple[str, str]] = []
    for key, statement in PAID_PREREQUISITES:
        recorded = evidence.get(key)
        if key == "sweep_settled_lease":
            if lease_dir is not None and _settled_lease_in_store(name, lease_dir):
                continue
            if not recorded:
                unproven.append((key, statement))
            continue
        if not isinstance(recorded, dict):
            unproven.append((key, statement))
            continue
        absent = [path for path in TEARDOWN_PROOF_PATHS
                  if not recorded.get(path)]
        if absent:
            unproven.append(
                (key, "%s (unproven: %s)" % (statement, ", ".join(absent))))
    return tuple(unproven)

def _missing(name: str, methods: Tuple[str, ...]) -> Tuple[str, ...]:
    adapter = adapter_class(name)
    return tuple(
        method for method in methods
        if not callable(getattr(adapter, method, None)))


def missing_contract_methods(name: str) -> Tuple[str, ...]:
    """Which of the twelve the adapter does not have.  Computed, never declared.

    `callable` cannot tell a real implementation from one that refuses -- that
    is what the human blocker tuple is for.  JarvisLabs deliberately ships
    `billing_history` as a refusal because `jl` has no such read, and its
    blocker line says so.
    """
    return _missing(name, PROVIDER_CONTRACT)


def missing_sweep_methods(name: str) -> Tuple[str, ...]:
    return _missing(name, SWEEP_BASE + SWEEP_CONTRACT)


def blockers(name: str) -> Tuple[str, ...]:
    return tuple(PROVIDER_BLOCKERS[_known(name)])


def sweep_refusal(name: str) -> Optional[ProviderRefusal]:
    """Whether the GENERIC lease sweep can drive this provider.

    Nothing here consults the blocker tuple.  Refusing to reap is itself a
    leak, so a sweep is admitted the moment the adapter can be driven through
    it -- a provider may be unfit to MEASURE on and still have to be cleaned
    up.
    """
    missing = missing_sweep_methods(_known(name))
    if missing:
        return ProviderRefusal(
            "the reaper cannot sweep %s: its adapter is missing %d of the %d "
            "methods a sweep drives (%s)"
            % (name, len(missing), len(SWEEP_BASE) + len(SWEEP_CONTRACT),
               ", ".join(missing)),
            ["implement them in bin/fidelity/%sapi.py against the provider's "
             "official API, with the same refuse-with-advice semantics as "
             "bin/fidelity/runpodapi.py" % name,
             "%s names every method and the per-provider blockers" % PARITY_DOC])
    return None


def reaper_refusal(name: str) -> Optional[ProviderRefusal]:
    """Whether `measure-cloud reaper --provider <name>` may run at all."""
    if _known(name) in LEGACY_SWEEP_PROVIDERS:
        return None
    return sweep_refusal(name)


def measurement_refusal(
        name: str, lease_dir: Any = None) -> Optional[ProviderRefusal]:
    """None iff a paid measurement may execute on this provider.

    The refusal names the REAL reason -- which methods are missing, which
    safety properties cannot be met, which definition-of-done item has no
    proof, which blockers remain -- rather than a hardcoded provider name.
    Four of the five reasons are COMPUTED; only the blocker tuple and the
    item-6 teardown record are declared.
    """
    _known(name)
    missing = missing_contract_methods(name)
    profile_gap = profile_refusal_reason(name)
    unmet = unmet_safety_properties(name)
    unproven = missing_paid_prerequisites(name, lease_dir)
    declared = blockers(name)
    reasons: List[str] = []
    advice: List[str] = []
    if missing:
        reasons.append(
            "%d of the %d contract methods are not implemented"
            % (len(missing), len(PROVIDER_CONTRACT)))
        advice.append("missing: " + ", ".join(missing))
    if profile_gap is not None:
        reasons.append("its paid execution profile is missing or invalid")
        advice.append("profile: %s" % profile_gap)
        advice.append(
            "the shared paid lifecycle needs %s; declare them in "
            "PAID_EXECUTION_PROFILES and they are cross-checked against the "
            "adapter, so a row cannot claim what the code contradicts"
            % ", ".join(PAID_EXECUTION_CONTRACT))
    if name not in EXECUTION_ENTRYPOINTS:
        # Not reachable while every provider is registered, and kept because
        # the predicate must stay true of a name someone adds tomorrow.
        reasons.append("no paid execution path is registered for it")
        advice.append(
            "bin/fidelity/providers.py EXECUTION_ENTRYPOINTS maps a provider "
            "to the shared paid lifecycle in bin/measure_cloud.py")
    if unmet:
        reasons.append(
            "%d of the %d enumerated safety propert%s cannot be met"
            % (len(unmet), len(SAFETY_PROPERTIES),
               "y" if len(unmet) == 1 else "ies"))
        for identifier, why in unmet:
            advice.append("safety property %s is unmet: %s" % (identifier, why))
    if unproven:
        reasons.append(
            "%d definition-of-done prerequisite%s unproven"
            % (len(unproven), " is" if len(unproven) == 1 else "s are"))
        for identifier, statement in unproven:
            advice.append("unproven (%s): %s" % (identifier, statement))
    for blocker in declared:
        advice.append("blocker: " + blocker)
    if declared:
        reasons.append(
            "%d declared blocker%s remain%s"
            % (len(declared), "" if len(declared) == 1 else "s",
               "s" if len(declared) == 1 else ""))
    if not reasons:
        return None
    advice.append(
        "%s carries the definition of done. The paid lifecycle itself is "
        "shared: a provider is enabled by implementing the twelve, declaring "
        "a paid execution profile that meets every safety property, proving "
        "items 3 and 6, and DELETING its blocker lines in "
        "bin/fidelity/providers.py -- which is a human judgement about "
        "evidence, not a lifecycle edit" % PARITY_DOC)
    return ProviderRefusal(
        "paid measurement execution is refused on %s: %s"
        % (name, "; ".join(reasons)), advice)


def measurement_ready(name: str, lease_dir: Any = None) -> bool:
    return measurement_refusal(name, lease_dir) is None


def execution_entrypoint(name: str, module: Any) -> Any:
    """Resolve the paid execution function against the controller module.

    The table names an ATTRIBUTE, and the module is passed in, so this module
    never imports the controller (which would be a cycle, and would import a
    second copy of it when the controller runs as `__main__`).
    """
    attribute = EXECUTION_ENTRYPOINTS.get(_known(name))
    if attribute is None:
        raise KeyError("no paid execution entrypoint for %s" % name)
    entry = getattr(module, attribute, None)
    if not callable(entry):
        raise KeyError(
            "execution entrypoint %s.%s for %s is missing"
            % (getattr(module, "__name__", "?"), attribute, name))
    return entry
