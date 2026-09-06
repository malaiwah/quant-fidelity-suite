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
    and its blocker tuple is empty                         (DECLARED)

Method conformance is a fact about the code, so it is computed from the adapter
class and never declared.  That is what makes the table safe to leave alone
while three ports land: an adapter reaching twelve-of-twelve needs no edit here
and stays correctly refused for whatever non-method blockers remain.  It also
means a table cannot claim conformance it does not have.

`PROVIDER_BLOCKERS` carries only the residue that no offline test can compute --
"no credential on this box", "teardown never proven on this provider", "no paid
capture has reproduced the published root".  Each entry is an explanatory
sentence, because a blocker nobody can read is a blocker nobody will clear, and
ENABLING A PROVIDER IS DONE BY DELETING A BLOCKER LINE.  That deletion is a
human judgement about evidence; it is deliberately not something an adapter can
do to itself by growing a method.

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
EXECUTION_ENTRYPOINTS: Dict[str, str] = {
    "runpod": "_main_runpod",
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

#: What is NOT derivable, and only that.  Delete a line when its evidence
#: exists; never when you would like it to.  Numbered items refer to the
#: definition of done in docs/PROVIDER-PARITY.md.
PROVIDER_BLOCKERS: Dict[str, Tuple[str, ...]] = {
    "runpod": (),
    "vast": (
        "no Vast lease has ever been swept end to end: the generic reaper "
        "sweep serves any adapter carrying the twelve, but a first paid run "
        "needs one proven settle-and-prove-absence pass on a real lease "
        "(item 3)",
        "teardown is unproven on Vast for success, failure, exception and "
        "interrupt (item 6)",
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
        "no Lambda lease has ever been swept end to end (item 3) and teardown "
        "is unproven for success, failure, exception and interrupt (item 6)",
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
        "unproven against an old lease shape (item 3)",
        "teardown is unproven under the lease store for success, failure, "
        "exception and interrupt (item 6), and no paid JarvisLabs capture has "
        "been compared against malaiwah/fruit-fidelity-root-v1 (item 5)",
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


def measurement_refusal(name: str) -> Optional[ProviderRefusal]:
    """None iff a paid measurement may execute on this provider.

    The refusal names the REAL reason -- the missing methods if any, the
    declared blockers if not -- rather than a hardcoded provider name.
    """
    _known(name)
    missing = missing_contract_methods(name)
    declared = blockers(name)
    reasons: List[str] = []
    advice: List[str] = []
    if missing:
        reasons.append(
            "%d of the %d contract methods are not implemented"
            % (len(missing), len(PROVIDER_CONTRACT)))
        advice.append("missing: " + ", ".join(missing))
    if name not in EXECUTION_ENTRYPOINTS:
        reasons.append("no paid execution path is implemented for it")
        advice.append(
            "bin/measure_cloud.py implements execution for %s only; a paid "
            "lease for another provider is also refused by "
            "cloudlease._validate_request, whose campaign-bound request "
            "policy is RunPod-exact"
            % ", ".join(sorted(EXECUTION_ENTRYPOINTS)))
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
        "%s carries the definition of done; a provider is enabled by "
        "implementing the twelve and DELETING its blocker lines in "
        "bin/fidelity/providers.py, which is a human judgement about evidence"
        % PARITY_DOC)
    return ProviderRefusal(
        "paid measurement execution is refused on %s: %s"
        % (name, "; ".join(reasons)), advice)


def measurement_ready(name: str) -> bool:
    return measurement_refusal(name) is None


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
