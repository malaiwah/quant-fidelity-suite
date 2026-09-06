#!/usr/bin/env python3
"""Offline conformance rungs for the Lambda Cloud provider contract.

    python3 bin/selftest_lambda_contract.py

WHAT THIS CAN AND CANNOT PROVE. There is no Lambda credential on the
controller this was written on (`LambdaCloud().available()` is False), so
`bin/fidelity/lambdaapi.py`'s twelve parity methods have never run against a
live Lambda account. Every fixture below is hand-built from the official
published OpenAPI document (`GET /api/v1/openapi.json`, version 1.10.0,
retrieved 2026-09-06) -- `Instance`, `InstanceStatus`, `InstanceType`,
`InstanceTypeSpecs`, `Filesystem`, `SSHKey`, `Region`, `InstanceLaunchResponse`
and the `{"error": {code, message, suggestion, request_id}}` envelope.

So these rungs prove SHAPE, ARITHMETIC and REFUSAL BEHAVIOUR: that the adapter
reads the documented fields, that it refuses rather than guesses, and that it
never fabricates a completeness claim, a cost or an authenticated host key.
They cannot prove Lambda answers the way its own schema says it does. That gap
is named in every affected docstring and in the report; an unverified
implementation labelled as verified is worse than none.

Nothing here contacts a network and nothing here creates anything. There is no
optional interpreter and nothing is conditionally skipped: every assertion
runs on stock python3 every time, so a PASS is full coverage of what it claims.
"""

from __future__ import annotations

import calendar
import datetime
import email.utils
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.jlapi import JLError                         # noqa: E402
from fidelity.lambdaapi import (  # noqa: E402
    LAMBDA_INSTANCE_STATUSES,
    LambdaCloud,
    LambdaCreateRejectedError,
    LambdaError,
    PreparedLambdaCreate,
    _read_key_file,
)

PASS: list = []
FAIL: list = []

CONTRACT = (
    "prepare_safe_create", "submit_prepared_create",
    "validate_safe_resource_binding", "attest_live_resource",
    "list_lifecycle_resources", "get_lifecycle_resource",
    "list_network_volumes", "chargeable_inventory",
    "server_time_evidence", "ssh_host_ed25519_fingerprint",
    "billing_history", "reconcile_billing",
)


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ")
          + name + (("  -- " + detail) if detail else ""))


def refuses(call, *args, **kw):
    """(refused, message). A return value counts as NOT refusing."""
    try:
        call(*args, **kw)
    except LambdaError as exc:
        return True, str(exc)
    except Exception as exc:                                # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, exc)
    return False, "returned instead of refusing"


REGION = {"name": "us-east-1", "description": "Virginia, USA"}
SPECS = {"vcpus": 30, "memory_gib": 200, "storage_gib": 512, "gpus": 1}
INSTANCE_TYPE = {
    "name": "gpu_1x_a100_sxm4",
    "description": "1x A100 (40 GB SXM4)",
    "gpu_description": "A100 (40 GB SXM4)",
    "price_cents_per_hour": 129,
    "specs": SPECS,
    "architecture": "x86_64",
}
LIVE_ID = "0920582c7ff041399e34823a0be62549"
GONE_ID = "ddaedf1b7a0e41ac981711504493b242"
ODD_ID = "1111582c7ff041399e34823a0be62549"


def instance(identifier=LIVE_ID, status="active", name="fidcloud-lambda-1",
             keys=("fidelity-controller",), filesystems=(), mounts=None):
    row = {
        "id": identifier,
        "name": name,
        "ip": "198.51.100.2",
        "private_ip": "10.0.2.100",
        "status": status,
        "ssh_key_names": list(keys),
        "file_system_names": list(filesystems),
        "region": REGION,
        "instance_type": INSTANCE_TYPE,
        "image": {"id": "43336648-096d-4cba-9aa2-f9bb7727639d",
                  "family": "ubuntu-lts"},
        "hostname": "headnode1",
        "jupyter_token": "03b7d30d9d3e4d8fa41657bc0d478c1b",
        "jupyter_url": "https://jupyter-x.lambdaspaces.com/?token=x",
    }
    if mounts is not None:
        row["file_system_mounts"] = list(mounts)
    return row


FILESYSTEM = {
    "id": "398578a2336b49079e74043f0bd2cfe8",
    "name": "fidelity-scratch",
    "mount_point": "/lambda/nfs/fidelity-scratch",
    "created": "2026-08-30T11:02:41.512345Z",
    "created_by": {"id": "3da5a70a57a7422ea8a7203f98b2198b",
                   "email": "operator@example.com", "status": "active"},
    "is_in_use": False,
    "region": REGION,
    "bytes_used": 41_231_552,
}


def envelope(data):
    return {"data": data}


def gmt_now() -> str:
    return email.utils.format_datetime(
        datetime.datetime.now(datetime.timezone.utc), usegmt=True)


class Stub(LambdaCloud):
    """A LambdaCloud whose HTTP layer replays official-schema fixtures.

    `_request` is the single seam every parity method reads through, so the
    replay is honest about which endpoint each answer came from. GET responses
    still run the REAL `_capture_server_time` against a synthesised HTTP
    `Date` header, so the strict-GMT parser is exercised rather than bypassed.
    """

    def __init__(self, responses=None, **kw):
        super().__init__(**kw)
        self.responses = dict(responses or {})
        self.calls: list = []
        self.date_header = None

    def _load_key(self) -> str:
        return "fixture-key-not-a-credential"

    def _request(self, method, path, body=None, *, query=None, timeout=90):
        self.calls.append((method, path, body, query))
        if method == "GET":
            header = (self.date_header if self.date_header is not None
                      else gmt_now())

            class Response:
                headers = {"Date": header}

            self._capture_server_time(Response(), "https://x.invalid" + path)
        key = (method, path)
        if key not in self.responses:
            error = LambdaError(
                "fixture has no %s %s (the test's fault, not the adapter's)"
                % (method, path))
            setattr(error, "status", 404)
            setattr(error, "code", "global/object-does-not-exist")
            raise error
        answer = self.responses[key]
        if isinstance(answer, Exception):
            raise answer
        return 200, answer


def keypair(directory: str) -> str:
    """A real ed25519 pair on disk: the adapter proves .pub matches private."""
    path = os.path.join(directory, "id_ed25519")
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "selftest",
         "-f", path],
        stdin=subprocess.DEVNULL, check=True, timeout=30)
    return path


def base_responses(instances=None, filesystems=None, keys=None):
    return {
        ("GET", "/instances"): envelope(
            instances if instances is not None else [instance()]),
        ("GET", "/instances/%s" % LIVE_ID): envelope(instance()),
        ("GET", "/file-systems"): envelope(
            filesystems if filesystems is not None else [FILESYSTEM]),
        ("GET", "/instance-types"): envelope({
            INSTANCE_TYPE["name"]: {
                "instance_type": INSTANCE_TYPE,
                "regions_with_capacity_available": [REGION],
            }}),
        ("GET", "/ssh-keys"): envelope(keys if keys is not None else []),
    }


def credential_rungs() -> None:
    print("\n[0a] THE CREDENTIAL FILE'S PROMISED MODE IS CHECKED")
    # This was a bare open().read() while the refusal text promised a 0600
    # file. An error message that asserts a guarantee the code does not check
    # is worse than silence, because it is read as evidence.
    with tempfile.TemporaryDirectory(prefix="lambda-key-") as vault:
        good = os.path.join(vault, "api_key")
        with open(good, "w", encoding="utf-8") as handle:
            handle.write("secret-key-value\n")
        os.chmod(good, 0o600)
        check("a 0600 owner-owned key file is read",
              _read_key_file(good) == "secret-key-value")
        os.chmod(good, 0o644)
        refused, message = refuses(_read_key_file, good)
        check("a 0644 key file is REFUSED, as the error text has always "
              "promised", refused and "mode 0600" in message)
        os.chmod(good, 0o600)
        os.symlink(good, os.path.join(vault, "link_key"))
        refused, _ = refuses(_read_key_file, os.path.join(vault, "link_key"))
        check("a symlinked key file is refused (O_NOFOLLOW)", refused)
        refused, _ = refuses(_read_key_file, "config/lambda/api_key")
        check("a relative key path is refused", refused)
        empty = os.path.join(vault, "empty_key")
        with open(empty, "w", encoding="utf-8") as handle:
            handle.write("")
        os.chmod(empty, 0o600)
        refused, message = refuses(_read_key_file, empty)
        check("an empty key file is refused rather than sending an empty "
              "Authorization header", refused and "empty" in message)
        refused, _ = refuses(_read_key_file, os.path.join(vault, "absent"))
        check("a missing key file is refused with its path", refused)
        os.environ["LAMBDA_KEY_FILE"] = good
        try:
            os.chmod(good, 0o640)
            refused, _ = refuses(LambdaCloud().require)
            check("the group-readable file is refused THROUGH the adapter, "
                  "not just the helper", refused)
            check("...so available() is False rather than half-configured",
                  LambdaCloud().available() is False)
            os.chmod(good, 0o600)
            check("and a correct file makes the adapter available",
                  LambdaCloud().available() is True)
        finally:
            os.environ.pop("LAMBDA_KEY_FILE", None)


def transport_rungs() -> None:
    print("\n[0b] THE TRANSPORT THIS ADAPTER INHERITS REALLY VERIFIES HOSTS")
    # The jarvislabs CLI sets StrictHostKeyChecking=no AND
    # UserKnownHostsFile=/dev/null (jarvislabs/ssh.py:22-30), so it
    # authenticates no host on any invocation -- and a pin the transport
    # ignores buys nothing. Lambda rides sshbase, so assert the opposite.
    #
    # This inspects the OPTION LIST the transport actually builds, not the
    # file's text. A grep for "UserKnownHostsFile=/dev/null" went red the
    # moment sshbase grew prose EXPLAINING the jarvislabs anti-pattern -- the
    # same trap the pgrep rung in selftest_provider_portability.py already
    # records: the file documents at length why the bad thing is bad, and
    # that prose is the point. A rung that cannot tell a warning from the
    # thing it warns about is the rung under review.
    probe = LambdaCloud(ssh_key="/nonexistent")
    # The refusal comes from the shared transport, so it is the base
    # `JLError` family rather than LambdaError -- caught explicitly rather
    # than widened, because which layer refuses is part of the claim.
    try:
        probe._known_hosts_file()
        transport_refusal = ""
    except JLError as exc:
        transport_refusal = str(exc)
    check("a Lambda exec is IMPOSSIBLE before a host key is authenticated: "
          "the transport refuses to build ssh options at all",
          "has not been authenticated" in transport_refusal,
          transport_refusal[:70])
    with tempfile.TemporaryDirectory(prefix="lambda-hosts-") as hosts:
        authenticated = os.path.join(hosts, "known_hosts")
        with open(authenticated, "w", encoding="utf-8") as handle:
            handle.write("198.51.100.2 ssh-ed25519 AAAA\n")
        # The transport requires the per-attempt file to be an owner-only
        # regular file, and refuses it otherwise -- assert by satisfying it.
        os.chmod(authenticated, 0o600)
        probe.set_known_hosts(authenticated)
        options = probe._ssh_opts()
    check("once authenticated, the options pin strict checking",
          "StrictHostKeyChecking=yes" in options)
    check("...at the per-attempt known_hosts this run wrote, never /dev/null",
          "UserKnownHostsFile=%s" % authenticated in options
          and not any("/dev/null" in item and "UserKnownHosts" in item
                      for item in options))
    check("...with the global known_hosts neutralised and host keys pinned "
          "to ed25519, so a downgrade cannot be negotiated",
          "GlobalKnownHostsFile=/dev/null" in options
          and "HostKeyAlgorithms=ssh-ed25519" in options)


def clock_rungs() -> None:
    print("\n[1] WHOSE CLOCK SAYS SO")
    provider = Stub(base_responses())
    refused, message = refuses(provider.server_time_evidence)
    check("server time refuses before any authenticated read", refused)
    check("...and the refusal names what to call first",
          "authenticated read" in message, message[:80])
    provider.list_lifecycle_resources()
    evidence = provider.server_time_evidence()
    check("an authenticated read captures the provider's clock",
          evidence["schema"] == "fidelity-suite/lambda-server-time.v1"
          and abs(evidence["local_minus_server_seconds"]) < 5.0)
    check("the evidence states that Lambda enforces NO deadline of its own "
          "(the guarantee is controller clock plus the on-instance watchdog)",
          evidence["provider_enforced_deadline_available"] is False)
    stale = Stub(base_responses())
    stale.list_lifecycle_resources()
    stale._server_time["local_received_epoch"] -= 600
    refused, message = refuses(stale.server_time_evidence)
    check("stale server-time evidence is refused, not reused", refused,
          message[:60])
    skewed = Stub(base_responses())
    skewed.list_lifecycle_resources()
    skewed._server_time["local_minus_server_seconds"] = 45.0
    refused, message = refuses(skewed.server_time_evidence)
    check("a 45 s clock disagreement is refused at a 30 s bound", refused)
    check("...and the refusal explains why it matters on THIS provider",
          "no termination deadline of its own" in message, message[:110])
    for bad, why in (("Mon, 01 Sep 2026 10:00:00 +0200", "not GMT"),
                     ("2026-09-01T10:00:00Z", "not RFC 5322"),
                     ("", "absent")):
        broken = Stub(base_responses())
        broken.date_header = bad
        refused, _ = refuses(broken.list_lifecycle_resources)
        check("an HTTP Date that is %s is refused, never replaced with ours"
              % why, refused)


def lifecycle_rungs() -> None:
    print("\n[2] IS ANYTHING OF MINE STILL ALIVE")
    rows = Stub(base_responses(instances=[
        instance(), instance(GONE_ID, "terminated", name="old"),
        instance(ODD_ID, "hibernating", name="odd")]))
    lifecycle = rows.list_lifecycle_resources()
    ids = [row["id"] for row in lifecycle]
    check("a terminated instance is not reported as live", GONE_ID not in ids)
    check("a live instance is reported", LIVE_ID in ids)
    check("an UNDOCUMENTED status is kept as live, not read as absence "
          "(an unknown state read as gone is how an instance leaks)",
          ODD_ID in ids)
    check("...and it is flagged so a caller can see the API drifted",
          next(row for row in lifecycle if row["id"] == ODD_ID)
          ["known_status"] is False)
    check("every documented status is recognised",
          all(LambdaCloud._lifecycle_row(instance(status=s))["known_status"]
              for s in LAMBDA_INSTANCE_STATUSES))
    check("`terminating` is LIVE: it still has a machine attached",
          LambdaCloud._lifecycle_row(instance(status="terminating"))["live"])
    check("`preempted` is LIVE: it is not `terminated`, and Lambda sells no "
          "spot capacity, so it needs an operator not an assumption",
          LambdaCloud._lifecycle_row(instance(status="preempted"))["live"])
    row = lifecycle[0]
    check("a lifecycle row carries Lambda's OWN fields (instance type, "
          "region, ssh key names, filesystems), not RunPod's",
          row["instance_type_name"] == "gpu_1x_a100_sxm4"
          and row["region_name"] == "us-east-1"
          and row["ssh_key_names"] == ["fidelity-controller"]
          and row["file_system_names"] == []
          and row["storage_gib"] == 512 and row["gpu_count"] == 1)
    check("the id is the exact provider string, never an int",
          isinstance(row["id"], str) and row["id"] == LIVE_ID)
    exact = Stub(base_responses())
    check("get_lifecycle_resource reads the exact-id endpoint",
          exact.get_lifecycle_resource(LIVE_ID)["id"] == LIVE_ID
          and ("GET", "/instances/%s" % LIVE_ID)
          in [(call[0], call[1]) for call in exact.calls])
    check("a NAME is not an id: it 404s to None rather than matching a row",
          exact.get_lifecycle_resource("fidcloud-lambda-1") is None)
    outage = LambdaError("Lambda HTTP 503 on GET /instances/x: upstream")
    setattr(outage, "status", 503)
    refused, _ = refuses(
        Stub({("GET", "/instances/%s" % LIVE_ID): outage})
        .get_lifecycle_resource, LIVE_ID)
    check("a 503 on the exact-id read RAISES: 'I could not tell' must never "
          "read as 'it is gone'", refused)
    for bad, why in (
            ({"id": LIVE_ID}, "no status"),
            (dict(instance(), status=5), "a non-string status"),
            (dict(instance(), ssh_key_names="fidelity"), "a string key list"),
            (dict(instance(), instance_type=[]), "a non-object instance_type")):
        refused, _ = refuses(LambdaCloud._lifecycle_row, bad)
        check("an instance row with %s is refused, not coerced" % why, refused)
    refused, _ = refuses(
        Stub(base_responses(instances=[instance(), instance()]))
        .list_lifecycle_resources)
    check("a duplicated instance id in the listing is refused", refused)


def volume_rungs() -> None:
    print("\n[3] PERSISTENT FILESYSTEMS ARE REAL WORK HERE")
    volumes = Stub(base_responses()).list_network_volumes()
    check("GET /file-systems is enumerated (hyphen on list, none on "
          "create/delete -- the API's own inconsistency, read not corrected)",
          len(volumes) == 1 and volumes[0]["id"] == FILESYSTEM["id"])
    check("a Lambda filesystem is recorded as OUTLIVING its instance -- "
          "copying Vast's pod-scoped empty shape here would publish a false "
          "absence proof",
          volumes[0]["persists_after_instance_termination"] is True)
    check("its region, mount point, in-use flag and bytes_used are read",
          volumes[0]["region_name"] == "us-east-1"
          and volumes[0]["is_in_use"] is False
          and volumes[0]["bytes_used"] == 41_231_552
          and volumes[0]["mount_point"].startswith("/lambda/nfs/"))
    check("no per-filesystem rate is invented: the API publishes none",
          volumes[0]["cost_per_hr"] is None)
    check("`created` is normalised from the API's fractional ISO 8601",
          volumes[0]["created_utc"] == "2026-08-30T11:02:41Z")
    for mutate, why in (
            ({"is_in_use": "false"}, "a stringly in-use flag"),
            ({"created": "2026-08-30 11:02:41"}, "a timestamp with no offset"),
            ({"region": "us-east-1"}, "a non-object region"),
            ({"name": ""}, "an empty name"),
            ({"bytes_used": "41231552"}, "a stringly bytes_used")):
        refused, _ = refuses(
            Stub(base_responses(filesystems=[dict(FILESYSTEM, **mutate)]))
            .list_network_volumes)
        check("a filesystem row with %s is refused, not coerced" % why, refused)
    refused, _ = refuses(
        Stub(base_responses(filesystems=[FILESYSTEM, dict(FILESYSTEM)]))
        .list_network_volumes)
    check("a duplicated filesystem id is refused", refused)


def inventory_rungs() -> None:
    print("\n[4] CHARGEABLE INVENTORY REPORTS COMPLETENESS EXPLICITLY")
    inventory = Stub(base_responses()).chargeable_inventory()
    check("schema and provider are the reaper's contract",
          inventory["schema"] == "fidelity-suite/lambda-chargeable-inventory.v1"
          and inventory["provider"] == "lambda")
    check("observed_at_utc is exact UTC",
          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.strptime(
              inventory["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ"))
          == inventory["observed_at_utc"])
    check("both families are present: instances AND network_volumes",
          set(inventory["families"]) == {"instances", "network_volumes"})
    check("complete is an exact bool and unknown_families is empty",
          inventory["complete"] is True and inventory["unknown_families"] == [])
    check("every family names the endpoint it came from",
          all(family["source"].startswith("GET https://")
              for family in inventory["families"].values()))
    check("each resource exposes an exact string id, and compute rows a status",
          all(isinstance(res["id"], str)
              for family in inventory["families"].values()
              for res in family["resources"])
          and all(isinstance(res["status"], str) for res
                  in inventory["families"]["instances"]["resources"]))
    check("a terminated instance is still INVENTORIED (the reaper needs the "
          "full picture, not the live subset)",
          len(Stub(base_responses(instances=[
              instance(), instance(GONE_ID, "terminated")]))
              .chargeable_inventory()["families"]["instances"]
              ["resources"]) == 2)
    fs_outage = LambdaError("Lambda HTTP 500 on GET /file-systems: upstream")
    setattr(fs_outage, "status", 500)
    partial = Stub({**base_responses(),
                    ("GET", "/file-systems"): fs_outage}
                   ).chargeable_inventory()
    check("a filesystem OUTAGE makes the inventory incomplete rather than "
          "empty -- a partial inventory cannot prove no leak",
          partial["complete"] is False
          and partial["unknown_families"] == ["network_volumes"]
          and partial["families"]["network_volumes"]["complete"] is False)
    check("...and it says WHY, so an operator can tell an outage from an "
          "empty account",
          "500" in partial["families"]["network_volumes"]["unknown"])
    check("the instance family is still complete and still populated",
          partial["families"]["instances"]["complete"] is True
          and partial["families"]["instances"]["resources"])
    both = Stub({}).chargeable_inventory()
    check("with nothing reachable, complete is False and BOTH families are "
          "named -- never a fabricated complete: true",
          both["complete"] is False
          and both["unknown_families"] == ["instances", "network_volumes"])


def cost_rungs() -> None:
    print("\n[5] COST: THE PROVIDER PUBLISHES NO BILL, SO NOTHING IS PRICED")
    cost = Stub(base_responses()).billing_history(
        LIVE_ID, start_time="2026-09-06T10:00:00Z",
        end_time="2026-09-06T12:00:00Z")
    check("the published rate IS read (129 cents/h for this type)",
          cost["published_price_cents_per_hour"] == 129)
    check("the window is recorded exactly", cost["window_seconds"] == 7200)
    check("NO dollar figure is computed: an hourly-billed residual the "
          "provider never priced stays unpriced",
          cost["cost_usd"] is None
          and not [key for key in cost if "usd" in key and cost[key]])
    check("the document says it is not the provider's statement",
          cost["provider_authoritative"] is False
          and cost["unreconcilable_by_provider"] is True
          and cost["provider_billing_api"] is None
          and cost["records"] == [])
    check("...and names the API version it read that from",
          "1.10.0" in cost["provider_billing_api_note"])
    for kwargs, why in (
            ({"start_time": "2026-09-06T12:00:00Z",
              "end_time": "2026-09-06T10:00:00Z"}, "an inverted window"),
            ({"start_time": "2026-09-06 10:00:00",
              "end_time": "2026-09-06T12:00:00Z"}, "a non-exact-UTC start"),
            ({"start_time": "2026-09-06T10:00:00Z",
              "end_time": "2026-09-06T12:00:00Z", "bucket_size": "day"},
             "an unsupported bucket")):
        refused, _ = refuses(Stub(base_responses()).billing_history,
                             LIVE_ID, **kwargs)
        check("billing with %s is refused" % why, refused)
    refused, message = refuses(
        Stub(base_responses(instances=[])).billing_history, GONE_ID,
        start_time="2026-09-06T10:00:00Z", end_time="2026-09-06T12:00:00Z")
    check("a terminated instance with no type supplied is refused, not "
          "priced from a guess", refused)
    check("...and the refusal says to pass the type from the lease",
          "instance_type_name" in message, message[:90])
    recovered = Stub(base_responses(instances=[])).billing_history(
        GONE_ID, start_time="2026-09-06T10:00:00Z",
        end_time="2026-09-06T12:00:00Z",
        instance_type_name="gpu_1x_a100_sxm4")
    check("...and with the lease's type it reads the rate from the catalogue",
          recovered["published_price_cents_per_hour"] == 129
          and recovered["cost_usd"] is None)


def closure_rungs() -> None:
    print("\n[6] COST CLOSURE IS POST-ABSENCE, STABLE, AND HONEST")
    now = calendar.timegm(time.strptime("2026-09-06T13:00:00Z",
                                        "%Y-%m-%dT%H:%M:%SZ"))
    lease = {
        "provider_resource_ids": [LIVE_ID],
        "create": {"provider": "lambda",
                   "pre_create_observed_at": "2026-09-06T10:00:00Z",
                   "instance_type_name": "gpu_1x_a100_sxm4"},
        "history": [{"to": "ABSENCE_CONFIRMED", "at": "2026-09-06T12:00:00Z"}],
    }
    settled = Stub(base_responses(instances=[])).reconcile_billing(
        lease, now=now)
    check("the closure is UNSETTLED and says so in both spellings the leases "
          "use", settled["reconciled"] is False and settled["settled"] is False)
    check("no total is invented", settled["total_amount"] is None)
    check("it carries the unpriced cost snapshot and the provider verdict",
          settled["unreconcilable_by_provider"] is True
          and settled["cost_snapshot"][0]
          ["published_price_cents_per_hour"] == 129)
    check("the remedy is the dashboard invoice, and it says the measurement "
          "itself is unaffected (the registry publishes no cost field)",
          any("invoice" in item for item in settled["remedy"])
          and any("no cost field" in item for item in settled["remedy"]))
    check("the closure is sealed by digest over the stable content",
          len(settled["evidence"]["closure_sha256"]) == 64
          and settled["evidence"]["absence_confirmed_at"]
          == "2026-09-06T12:00:00Z")
    refused, _ = refuses(
        Stub(base_responses(instances=[])).reconcile_billing,
        dict(lease, history=[]), now=now)
    check("a lease with no absence event is refused", refused)
    refused, message = refuses(
        Stub(base_responses(instances=[])).reconcile_billing, lease,
        now=calendar.timegm(time.strptime("2026-09-06T12:01:00Z",
                                          "%Y-%m-%dT%H:%M:%SZ")))
    check("a closure inside the 300 s post-absence window is refused",
          refused and "stabilization" in message)
    refused, _ = refuses(
        Stub(base_responses(instances=[])).reconcile_billing,
        dict(lease, provider_resource_ids=[]), now=now)
    check("a closure with no exact id is refused", refused)

    class MovingRate(Stub):
        """The published rate changes between the two retrievals."""

        def __init__(self):
            super().__init__(base_responses(instances=[]))
            self.reads = 0

        def _request(self, method, path, body=None, *, query=None, timeout=90):
            if path == "/instance-types":
                self.reads += 1
                cents = 129 if self.reads <= 1 else 149
                return 200, envelope({INSTANCE_TYPE["name"]: {
                    "instance_type": dict(INSTANCE_TYPE,
                                          price_cents_per_hour=cents),
                    "regions_with_capacity_available": [REGION]}})
            return super()._request(method, path, body, query=query,
                                    timeout=timeout)

    refused, message = refuses(MovingRate().reconcile_billing, lease, now=now)
    check("a rate that MOVES between the two independent retrievals is a "
          "refusal, not a seal", refused and "changed between" in message)


def create_rungs(scratch: str) -> None:                     # noqa: C901
    private = keypair(scratch)
    fields = open(private + ".pub", encoding="utf-8").read().split()
    canonical = "%s %s" % (fields[0], fields[1])
    registered = [{"id": "ddf9a910ceb744a0bb95242cbba6cb50",
                   "name": "fidelity-controller",
                   "public_key": canonical + " noname"}]
    plan = dict(
        gpu_type="gpu_1x_a100_sxm4", region="us-east-1", storage=400,
        num_gpus=1, name="fidcloud-lambda-1",
        terminate_after=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 7200)))

    def prov(**over):
        return Stub(base_responses(keys=registered), ssh_key=private, **over)

    print("\n[7] TWO-PHASE CREATE, AND THE SSH KEY NAME BINDING")
    prepared = prov().prepare_safe_create(**plan)
    check("prepare_safe_create freezes a request before any mutation",
          isinstance(prepared, PreparedLambdaCreate)
          and prepared.instance_type_name == "gpu_1x_a100_sxm4")
    body = json.loads(prepared.launch_body.decode("utf-8"))
    check("the frozen body is the documented launch shape",
          set(body) == {"region_name", "instance_type_name", "ssh_key_names",
                        "name", "user_data"}
          and body["ssh_key_names"] == ["fidelity-controller"])
    check("Lambda accepts exactly one ssh key name at launch, and exactly one "
          "is sent", len(body["ssh_key_names"]) == 1)
    check("no filesystem is attached: one would outlive the instance",
          "file_system_names" not in body)
    check("the frozen request records that the deadline is NOT provider "
          "enforced",
          json.loads(prepared.request_identity_json.decode("utf-8"))
          ["provider_enforced_deadline"] is False)
    check("the request identity is digest-sealed",
          len(prepared.to_dict()["request_identity_sha256"]) == 64)
    check("user_data pins an ED25519 host key through cloud-init",
          body["user_data"].startswith("#cloud-config")
          and "ssh_keys:" in body["user_data"]
          and "ed25519_private" in body["user_data"]
          and "ssh_deletekeys: true" in body["user_data"])
    check("the pinned FINGERPRINT is canonical SHA256",
          prepared.host_key_fingerprint.startswith("SHA256:")
          and len(prepared.host_key_fingerprint) == 50)
    rendered = json.dumps(prepared.to_dict())
    check("the private host key appears in NO representation of the prepared "
          "create",
          "PRIVATE KEY" not in rendered and "ed25519_private" not in rendered)
    for why, kwargs in (
            ("a filesystem attachment", dict(plan, file_system_names=["x"])),
            ("caller-supplied user_data",
             dict(plan, user_data="#cloud-config\n")),
            ("provider env (a create body is provider-persisted, so a "
             "credential in it can never be attested first)",
             dict(plan, env={"HF_TOKEN": "hf_x"})),
            ("a docker command", dict(plan, docker_cmd="python x.py")),
            ("spot", dict(plan, spot=True)),
            ("a missing deadline",
             {k: v for k, v in plan.items() if k != "terminate_after"}),
            ("a deadline inside the setup window",
             dict(plan, terminate_after=time.strftime(
                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 60)))),
            ("a disk larger than the type has", dict(plan, storage=9000)),
            ("a GPU count the type does not have", dict(plan, num_gpus=8)),
            ("an unknown instance type", dict(plan, gpu_type="gpu_1x_nope")),
            ("no region at all",
             {k: v for k, v in plan.items() if k != "region"}),
            ("a region with no capacity", dict(plan, region="us-west-2")),
            ("an empty lease name", dict(plan, name="")),
            ("a 65-character lease name", dict(plan, name="x" * 65))):
        refused, message = refuses(prov().prepare_safe_create, **kwargs)
        check("prepare refuses %s" % why, refused, message[:70])
    mismatched = Stub(base_responses(keys=[dict(registered[0], public_key=(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHZmb28gYmFyIGJheiBxdXV4IGNv"
        "cmdlIGdyYXVsdA== other"))]), ssh_key=private)
    refused, message = refuses(
        mismatched.prepare_safe_create,
        **dict(plan, ssh_key_names=["fidelity-controller"]))
    check("an EXPLICIT key NAME registered against a DIFFERENT public key is "
          "refused: a name is not a key, and launching it would bill a box we "
          "cannot log into", refused)
    check("...and the refusal says exactly that",
          "A name is not a key" in message, message[:120])
    refused, message = refuses(mismatched.prepare_safe_create, **plan)
    check("...and with no name given, discovery finds no key matching our "
          ".pub and refuses rather than picking one",
          refused and "matches" in message, message[:80])
    refused, message = refuses(
        Stub(base_responses(keys=[]), ssh_key=private).prepare_safe_create,
        **plan)
    check("no registered key at all is refused with the console remedy",
          refused and "registered in the console" in message)
    check("the key NAME is discovered by matching our .pub, never assumed",
          prov().prepare_safe_create(**plan).ssh_key_names
          == ("fidelity-controller",))
    refused, _ = refuses(
        Stub(base_responses(keys=registered),
             ssh_key=os.path.join(scratch, "no_such_key")).prepare_safe_create,
        **plan)
    check("a local key pair that does not exist is refused before launch",
          refused)
    os.chmod(private, 0o644)
    refused, _ = refuses(prov().prepare_safe_create, **plan)
    check("a group/other-readable private key is refused", refused)
    os.chmod(private, 0o600)

    print("\n[7a] NO CREDENTIAL MAY ENTER A PROVIDER-PERSISTED CREATE BODY")
    # A create body is stored by the provider and lands in the host's
    # environment BEFORE the instance exists, so no ordering and no
    # attestation can protect it -- there is nothing to attest yet. The guard
    # is tlsguard's single implementation of "looks like a secret", imported
    # lazily with a fail-closed local branch so no window exists in which
    # neither guard runs.
    token = "hf_" + "z" * 34
    try:
        from fidelity import tlsguard as _tlsguard                # noqa: F401
        value_scanning = True
    except ImportError:
        value_scanning = False
    # Carrier-key refusals hold either way: they are this adapter's own
    # fail-closed branch, and they are what makes the absence of tlsguard a
    # refusal rather than a gap.
    for payload, why in (
            ({"env": {"HF_TOKEN": token}}, "a token in env"),
            ({"user_data": "#cloud-config\nruncmd: [echo %s]" % token},
             "a token smuggled into user_data"),
            ({"docker_cmd": "python x.py --token %s" % token},
             "a token in a docker command")):
        refused, message = refuses(
            prov().create, gpu_type="gpu_1x_a100_sxm4", storage=400, **payload)
        check("create() refuses %s" % why, refused, message[:80])
        check("...and the refusal never echoes the credential VALUE",
              token not in message and token[3:] not in message)
    refused, _ = refuses(
        prov(dry=True).create, gpu_type="gpu_1x_a100_sxm4",
        env={"HF_TOKEN": token})
    check("...even under dry, where nothing would be transmitted: the "
          "caller's intent is the defect and a dry run is where to catch it",
          refused)
    check("a credential-free create body is NOT refused by the guard",
          prov(dry=True).create(gpu_type="gpu_1x_a100_sxm4", storage=400,
                                name="fidcloud-lambda-1")["dry_run"] is True)
    # VALUE scanning is tlsguard's property, not a second matcher of ours, so
    # it is asserted only where that module exists -- and its absence is
    # STATED here rather than printed as a pass. Without it a token smuggled
    # into a field no carrier list mentions (`name`) is NOT caught, which is
    # exactly why the fail-closed branch above refuses the carriers outright.
    if value_scanning:
        refused, message = refuses(
            prov(dry=True).create, gpu_type="gpu_1x_a100_sxm4", storage=400,
            name="fidcloud-%s" % token)
        check("create() refuses a token smuggled into `name`, which no "
              "carrier-key list mentions", refused, message[:80])
        check("...and withholds the value", token not in message)
        refused, message = refuses(
            prov().prepare_safe_create,
            **dict(plan, name="fidcloud-%s" % token))
        check("prepare_safe_create scans VALUES too, so a smuggled token is "
              "refused before the request is frozen", refused)
        check("...and that refusal also withholds the value",
              token not in message)
    else:
        print("      NOT ASSERTED HERE: fidelity.tlsguard is absent from this "
              "checkout, so value-level")
        print("      scanning cannot be exercised. The adapter's fail-closed "
              "branch refuses every")
        print("      carrier key instead (asserted above), and these two "
              "rungs arm themselves as")
        print("      soon as tlsguard is present -- owner: TlsAttestation, "
              "landed on main at 7a0a637.")

    print("\n[8] SUBMIT: A LOST RESPONSE IS RECONCILABLE, NOT AMBIGUOUS")
    dry = Stub(base_responses(keys=registered), ssh_key=private, dry=True)
    submitted = dry.submit_prepared_create(dry.prepare_safe_create(**plan))
    check("a dry submit mutates nothing and returns the request",
          submitted["dry_run"] is True
          and submitted["request"]["instance_type_name"] == "gpu_1x_a100_sxm4")
    check("the dry submit leaks no private host key",
          "PRIVATE KEY" not in json.dumps(submitted))

    class Launcher(Stub):
        """Replays one launch outcome through the frozen request's opener."""

        def __init__(self, outcome):
            super().__init__(base_responses(keys=registered),
                             ssh_key=private)
            self.outcome = outcome
            self.opened = 0

        def frozen(self):
            prepared_create = self.prepare_safe_create(**plan)
            launcher = self

            class Body:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def read(self, limit=None):
                    return json.dumps(launcher.outcome).encode("utf-8")

            class Opener:
                def open(self, request, timeout=None):
                    launcher.opened += 1
                    if isinstance(launcher.outcome, Exception):
                        raise launcher.outcome
                    return Body()

            return PreparedLambdaCreate(
                **dict(prepared_create.__dict__, http_opener=Opener()))

    launcher = Launcher({"data": {"instance_ids": [LIVE_ID]}})
    real = launcher.frozen()
    created = launcher.submit_prepared_create(real)
    check("a committed launch returns the exact provider id",
          created["machine_id"] == LIVE_ID
          and isinstance(created["machine_id"], str))
    check("the launch records the terminate_after the controller must enforce "
          "itself", created["terminate_after"] == plan["terminate_after"])
    check("no private host key in the create result",
          "PRIVATE KEY" not in json.dumps(created))
    refused, message = refuses(launcher.submit_prepared_create, real)
    check("a second launch inside Lambda's 12 s launch rate limit is REFUSED "
          "rather than risking an ambiguous 429",
          refused and "AMBIGUOUS" in message)

    def http_error(code, payload):
        return urllib.error.HTTPError(
            "https://x.invalid/launch", code, "err", {},
            io.BytesIO(json.dumps(payload).encode("utf-8")))

    for code, api_code, definitive in (
            (400, "instance-operations/launch/insufficient-capacity", True),
            (400, "global/quota-exceeded", True),
            (400, "global/invalid-parameters", True),
            (500, "global/internal-error", False)):
        rejecting = Launcher(http_error(code, {"error": {
            "code": api_code, "message": "no", "suggestion": "later",
            "request_id": "req-1"}}))
        try:
            rejecting.submit_prepared_create(rejecting.frozen())
            outcome = "returned"
        except LambdaCreateRejectedError as exc:
            outcome = "rejected:%s" % exc.rejection_code
        except LambdaError:
            outcome = "ambiguous"
        expected = ("rejected:%s" % api_code) if definitive else "ambiguous"
        check("%s is classified %s" % (
            api_code, "a definitive refusal (no instance exists)" if definitive
            else "AMBIGUOUS (fail closed)"), outcome == expected, outcome)
    lost = Launcher(OSError("connection reset"))
    refused, message = refuses(lost.submit_prepared_create, lost.frozen())
    check("a LOST launch response is refused with the reconciliation route -- "
          "by exact lease name, because Lambda echoes no id beforehand",
          refused and "UNKNOWN" in message
          and "fidcloud-lambda-1" in message)
    for payload_doc, why in (
            ({"data": {"instance_ids": []}}, "no id"),
            ({"data": {"instance_ids": [LIVE_ID, GONE_ID]}}, "two ids"),
            ({"data": {}}, "no instance_ids at all"),
            ({"instance_ids": [LIVE_ID]}, "no data envelope")):
        odd = Launcher(payload_doc)
        refused, _ = refuses(odd.submit_prepared_create, odd.frozen())
        check("a launch response with %s is refused" % why, refused)

    print("\n[9] HOST KEY: PINNED, NOT TRUSTED ON FIRST USE")
    refused, message = refuses(prov().ssh_host_ed25519_fingerprint, LIVE_ID)
    check("with no pin, the host key refuses rather than accepting a keyscan",
          refused)
    check("...and the refusal states that Lambda has no console or log "
          "endpoint at all", "no console or log endpoint" in message)
    check("...and warns that the HF token rides that session",
          "HF token" in message, message[-90:])
    proof = launcher.ssh_host_ed25519_fingerprint(LIVE_ID)
    check("after a pinned launch, the fingerprint is available",
          proof["fingerprint"] == real.host_key_fingerprint)
    check("the evidence says it is NOT trust-on-first-use and names no "
          "provider log",
          proof["trust_on_first_use"] is False
          and proof["provider_log_endpoint"] is None)
    check("the evidence binds the pin to the exact launch request",
          proof["launch_request_identity_sha256"]
          == launcher._host_key_pins[LIVE_ID]["request_identity_sha256"])
    check("the pin's public half is published, the private half never is",
          proof["public_key"].startswith("ssh-ed25519 ")
          and "PRIVATE" not in json.dumps(proof))
    refused, _ = refuses(launcher.ssh_host_ed25519_fingerprint, GONE_ID)
    check("a pin is per exact id: another id does not inherit it", refused)

    print("\n[10] BINDING: THE LIVE BOX IS THE ONE REQUESTED")
    expectation = dict(
        expected_name="fidcloud-lambda-1",
        instance_type_name="gpu_1x_a100_sxm4", region_name="us-east-1",
        ssh_key_names=["fidelity-controller"], storage_gib=512,
        gpu_count=1, terminate_after=plan["terminate_after"])
    binding = prov().validate_safe_resource_binding(LIVE_ID, **expectation)
    check("a matching live instance passes", binding["passed"] is True)
    check("the deadline is reported as NOT observable on Lambda, so a caller "
          "cannot mistake a missing field for an enforced one",
          binding["terminate_after_observable"] is False
          and binding["provider_enforced_deadline"] is False
          and "terminateAfter" in binding["deadline_note"])
    for over, why in (
            ({"region_name": "us-west-1"}, "a different region"),
            ({"instance_type_name": "gpu_1x_h100_sxm5"}, "a different type"),
            ({"ssh_key_names": ["someone-else"]}, "different key names"),
            ({"expected_name": "other"}, "a different name"),
            ({"storage_gib": 1024}, "a different disk"),
            ({"gpu_count": 8}, "a different GPU count")):
        refused, message = refuses(
            prov().validate_safe_resource_binding, LIVE_ID,
            **dict(expectation, **over))
        check("binding refuses %s" % why, refused, message[:70])
    refused, message = refuses(
        Stub({("GET", "/instances/%s" % LIVE_ID): envelope(instance(
            filesystems=["fidelity-scratch"],
            mounts=[{"mount_point": "/lambda/nfs/fidelity-scratch",
                     "file_system_id": FILESYSTEM["id"]}]))})
        .validate_safe_resource_binding, LIVE_ID, **expectation)
    check("an UNEXPECTED persistent filesystem on the box is refused: it "
          "outlives the instance and the lease does not know about it",
          refused and "file_system" in message)
    refused, _ = refuses(
        Stub({}).validate_safe_resource_binding, LIVE_ID, **expectation)
    check("an instance absent from the exact-id endpoint is refused", refused)
    refused, message = refuses(
        Stub({("GET", "/instances/%s" % LIVE_ID):
              envelope(instance(status="hibernating"))})
        .validate_safe_resource_binding, LIVE_ID, **expectation)
    check("a status outside the documented enum is refused at binding",
          refused and "InstanceStatus" in message)
    refused, _ = refuses(
        Stub({("GET", "/instances/%s" % LIVE_ID):
              envelope(instance(status="terminated"))})
        .validate_safe_resource_binding, LIVE_ID, **expectation)
    check("a terminated instance does not pass binding", refused)

    print("\n[11] ATTESTATION: THE DEVICE, PROVEN FROM THE BOX")
    attest_kw = dict(
        expected_gpu_model="NVIDIA A100-SXM4-40GB",
        expected_vram_bytes=42_949_672_960, min_vcpu=8, min_ram_gb=32,
        storage_gib=512, root_available_bytes_minimum=100 * 10 ** 9,
        run_root_available_bytes_minimum=100 * 10 ** 9,
        expected_gpu_count=1,
        expected_ssh_key_names=["fidelity-controller"])

    def payload(**over):
        disk = {"path": "/", "mount_point": "/", "fs_type": "ext4",
                "source": "/dev/vda1", "device": 2049,
                "total_bytes": 512 * (1 << 30),
                "available_bytes": 480 * 10 ** 9}
        base = {
            "remote_time_epoch": int(time.time()),
            "remote_time_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "login_user": "ubuntu", "uid": 1000,
            "logical_cpus": 30, "memtotal_bytes": 200 * 10 ** 9,
            "nvidia_smi_exit_code": 0, "nvidia_smi_error": "",
            "gpus": [{"index": 0, "name": "NVIDIA A100-SXM4-40GB",
                      "vram_bytes": 42_949_672_960,
                      "vram_used_bytes": 0,
                      "vram_free_bytes": 42_781_900_800,
                      "driver_version": "580.65.06"}],
            "cuda": {"available": True, "usable": True, "count": 1,
                     "name": "NVIDIA A100-SXM4-40GB",
                     "vram_bytes": 42_949_672_960,
                     "interpreter": "/usr/bin/python3", "error": None},
            "compute_apps": [],
            "filesystems": {"root": disk,
                            "run_root": dict(disk, path="/home/ubuntu")},
            "run_root_write": {"writable": True, "error": None},
        }
        base.update(over)
        return json.dumps(base)

    class Box(Stub):
        def __init__(self, output, responses=None, **kw):
            super().__init__(responses or base_responses(keys=registered),
                             ssh_key=private, **kw)
            self.output = output

        def exec_stdout(self, machine_id, command, *, timeout=600,
                        check=True):
            if isinstance(self.output, Exception):
                raise self.output
            self.command = command
            return self.output

    ok = Box(payload()).attest_live_resource(LIVE_ID, **attest_kw)
    check("a correct A100 box attests OK", ok["ok"] is True,
          ", ".join(ok["failures"]))
    check("the attestation is sealed by digest",
          len(ok["attestation_sha256"]) == 64
          and ok["schema"] == "fidelity-suite/lambda-live-attestation.v2")
    # A postcondition evaluated by the party it constrains is not independent
    # evidence: every `observed` field is the BOX's report of itself, and a
    # host with root can lie about nvidia-smi exactly as it can lie about
    # erasing a secret. The document must name its attester rather than let
    # "attested" be read as proof.
    check("the document NAMES its attester -- the box itself, over the "
          "channel whose host key was pinned at launch",
          "the instance itself" in ok["attested_by"]
          and "pinned at launch" in ok["attested_by"])
    check("...and says it is not independently verifiable",
          ok["independently_verifiable"] is False)
    check("...and separates what came from the box from what came from the "
          "provider API over TLS",

          "nvidia_smi_exit_code" in ok["box_self_reported_fields"]
          and "gpus" in ok["box_self_reported_fields"]
          and "provider_record" in ok["independent_of_the_box"])
    check("the run root's WRITABILITY is a check, not an assumption -- the "
          "EACCES that killed a paid gh200 two minutes in",
          ok["checks"]["run_root_writable"] is True)
    check("the login user is proven to be `ubuntu`, not assumed",
          ok["checks"]["login_user"] is True)
    check("the SSH key NAME binding is part of the attestation, because "
          "Lambda binds keys by name",
          ok["checks"]["ssh_key_binding"] is True)
    check("the provider record says WHERE it ran (region, type, image)",
          ok["provider_record"]["region_name"] == "us-east-1"
          and ok["provider_record"]["instance_type_name"] == "gpu_1x_a100_sxm4"
          and ok["provider_record"]["image_family"] == "ubuntu-lts")
    no_torch = Box(payload(cuda={
        "available": False, "usable": False, "count": 0, "name": None,
        "vram_bytes": None, "interpreter": None,
        "error": "no torch on this VM yet"})).attest_live_resource(
            LIVE_ID, **attest_kw)
    check("a VM with NO torch yet still attests: the stack is rebuilt by "
          "bootstrap_measure.sh, and the DEVICE is proven by nvidia-smi",
          no_torch["ok"] is True, ", ".join(no_torch["failures"]))
    check("...and it is flagged for re-attestation after bootstrap rather "
          "than silently passing",
          no_torch["cuda_probe_available"] is False
          and no_torch["revalidate_cuda_after_bootstrap"] is True)
    broken_torch = Box(payload(cuda={
        "available": True, "usable": False, "count": 0, "name": None,
        "vram_bytes": None, "interpreter": "/usr/bin/python3",
        "error": None})).attest_live_resource(LIVE_ID, **attest_kw)
    check("a torch that EXISTS and cannot see the card is a failure -- 7 of 8 "
          "H100 launches in one survey billed with "
          "torch.cuda.is_available() False",
          broken_torch["ok"] is False
          and "cuda_usable" in broken_torch["failures"])
    wrong_card = Box(payload(gpus=[{
        "index": 0, "name": "NVIDIA A100-PCIE-40GB",
        "vram_bytes": 42_949_672_960, "vram_used_bytes": 0,
        "vram_free_bytes": 42_781_900_800,
        "driver_version": "580.65.06"}])).attest_live_resource(
            LIVE_ID, **attest_kw)
    check("a DIFFERENT A100 variant is refused: the device model is what a "
          "comparison binds, not the catalogue name",
          wrong_card["ok"] is False and "gpu_model" in wrong_card["failures"])
    # The oversubscribed card. Total VRAM is honest and useless: host 434175
    # rented a "24 GB" 4090 with 23,424 of 24,564 MiB held by four foreign
    # PIDs, and every total-based check passes it.
    squatted = Box(payload(
        gpus=[{"index": 0, "name": "NVIDIA A100-SXM4-40GB",
               "vram_bytes": 42_949_672_960,
               "vram_used_bytes": 40_000_000_000,
               "vram_free_bytes": 2_949_672_960,
               "driver_version": "580.65.06"}],
        compute_apps=[{"gpu_uuid": "GPU-abc", "pid": 4242,
                       "process_name": "python",
                       "used_memory_bytes": 40_000_000_000}]))
    document = squatted.attest_live_resource(LIVE_ID, **attest_kw)
    check("an OVERSUBSCRIBED card is refused even though its TOTAL VRAM is "
          "exactly right -- free VRAM is the attestable quantity",
          document["ok"] is False
          and "gpu_free_vram" in document["failures"]
          and document["checks"]["gpu_vram"] is True)
    check("...and the foreign PID holding it is RECORDED, because who else "
          "was on the GPU is not knowable after the fact",
          document["observed"]["compute_apps"][0]["pid"] == 4242
          and "no_foreign_compute_apps" in document["failures"])
    check("the free-VRAM floor defaults to 90% of the expected card, so it "
          "gates without a caller remembering to ask",
          document["expected"]["free_vram_bytes_minimum"]
          == 42_949_672_960 * 9 // 10)
    check("...and an explicit lower floor lets a caller accept a shared card "
          "deliberately rather than by omission",
          squatted.attest_live_resource(
              LIVE_ID, **dict(attest_kw,
                              free_vram_bytes_minimum=2 * 10 ** 9)
              )["checks"]["gpu_free_vram"] is True)
    for over, failure, why in (
            ({"run_root_write": {"writable": False, "error": "EACCES"}},
             "run_root_writable", "an unwritable run root"),
            ({"logical_cpus": 2}, "logical_cpu_floor", "too few vCPUs"),
            ({"memtotal_bytes": 8 * 10 ** 9}, "memory_floor",
             "too little RAM"),
            ({"login_user": "root"}, "login_user", "the wrong login user"),
            ({"nvidia_smi_exit_code": 9, "gpus": []}, "nvidia_gpu_count",
             "no GPU at all"),
            ({"gpus": [{"index": 0, "name": "NVIDIA A100-SXM4-40GB",
                        "vram_bytes": 21_474_836_480,
                        "vram_used_bytes": 0,
                        "vram_free_bytes": 21_000_000_000,
                        "driver_version": "580.65.06"}]},
             "gpu_vram", "half the VRAM (a MiG slice or a mislabel)")):
        document = Box(payload(**over)).attest_live_resource(
            LIVE_ID, **attest_kw)
        check("attestation refuses %s" % why,
              document["ok"] is False and failure in document["failures"],
              ", ".join(document["failures"]))
    rebound = Box(payload(), responses={
        **base_responses(keys=registered),
        ("GET", "/instances/%s" % LIVE_ID):
            envelope(instance(keys=["someone-elses-key"]))})
    document = rebound.attest_live_resource(LIVE_ID, **attest_kw)
    check("a box carrying a DIFFERENT ssh key name is refused even though our "
          "key reached it",
          document["ok"] is False
          and "ssh_key_binding" in document["failures"])
    document = Box(RuntimeError("ssh: connect to host port 22: timeout")) \
        .attest_live_resource(LIVE_ID, **attest_kw)
    check("an unreachable box is NOT ok, and the transport error is recorded "
          "verbatim rather than hidden",
          document["ok"] is False and "timeout" in document["transport_error"])
    dry_attest = Box(payload(), dry=True).attest_live_resource(
        LIVE_ID, **attest_kw)
    check("dry mode cannot attest and says so instead of passing",
          dry_attest["ok"] is False
          and "dry mode" in dry_attest["transport_error"])
    for bad, why in (
            ({"extra_key": 1}, "an unexpected key"),
            ({"cuda": {"usable": True}}, "a truncated cuda block"),
            ({"filesystems": {"root": {}}}, "a truncated filesystem block")):
        document = Box(payload(**bad)).attest_live_resource(
            LIVE_ID, **attest_kw)
        check("attestation refuses a payload with %s" % why,
              document["ok"] is False)
    for over, why in (
            ({"expected_vram_bytes": 0}, "a zero VRAM expectation"),
            ({"min_vcpu": -1}, "a negative vCPU floor"),
            ({"expected_gpu_model": ""}, "an empty GPU model")):
        refused, _ = refuses(Box(payload()).attest_live_resource, LIVE_ID,
                             **dict(attest_kw, **over))
        check("attestation refuses %s from the CALLER" % why, refused)


def main() -> int:
    print("\n[0] THE TWELVE EXIST, AND THE ADAPTER IS OFFLINE-ONLY")
    for method in CONTRACT:
        check("LambdaCloud implements %s()" % method,
              callable(getattr(LambdaCloud, method, None)))
    check("no Lambda credential on this controller, so every rung below is a "
          "FIXTURE and nothing here has been verified live",
          LambdaCloud(dry=True).available() is False)
    text = (Path(__file__).resolve().parent / "fidelity" / "lambdaapi.py") \
        .read_text(encoding="utf-8")
    unlabelled = [
        method for method in CONTRACT
        if "UNVERIFIED against a live Lambda account"
        not in text.split("def %s(" % method, 1)[-1][:2800]]
    check("every one of the twelve says in its OWN docstring that it is "
          "unverified against a live account",
          not unlabelled, ", ".join(unlabelled))
    credential_rungs()
    transport_rungs()
    clock_rungs()
    lifecycle_rungs()
    volume_rungs()
    inventory_rungs()
    cost_rungs()
    closure_rungs()
    with tempfile.TemporaryDirectory(prefix="lambda-selftest-") as scratch:
        create_rungs(scratch)
    print("\n" + "-" * 72)
    print("selftest_lambda_contract: %d passed, %d failed"
          % (len(PASS), len(FAIL)))
    if FAIL:
        for name, detail in FAIL:
            print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
