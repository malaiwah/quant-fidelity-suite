#!/usr/bin/env python3
"""The twelve provider-contract methods on JarvisLabs, offline.

A provider you can RENT from is not a provider you can PUBLISH from. The
difference is twelve methods: four that prove the live resource is the one
requested (and, through `attest_live_resource`, that it is the DEVICE the root
was captured on), four that prove nothing of ours is still alive, and four that
reconcile what it cost. `bin/fidelity/runpodapi.py` is the reference; this
suite is the JarvisLabs port's evidence.

Why the port is science rather than convenience: provider is not a
comparability axis. Two A100s in two clouds agreed BITWISE while an H200 sat
2.973e-04 nats away, and four GLM-5.3-Flash rows landed on one comparability
key across three datacenters. What a comparison binds is the GPU MODEL and the
rebuilt stack -- so `attest_live_resource` is what makes a JarvisLabs number
comparable to a root captured anywhere else, and without it the device term is
a catalogue string rather than a reading.

Why THIS adapter needs the guards most: every portability bug this project has
found was a JarvisLabs representation treated as universal truth, and two of
the three could have leaked a billing instance.

  * machine ids are integers        -> `int(pod_id)` raised AFTER the resource
                                       existed, so the controller died holding
                                       an instance it had never adopted
  * the running state is "Running"  -> a "RUNNING" provider made every healthy
                                       poll count as not-running, so the
                                       controller declared a PREEMPTION and
                                       tore down a box mid-bootstrap
  * ids compare as ints in a set    -> the "is it really gone?" check reported
                                       a LIVE instance as destroyed

Rungs for all three are first, and they are written against what `jl` actually
returns: every provider payload below is verbatim from a read-only `jl 0.2.17
--json` call against the live account on 2026-09-06. Nothing here contacts a
provider or creates anything.

Labels in the rung titles: [LIVE] means the underlying provider read was
performed live today; [OFFLINE] means the behaviour is exercised only against a
fixture, because the account cannot produce that state read-only -- it holds no
filesystem, and no instance may be created to attest or to pin a key against.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fidelity.jlapi import (                                  # noqa: E402
    JL, JL_REGIONS, JL_WORKSPACE, Instance, JLBillingUnreconcilable,
    JLCreateResponseError, JLError, JLUnsupportedByCli, PreparedJLCreate,
    _HOST_KEY_PIN_SCRIPT, _is_running_status, _jl_id)

FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


def refuses(fn, *needles, exc=JLError):
    """Call fn; return True iff it raised `exc` naming every needle."""
    try:
        fn()
    except exc as raised:
        text = str(raised)
        missing = [n for n in needles if n not in text]
        if missing:
            print("      refusal missing %r in: %s" % (missing, text[:300]))
            return False
        return True
    except Exception as other:                                # noqa: BLE001
        print("      wrong exception %s: %s" % (type(other).__name__, other))
        return False
    return False


# --------------------------------------------------------------------- fixtures
# Verbatim from `jl list --json` on the live account, 2026-09-06. This is the
# one instance the account holds: a PAUSED spot box, which still charges for
# its stored image and disk, which is why a paused row is still "listed".
LIVE_LIST = [{
    "machine_id": 483634, "cost": 40.488,
    "runtime": "10 days, 1 hours 07 minutes", "gpu_type": "RTX-PRO6000",
    "ram": 160, "storage_gb": 1200, "cores": 28, "template": "pytorch",
    "framework_id": "pytorch", "version": None, "fs_id": None, "num_gpus": 1,
    "url": None, "ssh_command": None, "status": "Paused",
    "paused_image_size": 0.0, "endpoints": None, "name": "glm53-k6-forge",
    "is_reserved": None, "is_spot": True, "committed_resource_id": None,
    "reservation_info": None, "billing_frequency": "hour", "vs_url": None,
    "deployment_id": None, "user_id": "michel.belleau@malaiwah.com",
    "disk_type": "ssd", "public_ip": None, "private_ip": None, "vpc_id": None,
    "http_ports": "", "region": "IN1",
}]
# Verbatim from `jl status --json`, same session. The `resources` counters are
# what makes completeness provable rather than asserted: they are a SECOND
# provider read of the same facts.
LIVE_STATUS = {
    "user": {"user_id": "michel.belleau@malaiwah.com", "name": "Michel Belleau"},
    "balance": {"balance": 92.9746, "grants": 0.0},
    "resources": {"running_instances": 0, "paused_instances": 1,
                  "running_vms": 0, "paused_vms": 0, "deployments": 0,
                  "filesystems": 0},
    "currency": "USD",
}
# Verbatim: the account holds no filesystem and no deployment today.
LIVE_FS: list = []
LIVE_DEPLOY = {"deployments": [], "region_errors": []}


class StubJL(JL):
    """A JL whose `_call` serves scripted payloads instead of running `jl`.

    Keyed by the leading argv words, which is how `_call` itself decides
    things, so a rung cannot pass by accident of argv shape.
    """

    def __init__(self, payloads=None, **kw):
        super().__init__(**kw)
        self.payloads = dict(payloads or {})
        self.calls = []
        self._version = (0, 2, 17)

    def _call(self, argv, *, mutating=False, timeout=None, check=True):
        head = [a for a in argv if not a.startswith("-")]
        self.calls.append(list(argv))
        for width in (2, 1):
            key = " ".join(head[:width])
            if key in self.payloads:
                value = self.payloads[key]
                if isinstance(value, Exception):
                    raise value
                return copy.deepcopy(value)
        raise AssertionError("stub has no payload for %r" % (argv,))


def account(**overrides):
    payloads = {
        "list": LIVE_LIST, "status": LIVE_STATUS,
        "filesystem list": LIVE_FS, "deploy list": LIVE_DEPLOY,
    }
    dry = overrides.pop("dry", False)
    payloads.update(overrides)
    return StubJL(payloads, dry=dry)


print("== the three historical representation bugs cannot come back ==")

# BUG 1: an id is never int()ed. JarvisLabs numbers its machines and `jl get
# --help` types the argument <int>; treating that as universal truth is what
# killed the controller AFTER a pod existed. Both spellings must normalise to
# one exact string, and an opaque id must pass through untouched.
check("_jl_id(483634) and _jl_id('483634') are the SAME exact string",
      _jl_id(483634) == _jl_id("483634") == "483634")
check("an opaque provider id survives _jl_id unchanged",
      _jl_id("uqlk708fxtoz8n") == "uqlk708fxtoz8n")
check("Instance.from_json keeps an opaque id instead of raising",
      Instance.from_json({"machine_id": "uqlk708fxtoz8n"}).machine_id
      == "uqlk708fxtoz8n")
check("Instance.from_json keeps the provider's integer as-is",
      Instance.from_json(LIVE_LIST[0]).machine_id == 483634)
check("a bool is not an id (True would otherwise become '1')",
      refuses(lambda: _jl_id(True), "exact provider id"))
check("None is not an id", refuses(lambda: _jl_id(None), "exact provider id"))
check("a float is not an id",
      refuses(lambda: _jl_id(4.0), "exact provider id"))
check("a shell-injecting id is refused",
      refuses(lambda: _jl_id("483634; rm -rf /"), "invalid characters"))

# BUG 2: the running state is spelled per provider. Nothing may compare the
# spelling; JarvisLabs says "Running", RunPod "RUNNING", Lambda "active".
for status, want in (("Running", True), ("RUNNING", True), ("running", True),
                     ("ready", True), ("active", True), ("Paused", False),
                     ("Launching", False), ("EXITED", False), ("", False),
                     (None, False)):
    check("status %-12r -> running=%s" % (status, want),
          _is_running_status(status) is want)

# BUG 3: ids compare as ints in a set. The set operation that answers "is it
# really gone?" must have strings on BOTH sides. The reconcile_billing rungs
# at the end drive this with a legacy integer lease id against a live listing.
check("a listing id and a legacy integer lease id land in one set",
      len({_jl_id(483634), _jl_id("483634")}) == 1)

print("\n== [LIVE] list_lifecycle_resources: every listed row is a resource ==")
rows = account().list_lifecycle_resources()
check("the live account's single instance is listed", len(rows) == 1)
row = rows[0]
check("its id is the exact string form of the provider's number",
      row["id"] == "483634" and isinstance(row["id"], str))
check("a PAUSED box is still listed (it charges for its stored image)",
      row["listed"] is True and row["status"] == "Paused")
check("...and is reported as not running, without comparing spellings",
      row["running"] is False)
check("cost is carried as an exact decimal STRING, not a float",
      row["cost_usd_total"] == "40.488"
      and isinstance(row["cost_usd_total"], str))
check("the row names the attached filesystem slot (None here)",
      row["filesystem_id"] is None and row["container_disk_gb"] == 1200)
check("the row keeps the provider's own region and template",
      row["region"] == "IN1" and row["template"] == "pytorch")

# The refusal `list_instances` already owns must survive into the contract
# surface: an unreadable answer is not an empty account.
check("[OFFLINE] an unrecognised list envelope refuses instead of reporting "
      "an empty account",
      refuses(account(**{"list": {"data": []}}).list_lifecycle_resources,
              "not an instance list"))

print("\n== [LIVE] get_lifecycle_resource: exact ids, never names ==")
acct = account()
check("the exact id resolves",
      acct.get_lifecycle_resource(483634)["id"] == "483634")
check("the string spelling of the same id resolves identically",
      acct.get_lifecycle_resource("483634")["id"] == "483634")
check("an id that is not on the account is absent, not an error",
      acct.get_lifecycle_resource("999999") is None)
check("a NAME is REFUSED, with the exact id named in the refusal",
      refuses(lambda: acct.get_lifecycle_resource("glm53-k6-forge"),
              "is the NAME", "483634"))

print("\n== [LIVE/OFFLINE] list_network_volumes: storage OUTLIVES the box ==")
check("[LIVE] the live account has no filesystem",
      account().list_network_volumes() == [])
FS_ROWS = [{"fs_id": 907, "name": "glm53-fs", "storage": 1200, "region": "IN1"}]
vols = account(**{"filesystem list": FS_ROWS}).list_network_volumes()
check("[OFFLINE] a filesystem row normalises to an exact string id and size",
      vols == [{"id": "907", "name": "glm53-fs", "size_gb": 1200,
                "region": "IN1", "attached_machine_ids": [],
                "raw": FS_ROWS[0]}])
check("[OFFLINE] an {'filesystems': [...]} envelope is accepted too",
      len(account(**{"filesystem list":
                     {"filesystems": FS_ROWS}}).list_network_volumes()) == 1)
check("[OFFLINE] an unreadable payload REFUSES rather than reporting zero "
      "volumes",
      refuses(account(**{"filesystem list":
                         {"detail": "auth failed"}}).list_network_volumes,
              "not a filesystem list", "outlives its instance"))
check("[OFFLINE] a row with no recognised id is refused, never dropped",
      refuses(account(**{"filesystem list":
                         [{"label": "mystery", "storage": 100}]
                         }).list_network_volumes,
              "no recognised id", "chargeable resource"))
check("[OFFLINE] a duplicated filesystem id is refused",
      refuses(account(**{"filesystem list":
                         FS_ROWS + FS_ROWS}).list_network_volumes,
              "twice"))

print("\n== [LIVE] chargeable_inventory: completeness is stated, not implied ==")
inv = account().chargeable_inventory()
check("schema and provider are the generic sweep's contract",
      inv["schema"] == "fidelity-suite/jarvislabs-chargeable-inventory.v1"
      and inv["provider"] == "jarvislabs")
check("observed_at_utc is an exact UTC instant",
      time.strptime(inv["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ") is not None)
check("complete is a real bool and no family is unknown",
      inv["complete"] is True and inv["unknown_families"] == []
      and isinstance(inv["complete"], bool))
check("the volume family is PRESENT and separate from the compute family",
      set(inv["families"]) == {"instances", "network_volumes", "deployments"})
check("every family declares completeness and carries resources",
      all(fam["complete"] is True and isinstance(fam["resources"], list)
          for fam in inv["families"].values()))
check("the compute family's resource has an exact string id, a name, a status",
      inv["families"]["instances"]["resources"][0]["id"] == "483634"
      and inv["families"]["instances"]["resources"][0]["name"]
      == "glm53-k6-forge"
      and inv["families"]["instances"]["resources"][0]["status"] == "Paused")
# The completeness EVIDENCE: `jl status` counts the same resources
# independently, so a disagreement between two provider reads is an outage,
# not something to reconcile by choosing one.
check("each family records the account-summary counter it agreed with",
      inv["families"]["instances"]["account_summary_count"] == 1
      and inv["families"]["network_volumes"]["account_summary_count"] == 0)
skewed = dict(LIVE_STATUS)
skewed["resources"] = dict(LIVE_STATUS["resources"], paused_instances=2)
inv_skew = account(**{"status": skewed}).chargeable_inventory()
check("[OFFLINE] two provider reads that DISAGREE make the family incomplete",
      inv_skew["complete"] is False
      and inv_skew["unknown_families"] == ["instances"]
      and "disagree" in inv_skew["families"]["instances"]["unknown"])
check("...and the resources are still reported, so the operator sees them",
      len(inv_skew["families"]["instances"]["resources"]) == 1)
inv_nofs = account(**{"filesystem list":
                      {"detail": "auth failed"}}).chargeable_inventory()
check("[OFFLINE] an unreadable VOLUME family is named, never faked complete",
      inv_nofs["complete"] is False
      and inv_nofs["unknown_families"] == ["network_volumes"]
      and inv_nofs["families"]["network_volumes"]["resources"] == []
      and inv_nofs["families"]["network_volumes"]["complete"] is False)
inv_region = account(**{"deploy list":
                        {"deployments": [],
                         "region_errors": ["EU1 unreachable"]}
                        }).chargeable_inventory()
check("[OFFLINE] the provider's own partiality signal (region_errors) is "
      "honoured",
      inv_region["complete"] is False
      and inv_region["unknown_families"] == ["deployments"])
inv_nostatus = account(**{"status": {"balance": {}}}).chargeable_inventory()
check("[OFFLINE] no comparable counter means UNCORROBORATED, not complete",
      inv_nostatus["complete"] is False
      and len(inv_nostatus["unknown_families"]) == 3)

print("\n== [OFFLINE] validate_safe_resource_binding: the live box is the ask ==")
BIND = dict(expected_name="glm53-k6-forge", gpu_type_id="RTX-PRO6000",
            secure_cloud=False, gpu_count=1, volume_gb=1200,
            container_disk_gb=1200, image_name="pytorch",
            terminate_after="2099-01-01T00:00:00Z")
bound = account().validate_safe_resource_binding(483634, **BIND)
check("a matching instance binds",
      bound["passed"] is True and bound["provider_id"] == "483634")
check("the binding says the deadline is NOT provider-observable here",
      bound["terminate_after_observable"] is False
      and "local reaper" in bound["terminate_after_enforced_by"])
check("...and that there is no secure-cloud property to have asked for",
      bound["secure_cloud_supported"] is False)
check("secure_cloud=True is REFUSED, not silently accepted",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, secure_cloud=True)),
          "no secure-cloud", "region"))
check("a non-bool secure_cloud is refused",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, secure_cloud=1)), "exact bool"))
check("a docker image reference is refused with the flag that DOES exist",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, image_name="runpod/pytorch@sha256:abc")),
          "--template"))
check("a different GPU model is refused",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, gpu_type_id="H200")),
          "identity mismatch", "gpu_type"))
check("a wrong name is refused",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, expected_name="somebody-elses-box")),
          "identity mismatch", "name"))
check("more GPUs than the box has is refused",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, gpu_count=8)), "gpu_count"))
# The ENOSPC-after-paid-setup case: separable storage asked for with no
# filesystem to put it on.
check("volume_gb above the instance disk with NO filesystem is refused",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, volume_gb=4000)),
          "no separable filesystem is attached", "--fs-id"))
ATTACHED = [dict(LIVE_LIST[0], fs_id=907)]
check("an attached filesystem SMALLER than volume_gb is refused",
      refuses(lambda: account(
          **{"list": ATTACHED,
             "filesystem list": [{"fs_id": 907, "storage": 100}]}
      ).validate_safe_resource_binding(483634, **dict(BIND, volume_gb=1200)),
          "below the requested volume_gb"))
check("an attached filesystem missing from the fs listing is refused",
      refuses(lambda: account(
          **{"list": ATTACHED, "filesystem list": []}
      ).validate_safe_resource_binding(483634, **dict(BIND, volume_gb=1200)),
          "absent from `jl filesystem"))
check("an id absent from the listing refuses and names name-reconciliation",
      refuses(lambda: account().validate_safe_resource_binding(
          "999999", **BIND), "absent from the complete listing",
          "reconcile by the exact name"))
check("a malformed terminate_after is refused",
      refuses(lambda: account().validate_safe_resource_binding(
          483634, **dict(BIND, terminate_after="2099-01-01 00:00:00")),
          "exact UTC"))

print("\n== [OFFLINE] attest_live_resource: the SCIENTIFIC gate ==")
ATTEST = dict(expected_gpu_model="NVIDIA RTX PRO 6000 Blackwell",
              expected_vram_bytes=96 * 1024 ** 3, min_vcpu=8,
              min_ram_gb=64, volume_gb=1000, container_disk_gb=100,
              workspace_available_bytes_minimum=500 * 10 ** 9,
              container_available_bytes_minimum=20 * 10 ** 9)


def payload(**over):
    vram = 96 * 1024 ** 3
    doc = {
        "remote_time_epoch": int(time.time()),
        "remote_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logical_cpus": 28, "memtotal_bytes": 160 * 10 ** 9,
        "effective_memory_bytes": 160 * 10 ** 9,
        "nvidia_smi_exit_code": 0, "nvidia_smi_error": "",
        "gpus": [{"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell",
                  "vram_bytes": vram, "vram_free_bytes": vram - 400 * 2 ** 20,
                  "vram_used_bytes": 400 * 2 ** 20,
                  "driver_version": "580.126.09"}],
        "cuda": {"usable": True, "count": 1,
                 "name": "NVIDIA RTX PRO 6000 Blackwell", "vram_bytes": vram,
                 "error": None, "interpreter": "/usr/bin/python3"},
        "compute_processes": [], "compute_apps_exit_code": 0,
        "filesystems": {
            "container": {"path": "/", "mount_point": "/",
                          "fs_type": "overlay", "source": "overlay",
                          "device": 40, "total_bytes": 200 * 10 ** 9,
                          "available_bytes": 150 * 10 ** 9},
            "workspace": {"path": JL_WORKSPACE, "mount_point": JL_WORKSPACE,
                          "fs_type": "nfs4", "source": "10.0.0.5:/fs907",
                          "device": 41, "total_bytes": 1200 * 10 ** 9,
                          "available_bytes": 1100 * 10 ** 9},
        },
    }
    doc.update(over)
    return json.dumps(doc)


class AttestJL(StubJL):
    def __init__(self, out, **kw):
        super().__init__({"list": LIVE_LIST}, **kw)
        self.out = out

    def exec_stdout(self, machine_id, command, *, timeout=600, check=True):
        self.probe = (machine_id, command)
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


good = AttestJL(payload()).attest_live_resource(483634, **ATTEST)
check("a matching box attests OK",
      good["ok"] is True and good["failures"] == [])
check("the attestation is sealed over its own bytes",
      len(good["attestation_sha256"]) == 64)
check("the probe was read-only and went through jl exec",
      "jl exec" in good["transport"])
check("the document records the provider's catalogue view beside the silicon",
      good["provider_record"]["gpu_type"] == "RTX-PRO6000"
      and good["provider_record"]["region"] == "IN1")
check("the clock comparison is LABELLED as controller-vs-instance, because "
      "this provider publishes no clock",
      good["clock"]["provider_clock_available"] is False
      and "instance clock" in good["clock"]["reference"])


# A GPU row as nvidia-smi now reports it: total, free and used together,
# because total alone is not the attestable quantity.
def gpu(name="NVIDIA RTX PRO 6000 Blackwell", total=96 * 1024 ** 3,
        free=None, index=0):
    total = int(total)
    free = total - 400 * 2 ** 20 if free is None else int(free)
    return {"index": index, "name": name, "vram_bytes": total,
            "vram_free_bytes": free, "vram_used_bytes": total - free,
            "driver_version": "580.126.09"}


# The whole point of the gate: a different device must not pass.
wrong = AttestJL(payload(gpus=[gpu("NVIDIA H200", 141 * 1024 ** 3)])
                 ).attest_live_resource(483634, **ATTEST)
check("a DIFFERENT GPU MODEL fails the gate",
      wrong["ok"] is False and "gpu_model" in wrong["failures"]
      and "nvidia-smi GPU keys differ" not in wrong["failures"])
short = AttestJL(payload(gpus=[gpu(total=48 * 1024 ** 3)])
                 ).attest_live_resource(483634, **ATTEST)
check("the right model with HALF the VRAM fails the gate",
      short["ok"] is False and "gpu_vram" in short["failures"])
# DecoderParity, host 434175, 2026-09-06: a rented "24 GB" 4090 with 23,424
# of 24,564 MiB already held by four foreign PIDs. The model and the total
# are both correct and the card cannot hold the weights, so an attestation
# that reads TOTAL passes it and the run OOMs after the bootstrap is paid
# for. Free VRAM is the attestable quantity.
held = AttestJL(payload(gpus=[gpu(free=1140 * 2 ** 20)])
                ).attest_live_resource(483634, **ATTEST)
check("an OVERSUBSCRIBED card fails on free VRAM while total still matches",
      held["ok"] is False and "gpu_vram_free" in held["failures"]
      and "gpu_vram" not in held["failures"]
      and "gpu_model" not in held["failures"])
foreign = AttestJL(payload(compute_processes=[
    {"pid": 3141, "used_bytes": 23424 * 2 ** 20,
     "gpu_uuid": "GPU-0d1e2f3a"}])).attest_live_resource(483634, **ATTEST)
check("a FOREIGN compute process fails: shared silicon changes memory and "
      "every timing measured on it",
      foreign["ok"] is False
      and "no_foreign_compute_processes" in foreign["failures"])
check("...and the foreign PID and its held bytes are recorded, not just "
      "counted",
      foreign["observed"]["compute_processes"][0]["pid"] == 3141
      and foreign["observed"]["compute_processes"][0]["used_bytes"]
      == 23424 * 2 ** 20)
check("an unreadable compute-app query fails rather than reading as 'idle'",
      "no_foreign_compute_processes" in AttestJL(
          payload(compute_apps_exit_code=9)
      ).attest_live_resource(483634, **ATTEST)["failures"])
mixed = AttestJL(payload(
    gpus=[gpu(), gpu("NVIDIA H200", index=1)],
    cuda={"usable": True, "count": 2,
          "name": "NVIDIA RTX PRO 6000 Blackwell",
          "vram_bytes": 96 * 1024 ** 3, "error": None,
          "interpreter": "/usr/bin/python3"})).attest_live_resource(
              483634, **ATTEST)
check("a MIXED multi-GPU box fails (jl create --num-gpus rents 8 routinely)",
      mixed["ok"] is False and "gpu_model" in mixed["failures"])
nocuda = AttestJL(payload(cuda={"usable": False, "count": 0, "name": None,
                                "vram_bytes": None, "error": "no cuda",
                                "interpreter": None})).attest_live_resource(
                                    483634, **ATTEST)
check("nvidia-smi alone is not enough: unusable CUDA fails",
      nocuda["ok"] is False and "cuda_usable" in nocuda["failures"])
# The separable-storage proof: if the "big filesystem" is the container's own
# disk, a 200 GB fetch fills it after the bootstrap is already paid for.
same = json.loads(payload())
same["filesystems"]["workspace"].update(device=40, source="overlay")
notsep = AttestJL(json.dumps(same)).attest_live_resource(483634, **ATTEST)
check("a workspace on the SAME device as / fails the separable-storage check",
      notsep["ok"] is False
      and "workspace_is_separate_device" in notsep["failures"])
smallfs = json.loads(payload())
smallfs["filesystems"]["workspace"]["total_bytes"] = 10 * 10 ** 9
tiny = AttestJL(json.dumps(smallfs)).attest_live_resource(483634, **ATTEST)
check("a workspace smaller than volume_gb fails",
      tiny["ok"] is False and "workspace_volume_size" in tiny["failures"])
fewcpu = AttestJL(payload(logical_cpus=2)).attest_live_resource(
    483634, **ATTEST)
check("a box under the vCPU floor fails",
      fewcpu["ok"] is False and "logical_cpu_floor" in fewcpu["failures"])
lowram = AttestJL(payload(effective_memory_bytes=8 * 10 ** 9)
                  ).attest_live_resource(483634, **ATTEST)
check("a box under the RAM floor fails (cgroup limit, not just MemTotal)",
      lowram["ok"] is False and "memory_floor" in lowram["failures"])
broken = AttestJL(RuntimeError("remote command exited 127")
                  ).attest_live_resource(483634, **ATTEST)
check("a transport failure is RECORDED and never passes",
      broken["ok"] is False and "127" in broken["transport_error"])
dry_attest = AttestJL(payload(), dry=True).attest_live_resource(
    483634, **ATTEST)
check("dry mode cannot attest, and says so instead of claiming ok",
      dry_attest["ok"] is False
      and "dry mode" in dry_attest["transport_error"])
check("a zero or negative floor is refused rather than trivially satisfied",
      refuses(lambda: AttestJL(payload()).attest_live_resource(
          483634, **dict(ATTEST, min_vcpu=0)), "positive integer"))

print("\n== [OFFLINE] two-phase create: a lost RESPONSE stays reconcilable ==")
FUTURE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 7200))
CREATE = dict(gpu_type="RTX-PRO6000", num_gpus=1, region="IN1", storage=100,
              name="fidcloud-jl-0001", template="pytorch",
              terminate_after=FUTURE)
prep_acct = account()
prep = prep_acct.prepare_safe_create(**CREATE)
check("prepare_safe_create returns a frozen request and mutates nothing",
      isinstance(prep, PreparedJLCreate) and prep_acct.calls == [])
check("the frozen argv carries the EXACT name -- the only thing that "
      "reconciles a lost response",
      "--name" in prep.argv and "fidcloud-jl-0001" in prep.argv)
check("the frozen argv pins the region and the GPU model",
      "--region" in prep.argv and "IN1" in prep.argv
      and "RTX-PRO6000" in prep.argv)
frozen = prep.to_dict()
check("to_dict is a persistable intent record with a digest",
      frozen["schema"] == "fidelity-suite/jarvislabs-prepared-create.v1"
      and len(frozen["argv_sha256"]) == 64)
check("the same inputs freeze to the same digest (the record is stable)",
      account().prepare_safe_create(**CREATE).to_dict()["argv_sha256"]
      == frozen["argv_sha256"])
check("a different name freezes to a DIFFERENT digest",
      account().prepare_safe_create(
          **dict(CREATE, name="fidcloud-jl-0002")
      ).to_dict()["argv_sha256"] != frozen["argv_sha256"])
check("the request identity records that the deadline is ours to enforce",
      frozen["request_identity"]["terminate_after"] == FUTURE
      and frozen["request_identity"]["terminate_after_enforced_by"]
      == "local reaper only"
      and frozen["request_identity"]["is_spot"] is False)
for kw, needles in (
        (dict(spot=True), ("spot exactly false", "preemption")),
        (dict(name=""), ("exact lease name",)),
        (dict(name="Name me"), ("Name me", "double-spend")),
        (dict(region=None), ("exact region pin", JL_REGIONS[0])),
        (dict(region="US-CA-2"), ("exact region pin",)),
        (dict(gpu_type=None), ("exact GPU model",)),
        (dict(template="ghcr.io/malaiwah/quant-fidelity-measure:main"),
         ("--template",)),
        (dict(terminate_after=None), ("terminate_after", "local reaper")),
        (dict(terminate_after=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 60))),
         ("at least 300 seconds",)),
        (dict(storage=0), ("positive integer",)),
        (dict(script_id="s-123"), ("caller-supplied startup script",
                                   "pin_host_key=True")),
        (dict(script_args="--epochs 5"), ("caller-supplied startup script",)),
        (dict(http_ports="7860"), ("--http-ports",)),
        (dict(cpu=True), ("CPU VM",)),
        (dict(volume_gb=1200), ("no fs_id", "jl filesystem create")),
):
    check("the safe profile refuses %s" % ", ".join(sorted(kw)),
          refuses(lambda kw=kw: account().prepare_safe_create(
              **dict(CREATE, **kw)), *needles))

dry_acct = account(dry=True)
check("submit under dry mutates nothing and echoes the frozen request",
      dry_acct.submit_prepared_create(
          dry_acct.prepare_safe_create(**CREATE))["dry_run"] is True
      and dry_acct.calls == [])
check("a raw kwargs dict cannot be submitted (that would bypass the freeze)",
      refuses(lambda: account().submit_prepared_create(dict(CREATE)),
              "PreparedJLCreate", "bypasses"))

sub_acct = account(**{"create": {"machine_id": 500001,
                                    "name": "fidcloud-jl-0001"}})
submitted = sub_acct.submit_prepared_create(prep)
check("a good create response yields the EXACT string id",
      submitted["machine_id"] == "500001"
      and isinstance(submitted["machine_id"], str))
check("the submitted argv is byte-identical to the frozen argv",
      sub_acct.calls == [list(prep.argv)])
check("a response with NO id refuses and names reconciliation by name",
      refuses(lambda: account(
          **{"create": {"status": "ok"}}).submit_prepared_create(prep),
          "no machine id", "econcile by the exact name",
          "do not create a second instance"))
check("a response naming a DIFFERENT instance raises with the id we now own",
      refuses(lambda: account(
          **{"create": {"machine_id": 500002, "name": "somebody-else"}}
      ).submit_prepared_create(prep), "500002", "is billing",
          exc=JLCreateResponseError))

print("\n== [OFFLINE] host-key pinning: the trust is REMOVED, not tolerated ==")
# `jl` authenticates NO host: jarvislabs/ssh.py:22-30 appends
# UserKnownHostsFile=/dev/null and StrictHostKeyChecking=no to every ssh it
# builds, and cli/instance.py drives exec/upload/download through
# subprocess.call on exactly those options. There is also no boot log, so
# RunPod's read-the-fingerprint-from-the-provider-API anchor does not exist
# here. What IS available is `jl create --script-id` + `jl scripts add`, so
# the key can be decided BEFORE the instance exists.
check("the pinning script never traces itself (it holds a private key)",
      "set -x" not in _HOST_KEY_PIN_SCRIPT
      and "set -eu" in _HOST_KEY_PIN_SCRIPT)
check("the pinning script installs the ED25519 host key 0600",
      "/etc/ssh/ssh_host_ed25519_key" in _HOST_KEY_PIN_SCRIPT
      and "chmod 600" in _HOST_KEY_PIN_SCRIPT)

pin_acct = account(**{"scripts add": {"script_id": "sc-77"}})
pinned = pin_acct.prepare_safe_create(**dict(CREATE, pin_host_key=True))
identity = pinned.to_dict()["request_identity"]
check("the expected fingerprint is known BEFORE the instance exists",
      identity["host_key_fingerprint_sha256"].startswith("SHA256:")
      and len(identity["host_key_fingerprint_sha256"]) == 50)
check("the pinning script is registered and named in the frozen argv",
      "--script-id" in pinned.argv and "sc-77" in pinned.argv)
check("the only provider call made was registering that script",
      len(pin_acct.calls) == 1
      and pin_acct.calls[0][:2] == ["scripts", "add"]
      and pin_acct.calls[0][-1] == "fidelity-hostkey-fidcloud-jl-0001")
check("the PRIVATE host key never appears in the recorded argv",
      not any("PRIVATE KEY" in item for item in pinned.to_dict()["argv"]))
check("the script that carries it is recorded by DIGEST, not by content",
      len(identity["host_key_script_sha256"]) == 64
      and "host_key_script" not in identity)
check("the request identity records that jl's channel checks NOTHING",
      identity["channel_verifies_host_key"] is False)
check("two pinned creates get DIFFERENT keys (a key is never reused)",
      account(**{"scripts add": {"script_id": "sc-78"}}
              ).prepare_safe_create(**dict(CREATE, pin_host_key=True)
                                    ).host_key_fingerprint_sha256
      != pinned.host_key_fingerprint_sha256)
proof = pin_acct.ssh_host_ed25519_fingerprint(483634, prepared=pinned)
check("ssh_host_ed25519_fingerprint returns the PINNED expectation, sealed",
      proof["fingerprint_sha256"] == pinned.host_key_fingerprint_sha256
      and proof["algorithm"] == "ssh-ed25519"
      and len(proof["host_key_proof_sha256"]) == 64)
check("...and records BOTH what we can prove and what we cannot",
      proof["channel_verifies_host_key"] is False
      and proof["verified_on_live_image"] is False
      and proof["provider_log_endpoint_origin"] is None)
check("without a pinned key it REFUSES: there is no boot log, and first "
      "contact is not trust",
      refuses(lambda: account().ssh_host_ed25519_fingerprint(483634),
              "no boot log", "StrictHostKeyChecking=no", "pin_host_key=True",
              exc=JLUnsupportedByCli))
check("a scripts-add response with no id refuses rather than creating a box "
      "with an unpinned key",
      refuses(lambda: account(**{"scripts add": {"ok": True}}
                              ).prepare_safe_create(
                                  **dict(CREATE, pin_host_key=True)),
              "no script id", "would be unpinned"))
check("pinning is refused under dry, because nothing may be registered",
      refuses(lambda: account(dry=True).prepare_safe_create(
          **dict(CREATE, pin_host_key=True)), "under dry", "mutation"))

print("\n== [LIVE] ssh_endpoint: coordinates for a VERIFYING transport ==")
check("a paused box has no endpoint, and that refuses rather than guessing",
      refuses(lambda: account().ssh_endpoint(483634),
              "no ssh_command yet", "Paused"))
RUNNING_ROW = [dict(LIVE_LIST[0], status="Running",
                    ssh_command="ssh root@sshc.jarvislabs.ai -p 11393")]
endpoint = account(**{"list": RUNNING_ROW}).ssh_endpoint(483634)
check("[OFFLINE] a running box's ssh_command parses to user/host/port",
      endpoint["user"] == "root"
      and endpoint["host"] == "sshc.jarvislabs.ai"
      and endpoint["port"] == 11393)
check("...and it says plainly that jl's own channel verifies nothing",
      endpoint["channel_verifies_host_key"] is False)

print("\n== [LIVE] the genuine provider gaps are refusals, not guesses ==")
check("server_time_evidence refuses, naming the degradation precisely",
      refuses(account().server_time_evidence, "no absolute provider clock",
              "runtime", "watchdog", exc=JLUnsupportedByCli))
check("...and it is a JLError too, so no caller needs a new except clause",
      refuses(account().server_time_evidence, "provider clock", exc=JLError))
check("billing_history refuses and names the ONE provider figure that exists",
      refuses(lambda: account().billing_history(
          483634, start_time="2026-09-06T00:00:00Z",
          end_time="2026-09-06T02:00:00Z"),
          "no per-resource billing API", "cost_snapshot",
          "balance delta is not a substitute", exc=JLUnsupportedByCli))

print("\n== [LIVE] cost_snapshot: the provider's own number, labelled ==")
snap = account().cost_snapshot(483634)
check("the snapshot carries the provider's running total, not a rate",
      snap["cost_usd_total"] == "40.488" and "not a rate" in snap["basis"])
check("the observation time is named as the CONTROLLER's, not the provider's",
      "controller_observed_at_utc" in snap
      and snap["provider_clock_available"] is False
      and "observed_at_utc" not in snap)
check("a destroyed instance has NO cost figure, and that is said plainly",
      refuses(lambda: account().cost_snapshot("999999"),
              "already unreadable", "before destroy"))

print("\n== [OFFLINE] reconcile_billing copes with EVERY stored lease shape ==")
# The flat shape `measure_cloud.write_lease` wrote, which is what the 137
# settled JarvisLabs leases looked like -- machine_id a JSON INTEGER, an fs_id
# beside it, and no history at all.
LEGACY = {"job_id": "a1b2c3d4", "job_id_full": "a1b2c3d4" * 8,
          "name": "fidcloud-a1b2c3d4", "provider": "jarvislabs",
          "machine_id": 483634, "fs_id": 907,
          "deadline_epoch": 1757000000.0,
          "deadline_utc": "2026-09-04T14:13:20Z",
          "created_at": "2026-09-04T10:13:20Z", "pid": 4242}
check("a legacy flat lease PARSES (its ids are read, not crashed on)",
      JL._lease_facts(LEGACY)["provider_resource_ids"] == ["483634"])
check("...and its fs_id is carried as a SECOND chargeable resource",
      JL._lease_facts(LEGACY)["filesystem_ids"] == ["907"])
check("the legacy shape is identified as such",
      JL._lease_facts(LEGACY)["shape"] == "legacy-flat")
check("a legacy lease with no absence event refuses, naming both families",
      refuses(lambda: account().reconcile_billing(LEGACY),
              "no provider-absence event", "483634", "907",
              "outlives its instance"))


# The v2 shape, keys and history verbatim from a real settled lease
# (~/.fidelity-cloud/leases-v2, written 2026-09-05), re-keyed to jarvislabs.
def v2_lease(ids=("483634",), absent_at="2026-09-05T03:33:59Z",
             snapshots=None):
    doc = {
        "schema": "fidelity-suite/cloud-lease.v2",
        "job_hash": "02" * 32, "attempt_id": "11" * 12, "generation": 8,
        "state": "ABSENCE_CONFIRMED",
        "create": {"provider": "jarvislabs", "exact_name": "fidcloud-jl-0001",
                   "pre_create_observed_at": "2026-09-05T02:49:29Z",
                   "pre_create_provider_ids": [],
                   "pre_create_network_volume_ids": [],
                   "reap_deadline_epoch": 1757040000.0},
        "provider_resource_ids": list(ids),
        "history": [
            {"generation": 5, "from": "ACTIVE", "to": "DESTROYING",
             "event": "DESTROY_REQUESTED", "at": "2026-09-05T03:33:57Z",
             "evidence": {}},
            {"generation": 6, "from": "DESTROYING", "to": "ABSENCE_CONFIRMED",
             "event": "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING",
             "at": absent_at, "evidence": {}},
        ],
        "terminal_proof": {"provider_absence": {"at": absent_at}},
        "billing_reconciliation": None,
    }
    if snapshots is not None:
        doc["history"][0]["evidence"] = {"cost_snapshots": snapshots}
    return doc


EMPTY_STATUS = dict(LIVE_STATUS,
                    resources=dict(LIVE_STATUS["resources"],
                                   paused_instances=0))


def gone(**over):
    payloads = {"list": [], "status": EMPTY_STATUS}
    payloads.update(over)
    return account(**payloads)


check("a v2 lease whose id is GONE but has no cost figure is UNRECONCILABLE, "
      "and says the receipt is unpublishable on cost",
      refuses(lambda: gone().reconcile_billing(v2_lease()),
              "no provider cost figure", "UNRECONCILABLE",
              "cost_snapshot", exc=JLBillingUnreconcilable))
ev = {}
try:
    gone().reconcile_billing(v2_lease())
except JLBillingUnreconcilable as exc:
    ev = exc.evidence
check("...and the refusal still carries the full accounting for the operator",
      ev.get("absence_reverified") is True
      and ev.get("inventory_complete") is True
      and ev.get("ids_without_snapshot") == ["483634"]
      and ev.get("provider_billing_api_available") is False)

PRE = [{"provider_id": 483634, "cost_usd_total": "40.488",
        "controller_observed_at_utc": "2026-09-05T03:33:40Z"}]
check("a PRE-destroy snapshot is a LOWER BOUND, not a closure, and the "
      "unpriced residual is named in seconds",
      refuses(lambda: gone().reconcile_billing(v2_lease(snapshots=PRE)),
              "predates the absence proof", "19 second", "LOWER BOUND",
              "$40.488", exc=JLBillingUnreconcilable))
POST = [{"provider_id": "483634", "cost_usd_total": "40.488",
         "controller_observed_at_utc": "2026-09-05T03:33:59Z"}]
closed = gone().reconcile_billing(v2_lease(snapshots=POST))
check("a snapshot taken AT/AFTER the proven absence closes the lease",
      closed["reconciled"] is True and closed["total_amount"] == "40.488"
      and closed["provider_resource_ids"] == ["483634"])
check("the closure is sealed over its evidence",
      len(closed["evidence"]["closure_sha256"]) == 64)
check("the closure states there was no provider billing API behind it",
      closed["evidence"]["provider_billing_api_available"] is False
      and closed["evidence"]["provider_clock_available"] is False)
check("two closures of the same lease are byte-identical (stable)",
      gone().reconcile_billing(v2_lease(snapshots=POST))["evidence"]
      ["closure_sha256"] == closed["evidence"]["closure_sha256"])
check("an integer snapshot id and a string lease id reconcile as ONE resource",
      gone().reconcile_billing(v2_lease(snapshots=[
          dict(POST[0], provider_id=483634)]))["total_amount"] == "40.488")

# BUG 3, driven end to end: a lease id stored as an INTEGER against a listing
# id that is a STRING. If those compare as different objects, a live instance
# is settled as gone -- which is the "is it really gone?" bug, and here it
# would seal the cost of a box that is still billing.
check("a lease claiming absence for a STILL-LISTED integer id is REFUSED",
      refuses(lambda: account().reconcile_billing(
          dict(v2_lease(), provider_resource_ids=[], machine_id=483634)),
          "still listed", "483634", "re-confirm"))
check("...and so is a still-listed FILESYSTEM (storage outlives the box)",
      refuses(lambda: gone(**{
          "filesystem list": FS_ROWS,
          "status": dict(EMPTY_STATUS, resources=dict(
              EMPTY_STATUS["resources"], filesystems=1))}
      ).reconcile_billing(dict(v2_lease(snapshots=POST), fs_id=907)),
          "still listed", "907"))
check("an INCOMPLETE inventory refuses to conclude absence at all",
      refuses(lambda: gone(**{"filesystem list": {"detail": "auth"}}
                           ).reconcile_billing(v2_lease(snapshots=POST)),
              "inventory is incomplete", "cannot prove no leak"))
check("another provider's lease is never reconciled through the jl CLI",
      refuses(lambda: account().reconcile_billing(
          {"provider": "runpod", "machine_id": "soq7rpke4sccx2",
           "history": []}),
          "names provider 'runpod'", "aim JarvisLabs calls"))
check("a lease naming no resource at all is refused",
      refuses(lambda: account().reconcile_billing(
          {"provider": "jarvislabs", "history": []}),
          "at least one exact provider id"))
check("a cost snapshot with no observation time is refused, never dated from "
      "our clock",
      refuses(lambda: JL._lease_facts(v2_lease(snapshots=[
          {"provider_id": 483634, "cost_usd_total": "1.0"}])),
          "no observation time", "local clock"))
check("a non-object lease is refused",
      refuses(lambda: JL._lease_facts(["not", "a", "lease"]),
              "must be a JSON object"))

print("\n== [OFFLINE] no credential may enter a provider payload ==")
# Vast put `-e HF_TOKEN=...` into the create body at 2322-2330, so the
# credential was in the provider's own records and the host's docker
# environment BEFORE the instance existed -- there is nothing to attest yet,
# which is why this cannot be fixed by ordering. `jl create` has no env flag,
# so JL cannot leak by that exact route, but every create value becomes ARGV
# (visible in `ps` on a shared host, echoed back by the CLI) and
# --script-args is provider-stored. The guard is tlsguard's single
# implementation; the fallback here is fail-closed.
SECRET = "hf_" + "aBcD" * 8
for path, kw in (("create", dict(gpu_type="A100", name=SECRET)),
                 ("create", dict(gpu_type="A100", script_args="HF_TOKEN=" + SECRET)),
                 ("prepare_safe_create", dict(CREATE, name=SECRET))):
    acct_guard = account(**{"create": {"machine_id": 1}})
    call = (acct_guard.create if path == "create"
            else acct_guard.prepare_safe_create)
    raised = False
    try:
        call(**kw)
    except JLError as exc:
        raised = SECRET not in str(exc)
        if not raised:
            print("      REFUSAL LEAKED THE VALUE: %s" % str(exc)[:120])
    check("%s refuses a credential-shaped value, and the refusal never "
          "repeats the value" % path, raised)
    check("...and nothing was transmitted to the provider",
          acct_guard.calls == [])
check("a create with no credential still goes through",
      account(**{"create": {"machine_id": 7}}).create(
          gpu_type="A100", name="fidcloud-clean")["machine_id"] == 7)
print()
if FAILED:
    print("selftest_jl_parity: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_jl_parity: all passed")

