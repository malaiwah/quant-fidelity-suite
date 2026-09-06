#!/usr/bin/env python3
"""The twelve provider-contract methods, on Vast, offline.

`docs/PROVIDER-PARITY.md` names twelve methods that separate a provider you
can RENT from one you can PUBLISH from: four that prove the live resource is
the one requested, four that prove nothing of ours is still alive, four that
reconcile what it cost against the provider's own clock. This suite is the
offline rung for the Vast port of all twelve.

Every fixture here is a RESPONSE SHAPE observed live on 2026-09-06, not an
invention, because each portability bug this project has already paid for was
a provider representation assumed rather than read:

  * Vast ids are INTEGERS (`id`, `machine_id`, `new_contract`)
  * the contract state is lowercase `"running"`, and `actual_status` can say
    `"loading"` while it bills -- and BOTH said running for ~14 minutes on a
    box whose sshd could not be reached at all, which is why a status field is
    never evidence of reachability
  * `duration` on a live contract is time REMAINING and counts DOWN
  * the ask's advertised rate under-quotes the contract's by ~23% (it excludes
    the disk), so anything pricing a run reads the contract
  * `/charges/` IGNORES server-side per-instance filters, so selection is done
    here on the exact `source` string
  * one healthy Tesla T4 reports VRAM three ways: API 15360 MiB, nvidia-smi
    15360 MiB, torch total_memory 14912 MB -- an attestation demanding they
    match refuses every healthy card

No provider is contacted: `_req` is stubbed and every HTTP path is a fixture.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity import vastapi                              # noqa: E402
from fidelity.vastapi import (                            # noqa: E402
    KNOWN_BAD_MACHINE_IDS, Vast, VastCreateRejectedError, VastError)

FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


def refuses(label, call, *, needle=None, kind=VastError):
    try:
        call()
    except kind as exc:
        ok = needle is None or needle in str(exc)
        check("%s [refusal names %r]" % (label, needle or "-"), ok)
        return
    except Exception as exc:                              # noqa: BLE001
        check("%s [wrong exception %s: %s]"
              % (label, type(exc).__name__, str(exc)[:120]), False)
        return
    check("%s [NO refusal raised]" % label, False)


DAY = 20701

# One live instance row, trimmed to the fields the adapter reads: contract
# 50055626 on machine 150014 (Pennsylvania), a Tesla T4 at $0.16667/h.
LIVE_INSTANCE = {
    "id": 50055626, "machine_id": 150014, "host_id": 579267,
    "cur_state": "running", "actual_status": "loading",
    "intended_status": "running", "label": "vast-fruit",
    "dph_total": 0.16666666666666666, "dph_base": 0.13333333333333333,
    "storage_total_cost": 0.033333333333333326,
    "num_gpus": 1, "gpu_name": "Tesla T4", "gpu_ram": 15360,
    "disk_space": 120.0, "geolocation": "Pennsylvania, US",
    "image_uuid": "ghcr.io/malaiwah/quant-fidelity-measure:main",
    "verification": "unverified", "hosting_type": None,
    "driver_version": "595.71.05", "cuda_max_good": 13.2,
    "ssh_host": "ssh3.vast.ai", "ssh_port": 15626,
    "start_date": 1788694346.9469156, "end_date": 1788704346.0,
    "duration": 15596229.0, "volume_info": [],
}
# The advertised half of the same machine's ask: cheaper than the contract.
LIVE_OFFER = {
    "id": 50006709, "machine_id": 150014, "host_id": 579267,
    "gpu_name": "Tesla T4", "gpu_ram": 15360, "num_gpus": 1,
    "disk_space": 655.95, "dph_total": 0.13555555555555554,
    "geolocation": "Pennsylvania, US", "cuda_max_good": 13.2,
}


def charge_row(source, day=DAY, gpu=0.007, disk=0.004, label="vast-fruit"):
    # Vast reports charge amounts to the MILL (three decimals) and a row's
    # amount equals its items' sum EXACTLY -- verified over 39 real rows, so
    # the adapter enforces exact equality and the fixture must be built the
    # way the provider builds it. Rounding here is fixture fidelity, not a
    # tolerance: a binary artefact like 0.013000000000000001 is not a shape
    # Vast ever returns.
    start = day * 86400
    gpu = round(gpu, 3)
    disk = round(disk, 3)
    items = [
        {"start": start, "end": start, "type": kind, "source": None,
         "description": "%s charge" % kind, "amount": amount,
         "metadata": {}, "items": []}
        for kind, amount in (("gpu", gpu), ("disk", disk), ("bwd", 0.0),
                             ("bwu", 0.0))]
    return {
        "start": start, "end": start, "type": "instance", "source": source,
        "description": "Instance charges - 1 day",
        "amount": round(gpu + disk, 3), "metadata": {"label": label},
        "items": items,
    }


class StubVast(Vast):
    """A Vast whose HTTP layer is a fixture table."""

    def __init__(self, **kw):
        self.responses = kw.pop("responses", {})
        self.exec_payload = kw.pop("exec_payload", None)
        self.log_text = kw.pop("log_text", "")
        self.calls = []
        super().__init__(**kw)
        self._key = "stub-key"

    def _load_key(self):
        return "stub-key"

    def _req(self, method, path, body=None, **kw):
        self.calls.append((method, path.split("?")[0], body))
        for prefix, value in self.responses.items():
            if path.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value
        raise VastError("stub has no fixture for %s %s" % (method, path))

    def exec_stdout(self, machine_id, command, **kw):
        if self.exec_payload is None:
            raise VastError("stub SSH is unavailable")
        return self.exec_payload

    def _instance_log_text(self, instance_id, *, tail, timeout):
        return self.log_text


def instances(rows=(LIVE_INSTANCE,)):
    return {"/instances/": {"instances": list(rows)}}


print("== exact integral ids: a label is never an id ==")
for value, want in ((50055626, "50055626"), ("50055626", "50055626"),
                    (" 50055626 ", "50055626")):
    check("_provider_id(%r) -> %r" % (value, want),
          vastapi._provider_id(value) == want)
for bad in ("vast-fruit", "", "0", 0, True, None, 1.5, "12x", "-5"):
    refuses("_provider_id(%r) refuses" % (bad,),
            lambda bad=bad: vastapi._provider_id(bad),
            needle="exact integral id")

print()
print("== server_time_evidence: the PROVIDER's clock, never ours ==")


class FakeResponse:
    def __init__(self, date):
        self.headers = {"Date": date}


now = time.time()
gmt = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(now))
v = StubVast()
refuses("refuses before any authenticated read",
        v.server_time_evidence, needle="server time is unavailable")
v._capture_server_time(FakeResponse(gmt), vastapi.API + "/users/current/")
evidence = v.server_time_evidence()
check("accepts the strict GMT Date header Vast actually sends",
      evidence["schema"] == "fidelity-suite/vast-server-time.v1"
      and evidence["endpoint_origin"] == "https://console.vast.ai"
      and abs(evidence["local_minus_server_seconds"]) < 5)
refuses("refuses a non-GMT Date",
        lambda: v._capture_server_time(
            FakeResponse("Sun, 06 Sep 2026 11:37:15 +0200"),
            vastapi.API + "/users/current/"),
        needle="strict GMT")
refuses("refuses a missing Date",
        lambda: v._capture_server_time(
            FakeResponse(""), vastapi.API + "/users/current/"),
        needle="lacks HTTP Date")
stale = StubVast()
stale._capture_server_time(FakeResponse(gmt), vastapi.API + "/users/current/")
stale._server_time["local_received_epoch"] = now - 600
refuses("refuses stale evidence", stale.server_time_evidence, needle="stale")
skewed = StubVast()
skewed._capture_server_time(
    FakeResponse(time.strftime("%a, %d %b %Y %H:%M:%S GMT",
                               time.gmtime(now - 3600))),
    vastapi.API + "/users/current/")
refuses("refuses a clock delta beyond the bound",
        skewed.server_time_evidence, needle="differs from Vast server UTC")

print()
print("== lifecycle: exact-id rows, lowercase provider states ==")
v = StubVast(responses=instances())
rows = v.list_lifecycle_resources()
check("one row with an exact STRING id",
      [r["id"] for r in rows] == ["50055626"])
check("status is the CONTRACT state, verbatim lowercase, with actual beside",
      rows[0]["status"] == "running" and rows[0]["actual_status"] == "loading")
check("machine and host ids are strings, not ints",
      rows[0]["provider_machine_id"] == "150014"
      and rows[0]["provider_host_id"] == "579267")
check("the pod-scoped disk answers both disk roles",
      rows[0]["volume_gb"] == 120.0 and rows[0]["container_disk_gb"] == 120.0)
check("terminate_after is derived from the contract end_date",
      rows[0]["terminate_after"] == "2026-09-06T14:19:06Z")
check("runtime is ELAPSED, not the contract's remaining duration",
      0 <= rows[0]["runtime"] < 15596229.0)
check("get_lifecycle_resource finds the exact id",
      (v.get_lifecycle_resource("50055626") or {}).get("id") == "50055626")
check("...and an absent id is None, not an error",
      v.get_lifecycle_resource(99999999) is None)
refuses("a LABEL is not an id",
        lambda: v.get_lifecycle_resource("vast-fruit"),
        needle="exact integral id")
refuses("a duplicate id in the listing refuses",
        lambda: StubVast(responses=instances(
            (LIVE_INSTANCE, dict(LIVE_INSTANCE)))).list_lifecycle_resources(),
        needle="repeats id")
refuses("a non-list instances payload refuses",
        lambda: StubVast(responses={"/instances/": {"instances": None}}
                         ).list_lifecycle_resources(),
        needle="lacks an instances array")

print()
print("== network volumes: a REAL family, empty only on this account ==")
check("an empty account volume list is empty, not unimplemented",
      StubVast(responses={"/volumes/": {"volumes": []}}
               ).list_network_volumes() == [])
volumes = StubVast(responses={"/volumes/": {"volumes": [
    {"id": 771, "label": "root-cache", "size": 512, "machine_id": 150014,
     "dph_total": 0.02}]}}).list_network_volumes()
check("a sized volume row normalises to an exact string id",
      [(x["id"], x["size_gb"], x["provider_machine_id"]) for x in volumes]
      == [("771", 512.0, "150014")])
refuses("an unrecognised volume shape REFUSES rather than counting zero",
        lambda: StubVast(responses={"/volumes/": {"volumes": [
            {"id": 771, "label": "mystery"}]}}).list_network_volumes(),
        needle="no recognised size field")

print()
print("== chargeable_inventory: explicit completeness, both families ==")
v = StubVast(responses=dict(instances(), **{"/volumes/": {"volumes": []}}))
inventory = v.chargeable_inventory()
check("schema and provider are the contracted strings",
      inventory["schema"] == "fidelity-suite/vast-chargeable-inventory.v1"
      and inventory["provider"] == "vast")
check("complete is a real bool with no unknown families",
      inventory["complete"] is True and inventory["unknown_families"] == [])
check("observed_at_utc is exact UTC",
      vastapi._exact_utc(inventory["observed_at_utc"], "observed")
      == inventory["observed_at_utc"])
check("families carry network_volumes plus a compute family",
      "network_volumes" in inventory["families"]
      and "instances" in inventory["families"])
check("each family declares completeness and a resource list",
      all(isinstance(f["complete"], bool) and isinstance(f["resources"], list)
          for f in inventory["families"].values()))
check("compute resources expose an exact string id, a name and a status",
      [(r["id"], r["name"], r["status"])
       for r in inventory["families"]["instances"]["resources"]]
      == [("50055626", "vast-fruit", "running")])
partial = StubVast(responses=dict(
    instances(), **{"/volumes/": VastError("Vast HTTP 503 on /volumes/")}
)).chargeable_inventory()
check("a family that cannot be read is NAMED, never counted as empty",
      partial["complete"] is False
      and partial["unknown_families"] == ["network_volumes"]
      and partial["families"]["network_volumes"]["complete"] is False
      and "503" in partial["families"]["network_volumes"]["unknown"])
check("...and the readable family is still complete",
      partial["families"]["instances"]["complete"] is True)

print()
print("== validate_safe_resource_binding: is this the rental I asked for ==")
BIND = dict(expected_name="vast-fruit", gpu_type_id="Tesla T4",
            secure_cloud=False, gpu_count=1, volume_gb=60,
            container_disk_gb=60, image_name=LIVE_INSTANCE["image_uuid"],
            terminate_after="2026-09-06T14:19:06Z")
v = StubVast(responses=instances())
bound = v.validate_safe_resource_binding(50055626, **BIND)
check("the matching contract passes and reports the exact id",
      bound["passed"] is True and bound["provider_id"] == "50055626")
check("the CONTRACT rate is recorded as an exact decimal string",
      bound["observed"]["cost_per_hr"].startswith("0.1666666666"))
refuses("secure_cloud=True is refused, not silently accepted",
        lambda: v.validate_safe_resource_binding(
            50055626, **dict(BIND, secure_cloud=True)),
        needle="cannot attest a secure-datacenter")
refuses("a non-bool secure_cloud is refused",
        lambda: v.validate_safe_resource_binding(
            50055626, **dict(BIND, secure_cloud=1)),
        needle="exact bool")
refuses("volume+container above the rented disk is refused",
        lambda: v.validate_safe_resource_binding(
            50055626, **dict(BIND, volume_gb=100, container_disk_gb=100)),
        needle="cannot grow after rent time")
refuses("a different GPU model is refused",
        lambda: v.validate_safe_resource_binding(
            50055626, **dict(BIND, gpu_type_id="RTX 3090")),
        needle="gpu_type_id expected")
refuses("a contract outliving the deadline is refused",
        lambda: v.validate_safe_resource_binding(
            50055626, **dict(BIND, terminate_after="2026-09-06T12:00:00Z")),
        needle="not holding our teardown deadline")
check("a contract ending EARLIER than the deadline is recorded, not refused",
      v.validate_safe_resource_binding(
          50055626, **dict(BIND, terminate_after="2026-09-07T00:00:00Z")
      )["terminate_after_earlier_than_requested"] is True)
refuses("an attached network volume is refused",
        lambda: StubVast(responses=instances((dict(
            LIVE_INSTANCE, volume_info=[{"id": 771}]),))
        ).validate_safe_resource_binding(50055626, **BIND),
        needle="no network volume may be attached")
refuses("a known-bad host is refused even when everything else matches",
        lambda: StubVast(responses=instances((dict(
            LIVE_INSTANCE, machine_id=68004),))
        ).validate_safe_resource_binding(50055626, **BIND),
        needle="known-bad host")
refuses("an absent contract is refused",
        lambda: StubVast(responses=instances(())
                         ).validate_safe_resource_binding(50055626, **BIND),
        needle="absent from the complete listing")

print()
print("== attest_live_resource: the scientific gate ==")
T4_SMI_BYTES = 15360 * 1024 * 1024          # nvidia-smi memory.total
T4_TORCH_BYTES = 14912 * 1024 * 1024        # torch total_memory, 448 MiB less
ATTEST = dict(expected_gpu_model="Tesla T4",
              expected_vram_bytes=T4_SMI_BYTES, min_vcpu=4, min_ram_gb=16,
              volume_gb=60, container_disk_gb=60,
              workspace_available_bytes_minimum=40_000_000_000,
              container_available_bytes_minimum=10_000_000_000)


def payload(**over):
    epoch = int(time.time())
    disk = {"path": "/", "mount_point": "/", "fs_type": "overlay",
            "source": "overlay", "device": 66,
            "total_bytes": 128_000_000_000,
            "available_bytes": 120_000_000_000, "error": None}
    document = {
        "remote_time_epoch": epoch,
        "remote_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(epoch)),
        "logical_cpus": 36, "memtotal_bytes": 263_872_512_000,
        "effective_memory_bytes": 263_872_512_000,
        "nvidia_smi_exit_code": 0, "nvidia_smi_error": "",
        "gpus": [{"index": 0, "name": "Tesla T4",
                  "vram_bytes": T4_SMI_BYTES,
                  "driver_version": "595.71.05"}],
        "cuda": {"usable": True, "count": 1, "name": "Tesla T4",
                 "vram_bytes": T4_TORCH_BYTES, "error": None,
                 "interpreter": "/usr/bin/python3.12"},
        "filesystems": {"root": disk,
                        "workspace": dict(disk, path="/workspace")},
        "hub_reachability": [
            {"host": "huggingface.co", "tls_ok": True,
             "cert_subject_cn": "huggingface.co",
             "cert_issuer_cn": "Amazon RSA 2048 M01",
             "cert_not_after": "Feb 27 23:59:59 2027 GMT",
             "http_status": 200, "error": None},
            {"host": "cdn-lfs.hf.co", "tls_ok": True,
             "cert_subject_cn": "hf.co",
             "cert_issuer_cn": "Amazon RSA 2048 M01",
             "cert_not_after": "Feb  9 23:59:59 2027 GMT",
             "http_status": None, "error": None}],
    }
    document.update(over)
    return json.dumps(document)


def attest(payload_text, rows=(LIVE_INSTANCE,), **over):
    box = StubVast(responses=instances(rows), exec_payload=payload_text)
    return box.attest_live_resource(50055626, **dict(ATTEST, **over))


good = attest(payload())
HAS_TLSGUARD = good["hub_tls_verdict_source"] == "tlsguard"
check("a healthy T4 attests: torch's 14912 MB against nvidia-smi's 15360 MiB "
      "is inside the band, so a real card is not refused for 448 MiB",
      good["checks"]["gpu_model"] and good["checks"]["gpu_vram"]
      and good["checks"]["cuda_usable"])
check("the document is sealed and names its hub verdict source",
      len(good["attestation_sha256"]) == 64
      and good["hub_tls_verdict_source"] in ("tlsguard", "inline-floor"))
check("the document NAMES the attester of each half: a postcondition "
      "evaluated by the party it constrains is not proof, so the box's "
      "self-report is labelled as not independently verifiable",
      set(good["evidence_sources"]) == {"observed", "provider_record",
                                        "checks", "hub_tls_verdict"}
      and "not independently verifiable"
      in good["evidence_sources"]["observed"]
      and "independent of" in good["evidence_sources"]["provider_record"])
check("...and the two independent parties are required to AGREE, which is "
      "what a single party's word cannot give",
      good["checks"]["provider_gpu_model_agrees"] is True
      and good["checks"]["provider_vram_agrees"] is True)
if HAS_TLSGUARD:
    check("[INFO] tlsguard is present, so its verdict governs Hub identity",
          "hub_identity_attested" in good["checks"])
else:
    check("without tlsguard the document DISCLOSES it is not Hub-identity "
          "proof",
          any("NOT that the peer is the real Hub" in text
              for text in good["hub_tls_verdict"]["disclosures"]))
    check("...and the attestation still passes on its own floor",
          good["ok"] is True and good["failures"] == [])
check("one pod-scoped filesystem is recognised as one",
      good["single_filesystem"] is True)
check("the provider record carries the machine and host that answered -- the "
      "machine id matters because a Vast host key identifies the MACHINE",
      good["provider_record"]["provider_machine_id"] == "150014"
      and good["provider_record"]["provider_host_id"] == "579267"
      and good["provider_record"]["known_bad_host"] is None)

wrong_gpu = attest(payload(gpus=[{
    "index": 0, "name": "Tesla V100", "vram_bytes": T4_SMI_BYTES,
    "driver_version": "595.71.05"}]))
check("a DIFFERENT GPU model fails the gate",
      wrong_gpu["ok"] is False and "gpu_model" in wrong_gpu["failures"])
small_vram = attest(payload(gpus=[{
    "index": 0, "name": "Tesla T4", "vram_bytes": 8 * 1024 ** 3,
    "driver_version": "595.71.05"}]))
check("the right model with the wrong VRAM fails the gate",
      small_vram["ok"] is False and "gpu_vram" in small_vram["failures"])
no_cuda = attest(payload(cuda={
    "usable": False, "count": 0, "name": None, "vram_bytes": None,
    "error": "CUDA driver version is insufficient", "interpreter": None}))
check("a box whose torch cannot see CUDA fails the gate",
      no_cuda["ok"] is False and "cuda_usable" in no_cuda["failures"])
mitm = attest(payload(hub_reachability=[
    {"host": "huggingface.co", "tls_ok": False, "cert_subject_cn": None,
     "cert_issuer_cn": None, "cert_not_after": None, "http_status": None,
     "error": "SSLEOFError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred "
              "in violation of protocol"},
    {"host": "cdn-lfs.hf.co", "tls_ok": True, "cert_subject_cn": "hf.co",
     "cert_issuer_cn": "Amazon RSA 2048 M01",
     "cert_not_after": "Feb  9 23:59:59 2027 GMT",
     "http_status": None, "error": None}]))
check("the 2026-09-05 failure mode (UNEXPECTED_EOF to the Hub, the signature "
      "of an intercepting proxy) fails the gate BEFORE upload or spend",
      mitm["ok"] is False and "hub_tls_verified" in mitm["failures"])
proxy = attest(payload(hub_reachability=[
    {"host": "huggingface.co", "tls_ok": True,
     "cert_subject_cn": "proxy.local", "cert_issuer_cn": "Host CA",
     "cert_not_after": "Feb 27 23:59:59 2027 GMT", "http_status": 502,
     "error": None},
    {"host": "cdn-lfs.hf.co", "tls_ok": True, "cert_subject_cn": "hf.co",
     "cert_issuer_cn": "Amazon RSA 2048 M01",
     "cert_not_after": "Feb  9 23:59:59 2027 GMT",
     "http_status": None, "error": None}]))
check("a Hub the box cannot fetch from (5xx) fails the gate",
      proxy["ok"] is False and "hub_api_answers" in proxy["failures"])
# Live on a healthy Vast T4, 2026-09-06: HEAD /api/models/gpt2 answered 307.
# An equality check on 200 refused that box, so this rung is the regression.
redirected = attest(payload(hub_reachability=[
    {"host": "huggingface.co", "tls_ok": True,
     "cert_subject_cn": "huggingface.co",
     "cert_issuer_cn": "Amazon RSA 2048 M01",
     "cert_not_after": "Feb 27 23:59:59 2027 GMT", "http_status": 307,
     "error": None},
    {"host": "cdn-lfs.hf.co", "tls_ok": True, "cert_subject_cn": "hf.co",
     "cert_issuer_cn": "Amazon RSA 2048 M01",
     "cert_not_after": "Feb  9 23:59:59 2027 GMT",
     "http_status": None, "error": None}]))
check("the Hub's real 307 redirect is an ANSWER, not a failure: identity is "
      "the verified certificate, reachability is only that it spoke HTTP",
      redirected["ok"] is True
      and redirected["checks"]["hub_api_answers"] is True)
throttled = attest(payload(hub_reachability=[
    {"host": "huggingface.co", "tls_ok": True,
     "cert_subject_cn": "huggingface.co",
     "cert_issuer_cn": "Amazon RSA 2048 M01",
     "cert_not_after": "Feb 27 23:59:59 2027 GMT", "http_status": 429,
     "error": None},
    {"host": "cdn-lfs.hf.co", "tls_ok": True, "cert_subject_cn": "hf.co",
     "cert_issuer_cn": "Amazon RSA 2048 M01",
     "cert_not_after": "Feb  9 23:59:59 2027 GMT",
     "http_status": None, "error": None}]))
check("a 429 fails the reachability floor while the certificate still "
      "verifies, so the host is not implied to be hostile",
      throttled["ok"] is False
      and "hub_api_answers" in throttled["failures"]
      and throttled["checks"]["hub_tls_verified"] is True)
# Live shape on machine 150014: / is a 128 GB overlay and /workspace is a
# SEPARATE xfs on the host's own 1.6 TB disk, so the pod-scoped-single-disk
# assumption is not universal and both branches must work.
two_fs = attest(payload(filesystems={
    "root": {"path": "/", "mount_point": "/", "fs_type": "overlay",
             "source": "overlay", "device": 42,
             "total_bytes": 128_849_018_880,
             "available_bytes": 128_021_987_328, "error": None},
    "workspace": {"path": "/workspace", "mount_point": "/workspace",
                  "fs_type": "xfs", "source": "/dev/sdb", "device": 2064,
                  "total_bytes": 1_599_539_908_608,
                  "available_bytes": 1_507_561_123_840, "error": None}}))
check("a Vast box whose /workspace is a SEPARATE host filesystem attests, "
      "with each byte floor applied to its own mount",
      two_fs["ok"] is True and two_fs["single_filesystem"] is False)
dead = StubVast(responses=instances(), exec_payload=None
                ).attest_live_resource(50055626, **ATTEST)
check("a box the provider calls running but SSH cannot reach fails the gate "
      "(a status field is never evidence of reachability)",
      dead["ok"] is False and dead["transport_error"]
      and "live SSH attestation unavailable" in dead["failures"])
bad_host = attest(payload(), rows=(dict(LIVE_INSTANCE, machine_id=68004),))
check("attesting on a known-bad host fails and records why",
      bad_host["ok"] is False
      and "host_not_known_bad" in bad_host["failures"]
      and "68004" in bad_host["provider_record"]["known_bad_host"])
thin_disk = attest(payload(), volume_gb=200, container_disk_gb=200)
check("a pod disk too small for volume+container fails the gate",
      thin_disk["ok"] is False
      and "disk_total_bytes" in thin_disk["failures"])
starved = attest(payload(), workspace_available_bytes_minimum=100_000_000_000,
                 container_available_bytes_minimum=100_000_000_000)
check("on ONE filesystem the two free-space floors must be met TOGETHER",
      starved["ok"] is False
      and "disk_available_bytes" in starved["failures"])
skew = attest(payload(
    remote_time_epoch=int(time.time()) - 4000,
    remote_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                  time.gmtime(time.time() - 4000))))
check("a remote clock outside the bound fails the gate",
      skew["ok"] is False and "remote_clock" in skew["failures"])
missing_ws = attest(payload(filesystems={
    "root": {"path": "/", "mount_point": "/", "fs_type": "overlay",
             "source": "overlay", "device": 66,
             "total_bytes": 128_000_000_000,
             "available_bytes": 120_000_000_000, "error": None},
    "workspace": {"path": "/workspace", "mount_point": None, "fs_type": None,
                  "source": None, "device": None, "total_bytes": None,
                  "available_bytes": None,
                  "error": "FileNotFoundError: /workspace"}}))
check("a box with no /workspace fails the gate",
      missing_ws["ok"] is False
      and "workspace_present" in missing_ws["failures"])
for bad in ({"min_vcpu": 0}, {"min_ram_gb": -1}, {"volume_gb": True}):
    refuses("attest refuses a non-positive expectation %r" % bad,
            lambda bad=bad: attest(payload(), **bad),
            needle="must be a positive integer")
refuses("attest refuses a label as an id",
        lambda: StubVast(responses=instances(), exec_payload=payload()
                         ).attest_live_resource("vast-fruit", **ATTEST),
        needle="exact integral id")

print()
print("== the on-box attestation script ==")
compile(vastapi._LIVE_ATTEST_SCRIPT, "<attest>", "exec")
check("the script compiles as python", True)
check("it probes exactly the declared hub hosts",
      all('"%s"' % host in vastapi._LIVE_ATTEST_SCRIPT
          for host in vastapi.HUB_PROBE_HOSTS))
check("it verifies certificates rather than merely connecting",
      "check_hostname = True" in vastapi._LIVE_ATTEST_SCRIPT
      and "CERT_REQUIRED" in vastapi._LIVE_ATTEST_SCRIPT)
check("it never weakens TLS",
      "CERT_NONE" not in vastapi._LIVE_ATTEST_SCRIPT
      and "check_hostname = False" not in vastapi._LIVE_ATTEST_SCRIPT
      and "_create_unverified" not in vastapi._LIVE_ATTEST_SCRIPT)
check("it touches no credential and no token path",
      "token" not in vastapi._LIVE_ATTEST_SCRIPT.lower())

print()
print("== no credential may enter a Vast CREATE body ==")
# A create payload is provider-persisted and reaches the host before any host
# key, attestation or TLS check can exist, so no ordering makes it safe. The
# CROSS-PROVIDER version of this rung is RP7b in selftest_root_publish.py,
# parameterised over all four adapters; these two keep the Vast-specific
# refusal text honest.
ASK_OK = {"/asks/": {"success": True, "new_contract": 50055626}}
refuses("create refuses a provider-carried HF credential",
        lambda: StubVast(responses=ASK_OK,
                         ssh_key="/nonexistent/id_ed25519").create(
            ask_id=42, storage=80, env={"HF_TOKEN": "hf_" + "a" * 34},
            docker_cmd=["capture"], onstart="mkdir -p /workspace",
            name="vast-x"),
        needle="provider-persisted")
refuses("create refuses a token embedded in onstart TEXT too",
        lambda: StubVast(responses=ASK_OK,
                         ssh_key="/nonexistent/id_ed25519").create(
            ask_id=42, storage=80,
            onstart="export HF_TOKEN=hf_" + "c" * 34, name="vast-x"),
        needle="onstart carries a known token shape")
refuses("create refuses a result-sink URL: a URL with a path is a bearer "
        "capability, exactly like this morning's ntfy topic",
        lambda: StubVast(responses=ASK_OK,
                         ssh_key="/nonexistent/id_ed25519").create(
            ask_id=42, storage=80,
            env={"FIDELITY_RESULT_SINK": "https://sink.invalid/topic-cred"},
            name="vast-x"),
        needle="credential-shaped")
public_run = StubVast(responses=ASK_OK,
                      ssh_key="/nonexistent/id_ed25519").create(
    ask_id=42, storage=80, docker_cmd=["capture", "--model", "gpt2"],
    env={"FIDELITY_PANEL_ID": "panel--fruit.malaiwah.heldout-v1"},
    onstart="mkdir -p /workspace", name="vast-public")
check("a PUBLIC-artifact container run with no credential still works",
      public_run["machine_id"] == 50055626)

print()
print("== prepare_safe_create: everything refuses before anything bills ==")
CREATE = dict(ask_id=50006709, name="vast-fruit", gpu_type="Tesla T4",
              storage_gb=60, container_disk_gb=60,
              image="ghcr.io/malaiwah/quant-fidelity-measure:main",
              terminate_after_epoch=time.time() + 3600)
OFFERS = {"/bundles/": {"offers": [LIVE_OFFER]}}
KEYPAIR = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       "fid-vast-selftest-key")
if not os.path.isfile(KEYPAIR + ".pub"):
    os.system("ssh-keygen -q -t ed25519 -N '' -C selftest -f %s "
              "</dev/null >/dev/null 2>&1" % KEYPAIR)
HAVE_KEYPAIR = os.path.isfile(KEYPAIR + ".pub")


def preparer(offers=None, **kw):
    return StubVast(responses=offers or OFFERS, **kw)


if not HAVE_KEYPAIR:
    check("[SKIP] no scratch SSH keypair, prepare/submit rungs not run", False)
else:
    box = preparer(ssh_key=KEYPAIR)
    prepared = box.prepare_safe_create(**CREATE)
    frozen = prepared.to_dict()
    identity = frozen["request_identity"]
    body = json.loads(prepared.request_body.decode("utf-8"))
    check("the frozen request is a PUT to the exact ask",
          frozen["method"] == "PUT"
          and frozen["url"].endswith("/asks/50006709/"))
    check("it freezes the ADVERTISED identity of the ask",
          identity["provider_machine_id"] == "150014"
          and identity["advertised_gpu_name"] == "Tesla T4"
          and identity["advertised_dph_total"].startswith("0.1355555"))
    check("it sums the two disk roles into the one disk Vast rents",
          identity["disk_gb"] == 120 and body["disk"] == 120)
    check("it pins a FRESHLY GENERATED ED25519 host key before the box "
          "exists (reading the box's own key would authenticate the MACHINE, "
          "which is stable across contracts on Vast)",
          identity["pinned_host_key_fingerprint"].startswith("SHA256:")
          and len(identity["pinned_host_key_fingerprint"]) == 50)
    check("the private half travels in env, never in onstart text",
          "FIDELITY_VAST_HOST_KEY_B64" in body["env"]
          and "PRIVATE KEY" not in body["onstart"])
    check("no key material appears in the frozen evidence document",
          "PRIVATE" not in json.dumps(frozen)
          and body["env"].split("=", 1)[1] not in json.dumps(frozen))
    check("a provider-side deadline is requested as a duration",
          3500 <= body["duration"] <= 3600
          and identity["terminate_after"].endswith("Z"))
    check("the rental is on-demand and SSH-driven",
          body["runtype"] == "ssh" and identity["is_spot"] is False)
    check("the request identity is digest-sealed",
          len(frozen["request_identity_sha256"]) == 64
          and len(frozen["request_body_sha256"]) == 64)

    for kw, needle in (
            ({"docker_cmd": ["capture"]}, "refuses native docker launch"),
            ({"env": {"HF_TOKEN": "x"}}, "caller-supplied provider env"),
            ({"spot": True}, "never a bid"),
            ({"is_bid": True}, "never a bid"),
            ({"offer": "bid"}, "offer exactly on-demand"),
            ({"region": "secure"}, "no region tiers"),
            ({"network_volume_id": "771"}, "refuses network/custom volumes"),
            ({"name": ""}, "exact lease name"),
            ({"storage_gb": 0}, "must be a positive integer"),
            ({"container_disk_gb": "x"}, "must be a positive integer"),
            ({"terminate_after_epoch": time.time() + 60},
             "at least 300 seconds in the future"),
            ({"gpu_type": "RTX 3090"}, "advertises"),
    ):
        refuses("prepare refuses %s" % sorted(kw),
                lambda kw=kw: preparer(ssh_key=KEYPAIR).prepare_safe_create(
                    **dict(CREATE, **kw)),
                needle=needle)
    refuses("prepare refuses a deadline-free create",
            lambda: preparer(ssh_key=KEYPAIR).prepare_safe_create(
                **{k: value for k, value in CREATE.items()
                   if k != "terminate_after_epoch"}),
            needle="requires terminate_after")
    refuses("prepare refuses an ask on the known-bad host",
            lambda: preparer(
                {"/bundles/": {"offers": [dict(LIVE_OFFER,
                                               machine_id=68004)]}},
                ssh_key=KEYPAIR).prepare_safe_create(**CREATE),
            needle=KNOWN_BAD_MACHINE_IDS["68004"][:28])
    refuses("prepare refuses an ask whose disk cannot hold the plan",
            lambda: preparer(
                {"/bundles/": {"offers": [dict(LIVE_OFFER, disk_space=80)]}},
                ssh_key=KEYPAIR).prepare_safe_create(**CREATE),
            needle="below the 120 GB the plan needs")
    refuses("prepare refuses a vanished ask instead of renting another",
            lambda: preparer({"/bundles/": {"offers": []}},
                             ssh_key=KEYPAIR).prepare_safe_create(**CREATE),
            needle="not exactly one rentable on-demand offer")
    refuses("prepare refuses without the controller's public key",
            lambda: preparer(ssh_key="/nonexistent/id_ed25519"
                             ).prepare_safe_create(**CREATE),
            needle="needs the controller's SSH public key")

    print()
    print("== submit_prepared_create: a lost response stays reconcilable ==")
    dry = StubVast(responses=OFFERS, ssh_key=KEYPAIR, dry=True)
    dry_prepared = dry.prepare_safe_create(**CREATE)
    check("dry mode submits nothing and returns the frozen request",
          dry.submit_prepared_create(dry_prepared)["dry_run"] is True
          and not [c for c in dry.calls if c[0] == "PUT"])

    def submit(response):
        box_local = StubVast(responses=OFFERS, ssh_key=KEYPAIR)
        prepared_local = box_local.prepare_safe_create(**CREATE)
        raw = json.dumps(response).encode("utf-8")

        class Fake:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                return raw

        object.__setattr__(prepared_local, "http_opener",
                           lambda request, timeout=180: Fake())
        return box_local, box_local.submit_prepared_create(prepared_local)

    box, accepted = submit({"success": True, "new_contract": 50055626})
    check("an accepted rental yields the contract id as an exact string",
          accepted["machine_id"] == "50055626"
          and accepted["pinned_host_key_fingerprint"].startswith("SHA256:"))
    check("...and the pin is remembered for the host-key check",
          box._pinned_host_keys["50055626"]
          == accepted["pinned_host_key_fingerprint"])
    refuses("an explicit refusal is a REJECTION, so no id needs hunting",
            lambda: submit({"success": False, "msg": "no_such_ask"}),
            needle="no_such_ask", kind=VastCreateRejectedError)
    try:
        submit({"success": False, "msg": "timeout",
                "new_contract": 50055626})
        check("an ambiguous failure raises", False)
    except VastCreateRejectedError:
        check("an ambiguous failure is NOT classified as a definitive "
              "rejection", False)
    except VastError as exc:
        check("an ambiguous failure stays fail-closed, never a rejection",
              "refused the rental" in str(exc))
    refuses("a success without a contract id refuses",
            lambda: submit({"success": True}), needle="new_contract")

print()
print("== ssh_host_ed25519_fingerprint: no trust on first use ==")
VAST_LOG_TAIL = (
    "Server listening on 0.0.0.0 port 22.\n"
    "Server listening on :: port 22.\n"
    "Warning: Permanently added 'ssh2.vast.ai' (ED25519) to the list of "
    "known hosts.\n"
    "Error: remote port forwarding failed for listen port 14270\n")
refuses("an unmodified Vast image exposes NO instance fingerprint, so this "
        "refuses instead of trusting the jump proxy's key",
        lambda: StubVast(responses=instances(), log_text=VAST_LOG_TAIL
                         ).ssh_host_ed25519_fingerprint(50055626, timeout=1),
        needle="belong to the ssh2.vast.ai jump proxy")
check("the proxy warning line does not match the fingerprint pattern",
      vastapi._HOST_KEY_LOG_RE.fullmatch(
          "Warning: Permanently added 'ssh2.vast.ai' (ED25519) to the list "
          "of known hosts.") is None)
PRINTED = "256 SHA256:" + "A" * 43 + " root@c.50055626 (ED25519)"
evidence = StubVast(responses=instances(),
                    log_text=VAST_LOG_TAIL + PRINTED + "\n"
                    ).ssh_host_ed25519_fingerprint(50055626, timeout=1)
check("a printed listing line IS accepted from the authenticated log channel",
      evidence["source"] == "authenticated-instance-log"
      and evidence["fingerprint"] == "SHA256:" + "A" * 43
      and len(evidence["line_sha256"]) == 64)


class PinnedVast(StubVast):
    def __init__(self, scanned, **kw):
        super().__init__(**kw)
        self._scanned = scanned

    def scan_host_key(self, machine_id):
        if isinstance(self._scanned, Exception):
            raise self._scanned
        return self._scanned


PIN = "SHA256:" + "B" * 43
SCAN = {"host": "ssh3.vast.ai", "port": 15626, "algorithm": "ssh-ed25519",
        "fingerprint": PIN, "known_hosts_entry": "[ssh3.vast.ai]:15626 x y\n"}
v = PinnedVast(SCAN, responses=instances())
v._pinned_host_keys["50055626"] = PIN
pinned_evidence = v.ssh_host_ed25519_fingerprint(50055626, timeout=5)
check("a pinned key is authenticated with no first-contact trust at all",
      pinned_evidence["source"] == "pinned-at-create"
      and pinned_evidence["fingerprint"] == PIN
      and pinned_evidence["pinned_fingerprint"] == PIN)
other = PinnedVast(dict(SCAN, fingerprint="SHA256:" + "C" * 43),
                   responses=instances())
other._pinned_host_keys["50055626"] = PIN
refuses("a box presenting ANOTHER key is refused and its host recorded",
        lambda: other.ssh_host_ed25519_fingerprint(50055626, timeout=1),
        needle="not the box we asked for")
for bad in (0, -1, float("nan")):
    refuses("a non-positive host-key timeout refuses (%r)" % bad,
            lambda bad=bad: StubVast(responses=instances()
                                     ).ssh_host_ed25519_fingerprint(
                                         50055626, timeout=bad),
            needle="finite and positive")

print()
print("== billing_history: per-instance, per UTC day, client-side selection ==")
CHARGES = {"success": True, "count": 3, "total": 3, "next_token": None,
           "results": [charge_row("instance-50055626"),
                       charge_row("instance-49993424", gpu=0.0, disk=0.003),
                       charge_row("instance-50055626", day=DAY - 1,
                                  gpu=0.01, disk=0.005)]}
v = StubVast(responses={"/charges/": CHARGES})
history = v.billing_history(50055626, start_time="2026-09-04T00:00:00Z",
                            end_time="2026-09-05T23:59:59Z")
check("only the rows whose source is THIS instance are counted",
      history["metadata"]["matched_row_count"] == 2
      and history["metadata"]["account_rows_examined"] == 3)
check("the total is the exact decimal sum of the matched rows",
      history["total_amount"] == "0.026")
check("the per-kind totals are recorded",
      history["metadata"]["totals"]["gpu"] == "0.017"
      and history["metadata"]["totals"]["disk"] == "0.009")
check("the response records that selection is OURS, not the query's",
      "ignores server-side instance filters"
      in history["metadata"]["selection"])
refuses("an hourly bucket is refused, not approximated",
        lambda: v.billing_history(50055626,
                                  start_time="2026-09-04T00:00:00Z",
                                  end_time="2026-09-05T00:00:00Z",
                                  bucket_size="hour"),
        needle="publishes no finer bucket")
refuses("a paginated window cannot prove a total",
        lambda: StubVast(responses={"/charges/": dict(
            CHARGES, next_token="abc")}).billing_history(
            50055626, start_time="2026-09-04T00:00:00Z",
            end_time="2026-09-05T00:00:00Z"),
        needle="paginated")
refuses("a row whose items do not sum to its amount is refused",
        lambda: StubVast(responses={"/charges/": {
            "success": True, "count": 1, "total": 1, "next_token": None,
            "results": [dict(charge_row("instance-50055626"), amount=0.99)]}}
        ).billing_history(50055626, start_time="2026-09-04T00:00:00Z",
                          end_time="2026-09-05T00:00:00Z"),
        needle="items sum to")
refuses("metadata disagreeing with the rows is refused",
        lambda: StubVast(responses={"/charges/": dict(CHARGES, count=9)}
                         ).billing_history(
            50055626, start_time="2026-09-04T00:00:00Z",
            end_time="2026-09-05T00:00:00Z"),
        needle="disagrees with the rows")
refuses("no charge row for this instance leaves reconciliation UNRESOLVED",
        lambda: StubVast(responses={"/charges/": {
            "success": True, "count": 1, "total": 1, "next_token": None,
            "results": [charge_row("instance-49993424")]}}).billing_history(
            50055626, start_time="2026-09-04T00:00:00Z",
            end_time="2026-09-05T00:00:00Z"),
        needle="remains unresolved")
refuses("an end before the start is refused",
        lambda: v.billing_history(50055626,
                                  start_time="2026-09-05T00:00:00Z",
                                  end_time="2026-09-04T00:00:00Z"),
        needle="must follow start_time")

print()
print("== reconcile_billing: post-absence, day-closed, twice-stable ==")
ABSENCE = "2026-09-05T18:00:00Z"
LEASE = {
    "provider_resource_ids": [50055626],
    "create": {"pre_create_observed_at": "2026-09-05T17:00:00Z"},
    "history": [{"to": "ABSENCE_CONFIRMED", "at": ABSENCE}],
}
absence_epoch = vastapi._utc_epoch(ABSENCE, "absence")
day_end = (absence_epoch // 86400 + 1) * 86400
ONE_ROW = {"success": True, "count": 1, "total": 1, "next_token": None,
           "results": [charge_row("instance-50055626", day=DAY)]}
v = StubVast(responses={"/charges/": ONE_ROW})
refuses("inside the 300 s post-absence window it refuses",
        lambda: v.reconcile_billing(LEASE, now=absence_epoch + 10),
        needle="stabilization window")
refuses("before the absence DAY closes it refuses, naming the sweep time",
        lambda: v.reconcile_billing(LEASE, now=absence_epoch + 3600),
        needle="has not closed and stabilized")
closed = v.reconcile_billing(LEASE, now=day_end + 400)
check("after the day closes and stabilizes it closes with a sealed digest",
      closed["reconciled"] is True and closed["total_amount"] == "0.011"
      and len(closed["evidence"]["closure_sha256"]) == 64
      and closed["evidence"]["first_retrieval"]["retrieval_id"]
      != closed["evidence"]["second_retrieval"]["retrieval_id"])
check("the closure records when the absence day closed",
      closed["evidence"]["absence_day_closed_at"] == "2026-09-06T00:00:00Z")


class MovingVast(StubVast):
    """A bill that is still moving: the exact state a closure must refuse."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._served = 0

    def _req(self, method, path, body=None, **kw):
        self._served += 1
        return {"success": True, "count": 1, "total": 1, "next_token": None,
                "results": [charge_row("instance-50055626", day=DAY,
                                       gpu=0.007 + 0.001 * self._served)]}


refuses("a bill that MOVES between two independent retrievals is refused",
        lambda: MovingVast().reconcile_billing(LEASE, now=day_end + 400),
        needle="changed between independent retrievals")
refuses("a lease with no absence event cannot be reconciled",
        lambda: v.reconcile_billing(dict(LEASE, history=[]),
                                    now=day_end + 400),
        needle="no provider-absence event")
refuses("a lease with no exact ids cannot be reconciled",
        lambda: v.reconcile_billing(dict(LEASE, provider_resource_ids=[]),
                                    now=day_end + 400),
        needle="at least one exact")
refuses("a label in provider_resource_ids is refused, never coerced",
        lambda: v.reconcile_billing(
            dict(LEASE, provider_resource_ids=["vast-fruit"]),
            now=day_end + 400),
        needle="exact integral id")

print()
print("== elapsed time, the contract rate, and finding the credential ==")
inst = Vast._to_instance(LIVE_INSTANCE)
check("machine_id is the provider's own integral id",
      inst.machine_id == 50055626)
check("runtime is elapsed wall time, not the contract's remaining duration",
      inst.raw["contract_seconds_remaining"] == 15596229.0
      and inst.runtime < 15596229.0)
check("cost is priced from elapsed time, so it is dollars not thousands",
      inst.cost < 1000)
check("the CONTRACT rate is exposed beside the ask's",
      inst.raw["dph_total"] == LIVE_INSTANCE["dph_total"])
check("the conventional 0600 key path is a default, so a credential that "
      "EXISTS is found instead of teaching an operator to export secrets",
      vastapi.DEFAULT_KEY_FILE.endswith("/.config/vastai/vast_api_key")
      and os.path.isabs(vastapi.DEFAULT_KEY_FILE))
refuses("an explicitly named key file that is absent refuses rather than "
        "silently using a different credential",
        lambda: Vast(key_file="/nonexistent/vast_api_key")._load_key(),
        needle="does not exist")

print()
if FAILED:
    print("selftest_vast_contract: %d FAILED" % len(FAILED))
    for label in FAILED:
        print("  - %s" % label)
    sys.exit(1)
print("selftest_vast_contract: all passed")
