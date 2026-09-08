#!/usr/bin/env python3
"""Isolated experiment backstop. No provider cutoff or billing guarantee.

Run by a root systemd service; its credential is a separate root-owned 0600
file, never command text. Only the exact frozen instance can be terminated.
No raw provider documents, response bodies or exception messages are logged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys
import time


class Refusal(RuntimeError):
    pass




def private_bytes(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o600 or st.st_uid != os.getuid():
            raise Refusal("invalid private file")
        data = stream.read(65537)
        if len(data) > 65536:
            raise Refusal("private file too large")
        return data


def publish(root, state, **fields):
    path = root / "heartbeat.json"
    temporary = root / (".heartbeat-%d-%d" % (os.getpid(), time.time_ns()))
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(dict(state=state, heartbeat_epoch=time.time(), **fields), stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Backstop:
    def __init__(self, root):
        self.root = root
        self.config = json.loads(private_bytes(root / "binding.json"))
        c = self.config
        if set(c) != {"provider_id", "name", "gpu_type", "region", "deadline_epoch"}:
            raise Refusal("invalid binding fields")
        if not isinstance(c["provider_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", c["provider_id"]):
            raise Refusal("invalid instance ID")
        if not re.fullmatch(r"qfs-native-[0-9a-f]{32}", c["name"]):
            raise Refusal("invalid owned name")
        if type(c["deadline_epoch"]) not in (float, int) or not 0 < c["deadline_epoch"] < 10**11:
            raise Refusal("invalid deadline")
        sys.path.insert(0, str(root))
        from fidelity.lambdaapi import LambdaCloud
        self.provider = LambdaCloud(key_file=str(root / "api_key"))

    def request(self, method, path, body=None):
        from fidelity.lambdaapi import LambdaError
        try:
            _, document = self.provider._request(method, path, body, timeout=10)
        except LambdaError as exc:
            if (method == "GET" and path == "/instances/" + self.config["provider_id"]
                    and getattr(exc, "status", None) == 404):
                return None
            raise
        self.provider.server_time_evidence(max_clock_delta_seconds=60)
        if not isinstance(document, dict) or "error" in document or "data" not in document:
            raise Refusal("invalid provider response")
        return document["data"]

    def verify(self, row):
        c = self.config
        if (not isinstance(row, dict) or row.get("id") != c["provider_id"]
                or row.get("name") != c["name"]
                or (row.get("instance_type") or {}).get("name") != c["gpu_type"]
                or (row.get("region") or {}).get("name") != c["region"]):
            raise Refusal("exact owned instance binding mismatch")

    def inventory(self):
        rows = self.request("GET", "/instances")
        if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows):
            raise Refusal("incomplete inventory")
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise Refusal("duplicate inventory ID")
        return rows

    def run(self):
        c = self.config
        instance_path = "/instances/" + c["provider_id"]
        # A readable API credential and a live exact binding are prerequisites
        # to ARMED, not assumptions made after expensive bootstrap starts.
        self.verify(self.request("GET", instance_path))
        owned = [r for r in self.inventory() if r["id"] == c["provider_id"]]
        if len(owned) != 1:
            raise Refusal("owned ID missing from inventory")
        self.verify(owned[0])
        while time.time() < c["deadline_epoch"]:
            private_bytes(self.root / "api_key")
            publish(self.root, "armed", provider_id=c["provider_id"], deadline_epoch=c["deadline_epoch"])
            time.sleep(min(5, max(0, c["deadline_epoch"] - time.time())))
        confirmations = 0
        while True:
            try:
                detail = self.request("GET", instance_path)
                rows = self.inventory()
                owned = [r for r in rows if r["id"] == c["provider_id"]]
                if detail is None and not owned:
                    confirmations += 1
                    publish(self.root, "absence_observed", confirmations=confirmations)
                    if confirmations >= 2:
                        (self.root / "api_key").unlink()
                        publish(self.root, "closed", confirmations=confirmations)
                        return 0
                else:
                    confirmations = 0
                    if detail is not None:
                        self.verify(detail)
                    if owned:
                        self.verify(owned[0])
                    # Never mutate using an inventory-only/absent exact detail.
                    if detail is None:
                        raise Refusal("detail/inventory disagreement")
                    if detail.get("status") not in ("terminating", "terminated"):
                        self.request("POST", "/instance-operations/terminate", {"instance_ids": [c["provider_id"]]})
                    publish(self.root, "cleanup_pending", provider_id=c["provider_id"])
            except Exception as exc:
                confirmations = 0
                publish(self.root, "unresolved", error_class=type(exc).__name__)
            time.sleep(5)


def main():
    os.umask(0o077)
    if len(sys.argv) != 2 or os.getuid() != 0:
        return 2
    root = Path(sys.argv[1])
    st = root.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or stat.S_IMODE(st.st_mode) != 0o700:
        return 2
    try:
        return Backstop(root).run()
    except Exception as exc:
        # systemd restarts failures, including a transient initial API outage.
        publish(root, "unarmed", error_class=type(exc).__name__,
                http_status=getattr(exc, "status", None), provider_code=getattr(exc, "code", None))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
