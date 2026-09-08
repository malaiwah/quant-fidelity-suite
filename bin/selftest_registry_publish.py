#!/usr/bin/env python3
"""Guarded publication behavior, stock Python, hermetic, no token/network/spend.

The fake public repository enforces parent-CAS and retains actual bytes. Tests
observe refusals, unchanged state and consumer retrieval, not SDK call echoes.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import tempfile
from unittest import mock

sys.dont_write_bytecode = True
import registry_publish as P

PARENT = "a" * 40
NEXT = "b" * 40
FAILURES = []
ASSERTIONS = 0


def check(label, condition):
    global ASSERTIONS
    ASSERTIONS += 1
    print("  %s  %s" % ("PASS" if condition else "FAIL", label))
    if not condition:
        FAILURES.append(label)


def refused(call):
    try:
        call()
    except (P.Refusal, P.dshub.HubError):
        return True
    return False


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def snapshot(rows=None):
    files = {"README.md": b"Public fixture registry.\n", "tools/seed_registry.py": b"# source\n"}
    collections = {}
    for name in P.COLLECTIONS:
        records = (rows if rows is not None else [{"id": "measurement--old", "value": 0.25}]) if name == "measurements" else []
        path = "data/%s.jsonl" % name
        files[path] = b"".join(encoded(row) for row in records)
        collections[name] = {"file": path, "sha256": P.sha256(files[path]), "record_count": len(records)}
    files["index.json"] = encoded({"collections": collections})
    return files


def approval_for(plan):
    return {"schema": P.APPROVAL_SCHEMA, "repository": P.REPOSITORY,
            "parent_commit": plan["parent_commit"], "collection_sha256": plan["collection_sha256"],
            "field_changes": plan["field_changes"], "review": {"reviewer": "selftest", "reason": "Reviewed fixture correction"}}


def entry(body):
    return {"size": len(body), "blobId": hashlib.sha1(("blob %d\0" % len(body)).encode() + body).hexdigest()}


class Repository:
    verify = P.PublicHub.verify

    def __init__(self, files):
        self.files = dict(files)
        self.head = PARENT
        self.commits = 0
        self.race = False
        self.corrupt_download = False

    def inventory(self, revision="main"):
        if revision != "main" and revision != self.head:
            raise P.Refusal("unknown revision")
        return self.head, {path: entry(body) for path, body in self.files.items()}

    def snapshot(self, local):
        return self.head, dict(self.files)

    def commit(self, parent, changes, token, message):
        if self.race:
            self.head = "c" * 40
        if parent != self.head:
            raise P.Refusal("remote parent changed")
        self.files.update(changes)
        self.head = NEXT
        self.commits += 1
        return self.head

    def download(self, revision, path, metadata):
        if revision != self.head:
            raise P.Refusal("un-pinned retrieval")
        return b"corrupt" if self.corrupt_download else self.files[path]


def main():
    remote = snapshot()
    deleted = P.build_plan(snapshot([]), PARENT, remote)
    check("public-only row deletion refuses and names the lost id",
          bool(deleted["refusals"]) and deleted["collections"]["measurements"]["removed_ids"] == ["measurement--old"])

    changed = snapshot([{"id": "measurement--old", "value": 0.25000000000000006}])
    plan = P.build_plan(changed, PARENT, remote)
    check("one representable numeric step refuses without exact approval",
          bool(plan["refusals"]) and len(plan["field_changes"]) == 1)
    approval = approval_for(plan)
    accepted = P.build_plan(changed, PARENT, remote, approval)
    check("reviewed exact numeric correction is accepted", not accepted["refusals"])
    for key, replacement in (("parent_commit", "d" * 40), ("collection_sha256", {}), ("field_changes", [])):
        bad = dict(approval, **{key: replacement})
        check("approval tamper refuses: " + key, bool(P.build_plan(changed, PARENT, remote, bad)["refusals"]))
    changed_again = snapshot([{"id": "measurement--old", "value": 0.2500000000000001}])
    check("approval cannot authorize a neighboring unreviewed number",
          bool(P.build_plan(changed_again, PARENT, remote, approval)["refusals"]))

    equivalent = snapshot([{"id": "measurement--old", "value": 1.0}])
    integer = snapshot([{"id": "measurement--old", "value": 1}])
    noop = P.build_plan(equivalent, PARENT, integer)
    check("equal numeric spelling is not a scientific delta", not noop["field_changes"] and not noop["refusals"])
    boolean = P.build_plan(snapshot([{"id": "measurement--old", "value": True}]), PARENT, integer)
    check("boolean is not an equal numeric spelling", bool(boolean["field_changes"]) and bool(boolean["refusals"]))

    added = snapshot([{"id": "measurement--old", "value": 0.25}, {"id": "measurement--new", "value": 0.5}])
    hub = Repository(remote)
    add_plan = P.build_plan(added, PARENT, remote)
    P.publish(add_plan, added, remote, hub, "fixture-token", "fixture addition")
    check("addition publishes exact consumer bytes and preserves every existing id",
          add_plan["status"] == "published_verified" and hub.files == added and hub.commits == 1)
    hub = Repository(remote)
    hub.head = "d" * 40
    check("head changed before upload refuses without mutation",
          refused(lambda: P.publish(P.build_plan(added, PARENT, remote), added, remote, hub, "fixture", "m"))
          and hub.files == remote and hub.commits == 0)
    class SDKRepository(Repository):
        commit = P.PublicHub.commit

        def whoami(self, **kwargs):
            return {"name": "malaiwah"}

        def create_commit(self, *, operations, parent_commit=None, **kwargs):
            if self.race:
                self.head = "c" * 40
                self.files = snapshot([{"id": "measurement--old", "value": 0.25},
                                       {"id": "measurement--concurrent", "value": 0.125}])
            if parent_commit is not None and parent_commit != self.head:
                raise P.Refusal("parent mismatch")
            self.files.update({op.path_in_repo: op.path_or_fileobj for op in operations})
            self.head = NEXT
            self.commits += 1
            return SimpleNamespace(oid=NEXT)

    # Exercise the real SDK-facing commit implementation. The fake SDK has
    # the Hub's actual dangerous default: no parent means unconditional write.
    sdk = SDKRepository(remote)
    sdk.race = True
    sdk_module = SimpleNamespace(HfApi=lambda **kwargs: sdk,
                                 CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs))
    with mock.patch.dict(sys.modules, {"huggingface_hub": sdk_module}):
        cas_refused = refused(lambda: P.publish(P.build_plan(added, PARENT, remote),
                                               added, remote, sdk, "fixture", "m"))
    concurrent_ids = [row["id"] for row in map(json.loads, sdk.files["data/measurements.jsonl"].splitlines())]
    check("real SDK parent CAS preserves a row published during commit",
          cas_refused and sdk.commits == 0 and concurrent_ids == ["measurement--old", "measurement--concurrent"])
    sdk.race = False
    sdk.head, sdk.files = PARENT, dict(remote)
    with mock.patch.dict(sys.modules, {"huggingface_hub": sdk_module}):
        sdk_plan = P.build_plan(added, PARENT, remote)
        P.publish(sdk_plan, added, remote, sdk, "fixture", "m")
    check("real SDK commit route publishes retrievable exact bytes",
          sdk_plan["status"] == "published_verified" and sdk.files == added)
    hub = Repository(remote)
    hub.corrupt_download = True
    corrupt_plan = P.build_plan(added, PARENT, remote)
    check("post-commit byte mismatch is never reported as verified",
          refused(lambda: P.publish(corrupt_plan, added, remote, hub, "fixture", "m"))
          and corrupt_plan["status"] == "committed_unverified" and corrupt_plan["mutation_performed"])
    class LostResponseRepository(Repository):
        def commit(self, parent, changes, token, message):
            super().commit(parent, changes, token, message)
            raise P.Refusal("connection lost after server accepted commit")

    lost = LostResponseRepository(remote)
    lost_plan = P.build_plan(added, PARENT, remote)
    check("lost commit response cannot claim no mutation",
          refused(lambda: P.publish(lost_plan, added, remote, lost, "fixture", "m"))
          and lost.files == added and lost_plan["mutation_performed"] is None
          and lost_plan["status"] == "commit_outcome_unknown")
    hub = Repository(remote)
    identical_plan = P.build_plan(remote, PARENT, remote)
    P.publish(identical_plan, remote, remote, hub, "fixture", "m")
    check("already converged bytes create no commit", hub.commits == 0 and identical_plan["status"] == "already_converged")

    extra_remote = dict(remote, **{"protocol/public-receipt.json": b"sealed historical bytes"})
    omission = P.build_plan(remote, PARENT, extra_remote)
    check("remote-only evidence cannot be silently omitted", omission["remote_only"] == ["protocol/public-receipt.json"] and bool(omission["refusals"]))
    altered = dict(extra_remote, **{"protocol/public-receipt.json": b"rewritten"})
    rewrite = P.build_plan(altered, PARENT, extra_remote)
    check("approval cannot rewrite historical receipt bytes",
          bool(P.build_plan(altered, PARENT, extra_remote, approval_for(rewrite))["refusals"]))
    metadata = dict(remote, **{".gitattributes": b"*.json filter=lfs\n"})
    metadata_plan = P.build_plan(added, PARENT, metadata)
    hub = Repository(metadata)
    P.publish(metadata_plan, added, metadata, hub, "fixture", "m")
    check("Hub transport metadata is explicitly retained, never deleted",
          hub.files[".gitattributes"] == metadata[".gitattributes"] and bool(metadata_plan["retained_remote_metadata"]))

    old_paths = dict(remote, **{"README.md": b"Recorded path /home/historical/capture.json\n"})
    same_paths = dict(old_paths, **{"README.md": old_paths["README.md"] + b"Additive public caveat.\n"})
    historical = P.build_plan(same_paths, PARENT, old_paths)
    check("previously public historical paths retain exact bytes without a blanket exemption",
          not historical["refusals"] and historical["historical_private_paths"] == {"README.md": 1})
    new_paths = dict(same_paths, **{"README.md": same_paths["README.md"] + b"New /home/private/current.json\n"})
    private_plan = P.build_plan(new_paths, PARENT, old_paths)
    check("new private host path refuses", bool(private_plan["refusals"]) and len(private_plan["scan_findings"]) == 1)
    scan_approval = approval_for(private_plan)
    finding = private_plan["scan_findings"][0]
    scan_approval["scan_exceptions"] = [{k: finding[k] for k in ("path", "sha256", "finding_sha256")} | {"reason": "Reviewed synthetic documentation example, not a host disclosure"}]
    check("exact reviewed false positive is explicit in plan",
          not P.build_plan(new_paths, PARENT, old_paths, scan_approval)["refusals"])
    changed_path_file = dict(new_paths, **{"README.md": new_paths["README.md"] + b"altered\n"})
    check("scan exception is invalid after any file-byte change",
          bool(P.build_plan(changed_path_file, PARENT, old_paths, scan_approval)["refusals"]))
    secret = "hf_" + "X" * 32
    leaked = dict(remote, **{"README.md": secret.encode()})
    secret_plan = P.build_plan(leaked, PARENT, remote, token=secret)
    check("exact credential bytes cannot be approved", any(f["kind"] == "exact_credential" for f in secret_plan["scan_findings"]) and bool(secret_plan["refusals"]))

    with tempfile.TemporaryDirectory(prefix="qfs-publisher-selftest-") as temporary:
        root = Path(temporary)
        credential = root / "credential"
        credential.write_text(secret)
        credential.chmod(0o644)
        check("world-readable credential refuses", refused(lambda: P.dshub.read_token(str(credential))))
        credential.chmod(0o600)
        link = root / "credential-link"
        link.symlink_to(credential)
        check("symlink credential refuses", refused(lambda: P.dshub.read_token(str(link))))
        check("explicit owned 0600 token works and is redacted",
              P.dshub.read_token(str(credential)) == secret and secret not in P.common.redact(secret))
        approval_file = root / "approval.json"
        approval_file.write_bytes(encoded(approval))
        digest = P.sha256(approval_file.read_bytes())
        check("reviewed approval digest is accepted", P.load_approval(approval_file, digest) == approval)
        approval_file.write_bytes(encoded(dict(approval, review={"reviewer": "other", "reason": "changed"})))
        check("approval file-byte tamper refuses despite unchanged numeric permission",
              refused(lambda: P.load_approval(approval_file, digest)))

        # Exercise the actual CLI entrypoint with a temporary byte-backed Hub.
        # No constructor/import of the upload SDK and no token read is possible.
        hub = Repository(remote)
        before = dict(hub.files)
        out = io.StringIO()
        checks = [{"target": "validate-strict", "returncode": 0,
                   "report": {"errors": 0, "warnings": 0, "findings": []}}]
        with mock.patch.object(P, "authored_snapshot", return_value=added), \
                mock.patch.object(P, "prerequisites", return_value=checks), \
                mock.patch.object(P, "PublicHub", return_value=hub), \
                mock.patch.object(P.dshub, "read_token", side_effect=AssertionError("dry-run read credential")), \
                mock.patch.dict(os.environ, {"HF_TOKEN": secret}), contextlib.redirect_stdout(out):
            rc = P.main(["--dry-run", "--token-file", str(credential)])
        cli_plan = json.loads(out.getvalue())
        check("CLI dry-run never reads credentials or mutates the remote",
              rc == 0 and not cli_plan["mutation_performed"] and hub.files == before and hub.commits == 0)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"HF_TOKEN": secret}), contextlib.redirect_stdout(out):
            rc = P.main(["--execute"])
        check("CLI execute refuses ambient credentials without explicit file",
              rc == 2 and not json.loads(out.getvalue())["mutation_performed"] and secret not in out.getvalue())

        # Git-authored source discovery excludes caches, refuses omissions and
        # does not sweep untracked evidence into an upload without review.
        checkout = root / "checkout"
        (checkout / "registry").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        for path, body in remote.items():
            dest = checkout / "registry" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
        subprocess.run(["git", "add", "registry"], cwd=str(checkout), check=True)
        check("authored inventory reads exact staged public sources", P.authored_snapshot(checkout) == remote)
        (checkout / "registry/new-evidence.json").write_bytes(b"{}\n")
        check("unreviewed untracked evidence refuses rather than being silently omitted",
              refused(lambda: P.authored_snapshot(checkout)))
        (checkout / "registry/new-evidence.json").unlink()
        (checkout / "registry/README.md").unlink()
        check("missing authored source refuses rather than being silently deleted",
              refused(lambda: P.authored_snapshot(checkout)))

        # Warning disposition is a separate explicit approval, not a weakening
        # of the canonical strict warning-free release gate.
        finding = {"check": "TEST-001", "severity": "warn", "id": "measurement--old", "message": "Missing independent evidence", "remedy": None}
        counts = {name: (1 if name == "measurements" else 0) for name in P.COLLECTIONS}
        audit = {"validator": {"errors": 0, "warnings": 1}, "record_counts": counts,
                 "data_sha256": {p: P.sha256(b) for p, b in remote.items() if p.startswith("data/") or p == "index.json"},
                 "dispositions": [dict(finding, status="missing-evidence-retained", rationale="Retain disclosed limitation")]}
        audited = dict(remote, **{"publication-audit.json": encoded(audit)})
        warn_plan = P.build_plan(audited, PARENT, audited)
        warn_approval = approval_for(warn_plan)
        warn_approval["warning_disposition"] = {"audit_sha256": P.sha256(audited["publication-audit.json"]), "warning_count": 1}
        warning_checks = [{"target": "validate-strict", "returncode": 2,
                           "report": {"errors": 0, "warnings": 1, "findings": [finding], "counts": counts}}]
        warn_plan = P.build_plan(audited, PARENT, audited, warn_approval)
        P.check_prerequisites(warn_plan, audited, warn_approval, warning_checks, root)
        check("fully dispositioned warning permits snapshot, not warning-free release certification",
              not warn_plan["refusals"] and warn_plan["warnings_dispositioned"] and not warn_plan["strict_warning_free_release_gate_passed"])
        changed_warning = copy.deepcopy(warning_checks)
        changed_warning[0]["report"]["findings"][0]["message"] = "Different missing evidence"
        warn_plan = P.build_plan(audited, PARENT, audited, warn_approval)
        P.check_prerequisites(warn_plan, audited, warn_approval, changed_warning, root)
        check("same warning count cannot hide changed findings", bool(warn_plan["refusals"]))
        warn_plan = P.build_plan(audited, PARENT, audited)
        P.check_prerequisites(warn_plan, audited, None, warning_checks, root)
        check("unapproved warnings remain a strict publication refusal", bool(warn_plan["refusals"]))

    print("selftest_registry_publish: %d assertions, %d failures, 0 skipped" % (ASSERTIONS, len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
