#!/usr/bin/env python3
"""Plan a guarded, no-spend publication of the canonical public HF registry.

Dry-run is the default. Reads are anonymous and pinned; only --execute imports
huggingface_hub or reads the explicit protected credential file. No mirror,
delete, merge, regeneration, audit rewriting, repository creation or paid API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
from fidelity import common, dshub, dsformat

REPOSITORY = "malaiwah/quant-fidelity-registry"
ENDPOINT = "https://huggingface.co"
SCHEMA = "qfs/registry-publication-plan/v1"
APPROVAL_SCHEMA = "qfs/registry-publication-approval/v1"
COLLECTIONS = ("artifacts", "measurements", "models", "panels", "pipelines", "references")
# Hub owns this transport metadata. It is explicitly retained, never deleted.
REMOTE_METADATA = {".gitattributes"}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{40}\Z")
PRIVATE_PATH = re.compile(
    r"(?:/(?:home|root|Users|private|tmp|workspace|var/tmp)/|file:///|[A-Za-z]:\\Users\\)"
    r"[^\s\"'<>`\]\[{},;)]*")
KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bAKIA[A-Z0-9]{16}\b")
MISSING = object()


class Refusal(Exception):
    pass


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def parse_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Refusal("duplicate JSON key %r" % key)
            result[key] = value
        return result
    return json.loads(body, object_pairs_hook=pairs,
                      parse_constant=common.reject_nonfinite_token)


def safe_path(path):
    if (not isinstance(path, str) or not path or "\\" in path
            or any(ord(ch) < 32 for ch in path)
            or str(PurePosixPath(path)) != path or path.startswith("/")
            or ".." in PurePosixPath(path).parts):
        raise Refusal("unsafe publication member path")
    return path


def authored_snapshot(root):
    """Git's authored inventory, not a recursive upload of a working directory."""
    def git_paths(*args):
        return subprocess.check_output(
            ["git", "ls-files", "-z"] + list(args) + ["--", "registry/"],
            cwd=str(root)).decode("utf-8").split("\0")
    untracked = sorted(p for p in git_paths("--others", "--exclude-standard") if p)
    if untracked:
        raise Refusal("unstaged new registry sources: git add the reviewed files first: "
                      + ", ".join(untracked))
    files = {}
    total = 0
    registry = root / "registry"
    for name in sorted(set(p for p in git_paths("--cached") if p)):
        rel = safe_path(name[len("registry/"):])
        if (dsformat.looks_like_a_credential(rel)
                or any(p in (".cache", "__pycache__", ".venv") for p in PurePosixPath(rel).parts)
                or rel.endswith((".pyc", ".pyo"))):
            raise Refusal("credential/cache member is authored in registry: %s" % rel)
        source = registry / rel
        if any(p.is_symlink() for p in (source,) + tuple(source.parents)
               if p != root.parent):
            raise Refusal("symlink in authored publication path: %s" % rel)
        if not source.is_file():
            raise Refusal("authored public source is missing/nonregular: %s" % rel)
        size = source.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_SNAPSHOT_BYTES:
            raise Refusal("registry exceeds metadata-only publication size limit: %s" % rel)
        files[rel] = source.read_bytes()
    if not files:
        raise Refusal("no Git-authored registry sources found")
    return files


def matches_member(body, entry):
    if len(body) != entry.get("size"):
        return False
    lfs = entry.get("lfs")
    if isinstance(lfs, dict):
        return sha256(body) == lfs.get("sha256")
    header = ("blob %d\0" % len(body)).encode("ascii")
    return hashlib.sha1(header + body).hexdigest() == entry.get("blobId")


class PublicHub:
    """Anonymous stdlib reads; neither ambient tokens nor login caches are used."""
    def get(self, url):
        request = urllib.request.Request(url, headers={"User-Agent": "qfs-registry-publish/1"})
        with common.safe_urlopen(request, timeout=90) as response:
            body = response.read(MAX_FILE_BYTES + 1)
        if len(body) > MAX_FILE_BYTES:
            raise Refusal("remote metadata member exceeds size limit")
        return body

    def inventory(self, revision="main"):
        url = ENDPOINT + "/api/datasets/" + REPOSITORY + "/revision/"
        doc = parse_json(self.get(url + urllib.parse.quote(revision, safe="") + "?blobs=true"))
        if doc.get("id") != REPOSITORY or doc.get("private") is not False:
            raise Refusal("canonical destination is not confirmed public")
        parent = doc.get("sha")
        if not isinstance(parent, str) or not SHA.fullmatch(parent):
            raise Refusal("remote did not resolve to an immutable 40-hex revision")
        if revision != "main" and parent != revision:
            raise Refusal("remote immutable revision did not resolve exactly")
        siblings = doc.get("siblings")
        if not isinstance(siblings, list) or not siblings:
            raise Refusal("remote did not return its complete file inventory")
        entries = {}
        total = 0
        for entry in siblings:
            path = safe_path(entry.get("rfilename"))
            size = entry.get("size")
            if path in entries or type(size) is not int or size < 0:
                raise Refusal("invalid/duplicate remote member metadata: %s" % path)
            total += size
            if size > MAX_FILE_BYTES or total > MAX_SNAPSHOT_BYTES:
                raise Refusal("remote registry exceeds metadata-only size limit")
            entries[path] = entry
        return parent, entries

    def download(self, revision, path, entry):
        url = (ENDPOINT + "/datasets/" + REPOSITORY + "/resolve/" + revision
               + "/" + urllib.parse.quote(path, safe="/"))
        body = self.get(url)
        if not matches_member(body, entry):
            raise Refusal("download bytes do not match pinned remote inventory: %s" % path)
        return body

    def snapshot(self, local):
        parent, entries = self.inventory()
        remote = {}
        for path, entry in sorted(entries.items()):
            # Always download the public data and index. Other byte-identical
            # members need no redundant download: the Hub's blob/LFS digest
            # binds the exact local bytes to this pinned remote tree.
            if (not path.startswith("data/") and path != "index.json"
                    and path in local and matches_member(local[path], entry)):
                remote[path] = local[path]
            else:
                remote[path] = self.download(parent, path, entry)
        return parent, remote

    def commit(self, parent, changes, token, message):
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ImportError:
            raise Refusal("--execute needs huggingface_hub in the selected interpreter; no install was attempted")
        api = HfApi(endpoint=ENDPOINT, token=token)
        identity = api.whoami(token=token)
        if not dshub._write_namespace_allowed(identity, REPOSITORY.split("/", 1)[0]):
            raise Refusal("credential principal lacks declared namespace write authority")
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=body)
                      for path, body in sorted(changes.items())]
        try:
            commit = api.create_commit(
                repo_id=REPOSITORY, repo_type="dataset", revision="main",
                operations=operations, parent_commit=parent, token=token,
                commit_message=message)
        except Exception as exc:
            raise Refusal("parent-commit publication failed; re-resolve/review before retrying (no force): %s"
                          % common.redact(str(exc)))
        revision = getattr(commit, "oid", None)
        if not isinstance(revision, str) or not SHA.fullmatch(revision):
            raise Refusal("upload returned no immutable revision; inspect remote before retrying")
        return revision

    def verify(self, revision, expected):
        _, entries = self.inventory(revision)
        if set(entries) != set(expected):
            raise Refusal("committed file inventory differs from the approved tree")
        for path, body in sorted(expected.items()):
            if not matches_member(body, entries[path]):
                raise Refusal("committed bytes differ from approved bytes: %s" % path)
        # Exercise public retrieval of every changed byte, in addition to
        # checking the exact complete tree against immutable blob/LFS digests.
        return entries


def load_collections(files, label):
    if "index.json" not in files:
        raise Refusal("%s index.json is absent" % label)
    index = parse_json(files["index.json"])
    descriptors = index.get("collections")
    if not isinstance(descriptors, dict) or set(descriptors) != set(COLLECTIONS):
        raise Refusal("%s index must name exactly the six canonical collections" % label)
    collections = {}
    hashes = {}
    for name in COLLECTIONS:
        path = "data/%s.jsonl" % name
        if path not in files:
            raise Refusal("%s collection is absent: %s" % (label, path))
        body = files[path]
        rows = {}
        for line in body.splitlines():
            if not line.strip():
                continue
            row = parse_json(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise Refusal("%s %s contains a row without an id" % (label, name))
            if row["id"] in rows:
                raise Refusal("%s %s duplicate row id: %s" % (label, name, row["id"]))
            rows[row["id"]] = row
        digest = sha256(body)
        desc = descriptors[name]
        if (not isinstance(desc, dict) or desc.get("file") != path
                or desc.get("sha256") != digest or desc.get("record_count") != len(rows)):
            raise Refusal("%s index digest/count/file mismatch: %s" % (label, name))
        collections[name] = rows
        hashes[name] = digest
    return collections, hashes


def field_changes(old, new, path=""):
    """Exact JSON-value changes; formatting and equal numeric spellings are no-ops."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes = []
        for key in sorted(set(old) | set(new)):
            pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
            changes.extend(field_changes(old.get(key, MISSING), new.get(key, MISSING), pointer))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        changes = []
        for i in range(max(len(old), len(new))):
            changes.extend(field_changes(old[i] if i < len(old) else MISSING,
                                         new[i] if i < len(new) else MISSING, path + "/%d" % i))
        return changes
    same_kind = type(old) is type(new) or (type(old) in (int, float) and type(new) in (int, float))
    if same_kind and old == new:
        return []
    def value(item):
        return {"present": False} if item is MISSING else {"present": True, "value": item}
    return [{"path": path, "old": value(old), "new": value(new)}]


def scan_member(path, body, previous, token=None):
    findings = []
    digest = sha256(body)
    text = body.decode("utf-8", errors="replace")
    before = (previous or b"").decode("utf-8", errors="replace")
    # JSON-escaped slashes/backslashes must not conceal an absolute host path.
    text = text.replace("\\/", "/").replace("\\\\", "\\")
    before = before.replace("\\/", "/").replace("\\\\", "\\")
    historical = Counter(match.group() for match in PRIVATE_PATH.finditer(before))
    recorded = 0
    patterns = [("credential_pattern", pattern) for pattern in common._TOKEN_SHAPES]
    patterns += [("credential_pattern", KEY_PATTERN), ("private_path", PRIVATE_PATH)]
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            if kind == "private_path" and historical[match.group()] > 0:
                historical[match.group()] -= 1
                recorded += 1
                continue
            location = {"kind": kind, "line": text.count("\n", 0, match.start()) + 1,
                        "match_sha256": sha256(match.group().encode("utf-8"))}
            finding = dict(location, path=path, sha256=digest)
            finding["finding_sha256"] = sha256(common.canonical_json(location).encode("utf-8"))
            findings.append(finding)
    if token and token.encode("utf-8") in body:
        findings.append({"kind": "exact_credential", "path": path, "sha256": digest})
    return findings, recorded


def load_approval(path, expected_digest):
    if not path:
        if expected_digest:
            raise Refusal("--approval-sha256 requires --approval")
        return None
    body = Path(path).read_bytes()
    if not expected_digest or sha256(body) != expected_digest:
        raise Refusal("approval bytes differ from --approval-sha256; obtain the digest of the reviewed file")
    doc = parse_json(body)
    if not isinstance(doc, dict):
        raise Refusal("approval is not a JSON object")
    return doc


def build_plan(local, parent, remote, approval=None, token=None):
    local_rows, hashes = load_collections(local, "local")
    remote_rows, remote_hashes = load_collections(remote, "remote")
    plan = {"schema": SCHEMA, "repository": REPOSITORY, "parent_commit": parent,
            "collection_sha256": hashes, "remote_collection_sha256": remote_hashes,
            "collections": {}, "field_changes": [], "files": [], "remote_only": [],
            "retained_remote_metadata": [], "scan_findings": [], "historical_private_paths": {},
            "refusals": [], "mutation_performed": False}
    refusals = plan["refusals"]
    for name in COLLECTIONS:
        old, new = remote_rows[name], local_rows[name]
        added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
        plan["collections"][name] = {"remote_count": len(old), "local_count": len(new),
                                      "added_ids": added, "removed_ids": removed}
        if removed:
            refusals.append("public row deletion in %s: %s; recover the rows and evidence" % (name, ", ".join(removed)))
        for identity in sorted(set(old) & set(new)):
            for change in field_changes(old[identity], new[identity]):
                plan["field_changes"].append(dict(change, collection=name, id=identity))
    for path in sorted(set(remote) - set(local)):
        if path in REMOTE_METADATA:
            plan["retained_remote_metadata"].append({"path": path, "sha256": sha256(remote[path])})
        else:
            plan["remote_only"].append(path)
            refusals.append("remote-only public source/evidence omitted locally: %s; recover it verbatim" % path)
    for path, body in sorted(local.items()):
        previous = remote.get(path)
        action = "unchanged" if previous == body else ("add" if previous is None else "update")
        plan["files"].append({"path": path, "action": action, "bytes": len(body),
                              "sha256": sha256(body), "remote_sha256": sha256(previous) if previous is not None else None})
        if action == "unchanged":
            continue
        if dsformat.looks_like_a_credential(path):
            refusals.append("credential-named publication member: %s" % path)
        if previous is not None and path.startswith(("protocol/", "receipts/")):
            refusals.append("immutable public protocol/receipt bytes changed: %s; use additive evidence" % path)
        findings, recorded = scan_member(path, body, previous, token)
        plan["scan_findings"].extend(findings)
        if recorded:
            plan["historical_private_paths"][path] = recorded
    approved = False
    exceptions = []
    if approval is not None:
        review = approval.get("review")
        expected = {"schema": APPROVAL_SCHEMA, "repository": REPOSITORY,
                    "parent_commit": parent, "collection_sha256": hashes,
                    "field_changes": plan["field_changes"]}
        approved = all(common.canonical_json(approval.get(k)) == common.canonical_json(v)
                       for k, v in expected.items())
        approved = approved and isinstance(review, dict) and all(
            isinstance(review.get(k), str) and review[k].strip() for k in ("reviewer", "reason"))
        allowed = set(expected) | {"review", "scan_exceptions", "warning_disposition"}
        approved = approved and not (set(approval) - allowed)
        exceptions = approval.get("scan_exceptions", [])
        if not isinstance(exceptions, list):
            approved = False
            exceptions = []
        if not approved:
            refusals.append("approval does not exactly bind this parent, local collections, field delta and review")
    if plan["field_changes"] and not approved:
        refusals.append("existing public row fields changed: exact reviewed approval required (no numeric tolerance)")
    unused = list(exceptions) if approved else []
    for finding in plan["scan_findings"]:
        matched = None
        for exception in unused:
            if (isinstance(exception, dict) and set(exception) == {"path", "sha256", "finding_sha256", "reason"}
                    and all(exception.get(k) == finding.get(k) for k in ("path", "sha256", "finding_sha256"))
                    and isinstance(exception.get("reason"), str) and exception["reason"].strip()
                    and finding["kind"] != "exact_credential"):
                matched = exception
                break
        if matched is not None:
            unused.remove(matched)
            finding["reviewed_exception"] = matched["reason"]
        else:
            refusals.append("unapproved %s finding in %s%s" % (finding["kind"], finding["path"],
                            ":%s" % finding["line"] if "line" in finding else ""))
    if unused:
        refusals.append("approval contains stale/unmatched scan exceptions")
    plan["approval_valid"] = approved
    return plan


def prerequisites(root, python):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env.pop(name, None)
    # Call the canonical strict validator directly to retain its exact rc=2
    # and structured warning findings. make check-release remains unchanged.
    commands = [("validate-strict", [python, "registry/tools/registry_validate.py",
                                    "--strict", "--json", "--jsonschema-lib", "mini"])]
    commands += [(target, ["make", "-C", "registry", "PY=" + python, target])
                 for target in ("render-check", "joint", "reseed-check")]
    results = []
    for target, command in commands:
        result = subprocess.run(command, cwd=str(root), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        item = {"target": target, "interpreter": python, "returncode": result.returncode}
        if target == "validate-strict":
            try:
                item["report"] = parse_json(result.stdout)
            except (ValueError, Refusal):
                print(common.redact(result.stdout), file=sys.stderr, end="")
        else:
            print(common.redact(result.stdout), file=sys.stderr, end="")
        print(common.redact(result.stderr), file=sys.stderr, end="")
        results.append(item)
    return results


def check_prerequisites(plan, local, approval, checks, root):
    plan["prerequisites"] = checks
    plan["publication_kind"] = "reviewed_snapshot_not_release_certification"
    plan["strict_warning_free_release_gate_passed"] = False
    plan["warnings_dispositioned"] = False
    for check in checks:
        if check["target"] != "validate-strict":
            if check["returncode"]:
                plan["refusals"].append("canonical prerequisite failed: %s; fix sources or select --python with NumPy"
                                        % check["target"])
            continue
        report = check.get("report", {})
        warnings = report.get("warnings")
        findings = report.get("findings", [])
        if (report.get("errors") != 0 or type(warnings) is not int
                or not isinstance(findings, list)
                or any(f.get("severity") != "warn" for f in findings)
                or len(findings) != warnings
                or check["returncode"] != (2 if warnings else 0)):
            plan["refusals"].append("strict validator failed or did not return a consistent zero-error report")
            continue
        plan["strict_warning_free_validator_passed"] = warnings == 0
        if not warnings:
            continue
        disposition = (approval or {}).get("warning_disposition")
        body = local.get("publication-audit.json")
        if (not plan["approval_valid"] or not isinstance(disposition, dict)
                or set(disposition) != {"audit_sha256", "warning_count"} or body is None
                or disposition["audit_sha256"] != sha256(body)
                or type(disposition["warning_count"]) is not int
                or disposition["warning_count"] != warnings):
            plan["refusals"].append("strict validator reports %d warnings; an exact reviewed audit/hash/count disposition is required"
                                    % warnings)
            continue
        audit = parse_json(body)
        records = audit.get("dispositions", [])
        keys = ("check", "severity", "id", "message", "remedy")
        def multiset(items):
            return Counter(common.canonical_json({k: item.get(k) for k in keys}) for item in items)
        valid = isinstance(records, list) and all(
            isinstance(item, dict) and item.get("status") and item.get("rationale")
            for item in records)
        valid = valid and multiset(records) == multiset(findings)
        counts = {name: values["local_count"] for name, values in plan["collections"].items()}
        valid = valid and audit.get("record_counts") == counts and report.get("counts") == counts
        validator = audit.get("validator", {})
        valid = valid and validator.get("errors") == 0 and validator.get("warnings") == warnings
        hashes = audit.get("data_sha256", {})
        required = {"data/%s.jsonl" % name for name in COLLECTIONS} | {"index.json"}
        valid = valid and isinstance(hashes, dict) and required <= set(hashes)
        for path, digest in hashes.items():
            safe_path(path)
            valid = valid and path in local and sha256(local[path]) == digest
        for path, digest in audit.get("checkout_sha256", {}).items():
            safe_path(path)
            source = root / path
            valid = valid and source.is_file() and common.sha256_file(str(source)) == digest
        if not valid:
            plan["refusals"].append("audit dispositions/findings or snapshot hashes/counts drifted; re-review, never regenerate silently")
            continue
        plan["warnings_dispositioned"] = True
        plan["warning_disposition"] = disposition


def publish(plan, local, remote, hub, token, message):
    if plan["refusals"]:
        raise Refusal("publication plan has refusals; nothing uploaded")
    observed, _ = hub.inventory()
    if observed != plan["parent_commit"]:
        raise Refusal("remote HEAD changed since the plan; re-resolve and review, never force")
    changed = {path: body for path, body in local.items() if remote.get(path) != body}
    if not changed:
        plan["status"] = "already_converged"
        return
    # A transport failure after the server accepted the commit is ambiguous.
    # Never report "no mutation" merely because its response did not arrive.
    plan["mutation_performed"] = None
    plan["status"] = "commit_outcome_unknown"
    revision = hub.commit(plan["parent_commit"], changed, token, message)
    plan["mutation_performed"] = True
    plan["committed_revision"] = revision
    plan["status"] = "committed_unverified"
    try:
        expected = dict(remote)
        expected.update(local)
        entries = hub.verify(revision, expected)
        for path, body in sorted(changed.items()):
            if hub.download(revision, path, entries[path]) != body:
                raise Refusal("public download differs from approved bytes: %s" % path)
    except Exception as exc:
        raise Refusal("commit %s exists but exact-byte verification FAILED; inspect this revision before retrying: %s"
                      % (revision, common.redact(str(exc))))
    plan["status"] = "published_verified"
    plan["verified_file_count"] = len(expected)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="publish one guarded parent-CAS commit; requires --token-file")
    mode.add_argument("--dry-run", action="store_true", help="explicit default: anonymous reads and read-only checks; no token read or upload")
    parser.add_argument("--token-file", help="explicit owned 0600 HF credential file, execute only; never ambient credentials")
    parser.add_argument("--python", default=os.environ.get("FIDELITY_PYTHON", sys.executable),
                        help="interpreter for canonical checks, including NumPy reseed (default FIDELITY_PYTHON or running Python)")
    parser.add_argument("--expected-parent", help="refuse unless the freshly resolved public HEAD equals this 40-hex SHA")
    parser.add_argument("--approval", help="reviewed machine-readable exact delta/scan approval JSON")
    parser.add_argument("--approval-sha256", help="SHA256 of the reviewed approval bytes (required with --approval)")
    parser.add_argument("--plan", help="also atomically save the complete JSON plan outside this checkout (stdout always contains JSON)")
    parser.add_argument("--message", default="Publish reviewed receipt-backed registry convergence", help="HF commit message")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    plan = {"schema": SCHEMA, "repository": REPOSITORY, "mutation_performed": False, "refusals": []}
    output_path = None
    try:
        if args.plan:
            output_path = Path(args.plan).resolve()
            if root == output_path or root in output_path.parents or any(
                    output_path == Path(p).expanduser().resolve() for p in (args.token_file, args.approval) if p):
                output_path = None
                raise Refusal("--plan must be outside the checkout and distinct from credential/approval files")
        if args.execute and not args.token_file:
            raise Refusal("--execute requires --token-file pointing to an owned 0600 file; ambient tokens are never used")
        if args.expected_parent and not SHA.fullmatch(args.expected_parent):
            raise Refusal("--expected-parent must be an exact lowercase 40-hex SHA")
        approval = load_approval(args.approval, args.approval_sha256)
        # Read the explicit credential before any subprocess, so exact secret
        # redaction is armed. Dry-run never opens even a supplied token file.
        token = dshub.read_token(str(Path(args.token_file).expanduser())) if args.execute else None
        local = authored_snapshot(root)
        checks = prerequisites(root, args.python)
        hub = PublicHub()
        parent, remote = hub.snapshot(local)
        plan = build_plan(local, parent, remote, approval, token)
        check_prerequisites(plan, local, approval, checks, root)
        plan["mode"] = "execute" if args.execute else "dry-run"
        if args.approval:
            plan["approval_sha256"] = args.approval_sha256
        if args.expected_parent and parent != args.expected_parent:
            plan["refusals"].append("fresh remote HEAD differs from --expected-parent")
        # Checks do not generate files. Recheck the captured source identity
        # after network reads to refuse concurrent edits rather than mixing them.
        if authored_snapshot(root) != local:
            plan["refusals"].append("authored registry tree changed during planning; retry from a stable reviewed tree")
        if args.execute and not plan["refusals"]:
            publish(plan, local, remote, hub, token, args.message)
    except Exception as exc:
        plan["refusals"].append(common.redact(str(exc)))
    except KeyboardInterrupt:
        plan["refusals"].append("interrupted; if publication began, inspect remote HEAD before retrying")
    plan.setdefault("status", "refused" if plan["refusals"] else "ready_dry_run")
    safe_plan = parse_json(common.redact(json.dumps(plan, sort_keys=True, ensure_ascii=False, allow_nan=False)))
    if output_path is not None:
        common.write_json(str(output_path), safe_plan)
    print(json.dumps(safe_plan, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 2 if plan["refusals"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
