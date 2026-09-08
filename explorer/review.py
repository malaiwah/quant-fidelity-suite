"""Authenticated public discussions and explicit, HEAD-bound registry acceptance.

Only bundled validation code runs. Public repository files are untrusted data.
Approval tickets are opaque, process-local, short-lived and never contain credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import hmac
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .auth import AuthError, require_registry_owner
from .contribute import _parse

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_REPOSITORY = os.environ.get("QFS_REGISTRY_REPOSITORY", "malaiwah/quant-fidelity-registry")
SCHEMA = "qfs.registry-review-request.v1"
MAX_FILE = 1024 * 1024
MAX_SNAPSHOT = 256 * 1024 * 1024
MAX_FILES = 4096
TTL = 900
_PREFIX = "QFS registry review request\n\n```json\n"
_SUFFIX = "\n```"
_LOCK = threading.Lock()
_TICKETS = {}
_SLOTS = threading.BoundedSemaphore(2)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _path(value):
    _require(isinstance(value, str) and len(value) <= 240 and "\\" not in value, "Invalid evidence path.")
    parts = PurePosixPath(value).parts
    _require(parts and not value.startswith("/") and all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in (".", "..") for p in parts)
             and "/".join(parts) == value, "Evidence paths must be safe repository-relative paths.")
    return value


def _identity(actor, *, owner=False):
    actor.require_lifetime(120)
    api = actor.client()
    identity = api.whoami()
    if identity.get("name") != actor.username or identity.get("type") == "org":
        raise AuthError("The acting token must identify this personal Hugging Face user.")
    if owner:
        require_registry_owner(actor, REGISTRY_REPOSITORY)
    return api


def _publication(value):
    value = _parse(_canonical(value).decode())
    _require(set(value) <= {"kind", "repository", "revision", "files", "metadata", "redistribution_consent", "attestation"}, "Unknown publication fields; pass only the bounded review descriptor, never private result state.")
    _require(value.get("kind") in ("root", "measurement"), "Request kind must be root or measurement.")
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value.get("repository", "")), "Supply a public HF dataset repository.")
    _require(re.fullmatch(r"[0-9a-f]{40}", value.get("revision", "")), "Publication must pin a complete immutable HF commit.")
    _require(value.get("redistribution_consent") is True, "Explicit publication and redistribution consent is required before posting evidence.")
    files = value.get("files")
    required = {"result", "plan", "execution", "submission", "comparison"} if value["kind"] == "measurement" else {"result", "plan", "execution", "job", "dataset", "runtime", "capture", "qualification", "comparison", "config", "panel", "panel_receipt", "harness"}
    _require(isinstance(files, dict) and required <= files.keys() and len(files) <= 24, "Publication is missing required immutable evidence files: " + ", ".join(sorted(required)))
    for key, f in files.items():
        _require(re.fullmatch(r"[a-z][a-z_]{0,31}", key) and isinstance(f, dict) and set(f) == {"path", "sha256"}, "Each evidence pointer must contain exactly path and sha256.")
        _path(f["path"])
        _require(f["path"].endswith(".json") and re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), "Review accepts hashed JSON evidence only, not uploaded programs or archives.")
    _require(len({f["path"] for f in files.values()}) == len(files), "Evidence roles must use distinct files.")
    return value

def _anonymous():
    """token=False: no Authorization header, no ambient env/cache credential.
    Required-public evidence and the registry are read anonymously ONLY -- the
    anonymous read is the proof the bytes are public; a 401/403 is itself the
    disclosure that they are not, so it is a refusal, never an escalation."""
    from huggingface_hub import HfApi
    return HfApi(endpoint="https://huggingface.co", token=False)


def _public(repository, revision):
    try:
        info = _anonymous().repo_info(repository, repo_type="dataset", revision=revision)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            _require(False, "Evidence and registry must be ungated public datasets; %s refused an anonymous read, so it is not public. Publish with explicit consent first." % repository)
        raise
    _require(info.private is False and not getattr(info, "gated", False), "Evidence and registry must be ungated public datasets; publish with explicit consent first.")
    _require(info.sha == revision, "The requested immutable public commit did not resolve exactly.")


def _download(api, repo, revision, path, size, directory):
    _require(isinstance(size, int) and 0 <= size <= MAX_SNAPSHOT, "Invalid or oversized repository file.")
    cached = api.hf_hub_download(repo, path, repo_type="dataset", revision=revision, cache_dir=str(directory / "cache"))
    p = Path(cached)
    _require(p.stat().st_size == size, "Downloaded file size differs from the pinned Hub tree.")
    return p.read_bytes()


def _signing_key():
    key = os.environ.get("QFS_REVIEW_SIGNING_KEY", "")
    return key.encode() if len(key) >= 32 else None


def _attestation_verified(publication, author):
    key = _signing_key()
    att = publication.get("attestation")
    if key is None or not isinstance(att, dict):
        return False
    if set(att) != {"schema", "actor", "publication_sha256", "verified_at", "signature"}:
        return False
    payload = {k: v for k, v in att.items() if k != "signature"}
    unsigned = {k: v for k, v in publication.items() if k != "attestation"}
    return (att["schema"] == "qfs.review-provider-attestation.v1" and att["actor"] == author
            and att["publication_sha256"] == _sha(_canonical(unsigned))
            and isinstance(att["signature"], str)
            and hmac.compare_digest(att["signature"], hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()))


def attest_publication(actor, publication):
    """Internal controller helper, NEVER expose as a UI/API endpoint.

    Caller must first validate all recovered result bytes and scientific receipts.
    This certifies service provider readback, not independent model reproduction.
    """
    api = _identity(actor)
    pub = _publication({k: v for k, v in publication.items() if k != "attestation"})
    key = _signing_key()
    _require(key is not None, "Configure a stable QFS_REVIEW_SIGNING_KEY of at least 32 characters before issuing service attestations.")
    with tempfile.TemporaryDirectory(prefix="qfs-attest-") as td:
        evidence = _evidence(api, pub, Path(td))
        execution = _parse(evidence["execution"].decode())
        result = _parse(evidence["result"].decode())
        plan = _parse(evidence["plan"].decode())
        job = api.inspect_job(namespace=actor.username, job_id=execution["job_id"])
        from .jobs import _verify_provider, _verify_result_log
        _verify_provider(actor, job, plan)
        _verify_result_log(actor, execution["job_id"], result)
        _require(execution.get("namespace") == actor.username == result.get("owner") == plan.get("owner")
                 and result.get("status") == "complete" and job.id == execution["job_id"]
                 and job.status.stage == execution.get("status") == "COMPLETED"
                 and job.docker_image == execution.get("docker_image") == plan.get("image")
                 and job.flavor == execution.get("flavor") == plan["hardware"]["flavor"],
                 "Authenticated provider readback does not match the published completed Job.")
    payload = {"schema": "qfs.review-provider-attestation.v1", "actor": actor.username,
               "publication_sha256": _sha(_canonical(pub)), "verified_at": int(time.time())}
    pub["attestation"] = dict(payload, signature=hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest())
    return pub


def _evidence(api, publication, directory, *, require_public=True):
    repo, revision = publication["repository"], publication["revision"]
    # Required-public evidence is fetched anonymously only; the caller's api is
    # used solely for the caller's own private publication re-checks.
    reader = api if not require_public else _anonymous()
    if require_public:
        _public(repo, revision)
    files = publication["files"]
    infos = {f.path: f for f in reader.get_paths_info(repo, [f["path"] for f in files.values()], repo_type="dataset", revision=revision)}
    raw = {}
    for name, f in files.items():
        info = infos.get(f["path"])
        limit = MAX_FILE if require_public else 16 * 1024 * 1024
        _require(info is not None and isinstance(getattr(info, "size", None), int) and info.size <= limit, "Evidence missing or exceeds the bounded JSON review limit: " + f["path"])
        b = _download(reader, repo, revision, f["path"], info.size, directory)
        _require(_sha(b) == f["sha256"], "Published evidence hash mismatch: " + name)
        _parse(b.decode("utf-8")) if require_public else json.loads(b)
        raw[name] = b
    if require_public and publication["kind"] == "root":
        meta = publication.get("metadata") or {}
        root_repo, root_revision = meta.get("root_repository"), meta.get("root_revision")
        _require(isinstance(root_repo, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", root_repo)
                 and isinstance(root_revision, str) and re.fullmatch(r"[0-9a-f]{40}", root_revision),
                 "Root metadata must pin the canonical public capture repository and immutable revision.")
        _public(root_repo, root_revision)
        descriptor = reader.get_paths_info(root_repo, ["fidelity-dataset.json"], repo_type="dataset", revision=root_revision)
        _require(len(descriptor) == 1 and getattr(descriptor[0], "size", MAX_FILE + 1) <= MAX_FILE,
                 "Canonical public root descriptor is missing or exceeds the review limit.")
        canonical_raw = _download(reader, root_repo, root_revision, "fidelity-dataset.json", descriptor[0].size, directory)
        _require(canonical_raw == raw["dataset"], "Evidence bundle descriptor differs from the actual immutable public root descriptor.")
    return raw


def _summary(discussion):
    return {"discussion_id": discussion.num, "author": discussion.author, "status": discussion.status,
            "title": discussion.title, "url": "https://huggingface.co/datasets/" + REGISTRY_REPOSITORY + "/discussions/" + str(discussion.num)}


def _request(api, discussion_id):
    _require(type(discussion_id) is int and discussion_id > 0, "Select a positive discussion number.")
    d = api.get_discussion_details(REGISTRY_REPOSITORY, discussion_id, repo_type="dataset")
    _require(not d.is_pull_request and d.status == "open", "Only an open registry review discussion can be accepted.")
    _require(len(d.events) <= 200, "Discussion exceeds the bounded review event limit.")
    comments = [e for e in d.events if getattr(e, "type", None) == "comment"]
    _require(comments and comments[0].author == d.author, "The original request must be authored by the discussion creator.")
    body = comments[0].content
    _require(isinstance(body, str) and len(body) <= 65536 and body.startswith(_PREFIX) and body.endswith(_SUFFIX), "Discussion does not contain the typed QFS request envelope.")
    envelope = _parse(body[len(_PREFIX):-len(_SUFFIX)])
    _require(set(envelope) == {"schema", "requested_by", "publication"} and envelope.get("schema") == SCHEMA and envelope.get("requested_by") == d.author, "Request envelope authorship/schema is invalid.")
    envelope["publication"] = _publication(envelope["publication"])
    _require(envelope["publication"]["repository"].split("/")[0] == d.author, "A request must point to its author's public result dataset.")
    return d, envelope, _sha(_canonical(envelope))


def list_requests(actor):
    api = _identity(actor)
    result = []
    for n, d in enumerate(api.get_repo_discussions(REGISTRY_REPOSITORY, repo_type="dataset", discussion_status="open")):
        if n >= 200:
            break
        if not d.is_pull_request and d.title.startswith("QFS review:"):
            result.append(_summary(d))
    return {"registry_repository": REGISTRY_REPOSITORY, "requests": result, "limit": 200,
            "notice": "Discussion titles are untrusted; inspect a request before accepting."}


def request_review(actor, publication, *, confirm_public):
    _require(confirm_public is True, "Confirm public posting and redistribution before requesting review.")
    api = _identity(actor)
    pub = _publication(publication)
    _require(pub["repository"].split("/")[0] == actor.username, "Publish in the authenticated caller's personal namespace before requesting review.")
    with tempfile.TemporaryDirectory(prefix="qfs-request-") as td:
        evidence = _evidence(api, pub, Path(td))
        result = _parse(evidence["result"].decode())
        _require(result.get("owner") == actor.username and result.get("status") == "complete", "Only the producing caller's complete public Job result can request review.")
    envelope = {"schema": SCHEMA, "requested_by": actor.username, "publication": pub}
    body = _PREFIX + _canonical(envelope).decode() + _SUFFIX
    _require(len(body.encode()) <= 65536, "The public request envelope exceeds 64 KiB.")
    head = _anonymous().repo_info(REGISTRY_REPOSITORY, repo_type="dataset").sha
    _public(REGISTRY_REPOSITORY, head)
    unsigned = {k: v for k, v in pub.items() if k != "attestation"}
    incoming_verified = _attestation_verified(pub, actor.username)
    superseded = []
    for n, existing in enumerate(api.get_repo_discussions(REGISTRY_REPOSITORY, repo_type="dataset", discussion_status="open")):
        _require(n < 500, "Review recovery scan reached its limit; inspect existing requests before posting another.")
        if existing.author != actor.username or existing.is_pull_request or not existing.title.startswith("QFS review:"):
            continue
        try:
            original, prior, digest = _request(api, existing.num)
        except ValueError:
            continue
        if {k: v for k, v in prior["publication"].items() if k != "attestation"} == unsigned:
            if not incoming_verified or _attestation_verified(prior["publication"], actor.username):
                return dict(_summary(original), request_sha256=digest)
            superseded.append(existing.num)
    d = api.create_discussion(REGISTRY_REPOSITORY, "QFS review: " + pub["kind"] + " " + pub["repository"], description=body, repo_type="dataset")
    for number in superseded:
        api.change_discussion_status(REGISTRY_REPOSITORY, number, "closed", repo_type="dataset",
                                     comment="Superseded by canonically revalidated request #%s; original evidence remains linked in the history." % d.num)
    return dict(_summary(d), request_sha256=_sha(_canonical(envelope)))


def _expire():
    for key, state in list(_TICKETS.items()):
        if state["expires_at"] <= time.time() and not state.get("accepting"):
            shutil.rmtree(state["directory"], ignore_errors=True)
            del _TICKETS[key]


def _snapshot(head, destination, directory):
    """The public registry snapshot is read anonymously: tokenless bytes are the proof."""
    reader = _anonymous()
    count = total = 0
    original = {}
    for f in reader.list_repo_tree(REGISTRY_REPOSITORY, repo_type="dataset", revision=head, recursive=True):
        if not hasattr(f, "size"):
            continue
        path = _path(f.path)
        count += 1
        total += f.size
        _require(count <= MAX_FILES and total <= MAX_SNAPSHOT, "Registry snapshot exceeds review limits; use maintainer offline intake.")
        raw = _download(reader, REGISTRY_REPOSITORY, head, path, f.size, directory)
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        original[path] = _sha(raw)
    return original


def inspect_request(actor, discussion_id):
    api = _identity(actor, owner=True)
    d, envelope, digest = _request(api, discussion_id)
    _require(_SLOTS.acquire(blocking=False), "Review workers are busy; try again shortly.")
    directory = Path(tempfile.mkdtemp(prefix="qfs-review-"))
    keep = False
    try:
        head = _anonymous().repo_info(REGISTRY_REPOSITORY, repo_type="dataset").sha
        _public(REGISTRY_REPOSITORY, head)
        stage = directory / "registry"
        stage.mkdir()
        original = _snapshot(head, stage, directory)
        evidence = _evidence(api, envelope["publication"], directory)
        inputs = directory / "inputs"
        inputs.mkdir()
        for key, raw in evidence.items():
            (inputs / (key + ".json")).write_bytes(raw)
        context = {"envelope": envelope, "request_sha256": digest, "registry_head": head,
                   "registry_repository": REGISTRY_REPOSITORY, "discussion_id": discussion_id, "reviewed_by": actor.username,
                   "provider_metadata_verified": _attestation_verified(envelope["publication"], envelope["requested_by"])}
        (directory / "context.json").write_bytes(_canonical(context))
        command = [sys.executable, "-I", "-B", str(ROOT / "registry/tools/review_requests.py"), "stage", str(directory)]
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "HOME", "SYSTEMROOT"}}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        run = subprocess.run(command, cwd=directory, capture_output=True, timeout=120, env=env, check=False)
        _require(run.returncode == 0, "Registry review refused (exit %s): " % run.returncode
                 + (run.stdout + run.stderr).decode("utf-8", errors="replace")[-12000:])
        preview = _parse((directory / "preview.json").read_text())
        changes = {}
        for p in stage.rglob("*"):
            if p.is_file():
                path = p.relative_to(stage).as_posix()
                if _sha(p.read_bytes()) != original.get(path):
                    changes[path] = _sha(p.read_bytes())
        _require(all((stage / p).is_file() for p in original), "Staged intake would delete existing repository files.")
        _require(changes and len(changes) <= 200, "Acceptance must produce a bounded nonempty change set.")
        with _LOCK:
            _expire()
            _require(len(_TICKETS) < 16, "Too many outstanding approval tickets; wait for expiration.")
            ticket = secrets.token_urlsafe(32)
            expires = time.time() + TTL
            _TICKETS[ticket] = {"actor": actor.username, "discussion_id": discussion_id, "digest": digest, "head": head,
                                "expires_at": expires, "directory": str(directory), "changes": changes, "preview": preview}
            keep = True
        return dict(_summary(d), request_sha256=digest, registry_head=head, preview=preview,
                    approval_ticket=ticket, expires_at=expires, explicit_acceptance_required=True)
    finally:
        _SLOTS.release()
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)


def accept_request(actor, ticket, *, confirm_accept):
    _require(confirm_accept is True, "Explicit owner acceptance is required; inspection alone never commits.")
    api = _identity(actor, owner=True)
    _require(isinstance(ticket, str) and 32 <= len(ticket) <= 128, "Invalid approval ticket; inspect the request again.")
    with _LOCK:
        _expire()
        state = _TICKETS.get(ticket)
        _require(state is not None and state["actor"] == actor.username, "Approval ticket expired or belongs to another actor; inspect again.")
        _require(not state.get("accepting"), "This approval is already being committed.")
        state["accepting"] = True
    try:
        _, _, digest = _request(api, state["discussion_id"])
        _require(digest == state["digest"], "The request changed after inspection; inspect it again.")
        head = api.repo_info(REGISTRY_REPOSITORY, repo_type="dataset").sha
        _require(head == state["head"], "Registry HEAD changed after inspection; inspect against the new HEAD.")
        _public(api, REGISTRY_REPOSITORY, head)
        from huggingface_hub import CommitOperationAdd
        stage = Path(state["directory"]) / "registry"
        operations = []
        for path, sha in sorted(state["changes"].items()):
            raw = (stage / path).read_bytes()
            _require(_sha(raw) == sha, "Staged approval content changed; inspect again.")
            operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=raw))
        commit = api.create_commit(REGISTRY_REPOSITORY, repo_type="dataset", operations=operations, parent_commit=head,
                                   commit_message="Accept QFS review #%d (%s)" % (state["discussion_id"], digest[:12]))
        result = {"registry_repository": REGISTRY_REPOSITORY, "revision": commit.oid, "commit_url": commit.commit_url,
                "request_sha256": digest, "discussion_id": state["discussion_id"], "warnings": state["preview"]["warnings"],
                "acceptance_receipt": "https://huggingface.co/datasets/" + REGISTRY_REPOSITORY + "/resolve/" + commit.oid + "/protocol/review-requests/" + digest + "/acceptance.json",
                "independently_verified": False}
        try:
            api.change_discussion_status(
                REGISTRY_REPOSITORY, state["discussion_id"], "closed", repo_type="dataset",
                comment="Accepted by the registry owner at %s. Receipt integrity was validated; this is not independent model reproduction." % commit.commit_url)
        except Exception as exc:
            # The data commit already succeeded; never report a committed claim as rejected.
            result["discussion_update_warning"] = "Registry commit succeeded; discussion closure needs reconciliation (%s)." % type(exc).__name__
        return result
    finally:
        with _LOCK:
            _TICKETS.pop(ticket, None)
        shutil.rmtree(state["directory"], ignore_errors=True)
