#!/usr/bin/env python3
"""Transport the sealed malaiwah Qwen3.8 suite-v5 shard-0 token panel into a
``quant-pipeline.glm53-token-panel.v1`` tree that ``bin/fidelity/panel.py``
resolves and ``engines/tools/hf_capture.py`` consumes.

Why this exists
---------------
The existing Qwen3.8-27B registry rows were scored on
``panel--qwen38.malaiwah.suite-v5-shard0-1m``.  That panel's registry record
carries ``availability.status = "private"`` and ``uri = null``, and its only
recorded source is a receipt file on the author's laptop -- so on its face the
panel looked unreusable and a fresh panel looked mandatory.

It is in fact fully recoverable and byte-verifiable.  The public dataset
``malaiwah/qwen38-27b-fidelity-suite-v5`` carries every context's token ids
under ``suite/tokens/context-NNNN.json`` and the shard's sealed manifest under
``suite/shard-0000/suite-manifest.json``.  This tool reads (or, with
``--fetch``, downloads) the shard's 512 contexts at a pinned revision and
verifies four independent seals plus the tokenizer identity before it writes
anything:

  1. each context file's sha256 equals the ``token_sha256`` recorded for it in
     the sealed suite manifest;
  2. the sha256 of those 512 digests concatenated in ``context_index`` order --
     the suite's own aggregate rule -- equals the sealed
     ``suite_token_sha256`` AND the registry panel's
     ``identity.panel_token_sha256`` passed as ``--expect-panel-token-sha256``;
  3. optionally the legacy panel aggregate (empty-string join of the per-record
     compact-JSON token digests, ``fidelity.dsformat`` preimage 5 legacy), the
     ``panel_token_sha256_legacy`` the historical capture receipts carry; and
  4. optionally the normative canonical aggregate (newline join of the same
     per-record digests), which is what a resolved panel binding recomputes.

The tokenizer is pinned separately and proved against the suite: the manifest's
``model_identity.tokenizer_sha256`` and ``config_sha256`` must equal the sha256
of ``tokenizer.json`` and ``config.json`` fetched at the pinned MODEL revision,
and that ``config.json``'s ``vocab_size`` must equal the manifest's.  That is
what makes ``Qwen/Qwen3.8-27B@<revision>`` the tokenizer of THIS suite rather
than a plausible-looking assertion.  The emitted ``tokenizer.receipt.json``
(``quant-pipeline.glm53-tokenizer-receipt.v1``) is bound into the panel receipt
by seal, so a later rebuild or a paid admission verifies the tokenizer bytes
against the panel instead of trusting a repository name.

What reusing this panel does and does not buy
---------------------------------------------
It fixes the token content, the masks, the scored positions and the document
provenance: a new row and an old row are then measured on the SAME text.  It
does not make them rankable and it does not isolate a lane: the comparability
key binds the reference, the references differ, and lane, engine, pipeline,
precision policy and hardware differ too.  Equal tokens are a precondition for
an interpretable contrast, not a licence to attribute the difference to any one
of those factors.

No RNG.  No tokenizer is ever loaded and no text is ever re-tokenized: the
token ids are transported, and the tokenizer files are identity evidence only.
``--fetch`` reads public metadata (a manifest, token-id JSON, tokenizer/config
JSON) over HTTPS with no credential of any kind and never touches weights;
without it the tool is offline and refuses on the first missing input, naming
the URL to mirror.  The receipt carries no wall-clock stamp so that two runs of
the same pinned inputs are byte-identical.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REVISION40 = re.compile(r"^[0-9a-f]{40}$")
HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
CONTEXT_FILE = re.compile(r"^context-[0-9]{4,}\.json$")

PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"
BUILD_RECEIPT_SCHEMA = "malaiwah.token-panel-build-receipt.v1"
TOKENIZER_RECEIPT_SCHEMA = "quant-pipeline.glm53-tokenizer-receipt.v1"
PROVENANCE_SCHEMA = "malaiwah.transported-token-panel.v1"

#: Bound tokenizer identity: the files that determine a token id. A resolved
#: binding verifies every one of them against the target's own directory
#: before any spend, so this set is the panel's tokenizer contract.
DEFAULT_BOUND_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
#: Additionally fetched as evidence: config.json carries the suite-sealed
#: architecture digest and the vocabulary width; the rest is provenance.
DEFAULT_EVIDENCE_TOKENIZER_FILES = (
    "config.json", "generation_config.json", "chat_template.jinja")

SELECTION_RULE = (
    "every context of the sealed shard, in the shard suite-manifest's "
    "context_index order; window N carries context_index[N]'s token ids "
    "verbatim, its stratum as the window domain and its source cluster as the "
    "window document_id. Masks are all-ones over the full context, so every "
    "causal next-token position is scored. No RNG, no tokenizer, no "
    "re-derivation from text.")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    handle = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            handle.update(chunk)
    return handle.hexdigest()


def canonical(value, newline=False):
    """The repository's canonical JSON preimage (``fidelity.panel._canonical``)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return (text + ("\n" if newline else "")).encode("utf-8")


def seal(doc):
    """The modern self-blanked seal every resolver-visible receipt uses."""
    body = dict(doc)
    body["receipt_sha256"] = ""
    return sha256_bytes(canonical(body))


def seal_conventions(doc):
    """Every seal convention ``fidelity.panel`` or its history accepts.

    Returns ``(declared, convention)`` for the first that reproduces the
    document's own ``receipt_sha256``, else ``(declared, None)``.
    """
    declared = doc.get("receipt_sha256")
    if not isinstance(declared, str) or not HEX64.match(declared):
        return declared, None
    blanked = dict(doc)
    blanked["receipt_sha256"] = ""
    absent = dict(doc)
    del absent["receipt_sha256"]
    candidates = (
        ("self-blank", sha256_bytes(canonical(blanked))),
        ("legacy-field-absent", sha256_bytes(canonical(absent, newline=True))),
        ("legacy-field-absent-ascii",
         sha256_bytes(json.dumps(absent, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))),
    )
    for name, value in candidates:
        if value == declared:
            return declared, name
    return declared, None


def token_ids_json_sha256(ids):
    """Spec 5.1 per-record preimage: compact separators."""
    return sha256_bytes(json.dumps([int(v) for v in ids],
                                   separators=(",", ":")).encode("utf-8"))


def suite_token_hash_sha256(per_record_hex):
    """Spec 5.1 aggregate preimage (NORMATIVE): newline join, ascending order."""
    return sha256_bytes("\n".join(per_record_hex).encode("ascii"))


def suite_token_hash_sha256_legacy(per_record_hex):
    """The historical aggregate: empty-string join. Never normative."""
    return sha256_bytes("".join(per_record_hex).encode("utf-8"))


def die(msg):
    sys.stderr.write("build_panel_from_v5_suite: %s\n" % msg)
    raise SystemExit(4)


def write_atomic(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    # O_CREAT with 0666 applies the process umask, so a receipt lands with the
    # same mode as the numpy arrays beside it instead of tempfile's 0600.
    temporary = os.path.join(directory, ".tmp-%d-%s"
                             % (os.getpid(), os.path.basename(path)))
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(data)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json_atomic(path, doc, sort_keys):
    body = json.dumps(doc, indent=2, sort_keys=sort_keys,
                      ensure_ascii=False, allow_nan=False) + "\n"
    write_atomic(path, body.encode("utf-8"))


def hex64(value, label):
    if not isinstance(value, str) or not HEX64.match(value):
        die("%s must be a lowercase 64-hex sha256 (got %r)" % (label, value))
    return value


def revision40(value, label):
    if not isinstance(value, str) or not REVISION40.match(value):
        die("%s must be an immutable lowercase 40-hex commit, not a branch "
            "name (got %r)" % (label, value))
    return value


def repository(value, label):
    if not isinstance(value, str) or not HF_REPO.match(value):
        die("%s must have owner/name form (got %r)" % (label, value))
    return value


def resolve_url(endpoint, repo_type, repo, revision, path):
    prefix = "" if repo_type == "model" else "%ss/" % repo_type
    return "%s/%s%s/resolve/%s/%s" % (endpoint.rstrip("/"), prefix, repo,
                                      revision, path.lstrip("/"))


def http_get(url, timeout):
    """Fetch public bytes. No credential is read, sent, or logged."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "quant-fidelity-suite/build_panel_from_v5_suite",
                      "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        die("GET %s failed with HTTP %s; this tool fetches PUBLIC files only "
            "and sends no token" % (url, exc.code))
    except urllib.error.URLError as exc:
        die("GET %s failed: %s" % (url, exc.reason))


class Source(object):
    """A pinned public tree mirrored into a local cache directory."""

    def __init__(self, endpoint, repo_type, repo, revision, cache, fetch, timeout):
        self.endpoint = endpoint
        self.repo_type = repo_type
        self.repo = repo
        self.revision = revision
        self.cache = cache
        self.fetch = fetch
        self.timeout = timeout
        self.fetched = 0
        self.reused = 0

    def url(self, remote_path):
        return resolve_url(self.endpoint, self.repo_type, self.repo,
                           self.revision, remote_path)

    def get(self, remote_path, local_path, expect_sha256=None, label=None):
        """Bytes of one pinned file, cached locally and digest-checked."""
        name = label or remote_path
        if os.path.isfile(local_path):
            with open(local_path, "rb") as handle:
                raw = handle.read()
            self.reused += 1
        elif not self.fetch:
            die("missing local input %s (%s). Re-run with --fetch, or mirror "
                "%s into it. This tool never fetches without --fetch."
                % (local_path, name, self.url(remote_path)))
        else:
            raw = http_get(self.url(remote_path), self.timeout)
            self.fetched += 1
            write_atomic(local_path, raw)
        if expect_sha256 is not None and sha256_bytes(raw) != expect_sha256:
            die("%s: sha256 %s does not equal the sealed %s (cached copy at %s "
                "is not the pinned bytes)"
                % (name, sha256_bytes(raw), expect_sha256, local_path))
        return raw


def manifest_field(manifest, key, kinds, label):
    value = manifest.get(key)
    if not isinstance(value, kinds) or isinstance(value, bool):
        die("suite manifest %s is missing or not %s" % (label, kinds))
    return value


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="build_panel_from_v5_suite", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="panel directory to write")
    ap.add_argument("--panel-id", required=True)
    ap.add_argument("--panel-name", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="local mirror of the pinned public inputs: "
                         "suite-manifest.json, tokens/, tokenizer/. Kept "
                         "outside the panel tree; never published.")
    ap.add_argument("--fetch", action="store_true",
                    help="allow HTTPS fetches of missing cache entries from the "
                         "pinned revisions. Public files only, no credential, "
                         "no weights. Without it the run is offline.")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT",
                                                            "https://huggingface.co"))
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--corpus-repository", default="malaiwah/qwen38-27b-fidelity-suite-v5")
    ap.add_argument("--corpus-revision", required=True,
                    help="immutable 40-hex dataset commit")
    ap.add_argument("--suite-manifest-path",
                    default="suite/shard-0000/suite-manifest.json",
                    help="repository-relative path of the shard's sealed manifest")
    ap.add_argument("--tokens-path-prefix", default="suite/tokens/",
                    help="repository-relative prefix holding context-NNNN.json")
    ap.add_argument("--expect-suite-manifest-sha256", required=True,
                    help="sha256 of the shard manifest bytes; the pin that makes "
                         "every other seal in it meaningful")
    ap.add_argument("--expect-panel-token-sha256", required=True,
                    help="the registry panel's sealed identity.panel_token_sha256")
    ap.add_argument("--expect-panel-token-sha256-legacy", default=None,
                    help="the historical capture receipts' panel_token_sha256_legacy")
    ap.add_argument("--expect-suite-token-hash-sha256", default=None,
                    help="the normative canonical per-record aggregate")
    ap.add_argument("--tokenizer-repository", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--tokenizer-revision", required=True,
                    help="immutable 40-hex model commit; a branch name is refused")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="directory of tokenizer metadata files "
                         "(default <cache-dir>/tokenizer)")
    ap.add_argument("--bind-tokenizer-file", action="append", dest="bound_files",
                    default=None,
                    help="tokenizer file to BIND into the panel's tokenizer "
                         "identity; repeatable. Default: %s"
                         % ", ".join(DEFAULT_BOUND_TOKENIZER_FILES))
    ap.add_argument("--evidence-tokenizer-file", action="append",
                    dest="evidence_files", default=None,
                    help="tokenizer file fetched and digested as evidence but "
                         "not bound; repeatable. Default: %s"
                         % ", ".join(DEFAULT_EVIDENCE_TOKENIZER_FILES))
    ap.add_argument("--prior-receipt", action="append", dest="prior_receipts",
                    default=None,
                    help="a historical receipt for this panel whose provenance "
                         "this build carries forward; repeatable")
    ap.add_argument("--registry-panel-manifest-sha256", default=None,
                    help="the registry panel record's identity.manifest_sha256")
    ap.add_argument("--registry-panel-source-uri", default=None,
                    help="the registry panel record's recorded source uri")
    ap.add_argument("--provenance-out", default=None,
                    help="optional small recovery sidecar to write beside the tree")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing panel tree at --out")
    return ap.parse_args(argv)


def prepare_out(out, force):
    if os.path.lexists(out):
        if not force:
            die("%s exists (use --force to replace the panel tree)" % out)
        if os.path.islink(out) or not os.path.isdir(out):
            die("%s exists and is not a directory; refusing to replace it" % out)
        entries = set(os.listdir(out))
        allowed = {"panel.json", "panel.receipt.json", "tokenizer.receipt.json",
                   "arrays"}
        if not entries or not entries.issubset(allowed):
            die("%s does not look like a panel tree (unexpected entries %s); "
                "refusing to delete it"
                % (out, sorted(entries - allowed) or ["<empty>"]))
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "arrays"))


def load_manifest(source, cache, args):
    raw = source.get(args.suite_manifest_path,
                     os.path.join(cache, "suite-manifest.json"),
                     expect_sha256=hex64(args.expect_suite_manifest_sha256,
                                         "--expect-suite-manifest-sha256"),
                     label="suite manifest")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        die("suite manifest is not strict UTF-8 JSON: %s" % exc)
    if not isinstance(manifest, dict):
        die("suite manifest must be a JSON object")
    return manifest, raw


def tokenizer_identity(source, cache, args, manifest):
    """Fetch, verify and describe the pinned tokenizer metadata."""
    tokenizer_dir = args.tokenizer_dir or os.path.join(cache, "tokenizer")
    bound = list(args.bound_files or DEFAULT_BOUND_TOKENIZER_FILES)
    evidence = list(args.evidence_files or DEFAULT_EVIDENCE_TOKENIZER_FILES)
    if "config.json" not in bound and "config.json" not in evidence:
        die("config.json must be fetched: it carries the suite-sealed "
            "config_sha256 and the vocabulary width")
    names = []
    for name in bound + evidence:
        if name in names:
            continue
        if name != os.path.basename(name) or name.startswith("."):
            die("tokenizer file %r must be a plain file name" % name)
        names.append(name)

    digests = {}
    sizes = {}
    for name in names:
        raw = source.get(name, os.path.join(tokenizer_dir, name),
                         label="tokenizer file %s" % name)
        digests[name] = sha256_bytes(raw)
        sizes[name] = len(raw)

    identity_seals = manifest_field(manifest, "model_identity", dict,
                                    "model_identity")
    sealed_tokenizer = hex64(identity_seals.get("tokenizer_sha256"),
                             "suite manifest model_identity.tokenizer_sha256")
    sealed_config = hex64(identity_seals.get("config_sha256"),
                          "suite manifest model_identity.config_sha256")
    if digests.get("tokenizer.json") != sealed_tokenizer:
        die("tokenizer.json fetched at %s@%s has sha256 %s but the suite "
            "manifest was built with tokenizer_sha256 %s -- this is NOT the "
            "tokenizer that produced these token ids"
            % (args.tokenizer_repository, args.tokenizer_revision,
               digests.get("tokenizer.json"), sealed_tokenizer))
    if digests.get("config.json") != sealed_config:
        die("config.json fetched at %s@%s has sha256 %s but the suite manifest "
            "was built against config_sha256 %s -- this is NOT the pinned model"
            % (args.tokenizer_repository, args.tokenizer_revision,
               digests.get("config.json"), sealed_config))

    with open(os.path.join(tokenizer_dir, "config.json"), "rb") as handle:
        config = json.loads(handle.read().decode("utf-8"))
    if not isinstance(config, dict):
        die("model config.json must be a JSON object")
    vocab_size = manifest_field(manifest, "vocab_size", int, "vocab_size")
    # A multimodal checkpoint carries the text vocabulary under `text_config`
    # (the width every capture and every KL reduction in this suite uses).
    text_config = config.get("text_config")
    config_vocab, vocab_path = config.get("vocab_size"), "vocab_size"
    if ((isinstance(config_vocab, bool) or not isinstance(config_vocab, int))
            and isinstance(text_config, dict)):
        config_vocab, vocab_path = text_config.get("vocab_size"), "text_config.vocab_size"
    if isinstance(config_vocab, bool) or not isinstance(config_vocab, int):
        die("model config.json has no integer vocab_size or text_config.vocab_size")
    if config_vocab != vocab_size:
        die("model config.json %s is %d and disagrees with the suite "
            "manifest's vocab_size %d" % (vocab_path, config_vocab, vocab_size))

    tokenizer_class = None
    if "tokenizer_config.json" in digests:
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "rb") as handle:
            tokenizer_config = json.loads(handle.read().decode("utf-8"))
        if not isinstance(tokenizer_config, dict):
            die("tokenizer_config.json must be a JSON object")
        declared = tokenizer_config.get("tokenizer_class")
        if isinstance(declared, str) and declared.strip():
            tokenizer_class = declared

    identity = {
        "class": tokenizer_class,
        "model_id": args.tokenizer_repository,
        "model_revision": args.tokenizer_revision,
        "files": [{"path": name, "bytes": sizes[name], "sha256": digests[name]}
                  for name in sorted(bound)],
    }
    return {
        "dir": tokenizer_dir,
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical(identity, newline=True)),
        "bound": sorted(bound),
        "evidence": {name: {"bytes": sizes[name], "sha256": digests[name]}
                     for name in names if name not in bound},
        "digests": digests,
        "sizes": sizes,
        "vocab_size": vocab_size,
        "config_vocab_size_path": vocab_path,
        "sealed_tokenizer_sha256": sealed_tokenizer,
        "sealed_config_sha256": sealed_config,
        "trust_remote_code": identity_seals.get("trust_remote_code"),
        "model_type": config.get("model_type"),
    }


def transport_contexts(source, cache, args, manifest, arrays, mask_sha, vocab_size):
    """Verify and write every context of the shard. Returns (windows, digests)."""
    import numpy as np

    index = manifest_field(manifest, "context_index", list, "context_index")
    if not index:
        die("suite manifest context_index is empty")
    contexts = manifest_field(manifest, "contexts", int, "contexts")
    if contexts != len(index):
        die("suite manifest declares %d contexts but context_index has %d rows"
            % (contexts, len(index)))
    context_length = manifest_field(manifest, "context_length", int, "context_length")
    scored = manifest.get("scored_positions_per_context")
    if scored is not None and scored != context_length - 1:
        die("suite manifest scores %r of %d positions per context; this "
            "transport emits full-context all-ones masks only"
            % (scored, context_length))

    tokens_dir = os.path.join(cache, "tokens")
    windows = []
    file_digests = []
    observed_maximum = -1
    for position, row in enumerate(index):
        if not isinstance(row, dict):
            die("context_index[%d] is not an object" % position)
        remote = row.get("file")
        if not isinstance(remote, str) or not remote:
            die("context_index[%d] has no file" % position)
        name = os.path.basename(remote)
        if not CONTEXT_FILE.match(name):
            die("context_index[%d] file %r is not a context-NNNN.json name"
                % (position, remote))
        sealed = hex64(row.get("token_sha256"),
                       "context_index[%d].token_sha256" % position)
        raw = source.get(args.tokens_path_prefix + name,
                         os.path.join(tokens_dir, name),
                         expect_sha256=sealed, label="context %s" % name)
        file_digests.append(sealed)

        try:
            ids = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            die("context %s is not strict UTF-8 JSON: %s" % (name, exc))
        if not isinstance(ids, list) or not ids:
            die("context %s is not a non-empty token-id array" % name)
        declared = row.get("tokens")
        if isinstance(declared, bool) or not isinstance(declared, int):
            die("context_index[%d].tokens is not an integer" % position)
        if len(ids) != declared:
            die("context %s: expected %d ids, got %d" % (name, declared, len(ids)))
        if len(ids) != context_length:
            die("context %s: length %d != context_length %d"
                % (name, len(ids), context_length))
        values = []
        for value in ids:
            if isinstance(value, bool) or not isinstance(value, int):
                die("context %s contains a non-integer token id" % name)
            if value < 0 or value >= vocab_size:
                die("context %s has token id %d outside [0, %d) -- the pinned "
                    "tokenizer cannot represent it" % (name, value, vocab_size))
            values.append(int(value))
        observed_maximum = max(observed_maximum, max(values))

        window_id = "final-%04d" % position
        token_path = os.path.join(arrays, "%s.tokens.npy" % window_id)
        np.save(token_path, np.asarray(values, dtype=np.int32), allow_pickle=False)

        windows.append({
            "window_id": window_id,
            "role": "final",
            "domain": row.get("stratum"),
            "document_id": row.get("source_cluster"),
            "prediction_positions": context_length - 1,
            "token_ids_sha256": sha256_file(token_path),
            "attention_mask_sha256": mask_sha,
            "token_ids_json_sha256": token_ids_json_sha256(values),
            "token_ids_first16": values[:16],
            "token_ids_last16": values[-16:],
            "num_tokens": len(values),
            # provenance back to the sealed suite
            "suite_context_index": row.get("index"),
            "suite_context_file": remote,
            "suite_token_sha256": sealed,
            "suite_stratum": row.get("stratum"),
            "suite_source_cluster": row.get("source_cluster"),
            "suite_partition": row.get("partition"),
            "source_window_index": row.get("source_window_index"),
            "source_char_start": row.get("source_char_start"),
            "source_char_end": row.get("source_char_end"),
            "calibration_shingle_hits": row.get("calibration_shingle_hits"),
            "sentinel": row.get("sentinel"),
        })
    return windows, file_digests, context_length, observed_maximum


def document_provenance(manifest, windows):
    """Actual per-cluster provenance for the contexts this shard carries."""
    documents = manifest.get("documents")
    catalogue = {}
    if isinstance(documents, list):
        for entry in documents:
            if (isinstance(entry, list) and len(entry) == 2
                    and isinstance(entry[0], str) and isinstance(entry[1], dict)):
                catalogue[entry[0]] = entry[1]
    elif isinstance(documents, dict):
        catalogue = {k: v for k, v in documents.items() if isinstance(v, dict)}

    used = {}
    for window in windows:
        cluster = window["document_id"]
        row = used.setdefault(cluster, {"source_cluster": cluster, "contexts": 0,
                                        "strata": set(), "partitions": set()})
        row["contexts"] += 1
        row["strata"].add(window["domain"])
        row["partitions"].add(window["suite_partition"])

    rows = []
    missing = 0
    for cluster in sorted(used):
        row = used[cluster]
        entry = catalogue.get(cluster)
        if entry is None:
            missing += 1
        rows.append({
            "source_cluster": cluster,
            "contexts": row["contexts"],
            "strata": sorted(s for s in row["strata"] if s is not None),
            "partitions": sorted(p for p in row["partitions"] if p is not None),
            "stratum": (entry or {}).get("stratum"),
            "file": (entry or {}).get("file"),
            "bytes": (entry or {}).get("bytes"),
            "sha256": (entry or {}).get("sha256"),
        })
    by_stratum = {}
    for row in rows:
        for stratum in row["strata"]:
            by_stratum[stratum] = by_stratum.get(stratum, 0) + 1
    contexts_by_stratum = {}
    contexts_by_partition = {}
    for window in windows:
        contexts_by_stratum[window["domain"]] = (
            contexts_by_stratum.get(window["domain"], 0) + 1)
        key = window["suite_partition"]
        contexts_by_partition[key] = contexts_by_partition.get(key, 0) + 1
    return rows, {
        "source_clusters_in_shard": len(rows),
        "source_clusters_without_manifest_document_record": missing,
        "source_clusters_by_stratum": dict(sorted(by_stratum.items())),
        "contexts_by_stratum": dict(sorted(contexts_by_stratum.items())),
        "contexts_by_partition": dict(sorted(
            (str(k), v) for k, v in contexts_by_partition.items())),
        "note": ("counts are recomputed from this shard's own context_index "
                 "rows; they are not transferable to another panel"),
    }


def prior_receipt_records(paths):
    records = []
    for path in paths or ():
        if not os.path.isfile(path):
            die("--prior-receipt %s does not exist" % path)
        with open(path, "rb") as handle:
            raw = handle.read()
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            die("--prior-receipt %s is not strict UTF-8 JSON: %s" % (path, exc))
        if not isinstance(doc, dict):
            die("--prior-receipt %s is not a JSON object" % path)
        declared, convention = seal_conventions(doc)
        if convention is None:
            die("--prior-receipt %s does not verify under any seal convention "
                "this repository accepts; its provenance cannot be carried"
                % path)
        records.append({
            "path": os.path.relpath(path, os.getcwd()),
            "schema": doc.get("schema"),
            "file_sha256": sha256_bytes(raw),
            "file_bytes": len(raw),
            "declared_receipt_sha256": declared,
            "receipt_seal_convention": convention,
            "panel_id": doc.get("panel_id"),
            "suite_token_hash_sha256": doc.get("suite_token_hash_sha256"),
            "tool": doc.get("tool"),
        })
    return records


def main(argv=None):
    args = parse_args(argv)
    repository(args.corpus_repository, "--corpus-repository")
    repository(args.tokenizer_repository, "--tokenizer-repository")
    revision40(args.corpus_revision, "--corpus-revision")
    revision40(args.tokenizer_revision, "--tokenizer-revision")
    expect_panel_token = hex64(args.expect_panel_token_sha256,
                               "--expect-panel-token-sha256")
    expect_legacy = (hex64(args.expect_panel_token_sha256_legacy,
                           "--expect-panel-token-sha256-legacy")
                     if args.expect_panel_token_sha256_legacy else None)
    expect_canonical = (hex64(args.expect_suite_token_hash_sha256,
                              "--expect-suite-token-hash-sha256")
                        if args.expect_suite_token_hash_sha256 else None)
    if args.registry_panel_manifest_sha256:
        hex64(args.registry_panel_manifest_sha256,
              "--registry-panel-manifest-sha256")
    if not args.tokens_path_prefix.endswith("/"):
        die("--tokens-path-prefix must end with '/'")

    import numpy as np

    cache = os.path.abspath(args.cache_dir)
    out = os.path.abspath(args.out)
    if cache == out or cache.startswith(out + os.sep):
        die("--cache-dir must live outside the panel tree: the panel is an "
            "exact sealed file closure and refuses unlisted files")
    prior = prior_receipt_records(args.prior_receipts)

    corpus = Source(args.hf_endpoint, "dataset", args.corpus_repository,
                    args.corpus_revision, cache, args.fetch, args.timeout)
    tokenizer_source = Source(args.hf_endpoint, "model", args.tokenizer_repository,
                              args.tokenizer_revision, cache, args.fetch,
                              args.timeout)

    manifest, manifest_raw = load_manifest(corpus, cache, args)
    tokenizer = tokenizer_identity(tokenizer_source, cache, args, manifest)
    sealed_suite_token = manifest_field(manifest, "suite_token_sha256", str,
                                        "suite_token_sha256")
    hex64(sealed_suite_token, "suite manifest suite_token_sha256")
    if sealed_suite_token != expect_panel_token:
        die("suite manifest suite_token_sha256 %s is not the expected panel "
            "token digest %s -- this manifest is a different shard"
            % (sealed_suite_token, expect_panel_token))

    prepare_out(out, args.force)
    arrays = os.path.join(out, "arrays")
    context_length = manifest_field(manifest, "context_length", int, "context_length")
    mask_path = os.path.join(arrays, "causal-mask-%d.npy" % context_length)
    np.save(mask_path, np.ones(context_length, dtype=np.uint8), allow_pickle=False)
    mask_sha = sha256_file(mask_path)

    windows, file_digests, context_length, observed_maximum = transport_contexts(
        corpus, cache, args, manifest, arrays, mask_sha, tokenizer["vocab_size"])

    # SEAL 2: the suite's own aggregate rule must reproduce the registry
    # panel's sealed digest, or this is not that panel.
    shard_aggregate = sha256_bytes("".join(file_digests).encode("ascii"))
    if shard_aggregate != expect_panel_token:
        die("shard token digest %s != sealed panel_token_sha256 %s -- this is "
            "NOT the registry panel" % (shard_aggregate, expect_panel_token))
    per_record = [window["token_ids_json_sha256"] for window in windows]
    canonical_aggregate = suite_token_hash_sha256(per_record)
    legacy_aggregate = suite_token_hash_sha256_legacy(per_record)
    # SEAL 3/4: the aggregates the historical receipts and a resolved binding
    # recompute, when the caller pins them.
    if expect_legacy is not None and legacy_aggregate != expect_legacy:
        die("legacy panel token aggregate %s != expected %s -- the transported "
            "token ids are not the ids the historical receipts scored"
            % (legacy_aggregate, expect_legacy))
    if expect_canonical is not None and canonical_aggregate != expect_canonical:
        die("canonical suite token aggregate %s != expected %s"
            % (canonical_aggregate, expect_canonical))

    documents, provenance_summary = document_provenance(manifest, windows)

    token_receipt = {
        "schema": TOKENIZER_RECEIPT_SCHEMA,
        "format_version": 1,
        "receipt_sha256": "",
        "tokenizer_identity": tokenizer["identity"],
        "tokenizer_identity_sha256": tokenizer["identity_sha256"],
        "artifacts": list(tokenizer["identity"]["files"]),
        "vocab_size": tokenizer["vocab_size"],
        "minimum_token_id": 0,
        "maximum_token_id_exclusive": tokenizer["vocab_size"],
        "observed_token_id_maximum": observed_maximum,
        "model_type": tokenizer["model_type"],
        "trust_remote_code": tokenizer["trust_remote_code"],
        "evidence_files": tokenizer["evidence"],
        "sealed_by_suite": {
            "suite_manifest_sha256": sha256_bytes(manifest_raw),
            "model_identity_tokenizer_sha256": tokenizer["sealed_tokenizer_sha256"],
            "model_identity_config_sha256": tokenizer["sealed_config_sha256"],
            "rule": ("the suite manifest's model_identity digests equal the "
                     "sha256 of tokenizer.json and config.json at the pinned "
                     "model revision, so this tokenizer is the one that "
                     "produced these token ids"),
        },
        "add_special_tokens": False,
        "chat_template_applied": False,
        "note": ("no tokenizer was loaded and no text was tokenized: the token "
                 "ids are transported from the sealed suite. These files are "
                 "identity evidence, verified byte-for-byte against the "
                 "target's own directory before any spend."),
    }
    token_receipt["receipt_sha256"] = seal(token_receipt)

    panel = {
        "schema": PANEL_SCHEMA,
        "panel_id": args.panel_id,
        "name": args.panel_name,
        "sealed_corpus_sha256": None,
        "suite_token_hash_sha256": canonical_aggregate,
        "panel_token_sha256_legacy": legacy_aggregate,
        "provenance": {
            "derivation": "transport, not reconstruction",
            "source": "%s@%s %s + %s"
                      % (args.corpus_repository, args.corpus_revision,
                         args.suite_manifest_path, args.tokens_path_prefix),
            "shard_token_sha256": shard_aggregate,
            "tokenizer": "%s@%s" % (args.tokenizer_repository,
                                    args.tokenizer_revision),
            "note": ("window N carries context_index[N]'s token ids verbatim; "
                     "masks are all-ones, so every causal position is scored"),
        },
        "tokenizer": {
            "id": args.tokenizer_repository,
            "repository": args.tokenizer_repository,
            "revision": args.tokenizer_revision,
            "vocab_size": tokenizer["vocab_size"],
            "add_special_tokens": False,
            "chat_template_applied": False,
        },
        "windows": windows,
    }

    receipt = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "format_version": 1,
        "receipt_sha256": "",
        "tool": "engines/tools/build_panel_from_v5_suite.py",
        "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "panel_id": args.panel_id,
        "panel_name": args.panel_name,
        "suite_token_hash_sha256": canonical_aggregate,
        "panel_token_sha256_legacy": legacy_aggregate,
        "token_panel_schema": PANEL_SCHEMA,
        "derivation": "transport, not reconstruction",
        "selection_rule": SELECTION_RULE,
        "parameters": {
            "context_length": context_length,
            "prediction_positions_per_window": context_length - 1,
            "windows_total": len(windows),
            "scored_positions_total": len(windows) * (context_length - 1),
            "score_from": 0,
            "windowed": False,
            "mask": "all ones over the full context",
        },
        "corpus": {
            "repository": args.corpus_repository,
            "revision": args.corpus_revision,
            "repo_type": "dataset",
            "suite_manifest_path": args.suite_manifest_path,
            "suite_manifest_sha256": sha256_bytes(manifest_raw),
            "suite_manifest_bytes": len(manifest_raw),
            "suite_schema": manifest.get("schema"),
            "suite_token_sha256_parent": sealed_suite_token,
            "path_prefix": args.tokens_path_prefix,
            "shard": {
                "index": (manifest.get("shard") or {}).get("index")
                         if isinstance(manifest.get("shard"), dict) else None,
                "contexts": len(windows),
                "partial": (manifest.get("shard") or {}).get("partial")
                           if isinstance(manifest.get("shard"), dict) else None,
            },
            "suite_contexts_requested": manifest.get("contexts_requested"),
            "suite_strata": manifest.get("strata"),
            "suite_partitions": manifest.get("partitions"),
            "suite_source_clusters": manifest.get("source_clusters"),
            "corpus_note": manifest.get("corpus_note"),
        },
        "document_provenance": provenance_summary,
        "documents": documents,
        "seals_verified": {
            "per_context_file_sha256": {
                "rule": "sha256(context-NNNN.json bytes) == context_index[i].token_sha256",
                "contexts_checked": len(windows),
                "mismatches": 0,
            },
            "shard_aggregate": {
                "rule": ("sha256(concat of per-context token_sha256 hex in "
                         "context_index order) == suite manifest "
                         "suite_token_sha256 == registry panel "
                         "identity.panel_token_sha256"),
                "expected": expect_panel_token,
                "recomputed": shard_aggregate,
                "match": True,
            },
            "panel_token_legacy_aggregate": {
                "rule": ("sha256(empty-string join of per-record "
                         "token_ids_json_sha256) == the historical receipts' "
                         "panel_token_sha256_legacy"),
                "expected": expect_legacy,
                "recomputed": legacy_aggregate,
                "match": expect_legacy is None or legacy_aggregate == expect_legacy,
                "pinned_by_caller": expect_legacy is not None,
            },
            "canonical_token_aggregate": {
                "rule": ("sha256(newline join of per-record "
                         "token_ids_json_sha256) == suite_token_hash_sha256, "
                         "the aggregate a resolved panel binding recomputes"),
                "expected": expect_canonical,
                "recomputed": canonical_aggregate,
                "match": (expect_canonical is None
                          or canonical_aggregate == expect_canonical),
                "pinned_by_caller": expect_canonical is not None,
            },
            "tokenizer_identity": {
                "rule": ("sha256(tokenizer.json) == suite manifest "
                         "model_identity.tokenizer_sha256 and "
                         "sha256(config.json) == model_identity.config_sha256 "
                         "and the model config's declared text vocabulary "
                         "equals the manifest's vocab_size"),
                "tokenizer_sha256": tokenizer["sealed_tokenizer_sha256"],
                "config_sha256": tokenizer["sealed_config_sha256"],
                "config_vocab_size_path": tokenizer["config_vocab_size_path"],
                "vocab_size": tokenizer["vocab_size"],
                "observed_token_id_maximum": observed_maximum,
                "match": True,
            },
        },
        "tokenizer": {
            # PATH-2: the repository is the identity; never a local directory.
            "id": args.tokenizer_repository,
            "repository": args.tokenizer_repository,
            "revision": args.tokenizer_revision,
            "vocab_size": tokenizer["vocab_size"],
            "minimum_token_id": 0,
            "maximum_token_id_exclusive": tokenizer["vocab_size"],
            "add_special_tokens": False,
            "chat_template_applied": False,
            "files_sha256": {name: tokenizer["digests"][name]
                             for name in tokenizer["bound"]},
            "evidence_files_sha256": {name: row["sha256"] for name, row
                                      in sorted(tokenizer["evidence"].items())},
            "note": ("no tokenizer was loaded by this tool; token ids are "
                     "transported from the sealed suite. Identity is PROVED "
                     "against the suite manifest's model_identity digests, not "
                     "inherited from a repository name."),
        },
        "tokenizer_receipt_sha256": token_receipt["receipt_sha256"],
        "tokenizer_identity_sha256": tokenizer["identity_sha256"],
        "separation": {
            "checked": False,
            "method": ("this tool runs no scan; the parent suite manifest "
                       "records the scan its builder ran"),
            "inherited_scan": {
                "present": isinstance(manifest.get("contamination_scan"), dict),
                "contamination_scan": manifest.get("contamination_scan", {}).get(
                    "normalization") if isinstance(
                        manifest.get("contamination_scan"), dict) else None,
                "calibration_shingles": manifest.get(
                    "contamination_scan", {}).get("calibration_shingles")
                    if isinstance(manifest.get("contamination_scan"), dict) else None,
                "documents_excluded_any_reason": manifest.get(
                    "document_scan", {}).get("documents_excluded_any_reason")
                    if isinstance(manifest.get("document_scan"), dict) else None,
                "shard_calibration_shingle_hits": sum(
                    int(window["calibration_shingle_hits"] or 0)
                    for window in windows),
            },
            "note": manifest.get("corpus_note"),
        },
        "registry_panel": {
            "panel_id": args.panel_id,
            "panel_token_sha256": expect_panel_token,
            "manifest_sha256": args.registry_panel_manifest_sha256,
            "source_uri": args.registry_panel_source_uri,
            "note": ("the registry record's own source is a GitHub receipt; "
                     "this build re-derives the same token digest from the "
                     "public dataset instead of trusting that record"),
        },
        "source_receipts": prior,
    }
    receipt["receipt_sha256"] = seal(receipt)

    write_json_atomic(os.path.join(out, "tokenizer.receipt.json"), token_receipt,
                      sort_keys=True)
    write_json_atomic(os.path.join(out, "panel.json"), panel, sort_keys=False)
    write_json_atomic(os.path.join(out, "panel.receipt.json"), receipt,
                      sort_keys=True)

    # The panel is only real if the resolver accepts it: exact file closure,
    # every array digest, every mask/position relation, the receipt seals and
    # the tokenizer files verified against the fetched directory.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "bin"))
    from fidelity import panel as panel_api

    try:
        binding = panel_api.resolve_panel(
            out, role="final", tokenizer_root=tokenizer["dir"]).to_dict()
    except panel_api.PanelError as exc:
        die("the emitted panel does not resolve: %s" % exc)
    if binding["tokenizer"]["files_verified"] is not True:
        die("the emitted panel's tokenizer files did not verify against %s"
            % tokenizer["dir"])
    if binding["panel"]["suite_token_hash_sha256"] != canonical_aggregate:
        die("resolved binding aggregate %s != %s"
            % (binding["panel"]["suite_token_hash_sha256"], canonical_aggregate))
    if binding["tokenizer"]["identity_sha256"] != tokenizer["identity_sha256"]:
        die("resolved binding tokenizer identity differs from the emitted receipt")

    summary = {
        "panel_id": args.panel_id,
        "out": os.path.relpath(out, os.getcwd()),
        "windows": len(windows),
        "context_length": context_length,
        "scored_positions_total": len(windows) * (context_length - 1),
        "suite_token_hash_sha256": canonical_aggregate,
        "panel_token_sha256_legacy": legacy_aggregate,
        "shard_token_sha256_verified": shard_aggregate,
        "panel_receipt_sha256": receipt["receipt_sha256"],
        "tokenizer_receipt_sha256": token_receipt["receipt_sha256"],
        "tokenizer_identity_sha256": tokenizer["identity_sha256"],
        "source_clusters_in_shard": provenance_summary["source_clusters_in_shard"],
        "inputs_fetched": corpus.fetched + tokenizer_source.fetched,
        "inputs_reused_from_cache": corpus.reused + tokenizer_source.reused,
        "resolved_binding": {
            "panel_file_sha256": binding["panel"]["sha256"],
            "receipt_file_sha256": binding["receipt"]["receipt_file_sha256"],
            "manifest_sha256": binding["content"]["manifest_sha256"],
            "archive_sha256": binding["content"]["archive"]["sha256"],
            "files": len(binding["content"]["manifest"]),
            "bytes": sum(row["bytes"] for row in binding["content"]["manifest"]),
            "tokenizer_files_verified": True,
        },
    }

    if args.provenance_out:
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "panel_id": args.panel_id,
            "panel_name": args.panel_name,
            "tool": "engines/tools/build_panel_from_v5_suite.py",
            "tool_sha256": receipt["tool_sha256"],
            "derivation": "transport, not reconstruction",
            "source": {
                "repo_type": "dataset",
                "repository": args.corpus_repository,
                "revision": args.corpus_revision,
                "suite_manifest_path": args.suite_manifest_path,
                "suite_manifest_sha256": sha256_bytes(manifest_raw),
                "tokens_path_prefix": args.tokens_path_prefix,
                "contexts": len(windows),
            },
            "tokenizer_source": {
                "repo_type": "model",
                "repository": args.tokenizer_repository,
                "revision": args.tokenizer_revision,
                "bound_files": tokenizer["bound"],
                "identity_sha256": tokenizer["identity_sha256"],
            },
            "seals": {
                "shard_token_sha256": shard_aggregate,
                "panel_token_sha256_legacy": legacy_aggregate,
                "suite_token_hash_sha256": canonical_aggregate,
                "panel_receipt_sha256": receipt["receipt_sha256"],
                "tokenizer_receipt_sha256": token_receipt["receipt_sha256"],
            },
            "content": {
                "file_count": summary["resolved_binding"]["files"],
                "total_bytes": summary["resolved_binding"]["bytes"],
                "manifest_sha256": summary["resolved_binding"]["manifest_sha256"],
                "archive_sha256": summary["resolved_binding"]["archive_sha256"],
            },
            "source_receipts": prior,
            "verification": ("every context file checked against the sealed "
                             "suite manifest digest, the shard aggregate against "
                             "the registry panel token digest, the tokenizer "
                             "files against the manifest's model_identity "
                             "digests, and the emitted tree against "
                             "fidelity.panel.resolve_panel"),
        }
        write_json_atomic(args.provenance_out, provenance, sort_keys=True)
        summary["provenance_out"] = os.path.relpath(args.provenance_out, os.getcwd())

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
