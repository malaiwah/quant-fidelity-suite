#!/usr/bin/env python3
"""build_panel_from_v5_suite transports a sealed suite shard, or it refuses.

The producer's whole job is to turn a published token suite into a panel tree
that `fidelity.panel.resolve_panel` accepts, WITHOUT re-deriving a single token
id and without letting a plausible-looking input through.  What this file
asserts, on a synthetic sealed suite built here (no network, no tokenizer, no
GPU, no credential):

  [1] the happy path: every context byte digest, the shard aggregate, the
      legacy panel aggregate and the canonical aggregate verify; the emitted
      tree resolves with `tokenizer.files_verified == True` against the fetched
      tokenizer directory; the panel receipt's parameters, its bound tokenizer
      receipt seal and the panel's own aggregate all agree with the binding;
      per-window suite provenance (index, stratum, cluster, partition, char
      offsets) survives the transport;
  [2] it is deterministic: a second build of the same pinned inputs is
      byte-identical, so the receipt carries no wall-clock stamp;
  [3] offline is the default: a missing input refuses and names the URL to
      mirror, and nothing is written;
  [4] tamper and boundary refusals, one seal at a time -- a flipped token id
      (context digest), a swapped manifest (manifest pin), a wrong registry
      digest, a wrong legacy aggregate, a wrong canonical aggregate, a
      tokenizer.json that is not the one the suite recorded, a config.json
      whose vocab_size disagrees with the suite, a token id at and beyond the
      vocabulary boundary, a mutable tokenizer revision, a context whose
      length disagrees with the manifest, and a prior receipt whose own seal
      does not verify;
  [5] the emitted tree stays tamper-evident after the build: flipping one byte
      of one array, adding a stray file, or editing the sealed receipt makes
      the resolver refuse.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "bin"))

TOOL = os.path.join(HERE, "build_panel_from_v5_suite.py")
FAILED = []

CORPUS_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40
PANEL_ID = "panel--selftest.transport.shard0"
PANEL_NAME = "selftest transport shard 0"
CONTEXTS = 4
CONTEXT_LENGTH = 8
VOCAB = 64


def check(label, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label,
                          ("  [%s]" % detail) if detail and not ok else ""))
    if not ok:
        FAILED.append(label)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "wb") as handle:
        handle.write(data)


def read(path):
    with open(path, "rb") as handle:
        return handle.read()


def token_ids(index):
    """A fixed, RNG-free id sequence per context, inside the vocabulary."""
    return [(index * 13 + position * 7 + 1) % VOCAB
            for position in range(CONTEXT_LENGTH)]


def build_suite(cache, *, ids_override=None, vocab_size=VOCAB,
                config_vocab=None, tokenizer_json=b'{"version": "1.0"}\n',
                nested_vocab=False, context_length=CONTEXT_LENGTH):
    """A synthetic sealed suite in the real manifest's shape.

    Returns the pins a caller must pass: manifest digest, shard aggregate,
    legacy aggregate, canonical aggregate.
    """
    tokens_dir = os.path.join(cache, "tokens")
    rows = []
    file_digests = []
    per_record = []
    for index in range(CONTEXTS):
        ids = (ids_override(index) if ids_override else token_ids(index))
        raw = (json.dumps(ids) + "\n").encode("utf-8")
        name = "context-%04d.json" % index
        write(os.path.join(tokens_dir, name), raw)
        digest = sha256_bytes(raw)
        file_digests.append(digest)
        per_record.append(sha256_bytes(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")))
        rows.append({
            "index": index,
            "stratum": ["code", "literary"][index % 2],
            "source_cluster": "cluster-%02d" % (index // 2),
            "source_window_index": index,
            "source_char_start": index * 100,
            "source_char_end": index * 100 + 90,
            "file": "tokens/" + name,
            "tokens": len(ids),
            "token_sha256": digest,
            "calibration_shingle_hits": 0,
            "partition": ["analysis", "qualification"][index % 2],
            "sentinel": False,
        })

    tokenizer_dir = os.path.join(cache, "tokenizer")
    declared_vocab = vocab_size if config_vocab is None else config_vocab
    config = {"model_type": "selftest_transport", "num_hidden_layers": 2}
    if nested_vocab:
        # A multimodal checkpoint keeps the text vocabulary one level down.
        config["text_config"] = {"vocab_size": declared_vocab,
                                 "hidden_size": 16}
    else:
        config["vocab_size"] = declared_vocab
    config_raw = (json.dumps(config, indent=1) + "\n").encode("utf-8")
    files = {
        "tokenizer.json": tokenizer_json,
        "tokenizer_config.json": b'{"tokenizer_class": "SelftestTokenizer"}\n',
        "vocab.json": b'{"a": 0, "b": 1}\n',
        "merges.txt": b"a b\n",
        "config.json": config_raw,
        "generation_config.json": b'{"do_sample": false}\n',
        "chat_template.jinja": b"{{ messages }}\n",
    }
    for name, raw in files.items():
        write(os.path.join(tokenizer_dir, name), raw)

    shard_aggregate = sha256_bytes("".join(file_digests).encode("ascii"))
    manifest = {
        "schema": "selftest-distribution-fidelity/1",
        "model": "/models/selftest",
        "model_identity": {
            "tokenizer_sha256": sha256_bytes(files["tokenizer.json"]),
            "config_sha256": sha256_bytes(config_raw),
            "trust_remote_code": False,
        },
        "context_length": context_length,
        "scored_positions_per_context": context_length - 1,
        "contexts": CONTEXTS,
        "contexts_requested": CONTEXTS * 2,
        "vocab_size": vocab_size,
        "hidden_size": 16,
        "corpus_note": "selftest fixture corpus; not a real separation claim",
        "source_clusters": 2,
        "strata": {"code": 2, "literary": 2},
        "partitions": {"analysis": 2, "qualification": 2},
        "shard": {"index": 0, "size": CONTEXTS, "contexts": CONTEXTS,
                  "partial": False},
        "contamination_scan": {"normalization": "selftest NFKC",
                               "calibration_shingles": 11},
        "document_scan": {"documents_excluded_any_reason": 1},
        "documents": [
            ["cluster-00", {"stratum": "code", "file": "cluster-00.txt",
                            "bytes": 10, "sha256": "0" * 64}],
            ["cluster-01", {"stratum": "literary", "file": "cluster-01.txt",
                            "bytes": 20, "sha256": "1" * 64}],
        ],
        "context_index": rows,
        "suite_token_sha256": shard_aggregate,
        "total_scored_positions": CONTEXTS * (context_length - 1),
    }
    manifest_raw = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    write(os.path.join(cache, "suite-manifest.json"), manifest_raw)
    return {
        "manifest_sha256": sha256_bytes(manifest_raw),
        "shard_aggregate": shard_aggregate,
        "legacy_aggregate": sha256_bytes("".join(per_record).encode("utf-8")),
        "canonical_aggregate": sha256_bytes("\n".join(per_record).encode("ascii")),
        "tokenizer_dir": tokenizer_dir,
    }


def run(out, cache, pins, *, extra=()):
    argv = [sys.executable, TOOL, "--out", out, "--panel-id", PANEL_ID,
            "--panel-name", PANEL_NAME, "--cache-dir", cache,
            "--corpus-repository", "selftest/suite",
            "--corpus-revision", CORPUS_REVISION,
            "--suite-manifest-path", "suite/shard-0000/suite-manifest.json",
            "--tokens-path-prefix", "suite/tokens/",
            "--expect-suite-manifest-sha256", pins["manifest_sha256"],
            "--expect-panel-token-sha256", pins["shard_aggregate"],
            "--expect-panel-token-sha256-legacy", pins["legacy_aggregate"],
            "--expect-suite-token-hash-sha256", pins["canonical_aggregate"],
            "--tokenizer-repository", "selftest/tokenizer",
            "--tokenizer-revision", TOKENIZER_REVISION, "--force"]
    argv.extend(extra)
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True)


def refuses(label, proc, needle, out=None):
    ok = proc.returncode == 4 and needle in (proc.stderr or "")
    if ok and out is not None:
        ok = not os.path.exists(os.path.join(out, "panel.json"))
    check(label, ok, (proc.stderr or proc.stdout or "").strip()[-200:])


def resolve(out, tokenizer_dir):
    from fidelity import panel as panel_api

    return panel_api.resolve_panel(out, role="final",
                                   tokenizer_root=tokenizer_dir).to_dict()


def resolver_refuses(label, out, tokenizer_dir):
    from fidelity import panel as panel_api

    try:
        resolve(out, tokenizer_dir)
    except panel_api.PanelError as exc:
        check("%s (refused: %s)" % (label, str(exc)[:70]), True)
        return
    check(label + " (resolver accepted it)", False)


def main():
    print("selftest_v5_panel_transport")
    work = tempfile.mkdtemp(prefix="v5panel-")
    try:
        # ---------------------------------------------------------------- [1]
        cache = os.path.join(work, "cache")
        pins = build_suite(cache)
        out = os.path.join(work, "panel")
        proc = run(out, cache, pins,
                   extra=["--provenance-out", os.path.join(work, "prov.json")])
        check("[1] transport of a sealed shard succeeds", proc.returncode == 0,
              (proc.stderr or "").strip()[-300:])
        if proc.returncode != 0:
            raise SystemExit(1 if FAILED else 0)
        summary = json.loads(proc.stdout)
        binding = resolve(out, pins["tokenizer_dir"])
        receipt = json.loads(read(os.path.join(out, "panel.receipt.json")))
        token_receipt = json.loads(read(os.path.join(out, "tokenizer.receipt.json")))
        panel = json.loads(read(os.path.join(out, "panel.json")))

        check("[1] tokenizer files verify against the fetched tokenizer dir",
              binding["tokenizer"]["files_verified"] is True)
        check("[1] the binding carries the pinned immutable tokenizer identity",
              binding["tokenizer"]["repository"] == "selftest/tokenizer"
              and binding["tokenizer"]["revision"] == TOKENIZER_REVISION
              and binding["tokenizer"]["identity_sha256"]
              == token_receipt["tokenizer_identity_sha256"]
              and len(binding["tokenizer"]["files"]) == 4)
        check("[1] the panel receipt binds the tokenizer receipt by seal",
              receipt["tokenizer_receipt_sha256"]
              == token_receipt["receipt_sha256"]
              == binding["tokenizer"]["receipt"]["declared_receipt_sha256"])
        check("[1] the receipt seals verify in the modern self-blank mode",
              binding["receipt"]["receipt_seal_mode"] == "self-blank"
              and binding["tokenizer"]["receipt"]["receipt_seal_mode"]
              == "self-blank")
        check("[1] the shard aggregate equals the registry panel token digest",
              receipt["seals_verified"]["shard_aggregate"]["recomputed"]
              == pins["shard_aggregate"]
              and receipt["seals_verified"]["shard_aggregate"]["expected"]
              == pins["shard_aggregate"])
        check("[1] both per-record aggregates are recorded and recomputed",
              receipt["suite_token_hash_sha256"] == pins["canonical_aggregate"]
              == binding["panel"]["suite_token_hash_sha256"]
              and receipt["panel_token_sha256_legacy"] == pins["legacy_aggregate"]
              == panel["panel_token_sha256_legacy"])
        check("[1] the binding shape is the full-context identity",
              binding["panel"]["contexts"] == CONTEXTS
              and binding["panel"]["context_length"] == CONTEXT_LENGTH
              and binding["panel"]["positions_per_context"] == CONTEXT_LENGTH - 1
              and binding["panel"]["scored_positions_total"]
              == CONTEXTS * (CONTEXT_LENGTH - 1)
              and binding["panel"]["id"] == PANEL_ID)
        check("[1] the resolved file closure is exactly the panel tree",
              sorted(row["path"] for row in binding["content"]["manifest"])
              == sorted(["panel.json", "panel.receipt.json",
                         "tokenizer.receipt.json",
                         "arrays/causal-mask-%d.npy" % CONTEXT_LENGTH]
                        + ["arrays/final-%04d.tokens.npy" % i
                           for i in range(CONTEXTS)]))
        first = panel["windows"][0]
        check("[1] transported ids are the suite's ids, not re-derived",
              first["token_ids_first16"] == token_ids(0)
              and first["num_tokens"] == CONTEXT_LENGTH
              and first["suite_token_sha256"]
              == sha256_bytes((json.dumps(token_ids(0)) + "\n").encode("utf-8")))
        check("[1] per-window suite provenance survives the transport",
              first["suite_context_index"] == 0
              and first["suite_context_file"] == "tokens/context-0000.json"
              and first["domain"] == "code" and first["document_id"] == "cluster-00"
              and first["suite_partition"] == "analysis"
              and first["source_char_start"] == 0
              and first["source_char_end"] == 90
              and first["prediction_positions"] == CONTEXT_LENGTH - 1)
        check("[1] document provenance is counted from this shard's own rows",
              receipt["document_provenance"]["source_clusters_in_shard"] == 2
              and receipt["document_provenance"]["contexts_by_stratum"]
              == {"code": 2, "literary": 2}
              and [row["source_cluster"] for row in receipt["documents"]]
              == ["cluster-00", "cluster-01"]
              and receipt["documents"][0]["sha256"] == "0" * 64)
        check("[1] this tool claims no separation scan of its own",
              receipt["separation"]["checked"] is False
              and receipt["separation"]["inherited_scan"]["present"] is True)
        check("[1] the tokenizer receipt bounds the vocabulary it saw",
              token_receipt["minimum_token_id"] == 0
              and token_receipt["maximum_token_id_exclusive"] == VOCAB
              and token_receipt["observed_token_id_maximum"] < VOCAB
              and token_receipt["vocab_size"] == VOCAB)
        check("[1] the reported snapshot is the resolved binding",
              summary["resolved_binding"]["tokenizer_files_verified"] is True
              and summary["resolved_binding"]["manifest_sha256"]
              == binding["content"]["manifest_sha256"]
              and summary["resolved_binding"]["archive_sha256"]
              == binding["content"]["archive"]["sha256"]
              and summary["suite_token_hash_sha256"] == pins["canonical_aggregate"])
        provenance = json.loads(read(os.path.join(work, "prov.json")))
        check("[1] the provenance sidecar names the pinned public sources",
              provenance["source"]["revision"] == CORPUS_REVISION
              and provenance["tokenizer_source"]["revision"] == TOKENIZER_REVISION
              and provenance["seals"]["shard_token_sha256"]
              == pins["shard_aggregate"])

        # ---------------------------------------------------------------- [2]
        again = os.path.join(work, "panel-again")
        proc = run(again, cache, pins)
        same = all(read(os.path.join(out, name)) == read(os.path.join(again, name))
                   for name in ("panel.json", "panel.receipt.json",
                                "tokenizer.receipt.json"))
        arrays_same = all(
            read(os.path.join(out, "arrays", name))
            == read(os.path.join(again, "arrays", name))
            for name in sorted(os.listdir(os.path.join(out, "arrays"))))
        check("[2] a rebuild of the same pinned inputs is byte-identical",
              proc.returncode == 0 and same and arrays_same)

        # ---------------------------------------------------------------- [3]
        partial = os.path.join(work, "cache-partial")
        shutil.copytree(cache, partial)
        os.unlink(os.path.join(partial, "tokens", "context-0002.json"))
        missing_out = os.path.join(work, "panel-missing")
        proc = run(missing_out, partial, pins)
        refuses("[3] a missing input refuses offline and names the URL",
                proc, "Re-run with --fetch", out=missing_out)
        check("[3] the refusal names the exact pinned resolve URL",
              "selftest/suite/resolve/%s/suite/tokens/context-0002.json"
              % CORPUS_REVISION in (proc.stderr or ""))

        # ---------------------------------------------------------------- [4]
        flipped = os.path.join(work, "cache-flip")
        shutil.copytree(cache, flipped)
        ids = token_ids(1)
        ids[3] = (ids[3] + 1) % VOCAB
        write(os.path.join(flipped, "tokens", "context-0001.json"),
              (json.dumps(ids) + "\n").encode("utf-8"))
        target = os.path.join(work, "panel-flip")
        refuses("[4] one flipped token id fails its sealed context digest",
                run(target, flipped, pins), "does not equal the sealed",
                out=target)

        swapped = os.path.join(work, "cache-swap")
        shutil.copytree(cache, swapped)
        doc = json.loads(read(os.path.join(swapped, "suite-manifest.json")))
        doc["corpus_note"] = "edited after sealing"
        write(os.path.join(swapped, "suite-manifest.json"),
              (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8"))
        target = os.path.join(work, "panel-swap")
        refuses("[4] an edited suite manifest fails its pin",
                run(target, swapped, pins), "suite manifest", out=target)

        target = os.path.join(work, "panel-wrong-registry")
        wrong = dict(pins, shard_aggregate="0" * 64)
        refuses("[4] a wrong registry panel token digest refuses",
                run(target, cache, wrong), "is not the expected panel token",
                out=target)

        target = os.path.join(work, "panel-wrong-legacy")
        refuses("[4] a wrong legacy panel aggregate refuses",
                run(target, cache, dict(pins, legacy_aggregate="1" * 64)),
                "legacy panel token aggregate", out=target)

        target = os.path.join(work, "panel-wrong-canonical")
        refuses("[4] a wrong canonical aggregate refuses",
                run(target, cache, dict(pins, canonical_aggregate="2" * 64)),
                "canonical suite token aggregate", out=target)

        foreign = os.path.join(work, "cache-foreign-tokenizer")
        shutil.copytree(cache, foreign)
        write(os.path.join(foreign, "tokenizer", "tokenizer.json"),
              b'{"version": "1.0", "note": "a different tokenizer"}\n')
        target = os.path.join(work, "panel-foreign-tokenizer")
        refuses("[4] a tokenizer.json the suite did not record refuses",
                run(target, foreign, pins),
                "is NOT the tokenizer that produced these token ids", out=target)

        vocab_cache = os.path.join(work, "cache-vocab")
        vocab_pins = build_suite(vocab_cache, config_vocab=VOCAB + 8)
        target = os.path.join(work, "panel-vocab")
        refuses("[4] a config.json vocab_size disagreeing with the suite refuses",
                run(target, vocab_cache, vocab_pins),
                "disagrees with the suite", out=target)

        nested_cache = os.path.join(work, "cache-nested-vocab")
        nested_pins = build_suite(nested_cache, nested_vocab=True)
        target = os.path.join(work, "panel-nested-vocab")
        proc = run(target, nested_cache, nested_pins)
        check("[4] a multimodal config's text_config vocabulary is accepted",
              proc.returncode == 0
              and json.loads(read(os.path.join(target, "panel.receipt.json")))[
                  "seals_verified"]["tokenizer_identity"][
                      "config_vocab_size_path"] == "text_config.vocab_size",
              (proc.stderr or "").strip()[-200:])

        nested_bad = os.path.join(work, "cache-nested-vocab-bad")
        nested_bad_pins = build_suite(nested_bad, nested_vocab=True,
                                      config_vocab=VOCAB + 8)
        target = os.path.join(work, "panel-nested-vocab-bad")
        refuses("[4] a text_config vocabulary disagreeing with the suite refuses",
                run(target, nested_bad, nested_bad_pins),
                "disagrees with the suite", out=target)

        edge_cache = os.path.join(work, "cache-edge")
        edge_pins = build_suite(
            edge_cache,
            ids_override=lambda index: ([VOCAB - 1] * CONTEXT_LENGTH if index == 2
                                        else token_ids(index)))
        target = os.path.join(work, "panel-edge")
        proc = run(target, edge_cache, edge_pins)
        check("[4] the highest representable token id is accepted",
              proc.returncode == 0
              and json.loads(read(os.path.join(
                  target, "tokenizer.receipt.json")))[
                      "observed_token_id_maximum"] == VOCAB - 1,
              (proc.stderr or "").strip()[-200:])

        over_cache = os.path.join(work, "cache-over")
        over_pins = build_suite(
            over_cache,
            ids_override=lambda index: ([VOCAB] * CONTEXT_LENGTH if index == 2
                                        else token_ids(index)))
        target = os.path.join(work, "panel-over")
        refuses("[4] a token id at the vocabulary boundary refuses",
                run(target, over_cache, over_pins), "outside [0, %d)" % VOCAB,
                out=target)

        target = os.path.join(work, "panel-mutable-revision")
        proc = run(target, cache, pins,
                   extra=["--tokenizer-revision", "main"])
        refuses("[4] a mutable tokenizer revision refuses before any write",
                proc, "immutable lowercase 40-hex commit", out=target)

        short_cache = os.path.join(work, "cache-short")
        short_pins = build_suite(short_cache)
        ids = token_ids(1)[:-1]
        raw = (json.dumps(ids) + "\n").encode("utf-8")
        write(os.path.join(short_cache, "tokens", "context-0001.json"), raw)
        # Re-seal the manifest around the short context so ONLY the declared
        # length disagrees: its byte digest and the shard aggregate still verify.
        manifest_doc = json.loads(read(os.path.join(short_cache,
                                                    "suite-manifest.json")))
        rows = manifest_doc["context_index"]
        rows[1]["token_sha256"] = sha256_bytes(raw)
        rows[1]["tokens"] = len(ids)
        manifest_doc["suite_token_sha256"] = sha256_bytes(
            "".join(row["token_sha256"] for row in rows).encode("ascii"))
        manifest_raw = (json.dumps(manifest_doc, indent=1, sort_keys=True)
                        + "\n").encode("utf-8")
        write(os.path.join(short_cache, "suite-manifest.json"), manifest_raw)
        short_pins = dict(short_pins,
                          manifest_sha256=sha256_bytes(manifest_raw),
                          shard_aggregate=manifest_doc["suite_token_sha256"])
        target = os.path.join(work, "panel-short")
        refuses("[4] a context shorter than context_length refuses",
                run(target, short_cache, short_pins),
                "!= context_length", out=target)

        bad_prior = os.path.join(work, "bad-prior-receipt.json")
        write(bad_prior, (json.dumps(
            {"schema": "malaiwah.token-panel-rebuild-receipt.v1",
             "panel_id": PANEL_ID, "receipt_sha256": "e" * 64}) + "\n").encode())
        target = os.path.join(work, "panel-bad-prior")
        refuses("[4] a prior receipt whose own seal fails cannot be carried",
                run(target, cache, pins, extra=["--prior-receipt", bad_prior]),
                "does not verify under any seal convention", out=target)

        good_prior = os.path.join(work, "prior-receipt.json")
        body = {"schema": "malaiwah.token-panel-rebuild-receipt.v1",
                "panel_id": PANEL_ID, "tool": "old/tool.py",
                "suite_token_hash_sha256": pins["canonical_aggregate"]}
        body["receipt_sha256"] = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write(good_prior, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode())
        target = os.path.join(work, "panel-prior")
        proc = run(target, cache, pins, extra=["--prior-receipt", good_prior])
        carried = (json.loads(read(os.path.join(target, "panel.receipt.json")))
                   ["source_receipts"] if proc.returncode == 0 else [])
        check("[4] the original rebuild receipt's provenance is carried forward",
              proc.returncode == 0 and len(carried) == 1
              and carried[0]["declared_receipt_sha256"] == body["receipt_sha256"]
              and carried[0]["receipt_seal_convention"]
              == "legacy-field-absent-ascii"
              and carried[0]["file_sha256"] == sha256_bytes(read(good_prior)),
              (proc.stderr or "").strip()[-200:])

        # ---------------------------------------------------------------- [5]
        tampered = os.path.join(work, "panel-tampered-array")
        shutil.copytree(out, tampered)
        array = os.path.join(tampered, "arrays", "final-0000.tokens.npy")
        raw = bytearray(read(array))
        raw[-1] = (raw[-1] + 1) % 256
        write(array, bytes(raw))
        resolver_refuses("[5] a flipped array byte breaks the panel",
                         tampered, pins["tokenizer_dir"])

        stray = os.path.join(work, "panel-stray-file")
        shutil.copytree(out, stray)
        write(os.path.join(stray, "notes.txt"), b"an unlisted file\n")
        resolver_refuses("[5] a stray file breaks the sealed closure",
                         stray, pins["tokenizer_dir"])

        edited = os.path.join(work, "panel-edited-receipt")
        shutil.copytree(out, edited)
        doc = json.loads(read(os.path.join(edited, "panel.receipt.json")))
        doc["parameters"]["scored_positions_total"] += 1
        write(os.path.join(edited, "panel.receipt.json"),
              (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        resolver_refuses("[5] an edited sealed receipt breaks its seal",
                         edited, pins["tokenizer_dir"])

        foreign_tokenizer = os.path.join(work, "foreign-tokenizer-dir")
        shutil.copytree(pins["tokenizer_dir"], foreign_tokenizer)
        write(os.path.join(foreign_tokenizer, "tokenizer.json"),
              b'{"version": "1.0", "note": "not the bound bytes"}\n')
        resolver_refuses("[5] a tokenizer directory that is not the bound bytes "
                         "cannot verify", out, foreign_tokenizer)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("%s: %d checks failed" % ("FAIL" if FAILED else "PASS", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
