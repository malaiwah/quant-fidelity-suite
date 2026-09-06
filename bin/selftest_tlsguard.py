#!/usr/bin/env python3
"""The TLS guard: explicit trust, peer attestation before a credential, and no
credential in a provider create body.

WHY THIS EXISTS
---------------
2026-09-05 (docs/CLOUD-RECIPES.md): a rented Vast host served a certificate for
`huggingface.co` with a hostname mismatch and then `SSLEOFError:
UNEXPECTED_EOF_WHILE_READING` -- a man-in-the-middle TLS proxy in front of a
box that was about to receive an HF token.  The token was written BEFORE
anything looked at that box's network.

A guard with only positive tests is a guard nobody has tested, so the rungs
below stand up REAL local TLS servers with scratch CAs and prove the guard
REFUSES: a self-signed peer, a hostname mismatch, a chain from a CA that the
ambient environment tries to add, and a leaf that differs from the one the
controller verified.

  T1   the shipped bundle: pinned digest, every root self-signed, every subject
       CN declared in tlsguard.EXPECTED_ROOT_SUBJECT_CNS, none expiring within
       a year, and each PEM header's claimed sha256 equal to the bytes below it.
  T2   the explicit context: check_hostname, CERT_REQUIRED, TLS >= 1.2.
  T3   NON-AMBIENT TRUST (gap 1).  A server signed by a scratch CA that
       `SSL_CERT_FILE` names is accepted by `ssl.create_default_context()` and
       REFUSED by ours -- so an ambient variable cannot widen trust.  It is
       still DISCLOSED.
  T4   `safe_urlopen` end to end: refuses that same server, keeps the
       auth-stripping redirect handler, and accepts under the DISCLOSED
       operator override.
  T5   a SELF-SIGNED peer is refused, with a remedy naming the host id.
  T6   a HOSTNAME MISMATCH is refused (cert covers wrong.invalid, served for
       localhost) -- the exact 2026-09-05 shape.
  T7   the operator overrides are DISCLOSURES, not silent fallbacks.
  T8   a tampered bundle refuses with the recomputed digest in the remedy.
  T9   reachability is not identity: a timeout is TLS-UNREACHABLE and
       retryable; HTTP 429 is a disclosure over an otherwise attested peer;
       leaf divergence refuses and its override discloses.
  T10  attest_before_credential over a stub exec channel: attests a good box,
       REFUSES a box whose leaf differs from the controller-verified leaf, and
       states that no credential was written.  Unparseable evidence is
       TLS-PEER-NO-EVIDENCE, also before any token.
  T11  the collector is ONE implementation: the base64 remote command executed
       by a separate interpreter returns the same fields as the in-process call.
  T12  the attestation document carries peer evidence, names its attester, and
       contains NO credential.
  T13  provider payload guard, PARAMETERISED over all four adapters (the RP7b
       contract that existed for RunPod only): env dict, `-e NAME=` text,
       onstart script, nested body and bearer header all refuse; the refusal
       NEVER echoes the credential; legitimate `HF_TOKEN_PATH` /
       `HF_HUB_DISABLE_IMPLICIT_TOKEN` shapes pass through.
  T14  network positive rung (SKIPPED offline): huggingface.co and every
       provider-API anchor verify against the shipped bundle.

Offline and stdlib-only except for the `openssl` CLI used to mint scratch
certificates; without it the local-TLS rungs print SKIP rather than passing
quietly.  No credential exists anywhere in this file: the fixture token is a
literal that matches the `hf_` shape on purpose.
"""
import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity import common                                      # noqa: E402
from fidelity import tlsguard                                    # noqa: E402

FAILED = []
SKIPPED = []

FIXTURE_TOKEN = "hf_" + "S" * 34
OPENSSL = shutil.which("openssl")


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:10]:
            print("        %s" % line)


def skip(label, why):
    print("  SKIP  %s (%s)" % (label, why))
    SKIPPED.append(label)


# --------------------------------------------------------------------------
# scratch PKI + local TLS servers
# --------------------------------------------------------------------------


def _run(args, **kw):
    out = subprocess.run(args, capture_output=True, text=True, **kw)
    if out.returncode != 0:
        raise RuntimeError("%s failed: %s" % (args[0], out.stderr[-400:]))
    return out


def make_ca(tmp, name):
    """A scratch CA.  keyUsage/basicConstraints are explicit because
    `ssl.create_default_context()` enables VERIFY_X509_STRICT on python3.13+,
    which refuses a CA certificate carrying no keyUsage extension -- and T3a
    needs the DEFAULT context to ACCEPT this CA in order to prove that ours
    does not."""
    key = tmp / ("%s-ca.key" % name)
    crt = tmp / ("%s-ca.crt" % name)
    _run([OPENSSL, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
          "-keyout", str(key), "-out", str(crt), "-days", "2",
          "-subj", "/CN=fidelity selftest %s CA" % name,
          "-addext", "basicConstraints=critical,CA:TRUE",
          "-addext", "keyUsage=critical,keyCertSign,cRLSign"])
    return key, crt


def make_leaf(tmp, name, cn, san, ca=None):
    key = tmp / ("%s.key" % name)
    crt = tmp / ("%s.crt" % name)
    csr = tmp / ("%s.csr" % name)
    if ca is None:
        _run([OPENSSL, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
              "-keyout", str(key), "-out", str(crt), "-days", "2",
              "-subj", "/CN=%s" % cn,
              "-addext", "subjectAltName=DNS:%s" % san])
        return key, crt
    ca_key, ca_crt = ca
    _run([OPENSSL, "req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
          "-out", str(csr), "-subj", "/CN=%s" % cn])
    _run([OPENSSL, "x509", "-req", "-in", str(csr), "-CA", str(ca_crt),
          "-CAkey", str(ca_key), "-out", str(crt), "-days", "2",
          "-extfile", "/dev/stdin"],
         input="subjectAltName=DNS:%s\nbasicConstraints=CA:FALSE\n" % san)
    return key, crt


class _Quiet(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):                                       # noqa: N802
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _QuietServer(http.server.HTTPServer):
    """A handshake-only client (the collector reads the certificate and hangs
    up) makes the stdlib server print a BrokenPipeError traceback.  That is
    expected here, and it would drown the rung output."""

    def handle_error(self, request, client_address):
        pass


class TlsStub:
    """A real TLS server on 127.0.0.1, addressed as `localhost`."""

    def __init__(self, key, crt):
        self.server = _QuietServer(("127.0.0.1", 0), _Quiet)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(crt), keyfile=str(key))
        self.server.socket = context.wrap_socket(self.server.socket,
                                                 server_side=True)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05},
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def fresh_guard(**env):
    """Reload tlsguard's cached trust source under a chosen environment."""
    for name in (tlsguard.OVERRIDE_BUNDLE_ENV, tlsguard.OVERRIDE_SYSTEM_ENV,
                 tlsguard.OVERRIDE_LEAF_ENV, "SSL_CERT_FILE", "SSL_CERT_DIR"):
        os.environ.pop(name, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})
    tlsguard.reset_cached_state()
    common._SAFE_OPENER = None


# --------------------------------------------------------------------------
# T1/T2
# --------------------------------------------------------------------------


def section_bundle():
    print("T1  shipped trust bundle")
    fresh_guard()
    source = tlsguard.trust_source()
    check("T1a trust source is the bundle this repo ships",
          source["kind"] == "vendored"
          and source["path"] == str(tlsguard.BUNDLE_PATH), source)
    text = tlsguard.BUNDLE_PATH.read_text()
    digests = tlsguard.bundle_root_digests()
    check("T1b the file digest is pinned and covers every root",
          source["sha256"] == tlsguard.BUNDLE_SHA256
          and len(digests) == source["certificates"], source)
    claimed = [line.split("sha256:")[1].strip()
               for line in text.splitlines() if "sha256:" in line]
    check("T1c each PEM header's claimed sha256 equals the bytes below it",
          claimed == digests,
          "claimed=%s\nactual=%s" % (claimed[:3], digests[:3]))
    if not OPENSSL:
        skip("T1d-T1f root audit", "no openssl on PATH")
        return
    subjects, issuers, expiries = [], [], []
    for chunk in text.split("-----BEGIN CERTIFICATE-----")[1:]:
        pem = ("-----BEGIN CERTIFICATE-----"
               + chunk.split("-----END CERTIFICATE-----")[0]
               + "-----END CERTIFICATE-----\n")
        out = _run([OPENSSL, "x509", "-noout", "-subject", "-issuer",
                    "-enddate"], input=pem).stdout.splitlines()
        subjects.append(out[0].split("subject=")[1].strip())
        issuers.append(out[1].split("issuer=")[1].strip())
        expiries.append(out[2].split("notAfter=")[1].strip())
    check("T1d every shipped root is self-signed (issuer == subject)",
          subjects == issuers, list(zip(subjects, issuers))[:2])
    cns = [s.split("CN=")[-1].strip() for s in subjects]
    undeclared = [c for c in cns
                  if c not in tlsguard.EXPECTED_ROOT_SUBJECT_CNS]
    check("T1e no undeclared root: every CN is in EXPECTED_ROOT_SUBJECT_CNS",
          not undeclared, undeclared)
    year = time.time() + 365 * 24 * 3600
    soon = [(c, e) for c, e in zip(cns, expiries)
            if time.mktime(time.strptime(e, "%b %d %H:%M:%S %Y %Z")) < year]
    check("T1f no shipped root expires within a year", not soon, soon)

    print("T2  the explicit context")
    context = tlsguard.explicit_ssl_context()
    check("T2a check_hostname and CERT_REQUIRED are both on",
          context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED,
          (context.check_hostname, context.verify_mode))
    check("T2b TLS 1.2 floor",
          context.minimum_version >= ssl.TLSVersion.TLSv1_2,
          context.minimum_version)


# --------------------------------------------------------------------------
# T3-T8
# --------------------------------------------------------------------------


def section_negatives(tmp):
    good_ca = make_ca(tmp, "good")
    other_ca = make_ca(tmp, "other")
    good = make_leaf(tmp, "good", "localhost", "localhost", ca=good_ca)
    selfsigned = make_leaf(tmp, "selfsigned", "localhost", "localhost")
    mismatch = make_leaf(tmp, "mismatch", "wrong.invalid", "wrong.invalid",
                         ca=good_ca)

    print("T3  an ambient trust variable must NOT widen trust (gap 1)")
    stub = TlsStub(*good)
    try:
        fresh_guard(SSL_CERT_FILE=str(good_ca[1]))
        default_ok = True
        try:
            with socket.create_connection(("localhost", stub.port),
                                          timeout=5) as raw:
                ssl.create_default_context().wrap_socket(
                    raw, server_hostname="localhost").close()
        except ssl.SSLError as exc:
            default_ok = exc
        check("T3a SSL_CERT_FILE really does widen python's DEFAULT context",
              default_ok is True, default_ok)
        evidence = tlsguard.collect_peer_evidence(
            "localhost", port=stub.port, timeout=5,
            cafile=tlsguard.trust_source()["path"])
        verdict = tlsguard.evaluate_peer_evidence(
            evidence, host="localhost", host_id="vast:68004")
        check("T3b our context REFUSES it anyway: verification is not ambient",
              not verdict["ok"]
              and any(f["code"] == "TLS-PEER-CHAIN-NOT-OURS"
                      for f in verdict["failures"]), verdict["failures"])
        check("T3c the ambient variable is DISCLOSED, not silently ignored",
              any("SSL_CERT_FILE" in d for d in verdict["disclosures"]),
              verdict["disclosures"])
        remedy = " ".join(f["remedy"] for f in verdict["failures"])
        check("T3d the refusal separates stale-bundle from lying-host and "
              "names both remedies",
              "STALE" in remedy and tlsguard.OVERRIDE_BUNDLE_ENV in remedy
              and "destroy" in remedy, remedy[:200])

        print("T4  safe_urlopen end to end")
        import urllib.error
        import urllib.request
        fresh_guard(SSL_CERT_FILE=str(good_ca[1]))
        request = urllib.request.Request("https://localhost:%d/" % stub.port)
        try:
            common.safe_urlopen(request, timeout=5)
            outcome = "ACCEPTED"
        except urllib.error.URLError as exc:
            outcome = exc.reason
        check("T4a safe_urlopen refuses a peer only the ambient store trusts",
              isinstance(outcome, ssl.SSLCertVerificationError), outcome)
        handlers = [h.__class__.__name__ for h in
                    (common._SAFE_OPENER.handlers if common._SAFE_OPENER else [])]
        check("T4b the auth-stripping redirect handler is still installed",
              "NoCrossOriginAuth" in handlers, handlers)
        fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(good_ca[1])})
        try:
            status = common.safe_urlopen(request, timeout=5).status
        except urllib.error.URLError as exc:
            status = exc.reason
        check("T4c with the DISCLOSED override the same peer is accepted",
              status == 200, status)
    finally:
        stub.close()

    print("T5  a self-signed peer is refused")
    stub = TlsStub(*selfsigned)
    try:
        fresh_guard()
        evidence = tlsguard.collect_peer_evidence(
            "localhost", port=stub.port, timeout=5,
            cafile=tlsguard.trust_source()["path"])
        verdict = tlsguard.evaluate_peer_evidence(
            evidence, host="localhost", host_id="vast:68004")
        codes = [f["code"] for f in verdict["failures"]]
        check("T5a refused", not verdict["ok"] and codes, codes)
        check("T5b the presented (unverifiable) leaf is still recorded",
              (evidence.get("presented_leaf") or {}).get("spki_sha256"),
              evidence.get("presented_leaf"))
        text = " ".join([f["message"] for f in verdict["failures"]]
                        + [f["remedy"] for f in verdict["failures"]])
        check("T5c the refusal names the host id and the next thing to try",
              "vast:68004" in text and "destroy" in text, text[:200])
    finally:
        stub.close()

    print("T6  a hostname mismatch is refused (the 2026-09-05 shape)")
    stub = TlsStub(*mismatch)
    try:
        fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(good_ca[1])})
        evidence = tlsguard.collect_peer_evidence(
            "localhost", port=stub.port, timeout=5,
            cafile=tlsguard.trust_source()["path"])
        verdict = tlsguard.evaluate_peer_evidence(
            evidence, host="localhost", host_id="vast:68004")
        codes = [f["code"] for f in verdict["failures"]]
        check("T6a refused even though the chain DOES terminate in the trusted "
              "CA: the NAME is wrong", not verdict["ok"], codes)
        check("T6b the mismatch is named as such",
              "TLS-PEER-HOSTNAME-MISMATCH" in codes
              or any("does not cover" in f["message"]
                     for f in verdict["failures"]), verdict["failures"])
        check("T6c the wrong name is recorded for forensics",
              "wrong.invalid" in json.dumps(verdict["peer"]), verdict["peer"])
        check("T6d hostname matching is not fooled by a suffix or a wildcard "
              "over-reach",
              not tlsguard.host_matches_names("huggingface.co",
                                              ["evilhuggingface.co"])
              and not tlsguard.host_matches_names("a.b.huggingface.co",
                                                  ["*.huggingface.co"])
              and tlsguard.host_matches_names("cdn.huggingface.co",
                                              ["*.huggingface.co"]))
    finally:
        stub.close()

    print("T7  every override is a disclosure, never a silent fallback")
    fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(other_ca[1])})
    source = tlsguard.trust_source()
    check("T7a an operator bundle is recorded with its own digest",
          source["kind"] == "operator-bundle" and source["sha256"], source)
    check("T7b and it announces itself, naming the variable",
          any(tlsguard.OVERRIDE_BUNDLE_ENV in d for d in source["disclosures"]),
          source["disclosures"])
    fresh_guard(**{tlsguard.OVERRIDE_SYSTEM_ENV: "1"})
    source = tlsguard.trust_source()
    check("T7c the ambient-store escape hatch is also a disclosure",
          source["kind"] == "system-store"
          and any(tlsguard.OVERRIDE_SYSTEM_ENV in d
                  for d in source["disclosures"]), source)
    fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(tmp / "absent.pem")})
    try:
        tlsguard.trust_source()
        check("T7d a missing override bundle refuses", False, "no refusal")
    except tlsguard.TlsRefusal as exc:
        check("T7d a missing override bundle refuses with a remedy",
              exc.code == "TLS-TRUST-OVERRIDE-MISSING" and exc.advice, exc.code)

    print("T8  a tampered bundle refuses")
    copy = tmp / "tampered-roots.pem"
    copy.write_text(tlsguard.BUNDLE_PATH.read_text()
                    + "\n# an extra root would go here\n")
    original = tlsguard.BUNDLE_PATH
    try:
        tlsguard.BUNDLE_PATH = copy
        fresh_guard()
        try:
            tlsguard.trust_source()
            check("T8a refused", False, "a tampered bundle was accepted")
        except tlsguard.TlsRefusal as exc:
            joined = " ".join(exc.advice)
            check("T8a refused, digest mismatch named",
                  exc.code == "TLS-TRUST-BUNDLE-DIGEST", exc.code)
            check("T8b the remedy covers BOTH deliberate rotation and host "
                  "tampering, with the recomputed digest",
                  "git checkout" in joined and "BUNDLE_SHA256" in joined
                  and "destroy" in joined, joined[:240])
    finally:
        tlsguard.BUNDLE_PATH = original
        fresh_guard()


# --------------------------------------------------------------------------
# T9
# --------------------------------------------------------------------------


def section_verdicts():
    print("T9  reachability is not identity")
    fresh_guard()
    attested = {
        "host": "huggingface.co", "tls_ok": True,
        "resolved_addresses": ["1.2.3.4"], "chain_depth": 3,
        "chain_depth_source": "ssl.get_verified_chain",
        "leaf": {"subject_cn": "huggingface.co",
                 "issuer_cn": "Amazon RSA 2048 M01",
                 "spki_sha256": "a" * 64, "san_dns": ["huggingface.co"]},
        "bundle_verify": {"ok": True}, "bundle_path": "/x/tls-roots.pem",
    }
    timed_out = {"host": "huggingface.co", "tls_ok": False,
                 "error_class": "TimeoutError", "error_text": "timed out",
                 "resolved_addresses": ["1.2.3.4"], "chain_depth": None}
    verdict = tlsguard.evaluate_peer_evidence(timed_out, host="huggingface.co",
                                              host_id="vast:1")
    check("T9a a timeout is TLS-UNREACHABLE and retryable",
          verdict["verdict"] == "unreachable" and verdict["retryable"]
          and verdict["failures"][0]["code"] == "TLS-UNREACHABLE",
          verdict["failures"])
    check("T9b and it does not accuse the host",
          "destroy this instance" not in verdict["failures"][0]["message"]
          and "interception" not in verdict["failures"][0]["message"],
          verdict["failures"][0]["message"])
    rate_limited = tlsguard.evaluate_peer_evidence(
        dict(attested, http_status=429), host="huggingface.co",
        host_id="vast:1")
    check("T9c HTTP 429 over a verified peer stays ATTESTED, as a disclosure",
          rate_limited["ok"]
          and any("429" in d for d in rate_limited["disclosures"]),
          rate_limited["disclosures"])
    eof = {"host": "huggingface.co", "tls_ok": False,
           "error_class": "SSLEOFError",
           "error_text": "[SSL: UNEXPECTED_EOF_WHILE_READING]",
           "resolved_addresses": ["1.2.3.4"], "chain_depth": None}
    verdict = tlsguard.evaluate_peer_evidence(eof, host="huggingface.co",
                                              host_id="vast:68004")
    remedy = verdict["failures"][0]["remedy"]
    # MEASURED 2026-09-06: machine 68004, the host that produced this exact
    # error on 2026-09-05, has NO interceptor -- its DNS answers are forged on
    # the path to 1.1.1.1. So the guard must refuse the credential WITHOUT
    # accusing the host's TLS, and must name the measurement that tells the
    # three candidate causes apart.
    check("T9d an UNEXPECTED_EOF refuses the credential but does NOT accuse "
          "the host of interception",
          verdict["verdict"] == "refused"
          and verdict["failures"][0]["code"] == "TLS-PEER-UNVERIFIED",
          verdict["failures"])
    check("T9e and it names all THREE candidate causes plus the discriminating "
          "measurement",
          "forged DNS" in remedy and "interception proxy" in remedy
          and "cache" in remedy and "compare leaf digests" in remedy,
          remedy)
    dns_forged = dict(eof, pinned_address_dials=[
        {"address": "3.168.73.106", "ok": True,
         "leaf": {"subject_cn": "huggingface.co", "spki_sha256": "a" * 64,
                  "san_dns": ["huggingface.co"]}}])
    verdict = tlsguard.evaluate_peer_evidence(dns_forged, host="huggingface.co",
                                              host_id="vast:68004")
    remedy = verdict["failures"][0]["remedy"]
    check("T9f when a controller-verified ADDRESS works and the resolved one "
          "does not, the verdict blames RESOLUTION, not the host",
          verdict["failures"][0]["code"] == "TLS-RESOLUTION-SUSPECT"
          and "host's TLS is not implicated"
          in verdict["failures"][0]["message"], verdict["failures"])
    check("T9g and it explicitly says not to report the operator, while still "
          "refusing the credential",
          "do NOT report the host operator" in remedy
          and "do NOT use a credential" in remedy and not verdict["ok"], remedy)
    diverged = tlsguard.evaluate_peer_evidence(
        attested, host="huggingface.co", host_id="vast:1",
        reference={"spki_sha256": "b" * 64})
    check("T9h a leaf differing from the controller-verified leaf is refused",
          any(f["code"] == "TLS-PEER-LEAF-DIVERGENCE"
              for f in diverged["failures"]), diverged["failures"])
    check("T9i that refusal names the override that would accept it",
          tlsguard.OVERRIDE_LEAF_ENV in diverged["failures"][0]["remedy"],
          diverged["failures"][0]["remedy"])
    os.environ[tlsguard.OVERRIDE_LEAF_ENV] = "1"
    try:
        allowed = tlsguard.evaluate_peer_evidence(
            attested, host="huggingface.co", host_id="vast:1",
            reference={"spki_sha256": "b" * 64})
    finally:
        os.environ.pop(tlsguard.OVERRIDE_LEAF_ENV, None)
    check("T9j with the override it is attested AND disclosed",
          allowed["ok"] and any(tlsguard.OVERRIDE_LEAF_ENV in d
                                for d in allowed["disclosures"]),
          allowed["disclosures"])


# --------------------------------------------------------------------------
# T10-T12
# --------------------------------------------------------------------------


def section_attestation(tmp):
    print("T10 attest_before_credential: ordering, and refusal before a token")
    ca = make_ca(tmp, "attest")
    good = make_leaf(tmp, "attest-good", "localhost", "localhost", ca=ca)
    rogue_ca = make_ca(tmp, "rogue")
    rogue = make_leaf(tmp, "attest-rogue", "localhost", "localhost", ca=rogue_ca)

    honest = TlsStub(*good)
    try:
        fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(ca[1])})
        port = honest.port

        def exec_honest(machine_id, command, timeout=None):
            return json.dumps(tlsguard.collect_peer_evidence(
                "localhost", port=port, timeout=5, cafile=str(ca[1]))) + "\n"

        document = tlsguard.attest_before_credential(
            None, "pod-1", host_id="vast:150014", hosts=("localhost",),
            port=port, exec_stdout=exec_honest, timeout=20)
        check("T10a a good box attests",
              document["ok"] and document["verdict"] == "attested",
              document.get("failures"))
        check("T10b the document records the ordering claim explicitly",
              "BEFORE any credential" in document["ordering"],
              document["ordering"])
        check("T10c both observations are recorded, and they agree",
              document["controller_observations"]["localhost"]["leaf_spki_sha256"]
              == document["pod_observations"]["localhost"]["leaf_spki_sha256"],
              document["pod_observations"])
        check("T10d the guarantee's end is stated in the document",
              "root" in document["guarantee_ends"]
              and "forge" in document["guarantee_ends"],
              document["guarantee_ends"])
        check("T10e the pod's CA bundle digests are recorded",
              isinstance(document["pod_ca_bundle_digests"]["localhost"], list),
              document["pod_ca_bundle_digests"])
        check("T10f no credential anywhere in the document",
              FIXTURE_TOKEN not in json.dumps(document)
              and "hf_" not in json.dumps(document))
    finally:
        honest.close()

    liar = TlsStub(*rogue)
    honest2 = TlsStub(*good)
    try:
        fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(ca[1])})
        rogue_port = liar.port

        def exec_lying(machine_id, command, timeout=None):
            # the controller verifies the honest peer; the BOX reports a
            # different leaf -- the re-signing-proxy signature.
            return json.dumps(tlsguard.collect_peer_evidence(
                "localhost", port=rogue_port, timeout=5)) + "\n"

        try:
            tlsguard.attest_before_credential(
                None, "pod-2", host_id="vast:68004", hosts=("localhost",),
                port=honest2.port, exec_stdout=exec_lying, timeout=20)
            check("T10g a lying box is refused", False, "attested a liar")
        except tlsguard.TlsRefusal as exc:
            joined = " ".join(exc.advice)
            check("T10g a lying box is refused", exc.code.startswith("TLS-PEER-"),
                  exc.code)
            check("T10h the refusal states no credential was written yet",
                  "no HF credential has been written" in joined, joined[:200])
            check("T10i and it names the host id to destroy",
                  "vast:68004" in exc.reason + joined, exc.reason)
    finally:
        liar.close()
        honest2.close()

    stub = TlsStub(*good)
    try:
        fresh_guard(**{tlsguard.OVERRIDE_BUNDLE_ENV: str(ca[1])})
        try:
            tlsguard.attest_before_credential(
                None, "pod-3", host_id="vast:7", hosts=("localhost",),
                port=stub.port,
                exec_stdout=lambda *a, **k: "bash: python3: not found\n",
                timeout=20)
            check("T10j unparseable evidence is refused", False, "accepted")
        except tlsguard.TlsRefusal as exc:
            check("T10j unparseable evidence is refused, not assumed good",
                  exc.code == "TLS-PEER-NO-EVIDENCE", exc.code)
            check("T10k with a remedy naming destroy-and-recreate",
                  any("destroy" in a for a in exc.advice), exc.advice)

        print("T11 ONE collector implementation, in-process and as text")
        inproc = tlsguard.collect_peer_evidence("localhost", port=stub.port,
                                                timeout=5, cafile=str(ca[1]))
        command = tlsguard.remote_collector_command(
            "localhost", port=stub.port, cafile=str(ca[1]),
            python=sys.executable)
        out = subprocess.run(["bash", "-c", command], capture_output=True,
                             text=True, timeout=60)
        lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
        remote = json.loads(lines[-1]) if lines else {}
        check("T11a the shipped script runs under a bare interpreter and emits "
              "one JSON object",
              isinstance(remote, dict) and remote.get("host"),
              out.stderr[-300:])
        check("T11b it reports the same field set as the in-process call",
              set(inproc) <= set(remote), sorted(set(inproc) - set(remote)))
        check("T11c and the same peer identity for the same peer",
              (remote.get("leaf") or {}).get("spki_sha256")
              == (inproc.get("leaf") or {}).get("spki_sha256"))
        # Parsed, not grepped: a comment saying "no Authorization header" must
        # not satisfy a rung about what the code actually does.
        import ast
        tree = ast.parse(tlsguard.collector_script_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        allowed = {"base64", "hashlib", "json", "os", "socket", "ssl", "sys",
                   "http.client"}
        check("T11d the collector imports ONLY the stdlib modules it declares",
              imported <= allowed, sorted(imported - allowed))
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        credentialish = [s[:40] for s in literals
                         if "TOKEN" in s.upper() and "HF_ENDPOINT" not in s
                         or "AUTHORIZATION" in s.upper()]
        check("T11e no code path in the collector names a credential",
              not credentialish, credentialish)

        print("T12 the attestation document as written to a receipt")
        target = tmp / "receipts" / "tls-peer-setup.json"
        verdict = tlsguard.attest_local_peer("localhost", port=stub.port,
                                             timeout=5, host_id="vast:150014")
        digest = tlsguard.write_attestation(verdict, target)
        written = json.loads(target.read_text())
        check("T12a written atomically and self-describing",
              target.exists() and written["schema"] == tlsguard.SCHEMA
              and len(digest) == 64, written.get("schema"))
        check("T12b the peer evidence a receipt needs is present",
              all(written["peer"][k] for k in
                  ("subject_cn", "issuer_cn", "leaf_spki_sha256",
                   "leaf_der_sha256")), written["peer"])
        check("T12c no temporary file is left behind",
              [p.name for p in target.parent.iterdir()] == [target.name],
              [p.name for p in target.parent.iterdir()])
    finally:
        stub.close()


# --------------------------------------------------------------------------
# T13
# --------------------------------------------------------------------------


def section_payload_guard():
    print("T13 no credential in a provider create body (all four adapters)")
    providers = ("runpod", "vast", "lambda", "jarvislabs")
    shapes = {
        "env dict": {"env": {"HF_TOKEN": FIXTURE_TOKEN}},
        "docker -e text": "-e HF_TOKEN=%s -e FIDELITY_PANEL_ID=panel--fruit"
                          % FIXTURE_TOKEN,
        "onstart script": {"onstart": "export HF_TOKEN=%s\nbash run.sh"
                                      % FIXTURE_TOKEN},
        "nested body": {"instance": {"config": [
            {"user_data": "HF_TOKEN=%s" % FIXTURE_TOKEN}]}},
        "bearer header": {"headers": {"Authorization": "Bearer %s"
                                                       % FIXTURE_TOKEN}},
    }
    for provider in providers:
        for label, payload in shapes.items():
            refusal = None
            try:
                tlsguard.refuse_credential_in_provider_payload(
                    payload, provider=provider,
                    field="env_str" if isinstance(payload, str) else None)
            except tlsguard.TlsRefusal as exc:
                refusal = exc
            if refusal is None:
                check("T13 %s/%s refuses" % (provider, label), False,
                      "transmitted a credential")
                continue
            blob = refusal.reason + " " + " ".join(refusal.advice)
            check("T13 %s/%s refuses, naming provider-persistence"
                  % (provider, label),
                  refusal.code == "PROVIDER-PAYLOAD-CREDENTIAL"
                  and "provider-persisted" in refusal.reason, refusal.code)
            check("T13 %s/%s refusal does NOT echo the credential"
                  % (provider, label),
                  FIXTURE_TOKEN not in blob and "hf_SS" not in blob, blob[:160])
            check("T13 %s/%s remedy names the next thing to try"
                  % (provider, label),
                  "0600 file" in blob and "PUBLIC artifacts" in blob, blob[:160])
    legitimate = {
        "env": {"HF_TOKEN_PATH": "/root/.secrets/hf_token",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_ENDPOINT": "https://huggingface.co",
                "FIDELITY_PANEL_ID": "panel--fruit-1a2b3c4d",
                "FIDELITY_FS_ROOT": "/workspace/fidelity"},
        "onstart": "python3 /workspace/fidelity/bin/container_entry.py",
        "image": "ghcr.io/malaiwah/quant-fidelity-measure@sha256:" + "a" * 64,
    }
    for provider in providers:
        try:
            tlsguard.refuse_credential_in_provider_payload(
                legitimate, provider=provider)
            check("T13 %s legitimate payload passes through" % provider, True)
        except tlsguard.TlsRefusal as exc:
            check("T13 %s legitimate payload passes through" % provider, False,
                  exc.reason)
    findings = tlsguard.credential_findings({"env": {"HF_TOKEN": FIXTURE_TOKEN}})
    check("T13 every finding names a path, a shape or a character COUNT, and "
          "never the value",
          findings
          and all("characters" in f or "contains a" in f for f in findings)
          and all(FIXTURE_TOKEN not in f and FIXTURE_TOKEN[3:12] not in f
                  for f in findings), findings)


def section_pinned_ssh(tmp):
    """T15: the verifying transport for a provider whose CLI verifies nothing.

    `jl` passes StrictHostKeyChecking=no AND UserKnownHostsFile=/dev/null, so
    a pinned fingerprint buys nothing there until an invocation of OURS does
    the verifying.  These rungs prove the ordering is structural: no ssh
    process can spawn before the key is authenticated.
    """
    print("T15 verifying pinned-endpoint SSH transport")
    from fidelity import sshbase

    key = "AAAAC3NzaC1lZDI1NTE5AAAAI" + "A" * 43
    entry = "[example.invalid]:2222 ssh-ed25519 %s\n" % key
    fingerprint = "SHA256:" + "b" * 43

    class _Scanned(sshbase.PinnedEndpointSSH):
        """Only the keyscan is stubbed: it is the one step that needs a live
        sshd.  Everything judged here -- the refusal ordering, the comparison,
        the file mode -- is the real implementation."""

        def scan_host_key(self, machine_id=None):
            return {"algorithm": "ssh-ed25519", "fingerprint": self.scanned,
                    "host": "example.invalid", "port": 2222,
                    "known_hosts_entry": entry}

    transport = _Scanned("example.invalid", 2222, dry=True)
    transport.scanned = fingerprint
    try:
        transport._ssh_opts()
        check("T15a ssh REFUSES to spawn before the host key is authenticated",
              False, "built ssh options with no known_hosts")
    except sshbase.JLError as exc:
        check("T15a ssh REFUSES to spawn before the host key is authenticated",
              "has not been authenticated" in str(exc), str(exc))

    liar = _Scanned("example.invalid", 2222, dry=True)
    liar.scanned = "SHA256:" + "c" * 43
    try:
        liar.attest_endpoint(fingerprint,
                             known_hosts=tmp / "kh-mismatch",
                             pin_source="construction")
        check("T15b a keyscan that differs from the pin is refused", False,
              "accepted a mismatched key")
    except sshbase.JLError as exc:
        check("T15b a keyscan that differs from the pin is refused",
              "differs" in str(exc), str(exc))
    check("T15c and no known_hosts file was written by the refused attempt",
          not (tmp / "kh-mismatch").exists())

    known_hosts = tmp / "kh-good"
    proof = transport.attest_endpoint(fingerprint, known_hosts=known_hosts,
                                      pin_source="construction")
    mode = oct(known_hosts.stat().st_mode & 0o777)
    check("T15d a matching pin writes an owner-0600 per-attempt known_hosts",
          known_hosts.is_file() and mode == "0o600", mode)
    check("T15e and only then does ssh get its options, pinned to that file",
          "StrictHostKeyChecking=yes" in transport._ssh_opts()
          and str(known_hosts) in " ".join(transport._ssh_opts()))
    check("T15f the proof says what it can prove, and what it cannot",
          proof["channel_verifies_host_key"] is True
          and proof["pin_source"] == "construction"
          and "not the instance's" in proof["does_not_attest"], proof)
    try:
        transport.attest_endpoint(fingerprint, known_hosts=known_hosts,
                                  pin_source="guesswork")
        check("T15g an undeclared pin source is refused", False, "accepted")
    except sshbase.JLError as exc:
        check("T15g an undeclared pin source is refused, because 'where did "
              "this fingerprint come from' is the whole question",
              "pin_source" in str(exc), str(exc))


# --------------------------------------------------------------------------
# T14
# --------------------------------------------------------------------------


def section_network():
    print("T14 live anchors (network)")
    fresh_guard()
    hosts = [tlsguard.HUB_HOST] + sorted(set(tlsguard.PROVIDER_API_HOSTS.values()))
    roots = set(tlsguard.trust_source()["root_der_sha256"])
    for host in hosts:
        try:
            verdict = tlsguard.attest_local_peer(host, timeout=10,
                                                 host_id="controller")
        except OSError as exc:
            skip("T14 %s" % host, "no network (%s)" % exc.__class__.__name__)
            continue
        if not verdict["ok"] and verdict["retryable"]:
            skip("T14 %s" % host, "unreachable")
            continue
        check("T14 %s verifies against the bundle we ship" % host,
              verdict["ok"], {"failures": verdict["failures"],
                              "peer": verdict["peer"]})
        chain = verdict["peer"]["chain_der_sha256"] or []
        if chain:
            check("T14 %s chain terminates in a shipped root" % host,
                  chain[-1] in roots, chain[-1])
        if host == tlsguard.HUB_HOST and verdict["ok"]:
            check("T14 the Hub's identity is what we expect",
                  verdict["peer"]["issuer_org"] == "Amazon"
                  and tlsguard.host_matches_names(host,
                                                  verdict["peer"]["san_dns"]),
                  verdict["peer"])


def main():
    print("selftest_tlsguard: explicit TLS trust, peer attestation, payload guard")
    print()
    section_bundle()
    print()
    section_verdicts()
    print()
    if not OPENSSL:
        skip("T3-T8 and T10-T12 (local TLS negatives)", "no openssl on PATH")
    else:
        with tempfile.TemporaryDirectory(prefix="tlsguard-selftest-") as td:
            tmp = Path(td)
            section_negatives(tmp)
            print()
            section_attestation(tmp)
            print()
            section_pinned_ssh(tmp)
    print()
    section_payload_guard()
    print()
    section_network()
    fresh_guard()
    print()
    if SKIPPED:
        print("skipped: %d (%s)" % (len(SKIPPED), ", ".join(SKIPPED)))
    if FAILED:
        print("FAILED: %d" % len(FAILED))
        for label in FAILED:
            print("  - %s" % label)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
