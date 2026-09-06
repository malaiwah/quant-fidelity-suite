#!/usr/bin/env python3
"""Explicit TLS trust, peer attestation BEFORE a credential exists, and the
one guard that refuses a credential in a provider request body.

WHY THIS EXISTS
---------------
2026-09-05, docs/CLOUD-RECIPES.md: a rented Vast host presented a certificate
for `huggingface.co` with a HOSTNAME MISMATCH and then
`ssl.SSLEOFError: UNEXPECTED_EOF_WHILE_READING`.  That is the signature of a
man-in-the-middle TLS proxy, on a box this suite was about to write an HF token
onto.  The run failed at setup instead of leaking, because verification was on
-- and the log order shows how close it was: `HF token installed 0600 file` at
19:57:39, THEN `stage setup starting`, THEN the SSL error.  The credential was
already on disk before anything had looked at the box's network.

Four properties are missing from a default client, and this module supplies all
four.

1. NON-AMBIENT TRUST.  `ssl.create_default_context()` (and therefore
   `urllib.request.build_opener()` with no context) calls `load_default_certs`,
   which reads the machine's store AND honours `SSL_CERT_FILE`/`SSL_CERT_DIR`.
   Where somebody else is root, "the certificate verified" can mean "the store
   the host controls said yes".  `explicit_ssl_context()` loads
   `tls-roots.pem` -- shipped in this repo, digest-pinned below -- and nothing
   else.  `load_verify_locations(cafile=...)` does not consult those variables,
   so no environment variable can WIDEN our trust; if one is set at all it is
   disclosed (the precedent is the controller's ambient `HF_ENDPOINT` warning).

2. ATTESTATION BEFORE THE CREDENTIAL.  `attest_before_credential()` runs the
   collector on the rented box over the provider's exec channel and refuses
   unless the box's own TLS to the Hub verifies, the hostname matches, and the
   chain terminates in a root we ship.  Call it BEFORE the token transport.

3. PROVIDER-API IDENTITY.  RunPod host-key attribution is trust-on-first-use
   anchored in an ed25519 fingerprint read out of `api.runpod.io` over TLS
   before first contact.  That anchor is only as strong as our TLS to the
   provider API -- an intercepted provider API silently poisons host-key
   attribution, and therefore result attribution, for every row measured
   through it.  So provider API hosts are first-class attestation targets here.

4. NO CREDENTIAL IN A CREATE BODY.  `refuse_credential_in_provider_payload()`
   is the pure-function guard every provider adapter calls before it transmits.
   A credential in a create body (Vast `env`/`onstart`/`docker_cmd`) is
   provider-persisted before the instance exists, so there is nothing to
   attest and no ordering that helps -- the only fix is to refuse at the
   adapter, which is the last place that can tell.  RunPod already refused it
   (rungs RP7/RP7b in bin/selftest_root_publish.py); the defect was that the
   rung was never made per-provider, so this is one implementation for four.

WHERE THE GUARANTEE ENDS.  A Vast/Lambda host has root.  Root can lie to a
collector that runs on it: replay a real chain, fabricate the JSON.  So this
module buys exactly three things and claims nothing more: (a) a passive or
naive interception proxy fails CLOSED -- it cannot produce a chain terminating
in our roots, and its re-signed leaf's SPKI differs from the leaf the
CONTROLLER verified against our own bundle; (b) whatever the box reported is
RECORDED, so tampering is attributable afterwards; (c) no credential is
transported until (a) and (b) have happened.  A root-privileged host that
forges consistent evidence is not detected by software running on that host --
destroy-and-recreate is the response, and the provider host id is quoted in
every refusal so the operator knows which box to destroy.

A hostname mismatch also has a BENIGN explanation -- a transparent Hugging
Face cache misconfigured by a well-meaning host operator is identical to a
harvester at the certificate layer.  So a refusal here says "this box's TLS to
the Hub is not the Hub's" and never "this operator steals credentials"; the
evidence recorded (issuer, SAN, SPKI, chain) is what tells those apart later.

DEPENDENCIES, NAMED IN FULL (AGENTS.md: "a guard must name every dependency it
guards").  Controller side: stdlib `ssl`, `socket`, `hashlib`, `base64`,
`json`, `re`, `argparse`, `http.client`, `subprocess`, plus this repo's
`tls-roots.pem`.  Box side: a `python3` with the stdlib `ssl` module compiled
in -- nothing else.  No `certifi`, no `requests`, no `huggingface_hub`, no
venv, no torch, no repo checkout.  The collector is shipped as SOURCE TEXT so
it also runs where none of this repo exists yet.

NEVER put a credential in here.  The evidence is about the PEER, never about
the credential: no token bytes are read, logged, receipted or included in any
finding message produced by this file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA = "fidelity.tls-attestation/1"

HUB_HOST = "huggingface.co"

# Every host whose answer we treat as an AUTHENTICATION ANCHOR.  MEASURED
# 2026-09-06: api.runpod.io and cloud.lambdalabs.com terminate in GTS Root R4
# via 'WE1'; console.vast.ai in Amazon Root CA 1; api.jarvislabs.net in ISRG
# Root X2 (`api.jarvislabs.ai` does not answer -- it timed out from here, and
# the JL path goes through the vendor CLI rather than a host we authenticate,
# which is why JL cannot host a credential-bearing run at all).
# RunPod also serves rest.runpod.io; api.runpod.io is the one listed because it
# is where the ed25519 host-key fingerprint is read from, so it is the anchor
# that result ATTRIBUTION rests on.
PROVIDER_API_HOSTS = {
    "runpod": "api.runpod.io",
    "vast": "console.vast.ai",
    "lambda": "cloud.lambdalabs.com",
    "jarvislabs": "api.jarvislabs.net",
}

BUNDLE_PATH = Path(__file__).resolve().parent / "tls-roots.pem"
BUNDLE_SHA256 = "4f933fbcdb16876a13a574d570b372a7c0ea81e8e978c993d4d1400748253dfb"

# Adding a root is a deliberate TWO-file change: the PEM and this list.  A root
# that appears in the bundle without appearing here is refused by
# bin/selftest_tlsguard.py, so a copied-in system store cannot widen trust
# quietly.
EXPECTED_ROOT_SUBJECT_CNS = (
    "Amazon Root CA 1", "Amazon Root CA 2", "Amazon Root CA 3",
    "Amazon Root CA 4", "ISRG Root X1", "ISRG Root X2",
    "GTS Root R1", "GTS Root R3", "GTS Root R4",
    "DigiCert Global Root G2", "DigiCert Global Root G3",
    "Starfield Services Root Certificate Authority - G2",
)

OVERRIDE_BUNDLE_ENV = "FIDELITY_TLS_TRUST_BUNDLE"
OVERRIDE_SYSTEM_ENV = "FIDELITY_TLS_ALLOW_SYSTEM_TRUST"
OVERRIDE_LEAF_ENV = "FIDELITY_TLS_ACCEPT_LEAF_DIVERGENCE"

AMBIENT_TRUST_VARS = (
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS", "PYTHONHTTPSVERIFY", "SSLKEYLOGFILE",
    "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HF_ENDPOINT",
)


class TlsRefusal(Exception):
    """An expected invalid state is a refusal, never a guess.

    `reason`/`advice` are exactly the controller's `Refusal(reason, advice)`
    arguments, so a controller writes:

        except tlsguard.TlsRefusal as exc:
            raise Refusal(exc.reason, exc.advice)

    `code` separates "this host is lying" (`TLS-PEER-*`) from "our bundle is
    stale" (`TLS-TRUST-*`) from "we could not reach anything"
    (`TLS-UNREACHABLE`).  One generic remedy for several distinct causes points
    an operator at the wrong fix -- the trap of a 429 that printed
    `--token-file` advice when the answer was "wait".
    """

    def __init__(self, code: str, reason: str, advice: Sequence[str],
                 *, evidence: Optional[Dict[str, Any]] = None,
                 retryable: bool = False) -> None:
        super().__init__("%s: %s" % (code, reason))
        self.code = code
        self.reason = reason
        self.advice = list(advice)
        self.evidence = evidence or {}
        self.retryable = retryable


# --------------------------------------------------------------------------
# The collector.  ONE implementation of "what does this peer look like",
# shipped as source text so it runs three ways: in this process, over a
# provider exec channel with nothing of ours on the box, and from the pod's own
# stage script.  A second implementation of a security property is how the two
# drift, and the drifting one is always the one nobody tests.
# --------------------------------------------------------------------------

_COLLECTOR_SOURCE = r'''
"""fidelity TLS peer evidence collector.  Stdlib only: ssl, socket, hashlib,
base64, json, os, sys, http.client.  Collects; never judges; never reads a
credential.  Writes one JSON object to stdout."""
import base64, hashlib, json, os, socket, ssl, sys

AMBIENT_TRUST_VARS = (
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS", "PYTHONHTTPSVERIFY", "SSLKEYLOGFILE",
    "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HF_ENDPOINT",
)
SYSTEM_BUNDLE_PATHS = (
    "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem", "/usr/lib/ssl/cert.pem",
)
_OID_CN = b"\x55\x04\x03"
_OID_O = b"\x55\x04\x0a"
_OID_SAN = b"\x55\x1d\x11"


def _der_read(buf, i):
    """(tag, element_start, content_start, element_end) for one DER element."""
    tag = buf[i]
    length = buf[i + 1]
    j = i + 2
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(bytes(buf[j:j + count]), "big")
        j += count
    return tag, i, j, j + length


def _der_kids(buf, start, end):
    kids = []
    i = start
    while i < end:
        item = _der_read(buf, i)
        kids.append(item)
        i = item[3]
    return kids


def _name_attrs(buf, start, end):
    """RDNSequence -> [(oid_bytes, text)], in encoding order."""
    attrs = []
    for _tag, _es, cs, ee in _der_kids(buf, start, end):
        for _t2, _es2, cs2, ee2 in _der_kids(buf, cs, ee):
            kids = _der_kids(buf, cs2, ee2)
            if len(kids) < 2:
                continue
            oid = bytes(buf[kids[0][2]:kids[0][3]])
            text = bytes(buf[kids[1][2]:kids[1][3]]).decode("utf-8", "replace")
            attrs.append((oid, text))
    return attrs


def _first(attrs, oid):
    for got, text in attrs:
        if got == oid:
            return text
    return None


def _time_text(buf, tag, cs, ee):
    raw = bytes(buf[cs:ee]).decode("ascii", "replace")
    if tag == 0x17 and len(raw) >= 13:
        century = "19" if int(raw[0:2]) >= 50 else "20"
        return "%s%s-%s-%sT%s:%s:%sZ" % (century, raw[0:2], raw[2:4], raw[4:6],
                                         raw[6:8], raw[8:10], raw[10:12])
    if tag == 0x18 and len(raw) >= 15:
        return "%s-%s-%sT%s:%s:%sZ" % (raw[0:4], raw[4:6], raw[6:8],
                                       raw[8:10], raw[10:12], raw[12:14])
    return raw


def describe_cert(der):
    """Subject/issuer/validity/SAN/SPKI straight out of the DER.

    Parsed from bytes rather than read from getpeercert() because the two cases
    that matter most have no dict: getpeercert() returns {} on an UNVERIFIED
    socket (exactly where a proxy's own leaf must be described), and
    get_verified_chain() exists only on python3.13+ while the paid box is
    python3.12 and the controller floor is stock python3.9.
    """
    buf = memoryview(der)
    _tag, _es, cs, ee = _der_read(buf, 0)
    kids = _der_kids(buf, cs, ee)
    tbs = kids[0]
    fields = _der_kids(buf, tbs[2], tbs[3])
    index = 1 if fields and fields[0][0] == 0xA0 else 0
    serial = fields[index]
    index += 2
    issuer = fields[index]
    index += 1
    validity = fields[index]
    index += 1
    subject = fields[index]
    index += 1
    spki = fields[index]
    index += 1
    extensions = None
    for item in fields[index:]:
        if item[0] == 0xA3:
            extensions = item
    subject_attrs = _name_attrs(buf, subject[2], subject[3])
    issuer_attrs = _name_attrs(buf, issuer[2], issuer[3])
    validity_kids = _der_kids(buf, validity[2], validity[3])
    san = []
    if extensions is not None:
        for _t, _e, ecs, eee in _der_kids(buf, extensions[2], extensions[3]):
            for _t2, _e2, xcs, xee in _der_kids(buf, ecs, eee):
                ext = _der_kids(buf, xcs, xee)
                if not ext or bytes(buf[ext[0][2]:ext[0][3]]) != _OID_SAN:
                    continue
                payload = ext[-1]
                for _t3, _e3, gcs, gee in _der_kids(buf, payload[2], payload[3]):
                    for tag3, _e4, ncs, nee in _der_kids(buf, gcs, gee):
                        if tag3 == 0x82:
                            san.append(bytes(buf[ncs:nee]).decode("ascii", "replace"))
    return {
        "der_sha256": hashlib.sha256(bytes(der)).hexdigest(),
        "spki_sha256": hashlib.sha256(bytes(buf[spki[1]:spki[3]])).hexdigest(),
        "serial_hex": bytes(buf[serial[2]:serial[3]]).hex(),
        "subject_cn": _first(subject_attrs, _OID_CN),
        "subject_org": _first(subject_attrs, _OID_O),
        "issuer_cn": _first(issuer_attrs, _OID_CN),
        "issuer_org": _first(issuer_attrs, _OID_O),
        "not_before": (_time_text(buf, validity_kids[0][0], validity_kids[0][2],
                                  validity_kids[0][3]) if validity_kids else None),
        "not_after": (_time_text(buf, validity_kids[1][0], validity_kids[1][2],
                                 validity_kids[1][3])
                      if len(validity_kids) > 1 else None),
        "san_dns": san,
    }


def _sanitize_url(value):
    """A proxy URL can carry user:pass@host.  Keep the shape, drop userinfo."""
    if "@" in value:
        scheme, sep, rest = value.partition("://")
        if sep and "@" in rest:
            return "%s://***@%s" % (scheme, rest.split("@", 1)[1])
        return "***"
    return value


def _bundle_facts(path):
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError:
        return None
    return {"path": path, "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "certificates": blob.count(b"-----BEGIN CERTIFICATE-----")}


def _connect(host, port, timeout, context, server_hostname):
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        return context.wrap_socket(sock, server_hostname=server_hostname)
    except BaseException:
        try:
            sock.close()
        except OSError:
            pass
        raise


def _chain_shas(tls, method):
    getter = getattr(tls, method, None)
    if getter is None:
        return None
    try:
        chain = getter()
    except (ssl.SSLError, ValueError):
        return None
    out = []
    for cert in chain or ():
        # CPython has changed this return type: 3.13 hands back
        # `_ssl.Certificate` objects, 3.14 hands back raw DER `bytes`.  Handle
        # both rather than silently reporting "chain depth unavailable" on a
        # version that does expose the chain.
        try:
            if isinstance(cert, (bytes, bytearray)):
                der = bytes(cert)
            else:
                der = cert.public_bytes(getattr(ssl, "ENCODING_DER", 1))
        except Exception:
            return None
        out.append(hashlib.sha256(der).hexdigest())
    return out


def collect(host, port=443, timeout=15.0, cafile=None, http_path=None,
            pinned=None):
    evidence = {
        "host": host, "port": port,
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "openssl_version": ssl.OPENSSL_VERSION,
        "uname": " ".join(list(os.uname())[:3]) if hasattr(os, "uname") else "",
        "ambient_trust_env": {},
        "ca_bundle_digests": [],
        "tls_ok": False, "error_class": None, "error_text": None,
        "tls_version": None, "cipher": None,
        "leaf": None, "presented_leaf": None,
        "chain_der_sha256": None, "chain_depth": None,
        "chain_depth_source": "unavailable",
        "bundle_verify": None, "bundle_path": cafile,
        "resolved_addresses": [], "http_status": None, "http_error": None,
    }
    for name in AMBIENT_TRUST_VARS:
        value = os.environ.get(name)
        if value:
            evidence["ambient_trust_env"][name] = _sanitize_url(value)
    for candidate in SYSTEM_BUNDLE_PATHS:
        facts = _bundle_facts(candidate)
        if facts:
            evidence["ca_bundle_digests"].append(facts)
    try:
        for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP):
            address = info[4][0]
            if address not in evidence["resolved_addresses"]:
                evidence["resolved_addresses"].append(address)
    except OSError as exc:
        evidence["error_class"] = exc.__class__.__name__
        evidence["error_text"] = str(exc)
        return evidence

    # (1) the box's OWN default trust.  This is the rung a passive interception
    # proxy fails, and the one that failed on Vast machine 68004.
    try:
        tls = _connect(host, port, timeout, ssl.create_default_context(), host)
        try:
            evidence["tls_ok"] = True
            evidence["tls_version"] = tls.version()
            cipher = tls.cipher()
            evidence["cipher"] = cipher[0] if cipher else None
            der = tls.getpeercert(binary_form=True)
            if der:
                evidence["leaf"] = describe_cert(der)
            shas = _chain_shas(tls, "get_verified_chain")
            if shas:
                evidence["chain_der_sha256"] = shas
                evidence["chain_depth"] = len(shas)
                evidence["chain_depth_source"] = "ssl.get_verified_chain"
        finally:
            tls.close()
    except Exception as exc:
        evidence["error_class"] = exc.__class__.__name__
        evidence["error_text"] = str(exc)

    # (2) verification against a bundle WE control, when it is on the box.
    if cafile:
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_verify_locations(cafile=cafile)
            tls = _connect(host, port, timeout, context, host)
            try:
                der = tls.getpeercert(binary_form=True)
                described = describe_cert(der) if der else None
                shas = _chain_shas(tls, "get_verified_chain")
                evidence["bundle_verify"] = {"ok": True, "leaf": described,
                                             "chain_der_sha256": shas}
                if evidence["leaf"] is None:
                    evidence["leaf"] = described
                if shas and not evidence["chain_der_sha256"]:
                    evidence["chain_der_sha256"] = shas
                    evidence["chain_depth"] = len(shas)
                    evidence["chain_depth_source"] = "ssl.get_verified_chain"
            finally:
                tls.close()
        except Exception as exc:
            evidence["bundle_verify"] = {
                "ok": False, "error_class": exc.__class__.__name__,
                "error_text": str(exc)}

    # (3) what was PRESENTED, verified or not.  A proxy's own leaf is the most
    # useful single piece of evidence and an unverified socket is the only way
    # to capture it, so this handshake deliberately does not verify.  Nothing
    # is sent over it: no request, no header, no credential -- handshake, read
    # the certificate, close.
    try:
        blind = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        blind.check_hostname = False
        blind.verify_mode = ssl.CERT_NONE
        tls = _connect(host, port, timeout, blind, host)
        try:
            der = tls.getpeercert(binary_form=True)
            if der:
                evidence["presented_leaf"] = describe_cert(der)
            shas = _chain_shas(tls, "get_unverified_chain")
            if shas:
                evidence["presented_chain_der_sha256"] = shas
                if evidence["chain_depth"] is None:
                    evidence["chain_depth"] = len(shas)
                    evidence["chain_depth_source"] = "ssl.get_unverified_chain"
        finally:
            tls.close()
    except Exception as exc:
        evidence["presented_error_class"] = exc.__class__.__name__
        evidence["presented_error_text"] = str(exc)

    # (3b) THE DISCRIMINATOR.  A failed handshake has two very different
    # causes and they must not be reported as one.  MEASURED 2026-09-06 on
    # Vast machine 68004, the host that produced the 2026-09-05 UNEXPECTED_EOF:
    # there is NO TLS interceptor there.  Its path to 1.1.1.1 is subject to
    # forged UDP DNS injection -- a query for huggingface.co gets three
    # replies, two of them third-party blackholes arriving before the real
    # CloudFront set -- so the box dials a stranger's address and the
    # handshake dies. Dialling the REAL addresses from that same box, with
    # SNI and full verification, succeeds with a byte-identical leaf.
    #
    # So: connect to each controller-supplied known-good ADDRESS with this
    # host's SNI and verify normally. Interception fails both dials (it is on
    # the path, whatever address we pick); DNS injection fails only the
    # resolved one. Without this, the guard blames a host operator whose TLS
    # is untouched -- and a certificate-layer observation cannot support that.
    if pinned:
        results = []
        for address in pinned:
            record = {"address": address, "ok": False}
            try:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                if cafile:
                    context.load_verify_locations(cafile=cafile)
                else:
                    context.load_default_certs(ssl.Purpose.SERVER_AUTH)
                sock = socket.create_connection((address, port),
                                                timeout=timeout)
                try:
                    tls = context.wrap_socket(sock, server_hostname=host)
                    try:
                        record["ok"] = True
                        der = tls.getpeercert(binary_form=True)
                        if der:
                            record["leaf"] = describe_cert(der)
                    finally:
                        tls.close()
                except BaseException:
                    sock.close()
                    raise
            except Exception as exc:
                record["error_class"] = exc.__class__.__name__
                record["error_text"] = str(exc)
            results.append(record)
        evidence["pinned_address_dials"] = results
        evidence["resolution_matches_pinned"] = bool(
            set(evidence["resolved_addresses"]) & set(pinned))

    # (4) REACHABILITY, kept separate from identity on purpose: a rate-limited
    # Hub (429) must never be reported as a lying host.  Anonymous request, no
    # Authorization header, no token read.
    if http_path and evidence["tls_ok"]:
        import http.client
        try:
            conn = http.client.HTTPSConnection(
                host, port, timeout=timeout, context=ssl.create_default_context())
            try:
                conn.request("HEAD", http_path,
                             headers={"User-Agent": "fidelity-tlsguard/1"})
                evidence["http_status"] = conn.getresponse().status
            finally:
                conn.close()
        except Exception as exc:
            evidence["http_error"] = "%s: %s" % (exc.__class__.__name__, exc)
    return evidence


if __name__ == "__main__":
    _host = sys.argv[1] if len(sys.argv) > 1 else "huggingface.co"
    _cafile = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    _path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    _port = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 443
    _pinned = ([a for a in sys.argv[5].split(",") if a]
               if len(sys.argv) > 5 and sys.argv[5] else None)
    sys.stdout.write(json.dumps(
        collect(_host, port=_port, cafile=_cafile, http_path=_path,
                pinned=_pinned), sort_keys=True) + "\n")
'''


def collector_script_text() -> str:
    """The collector, verbatim, for a caller that ships it to a box itself.

    A pre-transport probe (VastParity's `attest_live_resource`) runs before any
    bundle or token exists on the box, so nothing of ours is importable there.
    It embeds this text; the VERDICT stays here, controller-side, over the
    returned evidence.
    """
    return _COLLECTOR_SOURCE


_COLLECTOR_NS: Dict[str, Any] = {}


def _collector() -> Dict[str, Any]:
    if not _COLLECTOR_NS:
        exec(compile(_COLLECTOR_SOURCE, "<tlsguard-collector>", "exec"),
             _COLLECTOR_NS)
    return _COLLECTOR_NS


def describe_cert(der: bytes) -> Dict[str, Any]:
    """Subject/issuer/validity/SAN/SPKI of one DER certificate."""
    return _collector()["describe_cert"](der)


def collect_peer_evidence(host: str, *, port: int = 443, timeout: float = 15.0,
                          cafile: Optional[str] = None,
                          http_path: Optional[str] = None,
                          pinned: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Run the collector in THIS process -- the same code the box runs."""
    return _collector()["collect"](host, port, timeout, cafile, http_path,
                                   list(pinned) if pinned else None)


def remote_collector_command(host: str, *, port: int = 443,
                             cafile: Optional[str] = None,
                             http_path: Optional[str] = None,
                             pinned: Optional[Sequence[str]] = None,
                             python: str = "python3") -> str:
    """One shell command that runs the collector where nothing of ours exists.

    base64 so no provider exec channel can mangle the quoting, and so no repo
    file has to be present on the far side.  `pinned` carries the addresses
    the CONTROLLER resolved and verified, which is what lets the box's answer
    distinguish forged DNS from interception.
    """
    payload = base64.b64encode(_COLLECTOR_SOURCE.encode("utf-8")).decode("ascii")
    args = [host, cafile or "", http_path or "", str(port),
            ",".join(pinned or ())]
    quoted = " ".join("'%s'" % a.replace("'", "'\\''") for a in args)
    # `del sys.argv[1]` drops the base64 blob so the collector sees exactly the
    # argv it sees when run as a file: argv[1] is the host.  Without it the
    # script tries to resolve the payload as a hostname.
    return ("%s -c 'import base64,sys;"
            "src=base64.b64decode(sys.argv[1]).decode(\"utf-8\");"
            "del sys.argv[1];exec(src)' %s %s"
            % (python, payload, quoted))


# --------------------------------------------------------------------------
# Trust source: ours, digest-pinned, with a DISCLOSED operator override
# --------------------------------------------------------------------------


def ambient_trust_env() -> Dict[str, str]:
    """Ambient trust/proxy variables that are set (userinfo stripped)."""
    out: Dict[str, str] = {}
    sanitize = _collector()["_sanitize_url"]
    for name in AMBIENT_TRUST_VARS:
        value = os.environ.get(name)
        if value:
            out[name] = sanitize(value)
    return out


def bundle_root_digests(path: Optional[str] = None) -> List[str]:
    """sha256 of every root certificate in a PEM bundle, in file order."""
    text = Path(path or BUNDLE_PATH).read_text()
    digests: List[str] = []
    for chunk in text.split("-----BEGIN CERTIFICATE-----")[1:]:
        body = chunk.split("-----END CERTIFICATE-----")[0]
        digests.append(hashlib.sha256(
            base64.b64decode("".join(body.split()))).hexdigest())
    return digests


_TRUST_SOURCE: Optional[Dict[str, Any]] = None


def trust_source() -> Dict[str, Any]:
    """Where trust comes from, and every disclosure attached to that choice."""
    global _TRUST_SOURCE
    if _TRUST_SOURCE is not None:
        return _TRUST_SOURCE
    disclosures: List[str] = []
    ambient = ambient_trust_env()
    if ambient:
        disclosures.append(
            "ambient trust/proxy environment present and IGNORED for our own "
            "requests: %s" % ", ".join(sorted(ambient)))
    override_path = os.environ.get(OVERRIDE_BUNDLE_ENV)
    allow_system = os.environ.get(OVERRIDE_SYSTEM_ENV) in ("1", "true", "yes")
    if override_path:
        candidate = Path(override_path)
        if not candidate.is_file():
            raise TlsRefusal(
                "TLS-TRUST-OVERRIDE-MISSING",
                "%s names a PEM bundle that does not exist: %s"
                % (OVERRIDE_BUNDLE_ENV, override_path),
                ["point %s at a readable PEM file, or unset it to use the "
                 "bundle this repo ships (%s)"
                 % (OVERRIDE_BUNDLE_ENV, BUNDLE_PATH)])
        blob = candidate.read_bytes()
        source: Dict[str, Any] = {
            "kind": "operator-bundle", "path": str(candidate),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "certificates": blob.count(b"-----BEGIN CERTIFICATE-----"),
        }
        disclosures.append(
            "OPERATOR TRUST OVERRIDE in force: %s=%s (sha256 %s). Trust for "
            "this run is the operator's bundle, not the repo's."
            % (OVERRIDE_BUNDLE_ENV, candidate, source["sha256"]))
    elif allow_system:
        paths = ssl.get_default_verify_paths()
        source = {"kind": "system-store", "path": paths.cafile or paths.capath,
                  "sha256": None, "certificates": None}
        disclosures.append(
            "OPERATOR TRUST OVERRIDE in force: %s=1. Trust for this run is the "
            "AMBIENT system store (%s), which SSL_CERT_FILE/SSL_CERT_DIR can "
            "redirect. Use this only to unblock a CA rotation we have not "
            "shipped, and record it in the run's disclosures."
            % (OVERRIDE_SYSTEM_ENV, source["path"]))
    else:
        if not BUNDLE_PATH.is_file():
            raise TlsRefusal(
                "TLS-TRUST-BUNDLE-ABSENT",
                "the trust bundle this suite ships is missing: %s" % BUNDLE_PATH,
                ["restore it from git: "
                 "`git checkout -- bin/fidelity/tls-roots.pem`",
                 "on a rented box bin/BUNDLE.txt ships it, so re-upload the "
                 "bundle rather than editing the box",
                 "to proceed meanwhile under disclosure: %s=/path/to.pem"
                 % OVERRIDE_BUNDLE_ENV])
        blob = BUNDLE_PATH.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != BUNDLE_SHA256:
            raise TlsRefusal(
                "TLS-TRUST-BUNDLE-DIGEST",
                "our trust bundle does not match its pinned digest "
                "(expected %s, found %s)" % (BUNDLE_SHA256, digest),
                ["if you did NOT change the bundle this copy was edited: "
                 "`git checkout -- bin/fidelity/tls-roots.pem`, and on a "
                 "rented box treat it as host tampering (destroy, re-create)",
                 "if you DID rotate or add a root deliberately, set "
                 "BUNDLE_SHA256 in bin/fidelity/tlsguard.py to %s and re-run "
                 "python3 bin/selftest_tlsguard.py" % digest])
        source = {"kind": "vendored", "path": str(BUNDLE_PATH),
                  "sha256": digest,
                  "certificates": blob.count(b"-----BEGIN CERTIFICATE-----")}
    source["root_der_sha256"] = (
        [] if source["kind"] == "system-store"
        else bundle_root_digests(source["path"]))
    source["disclosures"] = disclosures
    _TRUST_SOURCE = source
    return source


_ANNOUNCED = False


def _announce(source: Dict[str, Any]) -> None:
    global _ANNOUNCED
    if _ANNOUNCED or not source.get("disclosures"):
        return
    _ANNOUNCED = True
    for line in source["disclosures"]:
        sys.stderr.write("tlsguard: %s\n" % line)


_CONTEXT: Optional[ssl.SSLContext] = None


def explicit_ssl_context() -> ssl.SSLContext:
    """The suite's ONE TLS client context: explicit, non-ambient, fail-closed.

    `load_default_certs()` is deliberately never called on the default path,
    which is the whole point: `SSL_CERT_FILE`/`SSL_CERT_DIR` are consulted only
    by that call and by `set_default_verify_paths()`, so an ambient variable
    cannot widen trust here.  It can only be observed and disclosed.
    """
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    source = trust_source()
    _announce(source)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if source["kind"] == "system-store":
        context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    else:
        context.load_verify_locations(cafile=source["path"])
    _CONTEXT = context
    return context


def reset_cached_state() -> None:
    """Drop the memoized context/trust source (tests move the environment)."""
    global _CONTEXT, _TRUST_SOURCE, _ANNOUNCED
    _CONTEXT = None
    _TRUST_SOURCE = None
    _ANNOUNCED = False


# --------------------------------------------------------------------------
# The verdict: a pure function over collected evidence
# --------------------------------------------------------------------------


def host_matches_names(host: str, names: Iterable[str]) -> bool:
    """RFC 6125 hostname match; a wildcard covers one leftmost label only."""
    want = (host or "").lower().rstrip(".")
    for name in names or ():
        got = (name or "").lower().rstrip(".")
        if not got:
            continue
        if got == want:
            return True
        if got.startswith("*.") and "." in want:
            if want.split(".", 1)[1] == got[2:]:
                return True
    return False


_UNREACHABLE_ERRORS = frozenset((
    "timeout", "TimeoutError", "socket.timeout", "ConnectionRefusedError",
    "ConnectionResetError", "gaierror", "OSError", "ConnectionAbortedError",
))


def _redact(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    try:
        from fidelity.common import redact
    except Exception:                       # pragma: no cover - path-dependent
        return text
    return redact(text)


def evaluate_peer_evidence(evidence: Dict[str, Any], *, host: str = HUB_HOST,
                           host_id: Optional[str] = None,
                           reference: Optional[Dict[str, Any]] = None,
                           trust: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Judge one collector document.  No network, no state, no credential.

    Every failure here is an IDENTITY failure -- fail closed, destroy the box
    -- except `TLS-UNREACHABLE`, which is retryable and explicitly not an
    accusation: a 429 or a timeout is the Hub saying "wait", not a host
    claiming to be the Hub.  Two lanes lost paid pods to that conflation.
    """
    trust = trust or trust_source()
    failures: List[Dict[str, str]] = []
    disclosures: List[str] = list(trust.get("disclosures") or ())
    where = "host %s" % (host_id or "unidentified")
    destroy = ("destroy this instance and re-create elsewhere: on a rented box "
               "a certificate that is not the Hub's is hostile until proven "
               "otherwise, and %s must not receive a credential" % where)
    record = ("record the provider machine id (%s) so that marketplace host is "
              "avoided on the next rental" % (host_id or "unknown"))
    # MEASURED 2026-09-06 (MitmForensics, on the very host that produced the
    # 2026-09-05 failure): Vast machine 68004 has NO TLS interceptor. Its path
    # to 1.1.1.1 suffers forged UDP DNS injection, so it dials a third party's
    # blackhole and the handshake dies -- while dialling the REAL addresses
    # from that same box, with SNI and full verification, succeeds with a
    # byte-identical leaf. So an UNEXPECTED_EOF or a hostname mismatch does
    # NOT license an accusation about the host's TLS: the discriminator is the
    # pinned-address dial (interception fails both dials, forged DNS fails
    # only the resolved one), and a misconfigured transparent Hub cache is a
    # third explanation. Name the candidates and the next measurement instead.
    dials = evidence.get("pinned_address_dials") or []
    pinned_ok = [d for d in dials if d.get("ok")]
    resolution_suspect = bool(dials) and bool(pinned_ok)
    diagnose = ("dial a known-good address for %s with SNI from that box and "
                "compare leaf digests: interception fails BOTH dials, forged "
                "DNS fails only the resolved one, and a transparent Hub cache "
                "fails neither but serves its own certificate" % host)

    leaf = evidence.get("leaf") or evidence.get("presented_leaf")
    presented = evidence.get("presented_leaf")
    status = evidence.get("http_status")
    error_class = evidence.get("error_class")
    verify = evidence.get("bundle_verify")
    # OUR bundle is authoritative; the box's own trust store is EVIDENCE.  A
    # container can legitimately ship a stale or empty CA store, and refusing a
    # peer that our own roots verified because the box's store did not would be
    # a guard that fires on a missing `ca-certificates` package -- which is
    # exactly the kind of noise that gets a guard routed around.
    ours_verified = bool(verify and verify.get("ok"))

    if not evidence.get("resolved_addresses") and error_class:
        failures.append({
            "code": "TLS-UNREACHABLE",
            "message": "%s could not resolve %s (%s)" % (where, host, error_class),
            "remedy": "retry; if it persists check the box's DNS and egress "
                      "before suspecting interception -- this is reachability, "
                      "not identity"})
    elif not evidence.get("tls_ok") and ours_verified:
        disclosures.append(
            "%s could not verify %s against its OWN trust store (%s) while our "
            "shipped roots did. Usually a stale or absent CA store in the "
            "image, not interception -- identity here rests on our bundle, and "
            "note that huggingface_hub/requests use certifi, a third store."
            % (where, host, error_class))
    elif not evidence.get("tls_ok"):
        text = _redact(evidence.get("error_text")) or ""
        if resolution_suspect:
            # The box CAN reach the real host at an address we verified, and
            # cannot at the one it resolved. That is name resolution being
            # tampered with, not the host operator's TLS.
            failures.append({
                "code": "TLS-RESOLUTION-SUSPECT",
                "message": "%s failed TLS to %s at the address it RESOLVED "
                           "(%s: %s) but succeeded at a controller-verified "
                           "address (%s) with a valid certificate. Its DNS "
                           "answers for %s are wrong; the host's TLS is not "
                           "implicated."
                           % (where, host, error_class, text,
                              pinned_ok[0].get("address"), host),
                "remedy": "do NOT use a credential here and do NOT report the "
                          "host operator for interception -- this is forged or "
                          "broken name resolution (measured on Vast machine "
                          "68004: a single query to 1.1.1.1 returning "
                          "third-party addresses ahead of the real ones). "
                          "Next: re-create elsewhere, or pin a resolver "
                          "(8.8.8.8 / 9.9.9.9 answered correctly there); " + record})
        elif error_class in _UNREACHABLE_ERRORS and not presented:
            failures.append({
                "code": "TLS-UNREACHABLE",
                "message": "%s could not open a TLS connection to %s (%s: %s)"
                           % (where, host, error_class, text),
                "remedy": "retry with backoff: a timeout or a refused "
                          "connection is reachability, not identity. If it "
                          "repeats, " + diagnose})
        else:
            extra = ("" if not presented else
                     "; the certificate it DID present names %r, issued by %r"
                     % (presented.get("subject_cn"), presented.get("issuer_cn")))
            failures.append({
                "code": "TLS-PEER-UNVERIFIED",
                "message": "%s could not verify %s (%s: %s)%s"
                           % (where, host, error_class, text, extra),
                "remedy": "this is the 2026-09-05 shape (UNEXPECTED_EOF / "
                          "hostname mismatch) and it has THREE candidate "
                          "causes -- forged DNS, an interception proxy, or a "
                          "misconfigured transparent Hub cache. No credential "
                          "goes here until it is told apart: " + diagnose
                          + ". Meanwhile the safe move is to re-create "
                          "elsewhere; " + record})

    if leaf is not None:
        names = list(leaf.get("san_dns") or ())
        if not names and leaf.get("subject_cn"):
            names = [leaf["subject_cn"]]
        if not host_matches_names(host, names):
            failures.append({
                "code": "TLS-PEER-HOSTNAME-MISMATCH",
                "message": "%s was served a certificate that does not cover %s "
                           "(subject %r, SAN %s, issuer %r)"
                           % (where, host, leaf.get("subject_cn"),
                              ",".join(names) or "none", leaf.get("issuer_cn")),
                # A forged DNS answer that happens to land on a stranger's
                # real TLS server produces exactly this, and so does an
                # interception proxy: the certificate alone cannot tell them
                # apart, so the refusal names the measurement that can.
                "remedy": ("no credential goes here. If the certificate names "
                           "an unrelated site, the likeliest cause is a forged "
                           "or stale DNS answer pointing at a third party -- "
                           "%s. If it names a private or host-local CA, it is "
                           "interception. Either way the safe move is to "
                           "re-create elsewhere; %s" % (diagnose, record))})

    if verify is not None and not verify.get("ok"):
        failures.append({
            "code": "TLS-PEER-CHAIN-NOT-OURS",
            "message": "%s presented a chain for %s that does not terminate in "
                       "a root this suite ships (%s: %s)"
                       % (where, host, verify.get("error_class"),
                          _redact(verify.get("error_text")) or ""),
            "remedy": "two distinct causes with different fixes. OUR BUNDLE IS "
                      "STALE (the Hub moved to a CA we do not ship): add the "
                      "root to bin/fidelity/tls-roots.pem, update "
                      "BUNDLE_SHA256, re-run python3 bin/selftest_tlsguard.py, "
                      "or unblock now under disclosure with %s=/path/to.pem. "
                      "THE HOST IS INTERCEPTING (its own store accepted what "
                      "ours refused): %s" % (OVERRIDE_BUNDLE_ENV, destroy)})

    roots = set(trust.get("root_der_sha256") or ())
    chain = (evidence.get("chain_der_sha256")
             or evidence.get("presented_chain_der_sha256"))
    if roots and chain and len(chain) > 1 and chain[-1] not in roots:
        failures.append({
            "code": "TLS-PEER-CHAIN-NOT-OURS",
            "message": "%s presented a chain for %s whose terminal certificate "
                       "(%s...) is not byte-identical to any root this suite "
                       "ships" % (where, host, chain[-1][:16]),
            "remedy": "our bundle may be stale (add the root, or "
                      "%s=/path/to.pem under disclosure) or the host is "
                      "intercepting: %s" % (OVERRIDE_BUNDLE_ENV, destroy)})

    accept_divergence = os.environ.get(OVERRIDE_LEAF_ENV) in ("1", "true", "yes")
    if reference and leaf and reference.get("spki_sha256"):
        if leaf.get("spki_sha256") != reference["spki_sha256"]:
            detail = ("%s sees leaf SPKI %s... for %s while the controller, "
                      "verifying against our own bundle, sees %s..."
                      % (where, (leaf.get("spki_sha256") or "?")[:16], host,
                         reference["spki_sha256"][:16]))
            if accept_divergence:
                disclosures.append(
                    "OPERATOR OVERRIDE %s=1: %s. Accepted as a DISCLOSURE; the "
                    "observed leaf identity is recorded in this document."
                    % (OVERRIDE_LEAF_ENV, detail))
            else:
                failures.append({
                    "code": "TLS-PEER-LEAF-DIVERGENCE",
                    "message": detail,
                    "remedy": "a re-signing proxy in front of the box is the "
                              "first explanation: " + destroy + ". A regional "
                              "edge certificate is the benign one -- check the "
                              "issuer and SAN recorded here, then re-run with "
                              "%s=1 to proceed as a recorded disclosure"
                              % OVERRIDE_LEAF_ENV})

    if status is not None and status >= 400:
        if status in (429, 500, 502, 503, 504):
            disclosures.append(
                "anonymous reachability probe returned HTTP %d -- the Hub "
                "saying \"wait\", not an identity problem; the TLS identity "
                "above is unaffected" % status)
        else:
            disclosures.append(
                "anonymous reachability probe returned HTTP %d; identity is "
                "judged from the certificate, never from this status" % status)

    ambient = evidence.get("ambient_trust_env") or {}
    if ambient:
        disclosures.append(
            "%s has ambient trust/proxy environment set: %s. Our own context "
            "ignores these, but huggingface_hub/requests/curl on that box do "
            "not." % (where, ", ".join("%s=%s" % kv
                                       for kv in sorted(ambient.items()))))
    if evidence.get("chain_depth") is None:
        disclosures.append(
            "chain depth unavailable on that interpreter (python %s exposes no "
            "verified-chain API before 3.13); leaf SPKI, issuer and SAN are "
            "recorded regardless" % evidence.get("python_version"))
    if verify is None and evidence.get("bundle_path") is None:
        disclosures.append(
            "the box did not verify against our bundle (it was not present at "
            "probe time); this verdict rests on the box's own store plus the "
            "controller-side leaf comparison")

    retryable = bool(failures) and all(f["code"] == "TLS-UNREACHABLE"
                                       for f in failures)
    return {
        "schema": SCHEMA,
        "host": host,
        "host_id": host_id,
        "ok": not failures,
        "verdict": ("attested" if not failures
                    else ("unreachable" if retryable else "refused")),
        "retryable": retryable,
        "failures": failures,
        "disclosures": disclosures,
        "peer": {
            "subject_cn": (leaf or {}).get("subject_cn"),
            "issuer_cn": (leaf or {}).get("issuer_cn"),
            "issuer_org": (leaf or {}).get("issuer_org"),
            "leaf_spki_sha256": (leaf or {}).get("spki_sha256"),
            "leaf_der_sha256": (leaf or {}).get("der_sha256"),
            "san_dns": list((leaf or {}).get("san_dns") or ()),
            "not_before": (leaf or {}).get("not_before"),
            "not_after": (leaf or {}).get("not_after"),
            "chain_depth": evidence.get("chain_depth"),
            "chain_depth_source": evidence.get("chain_depth_source"),
            "chain_der_sha256": chain,
            "tls_version": evidence.get("tls_version"),
            "cipher": evidence.get("cipher"),
        },
        "evidence": evidence,
        "trust_source": {k: trust[k] for k in ("kind", "path", "sha256",
                                               "certificates") if k in trust},
    }


def raise_for_verdict(verdict: Dict[str, Any]) -> Dict[str, Any]:
    """Return the verdict, or refuse with its codes and every remedy."""
    if verdict.get("ok"):
        return verdict
    first = verdict["failures"][0]
    advice = [f["remedy"] for f in verdict["failures"]]
    advice.extend("disclosure: %s" % d for d in verdict.get("disclosures") or ())
    raise TlsRefusal(first["code"], first["message"], advice,
                     evidence=verdict, retryable=verdict.get("retryable", False))


# --------------------------------------------------------------------------
# Attestation
# --------------------------------------------------------------------------


def attest_local_peer(host: str = HUB_HOST, *, port: int = 443,
                      timeout: float = 15.0, http_path: Optional[str] = None,
                      host_id: str = "controller") -> Dict[str, Any]:
    """Attest a peer from THIS machine, verified against our own bundle."""
    source = trust_source()
    _announce(source)
    cafile = None if source["kind"] == "system-store" else source["path"]
    evidence = collect_peer_evidence(host, port=port, timeout=timeout,
                                     cafile=cafile, http_path=http_path)
    return evaluate_peer_evidence(evidence, host=host, host_id=host_id,
                                  trust=source)


def attest_before_credential(provider: Any, machine_id: Any, *,
                             host_id: Optional[str] = None,
                             hosts: Sequence[str] = (HUB_HOST,),
                             port: int = 443,
                             python: str = "python3",
                             fs_root: Optional[str] = None,
                             timeout: float = 180.0,
                             exec_stdout: Optional[Any] = None) -> Dict[str, Any]:
    """Prove a rented box reaches the REAL hosts, before it holds a credential.

    THE ORDERING IS THE POINT.  Call this BEFORE `_transport_hf_token`, never
    after: on 2026-09-05 the token was already on the box when the proxy was
    hit.  When this refuses, no credential has been written, and the refusal
    says so along with the host id to destroy.

    `provider` needs only `exec_stdout(machine_id, command, timeout=...)`,
    which every adapter in this suite has.  `fs_root` (once the bundle has been
    uploaded) additionally makes the BOX verify against our own roots; without
    it the box verifies against its own store and the controller-side leaf
    comparison carries the identity claim.
    """
    runner = exec_stdout or getattr(provider, "exec_stdout")
    source = trust_source()
    _announce(source)
    box_bundle = ("%s/bin/fidelity/tls-roots.pem" % fs_root.rstrip("/")
                  if fs_root else None)
    observations: Dict[str, Any] = {}
    controller: Dict[str, Any] = {}
    disclosures: List[str] = list(source.get("disclosures") or ())
    failures: List[Dict[str, str]] = []
    for host in hosts:
        local = attest_local_peer(host, port=port, timeout=min(timeout, 30.0))
        controller[host] = local
        if not local["ok"]:
            # Our OWN network cannot verify this host.  That is "our bundle is
            # stale, or our controller is intercepted" -- a different problem
            # from a lying rented box, and it must not be reported as the box's.
            first = local["failures"][0]
            raise TlsRefusal(
                "TLS-TRUST-CONTROLLER-UNVERIFIED",
                "the CONTROLLER cannot verify %s against our own bundle: %s"
                % (host, first["message"]),
                ["this is not evidence about %s -- fix the controller side "
                 "first" % (host_id or "the rented box"),
                 first["remedy"],
                 "if the Hub rotated to a CA we do not ship: add the root to "
                 "bin/fidelity/tls-roots.pem, update BUNDLE_SHA256, re-run "
                 "python3 bin/selftest_tlsguard.py; to unblock now under "
                 "disclosure set %s=/path/to.pem" % OVERRIDE_BUNDLE_ENV],
                evidence=local)
        http_path = "/api/models/gpt2" if host == HUB_HOST else None
        # The addresses the CONTROLLER resolved and verified against our own
        # bundle. The box dials them too, with SNI, which is what separates
        # forged DNS on the box (only its resolved address fails) from an
        # interception proxy in front of it (every address fails).
        pinned = list((local["evidence"].get("resolved_addresses") or ())[:4])
        command = remote_collector_command(host, port=port, cafile=box_bundle,
                                           http_path=http_path, pinned=pinned,
                                           python=python)
        evidence = _parse_collector_output(
            runner(machine_id, command, timeout=timeout),
            host=host, host_id=host_id)
        verdict = evaluate_peer_evidence(
            evidence, host=host, host_id=host_id, trust=source,
            reference={"spki_sha256": local["peer"]["leaf_spki_sha256"]})
        observations[host] = verdict
        failures.extend(verdict["failures"])
        disclosures.extend(d for d in verdict["disclosures"]
                           if d not in disclosures)
    document = {
        "schema": SCHEMA,
        "attested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_id": host_id,
        "machine_id": str(machine_id),
        "hosts": list(hosts),
        "ordering": "attested BEFORE any credential was transported to the box",
        "trust_source": {k: source[k] for k in ("kind", "path", "sha256",
                                                "certificates")},
        "trust_source_root_der_sha256": list(source.get("root_der_sha256") or ()),
        "controller_observations": {h: v["peer"] for h, v in controller.items()},
        "pod_observations": {h: v["peer"] for h, v in observations.items()},
        "pod_ca_bundle_digests": {
            h: (v["evidence"].get("ca_bundle_digests") or [])
            for h, v in observations.items()},
        "pod_ambient_trust_env": {
            h: (v["evidence"].get("ambient_trust_env") or {})
            for h, v in observations.items()},
        "ok": not failures,
        "verdict": "attested" if not failures else "refused",
        "failures": failures,
        "disclosures": disclosures,
        "guarantee_ends": (
            "the box has root: a root-privileged host can forge this evidence. "
            "This proves a passive or naive interception proxy fails closed, "
            "records the peer identity the box reported, and guarantees no "
            "credential was transported before both happened."),
    }
    if failures:
        first = failures[0]
        raise TlsRefusal(
            first["code"], first["message"],
            [f["remedy"] for f in failures]
            + ["no HF credential has been written to %s: this refusal happened "
               "BEFORE the token transport" % (host_id or "the box")],
            evidence=document)
    return document


def _parse_collector_output(raw: Any, *, host: str,
                            host_id: Optional[str]) -> Dict[str, Any]:
    text = raw if isinstance(raw, str) else (raw or {}).get("stdout", "")
    for line in reversed([l.strip() for l in (text or "").splitlines()]):
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict) and parsed.get("host"):
                return parsed
    raise TlsRefusal(
        "TLS-PEER-NO-EVIDENCE",
        "host %s returned no parseable TLS evidence for %s"
        % (host_id or "unidentified", host),
        ["the box needs a python3 with the stdlib ssl module; check that the "
         "exec channel actually returns stdout",
         "treat an empty or mangled answer as untrusted: destroy the instance "
         "and re-create elsewhere rather than transporting a credential",
         "collector stdout began: %r" % ((text or "")[:200],)])


def write_attestation(document: Dict[str, Any], path: Any) -> str:
    """Atomically write an attestation document; returns its sha256.

    Atomic, and callers name a per-stage path, because stages now run
    CONCURRENTLY (fetch_reference alongside fetch_target): two leaders sharing
    one path is how a stale record satisfied a check instantly.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n"
    tmp = target.with_name(".%s.%d.tmp" % (target.name, os.getpid()))
    tmp.write_text(blob)
    os.replace(str(tmp), str(target))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def certifi_bundle_facts(python: Optional[str] = None) -> Dict[str, Any]:
    """The CA bundle `huggingface_hub`/`requests` will ACTUALLY use.

    Gap 3: `hf download` goes through requests -> certifi, not through our
    opener, so a green stdlib check does not attest the store the authenticated
    fetch uses -- they are two different stores on the same box.  The image is
    pinned by digest, so its certifi bundle digest is a checkable property of
    that image and a changed digest is evidence of tampering rather than noise.
    """
    import subprocess
    interpreter = python or sys.executable
    code = ("import certifi,hashlib,json;p=certifi.where();"
            "b=open(p,'rb').read();"
            "print(json.dumps({'path':p,'bytes':len(b),"
            "'sha256':hashlib.sha256(b).hexdigest(),"
            "'certificates':b.count(b'-----BEGIN CERTIFICATE-----')}))")
    try:
        out = subprocess.run([interpreter, "-c", code], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False,
                "note": "certifi bundle not readable via %s (%s)"
                        % (interpreter, exc.__class__.__name__)}
    if out.returncode != 0:
        return {"available": False,
                "note": "certifi not importable under %s, so that interpreter "
                        "is not the one `hf download` uses" % interpreter}
    try:
        facts = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"available": False, "note": "certifi probe returned no JSON"}
    facts["available"] = True
    facts["interpreter"] = interpreter
    return facts


# --------------------------------------------------------------------------
# No credential in a provider request body
# --------------------------------------------------------------------------
#
# `vastapi.py:2322` built `-e HF_TOKEN=...` into a `PUT /asks/{id}/` body, so
# the credential entered Vast's own records and the host's `docker run`
# environment BEFORE the instance existed.  There is no "attest first" for data
# handed over at create time, and no ordering that helps: the only fix is to
# refuse at the adapter, which is the last place that can tell.
#
# RunPod already refused exactly this (rungs RP7/RP7b in
# bin/selftest_root_publish.py, which assert the refusal AND that nothing was
# submitted).  The defect was never Vast's code in isolation -- it was a
# per-provider test that was never made per-provider.  Hence ONE
# implementation here, and one rung parameterised over every adapter.
#
# Findings NEVER contain the matched value.  They name the path, the key and
# the shape only: a guard that echoes the secret it found has moved the leak
# rather than closed it, and a refusal string is exactly what ends up in a
# receipt's warnings array.

_CREDENTIAL_SHAPES = (
    ("hf-user-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("hf-org-token", re.compile(r"\bapi_org_[A-Za-z0-9]{20,}\b")),
    ("openai-style-key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{12,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer-header", re.compile(r"(?i)authorization:\s*bearer\s+\S{16,}")),
    ("query-string-token", re.compile(r"(?i)\btoken=[A-Za-z0-9._\-]{24,}")),
)

# `NAME=value` inside a shell/docker payload.  `HF_TOKEN_PATH=/x` and
# `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` are the legitimate spellings and must not
# trip it, so a path-, file-, id- or name-suffixed variable is exempt, and so
# is an empty, numeric or path-shaped value.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|"
    r"CREDENTIAL|PRIVATE_KEY))\s*=\s*([^\s\"';]*)")
_NAME_EXEMPT_SUFFIX = ("_PATH", "_FILE", "_ID", "_NAME", "_DIR", "_ENDPOINT")
_SECRET_NAME_TAIL = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|CREDENTIAL|PRIVATE_KEY)$")

# A URL WITH A PATH is a bearer capability, and it matches no token shape --
# which is exactly why a shape-based matcher misses it (VastParity, 2026-09-06:
# `{"FIDELITY_RESULT_SINK": "https://sink.invalid/topic-cred"}` passed clean).
# Whoever holds a result-sink or ntfy-style topic URL can read the run's
# output, so in a provider-persisted payload it is a credential in every sense
# that matters. Two detectors, because either alone is wrong:
#   * a capability-shaped NAME whose value is a URL carrying a path -- a repo
#     or wheel URL (PIPE_REPO=https://github.com/owner/repo) is legitimate and
#     must not trip it, which is why the NAME decides here;
#   * a presigned-style path segment (>=20 chars of URL-safe alphabet) under
#     any name at all, which is what an S3/CloudFront signed URL looks like.
_CAPABILITY_NAME = re.compile(
    r"(?i)(SINK|WEBHOOK|CALLBACK|NOTIFY|HOOK|TOPIC|PRESIGN|SIGNED_URL|INVITE)")
_URL_WITH_PATH = re.compile(r"(?i)\b(https?)://([^\s/\"']+)(/[^\s\"']*)")
_OPAQUE_SEGMENT = re.compile(r"/[A-Za-z0-9_\-]{20,}(?:[/?#]|$)")


def _capability_findings(path: str, name: Optional[str], value: Any) -> List[str]:
    """A URL that IS an authorisation, reported without repeating it."""
    if not isinstance(value, str):
        return []
    found: List[str] = []
    for match in _URL_WITH_PATH.finditer(value):
        url_path = match.group(3)
        if url_path in ("/", ""):
            continue
        named = bool(name and _CAPABILITY_NAME.search(name))
        opaque = bool(_OPAQUE_SEGMENT.search(url_path))
        if not (named or opaque):
            continue
        found.append(
            "%s: %s carries a URL with a %d-character path on host %r -- a "
            "BEARER CAPABILITY (whoever holds it can read or write that "
            "endpoint), and it matches no token shape"
            % (path, ("key %r" % name) if name else "value",
               len(url_path), match.group(2)))
    return found


def _name_is_secret(name: str) -> bool:
    upper = name.upper()
    if upper.endswith(_NAME_EXEMPT_SUFFIX):
        return False
    return bool(_SECRET_NAME_TAIL.search(upper))


def _value_is_secretish(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text in ("0", "1", "true", "false", "True", "False"):
        return False
    if text.startswith(("/", "~", "./", "$")):    # a path, or a deferred lookup
        return False
    return len(text) >= 8


def credential_findings(payload: Any, *, path: str = "$") -> List[str]:
    """Every credential-shaped thing in a provider request body, by PATH.

    Three independent detectors, because each alone misses a real case: a
    token SHAPE anywhere in any string (covers `docker_cmd`/`onstart` text and
    an unknown key name); a secret-looking KEY or `NAME=value` assignment with
    a non-trivial value (covers a credential format we cannot pattern-match,
    e.g. after a rotation); and a URL that IS an authorisation -- a result
    sink, webhook or presigned link, which matches no token shape at all.
    """
    findings: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = "%s.%s" % (path, key)
            if isinstance(key, str) and _name_is_secret(key) \
                    and _value_is_secretish(value):
                findings.append(
                    "%s: key %r carries a credential-shaped value "
                    "(%d characters)" % (child, key, len(value.strip())))
            findings.extend(_capability_findings(
                child, key if isinstance(key, str) else None, value))
            findings.extend(credential_findings(value, path=child))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            findings.extend(credential_findings(
                value, path="%s[%d]" % (path, index)))
    elif isinstance(payload, str):
        for label, pattern in _CREDENTIAL_SHAPES:
            if pattern.search(payload):
                findings.append("%s: string contains a %s" % (path, label))
        for match in _SECRET_ASSIGNMENT.finditer(payload):
            name, value = match.group(1), match.group(2)
            if _name_is_secret(name) and _value_is_secretish(value):
                findings.append(
                    "%s: embeds %s=<%d characters> in a payload the provider "
                    "stores" % (path, name, len(value)))
        # `NAME=https://host/topic-cred` inside onstart/docker_cmd text: the
        # NAME is what makes it a capability rather than a repo pin, so the
        # assignment is matched, not the bare URL.
        for match in re.finditer(
                r"(?i)\b([A-Za-z0-9_]+)\s*=\s*(https?://[^\s\"';]+)", payload):
            findings.extend(_capability_findings(path, match.group(1),
                                                 match.group(2)))
        findings.extend(_capability_findings(path, None, payload))
    # Two detectors can name the same thing (a capability-shaped NAME whose
    # value is also presigned-shaped); a refusal should say it once.
    unique: List[str] = []
    for finding in findings:
        if finding not in unique:
            unique.append(finding)
    return unique


def refuse_credential_in_provider_payload(payload: Any,
                                          provider: Optional[str] = None,
                                          *, operation: str = "create",
                                          field: Optional[str] = None) -> None:
    """Refuse before transmitting a credential into a provider's own records.

    Call this in every adapter's `create`/`update` path, on the request body
    and on any `env`/`onstart`/`docker_cmd`/`user_data`/`script` text.  It
    raises `TlsRefusal`, whose `.reason`/`.advice` are `Refusal` arguments; an
    adapter with its own error type wraps it (`raise VastError(exc.reason)`).

    `provider` may be positional, keyword, or omitted entirely when the
    payload names itself (`{"provider": "vast", "env": ...}`) -- an adapter
    should not need a TypeError fallback to call its own security guard, and a
    fallback path is a second implementation waiting to drift.
    """
    if not provider and isinstance(payload, dict):
        named = payload.get("provider")
        provider = named if isinstance(named, str) else None
    provider = provider or "provider"
    findings = credential_findings(payload, path=field or "$")
    if not findings:
        return
    raise TlsRefusal(
        "PROVIDER-PAYLOAD-CREDENTIAL",
        "%s %s payload is provider-persisted and carries a credential: %s"
        % (provider, operation, "; ".join(findings)),
        ["a credential in a create body is stored by the provider and lands in "
         "the host's process environment BEFORE the instance exists, so no "
         "attestation and no ordering can protect it -- remove it from the "
         "payload",
         "transport the credential after the box is attested: a 0600 file over "
         "the authenticated exec channel (measure_cloud._transport_hf_token), "
         "shredded at teardown",
         "if the run only needs PUBLIC artifacts, pass no credential at all: "
         "anonymous fetches need none, and that is the least-privilege path",
         "a provider-visible payload is not private: Vast's own log endpoint "
         "hands back a world-readable s3.amazonaws.com URL for run output"])


# --------------------------------------------------------------------------
# CLI.  Runs on the pod (`python3 bin/fidelity/tlsguard.py attest ...`) and on
# the controller.  Prints peer identity, never a credential.
# --------------------------------------------------------------------------


def _cmd_attest(args: argparse.Namespace) -> int:
    verdict = attest_local_peer(
        args.host, port=args.port, timeout=args.timeout,
        http_path=args.http_path, host_id=args.host_id or "unidentified")
    document: Dict[str, Any] = {
        "schema": SCHEMA,
        "attested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_id": args.host_id,
        "role": args.role,
        "host": args.host,
        "ok": verdict["ok"],
        "verdict": verdict["verdict"],
        "peer": verdict["peer"],
        "failures": list(verdict["failures"]),
        "disclosures": list(verdict["disclosures"]),
        "trust_source": verdict["trust_source"],
        "ca_bundle_digests": verdict["evidence"].get("ca_bundle_digests") or [],
        "ambient_trust_env": verdict["evidence"].get("ambient_trust_env") or {},
        "python_version": verdict["evidence"].get("python_version"),
        "guarantee_ends": (
            "run on the box itself: a root-privileged host can forge this. It "
            "proves a passive interception proxy fails closed and records what "
            "the peer looked like."),
    }
    if args.certifi or args.certifi_python:
        facts = certifi_bundle_facts(args.certifi_python)
        document["certifi_bundle"] = facts
        if facts.get("available"):
            evidence = collect_peer_evidence(args.host, port=args.port,
                                             timeout=args.timeout,
                                             cafile=facts["path"])
            document["certifi_verify_ok"] = bool(
                (evidence.get("bundle_verify") or {}).get("ok"))
            document["certifi_peer"] = evaluate_peer_evidence(
                evidence, host=args.host, host_id=args.host_id)["peer"]
            if not document["certifi_verify_ok"]:
                document["ok"] = False
                document["verdict"] = "refused"
                document["failures"].append({
                    "code": "TLS-PEER-CERTIFI-CHAIN-NOT-OURS",
                    "message": "%s: the store `hf download` actually uses "
                               "(certifi at %s) does not verify %s"
                               % (args.host_id or "unidentified",
                                  facts["path"], args.host),
                    "remedy": "the authenticated fetch runs through THAT store, "
                              "so do not use the credential here: destroy and "
                              "re-create the instance, or re-pin the image if "
                              "its bundle legitimately changed"})
        else:
            document["disclosures"].append(
                "certifi store not attested: %s. The stdlib check above used "
                "the OS store, which is NOT the store `hf download` uses."
                % facts.get("note"))
    if args.json:
        document["document_sha256"] = write_attestation(document, args.json)
    peer = document["peer"]
    print("tls %s %s: subject=%s issuer=%s spki=%s... depth=%s (%s) trust=%s"
          % (document["verdict"], args.host, peer.get("subject_cn"),
             peer.get("issuer_cn"), (peer.get("leaf_spki_sha256") or "?")[:16],
             peer.get("chain_depth"), peer.get("chain_depth_source"),
             document["trust_source"].get("kind")))
    for line in document["disclosures"]:
        print("  disclosure: %s" % line)
    for failure in document["failures"]:
        print("  REFUSED %s: %s" % (failure["code"], failure["message"]))
        print("    remedy: %s" % failure["remedy"])
    if document["ok"]:
        return 0
    return 75 if document["verdict"] == "unreachable" else 3


def _cmd_collect(args: argparse.Namespace) -> int:
    print(json.dumps(collect_peer_evidence(
        args.host, port=args.port, timeout=args.timeout, cafile=args.cafile,
        http_path=args.http_path), sort_keys=True))
    return 0


def _cmd_ca_digest(args: argparse.Namespace) -> int:
    facts = _collector()["_bundle_facts"]
    document = {
        "schema": "fidelity.tls-ca-bundle-digest/1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_id": args.host_id,
        "system_bundles": [f for f in (facts(p) for p in
                                       _collector()["SYSTEM_BUNDLE_PATHS"]) if f],
        "certifi_bundle": certifi_bundle_facts(args.certifi_python),
        "suite_bundle": {k: trust_source()[k]
                         for k in ("kind", "path", "sha256", "certificates")},
        "ambient_trust_env": ambient_trust_env(),
    }
    if args.json:
        write_attestation(document, args.json)
    print(json.dumps(document, sort_keys=True, indent=2))
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    source = trust_source()
    print("trust source: %s %s" % (source["kind"], source["path"]))
    print("bundle sha256: %s (%s certificates)"
          % (source["sha256"], source["certificates"]))
    for digest in source.get("root_der_sha256") or ():
        print("  root %s" % digest)
    for line in source["disclosures"]:
        print("disclosure: %s" % line)
    context = explicit_ssl_context()
    print("check_hostname=%s verify_mode=%s minimum_version=%s"
          % (context.check_hostname, context.verify_mode, context.minimum_version))
    return 0


def _cmd_collector_source(args: argparse.Namespace) -> int:
    sys.stdout.write(_COLLECTOR_SOURCE)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="explicit TLS trust and peer attestation (prints peer "
                    "identity; never a credential)")
    sub = parser.add_subparsers(dest="command")

    attest = sub.add_parser("attest", help="verify a peer against our bundle")
    attest.add_argument("--host", default=HUB_HOST)
    attest.add_argument("--port", type=int, default=443)
    attest.add_argument("--timeout", type=float, default=15.0)
    attest.add_argument("--host-id", default=None,
                        help="provider machine id, quoted in every refusal")
    attest.add_argument("--role", default=None, help="free-form call-site tag")
    attest.add_argument("--http-path", default=None,
                        help="anonymous reachability probe path, e.g. "
                             "/api/models/gpt2 (never sends Authorization)")
    attest.add_argument("--json", default=None, help="write the document here")
    attest.add_argument("--certifi", action="store_true",
                        help="also verify through this interpreter's certifi")
    attest.add_argument("--certifi-python", default=None,
                        help="the interpreter whose certifi store `hf "
                             "download` will use, e.g. $VENV/bin/python")
    attest.set_defaults(func=_cmd_attest)

    collect = sub.add_parser("collect", help="print raw peer evidence as JSON")
    collect.add_argument("--host", default=HUB_HOST)
    collect.add_argument("--port", type=int, default=443)
    collect.add_argument("--timeout", type=float, default=15.0)
    collect.add_argument("--cafile", default=None)
    collect.add_argument("--http-path", default=None)
    collect.set_defaults(func=_cmd_collect)

    digest = sub.add_parser("ca-digest",
                            help="record the CA bundles present on this box")
    digest.add_argument("--host-id", default=None)
    digest.add_argument("--certifi-python", default=None)
    digest.add_argument("--json", default=None)
    digest.set_defaults(func=_cmd_ca_digest)

    context = sub.add_parser("context", help="report the trust source in force")
    context.set_defaults(func=_cmd_context)

    script = sub.add_parser("collector-source",
                            help="print the collector script text")
    script.set_defaults(func=_cmd_collector_source)

    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except TlsRefusal as exc:
        sys.stderr.write("REFUSED %s: %s\n" % (exc.code, exc.reason))
        for line in exc.advice:
            sys.stderr.write("  remedy: %s\n" % line)
        return 75 if exc.retryable else 3


if __name__ == "__main__":
    raise SystemExit(main())
