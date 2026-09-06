#!/usr/bin/env python3
"""A transient HTTP status is a WAIT, not a refusal -- on both fetch layers.

On 2026-09-06 a paid pod died eighteen seconds into a rental because the
anonymous reference fetch got HTTP 429 and the stage exited 3. The fetch is
anonymous on purpose: reading the published root WITHOUT a token is what proves
it is publicly readable, which is the property every "verified anonymously" row
claims. So there is no credential to fall back on, and with several lanes
pulling published roots at once a 429 is expected traffic rather than an
anomaly. Minutes later the SAME status refused a controller-side `blobs=true`
census in `_anonymous_hf_environment` -- no pod involved, three concurrent
dry-runs were enough.

Two layers therefore need the identical policy, and this suite holds both to it:

  fidelity/dshub.py   the pod's reference fetch and immutable-member streaming
  fidelity/hfmeta.py  controller-side repository metadata and file reads

What is NOT retried matters as much as what is. 401, 403 and 404 are answers
ABOUT THE REQUEST, and on an anonymous read a 403 is a statement about public
readability -- exactly the finding these rows exist to make -- so retrying it
would hide the finding behind a delay.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from fidelity import dshub as DS                                  # noqa: E402
from fidelity import hfmeta as HM                                 # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          "" if ok else "  (%s)" % detail))
    if not ok:
        FAILURES.append(name)


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://huggingface.co/x", code, "boom", headers, None)


class Recorder:
    """Stands in for time.sleep and records what the policy asked to wait."""

    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


def with_sleep(module, fn):
    """Run fn with the module's sleep captured; return (result, waits)."""
    recorder = Recorder()
    original = module._SLEEP
    module._SLEEP = recorder
    try:
        return fn(), recorder.waits
    finally:
        module._SLEEP = original


def opener_that(module, responses):
    """Replace the transport with one that yields `responses` in order.

    Each entry is either an HTTPError to raise or bytes to return.
    """
    state = {"i": 0}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

        def getcode(self):
            return 200

    def transport(request, timeout=None):
        item = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        if isinstance(item, urllib.error.HTTPError):
            raise item
        return Response(item)

    return transport, state


def main():
    print("== the retry POLICY (shared by both layers) ==")
    missing = [label for label, module in (("dshub", DS), ("hfmeta", HM))
               if not hasattr(module, "_retry_delay")
               or not hasattr(module, "_SLEEP")]
    if missing:
        # Without the policy there is nothing to hold to it, and a bare
        # AttributeError traceback would read like a broken test rather than
        # the absent behaviour it is.
        check("both fetch layers carry a transient-retry policy (%s do not)"
              % ", ".join(missing), False, "no _retry_delay/_SLEEP")
        print("\nselftest_hub_retry: %d FAILED" % len(FAILURES))
        return 1
    for label, module in (("dshub", DS), ("hfmeta", HM)):
        # A 429 with no Retry-After backs off exponentially, bounded.
        delays = [module._retry_delay(http_error(429), n, 0.0)
                  for n in range(1, 6)]
        check("%s: a 429 without Retry-After backs off 2,4,8,16 then gives up "
              "at the attempt cap" % label,
              delays[:4] == [2.0, 4.0, 8.0, 16.0] and delays[4] is None,
              repr(delays))
        # An explicit Retry-After wins over the backoff, but is still bounded:
        # a server (or a proxy) asking for an hour must not hold a rented pod.
        check("%s: an explicit Retry-After is honoured" % label,
              module._retry_delay(http_error(429, 7), 1, 0.0) == 7.0)
        check("%s: an absurd Retry-After is capped, not obeyed" % label,
              module._retry_delay(http_error(429, 86400), 1, 0.0)
              == module._RETRY_MAX_DELAY)
        check("%s: a malformed or HTTP-date Retry-After falls back to backoff "
              "rather than trusting a parse" % label,
              module._retry_delay(http_error(429, "Wed, 21 Oct 2026 07:28:00 GMT"),
                                  1, 0.0) == 2.0
              and module._retry_delay(http_error(429, "-5"), 1, 0.0) == 2.0)
        # The answers-about-the-request family must never be retried.
        for code in (400, 401, 403, 404, 410, 422):
            if module._retry_delay(http_error(code), 1, 0.0) is not None:
                check("%s: HTTP %d must NOT be retried" % (label, code), False)
                break
        else:
            check("%s: 400/401/403/404/410/422 are never retried -- an "
                  "anonymous 403 is a finding about public readability, not a "
                  "wait" % label, True)
        check("%s: 5xx transients are retried alongside 429" % label,
              all(module._retry_delay(http_error(code), 1, 0.0) == 2.0
                  for code in (500, 502, 503, 504)))
        # The total budget stops a slow-motion denial from holding a pod for
        # the whole rental.
        check("%s: the cumulative wait is bounded by a total budget" % label,
              module._retry_delay(http_error(429),
                                  1, module._RETRY_TOTAL_BUDGET) is None)

    print()
    print("== dshub: the pod's reference fetch ==")
    payload = b'{"ok": true}'
    original = DS._OPENER
    try:
        transport, state = opener_that(
            DS, [http_error(429, 1), http_error(429, 1), payload])
        DS._OPENER = type("O", (), {"open": staticmethod(transport)})
        got, waits = with_sleep(
            DS, lambda: DS._get("https://huggingface.co/api/x"))
        check("a 429 twice then success returns the body, and the two waits "
              "came from Retry-After",
              json.loads(got) == {"ok": True} and waits == [1.0, 1.0],
              "%r waits=%r" % (got, waits))

        transport, _ = opener_that(DS, [http_error(429, 1)])
        DS._OPENER = type("O", (), {"open": staticmethod(transport)})
        refused, waits = with_sleep(DS, lambda: _refusal(
            lambda: DS._get("https://huggingface.co/api/x")))
        check("a 429 that never clears still refuses, after the bounded "
              "attempts, and says 429",
              refused is not None and "429" in str(refused)
              and len(waits) == DS._RETRY_ATTEMPTS - 1,
              "%r waits=%r" % (refused, waits))

        transport, _ = opener_that(DS, [http_error(404)])
        DS._OPENER = type("O", (), {"open": staticmethod(transport)})
        refused, waits = with_sleep(DS, lambda: _refusal(
            lambda: DS._get("https://huggingface.co/api/x")))
        check("a 404 refuses immediately with no wait at all",
              refused is not None and "404" in str(refused) and waits == [],
              "%r waits=%r" % (refused, waits))
    finally:
        DS._OPENER = original

    print()
    print("== dshub: a retried STREAM re-derives its identity from zero ==")
    body = b"a" * 64
    import hashlib
    sha = hashlib.sha256(body).hexdigest()
    original = DS._OPENER
    try:
        # First attempt 429s, second delivers the whole member. The exact
        # byte/sha256 gates must still pass -- and must judge one whole
        # attempt, never a resumed or spliced one.
        transport, _ = opener_that(DS, [http_error(429, 1), body])
        DS._OPENER = type("O", (), {"open": staticmethod(transport)})
        got, waits = with_sleep(DS, lambda: DS._read_remote_exact(
            "https://huggingface.co/datasets/x/resolve/y/capture/hidden_0022.safetensors",
            len(body), sha, capture=True))
        check("a member that 429s once streams clean on the retry and still "
              "satisfies its exact byte and sha256 gates",
              got == body and waits == [1.0], "waits=%r" % (waits,))

        # And a retry must not be able to launder a WRONG body into a pass.
        transport, _ = opener_that(DS, [http_error(429, 1), b"b" * 64])
        DS._OPENER = type("O", (), {"open": staticmethod(transport)})
        refused, _ = with_sleep(DS, lambda: _refusal(
            lambda: DS._read_remote_exact(
                "https://huggingface.co/datasets/x/resolve/y/m", len(body),
                sha, capture=True)))
        check("a retry does NOT relax the identity gate: wrong bytes after a "
              "429 still refuse as differing from the verified source",
              refused is not None and "differs from" in str(refused),
              repr(refused))
    finally:
        DS._OPENER = original

    print()
    print("== hfmeta: the controller's anonymous metadata read ==")
    original_open = HM.safe_urlopen
    try:
        transport, _ = opener_that(
            HM, [http_error(429, 1), b'{"sha": "a"}'])
        HM.safe_urlopen = transport
        got, waits = with_sleep(
            HM, lambda: HM._get("https://huggingface.co/api/models/x?blobs=true"))
        check("the blobs=true census survives a 429 -- the exact refusal that "
              "killed a dry-run with no pod involved",
              got == {"sha": "a"} and waits == [1.0],
              "%r waits=%r" % (got, waits))

        transport, _ = opener_that(HM, [http_error(403)])
        HM.safe_urlopen = transport
        refused, waits = with_sleep(HM, lambda: _refusal(
            lambda: HM._get("https://huggingface.co/api/models/x")))
        check("a 403 on an anonymous read refuses at once and keeps its "
              "gated/private hint -- retrying would hide the finding",
              refused is not None and "403" in str(refused)
              and "gated" in str(refused) and waits == [],
              "%r waits=%r" % (refused, waits))

        transport, _ = opener_that(HM, [http_error(429, 1), b"raw-bytes"])
        HM.safe_urlopen = transport
        got, waits = with_sleep(HM, lambda: HM.fetch_file(
            "x/y", "config.json", revision="a" * 40))
        check("fetch_file retries too, so a 429 on one shard's metadata does "
              "not fail a plan", got == b"raw-bytes" and waits == [1.0],
              "%r waits=%r" % (got, waits))
    finally:
        HM.safe_urlopen = original_open

    print()
    print("== the REMEDY has to match the status ==")
    import importlib
    FD = importlib.import_module("fidelity_dataset")
    advice = FD._hub_error_advice(429)
    check("a 429 remedy says WAIT and says there is no token to add -- it used "
          "to print --token-file advice for the status that cost two pods",
          "rate limit" in advice and "--token-file" not in advice
          and "anonymous" in advice, advice[:110])
    advice = FD._hub_error_advice(403)
    check("a 403 remedy still offers --token-file, and names the anonymous "
          "case as a finding about public readability",
          "--token-file" in advice and "publicly readable" in advice,
          advice[:110])
    check("a 404 remedy still points at `adapt`",
          "adapt" in FD._hub_error_advice(404))
    advice = FD._hub_error_advice(None)
    check("a status-less hub failure keeps the full three-way advice rather "
          "than guessing one",
          "adapt" in advice and "--token-file" in advice
          and "rate limit" in advice, advice[:110])
    streamed = DS.HubError("HTTP 429 while streaming immutable public evidence")
    streamed.status = 429
    check("the STREAMING refusal carries its status, so a member fetch that "
          "429s past the retry budget gets the wait remedy and not the "
          "credential one",
          FD._hub_error_advice(getattr(streamed, "status", None))
          == FD._hub_error_advice(429))

    print()
    if FAILURES:
        print("selftest_hub_retry: %d FAILED" % len(FAILURES))
        return 1
    print("selftest_hub_retry: all passed")
    return 0


def _refusal(fn):
    try:
        fn()
    except (DS.HubError, HM.HFError) as exc:
        return exc
    return None


if __name__ == "__main__":
    raise SystemExit(main())
