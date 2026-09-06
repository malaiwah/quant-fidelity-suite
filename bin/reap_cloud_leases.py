#!/usr/bin/env python3
"""Isolated installed entrypoint for the paid cloud lease reaper.

One systemd template instance per provider
(`fidelity-cloud-reaper@<provider>.timer`), one sealed control manifest and
one health stamp per provider, and a sweep driven entirely through the
provider contract -- so a conforming adapter gets an autonomous teardown
backstop with no change here.  A provider whose adapter cannot be swept is
refused by name before any provider call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Python -I deliberately omits the script directory. Add only this private
# snapshot root; -S keeps user and system site packages out of the import graph.
SNAPSHOT_ROOT = Path(__file__).resolve().parent
if not SNAPSHOT_ROOT.is_absolute():
    raise SystemExit(90)
sys.path[:] = [str(SNAPSHOT_ROOT)] + [
    item for item in sys.path
    if item and os.path.isabs(item) and "site-packages" not in item
]

from fidelity.cloudlease import (  # noqa: E402
    _safe_directory_fd,
    LeaseStore,
    reap_once,
    verify_reaper_control_account,
    verify_reaper_runtime_invocation,
    write_reaper_health,
)
from fidelity import providers as P  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reap-cloud-leases")
    parser.add_argument("--provider", choices=P.PROVIDERS, required=True)
    parser.add_argument("--sweep", action="store_true", required=True)
    parser.add_argument("--lease-dir", required=True)
    parser.add_argument("--reaper-state-dir", required=True)
    # The credential path, per provider.  JarvisLabs is CLI-driven (`jl`, with
    # JL_API_KEY in the unit environment) and takes no key file; the three
    # REST adapters require one, and it is passed as a PATH so the secret
    # never reaches argv.
    parser.add_argument("--key-file")
    return parser


def _build_provider(name: str, key_file):
    refusal = P.sweep_refusal(name)
    if refusal is not None:
        raise RuntimeError(str(refusal))
    adapter = P.adapter_class(name)
    if name == "jarvislabs":
        if key_file:
            raise RuntimeError(
                "jarvislabs is CLI-driven and takes no --key-file")
        return adapter(dry=False)
    if not key_file:
        raise RuntimeError("%s requires --key-file" % name)
    return adapter(dry=False, key_file=key_file)


def _invalidate_prior_health(state: Path, provider: str) -> None:
    """Durably remove an older success before this invocation can fail."""
    directory_fd = _safe_directory_fd(state, create=False)
    try:
        try:
            os.unlink("reaper-health-%s.json" % provider, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    store = LeaseStore(Path(args.lease_dir))
    state = Path(args.reaper_state_dir)
    # A mutable checkout invocation cannot invalidate installed health or touch
    # provider state. Only the exact isolated snapshot command may proceed.
    verify_reaper_runtime_invocation(
        state, lease_dir=store.root, provider=args.provider)
    # Fail closed across provider outages, credential drift, process death, and
    # every exception below: an older success can never survive a newer sweep.
    _invalidate_prior_health(state, args.provider)
    provider = _build_provider(args.provider, args.key_file)
    status = provider.status()
    account = str(status.get("id") or "").strip()
    if not account:
        raise RuntimeError(
            "%s status lacks an exact account identity" % args.provider)
    # This check is unconditional, including an empty/terminal-only sweep.
    verify_reaper_control_account(
        state, lease_dir=store.root, provider=args.provider,
        provider_account_id=account)
    result = reap_once(store, {args.provider: provider}, dry_run=False)
    if not result.ok:
        for failure in result.failures:
            print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 90
    # Health is the final durable side effect and is never written for an
    # account mismatch or any failed cleanup/projection step.
    write_reaper_health(
        state, result, lease_dir=store.root, provider=args.provider,
        provider_account_id=account)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("reaper failed: %s" % exc, file=sys.stderr)
        raise SystemExit(90)
