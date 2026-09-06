#!/usr/bin/env python3
"""Offline selftest for bin/measurement_inventory.py.

The tool exists because `qualified-unpublished` was a validated state that
nothing enumerated, so a verified capture could wait a day for a row with no
signal. Its own value therefore depends entirely on the classification being
right, and on the REFUSAL being able to fire -- the STRANDED case does not
occur in the current estate, so nothing but a fixture can prove that arm works.

Scratch directories only. No network, no provider, no torch, no installs.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import measurement_inventory as MI  # noqa: E402

FAILURES: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print("PASS %s" % label)
    else:
        FAILURES.append(label)
        print("FAIL %s%s" % (label, (" -- " + detail) if detail else ""))


def write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_run(root: Path, name: str, *, status: Optional[str] = None,
             structural: Optional[str] = None, errors: Optional[int] = None,
             destination: Optional[str] = None,
             published_repo: Optional[str] = None,
             dataset: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    if status is not None:
        write(run / "terminal-receipt.json",
              {"combined_status": status, "scientific_status": status})
    if structural is not None:
        write(run / "local-verify.json",
              {"structural_status": structural, "error_count": errors})
    if dataset:
        write(run / "result" / "dataset" / "fidelity-dataset.json",
              {"dataset_sha256": "d" * 64,
               "capture": {"capture_content_digest": "c" * 64}})
    else:
        (run / "result").mkdir(parents=True, exist_ok=True)
    if destination is not None or status == MI.QUALIFIED_UNPUBLISHED:
        write(run / "result" / "receipts" / "root-qualification.json",
              {"destination_repository": destination})
    if published_repo is not None:
        write(run / "result" / "receipts" / "publish-root.json",
              {"repository": published_repo, "revision": "a" * 40,
               "verified_after_publish": True})
    return run


def main(argv: Optional[List[str]] = None) -> int:
    print("== classification ==")
    with tempfile.TemporaryDirectory(prefix="mi-selftest-") as tmp:
        root = Path(tmp)

        # The case that cost $6.59: verified, a destination IS named, and it
        # was never published. Nothing in the current estate looks like this,
        # so only a fixture can prove the refusal arm works at all.
        stranded = make_run(root, "stranded-root", status=MI.QUALIFIED_UNPUBLISHED,
                            structural="sealed", errors=0,
                            destination="malaiwah/some-root-v1")
        facts = MI.RunFacts(stranded)
        check("a verified, publishable, unpublished run is STRANDED",
              facts.classify() == "STRANDED", facts.classify())
        check("and it is recognised as verified",
              facts.verified is True, repr(facts.verified))

        # flashA-k2-run1's real shape: sealed and verified, but the sealed
        # qualification names NO destination, so no --repo can satisfy the
        # publish gate. That is a row gap, not a loss, and must not be
        # reported as the MiniMax shape.
        needs_row = make_run(root, "candidate-no-destination",
                             status=MI.QUALIFIED_UNPUBLISHED,
                             structural="sealed", errors=0, destination=None)
        check("a sealed run whose qualification names no destination is "
              "needs-row, NOT stranded",
              MI.RunFacts(needs_row).classify() == "needs-row",
              MI.RunFacts(needs_row).classify())

        published = make_run(root, "published-run", status="published",
                             structural="sealed", errors=0,
                             published_repo="malaiwah/published-v1")
        check("a run with a publish receipt is published",
              MI.RunFacts(published).classify() == "published")

        # A publish receipt wins even over the unpublished token: the receipt
        # is evidence the upload happened, and a stale status must not
        # resurrect a published run into the stranded bucket.
        both = make_run(root, "published-stale-status",
                        status=MI.QUALIFIED_UNPUBLISHED, structural="sealed",
                        errors=0, destination="malaiwah/x-v1",
                        published_repo="malaiwah/x-v1")
        check("a publish receipt outranks a stale unpublished status",
              MI.RunFacts(both).classify() == "published",
              MI.RunFacts(both).classify())

        no_capture = make_run(root, "refused-before-capture", status="failed",
                              dataset=False)
        check("a run with no sealed dataset is no-capture",
              MI.RunFacts(no_capture).classify() == "no-capture")

        # The two negative verdicts must stay distinct. This rung failed on
        # the first version of classify(), which returned no-capture for both
        # -- a real flaw found by the test rather than by reading the code.
        unknown = make_run(root, "dataset-but-no-receipt", status=None,
                           dataset=True)
        check("a run with a dataset but NO terminal receipt is unknown -- "
              "something was written and nothing says whether it sealed",
              MI.RunFacts(unknown).classify() == "unknown",
              MI.RunFacts(unknown).classify())

        bare = make_run(root, "neither-receipt-nor-dataset", status=None,
                        dataset=False)
        check("a run with neither a receipt nor a dataset is no-capture -- "
              "calling that unknown would understate what we know",
              MI.RunFacts(bare).classify() == "no-capture",
              MI.RunFacts(bare).classify())

        # A sealed-but-error-carrying verify must not read as verified.
        dirty = make_run(root, "sealed-with-errors",
                         status=MI.QUALIFIED_UNPUBLISHED, structural="sealed",
                         errors=3, destination="malaiwah/y-v1")
        check("a verify carrying errors is NOT verified",
              MI.RunFacts(dirty).verified is False)

        print()
        print("== discovery and exit codes ==")
        found = {r.name for r in MI.discover([root])}
        check("discovery finds every run directory",
              found == {"stranded-root", "candidate-no-destination",
                        "published-run", "published-stale-status",
                        "refused-before-capture", "dataset-but-no-receipt",
                        "neither-receipt-nor-dataset", "sealed-with-errors"},
              repr(sorted(found)))

        rc = MI.report(MI.discover([root]), show_all=False)
        check("a stranded run REFUSES with exit 3", rc == 3, "rc=%s" % rc)

    with tempfile.TemporaryDirectory(prefix="mi-rows-") as tmp:
        root = Path(tmp)
        make_run(root, "candidate", status=MI.QUALIFIED_UNPUBLISHED,
                 structural="sealed", errors=0, destination=None)
        rc = MI.report(MI.discover([root]), show_all=False)
        check("a needs-row run warns with exit 2, it does not refuse",
              rc == 2, "rc=%s" % rc)

    with tempfile.TemporaryDirectory(prefix="mi-clean-") as tmp:
        root = Path(tmp)
        make_run(root, "ok", status="published", structural="sealed", errors=0,
                 published_repo="malaiwah/ok-v1")
        rc = MI.report(MI.discover([root]), show_all=False)
        check("an estate with nothing outstanding exits 0", rc == 0,
              "rc=%s" % rc)

    with tempfile.TemporaryDirectory(prefix="mi-empty-") as tmp:
        rc = MI.main(["--runs-dir", tmp])
        check("an empty run root is not an error", rc == 0, "rc=%s" % rc)

    print()
    print("== the real estate, read-only ==")
    real = MI.discover([p for p in MI.DEFAULT_RUN_ROOTS if p.is_dir()])
    if not real:
        print("SKIP no run directory on this box -- nothing to cross-check "
              "(this is a fixture-only run)")
    else:
        # The audit found exactly two runs carrying the token by hand-walking
        # 99 directories. The tool must not silently find fewer.
        tokened = [r for r in real
                   if r.status == MI.QUALIFIED_UNPUBLISHED]
        check("every run carrying qualified-unpublished is surfaced, not "
              "silently dropped (%d found)" % len(tokened),
              all(r.classify() in ("STRANDED", "needs-row") for r in tokened),
              repr([(r.name, r.classify()) for r in tokened]))

    print()
    if FAILURES:
        print("selftest_measurement_inventory: %d FAILED" % len(FAILURES))
        return 1
    print("selftest_measurement_inventory: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
