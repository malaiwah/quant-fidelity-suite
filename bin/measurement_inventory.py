#!/usr/bin/env python3
"""List every local run by what it PRODUCED, not by whether it exited 0.

Why this exists, and it is a money reason rather than a tidiness one.

On 2026-08-31 the controller destroyed a sealed, twice-validated MiniMax-M3
root dataset at teardown -- **$6.59 of GPU time and the only copy of the
evidence** -- because nothing published or preserved a root first
(REVIEW-DEFERRED ROOT-1; the figure is the authors' contemporaneous one, cited
from bin/selftest_root_publish.py:6-9 and measure_cloud.py:347, and is NOT a
billing document). That ordering hole is closed: `--publish-root-to` uploads
before teardown can run, and teardown refuses to destroy a verified-but-
unpublished root.

What was NOT closed is the reporting hole, and the 2026-09-06 spend audit
found it the expensive way. `qualified-unpublished` is a first-class VALIDATED
state -- `bin/fidelity/resultsink.py` keys on it in about a dozen places,
including two symmetric refusals that such a result may not carry a
publication receipt; `jobcontract.py` binds `publication_mode` to it;
`container_entry.py` refuses local root publication with it; five selftests
assert on it -- and `registry/` contains ZERO occurrences of it. So every
component that could reject a malformed one does, and NOTHING anywhere
enumerates runs by that status to say "here is a verified capture waiting for
a row". Walking 99 run directories by hand was the only mechanism that found
them, which is why `flashA-k2-run1` sat for a day.

That is the same defect as a battery printing "0 skipped" while twelve
sub-rungs sit out: **the state is known to the code and invisible to the
operator.** The fix shape is the same too -- one token, asserted AND reported.

This tool reads only files. It contacts no provider, no Hub and no registry,
spends nothing, and writes nothing.

Exit codes follow the tree's convention:
  0  nothing stranded
  2  a run needs a registry row, or a record is missing, but nothing is at
     risk of being lost (warnings only)
  3  REFUSED: a verified capture is publishable and unpublished -- the
     MiniMax shape recurring, and the one case where evidence can still be
     destroyed by deleting a directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Run directories this project has actually used. Both spellings appear in
# receipts; on this workstation they are the same tree.
DEFAULT_RUN_ROOTS = (
    Path.home() / "code" / "fidelity-runs",
    Path.home() / "fidelity-runs",
)

QUALIFIED_UNPUBLISHED = "qualified-unpublished"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class RunFacts:
    """What one run directory can be shown to have produced."""

    __slots__ = ("path", "name", "status", "scientific", "structural",
                 "error_count", "capture_digest", "dataset_sha256",
                 "destination", "published_repo", "published_revision",
                 "verified_after_publish", "has_dataset")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name

        terminal = _load(path / "terminal-receipt.json") or {}
        self.status = terminal.get("combined_status")
        self.scientific = terminal.get("scientific_status")

        verify = _load(path / "local-verify.json") or {}
        self.structural = verify.get("structural_status")
        self.error_count = verify.get("error_count")

        dataset = _load(path / "result" / "dataset" / "fidelity-dataset.json") or {}
        self.has_dataset = bool(dataset)
        self.dataset_sha256 = dataset.get("dataset_sha256")
        self.capture_digest = (dataset.get("capture") or {}).get(
            "capture_content_digest")

        # The destination is sealed INSIDE the qualification receipt. A null
        # here means no later --repo can satisfy the publish gate, by design.
        qual = _load(path / "result" / "receipts" / "root-qualification.json") or {}
        self.destination = qual.get("destination_repository")

        publish = _load(path / "result" / "receipts" / "publish-root.json") or {}
        self.published_repo = publish.get("repository")
        self.published_revision = publish.get("revision")
        self.verified_after_publish = publish.get("verified_after_publish")

    @property
    def verified(self) -> bool:
        return self.structural == "sealed" and self.error_count == 0

    def classify(self) -> str:
        """One of: published, STRANDED, needs-row, no-capture, unknown.

        The two negative verdicts are deliberately NOT interchangeable, and
        the selftest caught them being collapsed here. A directory with
        neither a terminal receipt nor a sealed dataset demonstrably produced
        no capture -- calling that "unknown" would understate what we know.
        A directory WITH a dataset but no terminal receipt is genuinely
        unknown: something was written and nothing says whether it sealed,
        verified or was abandoned. That one wants a human, and saying so is
        the whole point of a tool built because a state was invisible.
        """
        if self.published_repo:
            # A publish receipt is evidence the upload happened; it outranks a
            # stale status token, which must not resurrect a published run.
            return "published"
        if self.status == QUALIFIED_UNPUBLISHED:
            # Sealed and not published. The destination sealed INSIDE the
            # qualification decides whether publishing is even reachable.
            return "STRANDED" if self.destination else "needs-row"
        if not self.has_dataset:
            return "no-capture"
        return "unknown"


def discover(roots: List[Path]) -> List[RunFacts]:
    seen: Dict[Path, RunFacts] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            resolved = child.resolve()
            if resolved in seen:
                continue
            if not (child / "terminal-receipt.json").is_file() \
                    and not (child / "result").is_dir():
                continue
            seen[resolved] = RunFacts(child)
    return list(seen.values())


REASONS = {
    "STRANDED": ("REFUSED: verified, publishable and UNPUBLISHED -- this is "
                 "the shape that cost $6.59 and the only copy of a MiniMax-M3 "
                 "root. Publish it before anything deletes the directory."),
    "needs-row": ("needs a registry row. The capture is verified and its "
                  "number is admissible, but its sealed qualification names no "
                  "destination_repository, so the dataset is structurally "
                  "unpublishable and publication is optional for a candidate."),
    "published": "published, with a publish receipt.",
    "no-capture": "no sealed dataset here.",
    "unknown": "no terminal receipt -- cannot say what this produced.",
}


def report(runs: List[RunFacts], *, show_all: bool) -> int:
    buckets: Dict[str, List[RunFacts]] = {}
    for run in runs:
        buckets.setdefault(run.classify(), []).append(run)

    print("== %d run directory(ies) ==" % len(runs))
    for kind in ("STRANDED", "needs-row", "published", "no-capture", "unknown"):
        rows = buckets.get(kind) or []
        if not rows:
            continue
        print("  %-11s %d" % (kind, len(rows)))
    print()

    for kind in ("STRANDED", "needs-row"):
        rows = buckets.get(kind) or []
        if not rows:
            continue
        print("== %s -- %s ==" % (kind, REASONS[kind]))
        for run in rows:
            print("  %s" % run.name)
            print("     path                   %s" % run.path)
            print("     combined_status        %s" % run.status)
            print("     structural_status      %s (errors: %s)"
                  % (run.structural, run.error_count))
            print("     capture_content_digest %s" % run.capture_digest)
            print("     dataset_sha256         %s" % run.dataset_sha256)
            print("     destination_repository %s" % run.destination)
        print()

    if show_all:
        published = buckets.get("published") or []
        if published:
            print("== published ==")
            for run in published:
                print("  %-28s %s@%s verified_after_publish=%s"
                      % (run.name, run.published_repo,
                         str(run.published_revision)[:12],
                         run.verified_after_publish))
            print()

    if buckets.get("STRANDED"):
        return 3
    if buckets.get("needs-row"):
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="List local runs by what they produced. Reads files only.")
    parser.add_argument("--runs-dir", action="append", default=None,
                        help="run directory root; repeatable "
                             "(default: ~/code/fidelity-runs, ~/fidelity-runs)")
    parser.add_argument("--all", action="store_true",
                        help="also list the published runs")
    args = parser.parse_args(argv)

    roots = [Path(p).expanduser() for p in (args.runs_dir or [])] \
        or list(DEFAULT_RUN_ROOTS)
    present = [r for r in roots if r.is_dir()]
    if not present:
        print("no run directory found; looked in: %s"
              % ", ".join(str(r) for r in roots), file=sys.stderr)
        return 0
    return report(discover(present), show_all=args.all)


if __name__ == "__main__":
    raise SystemExit(main())
