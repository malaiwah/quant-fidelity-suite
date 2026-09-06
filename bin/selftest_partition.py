#!/usr/bin/env python3
"""Every selftest in this tree is categorised, and CI runs a tier by name.

    python3 bin/selftest_partition.py                 # the guard rungs
    python3 bin/selftest_partition.py --list fast     # one path per line
    python3 bin/selftest_partition.py --run fast      # execute that job

WHY THIS EXISTS, rather than a list of rungs pasted into a workflow.

`.github/workflows/selftest.yml` runs the hermetic partition of the estate on
every push.  The obvious way to write that is a hand-maintained list of steps
in the YAML -- and that list starts drifting the day someone adds a selftest,
silently, in the direction of less coverage.  The failure mode is not
hypothetical: 549 commits in thirty days produced 7 manual CI runs and zero
automatic ones, and the only test GitHub had ever executed was
`selftest_container.py` (`local/CiCoverage-report.md` §1.1, §1.3).

So the partition is DECLARED in `bin/SELFTEST-PARTITION.json` and ENFORCED
here: a selftest file that is in neither the declaration nor the explicit
exclusion list FAILS this rung, by name, with the remedy.  A new selftest must
therefore be categorised before it can land, and CI coverage cannot quietly lag
the estate.  It is the same shape as the provider-parity table in
`bin/fidelity/providers.py`, which the controller also reads, so a declaration
cannot disagree with behaviour.

The tiers, and what each one means:

  hermetic     offline, stock python3 with NO installs, no GPU, no credential,
               no spend -- and it ASSERTS SOMETHING under those conditions.
               `job` splits it into `fast` (the push gate's first verdict) and
               `full` (the same push, in parallel, slower rungs).
  torch        needs torch / numpy / safetensors / PyYAML.  Nightly.
  network      needs a live endpoint.  Nightly, and never in the push job: a
               red push build must never be able to mean "the Hub hiccuped".
  gpu          needs a real accelerator.
  pipeline     needs a `quant_pipeline` tree (`--pipeline-root`), which is
               neither a device nor a wheel and is not installable on the
               buildbox or on a runner.
  orphan-dead  must not be wired anywhere yet, with the reason.

A hermetic entry MUST declare a non-zero measured assertion count.  That rule
is the whole point of the honesty half of this file: `race mode (T15)`,
`hf-transformers capture (A1-A22)`, `layer-outer (L1-L14)` and `zero-floor
(T4)` all exit 0 on a bare runner having asserted 0, 0, 0 and 0 things
(measured, `local/CiCoverage-report.md` §2; `local/LocalCoverage-report.md`
§2.2 reached the same result on the FIDELITY_PYTHON axis: 146 assertions
collapsing to 6 while five rungs printed PASS).  Putting those in the push job
would launder four empty runs into a green tick, which is worse than not
running them, because it is indistinguishable from coverage.

Stock python3.9, no installs, no network, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
DECLARATION = HERE / "SELFTEST-PARTITION.json"
SCHEMA = "fidelity-suite/selftest-partition.v1"

# Where selftests live, and what a selftest looks like. Both halves are
# asserted against the declaration, so a new tree or a new extension is a
# visible edit here rather than a silent omission there.
ROOTS = ("bin", "engines/tools")
PATTERNS = ("selftest_*.py", "selftest_*.sh")

TIERS = ("hermetic", "torch", "network", "gpu", "pipeline", "orphan-dead")
JOBS = ("fast", "full")
# The fast job's selling point IS its wall clock: it is the first verdict a
# pushing agent sees. Raising this ceiling is a deliberate edit, not a drift.
FAST_BUDGET_SECONDS = 60.0
# And no single fast rung may dominate it. 5 s on a 2009 Xeon is ~2 s on a
# runner; the two whole-tree static sweeps (the python3.9 floor and the
# annotation-name rung, ~4 s each because they parse 150+ modules) are the
# most valuable rungs in the tier and belong in the first verdict.
FAST_RUNG_CEILING_SECONDS = 5.0

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label,
                          ("  -- " + detail) if (detail and not ok) else ""))
    if not ok:
        FAILED.append(label)


def load():
    with open(DECLARATION, encoding="utf-8") as handle:
        return json.load(handle)


def discover():
    """Every selftest file on disk, repo-relative, sorted."""
    found = []
    for root in ROOTS:
        for pattern in PATTERNS:
            for path in (SUITE / root).glob(pattern):
                found.append(str(path.relative_to(SUITE)))
    return sorted(found)


def entries(doc, tier=None, job=None):
    out = []
    for path, record in sorted(doc["selftests"].items()):
        if tier is not None and record["tier"] != tier:
            continue
        if job is not None and record.get("job") != job:
            continue
        out.append(path)
    return out


def rung_declaration(doc):
    print("[P1] the declaration is complete and current")
    on_disk = set(discover())
    declared = set(doc["selftests"])
    excluded = set(doc["exclusions"])

    uncategorised = sorted(on_disk - declared - excluded)
    check("P1a every selftest file on disk is categorised or explicitly "
          "excluded", not uncategorised,
          "uncategorised: %s -- add each one to bin/SELFTEST-PARTITION.json "
          "under \"selftests\" with its tier (%s), or to \"exclusions\" with "
          "the reason it is not a CI rung. A selftest CI does not know about "
          "is a selftest CI does not run."
          % (", ".join(uncategorised), "/".join(TIERS)))

    stale = sorted((declared | excluded) - on_disk)
    check("P1b the declaration names no file that does not exist",
          not stale, "declared but absent: %s" % ", ".join(stale))

    both = sorted(declared & excluded)
    check("P1c nothing is both categorised and excluded", not both,
          "%s" % ", ".join(both))

    check("P1d the schema string is the one this tool reads",
          doc.get("schema") == SCHEMA, "%r" % doc.get("schema"))
    check("P1e every exclusion carries a reason",
          all(str(v).strip() for v in doc["exclusions"].values()),
          "%s" % [k for k, v in doc["exclusions"].items() if not str(v).strip()])


def rung_records(doc):
    print("[P2] every record says which tier, and hermetic means it asserts "
          "something")
    bad_tier = [p for p, r in doc["selftests"].items() if r["tier"] not in TIERS]
    check("P2a every tier is one of %s" % ", ".join(TIERS), not bad_tier,
          "%s" % bad_tier)

    herm = [(p, r) for p, r in doc["selftests"].items() if r["tier"] == "hermetic"]
    check("P2b a hermetic entry names the job that runs it",
          all(r.get("job") in JOBS for _p, r in herm),
          "%s" % [p for p, r in herm if r.get("job") not in JOBS])
    check("P2c a hermetic entry declares its measured wall clock",
          all(isinstance(r.get("seconds_xeon_x5570"), (int, float))
              and r["seconds_xeon_x5570"] > 0 for _p, r in herm),
          "%s" % [p for p, r in herm
                  if not isinstance(r.get("seconds_xeon_x5570"), (int, float))])
    # The anti-vacuity rule. A rung that exits 0 having asserted nothing is
    # not coverage, and a job full of them is worse than an empty job because
    # it reads as green.
    check("P2d a hermetic entry declares a NON-ZERO measured assertion count "
          "on a bare runner",
          all(isinstance(r.get("assertions_measured"), int)
              and r["assertions_measured"] > 0 for _p, r in herm),
          "%s -- a rung that asserts 0 things without its dependency belongs "
          "in the nightly tier with the measurement as its reason"
          % [p for p, r in herm if not r.get("assertions_measured")])

    others = [(p, r) for p, r in doc["selftests"].items() if r["tier"] != "hermetic"]
    check("P2e a non-hermetic entry says WHY, so the exclusion is reviewable",
          all(len(str(r.get("reason", "")).strip()) > 20 for _p, r in others),
          "%s" % [p for p, r in others
                  if len(str(r.get("reason", "")).strip()) <= 20])
    check("P2f a non-hermetic entry names no CI job",
          not [p for p, r in others if r.get("job")],
          "%s" % [p for p, r in others if r.get("job")])


def rung_budget(doc):
    print("[P3] the fast job stays fast, and the push jobs stay free")
    fast = [(p, r) for p, r in doc["selftests"].items()
            if r["tier"] == "hermetic" and r.get("job") == "fast"]
    total = round(sum(r["seconds_xeon_x5570"] for _p, r in fast), 2)
    over = [p for p, r in fast
            if r["seconds_xeon_x5570"] > FAST_RUNG_CEILING_SECONDS]
    check("P3a no single fast rung declares more than %.1f s"
          % FAST_RUNG_CEILING_SECONDS, not over,
          "%s -- move it to job \"full\"" % over)
    check("P3b the fast job's declared total is inside its %.0f s budget "
          "(declared %.1f s over %d rungs)"
          % (FAST_BUDGET_SECONDS, total, len(fast)),
          total <= FAST_BUDGET_SECONDS, "%.1f s" % total)
    # Stated as a property of the declaration rather than of the YAML: the
    # workflow runs `--run fast` and `--run full`, and both resolve only
    # through the hermetic tier, so a networked, GPU or paid rung cannot
    # reach a push job without changing a tier here.
    for job in JOBS:
        names = [p for p, r in doc["selftests"].items()
                 if r.get("job") == job and r["tier"] != "hermetic"]
        check("P3d job %r contains only hermetic rungs" % job, not names,
              "%s" % names)


def rung_runnable(doc):
    print("[P4] every hermetic rung is a file this repo can execute")
    missing = []
    for path in entries(doc, tier="hermetic"):
        target = SUITE / path
        if not target.is_file():
            missing.append(path)
    check("P4a every hermetic rung exists", not missing, "%s" % missing)
    check("P4b every hermetic rung is a .py or a .sh this tool knows how to "
          "invoke",
          all(p.endswith(".py") or p.endswith(".sh")
              for p in entries(doc, tier="hermetic")))


def command_for(path):
    if path.endswith(".sh"):
        return ["bash", str(SUITE / path)]
    # sys.executable, not "python3": CI pins the interpreter (3.9, the floor
    # bin/ and registry/ must hold) and this must run under exactly that one.
    return [sys.executable, str(SUITE / path)]


def is_internal_skip(line):
    """A rung's own SKIP line, not a PASS whose label mentions skipping.

    `bin/selftest_battery_harness.py` asserts a rung named "SKIP_RE catches
    every skip format", and a substring match reported that PASS as a skip.
    A verdict token at the start of the line settles it: a line whose first
    word is PASS, ok or FAIL is a result, whatever its label says.
    """
    stripped = line.strip()
    if not stripped:
        return False
    head = stripped.split(None, 1)[0].strip("[]")
    if head in ("PASS", "ok", "FAIL", "no"):
        return False
    return "SKIP" in stripped or "[skip]" in stripped


def run_job(doc, job):
    """Execute a job's rungs, keeping each rung's own SKIP lines visible."""
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    paths = (entries(doc, tier="hermetic", job=job) if job in JOBS
             else entries(doc, tier=job))
    if not paths:
        print("selftest_partition: %r selects no rung -- refusing to report a "
              "job that ran nothing as a pass" % job)
        return 2
    failures = []
    skips = []
    started = time.time()
    for path in paths:
        if annotate:
            print("::group::%s" % path)
        began = time.time()
        proc = subprocess.run(command_for(path), cwd=str(SUITE),
                              capture_output=True, text=True)
        took = time.time() - began
        body = (proc.stdout or "") + (proc.stderr or "")
        if annotate:
            print(body)
            print("::endgroup::")
        internal = [line.strip() for line in body.splitlines()
                    if is_internal_skip(line)]
        verdict = "ok  " if proc.returncode == 0 else "FAIL"
        print("  %s %6.1fs  %s" % (verdict, took, path))
        # An outer PASS must never hide an internal skip: that is how 140
        # assertions went missing while the battery printed 0 skipped
        # (local/LocalCoverage-report.md §3).
        for line in internal:
            print("         skip: %s" % line[:160])
            skips.append("%s: %s" % (path, line[:160]))
        if proc.returncode != 0:
            failures.append(path)
            tail = [ln for ln in body.splitlines() if ln.strip()][-12:]
            if annotate:
                print("::error file=%s::%s exited %d: %s"
                      % (path, path, proc.returncode,
                         (tail[-1] if tail else "no output")[:400]))
            else:
                for line in tail:
                    print("       | %s" % line[:200])
    print("")
    print("selftest_partition %s: %d rungs, %d failed, %d internal skips, "
          "%.1fs" % (job, len(paths), len(failures), len(skips),
                     time.time() - started))
    if skips:
        print("internal skips (a dependency this runner lacks, named so the "
              "verdict is not read as coverage):")
        for line in skips:
            print("  - %s" % line)
    if failures:
        print("FAILED %d:" % len(failures))
        for path in failures:
            print("  - %s" % path)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="the selftest partition: declare it, enforce it, run it")
    parser.add_argument("--list", metavar="TIER_OR_JOB",
                        help="print the paths a tier or job selects")
    parser.add_argument("--run", metavar="TIER_OR_JOB",
                        help="execute a tier or job and return its verdict")
    args = parser.parse_args(argv)
    doc = load()
    if args.list:
        selector = args.list
        paths = (entries(doc, tier="hermetic", job=selector)
                 if selector in JOBS else entries(doc, tier=selector))
        if not paths:
            sys.stderr.write("selftest_partition: %r selects no rung\n"
                             % selector)
            return 2
        for path in paths:
            print(path)
        return 0
    if args.run:
        return run_job(doc, args.run)

    rung_declaration(doc)
    rung_records(doc)
    rung_budget(doc)
    rung_runnable(doc)
    print("")
    if FAILED:
        print("FAILED %d:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    counts = {}
    for record in doc["selftests"].values():
        counts[record["tier"]] = counts.get(record["tier"], 0) + 1
    print("selftest_partition: all rungs pass (%d selftests categorised: %s; "
          "%d excluded)"
          % (len(doc["selftests"]),
             ", ".join("%s %d" % (k, counts[k]) for k in sorted(counts)),
             len(doc["exclusions"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
