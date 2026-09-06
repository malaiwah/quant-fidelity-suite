#!/usr/bin/env python3
"""Test the battery harness itself -- the one file that had no test.

`bin/selftest_all.sh` decides what "done" means for this repository, and until
today nothing checked it. LocalCoverage measured the consequence: the battery
printed "86 passed, 0 failed, 0 skipped" while five rungs ran 6 of their 146
assertions, and it contained the exact `A && B || C` construct its own guard
rung forbids elsewhere. A harness that mis-selects an interpreter or swallows
a skip does not fail loudly; it reports green, which is worse.

Four invariants, each with a defect behind it:

  1. INTERPRETER AGREEMENT. Inside `if have_module "$X" ...; then`, every rung
     in the block must be invoked with `"$X"`. On 2026-09-06 the gguf gate was
     moved to $TPY while its rung still ran `"$PY"`, so on any box where torch
     lives in a venv and FIDELITY_PYTHON is unset the gate passed and the rung
     died with ModuleNotFoundError -- an honest SKIP converted into a red whose
     only meaning is "wrong interpreter". That is failure #4's signature,
     reintroduced by the commit that fixed its other three instances.

  2. NO `A && B || C` RUNGS. C also runs when A succeeds and B fails, so the
     construct cannot express if-then-else. `bin/selftest_shell_guards.sh`'s
     own header records this class as already paid for once, and the battery
     was still doing it at line 467.

  3. EVERY RUNG'S OUTPUT IS CAPTURED. `t()` must write per-rung logs under a
     log directory rather than reusing one file, or AGENTS.md's instruction to
     "read the internal SKIP lines" is physically impossible and the summary
     can claim "0 skipped" while a whole dependency tier sat out.

  4. NO LINTER-NAMED COMMENT OPENERS. A comment line beginning with the
     linter's name is parsed as a directive and silently disables checking for
     the entire file -- a self-inflicted blind spot with no diagnostic.

Pure text and regex over the shell source. No shell executed, no network, no
torch, nothing created. Runs in milliseconds.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
BATTERY = ROOT / "bin" / "selftest_all.sh"

GATE = re.compile(r'^\s*if\s+.*have_module\s+"\$(\w+)"')
INVOKE = re.compile(r'"\$(\w+)"\s+\S*\.py\b')
BLOCK_END = re.compile(r'^\s*(else|elif|fi)\b')
# `cmd && t "..." || t "..."` -- the shape that cannot be if-then-else.
AND_OR = re.compile(r'&&.*\|\|')
LOG_REUSE = re.compile(r'>\s*"\$TMP/out\.log"')

FAILURES: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print("PASS %s" % label)
    else:
        FAILURES.append(label)
        print("FAIL %s%s" % (label, (" -- " + detail) if detail else ""))


def interpreter_mismatches(lines: List[str]) -> List[str]:
    """Rungs invoked with a different interpreter than their gate tested.

    Block-scoped on purpose: a naive lookahead reports the `TPY="$VPY"` /
    `have_module "$TPY" torch || TPY="$PY"` derivation lines as mismatches
    against the next unrelated rung. Those are not gates -- they are how $TPY
    is chosen -- so only `if ... have_module` OPENERS start a block.
    """
    out: List[str] = []
    for i, line in enumerate(lines):
        gate = GATE.match(line)
        if not gate:
            continue
        want = gate.group(1)
        for j in range(i + 1, len(lines)):
            if BLOCK_END.match(lines[j]):
                break
            used = INVOKE.search(lines[j])
            if used and used.group(1) != want:
                out.append(
                    "line %d invokes $%s inside a block gated on $%s: %s"
                    % (j + 1, used.group(1), want, lines[j].strip()[:70]))
    return out


def _join_continuations(lines: List[str]) -> List[Tuple[int, str]]:
    """Fold `\\`-continued shell lines into one logical line.

    Without this the `A && B || C` check is weaker than it claims: the real
    defect at origin/main was written as

        "$PY" - <<'PYEOF' && t "..." 0 true \\
          || t "..." 0 false

    and a per-physical-line regex scores it 0. Verified against 4681e30,
    where the naive version reported no findings and the defect was present.
    """
    out: List[Tuple[int, str]] = []
    buf = ""
    start = 1
    for i, line in enumerate(lines, 1):
        if not buf:
            start = i
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        out.append((start, (buf + stripped).strip()))
        buf = ""
    if buf:
        out.append((start, buf.strip()))
    return out


def and_or_rungs(lines: List[str]) -> List[str]:
    out: List[str] = []
    for lineno, logical in _join_continuations(lines):
        if logical.startswith("#"):
            continue
        if AND_OR.search(logical) and re.search(r'\bt\s+"', logical):
            out.append("line %d: %s" % (lineno, logical[:70]))
    return out


def directive_comments(lines: List[str]) -> List[str]:
    out: List[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        first = body.split(" ", 1)[0].lower() if body else ""
        if first in ("shellcheck", "ruff", "pyright", "mypy", "flake8"):
            out.append("line %d parses as a %s DIRECTIVE, not prose: %s"
                       % (i + 1, first, stripped[:60]))
    return out


def audit(text: str) -> Tuple[List[str], List[str], List[str], bool, bool]:
    lines = text.splitlines()
    return (interpreter_mismatches(lines),
            and_or_rungs(lines),
            directive_comments(lines),
            bool(re.search(r'LOG_DIR=', text)) and not LOG_REUSE.search(text),
            "inner_skip" in text)


def main(argv: Optional[List[str]] = None) -> int:
    if not BATTERY.is_file():
        print("selftest_battery_harness: %s not found" % BATTERY)
        return 1
    text = BATTERY.read_text(encoding="utf-8")
    mismatch, andor, directives, per_rung_logs, counts_skips = audit(text)

    print("== the battery harness, %d lines ==" % len(text.splitlines()))
    for m in mismatch:
        print("   %s" % m)
    check("every rung is invoked with the interpreter its gate tested",
          not mismatch, "%d mismatch(es)" % len(mismatch))
    for a in andor:
        print("   %s" % a)
    check("no rung is dispatched through `A && B || C`",
          not andor, "%d found" % len(andor))
    for d in directives:
        print("   %s" % d)
    check("no comment line opens with a linter name (SC1073 blind spot)",
          not directives, "%d found" % len(directives))
    check("t() writes one log per rung instead of reusing a single file",
          per_rung_logs)
    check("the summary counts skips that happened INSIDE passing rungs",
          counts_skips)

    # A gate nobody has seen fail is a gate nobody has tested. Each check must
    # go red on its own synthetic input.
    print()
    print("== each invariant can refuse ==")
    bad_gate = ('if have_module "$TPY" torch; then\n'
                '  t "x" 0 "$PY" bin/selftest_x.py\n'
                'fi\n')
    check("refuses a gate/invocation interpreter mismatch",
          len(interpreter_mismatches(bad_gate.splitlines())) == 1,
          repr(interpreter_mismatches(bad_gate.splitlines())))

    good_gate = ('if have_module "$TPY" torch; then\n'
                 '  t "x" 0 "$TPY" bin/selftest_x.py\n'
                 'fi\n')
    check("accepts a gate and invocation that agree",
          interpreter_mismatches(good_gate.splitlines()) == [])

    derivation = ('TPY="$VPY"\n'
                  'have_module "$TPY" torch || TPY="$PY"\n'
                  't "unrelated" 0 "$PY" bin/selftest_fit.py\n')
    check("does NOT flag the $TPY derivation lines as gates",
          interpreter_mismatches(derivation.splitlines()) == [],
          repr(interpreter_mismatches(derivation.splitlines())))

    andor_bad = '"$PY" - <<EOF && t "x" 0 true || t "x" 0 false\n'
    check("refuses an `A && B || C` rung",
          len(and_or_rungs(andor_bad.splitlines())) == 1)

    directive_bad = "# shellcheck flags this as SC2015 and it is prose\n"
    check("refuses a comment that opens with a linter name",
          len(directive_comments(directive_bad.splitlines())) == 1)

    prose_ok = "# This is flagged by shellcheck as SC2015; prose is fine here\n"
    check("accepts the same prose when the linter is not the first word",
          directive_comments(prose_ok.splitlines()) == [])

    print()
    if FAILURES:
        print("selftest_battery_harness: %d FAILED" % len(FAILURES))
        return 1
    print("selftest_battery_harness: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
