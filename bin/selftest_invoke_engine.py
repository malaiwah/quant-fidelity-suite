#!/usr/bin/env python3
"""CLI-17: the engine's output must reach the log WHILE it runs.

`fidelity.common.run(...)` buffers stdout and stderr and hands them back after
the child EXITS. Launching the engine through it meant a 79-minute capture
wrote exactly one line to its stage log -- the argv -- and nothing else until
it was over, so a healthy run and a wedged one were indistinguishable from the
file `measure_cloud.py` tells the operator to inspect
(`tail -50 <fs>/logs/*.log`). On a rented GPU that is the difference between
noticing a stall in minutes and paying for it for an hour (JOURNAL lesson 43).

This asserts the BEHAVIOUR, not the source text: a chatty child is launched
through `invoke_engine.spawn_streaming` and its output must be observable in
the log file BEFORE the child exits. A rung that grepped for `capture_output`
would pass just as happily on some future call that captured by another
spelling, and would go red on a refactor that kept the property.

Offline, stdlib only, no torch, no network, nothing created outside a scratch
directory. Runs in a couple of seconds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import invoke_engine as IE  # noqa: E402

FAILURES: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print("PASS %s" % label)
    else:
        FAILURES.append(label)
        print("FAIL %s%s" % (label, (" -- " + detail) if detail else ""))


# A child that speaks, then keeps running. The sleep has to outlast the poll
# below, or the test would pass on a buffered launch simply because the child
# had already exited.
CHATTY = (
    "import sys, time\n"
    "sys.stdout.write('layer 0 of 45\\n'); sys.stdout.flush()\n"
    "sys.stderr.write('a warning while running\\n'); sys.stderr.flush()\n"
    "time.sleep(6)\n"
    "sys.stdout.write('done\\n')\n"
)


def main(argv: List[str] = None) -> int:
    print("== CLI-17: engine output streams, it is not buffered to exit ==")
    with tempfile.TemporaryDirectory(prefix="cli17-") as tmp:
        root = Path(tmp)
        child = root / "chatty.py"
        child.write_text(CHATTY, encoding="utf-8")
        log = root / "measure-run-1.log"

        # Stand in for the stage driver: this process's own stdout is a file,
        # exactly as `stage_measure.sh` tees it. If the launch inherits the
        # streams, the child writes here live.
        saved_out, saved_err = os.dup(1), os.dup(2)
        handle = os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        rc = None
        early = b""
        try:
            os.dup2(handle, 1)
            os.dup2(handle, 2)
            proc_rc = []

            import threading

            def launch():
                proc_rc.append(
                    IE.spawn_streaming([sys.executable, str(child)], dict(os.environ)))

            thread = threading.Thread(target=launch)
            thread.start()
            # The child sleeps 6 s. Read at 2 s: with inherited streams the
            # first lines are already on disk; with a capturing launch the
            # file is still empty.
            time.sleep(2.0)
            early = log.read_bytes()
            thread.join(timeout=30)
            rc = proc_rc[0] if proc_rc else None
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(handle)
            os.close(saved_out)
            os.close(saved_err)

        text = early.decode("utf-8", "replace")
        check("the child's stdout is in the log BEFORE it exits",
              "layer 0 of 45" in text, repr(text[:120]))
        check("and its stderr too, so a warning is not held until exit",
              "a warning while running" in text, repr(text[:120]))
        check("the launch still returns the child's exit code",
              rc == 0, "rc=%r" % (rc,))

        final = log.read_text(encoding="utf-8", errors="replace")
        check("the later output arrives as well, so streaming did not drop it",
              "done" in final, repr(final[-80:]))

    print()
    print("== the launch does not go through the buffering helper ==")
    # A containment check, in the spirit of the MKL-01 one: the behavioural
    # rung above proves TODAY's launch streams, and this catches a future
    # rewrite that routes the engine back through the helper whose whole
    # documented behaviour is to hand the output back after exit.
    src = (Path(IE.__file__)).read_text(encoding="utf-8")
    body = src.split("def spawn_streaming", 1)[-1]
    check("spawn_streaming does not set text= or capture_output=",
          "capture_output" not in body and "text=True" not in body)
    check("and it passes the parent's streams through explicitly",
          "stdout=None" in body and "stderr=None" in body)

    print()
    print("== NUM-16: the authored profile is the authority ==")
    # engines.json advertises knobs that no composer fills from a job. Before
    # 2026-09-07 a job naming one was SILENTLY IGNORED, which is the worst of
    # the three possible behaviours: the operator believes the value took
    # effect and the receipt cannot show that it did not. It refuses now.
    import json as _json
    from fidelity.engines import load_engines
    from fidelity.jobcontract import JobContractError

    with open(str(Path(IE.__file__).parent / "engines.json"),
              encoding="utf-8") as fh:
        owned = _json.load(fh).get("profile_authoritative_flags") or []
    check("NUM-16 engines.json declares which flags the profile owns",
          bool(owned) and "ep_emulate" in owned and "inventory" in owned,
          repr(owned))

    engine = load_engines()["streaming"]
    base = {"role": "quant", "lane": "streaming",
            "profile": {"profile_id": "k6", "lane": "streaming",
                        "source": "x", "surface": "y", "bits": 6.0}}

    def gate(runtime):
        """Did the NUM-16 gate specifically refuse this runtime?"""
        try:
            IE._invocation_values(dict(base, runtime=runtime), "streaming",
                                  engine)
        except JobContractError as exc:
            return "authored profile owns" in str(exc)
        return False

    check("NUM-16 a job steering ep_emulate is REFUSED, not ignored",
          gate({"device": "cuda", "ep_emulate": 8}))
    check("NUM-16 a job steering inventory is REFUSED, not ignored",
          gate({"inventory": "/tmp/x"}))
    check("NUM-16 an ordinary runtime is NOT refused by this gate -- the rule "
          "is narrow, and no existing job.json carries an owned key",
          not gate({"device": "cuda", "reduce_order": "layer"}))
    # A guard that refuses everything would pass the three rungs above while
    # being useless, so assert the complement too.
    check("NUM-16 the owned set does not swallow the keys a job legitimately "
          "sets", not ({"device", "decode_cache", "decode_threads",
                        "reduce_order", "reader_threads"} & set(owned)),
          repr(sorted(set(owned))))

    print()
    if FAILURES:
        print("selftest_invoke_engine: %d FAILED" % len(FAILURES))
        return 1
    print("selftest_invoke_engine: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
