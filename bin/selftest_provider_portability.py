#!/usr/bin/env python3
"""A second provider must not be able to leak an instance.

Every bug found while porting to RunPod was the same bug: a JarvisLabs
*representation* treated as a universal truth. None of them were about the
measurement, and three of them could have left a billing instance running.

  * machine ids are integers        -> `int(pod_id)` raised AFTER the pod was
                                       created, so the controller died holding
                                       an instance it had never adopted
  * the running state is "Running"  -> RunPod says "RUNNING", so every healthy
                                       poll counted as not-running and the
                                       controller declared a PREEMPTION and
                                       tore down a box mid-bootstrap
  * ids compare as ints in a set    -> the "is it really gone?" check would
                                       report a live instance as destroyed

These are offline: no provider is contacted.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_cloud as mc                                # noqa: E402
from fidelity.jlapi import Instance                       # noqa: E402
from fidelity.runpodapi import RunPod, RunPodError        # noqa: E402

FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


def _fixture_key_file():
    """A 0600 fixture key path, injected so no rung reads a credential.

    Returns a path, not a key: the loader's contract is a 0600 regular file
    owned by the caller, and the 36 characters inside only have to survive
    that shape check.  Nothing is sent anywhere -- every provider here is
    constructed dry.
    """
    root = tempfile.mkdtemp(prefix="portability-key-")
    atexit.register(shutil.rmtree, root, True)
    path = os.path.join(root, "runpod-fixture-key")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("F" * 36 + "\n")
    return path


def inst(mid, status="Running", name="fidcloud-x"):
    i = Instance.from_json({"machine_id": 0, "status": status, "name": name})
    i.machine_id = mid
    return i


print("== an opaque machine id survives every hop ==")
for created, want in [
        ({"machine_id": 483634}, 483634),
        ({"machine_id": "483634"}, 483634),
        ({"pod_id": "uqlk708fxtoz8n"}, "uqlk708fxtoz8n"),
        ({"id": "yytmxz8vhh1qfk"}, "yytmxz8vhh1qfk"),
        ({}, None),
        (None, None),
]:
    got = mc._machine_id_of(created)
    check("_machine_id_of(%r) -> %r" % (created, want), got == want)

check("a non-numeric id is NOT dropped (the leak this caused)",
      mc._machine_id_of({"machine_id": "abc123xyz"}) == "abc123xyz")

print("\n== the running state is spelled differently per provider ==")
for status, want in [("Running", True), ("RUNNING", True), ("running", True),
                     ("ready", True), ("Paused", False), ("EXITED", False),
                     ("TERMINATED", False), ("", False)]:
    check("status %-12r -> running=%s" % (status, want),
          mc._is_running(inst(1, status)) is want)
check("None is not running", mc._is_running(None) is False)

print("\n== 'is it really gone?' compares like with like ==")


class FakeJL:
    provider = "runpod"

    def __init__(self, alive):
        self._alive = alive

    def list_instances(self):
        return [inst(m) for m in self._alive]


class Con:
    def __init__(self):
        self.lines = []

    def __getattr__(self, _):
        return lambda *a, **k: None


td = mc.Teardown(FakeJL(["uqlk708fxtoz8n"]), Con(), mc.Path("."))
check("a LIVE opaque-id instance is not reported gone",
      td._confirm_gone("uqlk708fxtoz8n") is False)
check("a destroyed opaque-id instance is reported gone",
      td._confirm_gone("gone000000") is True)

td_int = mc.Teardown(FakeJL([483634]), Con(), mc.Path("."))
check("the integer case still works", td_int._confirm_gone(483634) is False)
check("...and its negative", td_int._confirm_gone(999999) is True)

print("\n== the run root is provider-specific, and always exported ==")


class JLish:
    provider = "jarvislabs"


rp, jl = mc.Teardown(FakeJL([]), Con(), mc.Path(".")), \
    mc.Teardown(JLish(), Con(), mc.Path("."))
check("runpod runs under /workspace", rp.fs_root.startswith("/workspace"))
check("jarvislabs runs under /home/jl_fs", jl.fs_root.startswith("/home/jl_fs"))

# EVERY backend's run root must be WRITABLE BY THE USER IT LOGS IN AS. Lambda
# is the one that is not root: `ubuntu`, on an image whose `/home` is root-owned
# 0755 -- so the `/home/jl_fs` default is EACCES and `mkdir -p
# /home/jl_fs/fidelity/logs` kills the run during the bundle upload, two
# minutes in, on every rental. Observed on a live gpu_1x_gh200 before this
# check existed; the box had already been paid for.
for _p in ("jarvislabs", "runpod", "vast", "lambda"):
    _t = mc.Teardown(mc._make_provider(_p, dry=True), Con(), mc.Path("."))
    _user = getattr(mc._make_provider(_p, dry=True), "ssh_user", "root")
    # A non-root user may only be given a root under ITS OWN home (or a mount
    # the image gives it, which a backend states via `run_base`).
    _ok = _user == "root" or _t.fs_root.startswith("/home/%s/" % _user) \
        or _t.fs_root.startswith("/workspace/")
    check("%s (%s@) gets a run root that user can create: %s"
          % (_p, _user, _t.fs_root), _ok)
    check("...and its engine root sits beside it", _ok and
          _t.engine_root.rsplit("/", 1)[0] == _t.fs_root.rsplit("/", 1)[0])
    _env = mc._stage_env(_t)
    check("...and the stage env exports that root, not the default",
          _t.fs_root in _env and (_user == "root" or "/home/jl_fs" not in _env))
# The root the controller CHOOSES, not just the ones it exports: it used to be
# `/home/jl_fs/glm53-k6` -- one provider and one model baked into the same
# path, on a box that may be measuring MiniMax.
check("neither engine root names a model or a campaign (%s / %s)"
      % (rp.engine_root, jl.engine_root),
      not any(t in (rp.engine_root + jl.engine_root).lower()
              for t in ("glm", "qwen", "minimax", "deepseek", "fruit",
                        "k6", "k8", "tr3")))
for t in (rp, jl):
    env = mc._stage_env(t)
    check("%s stage env names both roots" % t.fs_root.split("/")[1],
          "FIDELITY_FS_ROOT=" in env and "FIDELITY_ENGINE_ROOT=" in env
          and t.fs_root in env)

print("\n== the backend switch ==")
check("--provider jarvislabs builds the jl backend",
      type(mc._make_provider("jarvislabs", dry=True)).__name__ == "JL")
check("--provider runpod builds the runpod backend",
      type(mc._make_provider("runpod", dry=True)).__name__ == "RunPod")
rpb = mc._make_provider("runpod", dry=True)
check("the runpod backend declares its provider name", rpb.provider == "runpod")
for m in ("create", "destroy", "exec", "exec_stdout", "upload", "download",
          "list_instances", "get", "gpus", "balance", "run_job", "run_status",
          "run_logs", "fs_create", "fs_delete", "available", "require",
          "pause", "resume"):
    check("runpod implements %s()" % m, callable(getattr(rpb, m, None)))
check("destroy is a no-op under dry",
      rpb.destroy("x").get("dry_run") is True)

print("\n== storage that dies with the instance must be sized at create ==")
for name, sep in (("jarvislabs", True), ("runpod", False), ("vast", False),
                  ("lambda", False)):
    prov = mc._make_provider(name, dry=True)
    check("%-10s separable_storage=%s" % (name, sep),
          getattr(prov, "separable_storage", True) is sep)
# 100 GB is right ONLY where the big disk is a separate filesystem. Getting
# this wrong is not a create error: it is "No space left on device" after paid
# setup. Exercise the actual dry RunPod request rather than source text.
#
# prepare_safe_create loads the API key BEFORE it builds the mutation, even
# under dry=True, and with no key_file it falls back to $RUNPOD_KEY_FILE and
# then to ~/.config/runpod/api_key. So this file used to die HERE with an
# unhandled RunPodError on any box without an operator credential -- a bare CI
# runner, a fresh clone -- taking the ~500 assertions below it with it, and to
# pass on a developer's box only because a real key happened to sit in $HOME
# (measured 2026-09-06: any 36-character fake key made it rc=0). The key path
# is injected as a 0600 fixture; a credential is never read, and a key the
# loader still refuses is a named FAIL with its reason rather than a traceback
# that suppresses the rest of the file.
rp_create = RunPod(dry=True, key_file=_fixture_key_file())
rp_create._validated_ssh_public_key = lambda: "ssh-ed25519 AAAA"
try:
    storage_request = rp_create.create(
        gpu_type="NVIDIA L4", storage_gb=237, container_disk_gb=41,
        region="secure", spot=False, offer="on-demand",
        name="fidcloud-" + "a" * 64 + "-a" + "b" * 24,
        terminate_after="2099-01-01T00:00:00Z")
except RunPodError as exc:
    check("a non-separable provider is sized from the plan, not 100 GB "
          "-- REFUSED, the injected fixture key path did not load: %s" % exc,
          False)
else:
    check("a non-separable provider is sized from the plan, not 100 GB",
          storage_request["request"]["volume_gb"] == 237
          and storage_request["request"]["container_disk_gb"] == 41)

print("\n== no path may assume JarvisLabs except the two exported roots ==")
# Every hardcoded /home/jl_fs literal in the on-instance tools is a run that
# silently does the wrong thing somewhere else. Three of them were found one at
# a time, each by a paid run: FIDELITY_FS_ROOT and FIDELITY_ENGINE_ROOT (a run
# written into a container's ephemeral layer), then QP_PIPELINE_ROOT -- which
# stalled a box at 0% GPU for two hours, at $1.59/h, AFTER the bootstrap, a
# 200 GB fetch and the panel were all paid for. This is the rule that finds the
# fourth one without renting anything.
EXPORTED_ROOTS = ("FIDELITY_FS_ROOT", "FIDELITY_ENGINE_ROOT", "QP_PIPELINE_ROOT")
ON_INSTANCE = ("invoke_engine.py", "invoke_scorer.py", "stage_measure.sh",
               "bootstrap_measure.sh")
here = os.path.dirname(os.path.abspath(__file__))
offenders = []
for fname in ON_INSTANCE:
    path = os.path.join(here, fname)
    if not os.path.isfile(path):
        continue
    lines = open(path, encoding="utf-8").read().splitlines()
    for n, line in enumerate(lines, 1):
        if "/home/jl_fs" not in line or line.lstrip().startswith("#"):
            continue
        # Allowed ONLY as the fallback of a root the controller exports. The
        # name and its literal are often on different lines (an os.environ.get
        # call wrapped across three), so the window is what is checked.
        window = "\n".join(lines[max(0, n - 4):n + 1])
        if not any(r in window for r in EXPORTED_ROOTS):
            offenders.append("%s:%d %s" % (fname, n, line.strip()[:88]))
for o in offenders:
    print("      %s" % o)
check("every /home/jl_fs literal is the default of an EXPORTED root",
      not offenders)

env = mc._stage_env(rp)
for r in EXPORTED_ROOTS:
    check("the stage env exports %s" % r, (r + "=") in env)
check("...and none of them point at /home/jl_fs on a runpod box",
      "/home/jl_fs" not in env)

print("\n== a run's identity includes WHERE it ran ==")
import argparse                                            # noqa: E402


def jid(**kw):
    base = dict(model="m", revision="r" * 40, panel="p", lane="streaming",
                spot=False, cold_runs=2, provider="jarvislabs", gpu=None,
                role="quant")
    base.update(kw)
    return mc.job_id_for(argparse.Namespace(**base))


# Two providers running the SAME measurement produced the same job id, the
# same fidelity-runs/<id>/ directory, and the second silently OVERWROTE the
# first one's sealed receipt. The earlier result had to be rescued from disk.
ids = {jid(), jid(provider="vast"), jid(provider="runpod"),
       jid(provider="runpod", gpu="NVIDIA A100-SXM4-80GB"),
       jid(provider="runpod", gpu="NVIDIA H100 PCIe")}
check("provider and GPU change the job id (no receipt collision)", len(ids) == 5)
check("the same inputs still give the same id", jid() == jid())
check("--role root is a different run from --role quant",
      jid(role="root") != jid(role="quant"))

print("\n== a receipt says which cloud it actually ran on ==")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "measure_cloud.py"), encoding="utf-8").read()
check('the environment host is read from --provider, not hardcoded',
      '"host": getattr(args, "provider"' in src)
check("...and no literal jarvislabs host remains",
      '"host": "jarvislabs"' not in src)

print("\n== the machine is measured before the run is spent on it ==")
from fidelity.bench import estimate, gate                   # noqa: E402

# Both are REAL readings from two Vast offers for the same GPU model, taken
# with the same benchmark minutes apart. The only difference is how the host
# wires the card.
X1 = {"gpu": "NVIDIA RTX 4000 Ada Generation", "h2d_GBps": 1.6,
      "h2d_cold_GBps": 1.6, "expert_gemm_TFLOPs": 82.2,
      "stream_matrix_ms": 10.587,
      "pcie_load": {"text": "Gen4 x1 of Gen4 x16"}}
X16 = {"gpu": "NVIDIA RTX 4000 Ada Generation", "h2d_GBps": 11.0,
       "h2d_cold_GBps": 10.8, "expert_gemm_TFLOPs": 85.7,
       "stream_matrix_ms": 1.755,
       "pcie_load": {"text": "Gen3 x16 of Gen3 x16"}}

check("a PCIe x1 host is REFUSED at an 8 GB/s floor",
      gate(X1, min_h2d_gbps=8.0) is not None)
check("...and the refusal names the link width, not just the number",
      "Gen4 x1" in (gate(X1, min_h2d_gbps=8.0) or ""))
check("the x16 sibling of the SAME GPU passes the same floor",
      gate(X16, min_h2d_gbps=8.0) is None)
check("no floor asked for means no refusal", gate(X1) is None)
check("a compute floor is separate from a bandwidth floor",
      gate(X16, min_gemm_tflops=200.0) is not None)

# The whole point: same card, same compute, 6x the wall clock.
e1 = estimate(X1, matrices_per_window=36288)
e16 = estimate(X16, matrices_per_window=36288)
check("the x1 host is ~6x slower per window on identical silicon",
      5.0 < e1["minutes_per_window"] / e16["minutes_per_window"] < 7.0)

# A parked link ramps under load; a narrow one does not. That distinction is
# what makes the refusal defensible rather than a guess.
check("cold and warm are reported separately so a ramp is visible",
      "h2d_cold_GBps" in X1 and "h2d_GBps" in X1)

print("\n== the bench waits for readiness on backends that have no socket ==")
from fidelity.bench import wait_ready                       # noqa: E402

# `fidelity-bench --provider jarvislabs` advertises a fourth backend and could
# never use it: run_bench called provider._endpoint(), which only the three SSH
# backends have. The AttributeError landed on the line AFTER create(), so every
# invocation rented a box, failed, and tore it down -- paying for a benchmark
# that produced nothing. Readiness has to be asked for in a way a CLI-transport
# provider can answer.


class SSHish:
    """Has an endpoint, like RunPod / Vast / Lambda."""

    def __init__(self):
        self.asked = 0

    def _endpoint(self, mid, *, wait=900):
        self.asked += 1
        return ("1.2.3.4", 22)

    def get(self, mid):                       # must NOT be consulted
        raise AssertionError("endpoint provider must not fall back to get()")


class CLIish:
    """No endpoint at all, like JarvisLabs: state is the only signal.

    The last state REPEATS rather than falling through to "Running": a fixture
    that quietly becomes ready once its script runs out cannot fail the
    never-comes-up case, and this one did not until it was made to repeat.
    """

    def __init__(self, states):
        self._states = list(states)

    def get(self, mid):
        s = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return inst(mid, s)


s = SSHish()
wait_ready(s, "pod-x", wait=5)
check("an SSH backend is still waited on via its endpoint", s.asked == 1)

check("a CLI backend with no endpoint has no _endpoint to call",
      getattr(mc._make_provider("jarvislabs", dry=True), "_endpoint", None) is None)

wait_ready(CLIish(["Running"]), 483634, wait=5, poll=0)
check("a CLI backend that is already Running returns at once", True)

wait_ready(CLIish(["Launching", "Launching", "Running"]), 483634, wait=5, poll=0)
check("...and one that is still launching is waited for", True)

try:
    wait_ready(CLIish(["Launching"]), 483634, wait=0.01, poll=0)
    timed_out = False
except RuntimeError as exc:
    timed_out = "never became ready" in str(exc)
check("a CLI backend that never comes up raises rather than hanging", timed_out)

print("\n== a detached job's liveness cannot be read from pgrep ==")
ssh_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fidelity", "sshbase.py"), encoding="utf-8").read()
# "pgrep appears nowhere" is the wrong assertion: the file DOCUMENTS at length
# why pgrep cannot work here, and that prose is the point. What must not exist
# is pgrep inside a command this code actually sends to a machine.
sent = [ln for ln in ssh_src.splitlines()
        if "pgrep" in ln and ("exec" in ln or "echo RUNNING" in ln
                              or "-f %s" in ln or "-f {" in ln)]
check("no command sent to a machine shells out to pgrep", not sent)
check("the wrapper records its OWN pid ($$), not the launcher's $!",
      "echo $$ > {d}/pid" in ssh_src and "echo $! > {d}/pid" not in ssh_src)
check("liveness is kill -0 on that pid",
      "kill -0 $(cat {d}/pid)" in ssh_src)
# Why not pgrep, stated so it is not "simplified" back: the probe names the run
# DIRECTORY, which contains the plain run id, so the id is in the probe's own
# cmdline whatever the pattern -- confirmed on procps-ng 4.0.4, where a dead
# target answers RUNNING for both the plain and the bracketed form.
check("...and the file explains why the bracket trick cannot work here",
      "procps-ng" in ssh_src and "bracket" in ssh_src)

print("\n== a benchmark that measured nothing must not write a receipt ==")
from fidelity.bench import _measure                          # noqa: E402

# A Lambda gpu_1x_h100_sxm5 answered `{"error": "no cuda"}` 238 s after the API
# called it active, and the run wrote a receipt of zeros, exit 0. That receipt
# tabulates as a very slow machine and nothing in it says the card was absent.


class Payload:
    """Replays a scripted sequence of payload stdouts, one per exec."""

    def __init__(self, outs):
        self.outs = list(outs)
        self.calls = 0

    def exec_stdout(self, mid, cmd, timeout=0):
        self.calls += 1
        return self.outs[min(self.calls - 1, len(self.outs) - 1)]


GOOD = '{"gpu": "X", "stream_matrix_ms": 0.9}'
NOCUDA = '{"error": "no cuda"}'

p_ok = Payload([GOOD])
check("a real result is returned on the first try",
      _measure(p_ok, "m", lambda *_: None)["stream_matrix_ms"] == 0.9
      and p_ok.calls == 1)

p_race = Payload([NOCUDA, NOCUDA, GOOD])
check("'no cuda' is RETRIED -- the API calls a box ready before the driver is",
      _measure(p_race, "m", lambda *_: None, settle=0)["stream_matrix_ms"] == 0.9
      and p_race.calls == 3)

for outs, why in ((["{\"error\": \"no cuda\"}"], "no cuda forever"),
                  (['{"gpu": "X"}'], "no stream_matrix_ms")):
    try:
        _measure(Payload(outs), "m", lambda *_: None, attempts=2, settle=0)
        raised = False
    except RuntimeError as exc:
        raised = "did not measure anything" in str(exc)
    check("a receipt of zeros is REFUSED (%s)" % why, raised)

print("\n== every launch is a retry loop, with backoff that is polite ==")
import time as _time

_sleeps = []
_real_sleep = mc.time.sleep
mc.time.sleep = lambda t: _sleeps.append(t)


class Flaky:
    """Capacity-fails n times, then succeeds."""
    def __init__(self, n, msg="SUPPLY_CONSTRAINT: no longer any instances available"):
        self.n, self.msg, self.calls = n, msg, 0

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.n:
            raise RuntimeError(self.msg)
        return {"machine_id": "ok-after-%d" % self.calls}


try:
    _sleeps.clear()
    got = mc._create_with_retry(Flaky(3), Con(), gpu_type="x")
    check("capacity failures are retried to success", got["machine_id"] == "ok-after-4")
    # THIS RUNG USED TO BE FLAKY, AND FLAKY IS WORSE THAN ABSENT. It asserted
    # 1.5 < s[i+1]/s[i] < 2.7 on two JITTERED samples. The implementation is
    # wait = base * 2**(n-1) * uniform(0.8, 1.2), so a ratio of two draws
    # spans 2*0.8/1.2 = 1.33 to 2*1.2/0.8 = 3.0 -- outside that band on
    # 13.7% of runs (200k trials). A battery that goes red one time in seven
    # trains people to re-run it until it is green, which costs more than the
    # rung was ever worth. Assert the CONTRACT rather than a sampled ratio:
    # every wait lies inside its OWN analytic band. Deterministic, and it
    # still fails if the exponent or the jitter is dropped.
    _bands = [(30.0 * (2 ** i) * 0.8, 30.0 * (2 ** i) * 1.2) for i in range(3)]
    check("...with EXPONENTIAL backoff (each wait inside its own band)",
          len(_sleeps) == 3
          and all(lo <= s <= hi for s, (lo, hi) in zip(_sleeps, _bands)))
    check("...and jitter, so fleets do not synchronise",
          all(s % 30 != 0 for s in _sleeps))

    _sleeps.clear()
    try:
        mc._create_with_retry(Flaky(99), Con(), attempts=3, gpu_type="x")
        check("exhausted retries become a Refusal", False)
    except mc.Refusal as r:
        check("exhausted retries become a Refusal", "backoff" in str(r))
        check("...that says $0.00 was spent",
              any("$0.00" in a for a in (r.advice or [])))

    class Broken:
        def create(self, **kw):
            raise RuntimeError("401 unauthorized: bad api key")
    _sleeps.clear()
    try:
        mc._create_with_retry(Broken(), Con(), gpu_type="x")
        check("a NON-capacity error is not retried", False)
    except RuntimeError:
        check("a NON-capacity error is not retried", _sleeps == [])
finally:
    mc.time.sleep = _real_sleep

# A box with no working CUDA is a refusal at preflight, not an advisory: 7 of 8
# H100-SXM5 launches in the survey billed with torch.cuda.is_available()==False.
src_mc = open(os.path.join(here, "measure_cloud.py"), encoding="utf-8").read()
check("preflight REFUSES a no-cuda machine",
      '"no cuda"' in src_mc and "no working CUDA device" in src_mc)

print("\n== Vast container mode: argv as a list, and NO credential at create ==")
from fidelity.vastapi import Vast                                  # noqa: E402

_captured = []


class StubVast(Vast):
    """A Vast whose _req captures bodies instead of hitting the API."""
    def _load_key(self):
        return "stub-key"
    def _req(self, method, path, body=None, **kw):
        _captured.append({"method": method, "path": path, "body": body})
        if "/asks/" in path and method == "PUT":
            return {"success": True, "new_contract": 999}
        return {}


_v = StubVast(dry=False, ssh_key="/nonexistent/id_ed25519")
_SECRET = "hf_secret_token_abc123xyz"
# A credential-bearing sink, used only to prove the value never reaches argv or
# onstart. It deliberately does NOT name ntfy.sh: SEC-02 refuses any tracked
# file that hardcodes an ntfy topic, and it cannot tell a fixture topic from a
# live one (a real topic IS the credential -- anyone holding it can read a
# run's results). `.invalid` is reserved by RFC 6761, so this can never be a
# reachable endpoint.
_SINK = "https://sink.invalid/s3cret-topic-with-cred"

# These rungs USED to pass a token and a sink here and assert that the create
# body "carries the HF token". That is a defect class worth naming: the
# ASSERTION encoded a leak as a CONTRACT, so fixing the leak would have turned
# the suite red and made "fix it" mean "fight the test suite". A create payload
# is provider-persisted and reaches the host BEFORE any host key, attestation
# or TLS check exists, so no ordering makes it safe. The shaping coverage
# these rungs were genuinely good at is kept, with a NON-credential variable.
_ENV_NAME, _ENV_VALUE = "FIDELITY_PANEL_ID", "panel--fruit-bf16-v1"
_v.create(
    ask_id=42, storage=80,
    image="ghcr.io/malaiwah/quant-fidelity-measure:main",
    docker_cmd=["capture", "--model", "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
                "--revision", "e" * 40,
                "--sanity-expect", ""],
    env={_ENV_NAME: _ENV_VALUE},
    onstart="mkdir -p /workspace",
    name="vast-container-test")

_body = _captured[-1]["body"]
check("container mode uses runtype ssh", _body.get("runtype") == "ssh")
check("container mode onstart embeds the capture argv",
      "container_entry.py" in _body.get("onstart", "")
      and "--sanity-expect" in _body.get("onstart", "")
      and "capture" in _body.get("onstart", ""))
check("container mode onstart preserves an empty argument",
      "''" in _body.get("onstart", ""))
check("container mode image is the measurement image",
      _body.get("image") == "ghcr.io/malaiwah/quant-fidelity-measure:main")
check("container mode env carries a NON-credential variable, in env rather "
      "than in the command text",
      "-e %s=%s" % (_ENV_NAME, _ENV_VALUE) in _body.get("env", "")
      and _ENV_VALUE not in _body.get("onstart", ""))

# The two rungs that used to sit here -- a Vast create body carrying a token,
# and one carrying a bearer-capability sink, each REFUSED naming that the
# payload is provider-persisted -- are SUBSUMED and deliberately deleted, not
# merely moved. `bin/selftest_root_publish.py` RP7c/RP7d now assert the shared
# guard over EVERY provider's create-body shape and that no refusal echoes the
# credential, and RP7e/RP7f read each adapter and assert it refuses before
# transmitting. Verified all four PASS (runpodapi, vastapi, lambdaapi, jlapi)
# before removing these, so there is no window in which neither exists.
#
# One rung, four providers, is the point: the original defect was not an
# oversight in Vast's code but a per-provider test that was never made
# per-provider, so the next adapter inherits the guard instead of needing
# someone to remember it. What stays here is the SHAPING coverage above,
# which is this file's concern and no part of the credential contract.

# The critical assertion: no secret appears in onstart text.
# A provider may echo the command back; environment variables it does not.
_onstart_text = _body.get("onstart", "")
check("no secret token in onstart", _SECRET not in _onstart_text)
check("no secret sink URL in onstart", _SINK not in _onstart_text)

# SSH path is byte-identical when docker_cmd is absent.
_captured.clear()
_v.create(ask_id=42, storage=80)
_ssh = _captured[-1]["body"]
check("ssh path uses runtype ssh", _ssh.get("runtype") == "ssh")
check("ssh path has no args field", "args" not in _ssh)
check("ssh path onstart is empty", _ssh.get("onstart") == "")
check("ssh path env is empty dict", _ssh.get("env") == {})

def _provider_action():
    for action in mc.build_parser()._actions:
        if "--provider" in getattr(action, "option_strings", ()):
            return action
    raise AssertionError("--provider is not a parser option any more")


def _provider_choices():
    return tuple(_provider_action().choices or ())


def _provider_default():
    return _provider_action().default


# ---------------------------------------------------------------- CONFORMANCE
# A provider you can RENT from is not a provider you can PUBLISH from. The
# controller drives a provider through a fixed surface, and the difference
# between the two is twelve methods: four that prove the live resource is the
# one requested (and, via attest_live_resource, that it is the DEVICE the root
# was captured on), four that prove nothing of ours is still alive, and four
# that reconcile what it cost against the provider's own clock and billing.
#
# Provider is not a comparability axis -- two A100s in two clouds agree bitwise
# (docs/ARCHITECTURE-DETERMINISM.md), and four GLM-5.3-Flash rows landed on one
# comparability key across three datacenters -- so parity here is legitimate
# science, not convenience. What blocks it is that we cannot yet prove what we
# rented, prove it is gone, or reconcile its cost on any provider but RunPod.
#
# The table lives in bin/fidelity/providers.py, which the CONTROLLER also
# reads, so a declaration cannot disagree with behaviour. It was here, beside
# a controller that hardcoded "runpod" in three places, and the two halves
# could drift silently.
#
# Parity is a PREDICATE, not a label: all twelve implemented AND a paid
# execution entrypoint AND an empty blocker tuple. Method conformance is
# COMPUTED from the adapter class and never declared, so an adapter reaching
# twelve-of-twelve needs no table edit and a table cannot claim conformance it
# does not have. What is declared is only the residue no offline test can
# compute, and a provider is enabled by DELETING a blocker line.
#
# This rung is offline and contacts nothing. docs/PROVIDER-PARITY.md carries
# the per-provider blockers and the definition of done.
from fidelity import providers as P                          # noqa: E402

PROVIDER_CONTRACT = P.PROVIDER_CONTRACT

print()
print("== provider contract conformance (offline) ==")
check("the table lives in bin/fidelity/providers.py, which the controller "
      "reads too (a declaration cannot disagree with behaviour)",
      mc._providers() is P)
check("the CLI's --provider choices ARE the declared providers",
      set(_provider_choices()) == set(P.PROVIDERS))
check("every provider the CLI accepts appears in the blocker table",
      set(P.PROVIDER_BLOCKERS) == set(P.PROVIDERS)
      and set(P.PROVIDER_DEGRADATIONS) == set(P.PROVIDERS))
# RunPod is the reference: if IT is missing a contract method, the LIST has
# drifted from the implementation and the list is what is wrong.
check("the contract names only methods RunPod actually has (the list cannot "
      "drift from the reference implementation)",
      not P.missing_contract_methods("runpod"))
check("the sweep's required methods are a subset of the twelve, plus the easy "
      "surface -- a sweep cannot require something no adapter is asked for",
      set(P.SWEEP_CONTRACT) <= set(P.PROVIDER_CONTRACT))

for _name in P.PROVIDERS:
    _missing = P.missing_contract_methods(_name)
    _blockers = P.blockers(_name)
    _ready = P.measurement_ready(_name)
    _refusal = P.measurement_refusal(_name)
    # A blocker nobody can read is a blocker nobody will clear.
    for _text in _blockers + P.degradations(_name):
        check("%s declares an explanatory blocker/degradation, not a flag "
              "(%r)" % (_name, _text[:48]),
              isinstance(_text, str) and len(_text) >= 40 and " " in _text)
    # BOTH DIRECTIONS. Ready means nothing is missing and nothing is
    # declared; not ready means the refusal NAMES the reason.
    if _ready:
        check("%s is dispatchable, so it must have all %d contract methods, "
              "an execution entrypoint and no blockers"
              % (_name, len(PROVIDER_CONTRACT)),
              not _missing and not _blockers
              and _name in P.EXECUTION_ENTRYPOINTS
              and _refusal is None)
        check("...and its entrypoint resolves to a real controller function",
              callable(P.execution_entrypoint(_name, mc)))
    else:
        check("%s is refused, and something really does block it (%d of %d "
              "methods missing, %d declared blocker(s))"
              % (_name, len(_missing), len(PROVIDER_CONTRACT), len(_blockers)),
              bool(_missing) or bool(_blockers)
              or _name not in P.EXECUTION_ENTRYPOINTS)
        _text = str(_refusal)
        check("...and the refusal names the real reason and points at the "
              "parity doc, rather than naming a provider",
              _refusal is not None and P.PARITY_DOC in _text
              and all(method in _text for method in _missing)
              and all(blocker in _text for blocker in _blockers))

# A DECLARATION THAT LIES MUST GO RED. The old gate was verified to fail
# (rc 1) when a declared level disagreed with the code. The equivalent
# property here is stronger, because the method half is computed: a table
# that declares NO blockers, names an execution entrypoint, and is still
# missing a contract method must remain refused. The liar is SYNTHETIC on
# purpose -- all four real adapters now carry the twelve, and a probe that
# depended on one of them being incomplete would evaporate exactly when the
# ports landed.
class PartialAdapter:
    """Eleven of the twelve. The absent one is the entire point."""


for _method in PROVIDER_CONTRACT[:-1]:
    setattr(PartialAdapter, _method, lambda self, *a, **k: None)
_ABSENT = PROVIDER_CONTRACT[-1]

_saved = (P.PROVIDERS, dict(P._ADAPTERS), dict(P.PROVIDER_BLOCKERS),
          dict(P.PROVIDER_DEGRADATIONS), dict(P.EXECUTION_ENTRYPOINTS),
          dict(P.PAID_EXECUTION_PROFILES),
          dict(P.PAID_PREREQUISITE_EVIDENCE))


def _probe_profile(**overrides):
    """A complete, honest paid execution profile for the synthetic adapter.

    Built by COPYING RunPod's row rather than by hand, so a new required
    profile field cannot be forgotten here and silently stop the probe from
    testing the thing it exists to test.
    """
    row = dict(P.PAID_EXECUTION_PROFILES["runpod"])
    row.update(overrides)
    return row


def _probe_evidence():
    return {
        "sweep_settled_lease": "synthetic: a settled lease with an absence "
                               "proof and a reconciled cost",
        "teardown_proven": {
            path: "synthetic" for path in P.TEARDOWN_PROOF_PATHS},
    }


try:
    P.PROVIDERS = P.PROVIDERS + ("probe",)
    # The selftest runs as __main__, so the table can point at a class
    # defined right here without writing a throwaway module.  If these rungs
    # ever move into an IMPORTED helper this stops resolving and the probe
    # quietly tests nothing, which is worse than a red one.
    P._ADAPTERS["probe"] = ("__main__", "PartialAdapter")
    P.PROVIDER_BLOCKERS["probe"] = ()
    P.PROVIDER_DEGRADATIONS["probe"] = ()
    # Whatever the real RunPod entrypoint is, not a literal: the point is
    # that an entrypoint EXISTS, and this survives the next rename.
    P.EXECUTION_ENTRYPOINTS["probe"] = P.EXECUTION_ENTRYPOINTS["runpod"]
    P.PAID_EXECUTION_PROFILES["probe"] = _probe_profile()
    P.PAID_PREREQUISITE_EVIDENCE["probe"] = _probe_evidence()
    _probe = P.measurement_refusal("probe")
    check("a provider declaring NO blockers, with an entrypoint and a "
          "complete profile, is STILL refused while a contract method is "
          "missing (the method half is computed, so the table cannot lie)",
          _probe is not None and not P.measurement_ready("probe")
          and _ABSENT in str(_probe))
    check("...and its sweep is refused too, naming a method a sweep drives "
          "and pointing at the parity doc",
          P.sweep_refusal("probe") is not None
          and P.reaper_refusal("probe") is not None
          and "status" in str(P.sweep_refusal("probe"))
          and P.PARITY_DOC in str(P.sweep_refusal("probe")))
    # And the other direction: complete the class and it becomes ready, so
    # the gate is not simply refusing everything.  THIS IS THE CENTRAL RUNG
    # OF THE PAID-PATH WORK: a synthetic adapter that conforms reaches the
    # shared paid path with no lifecycle edit and no per-provider branch.
    setattr(PartialAdapter, _ABSENT, lambda self, *a, **k: None)
    for _method in P.SWEEP_BASE:
        setattr(PartialAdapter, _method, lambda self, *a, **k: None)
    for _method in ("upload", "exec"):
        setattr(PartialAdapter, _method, lambda self, *a, **k: None)
    check("...and the SAME declaration becomes dispatchable the moment the "
          "twelfth method exists (both directions, no table edit)",
          P.measurement_ready("probe")
          and P.sweep_refusal("probe") is None)
    check("...and a CONFORMING synthetic provider resolves the SHARED paid "
          "lifecycle, the same function object RunPod runs (a second "
          "provider's paid path is no new lifecycle code)",
          P.execution_entrypoint("probe", mc)
          is P.execution_entrypoint("runpod", mc)
          and P.execution_entrypoint("probe", mc) is mc._main_paid)

    # A non-empty blocker tuple refuses, and NAMES the blocker.
    P.PROVIDER_BLOCKERS["probe"] = ("the synthetic blocker nobody cleared",)
    _blocked = P.measurement_refusal("probe")
    check("a conforming provider with a non-empty blocker tuple is refused, "
          "quoting the blocker verbatim",
          _blocked is not None
          and "the synthetic blocker nobody cleared" in str(_blocked))
    P.PROVIDER_BLOCKERS["probe"] = ()

    # A missing paid execution profile refuses, and says so as its own
    # reason rather than as twelve confusing safety-property failures.
    del P.PAID_EXECUTION_PROFILES["probe"]
    _no_profile = P.measurement_refusal("probe")
    check("a conforming provider with NO paid execution profile is refused, "
          "naming the profile rather than defaulting to RunPod's values",
          _no_profile is not None
          and "paid execution profile" in str(_no_profile)
          and "no paid execution profile is declared for probe"
          in str(_no_profile))
    check("...and every field the shared lifecycle needs is named in that "
          "refusal, so the remedy is readable",
          all(field in str(_no_profile)
              for field in P.PAID_EXECUTION_CONTRACT))

    # A profile that MEETS the twelve but cannot give a provider-enforced
    # deadline is refused as an unmet SAFETY PROPERTY -- the Lambda ruling,
    # made mechanical.  Lambda's launch has no terminateAfter, so its
    # deadline would be controller clock plus the on-instance watchdog, and
    # both of those die with the controller host and the instance OS
    # respectively.  A paid run needs one layer that survives both.
    P.PAID_EXECUTION_PROFILES["probe"] = _probe_profile(
        provider_enforced_deadline=None)
    _no_deadline = P.measurement_refusal("probe")
    check("a conforming provider with NO provider-enforced termination "
          "deadline is refused as an unmet safety property, not admitted "
          "with a degradation (the Lambda terminateAfter ruling, computed)",
          _no_deadline is not None
          and "provider-enforced-deadline" in str(_no_deadline)
          and "provider_enforced_deadline" in str(_no_deadline))

    # A profile whose credential transport is not the 0600-file-over-verified
    # -SSH mechanism is refused: this is the Vast container-mode gate, and it
    # fires on the MECHANISM before any provider call, ahead of the adapter's
    # complementary payload-shape refusal.
    P.PAID_EXECUTION_PROFILES["probe"] = _probe_profile(
        credential_transport="provider-create-env")
    _bad_transport = P.measurement_refusal("probe")
    check("a provider whose declared credential transport is not the "
          "0600-file-over-verified-SSH mechanism is refused before any "
          "provider call (Vast container mode cannot be fixed by ordering)",
          _bad_transport is not None
          and "credential-transport" in str(_bad_transport)
          and P.PAID_CREDENTIAL_TRANSPORT in str(_bad_transport))

    # A host key that authenticates the MACHINE rather than the resource is
    # refused: measured on Vast 2026-09-06, one key across two contracts on
    # machine 150014, surviving destroy-and-create.
    P.PAID_EXECUTION_PROFILES["probe"] = _probe_profile(
        host_key_attribution="machine")
    _machine_key = P.measurement_refusal("probe")
    check("a provider whose host key is machine-attributable rather than "
          "resource-attributable is refused (a verified channel to hardware "
          "we have seen before is not attribution to what we rented)",
          _machine_key is not None
          and "host-key-before-ssh" in str(_machine_key))

    # Definition of done, items 3 and 6: refused when unproven, and item 6
    # refused when only PARTLY proven.
    P.PAID_EXECUTION_PROFILES["probe"] = _probe_profile()
    P.PAID_PREREQUISITE_EVIDENCE["probe"] = {}
    _unproven = P.measurement_refusal("probe")
    check("a fully conforming provider is STILL refused until definition-of-"
          "done items 3 and 6 are proven (a reaper sweep that settled a "
          "lease, and teardown on all four paths)",
          _unproven is not None
          and "sweep_settled_lease" in str(_unproven)
          and "teardown_proven" in str(_unproven))
    _partial = _probe_evidence()
    _partial["teardown_proven"] = dict(_partial["teardown_proven"])
    del _partial["teardown_proven"]["interrupt"]
    P.PAID_PREREQUISITE_EVIDENCE["probe"] = _partial
    _partial_refusal = P.measurement_refusal("probe")
    check("...and a PARTIAL teardown record is refused naming the missing "
          "path, rather than rounded up to proven",
          _partial_refusal is not None
          and "interrupt" in str(_partial_refusal))

    # Item 3 is EARNED from the lease store, not only declared: a provider
    # with no declaration but a settled absence-proven lease is admitted.
    P.PAID_PREREQUISITE_EVIDENCE["probe"] = {
        "teardown_proven": {
            path: "synthetic" for path in P.TEARDOWN_PROOF_PATHS}}
    with tempfile.TemporaryDirectory() as _store:
        check("item 3 with no declaration and an EMPTY lease store is still "
              "refused",
              P.measurement_refusal("probe", _store) is not None)
        _settled = {
            "state": "TERMINAL",
            "create": {"provider": "probe"},
            "terminal_proof": {
                "provider_absence": {
                    "authoritative_inventory": {"schema": "synthetic"},
                    "complete_listing": True},
                "billing_reconciliation": {"reconciled": True}},
        }
        Path(_store, "settled.json").write_text(
            json.dumps(_settled), encoding="utf-8")
        check("...and the SAME provider is admitted once its store holds a "
              "TERMINAL lease with an absence proof and a reconciled cost "
              "(item 3 is earned by settling a real lease, not declared)",
              P.measurement_refusal("probe", _store) is None)
        _weak = dict(_settled)
        _weak["terminal_proof"] = {
            "provider_absence": {"complete_listing": True},
            "billing_reconciliation": {"reconciled": True}}
        Path(_store, "settled.json").write_text(
            json.dumps(_weak), encoding="utf-8")
        check("...but a TERMINAL lease with NO authoritative inventory does "
              "not earn it (a lifecycle listing alone is not absence)",
              P.measurement_refusal("probe", _store) is not None)
finally:
    (P.PROVIDERS, _adapters, _blockers, _degradations, _entries,
     _profiles, _evidence) = _saved
    P._ADAPTERS.clear(); P._ADAPTERS.update(_adapters)
    P.PROVIDER_BLOCKERS.clear(); P.PROVIDER_BLOCKERS.update(_blockers)
    P.PROVIDER_DEGRADATIONS.clear()
    P.PROVIDER_DEGRADATIONS.update(_degradations)
    P.EXECUTION_ENTRYPOINTS.clear(); P.EXECUTION_ENTRYPOINTS.update(_entries)
    P.PAID_EXECUTION_PROFILES.clear()
    P.PAID_EXECUTION_PROFILES.update(_profiles)
    P.PAID_PREREQUISITE_EVIDENCE.clear()
    P.PAID_PREREQUISITE_EVIDENCE.update(_evidence)
check("...and the table is restored after that probe",
      "probe" not in P.PROVIDERS and "probe" not in P.PROVIDER_BLOCKERS
      and "probe" not in P.PAID_EXECUTION_PROFILES
      and set(P.EXECUTION_ENTRYPOINTS) == set(P.PROVIDERS))
check("every provider the CLI accepts is registered on the SHARED paid "
      "lifecycle, so a refusal is never 'no paid execution path' -- it is "
      "always the provider's own unmet property",
      all(P.EXECUTION_ENTRYPOINTS[_p] == "_main_paid" for _p in P.PROVIDERS))

print()
print("== the controller's refusals are derived, not hardcoded ==")
_mc_src = open(os.path.join(here, "measure_cloud.py"), encoding="utf-8").read()
check("no refusal text hardcodes 'requires explicit --provider runpod'",
      "requires explicit --provider runpod" not in _mc_src)
check("the drill's refusal says what a provider without one may still do, "
      "instead of 'RunPod-only'",
      "drill subcommand is RunPod-only" not in _mc_src
      and "not under strict" in _mc_src)
check("the reaper no longer advertises RunPod + historical JarvisLabs as the "
      "only sweeps",
      "reaper supports RunPod and historical JarvisLabs cleanup"
      not in _mc_src)
check("'historical' is not a parity level any more (one axis: can this "
      "publish a number)",
      "historical" not in repr(P.PROVIDER_BLOCKERS))
# --provider must keep NO default: fcb9470 removed one that made two
# refusals unreachable.
check("--provider still has no default (a guessed account bills the wrong "
      "person)", _provider_default() is None)
for _cmd in (["reaper", "--list"], ["drill"]):
    check("`measure-cloud %s` with no --provider refuses" % " ".join(_cmd),
          mc.main(_cmd) == mc.EXIT_REFUSED)
# Sweep admission is deliberately WIDER than measurement admission: refusing
# to reap is itself a leak, so an adapter the sweep can drive may be swept
# even while the provider is unfit to measure on. All four now can be
# (the synthetic probe above covers the refusal direction).
check("every adapter the CLI accepts can now be driven through the sweep, so "
      "no provider is left without an autonomous teardown backstop",
      not [name for name in P.PROVIDERS if P.sweep_refusal(name) is not None])
check("...and sweep admission does not imply measurement admission",
      P.sweep_refusal("vast") is None
      and P.measurement_refusal("vast") is not None)
check("...and RunPod's sweep is admitted",
      P.sweep_refusal("runpod") is None
      and P.reaper_refusal("runpod") is None)
check("JarvisLabs keeps its legacy lease sweep while the generic one is "
      "unproven against an old lease shape",
      "jarvislabs" in P.LEGACY_SWEEP_PROVIDERS
      and P.reaper_refusal("jarvislabs") is None)
check("--role candidate still normalises to root (one code path)",
      "args.role = \"root\"" in _mc_src)
print()
if FAILED:
    print("selftest_provider_portability: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_provider_portability: all passed")
