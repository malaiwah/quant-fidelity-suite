#!/usr/bin/env python3
"""P1-06 known-answer tests for the tools/fidelity.py KL reducer.

The defect: the replay comparator computed logits, normalizers, probabilities and
the VOCABULARY SUM in float32 and cast the already-reduced result to float64,
while its receipts (and 37 Qwen registry rows seeded from them) declared float64
accumulation. On near-equal 50k-vocab distributions the float32 reduction returns
NEGATIVE "KL" values around -1e-6 where the true float64 KL is ~+2e-8 -- KL is
mathematically non-negative, so a negative value is pure estimator error, three
orders of magnitude larger than the quantity being measured.

These tests FAIL against the pre-fix reducer (verified by running them on the
parent commit): case 2 asserts the fixed path is non-negative and matches a dense
float64 reference on exactly the construction where the float32 path goes
negative.

Needs torch (run under the venv python). No network, no GPU, CPU-only, ~seconds.
"""
import math
import pathlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import torch  # noqa: E402

import fidelity  # noqa: E402

V, D, N = 50_000, 64, 8
CHUNK = 8_192  # not a divisor of V, so the ragged final chunk is exercised

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""))
    if not ok:
        failures.append(name)


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    head = torch.randn(V, D, generator=g) / math.sqrt(D)
    ref_h = torch.randn(N, D, generator=g)
    cand_h = ref_h + 2e-4 * torch.randn(N, D, generator=g)
    return head, ref_h, cand_h


def legacy_float32_reduce(ref_h, cand_h, head, chunk):
    """The pre-fix reduction, verbatim: float32 everything, .double() after the sum."""
    def norm(h):
        log_z = torch.full((h.shape[0],), -math.inf, dtype=torch.float32)
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            logits = (h @ head[s:e].T).float()
            log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
        return log_z
    rz, cz = norm(ref_h), norm(cand_h)
    kl = torch.zeros(ref_h.shape[0], dtype=torch.float64)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        rl = (ref_h @ head[s:e].T).float() - rz[:, None]
        cl = (cand_h @ head[s:e].T).float() - cz[:, None]
        kl += (rl.exp() * (rl - cl)).sum(-1).double()
    return kl


def dense_float64_truth(ref_h, cand_h, head):
    """Dense float64 reference over the SAME float32 logits the comparator sees.

    The contract under test is the receipts' split: logits_dtype fp32 (the matmul
    output), accumulation float64 (everything after the logits). So the reference
    casts the fp32 logits to float64 and does normalization, exp, product and the
    vocabulary sum densely in float64.
    """
    rl = (ref_h @ head.T).double()
    cl = (cand_h @ head.T).double()
    rl = rl - rl.logsumexp(-1, keepdim=True)
    cl = cl - cl.logsumexp(-1, keepdim=True)
    return (rl.exp() * (rl - cl)).sum(-1)


head, ref_h, cand_h = make_inputs()

# --- 1. the defect is real: the float32 reduction goes negative here ----------
kl_legacy = legacy_float32_reduce(ref_h, cand_h, head, CHUNK)
kl_truth = dense_float64_truth(ref_h, cand_h, head)
check("float32 reduction goes negative on near-equal 50k-vocab distributions",
      float(kl_legacy.min()) < 0 < float(kl_truth.min()),
      "legacy min %.3e, true range [%.3e, %.3e]"
      % (kl_legacy.min(), kl_truth.min(), kl_truth.max()))

# --- 2. the fixed reducer stays non-negative and matches dense float64 --------
kl, js, hits = fidelity.context_metrics(ref_h, cand_h, head, CHUNK)
check("context_metrics: per-token KL strictly positive on the failing case",
      float(kl.min()) > 0, "min %.3e" % kl.min())
max_err = float((kl - kl_truth).abs().max())
check("context_metrics: matches dense float64 known answer",
      max_err < 1e-12, "max abs err %.3e against values ~%.1e" % (max_err, float(kl_truth.mean())))
check("context_metrics: JSD non-negative", float(js.min()) >= 0.0, "min %.3e" % js.min())

# --- 3. exact known answer: self-compare is exactly zero ----------------------
kl0, js0, hits0 = fidelity.context_metrics(ref_h, ref_h.clone(), head, CHUNK)
check("context_metrics: self-compare KL exactly 0.0",
      float(kl0.abs().max()) == 0.0, "max %.3e" % kl0.abs().max())
check("context_metrics: self-compare top-1 agreement is total", hits0 == N)

# --- 4. qualification_metrics: same construction through the live-logit path --
live = (ref_h @ head.T).double()
live = (live - live.logsumexp(-1, keepdim=True)).float()  # normalized fp32 logprobs, as served
klq, hitsq = fidelity.qualification_metrics(live, cand_h, head, CHUNK)
# reference: KL(live || replayed candidate), both in float64; the live operand is
# re-normalised in float64 exactly as the fixed implementation does
liv64 = live.double()
liv64 = liv64 - liv64.logsumexp(-1, keepdim=True)
rep64 = (cand_h @ head.T).double()
rep64 = rep64 - rep64.logsumexp(-1, keepdim=True)
klq_truth = (liv64.exp() * (liv64 - rep64)).sum(-1)
qerr = float((klq - klq_truth).abs().max())
check("qualification_metrics: matches dense float64 known answer",
      qerr < 1e-12, "max abs err %.3e" % qerr)
check("qualification_metrics: non-negative", float(klq.min()) >= 0.0, "min %.3e" % klq.min())

# --- 5. the sanity guard refuses garbage rather than reporting it -------------
for bad, label in ((torch.tensor([0.0, float("nan")]), "NaN"),
                   (torch.tensor([0.0, float("-inf")]), "-inf"),
                   (torch.tensor([1e-8, -1e-6]), "materially negative")):
    try:
        fidelity._check_kl_sane(bad.double(), "selftest")
        check("_check_kl_sane refuses %s" % label, False, "accepted silently")
    except ValueError:
        check("_check_kl_sane refuses %s" % label, True)
ok = torch.tensor([0.0, 1e-8, -1e-15]).double()  # float64 rounding dust is not an error
try:
    fidelity._check_kl_sane(ok, "selftest")
    check("_check_kl_sane accepts rounding-scale dust", True)
except ValueError as e:
    check("_check_kl_sane accepts rounding-scale dust", False, str(e))

# MKL-01 containment. `torch.logsumexp` on a large fp32 tensor executes an
# ILLEGAL INSTRUCTION on a pre-AVX host: measured on this Xeon X5570 (Nehalem,
# sse4_2, no AVX) at 5 of 12 runs, and 0 of 12 with MKL_NUM_THREADS=1. The
# fault is inside mkl_vml_kernel in torch/lib/libtorch_cpu.so, so it is the
# CPU and MKL's threaded VML dispatch, not this code.
#
# It was isolated by elimination, and the elimination is why the guard can be
# narrow: log, exp, sum, log_softmax, logaddexp and matmul all measured 0/6 to
# 0/8 on the same box. logsumexp alone fails. The production scorers
# (engines/tools/kld_report.py, stream_score.py) normalise with log_softmax
# and are therefore NOT exposed.
#
# So the mitigation is one `env MKL_NUM_THREADS=1` on this file's battery rung
# -- and it stays sufficient only while this file remains the ONLY user of the
# op. This rung is what makes that an enforced invariant instead of a comment.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GUARDED = {"bin/selftest_fidelity_reducer.py"}
_users = set()
for _tree in ("bin", "engines/tools", "registry"):
    _base = _ROOT / _tree
    if not _base.is_dir():
        continue
    for _path in sorted(_base.rglob("*.py")):
        if any(part in {".venv", "__pycache__"} for part in _path.parts):
            continue
        try:
            if "logsumexp" in _path.read_text(encoding="utf-8"):
                _users.add(str(_path.relative_to(_ROOT)))
        except (OSError, UnicodeDecodeError):
            continue
check("MKL-01: torch.logsumexp stays confined to the file whose battery rung "
      "carries MKL_NUM_THREADS=1 (found: %s)" % ", ".join(sorted(_users) or ["none"]),
      _users <= _GUARDED,
      "unguarded users: %s" % ", ".join(sorted(_users - _GUARDED)))


print()
if failures:
    print("selftest_fidelity_reducer: FAILED: %s" % ", ".join(failures))
    sys.exit(1)
print("selftest_fidelity_reducer: all checks passed")
