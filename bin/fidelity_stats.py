#!/usr/bin/env python3
"""fidelity-stats -- same-lane KL contrasts and document-aware paired analysis.

    bin/fidelity-stats attributable --quant-summary K8.json \
        --floor-summary engines/native-bf16-kld.json
    bin/fidelity-stats paired-delta --report-a runA/kld-report.json \
        --report-b runB/kld-report.json --label-a K6 --label-b K8

WHY THE GATES EXIST. KL is not additive: subtracting a native baseline is a
descriptive contrast, not an identified quantization effect. Reference, panel
and lane identity must be checked before comparing runs. Document provenance
is required for source-document inference; window-level spread describes the
observed panel, not independent samples when windows share a document.

Stock python3.9, stdlib only (json, math, random, statistics, argparse).
Outputs are analysis receipts, not measurements: every output carries
not_submittable: true and a schema no registry adapter accepts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.common import Console, write_json           # noqa: E402
from jointstd.stats import document_level_paired           # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3

ATTRIBUTABLE_SCHEMA = "malaiwah.glm53-floor-attributable-report.v1"
PAIRED_SCHEMA = "malaiwah.glm53-paired-window-delta.v1"

# The sealed streaming-lane teacher; quoted in refusals so the operator can see
# WHICH teacher a floor belongs to without opening files.
SEALED_STREAM_TEACHER = "2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5"

# The only floor profile that legitimately pairs with the sealed streaming
# teacher today (engines/native-bf16-kld.json).  Gating on it closes the
# adversarial review's "same-teacher floor forgery" residual for every
# naturally-occurring artifact: the cross-stack 0.012712 number exists only
# in receipts that do NOT carry this profile.  (A deliberate forgery of the
# profile field too is out of scope -- these receipts are not signed.)
STREAM_FLOOR_PROFILE = "native-bf16-stream"


def _sha_pair(a: str, b: str, n: int = 16) -> "Tuple[str, str]":
    """Truncated display forms of two shas -- FULL forms when the truncations
    would collide, so a refusal can never print an identical-looking
    'X vs X' for two different values (adversarial review, 2026-08-28)."""
    a, b = str(a or "?"), str(b or "?")
    if a != b and a[:n] == b[:n]:
        return a, b
    return a[:n], b[:n]

ESTIMAND_TEXT = (
    "The equal-window mean difference describes the supplied finite panel. "
    "No determinism or zero measurement error is inferred from these reports. "
    "Source-document inference requires consistent provenance and assumptions "
    "of independent, exchangeable documents; window-level statistics are "
    "descriptive only. Equal-document and equal-window means need not coincide."
)

CROSS_LANE_WORKED_EXAMPLE = (
    "For example, 0.012384 - 0.012712 = -0.000328, whereas "
    "0.012384 - 0.011506 = +0.000878. The mismatch in provenance, not the "
    "negative sign, invalidates a cross-lane baseline interpretation. KL "
    "differences can legitimately be negative and do not isolate a codec effect."
)


class Refusal(RuntimeError):
    def __init__(self, reason: str, advice: Optional[List[str]] = None) -> None:
        self.reason, self.advice = reason, list(advice or [])
        super().__init__(reason)


# ==========================================================================
# Pure-stdlib statistics
# ==========================================================================


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes betacf)."""
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p for Student's t via the regularized incomplete beta."""
    if df <= 0:
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def t_quantile_975(df: int) -> float:
    """t_{df, 0.975} by bisecting the exact p-value function.

    (The plan called for a hardcoded df 1..40 table; inverting the same betai
    the p-value uses is exact for ANY df and cannot disagree with the p-value
    printed next to it.  Known answer asserted by the selftest: df=24 ->
    2.0639.)"""
    lo, hi = 0.0, 700.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _norm():
    return statistics.NormalDist()


def bca_interval(values: Sequence[float], B: int, seed: int,
                 alpha: float = 0.05) -> Dict[str, Any]:
    """Descriptive BCa spread of the equal-window mean, not document inference."""
    n = len(values)
    if n < 2 or B < 2 or not 0 < alpha < 1:
        raise ValueError("BCa needs at least two windows, two resamples and 0 < alpha < 1")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("BCa values must be finite")
    rnd = random.Random(seed)
    theta = statistics.fmean(values)
    boots = sorted(
        statistics.fmean(values[rnd.randrange(n)] for _ in range(n))
        for _ in range(B)
    )
    prop = sum(1 for b in boots if b < theta) / float(B)
    prop = min(max(prop, 1.0 / (B + 1)), 1.0 - 1.0 / (B + 1))
    z0 = _norm().inv_cdf(prop)
    jack = [statistics.fmean(list(values[:i]) + list(values[i + 1:]))
            for i in range(n)]
    jm = statistics.fmean(jack)
    num = sum((jm - j) ** 3 for j in jack)
    den = 6.0 * (sum((jm - j) ** 2 for j in jack)) ** 1.5
    accel = (num / den) if den else 0.0

    def adjusted(p: float) -> float:
        z = _norm().inv_cdf(p)
        w = z0 + (z0 + z) / (1.0 - accel * (z0 + z))
        return _norm().cdf(w)

    def quantile(p: float) -> float:
        idx = min(max(int(p * B), 0), B - 1)
        return boots[idx]

    return {
        "method": "BCa-bootstrap-over-windows",
        "B": B, "seed": seed,
        "low": quantile(adjusted(alpha / 2.0)),
        "high": quantile(adjusted(1.0 - alpha / 2.0)),
        "z0": z0, "acceleration": accel,
    }


def sign_test_two_sided(wins: int, n: int) -> Optional[float]:
    """Exact two-sided sign test; no informative pairs means no p-value."""
    if n < 0 or not 0 <= wins <= n:
        raise ValueError("sign counts must satisfy 0 <= wins <= n")
    if n == 0:
        return None
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilcoxon_signed_rank(deltas: Sequence[float]) -> Dict[str, Any]:
    """Wilcoxon signed-rank, normal approximation with tie correction."""
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "W_plus": None, "z": None, "p": None}
    ranked = sorted(nonzero, key=abs)
    ranks: List[float] = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(ranked[j + 1]) == abs(ranked[i]):
            j += 1
        avg_rank = (i + j + 2) / 2.0                    # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for d, r in zip(ranked, ranks) if d > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return {"n": n, "W_plus": w_plus, "z": None, "p": None}
    z = (w_plus - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - _norm().cdf(abs(z)))
    return {"n": n, "W_plus": w_plus, "z": z, "p": min(1.0, p)}


# ==========================================================================
# attributable
# ==========================================================================


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise Refusal("%s missing: %s" % (label, path))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Refusal("%s is not valid JSON: %s" % (label, exc))
    if not isinstance(doc, dict):
        raise Refusal("%s must be a JSON object" % label)
    return doc


def _gate_floor_receipt(floor: Dict[str, Any], floor_path: Path) -> None:
    schema = floor.get("schema", "")
    if schema == "malaiwah.glm53-bf16-floor-analysis.v1":
        raise Refusal(
            "%s is the floor ANALYSIS (schema %s), not a floor summary. It "
            "already contains the worked attributables, including the field "
            "'cross_stack_floor_do_not_mix' -- whose name is the instruction: "
            "that value (0.012712) belongs to a DIFFERENT lane and must never "
            "be subtracted from a streaming-lane mean. Pass the floor summary "
            "instead (engines/native-bf16-kld.json)." % (floor_path, schema))
    label = floor.get("student_label")
    if label != "native-bf16":
        raise Refusal(
            "floor receipt is labelled %r, not 'native-bf16' -- not a "
            "lossless-lane floor. A floor is the un-quantized weights pushed "
            "through the identical capture; a receipt labelled %r measured a "
            "quant, and subtracting one quant from another is a delta, not an "
            "attributable (use paired-delta for that)." % (label, label))
    if floor.get("measured_mean_kld") is None:
        raise Refusal("floor receipt %s carries no measured_mean_kld" % floor_path)


def _cross_lane_refusal(quant_mean: float, quant_teacher: str,
                        floor_mean: float, floor_teacher: str) -> Refusal:
    delta = quant_mean - floor_mean
    q_disp, f_disp = _sha_pair(quant_teacher, floor_teacher)
    return Refusal(
        "floor %.6f was measured against teacher %s; your panel mean %.6f was "
        "measured against teacher %s. Floors are (panel, teacher, lane)-"
        "specific. Your subtraction would be %.6f - %.6f = %+.6f. %s"
        % (floor_mean, f_disp, quant_mean,
           q_disp, quant_mean, floor_mean, delta,
           CROSS_LANE_WORKED_EXAMPLE),
        ["matching teacher_receipt_sha256 is necessary, not sufficient; "
         "panel and lane must also match",
         "the streaming lane's own floor summary is engines/native-bf16-kld.json"])


def _fetch_registry_quant(measurement_id: str, source: str,
                          con: Console) -> Tuple[Dict[str, Any], str]:
    """--from-registry: pull the row and fetch its public receipt for gating."""
    from fidelity import registry_client as RC
    reg = RC.load(source, purpose="rows", con=con)
    row = reg.collections.get("measurements", {}).get(measurement_id)
    if row is None:
        raise Refusal("measurement %s not in the registry (%s)"
                      % (measurement_id, reg.footer()))
    uri = None
    for src in (row.get("provenance") or {}).get("sources") or []:
        if src.get("kind") in ("hf_file", "github_file") and src.get("uri"):
            uri = src["uri"]
            break
    if not uri:
        raise Refusal(
            "measurement %s has no publicly fetchable receipt (its sources "
            "are local paths); pass --quant-summary with the file instead"
            % measurement_id)
    con.say("  fetching receipt: %s" % uri)
    import urllib.request
    req = urllib.request.Request(uri, headers={"User-Agent": "fidelity-suite/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        receipt = json.loads(resp.read().decode("utf-8"))
    value = (row.get("metric") or {}).get("value")
    if receipt.get("measured_mean_kld") is not None and \
            value is not None and receipt["measured_mean_kld"] != value:
        raise Refusal(
            "registry row %s says %.12g but its receipt says %.12g -- "
            "refusing to pick one" % (measurement_id, value,
                                      receipt["measured_mean_kld"]))
    return receipt, uri


def cmd_attributable(args: argparse.Namespace, con: Console) -> int:
    con.say("fidelity-stats attributable")
    con.rule()
    gates: List[str] = []

    if args.from_registry:
        quant, origin = _fetch_registry_quant(args.from_registry,
                                              args.registry, con)
    else:
        if not args.quant_summary:
            raise Refusal("pass --quant-summary PATH or --from-registry ID")
        quant = _load_json(Path(args.quant_summary), "quant summary")
        origin = str(args.quant_summary)

    floor_path = Path(args.floor_summary)
    floor = _load_json(floor_path, "floor summary")

    # Gate 1: the floor must BE a floor.
    _gate_floor_receipt(floor, floor_path)
    gates.append("floor receipt is student_label 'native-bf16' (a lossless-lane floor)")

    quant_mean = quant.get("measured_mean_kld")
    if quant_mean is None:
        raise Refusal(
            "%s carries no measured_mean_kld -- it is not a *-packed-kld-summary "
            "receipt (schema %r). Preview receipts deliberately use a different "
            "field name (preview_panel_mean_estimate) so they cannot be fed in "
            "here." % (origin, quant.get("schema")))
    try:
        floor_mean = float(floor["measured_mean_kld"])
        quant_mean = float(quant_mean)
    except (TypeError, ValueError):
        raise Refusal("quant and floor means must be numeric")
    if not math.isfinite(quant_mean) or not math.isfinite(floor_mean):
        raise Refusal("quant and floor means must be finite")
    quant_teacher = quant.get("teacher_receipt_sha256")
    floor_teacher = floor.get("teacher_receipt_sha256")
    if not isinstance(quant_teacher, str) or not isinstance(floor_teacher, str):
        raise Refusal("quant and floor must declare teacher_receipt_sha256 strings")

    # Gate 2: cross-lane guard -- the teacher sha is the identity of the
    # reference the number was measured against.
    if not quant_teacher or not floor_teacher or quant_teacher != floor_teacher:
        raise _cross_lane_refusal(float(quant_mean), quant_teacher or "(absent)",
                                  floor_mean, floor_teacher or "(absent)")
    gates.append("teacher_receipt_sha256 identical (%s...)" % quant_teacher[:16])

    # Gate 2b: the floor's PROFILE must name the lane its teacher belongs to.
    # The teacher-sha gate alone cannot tell the streaming floor (0.011506)
    # from a cross-stack summary re-labelled with the same teacher sha; the
    # profile field is the lane's name on the floor run itself.
    floor_profile = floor.get("profile")
    if not floor_profile:
        raise Refusal(
            "floor summary carries no 'profile' field -- a floor without a "
            "declared lane profile cannot be matched to the quant run's lane. "
            "The streaming lane's own floor (engines/native-bf16-kld.json) carries "
            "profile %r." % STREAM_FLOOR_PROFILE)
    if quant_teacher == SEALED_STREAM_TEACHER and \
            floor_profile != STREAM_FLOOR_PROFILE:
        raise Refusal(
            "floor claims the sealed streaming teacher %s... but declares "
            "profile %r; the only floor measured on that (panel, teacher, "
            "lane) is profile %r (engines/native-bf16-kld.json, 0.011506). A "
            "floor from any other lane must not be subtracted here. %s"
            % (quant_teacher[:16], floor_profile, STREAM_FLOOR_PROFILE,
               CROSS_LANE_WORKED_EXAMPLE))
    gates.append("floor profile %r names its lane" % floor_profile)

    # Gate 2c: lane-ONLY identity, when both receipts carry it (captures made
    # after 2026-08-29 emit student_lane_identity_sha256 -- a hash over the
    # lane fields alone, comparable across quants).  Equality VERIFIES the
    # lane; inequality refuses with both hashes printed.
    q_lid = (quant.get("student_lane_identity_sha256")
             or quant.get("lane_identity_sha256"))
    f_lid = (floor.get("student_lane_identity_sha256")
             or floor.get("lane_identity_sha256"))
    if q_lid and f_lid:
        if q_lid != f_lid:
            raise Refusal(
                "quant and floor declare DIFFERENT lane identities (%s vs %s) "
                "-- the floor is (panel, teacher, lane)-specific and this "
                "subtraction would mix lanes. %s"
                % (_sha_pair(q_lid, f_lid, 12) + (CROSS_LANE_WORKED_EXAMPLE,)))
        gates.append("lane_identity_sha256 identical -- lane equality "
                     "VERIFIED (%s...)" % q_lid[:12])

    # Gate 3: backend/lane drift, when a per-run report is provided.
    drift = False
    if args.quant_report:
        report = _load_json(Path(args.quant_report), "per-run kld report")
        if report.get("teacher_receipt_sha256") not in (None, quant_teacher):
            raise Refusal(
                "--quant-report %s names teacher %s but the summary names %s "
                "-- wrong report/summary pair"
                % ((args.quant_report,)
                   + _sha_pair(report["teacher_receipt_sha256"], quant_teacher)))
        for field in ("token_panel_receipt_sha256",):
            left, right = report.get(field), floor.get(field)
            if left and right and left != right:
                if not args.allow_backend_drift:
                    raise Refusal(
                        "field %s differs between the quant run (%s...) and "
                        "the floor (%s...). Pass --allow-backend-drift to "
                        "print both numbers anyway; the output will be "
                        "stamped floor_backend_drift: true."
                        % ((field,) + _sha_pair(left, right, 12)))
                drift = True
            elif left and not right:
                gates.append("field %s present on the quant side only (the "
                             "surviving floor summary does not carry it); "
                             "teacher sha is the operative guard" % field)
        gates.append("per-run report cross-checked against the summary")

    # This legacy scalar summary has no bound teacher/student tensor operands.
    # An evidence_kind label plus prose is not a content-identity proof.
    if floor_mean == 0.0:
        raise Refusal(
            "a legacy scalar floor summary cannot establish a zero control; "
            "zero_floor_evidence labels are not bound tensor evidence",
            ["use fidelity-dataset compare on two independently captured, sealed "
             "datasets with --self-compare --force-compute, and retain the "
             "comparison/qualification receipts; do not synthesize a scalar "
             "summary from an evidence-kind label"])

    attributable = float(quant_mean) - floor_mean
    label = quant.get("student_label") or quant.get("profile") or "?"
    con.say("  quant   %-14s mean %.18g   (teacher %s...)"
            % (label, quant_mean, quant_teacher[:12]))
    con.say("  floor   %-14s mean %.18g" % (floor.get("student_label"), floor_mean))
    con.say("  attributable = %.18g - %.18g = %+.18g nats"
            % (quant_mean, floor_mean, attributable))
    if drift:
        con.warn("floor_backend_drift: true -- the lanes differ in a declared "
                 "field; treat the subtraction as indicative, not sealed")

    out = {
        "schema": ATTRIBUTABLE_SCHEMA,
        "quant_label": label,
        "quant_source": origin,
        "quant_mean": float(quant_mean),
        "floor_mean": floor_mean,
        "attributable": attributable,
        "interpretation": "descriptive native-baseline KL contrast, not an identified codec effect",
        "equation": "%.18g - %.18g = %+.18g" % (quant_mean, floor_mean, attributable),
        "floor_source_path": str(floor_path),
        "teacher_receipt_sha256": quant_teacher,
        "gates_passed": gates,
        "floor_backend_drift": drift,
        "zero_floor": floor_mean == 0.0,
        "generated_by": "bin/fidelity-stats attributable",
        "not_submittable": True,
    }
    if args.out:
        write_json(args.out, out)
        con.say("  written: %s" % args.out)
    con.say("")
    con.say("  NOTE: 'attributable' is the historical field name for this "
            "descriptive baseline contrast. KL is not additive; even small "
            "same-lane values do not identify a causal quantization effect.")
    return EXIT_OK


# ==========================================================================
# paired-delta
# ==========================================================================


def _windows_from_report(path: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    doc = _load_json(path, "kld report")
    rows = doc.get("per_window")
    if not isinstance(rows, list) or not rows:
        raise Refusal("%s needs a nonempty per_window list; scalar summaries cannot be paired" % path)
    means = {}
    for row in rows:
        if not isinstance(row, dict):
            raise Refusal("per_window entries must be objects")
        window = row.get("window_id")
        if not isinstance(window, str) or not window.strip() or window in means:
            raise Refusal("window_id must be nonempty and unique: %r" % window)
        try:
            mean = float(row["summary"]["mean"])
        except (KeyError, TypeError, ValueError):
            raise Refusal("window %s needs a numeric summary.mean" % window)
        if not math.isfinite(mean):
            raise Refusal("window %s has a nonfinite mean" % window)
        means[window] = mean
    return means, doc


def _windows_from_anomaly(path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    doc = _load_json(path, "anomaly investigation")
    rows = doc.get("per_window")
    if not isinstance(rows, list) or not rows:
        raise Refusal("%s has no per_window engines/k8 rows" % path)
    a, b = {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise Refusal("anomaly per_window entries must be objects")
        window = row.get("window")
        if not isinstance(window, str) or not window.strip() or window in a:
            raise Refusal("anomaly window must be nonempty and unique: %r" % window)
        try:
            a[window], b[window] = float(row["k6"]), float(row["k8"])
        except (KeyError, TypeError, ValueError):
            raise Refusal("anomaly window %s needs numeric k6 and k8 means" % window)
        if not math.isfinite(a[window]) or not math.isfinite(b[window]):
            raise Refusal("anomaly window %s has a nonfinite mean" % window)
    return a, b


def _paired_provenance(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]],
                       window_key: str = "window_id") -> Optional[Dict[str, str]]:
    """Validate declared per-window identities without inventing source maps."""
    a = {r[window_key]: r for r in rows_a}
    b = {r[window_key]: r for r in rows_b}
    if set(a) != set(b):
        raise Refusal("window sets differ; pairing requires identical windows")
    fields = ("document_id", "token_ids_sha256", "token_ids_json_sha256",
              "attention_mask_sha256", "prediction_positions")
    for field in fields:
        if not any(r.get(field) is not None for r in rows_a + rows_b):
            continue
        for w in a:
            left, right = a[w].get(field), b[w].get(field)
            valid = (isinstance(left, int) and not isinstance(left, bool) and left > 0
                     if field == "prediction_positions"
                     else isinstance(left, str) and bool(left.strip()))
            if not valid or left != right:
                raise Refusal("window %s has incomplete or conflicting %s" % (w, field))
    for w in a:
        ca, cb = a[w].get("summary", {}).get("count"), b[w].get("summary", {}).get("count")
        if ca is not None or cb is not None:
            if not isinstance(ca, int) or isinstance(ca, bool) or ca <= 0 or ca != cb:
                raise Refusal("window %s has incomplete or conflicting scored count" % w)
    if any(r.get("document_id") is not None for r in rows_a + rows_b):
        return {w: a[w]["document_id"] for w in a}
    return None


def cmd_paired_delta(args: argparse.Namespace, con: Console) -> int:
    con.say("fidelity-stats paired-delta")
    con.rule()
    gates: List[str] = []
    lane_note: Optional[Dict[str, Any]] = None
    if args.bootstrap_b < 2:
        raise Refusal("--bootstrap-b must be at least 2")
    if args.anomaly_format:
        a_means, b_means = _windows_from_anomaly(Path(args.report_a))
        label_a = args.label_a or "k6"
        label_b = args.label_b or "k8"
        rows = _load_json(Path(args.report_a), "anomaly investigation")["per_window"]
        documents = _paired_provenance(rows, rows, "window")
        gates.append("anomaly-format input: both series supplied in one investigation; "
                     "reference/panel/lane pairing is asserted by its author, not reverified")
    else:
        if not args.report_b:
            raise Refusal("pass --report-b (or --anomaly-format for the "
                          "committed K8-ANOMALY.json)")
        a_means, doc_a = _windows_from_report(Path(args.report_a))
        b_means, doc_b = _windows_from_report(Path(args.report_b))
        label_a = args.label_a or doc_a.get("student_label") or "A"
        label_b = args.label_b or doc_b.get("student_label") or "B"
        documents = _paired_provenance(doc_a["per_window"], doc_b["per_window"])
        for field in ("teacher_receipt_sha256", "token_panel_receipt_sha256"):
            left, right = doc_a.get(field), doc_b.get(field)
            if not isinstance(left, str) or not left.strip() or not isinstance(right, str) or not right.strip():
                raise Refusal("both reports must declare %s; window IDs alone do not establish pairing" % field)
            if left and right and left != right:
                raise Refusal(
                    "runs disagree on %s (%s... vs %s...) -- these runs were "
                    "measured against different references; this command "
                    "requires matching reference and panel identities."
                    % ((field,) + _sha_pair(left, right, 12)))
            if left and right:
                gates.append("%s identical" % field)
        # LANE DISCLOSURE, not a gate: student_backend_identity_sha256 pins
        # the artifact AND the lane together (it hashes checkpoint identity
        # alongside torch/device/kernel), so two different quants on the SAME
        # lane still differ -- inequality proves nothing, equality is not
        # expected. Same-lane provenance supports a controlled contrast, not
        # an additive decomposition of KL into floor plus codec error.
        lane_a = doc_a.get("student_backend_identity_sha256")
        lane_b = doc_b.get("student_backend_identity_sha256")
        # LANE GATE, when both reports carry the lane-ONLY identity emitted
        # by captures made after 2026-08-29 (student_lane_identity_sha256:
        # sha256 over torch/device/kernel/numeric-policy/parallelism/reduce
        # order and nothing artifact-specific).  Unlike the backend hash it
        # IS comparable across quants, so equality verifies the lane and
        # inequality refuses.
        lids = []
        for doc in (doc_a, doc_b):
            declared = [doc[k] for k in ("student_lane_identity_sha256", "lane_identity_sha256")
                        if doc.get(k) is not None]
            if declared and (any(not isinstance(v, str) or not v.strip() for v in declared)
                             or len(set(declared)) != 1):
                raise Refusal("incomplete or conflicting lane identity aliases")
            lids.append(declared[0] if declared else None)
        lid_a, lid_b = lids
        if bool(lid_a) != bool(lid_b):
            raise Refusal("lane identity is declared on only one report")
        if lid_a and lid_b:
            if lid_a != lid_b:
                raise Refusal(
                    "runs declare DIFFERENT lane identities (%s vs %s) -- "
                    "this command requires a same-lane contrast, not a "
                    "mixture of device/kernel/numeric-policy effects."
                    % _sha_pair(lid_a, lid_b, 12))
            gates.append("student_lane_identity_sha256 identical -- lane "
                         "equality VERIFIED (%s...)" % lid_a[:12])
            lane_note = {
                "student_lane_identity_sha256": lid_a,
                "student_backend_identity_sha256_a": lane_a,
                "student_backend_identity_sha256_b": lane_b,
                "lane_equality_verified": True,
                "note": "lane-only identity hashes match; this controls lane "
                        "provenance but does not identify a causal codec effect",
            }
        else:
            lane_note = {
                "student_backend_identity_sha256_a": lane_a,
                "student_backend_identity_sha256_b": lane_b,
                "lane_equality_verified": False,
                "note": ("backend identity hashes pin artifact+lane together; "
                         "confirm same lane (device/torch/kernel/reduce order) "
                         "from the two runs' backend.json; the "
                         "KL contrast does not identify a codec effect"),
            }
            con.warn("lane equality NOT verifiable from these reports (backend "
                     "hashes pin artifact+lane together: %s... vs %s...); the "
                     "contrast does not establish a codec effect; "
                     "confirm lane provenance from backend.json"
                     % _sha_pair(lane_a or "?", lane_b or "?", 12))

    if set(a_means) != set(b_means):
        only_a = sorted(set(a_means) - set(b_means))[:3]
        only_b = sorted(set(b_means) - set(a_means))[:3]
        raise Refusal(
            "window sets differ (only-%s: %s; only-%s: %s) -- pairing "
            "requires identical windows" % (label_a, only_a, label_b, only_b))
    windows = sorted(a_means)
    n = len(windows)
    gates.append("identical window-id sets (n=%d)" % n)

    deltas = [b_means[w] - a_means[w] for w in windows]
    if not all(math.isfinite(d) for d in deltas):
        raise Refusal("paired differences must be finite")
    d_bar = statistics.fmean(deltas)
    s_d = statistics.stdev(deltas) if n > 1 else None
    se = s_d / math.sqrt(n) if s_d is not None else None
    t = d_bar / se if se else None
    df = n - 1 if n > 1 else None
    p_t = t_two_sided_p(abs(t), df) if t is not None else None
    t_crit = t_quantile_975(df) if df else None
    ci_t = ((d_bar - t_crit * se, d_bar + t_crit * se)
            if t_crit is not None else (None, None))
    boot = bca_interval(deltas, args.bootstrap_b, args.seed) if n > 1 else None
    wins_b = sum(1 for d in deltas if d < 0)             # b better = lower KLD
    p_sign = sign_test_two_sided(wins_b, sum(1 for d in deltas if d != 0.0))
    wilcox = wilcoxon_signed_rank(deltas)
    if documents is not None:
        # jointstd's contract is A-B. Preserve its A/B win labels while converting
        # signed estimates and interval endpoints to this CLI's historical B-A.
        doc_block = document_level_paired([-d for d in deltas], windows, documents)
        for row in doc_block["per_document"]:
            row["mean_diff"] = -row["mean_diff"]
        if "mean_diff_equal_documents" in doc_block:
            doc_block["mean_diff_equal_documents"] = -doc_block["mean_diff_equal_documents"]
        if "ci95_diff_t" in doc_block:
            lo, hi = doc_block["ci95_diff_t"]
            doc_block["ci95_diff_t"] = [-hi, -lo]
        doc_block["difference_convention"] = "B-A"
        inference_unit = "source_document"
    else:
        doc_block = {"available": False, "reason": "no source-document provenance supplied"}
        inference_unit = "none"

    con.say("  pairing %s (A) vs %s (B) over %d windows; d_j = B_j - A_j"
            % (label_a, label_b, n))
    con.say("  descriptive equal-window d_bar %+.6e; s_d=%r SE=%r" % (d_bar, s_d, se))
    con.say("  descriptive window t p=%r; t-CI95=%r; BCa=%r" % (p_t, ci_t, boot))
    con.say("  %s wins %d/%d windows; descriptive window sign p=%r"
            % (label_b, wins_b, n, p_sign))
    if documents is not None:
        con.say("  SOURCE DOCUMENTS: %d; %s wins %d; exact sign p=%r (conditional on "
                "independent, exchangeable documents)"
                % (doc_block["n_documents"], label_b, doc_block["documents_b_better"],
                   doc_block["sign_test_p"]))
    else:
        con.warn("DESCRIPTIVE ONLY: no source-document provenance; no population inference")
    caveat = "Only the supplied windows are described; no complete-panel scope is inferred."
    con.say("")
    con.say("  ESTIMAND: %s" % ESTIMAND_TEXT)
    con.say("")

    out = {
        "schema": PAIRED_SCHEMA,
        "label_a": label_a, "label_b": label_b,
        "window_count": n,
        "window_ids": windows,
        "deltas_b_minus_a": deltas,
        "d_bar": d_bar, "s_d": s_d, "se": se,
        "t": t, "df": df, "p_two_sided_t": p_t,
        "ci95_t": {"low": ci_t[0], "high": ci_t[1], "t_crit": t_crit},
        "ci95_bca": boot,
        "sign_test": {"wins_b": wins_b, "n": n,
                      "informative_n": sum(d != 0.0 for d in deltas),
                      "p_two_sided": p_sign},
        "wilcoxon": wilcox,
        "document_level": doc_block,
        "inference_unit": inference_unit,
        "window_stats_are": "descriptive_only",
        "degenerate_note": ("one window: variance and intervals unavailable" if n == 1 else
                            "zero window variance: t test unavailable; resampling spread "
                            "is degenerate, not proof of certainty" if se == 0 else None),
        "gates_passed": gates,
        "subset_caveat": caveat,
        "lane_disclosure": lane_note,
        "estimand": ESTIMAND_TEXT,
        "generated_by": "bin/fidelity-stats paired-delta",
        "not_submittable": True,
    }
    if args.out:
        write_json(args.out, out)
        con.say("  written: %s" % args.out)
    return EXIT_OK


# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fidelity-stats", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("attributable",
                       help="quant mean minus the SAME-lane floor, gated")
    a.add_argument("--quant-summary", help="a *-packed-kld-summary receipt")
    a.add_argument("--quant-report", help="optional per-run kld-report.json "
                                          "for extra drift gates")
    a.add_argument("--from-registry", metavar="MEASUREMENT_ID",
                   help="fetch the quant receipt via the registry instead")
    a.add_argument("--registry", default="auto")
    a.add_argument("--floor-summary", required=True,
                   help="the lane's floor summary (streaming lane: "
                        "engines/native-bf16-kld.json)")
    a.add_argument("--allow-backend-drift", action="store_true")
    a.add_argument("--out")

    d = sub.add_parser("paired-delta",
                       help="descriptive window contrast with conditional source-document inference")
    d.add_argument("--report-a", required=True)
    d.add_argument("--report-b")
    d.add_argument("--anomaly-format", action="store_true",
                   help="--report-a is engines/K8-ANOMALY.json (both series in one file)")
    d.add_argument("--label-a")
    d.add_argument("--label-b")
    d.add_argument("--bootstrap-b", type=int, default=10000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()
    try:
        if args.cmd == "attributable":
            return cmd_attributable(args, con)
        if args.cmd == "paired-delta":
            return cmd_paired_delta(args, con)
    except Refusal as exc:
        con.say("")
        con.say("REFUSED: %s" % exc.reason)
        for line in exc.advice:
            con.say("         %s" % line)
        return EXIT_REFUSED
    build_parser().print_help()
    return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
