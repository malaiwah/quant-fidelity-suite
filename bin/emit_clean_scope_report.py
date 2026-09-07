#!/usr/bin/env python3
"""Emit the calibration-clean-scope recompute as a publishable report.

    bin/emit_clean_scope_report.py --out reports/clean-scope-recompute.json

WHAT THIS IS.  brandonmusic ran a 13-gram overlap scan against the
calibration-role windows of his own sealed panel and found that one entire
domain of the FINAL windows -- the reasoning axis -- shares 37-39% of its
13-grams with calibration material, despite the panel being clean at the
DOCUMENT-hash level.  He excluded that domain and scored his primary numbers on
the 17 windows that survive.

Every malaiwah number published on that panel used all 25 windows, so every one
of them carries the same contamination.  This report recomputes them on his
clean scope.  It costs no GPU: our published receipts carry per-window arrays,
so the clean-scope mean is the equal-weight mean of the 17 windows his scan
keeps.  That is arithmetic on data we already published, not a re-measurement.

WHAT IT IS NOT.  It is not a correction to any published number.  The panel25
values remain correct FOR PANEL25.  What changes is that panel25 is now known to
include calibration-adjacent windows, so a panel25 number and a clean17 number
are answers to different questions and must never be differenced.  The registry
enforces that structurally: clean17 is its own derived panel with its own
comparability key, so a clean17-vs-panel25 table cannot be built by accident.

Scope deltas and FP8-minus-BF16 differences are descriptive arithmetic, not a
causal decomposition or evidence that subtraction is stable across populations.

Sources, all committed and all re-derivable:
  registry/protocol/per-window/*.json                 six per-window series
  registry/protocol/window-selection.brandonmusic-final25.json   his 17-window scope
  docs/joint-standard/analysis/*.json                 the emitted BCa analyses
  registry/protocol/glm53-joint-kld-protocol.v1.json  the frozen protocol

Stdlib only; no network; no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SERIES = [
    ("k6-sealed", "K6 tr3-6bpw sealed 5-run",
     "measurement--glm53.k6-6bpw.brandonmusic-final25", "sealed-ep8"),
    ("k6-streaming", "K6 tr3-6bpw streaming 2-run",
     "measurement--glm53.k6-6bpw.brandonmusic-final25-stream", "streaming"),
    ("k8-streaming", "K8 tr3-8bpw streaming 2-run",
     "measurement--glm53.k8-8bpw.brandonmusic-final25-stream", "streaming"),
    ("fp8-crossstack", "official FP8 (cross-stack)",
     "measurement--glm53.official-fp8.brandonmusic-final25", "cross-stack"),
    ("bf16-floor-crossstack", "BF16 replay floor (cross-stack)",
     "measurement--glm53.bf16-replay-floor.brandonmusic-final25", "cross-stack"),
    ("brandonmusic-4bpw", "brandonmusic tr3-4bpw (his artifact, our scorer)",
     "measurement--glm53.brandonmusic-4bpw.brandonmusic-final25", "cross-stack"),
]

PAIRED = [
    ("paired.K6-vs-K8", "K6 - K8"),
    ("paired.K6-vs-FP8", "K6 - FP8"),
    ("paired.K8-vs-FP8", "K8 - FP8"),
    ("paired.FP8-vs-BF16floor", "FP8 - BF16 floor"),
    ("paired.BM4bpw-vs-K6", "his 4bpw - K6"),
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def scope_mean(per_window, keep=None):
    rows = [w for w in per_window if keep is None or w["window_id"] in keep]
    counts = {w["count"] for w in rows}
    if len(counts) != 1:
        raise SystemExit("mixed window sizes %s -- the equal-weight mean is not "
                         "the token-weighted mean, so this report would be wrong"
                         % sorted(counts))
    return math.fsum(w["mean"] for w in rows) / len(rows), len(rows), counts.pop()


def atomic_text(path, text):
    """Publish a complete file; interruptions never truncate an existing report."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".clean-scope-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def analysis_with_source(path, root):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw), {
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def portable_locations(value):
    """Retain distinct producer-local identities without publishing host paths."""
    if isinstance(value, dict):
        return {key: portable_locations(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_locations(item) for item in value]
    if isinstance(value, str) and value.startswith(("/", "file://")):
        return "producer-local-path-sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    return value


def disclosures(source):
    """Carry scientific qualifications intact, including future nested fields."""
    keys = ("document_level", "document_map_source", "inference_unit",
            "window_stats_are", "cross_lane", "contract_a", "contract_b",
            "not_submittable", "scope_complete", "degenerate_note", "source_analysis")
    result = {k: source[k] for k in keys if k in source}
    result.setdefault("inference_unit", "unknown")
    result.setdefault("window_stats_are",
                      "Descriptive of this fixed panel only; source-document "
                      "sampling provenance and assumptions are not established.")
    return result

def render_disclosures(source, depth=0):
    """Readable nested qualifications without dropping assumptions or provenance."""
    lines = []
    for key, value in source.items():
        prefix = "  " * depth + "- **%s**:" % key.replace("_", " ")
        if isinstance(value, dict):
            lines.append(prefix)
            lines.extend(render_disclosures(value, depth + 1))
        elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
            lines.append(prefix)
            for index, item in enumerate(value, 1):
                lines.extend(render_disclosures({str(index): item}, depth + 1))
        else:
            lines.append(prefix + " " + (json.dumps(value) if not isinstance(value, str)
                                        else value))
    return lines



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", help="also render a human-readable table")
    args = ap.parse_args()
    R = args.root

    pw_dir = os.path.join(R, "registry/protocol/per-window")
    an_dir = os.path.join(R, "docs/joint-standard/analysis")
    sel_path = os.path.join(
        R, "registry/protocol/window-selection.brandonmusic-final25.json")
    proto_path = os.path.join(R, "registry/protocol/glm53-joint-kld-protocol.v1.json")
    sel = load(sel_path)
    proto = load(proto_path)
    clean = set(sel["selected_windows"])

    doc = {
        "schema": "malaiwah.glm53-clean-scope-recompute/2",
        "supersedes_schema": "malaiwah.glm53-clean-scope-recompute/1",
        "title": "GLM-5.3-Flash fidelity: brandonmusic's calibration-clean scope, "
                 "recomputed from published per-window data",
        "generated_by": "bin/emit_clean_scope_report.py",
        "gpu_used": False,
        "is_a_correction": False,
        "summary": (
            "brandonmusic's 13-gram calibration-overlap scan excludes 8 of the 25 "
            "windows of his sealed final panel -- the whole axis4_reasoning domain "
            "plus final-0021 and final-0022 -- leaving a 17-window "
            "calibration-clean scope. Every malaiwah number published on that panel "
            "used all 25 windows and therefore carries the same contamination. This "
            "report recomputes all six series on his clean scope from our own "
            "published per-window arrays. Scope movements describe these two "
            "finite panels only. FP8-minus-BF16 is a descriptive contrast, not "
            "identified quantization error; mixed-lane results do not rank codecs."),
    }

    # ---------------------------------------------------------------- provenance
    doc["provenance"] = {
        "protocol_file": "registry/protocol/glm53-joint-kld-protocol.v1.json",
        "protocol_file_sha256": sha256_file(proto_path),
        "protocol_scoring_sha256": proto.get("scoring_sha256"),
        "window_selection_file": "registry/protocol/window-selection.brandonmusic-final25.json",
        "window_selection_sha256": sha256_file(sel_path),
        "scan": {
            "method": sel.get("method"),
            "ngram_n": sel.get("ngram_n"),
            "threshold": sel.get("threshold"),
            "calibration_windows_scanned": sel.get("calibration_windows_scanned"),
            "calibration_grams": sel.get("calibration_grams"),
        },
        "cross_check_against_his_published_selection": sel.get("cross_check"),
        "credit": (
            "The contamination finding, the 13-gram scan and the 0.05 threshold "
            "are brandonmusic's. His harness, protocol, receipts and panel: "
            "huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/tree/main/eval/kld ; "
            "teacher logits: huggingface.co/datasets/brandonmusic/"
            "GLM-5.3-Flash-BF16-Teacher-Logits . This report is the arithmetic "
            "consequence of his finding applied to our published numbers."),
    }

    # -------------------------------------------------------------------- scopes
    doc["scopes"] = {
        "panel25": {
            "panel_ref": "panel--glm53.brandonmusic.final25",
            "windows": 25, "positions_per_window": 2047, "scored_positions": 25 * 2047,
            "note": "the full sealed panel, and the scope of every previously "
                    "published malaiwah number on it",
        },
        "clean17": {
            "panel_ref": "panel--glm53.brandonmusic.final25-clean17",
            "windows": len(clean), "positions_per_window": 2047,
            "scored_positions": len(clean) * 2047,
            "window_ids": sorted(clean),
            "excluded": [
                {"window_id": e["window_id"],
                 "shared_ngram_fraction": e.get("shared_ngram_fraction"),
                 "domain": e.get("domain")}
                for e in sel.get("excluded_windows", [])],
            "note": "a different window set is a different panel: clean17 carries "
                    "its own comparability key so that a clean17-vs-panel25 "
                    "difference is structurally impossible, not merely discouraged",
        },
    }

    # --------------------------------------------------------------------- rows
    means = {}
    rows = []
    for key, label, mid, lane in SERIES:
        pw = load(os.path.join(pw_dir, key + ".json"))
        row = {"series": key, "label": label, "lane": lane,
               "registry_measurement_ref": mid,
               "source_note": pw.get("note"), "scopes": {}}
        for scope, keep, suffix in (("panel25", None, "panel"),
                                    ("clean17", clean, "selected")):
            m, n, per = scope_mean(pw["per_window"], keep)
            means[(key, scope)] = m
            entry = {"mean_kld_nats": m, "windows": n, "scored_positions": n * per}
            ap_ = os.path.join(an_dir, "%s.%s.json" % (key, suffix))
            if os.path.exists(ap_):
                a, analysis_source = analysis_with_source(ap_, R)
                if abs(a["summary"]["mean"] - m) > 1e-15:
                    raise SystemExit("%s/%s: the emitted analysis disagrees with the "
                                     "recomputed mean" % (key, scope))
                entry.update(disclosures(a))
                entry["source_analysis"] = analysis_source
                entry["se_clustered_window"] = a["summary"]["se_clustered_window"]
                entry["ci95_bca"] = a["bootstrap"]["ci95_bca"]
                entry["ci95_percentile"] = a["bootstrap"]["ci95_percentile"]
                entry["bootstrap_b"] = a["bootstrap"]["b"]
                entry["bootstrap_seed"] = a["bootstrap"]["seed"]
                entry["interval_kind"] = "bca"
                entry["cluster_unit"] = "window"
                if a["summary"].get("deff_window") is not None:
                    entry["deff_window"] = a["summary"]["deff_window"]
                sr = a.get("sigma_run") or {}
                if sr.get("sigma_run") is not None:
                    entry["sigma_run"] = sr["sigma_run"]
                    entry["sigma_run_runs"] = sr["runs"]
                # The stratum interval stopped being BCa on 2026-08-30 (STAT-01: BCa
                # measures 81.3% coverage at the 5-7 windows a domain has). Emit
                # whichever interval the table actually produced, and NAME it --
                # a key called ci95_bca holding a Student-t interval would be the
                # same class of mislabel the reseed was done to remove.
                entry["by_domain"] = [
                    {**d, "domain": d["domain"], "windows": d["n_clusters_window"],
                     "scored_positions": d["n"], "mean_kld_nats": d["mean"],
                     "se_clustered_window": d.get("se_clustered_window"),
                     "interval_method": d.get("interval_method"),
                     "ci95": (d.get("ci95_t_log") or d.get("ci95_bca")),
                     "ci95_bca": d.get("ci95_bca")}
                    for d in a.get("by_domain", [])]
            row["scopes"][scope] = entry
        p, c = means[(key, "panel25")], means[(key, "clean17")]
        row["scope_delta"] = {
            "absolute_nats": c - p,
            "relative_pct": (c - p) / p * 100.0,
            "direction": "down" if c < p else "up",
        }
        rows.append(row)
    doc["rows"] = rows

    doc["headline"] = {
        "five_of_six_fall": [r["label"] for r in rows
                             if r["scope_delta"]["direction"] == "down"],
        "rises": [r["label"] for r in rows if r["scope_delta"]["direction"] == "up"],
        "note": ("Scope selection changes the finite-panel estimand. Opposite "
                 "movements do not identify a causal contamination effect or "
                 "establish a population ordering."),
    }

    # ------------------------------------------------------------- attributable
    att = {}
    for scope in ("panel25", "clean17"):
        f = means[("fp8-crossstack", scope)]
        b = means[("bf16-floor-crossstack", scope)]
        att[scope] = {"fp8": f, "same_lane_floor": b,
                      "difference_nats": f - b}
    a_p, a_c = att["panel25"]["difference_nats"], att["clean17"]["difference_nats"]
    doc["descriptive_fp8_bf16_contrast"] = {
        "lane": "cross_stack (both sides; not sufficient for causal subtraction)",
        "by_scope": att,
        "difference_move_pct": (a_c - a_p) / a_p * 100.0,
        "fp8_move_pct": ((means[("fp8-crossstack", "clean17")]
                          - means[("fp8-crossstack", "panel25")])
                         / means[("fp8-crossstack", "panel25")] * 100.0),
        "floor_move_pct": ((means[("bf16-floor-crossstack", "clean17")]
                            - means[("bf16-floor-crossstack", "panel25")])
                           / means[("bf16-floor-crossstack", "panel25")] * 100.0),
        "reading": ("Both inputs move by 9-16% between these scopes and their "
                    "difference moves by 1.4%. This is descriptive arithmetic "
                    "on two fixed panels, not evidence of stable subtraction "
                    "or an additive, causal quantization-error component."),
        "what_cannot_be_recomputed": (
            "The same-lane K6/K8 floor has a scalar-only receipt, so it cannot "
            "be recomputed on clean17. The historical 2.52x residual ratio "
            "remains withdrawn: uncertainty of the small control residuals "
            "was not established. A signed control contrast is not itself KL "
            "or automatically causal quantization error. No cross-lane floor is substituted."),
    }

    # -------------------------------------------------------------------- paired
    paired = []
    for stem, label in PAIRED:
        entry = {"comparison": label, "scopes": {}}
        for scope, suffix in (("panel25", "panel"), ("clean17", "selected")):
            p = os.path.join(an_dir, "%s.%s.json" % (stem, suffix))
            if not os.path.exists(p):
                continue
            d, analysis_source = analysis_with_source(p, R)
            r = d.get("paired") or d
            keep = ("label_a", "label_b", "mean_a", "mean_b", "mean_diff",
                    "mean_diff_se", "n_windows", "windows_a_better",
                    "windows_b_better", "windows_tied", "sign_test_n")
            missing = [k for k in ("mean_a", "mean_b", "mean_diff") if k not in r]
            if missing:
                raise SystemExit(
                    "%s: paired receipt is missing %s. Emitting a paired block "
                    "with silently absent fields is how an incomplete table gets "
                    "published." % (os.path.basename(p), ", ".join(missing)))
            entry["scopes"][scope] = {k: r[k] for k in keep if k in r}
            entry["scopes"][scope].update(disclosures({**d, **r}))
            entry["scopes"][scope]["source_analysis"] = analysis_source
            entry["scopes"][scope]["interpretation"] = (
                "Descriptive mixed-design contrast; not a codec ranking."
                if entry["scopes"][scope].get("cross_lane", {}).get("mixed") else
                "Descriptive finite-panel contrast; a ranking requires the "
                "registry's actual pair predicate, not shared windows alone.")
            # Preserve historical diagnostics under an explicitly non-inferential
            # namespace, but do not republish ratios across measurement designs.
            entry["scopes"][scope]["descriptive_window_diagnostics"] = {
                k: r[k] for k in ("ci95_diff_bca", "ci95_diff_percentile",
                                 "sign_test_p", "bootstrap_b", "seed") if k in r}
            mc = r.get("mcnemar")
            if mc:
                entry["scopes"][scope]["mcnemar"] = mc
        if entry["scopes"]:
            paired.append(entry)
    doc["paired"] = {
        "method": "Finite-panel paired differences are descriptive. Document-level "
                  "summaries retain their provenance, weighting and assumptions; "
                  "no unconditional significance or codec ranking is inferred.",
        "comparisons": paired,
    }

    # ---------------------------------------------------- per-domain, clean scope
    dom = {}
    for key in ("k6-sealed", "k8-streaming", "fp8-crossstack",
                "bf16-floor-crossstack"):
        p = os.path.join(an_dir, "%s.selected.json" % key)
        if os.path.exists(p):
            dom[key] = {d["domain"]: d for d in load(p)["by_domain"]}
    table = []
    if dom:
        for d in sorted(dom["k6-sealed"]):
            e = {"domain": d,
                 "windows": dom["k6-sealed"][d]["n_clusters_window"],
                 "scored_positions": dom["k6-sealed"][d]["n"]}
            for key in dom:
                e[key] = dom[key][d]["mean"]
            table.append(e)
        doc["per_domain_clean17"] = {
            "table": table,
            "reading": "Finite-panel domain means only. Cross-lane ratios, their "
                       "spread and a worst-codec/domain ranking are not identified "
                       "by this mixed measurement design.",
        }

    doc["producer_path_policy"] = (
        "Producer-local absolute path labels are replaced by distinct opaque hashes. "
        "source_analysis names the original committed analysis by portable path and "
        "file digest; no replacement path is guessed to contain the old bytes.")
    doc = portable_locations(doc)


    if args.markdown:
        # Rendered FROM the emitted document, never typed alongside it, so the
        # human-readable table cannot drift away from the machine-readable one.
        L = []
        A = L.append
        A("# Calibration-clean scope: GLM-5.3-Flash fidelity recomputed\n")
        A(doc["summary"] + "\n")
        A("Generated by `%s` from committed per-window data. No GPU, no "
          "re-measurement. **This is not a correction**: the panel25 values "
          "remain correct for panel25.\n" % doc["generated_by"])
        A(doc["producer_path_policy"] + "\n")
        A("## Scope\n")
        A("| scope | windows | scored positions |")
        A("|---|---:|---:|")
        for k in ("panel25", "clean17"):
            sc = doc["scopes"][k]
            A("| `%s` | %d | %s |" % (k, sc["windows"], "{:,}".format(sc["scored_positions"])))
        A("")
        A("Excluded by the 13-gram scan at threshold %s (his scan, his threshold):\n"
          % doc["provenance"]["scan"]["threshold"])
        A("| window | domain | shared 13-gram fraction |")
        A("|---|---|---:|")
        for e in doc["scopes"]["clean17"]["excluded"]:
            A("| `%s` | %s | %.4f |" % (e["window_id"], e["domain"],
                                        e["shared_ngram_fraction"]))
        A("")
        A("## Means, both scopes\n")
        A("| row | lane | panel25 | clean17 | move | descriptive window BCa 95% |")
        A("|---|---|---:|---:|---:|---|")
        for r in doc["rows"]:
            c = r["scopes"]["clean17"]
            ci = c.get("ci95_bca")
            A("| %s | `%s` | %.12f | %.12f | %+.2f %% | %s |"
              % (r["label"], r["lane"], r["scopes"]["panel25"]["mean_kld_nats"],
                 c["mean_kld_nats"], r["scope_delta"]["relative_pct"],
                 ("[%.6f, %.6f]" % (ci[0], ci[1])) if ci else "--"))
        A("")
        A("Window intervals above describe fixed-panel window variation, not "
          "independent-document or domain-population uncertainty.\n")
        for r in doc["rows"]:
            for scope, entry in r["scopes"].items():
                A("### %s / %s: scientific qualifications\n" % (r["label"], scope))
                L.extend(render_disclosures(disclosures(entry)))
                A("")
        A("## Descriptive FP8 minus BF16 contrast (not causal attribution)\n")
        A("| scope | FP8 | BF16 context | difference |")
        A("|---|---:|---:|---:|")
        contrast = doc["descriptive_fp8_bf16_contrast"]
        for k in ("panel25", "clean17"):
            a = contrast["by_scope"][k]
            A("| `%s` | %.9f | %.9f | %.9f |"
              % (k, a["fp8"], a["same_lane_floor"], a["difference_nats"]))
        A("")
        A(contrast["reading"] + "\n")
        A(contrast["what_cannot_be_recomputed"] + "\n")
        if "per_domain_clean17" in doc:
            pdm = doc["per_domain_clean17"]
            A("## Per-domain, clean scope\n")
            A("| domain | windows | K6 | K8 | FP8 | BF16 context |")
            A("|---|---:|---:|---:|---:|---:|")
            for e in pdm["table"]:
                A("| %s | %d | %.9f | %.9f | %.9f | %.9f |"
                  % (e["domain"], e["windows"], e["k6-sealed"], e["k8-streaming"],
                     e["fp8-crossstack"], e["bf16-floor-crossstack"]))
            A("")
            A(pdm["reading"] + "\n")
        A("## Paired comparisons\n")
        A(doc["paired"]["method"] + "\n")
        for c in doc["paired"]["comparisons"]:
            for scope in ("panel25", "clean17"):
                v = c["scopes"].get(scope)
                if not v:
                    continue
                A("### %s / %s\n" % (c["comparison"], scope))
                A(v["interpretation"] + "\n")
                A("A %.9f; B %.9f; descriptive A-B %+.9f nats.\n"
                  % (v["mean_a"], v["mean_b"], v["mean_diff"]))
                # Render the complete qualification block, not a handpicked p or
                # interval stripped of its weighting and sampling assumptions.
                L.extend(render_disclosures(disclosures(v)))
                A("")
        A("")
        A("## Credit\n")
        A(doc["provenance"]["credit"] + "\n")
        markdown = "\n".join(L) + "\n"
    # Finish both serializations before publishing either file.
    encoded = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_text(args.out, encoded)
    print("wrote %s" % args.out)
    if args.markdown:
        atomic_text(args.markdown, markdown)
        print("wrote %s" % args.markdown)
    print("  %d series x 2 scopes; %d paired comparisons; %d domains"
          % (len(rows), len(paired), len(table)))
    for r in rows:
        print("  %-46s %.12f -> %.12f  (%+.2f %%)"
              % (r["label"], r["scopes"]["panel25"]["mean_kld_nats"],
                 r["scopes"]["clean17"]["mean_kld_nats"],
                 r["scope_delta"]["relative_pct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
