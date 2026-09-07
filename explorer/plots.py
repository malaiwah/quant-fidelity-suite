"""Deterministic comparable-lane scatter plots for the QFS Explorer.

Read-only views over an :class:`ExplorerRegistry` snapshot. ``build_plot``
turns a comparability group (or a highlighted measurement's group) into a
portable, JSON-compatible payload; ``render_svg``/``render_png``/``render_csv``
turn that payload into standalone image/data bytes; ``plot_links`` builds the
canonical image, live and interactive URL snippets the web layer embeds.

Scientific contracts honoured (see AGENTS.md / WHAT-WE-MEASURE.md):

* x is the recorded serialized size in GiB. Weight files, tensor payload and
  whole-repository counts retain their declared basis; none means VRAM.
  Mixed size bases disable the Pareto guide. Unknown sizes are excluded.
* y is the full-vocabulary mean tokenwise KL in nats, with an adaptive
  nats / millinats / micronats display unit.
* Every original group member is represented; an invalid size basis or
  missing/non-finite KL excludes a point with an explicit, portable reason.
  The original full-key predicate and group context are preserved verbatim;
  false/unknown groups are inspection-only (no Pareto guide, no ranking claim).
* No substitution on bad input: an unknown scale, a conflicting
  measurement/group, or a highlighted measurement from the wrong group raise
  ``ValueError``.
* SVG/PNG are standalone, escaped and bounded; the SVG carries safe HTTPS deep
  links and accessibility ``<title>``/``<desc>`` but no scripts, ``foreignObject``
  or remote font loads.

This module never imports ``app.py``.
"""
from __future__ import annotations

import csv
import io
import json
import math
from urllib.parse import urlencode
from xml.sax.saxutils import escape, quoteattr

from . import links
from .data import _number, _safe_link

__all__ = ["build_plot", "render_svg", "render_png", "render_csv", "plot_links"]

# --- constants ---------------------------------------------------------------

_GIB = 1024 ** 3
_SCALES = ("auto", "linear", "symlog")
# Recorded serialized-byte bases; never infer missing sizes from nominal bits.
_SIZE_BASES = ("repo_weight_files", "tensor_payload", "repo_all_files")
_MAX_POINTS = 48
_DEFAULT_W = 840
_DEFAULT_H = 580
_MIN_W, _MAX_W = 360, 1200
_MIN_H, _MAX_H = 260, 900
_PARETO_MIN = 3  # minimum comparable quant points before a Pareto guide is allowed
_SYMLOG_RATIO = 1e4  # auto picks symlog only when positive KL spans >= 4 decades

# Accessible, deterministic palette (shape carries meaning, not colour alone).
_C_QUANT = "#1d4ed8"      # blue   -> quantized candidate (circle)
_C_CONTROL = "#047857"    # green  -> native control / floor (diamond)
_C_HIGHLIGHT = "#b91c1c"  # red    -> highlighted measurement (ring)
_C_PARETO = "#6b7280"     # gray   -> observed Pareto guide (dashed)
_C_AXIS = "#374151"
_C_GRID = "#e5e7eb"
_C_TEXT = "#111827"
_C_MUTED = "#6b7280"
_C_BG = "#ffffff"

_FONT_STACK = "Arial, 'Helvetica Neue', Helvetica, sans-serif"
_DEJAVU_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
)


# --- small format helpers ---------------------------------------------------

def _fmt(value):
    """Compact, deterministic number formatting for axes and labels."""
    if value is None:
        return ""
    try:
        return "%g" % (value,)
    except (TypeError, ValueError):
        return str(value)


def _truncate(text, max_chars):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)].rstrip() + "\u2026"


def _wrap(text, width):
    """Greedy word wrap; returns a list of lines no longer than ``width`` chars."""
    text = "" if text is None else str(text)
    if not text:
        return [""]
    out = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        cur = ""
        for word in words:
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= width:
                cur += " " + word
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return out or [""]


def _nice_step(span, count):
    if span <= 0:
        return 1.0
    raw = span / max(1, count)
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10):
        if mult * mag >= raw * 0.999999:
            return mult * mag
    return 10 * mag


def _axis_ticks_linear(lo, hi, count=5):
    if hi <= lo:
        return [float(lo)], [_fmt(lo)]
    step = _nice_step(hi - lo, count)
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    k = 0
    while True:
        v = start + k * step
        if v > hi + step * 1e-6:
            break
        if v >= lo - step * 1e-6:
            ticks.append(round(v, 12))
        k += 1
        if k > 1000:
            break
    if not ticks:
        ticks = [float(lo), float(hi)]
    return ticks, [_fmt(t) for t in ticks]


def _symlog_map(value, linthresh):
    """Map a non-negative value to symlog coordinates (0 -> 0, exact zero kept)."""
    if linthresh is None or linthresh <= 0:
        return float(value)
    if value <= linthresh:
        return value / linthresh
    return 1.0 + math.log10(value / linthresh)


def _symlog_ticks(hi, linthresh):
    ticks = [0.0]
    if linthresh and linthresh > 0:
        if hi <= linthresh:
            # Everything lives in the linear region; use plain linear ticks.
            return _axis_ticks_linear(0.0, hi, 4)
        if linthresh != 0.0:
            ticks.append(float(linthresh))
        v = linthresh * 10
        guard = 0
        while v <= hi * 1.0000001 and guard < 60:
            ticks.append(round(v, 12))
            v *= 10
            guard += 1
    return ticks, [_fmt(t) for t in ticks]


def _pareto_frontier(points):
    """Non-dominated quant points (smaller size and lower KL are both better).

    Ties are kept once (first occurrence) so duplicates do not both vanish.
    Returned sorted by ascending size, then KL.
    """
    frontier = []
    n = len(points)
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            qs, qk = q["size_gib"], q["kl_display"]
            ps, pk = p["size_gib"], p["kl_display"]
            if qs <= ps + 1e-15 and qk <= pk + 1e-15:
                strictly = qs < ps - 1e-15 or qk < pk - 1e-15
                equal = abs(qs - ps) <= 1e-15 and abs(qk - pk) <= 1e-15
                if strictly or (equal and j < i):
                    dominated = True
                    break
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda p: (p["size_gib"], p["kl_display"], p["measurement_id"]))
    return frontier


def _dataset_viewer_url(measurement_id):
    if not links.MEASUREMENT.fullmatch(measurement_id or ""):
        return None
    return "https://huggingface.co/datasets/%s/viewer/measurements/train?%s" % (
        links.REGISTRY, urlencode({"q": measurement_id}))


def _immutable_records_url(revision):
    if not (isinstance(revision, str) and links.SHA.fullmatch(revision)):
        return None
    return "https://huggingface.co/datasets/%s/resolve/%s/data/measurements.jsonl" % (
        links.REGISTRY, revision)


# --- payload construction ---------------------------------------------------

def build_plot(registry, *, measurement_id=None, group_id=None, scale="auto"):
    """Build a portable, JSON-compatible plot payload from a registry snapshot.

    Exactly one of ``measurement_id`` / ``group_id`` selects the comparability
    lane. A measurement may be supplied together with its own group to mark it
    as highlighted; a measurement from a *different* group is rejected. The
    payload records the actually resolved revision, every group member (plotted
    or excluded with a reason), the original predicate, and fixed source links.
    """
    if scale not in _SCALES:
        raise ValueError("Unknown plot scale %r; choose 'auto', 'linear' or 'symlog'." % (scale,))
    if measurement_id is None and group_id is None:
        raise ValueError("Specify exactly one of measurement_id or group_id for the plot.")

    highlight_mid = None
    if measurement_id is not None:
        if not isinstance(measurement_id, str):
            raise ValueError("measurement_id must be a published measurement id string.")
        # detail() raises ValueError for an unknown id -> we never substitute.
        detail = registry.detail(measurement_id)
        resolved_group = detail["group_id"]
        if group_id is not None and group_id != resolved_group:
            raise ValueError(
                "The highlighted measurement belongs to a different comparability group; "
                "refusing to plot the wrong group.")
        group_id = resolved_group
        highlight_mid = measurement_id
    else:
        if not isinstance(group_id, str):
            raise ValueError("group_id must be a comparability group id string.")

    group = registry.group(group_id)  # raises ValueError for an unknown group
    context = group.get("context") or {}
    rows = group.get("rows") or []
    overview = registry.overview()
    revision = overview.get("revision")
    snapshot_id = overview.get("snapshot")
    origin = overview.get("origin")
    notes = list(overview.get("notes") or [])
    control_ids = set(context.get("control_measurement_ids") or [])
    status = group.get("status") or "unknown"
    ranking_allowed = bool(context.get("ranking_allowed")) and status == "true"
    # A bundled snapshot (revision is None) may be ahead of or behind published
    # truth; it is never a certified ranking surface. Keep the original predicate
    # but render inspection-only and suppress the Pareto guide. This never alters
    # the registry's full-key certification; it only withholds a ranking claim.
    group_reasons = list(group.get("reasons") or [])
    if revision is None:
        ranking_allowed = False
        group_reasons.append(
            "Ranking disabled: this is a bundled snapshot without a pinned public "
            "registry revision; it may be ahead of or behind published truth. The "
            "plot is inspection-only and the original comparability predicate is "
            "preserved unchanged.")

    panel_id = panel_name = ref_id = ref_name = model_name = None
    points = []
    exclusions = []
    members_total = len(rows)

    for row in rows:
        mid = row["id"]
        d = registry.detail(mid)
        artifact = d.get("artifact") or {}
        weights = artifact.get("weights") or {}
        hf = artifact.get("huggingface") or {}
        availability = artifact.get("availability") or {}
        name = artifact.get("name") or row.get("artifact") or mid
        classification = row.get("classification") or "unknown"
        is_control = mid in control_ids
        is_highlight = mid == highlight_mid

        if panel_name is None:
            panel = d.get("panel") or {}
            reference = d.get("reference") or {}
            model = d.get("model") or {}
            panel_id, panel_name = panel.get("id"), panel.get("name")
            ref_id, ref_name = reference.get("id"), reference.get("name")
            model_name = model.get("name")

        size_bytes = _number(weights.get("size_bytes"))
        size_basis = weights.get("size_basis")
        kl_nats = row.get("kl")  # _row() already finite-checks and limits to tokenwise KLD
        art_revision = row.get("revision") or hf.get("revision")
        source_url = _safe_link(hf.get("url") or availability.get("uri"))

        size_reason = None
        size_gib = None
        if size_bytes is None:
            size_reason = "serialized size not recorded"
        elif size_bytes <= 0:
            size_reason = "serialized size is not positive (%s bytes)" % _fmt(size_bytes)
        elif size_basis not in _SIZE_BASES:
            size_reason = "serialized size basis is unknown or unsupported: %r" % (size_basis,)
        else:
            size_gib = size_bytes / _GIB

        kl_reason = None
        if kl_nats is None:
            kl_reason = "mean tokenwise KL is missing or non-finite"
        elif kl_nats < 0:
            kl_reason = "mean tokenwise KL is negative (%s nats); non-physical value excluded" % _fmt(kl_nats)
        label = name
        for redundant in (model_name, (hf.get("repository") or "").split("/")[0]):
            if redundant:
                label = label.replace(redundant, "").strip(" -")
        label = " ".join(label.split()) or name

        common = {
            "measurement_id": mid,
            "artifact": row.get("artifact") or artifact.get("name") or mid,
            "name": name,
            "label": label,
            "classification": classification,
            "is_control": is_control,
            "is_highlight": is_highlight,
            "revision": art_revision,
            "size_bytes": size_bytes,
            "size_basis": size_basis,
            "size_gib": size_gib,
            "kl_nats": kl_nats,
            "source_url": source_url,
            "dataset_viewer_url": _dataset_viewer_url(mid),
        }

        if size_reason or kl_reason:
            reasons = [r for r in (size_reason, kl_reason) if r]
            excl = dict(common)
            excl["plotted"] = False
            excl["kl_display"] = None
            excl["exclusion_reason"] = "; ".join(reasons)
            exclusions.append(excl)
            continue

        common["plotted"] = True
        common["exclusion_reason"] = None
        points.append(common)

    # Bound the rendered point count; overflow members stay portable as exclusions.
    plotted = points[:_MAX_POINTS]
    for overflow in points[_MAX_POINTS:]:
        excl = dict(overflow)
        excl["plotted"] = False
        excl["kl_display"] = None
        excl["exclusion_reason"] = "plot capped at %d points" % _MAX_POINTS
        exclusions.append(excl)

    # Adaptive y unit (nats / millinats / micronats) from the plotted KL span.
    kl_values = [p["kl_nats"] for p in plotted]
    max_kl = max(kl_values) if kl_values else 0.0
    if max_kl <= 0:
        y_unit, y_factor = "nats", 1.0
    elif max_kl < 1e-3:
        y_unit, y_factor = "micronats", 1e6
    elif max_kl < 1.0:
        y_unit, y_factor = "millinats", 1e3
    else:
        y_unit, y_factor = "nats", 1.0
    for p in plotted:
        p["kl_display"] = (p["kl_nats"] or 0.0) * y_factor

    display_values = [p["kl_display"] for p in plotted]
    max_display = max(display_values) if display_values else 0.0
    positive = [v for v in display_values if v > 0]
    wide = bool(positive) and max_display > 0 and (max_display / min(positive)) >= _SYMLOG_RATIO

    requested_scale = scale
    if scale == "auto":
        resolved_scale = "symlog" if wide else "linear"
    else:
        resolved_scale = scale

    scale_note = ""
    linthresh = None
    if resolved_scale == "symlog":
        if not positive or max_display <= 0:
            resolved_scale = "linear"
            scale_note = ("symlog requested but the KL data has no positive range; "
                          "shown on a linear axis.")
        else:
            # Threshold at the smallest positive value so exact zero and the
            # tiniest values stay in the linear region (zero maps to zero).
            linthresh = min(positive)
            scale_note = ("Symlog y-axis (display only): values at or below the linear "
                          "threshold are spaced linearly so exact zero stays at zero; "
                          "values above it use a base-10 log scale. The underlying KL "
                          "values are unchanged.")

    sizes = [p["size_gib"] for p in plotted]
    max_size = max(sizes) if sizes else 0.0
    if max_size <= 0:
        max_size = 1.0  # no plottable size; avoid division by zero in renderers

    x_ticks, x_labels = _axis_ticks_linear(0.0, max_size, 5)
    if resolved_scale == "symlog":
        y_ticks, y_labels = _symlog_ticks(max_display, linthresh)
    else:
        y_ticks, y_labels = _axis_ticks_linear(0.0, max_display, 5)

    # Observed Pareto guide: only when ranking is allowed and enough quants exist.
    pareto = {"shown": False, "order": [], "note": ""}
    quant_points = [p for p in plotted if not p["is_control"]]
    size_bases = sorted({p["size_basis"] for p in plotted})
    if len(size_bases) > 1:
        ranking_allowed = False
        group_reasons.append("Mixed serialized-size bases: inspection only, no size–KL ranking guide.")
    if ranking_allowed and len(quant_points) >= _PARETO_MIN:
        frontier = _pareto_frontier(quant_points)
        if len(frontier) >= 2:
            pareto = {
                "shown": True,
                "order": [p["measurement_id"] for p in frontier],
                "note": ("Dashed guide connects observed Pareto-optimal quant points "
                         "(smaller size and lower KL are both better). It is not "
                         "interpolation and not a claim of general model quality."),
            }

    highlight = None
    if highlight_mid:
        on_plot = next((p for p in plotted if p["measurement_id"] == highlight_mid), None)
        if on_plot:
            highlight = {"measurement_id": highlight_mid, "name": on_plot["name"],
                         "excluded": False, "exclusion_reason": None}
        else:
            ex = next((e for e in exclusions if e["measurement_id"] == highlight_mid), None)
            highlight = {
                "measurement_id": highlight_mid,
                "name": (ex or {}).get("name") or highlight_mid,
                "excluded": True,
                "exclusion_reason": ex["exclusion_reason"] if ex else "not a member of this group",
            }

    title = (model_name or "QFS measurement") + " — size vs KL"
    lane = context.get("lane")
    subtitle_parts = []
    if lane:
        subtitle_parts.append("Lane: %s" % lane)
    if panel_name:
        subtitle_parts.append("Panel: %s" % panel_name)
    if ref_name:
        subtitle_parts.append("Reference: %s" % ref_name)
    subtitle = " \u00b7 ".join(subtitle_parts)

    caution_parts = [
        "Lower KL is closer to the reference, only within this comparability group.",
        "x is recorded serialized size (GiB), not VRAM. Size basis is retained per point; "
        "unknown sizes and missing/non-finite KL are excluded with reasons.",
    ]
    if status == "true" and ranking_allowed:
        caution_parts.append("Comparability is certified for this group; the Pareto guide "
                             "(if shown) lists observed optima only, not a general quality ranking.")
    else:
        caution_parts.append("Group comparability is %s; this plot is inspection-only and "
                             "not a ranking." % status)
    if revision is None:
        caution_parts.append("This is a bundled snapshot without a pinned registry revision; "
                             "it is inspection-only and not pinned to an immutable SHA.")
    if pareto["shown"]:
        caution_parts.append(pareto["note"])
    if scale_note:
        caution_parts.append(scale_note)
    caution = " ".join(caution_parts)

    links_block = {
        "registry_dataset": "https://huggingface.co/datasets/%s" % links.REGISTRY,
        "immutable_registry_records": _immutable_records_url(revision),
        "dataset_viewer_measurements": "https://huggingface.co/datasets/%s/viewer/measurements/train" % links.REGISTRY,
        "viewer_note": ("The dataset viewer searches live data; only the Explorer and raw "
                        "records above pin this registry snapshot."),
    }

    payload = {
        "schema": "qfs-explorer-plots/v1",
        "kind": "scatter",
        "title": title,
        "subtitle": subtitle,
        "caution": caution,
        "scale_requested": requested_scale,
        "scale": resolved_scale,
        "scale_note": scale_note,
        "y_unit": y_unit,
        "y_unit_factor": y_factor,
        "registry_revision": revision,
        "registry_snapshot": snapshot_id,
        "registry_origin": origin,
        "registry_notes": notes,
        "highlight": highlight,
        "group": {
            "id": group_id,
            "title": title,
            "status": status,
            "lane": lane,
            "key": context.get("key"),
            "ranking_allowed": ranking_allowed,
            "ordering": context.get("ordering"),
            "predicate": context.get("original_predicate"),
            "predicate_scope": context.get("predicate_scope"),
            "reasons": group_reasons,
            "caveats": list(context.get("caveats") or []),
            "rule": context.get("rule"),
            "snapshot_notes": list(context.get("snapshot_notes") or []),
            "key_inputs": context.get("key_inputs"),
            "full_key_member_count": context.get("full_key_member_count"),
            "displayed_member_count": context.get("displayed_member_count"),
            "control_measurement_ids": sorted(control_ids),
            "member_count_total": members_total,
            "member_count_plotted": len(plotted),
            "member_count_excluded": len(exclusions),
        },
        "panel": {"id": panel_id, "name": panel_name},
        "reference": {"id": ref_id, "name": ref_name},
        "model": {"name": model_name},
        "axes": {
            "x": {"min": 0.0, "max": max_size, "label": "Serialized size (GiB; basis disclosed)",
                  "scale": "linear", "ticks": x_ticks, "tick_labels": x_labels},
            "y": {"min": 0.0, "max": max_display, "label": "Mean KL (%s)" % y_unit,
                  "scale": resolved_scale, "linthresh": linthresh,
                  "ticks": y_ticks, "tick_labels": y_labels},
        },
        "points": plotted,
        "exclusions": exclusions,
        "pareto": pareto,
        "links": links_block,
        "render": {"width": _DEFAULT_W, "height": _DEFAULT_H, "max_points": _MAX_POINTS},
    }
    return payload


# --- shared geometry / footer ----------------------------------------------

def _layout(w, h):
    w = int(max(_MIN_W, min(_MAX_W, w)))
    h = int(max(_MIN_H, min(_MAX_H, h)))
    left, right, top, bottom = 74, 28, 86, 120
    plot_w = max(40, w - left - right)
    plot_h = max(40, h - top - bottom)
    return {
        "W": w, "H": h,
        "left": left, "right": right, "top": top, "bottom": bottom,
        "plot_left": left, "plot_top": top,
        "plot_right": left + plot_w, "plot_bottom": top + plot_h,
        "plot_w": plot_w, "plot_h": plot_h,
    }


def _point_pixel(point, axes, geo):
    max_x = axes["x"]["max"] or 1.0
    px = geo["plot_left"] + (point["size_gib"] / max_x) * geo["plot_w"]
    d = point["kl_display"] or 0.0
    axy = axes["y"]
    if axy["scale"] == "symlog" and axy.get("linthresh"):
        m = _symlog_map(d, axy["linthresh"])
        mmax = _symlog_map(axy["max"] or 1.0, axy["linthresh"]) or 1.0
        frac = m / mmax if mmax else 0.0
    else:
        frac = d / (axy["max"] or 1.0) if (axy["max"] or 0) > 0 else 0.0
    py = geo["plot_top"] + (1.0 - frac) * geo["plot_h"]
    return px, py


def _footer_lines(payload):
    g = payload["group"]
    lines = []
    if g["status"] == "true" and g["ranking_allowed"]:
        lines.append("Comparability certified \u2014 Pareto guide (if shown) lists observed optima only.")
    else:
        lines.append("Comparability %s \u2014 inspection only, not a ranking." % g["status"])
    lines.append("x = recorded serialized GiB, not VRAM. Size bases and exclusions are in the CSV.")
    if payload["scale"] == "symlog":
        lines.append("y-axis is symlog: linear near zero (exact zero preserved), log above the threshold.")
    if g["member_count_excluded"]:
        lines.append("%d of %d members excluded (invalid size basis, missing/non-finite KL, or plot cap)."
                     % (g["member_count_excluded"], g["member_count_total"]))
    return lines


def _revision_label(payload):
    rev = payload.get("registry_revision")
    if isinstance(rev, str) and links.SHA.fullmatch(rev):
        return "Registry revision: %s" % rev
    snap = payload.get("registry_snapshot") or "bundled"
    return "Registry snapshot: %s (not a pinned public SHA)" % snap


# --- SVG renderer -----------------------------------------------------------

def _svg_text(x, y, text, size, fill, anchor="start", weight="normal", halo=False, cls=None):
    cls_attr = (' class="%s"' % cls) if cls else ""
    halo_attr = ' stroke="%s" stroke-width="3" paint-order="stroke"' % _C_BG if halo else ""
    return ('<text x="%s" y="%s" font-size="%d" font-family="%s" fill="%s" font-weight="%s" '
            'text-anchor="%s"%s%s>%s</text>' % (
                _fmt(x), _fmt(y), size, _FONT_STACK, fill, weight, anchor, halo_attr, cls_attr,
                escape(text)))


def render_svg(payload, *, base=None):
    """Render a standalone, escaped SVG scatter plot from a plot payload.

    No scripts, ``foreignObject`` or remote font loads; safe HTTPS deep links
    are attached to points that have one. Includes accessibility title/desc.
    """
    if not isinstance(payload, dict):
        raise ValueError("render_svg expects a build_plot payload dict.")
    geo = _layout(payload.get("render", {}).get("width", _DEFAULT_W),
                  payload.get("render", {}).get("height", _DEFAULT_H))
    axes = payload.get("axes") or {}
    points = payload.get("points") or []
    pareto = payload.get("pareto") or {}
    highlight = payload.get("highlight") or {}
    title = payload.get("title") or ""
    subtitle = payload.get("subtitle") or ""
    y_unit = payload.get("y_unit") or "nats"

    W, H = geo["W"], geo["H"]
    out = []
    out.append(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" width="%d" height="%d" '
        'viewBox="0 0 %d %d" font-family="%s" role="img" aria-labelledby="plottitle plotdesc">'
        % (W, H, W, H, _FONT_STACK))
    out.append('<title id="plottitle">%s</title>' % escape(_truncate(title, 160)))
    desc = ("QFS comparable-lane scatter plot. x: recorded serialized size in GiB, basis disclosed. "
            "y: full-vocabulary mean tokenwise KL in %s. %s"
            % (y_unit, escape(payload.get("caution") or "")))
    out.append('<desc id="plotdesc">%s</desc>' % desc)

    # background
    out.append('<rect x="0" y="0" width="%d" height="%d" fill="%s"/>' % (W, H, _C_BG))

    # title / subtitle / highlight
    out.append(_svg_text(geo["left"], 26, _truncate(title, 42), 17, _C_TEXT, weight="bold"))
    if subtitle:
        out.append(_svg_text(geo["left"], 48, _truncate(subtitle, 100), 12, _C_MUTED))
    if highlight.get("name"):
        out.append(_svg_text(geo["plot_right"], 26, "Highlighted: %s" % _truncate(highlight["name"], 30),
                             12, _C_HIGHLIGHT, anchor="end", weight="bold"))
        if highlight.get("excluded"):
            out.append(_svg_text(geo["plot_right"], 46, "(excluded: %s)" % _truncate(highlight.get("exclusion_reason"), 60),
                                 10, _C_HIGHLIGHT, anchor="end"))

    # legend (horizontal row)
    legend_y = 68
    lx = geo["left"]
    items = [("\u25cf", _C_QUANT, "Quantized"), ("\u25c6", _C_CONTROL, "Native control")]
    if highlight:
        items.append(("\u25cf", _C_HIGHLIGHT, "Highlighted"))
    for glyph, color, label in items:
        out.append('<text x="%d" y="%d" font-size="13" fill="%s">%s</text>'
                   % (lx, legend_y, color, escape(glyph)))
        lx += 16
        out.append(_svg_text(lx, legend_y, label, 11, _C_TEXT))
        lx += 8 + 7 * len(label)
    if pareto.get("shown"):
        seg_x = lx + 6
        out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5" stroke-dasharray="5 4"/>'
                   % (seg_x, legend_y - 3, seg_x + 26, legend_y - 3, _C_PARETO))
        out.append(_svg_text(seg_x + 32, legend_y, "Observed Pareto guide", 11, _C_TEXT))

    # gridlines + axes
    pl, pt = geo["plot_left"], geo["plot_top"]
    pr, pb = geo["plot_right"], geo["plot_bottom"]
    out.append('<rect x="%s" y="%s" width="%s" height="%s" fill="none" stroke="%s" stroke-width="1"/>'
               % (_fmt(pl), _fmt(pt), _fmt(pr - pl), _fmt(pb - pt), _C_AXIS))

    xticks = axes.get("x", {}).get("ticks") or []
    xlabels = axes.get("x", {}).get("tick_labels") or []
    for t, lab in zip(xticks, xlabels):
        if axes["x"]["max"] <= 0:
            continue
        xpx = pl + (t / axes["x"]["max"]) * geo["plot_w"]
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                   % (_fmt(xpx), _fmt(pt), _fmt(xpx), _fmt(pb), _C_GRID))
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                   % (_fmt(xpx), _fmt(pb), _fmt(xpx), _fmt(pb + 4), _C_AXIS))
        out.append(_svg_text(xpx, pb + 16, lab, 11, _C_MUTED, anchor="middle"))
    out.append(_svg_text((pl + pr) / 2.0, pb + 36, axes.get("x", {}).get("label", ""), 12, _C_TEXT, anchor="middle"))

    axy = axes.get("y", {})
    yticks = axy.get("ticks") or []
    ylabels = axy.get("tick_labels") or []
    for t, lab in zip(yticks, ylabels):
        frac = 0.0
        if axy.get("scale") == "symlog" and axy.get("linthresh"):
            mmax = _symlog_map(axy.get("max") or 1.0, axy["linthresh"]) or 1.0
            frac = _symlog_map(t, axy["linthresh"]) / mmax if mmax else 0.0
        elif (axy.get("max") or 0) > 0:
            frac = t / axy["max"]
        ypx = pt + (1.0 - frac) * geo["plot_h"]
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                   % (_fmt(pl), _fmt(ypx), _fmt(pr), _fmt(ypx), _C_GRID))
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1"/>'
                   % (_fmt(pl - 4), _fmt(ypx), _fmt(pl), _fmt(ypx), _C_AXIS))
        out.append(_svg_text(pl - 8, ypx + 4, lab, 11, _C_MUTED, anchor="end"))
    y_label = axy.get("label", "")
    out.append('<text x="%d" y="%d" font-size="12" font-family="%s" fill="%s" text-anchor="end" '
               'transform="rotate(-90 %d %d)">%s</text>'
               % (16, (pt + pb) / 2, _FONT_STACK, _C_TEXT, 16, (pt + pb) / 2, escape(y_label)))

    # observed Pareto guide (under points)
    if pareto.get("shown") and len(pareto.get("order") or []) >= 2:
        by_id = {p["measurement_id"]: p for p in points}
        coords = []
        for mid in pareto["order"]:
            p = by_id.get(mid)
            if p:
                coords.append(_point_pixel(p, axes, geo))
        if len(coords) >= 2:
            pts_str = " ".join("%s,%s" % (_fmt(x), _fmt(y)) for x, y in coords)
            out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.5" stroke-dasharray="5 4">'
                       '<title>%s</title></polyline>' % (pts_str, _C_PARETO, escape(pareto.get("note") or "")))

    # points
    for p in points:
        px, py = _point_pixel(p, axes, geo)
        px = min(max(px, pl), pr)
        py = min(max(py, pt), pb)
        is_ctrl = p.get("is_control")
        fill = _C_CONTROL if is_ctrl else _C_QUANT
        tip = ("%s \u2014 %s%s\nsize %s GiB \u00b7 KL %s nats (%s)\n%s"
               % (p.get("name"), p.get("classification"),
                  " \u00b7 native control" if is_ctrl else "",
                  _fmt(p.get("size_gib")), _fmt(p.get("kl_nats")), y_unit,
                  p.get("exclusion_reason") or "plotted"))
        tip += " · size basis: " + str(p.get("size_basis"))
        link = (plot_links(base, payload.get("registry_revision"),
                           measurement_id=p["measurement_id"],
                           scale=payload.get("scale_requested", "auto"))["interactive"]
                if base else p.get("source_url") or p.get("dataset_viewer_url"))
        open_a = ""
        close_a = ""
        if link:
            open_a = '<a xlink:href=%s target="_blank" rel="noopener">' % quoteattr(link)
            close_a = "</a>"
        inner = ["<title>%s</title>" % escape(tip)]
        if p.get("is_highlight"):
            inner.append('<circle cx="%s" cy="%s" r="9" fill="none" stroke="%s" stroke-width="2"/>'
                         % (_fmt(px), _fmt(py), _C_HIGHLIGHT))
        if is_ctrl:
            half = 6
            diamond = "%s,%s %s,%s %s,%s %s,%s" % (
                _fmt(px), _fmt(py - half), _fmt(px + half), _fmt(py),
                _fmt(px), _fmt(py + half), _fmt(px - half), _fmt(py))
            inner.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1"/>'
                         % (diamond, fill, _C_BG))
        else:
            r = 5
            inner.append('<circle cx="%s" cy="%s" r="%s" fill="%s" stroke="%s" stroke-width="1"/>'
                         % (_fmt(px), _fmt(py), r, fill, _C_BG))
        out.append(open_a + "<g>" + "".join(inner) + "</g>" + close_a)

        # label with white halo for readability over gridlines
        if px > pr - geo["plot_w"] * 0.32:
            out.append(_svg_text(px - 9, py + 4, _truncate(p.get("label") or p.get("name"), 22), 11, _C_TEXT,
                                 anchor="end", halo=True))
        else:
            out.append(_svg_text(px + 9, py + 4, _truncate(p.get("label") or p.get("name"), 22), 11, _C_TEXT, halo=True))

    if not points:
        out.append(_svg_text((pl + pr) / 2.0, (pt + pb) / 2.0,
                             "No plottable points \u2014 every member was excluded (see CSV).",
                             13, _C_MUTED, anchor="middle"))

    # footer: revision + concise caution
    fy = pb + 56
    out.append(_svg_text(geo["left"], fy, _revision_label(payload), 11, _C_MUTED))
    for line in _footer_lines(payload):
        fy += 16
        out.append(_svg_text(geo["left"], fy, line, 10, _C_MUTED))

    out.append("</svg>")
    return "".join(out).encode("utf-8")


# --- PNG renderer (Pillow) --------------------------------------------------

def _pil_font(size):
    from PIL import ImageFont
    for path in _DEJAVU_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.2 accepts size
    except TypeError:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _text_width(draw, text, font):
    try:
        return draw.textlength(text, font=font)
    except Exception:
        pass
    try:
        return font.getlength(text)
    except Exception:
        pass
    try:
        return font.getbbox(text)[2]
    except Exception:
        return len(text) * 7


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _dashed_line(draw, p1, p2, fill, width=1, dash=5, gap=4):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist <= 0:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    covered = 0.0
    while covered < dist:
        seg_end = min(covered + dash, dist)
        draw.line([(x1 + dx * covered, y1 + dy * covered),
                   (x1 + dx * seg_end, y1 + dy * seg_end)], fill=fill, width=width)
        covered = seg_end + gap


def render_png(payload):
    """Render a bounded PNG scatter plot from a plot payload using Pillow."""
    if not isinstance(payload, dict):
        raise ValueError("render_png expects a build_plot payload dict.")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - dependency supplied by the web layer
        raise ValueError("PNG rendering requires Pillow, which is not available: %s" % exc) from None

    geo = _layout(payload.get("render", {}).get("width", _DEFAULT_W),
                  payload.get("render", {}).get("height", _DEFAULT_H))
    axes = payload.get("axes") or {}
    points = payload.get("points") or []
    pareto = payload.get("pareto") or {}
    highlight = payload.get("highlight") or {}
    title = payload.get("title") or ""
    subtitle = payload.get("subtitle") or ""
    y_unit = payload.get("y_unit") or "nats"

    W, H = geo["W"], geo["H"]
    img = Image.new("RGB", (W, H), _rgb(_C_BG))
    draw = ImageDraw.Draw(img)
    f_title = _pil_font(17)
    f_sub = _pil_font(12)
    f_axis = _pil_font(11)
    f_label = _pil_font(11)
    f_small = _pil_font(10)

    def text(x, y, s, font, fill, anchor="la"):
        draw.text((x, y), s, font=font, fill=_rgb(fill))

    pl, pt = geo["plot_left"], geo["plot_top"]
    pr, pb = geo["plot_right"], geo["plot_bottom"]

    # title / subtitle / highlight
    text(pl, 8, _truncate(title, 42), f_title, _C_TEXT)
    if subtitle:
        text(pl, 30, _truncate(subtitle, 100), f_sub, _C_MUTED)
    if highlight.get("name"):
        hlabel = "Highlighted: %s" % _truncate(highlight["name"], 30)
        tw = _text_width(draw, hlabel, f_sub)
        text(pr - tw, 8, hlabel, f_sub, _C_HIGHLIGHT)
        if highlight.get("excluded"):
            elabel = "(excluded: %s)" % _truncate(highlight.get("exclusion_reason"), 56)
            tw2 = _text_width(draw, elabel, f_small)
            text(pr - tw2, 30, elabel, f_small, _C_HIGHLIGHT)

    # legend
    lx = pl
    for glyph, color, label in [("\u25cf", _C_QUANT, "Quantized"),
                                 ("\u25c6", _C_CONTROL, "Native control")] + (
                                [("\u25cf", _C_HIGHLIGHT, "Highlighted")] if highlight else []):
        draw.text((lx, 50), glyph, font=f_label, fill=_rgb(color))
        lx += 12
        text(lx, 50, label, f_label, _C_TEXT)
        lx += _text_width(draw, label, f_label) + 16
    if pareto.get("shown"):
        draw.line([(lx, 57), (lx + 26, 57)], fill=_rgb(_C_PARETO), width=1)
        # crude dash overlay
        _dashed_line(draw, (lx, 57), (lx + 26, 57), _rgb(_C_PARETO), width=1, dash=5, gap=4)
        lx += 32
        text(lx, 50, "Observed Pareto guide", f_label, _C_TEXT)

    # plot frame + gridlines + ticks
    draw.rectangle([pl, pt, pr, pb], outline=_rgb(_C_AXIS), width=1)
    xticks = axes.get("x", {}).get("ticks") or []
    xlabels = axes.get("x", {}).get("tick_labels") or []
    xmax = axes.get("x", {}).get("max") or 1.0
    for t, lab in zip(xticks, xlabels):
        if xmax <= 0:
            continue
        xpx = pl + (t / xmax) * geo["plot_w"]
        draw.line([(xpx, pt), (xpx, pb)], fill=_rgb(_C_GRID), width=1)
        draw.line([(xpx, pb), (xpx, pb + 4)], fill=_rgb(_C_AXIS), width=1)
        tw = _text_width(draw, lab, f_axis)
        text(xpx - tw / 2, pb + 6, lab, f_axis, _C_MUTED)
    xlabel = axes.get("x", {}).get("label", "")
    tw = _text_width(draw, xlabel, f_sub)
    text((pl + pr) / 2 - tw / 2, pb + 24, xlabel, f_sub, _C_TEXT)

    axy = axes.get("y", {})
    yticks = axy.get("ticks") or []
    ylabels = axy.get("tick_labels") or []
    for t, lab in zip(yticks, ylabels):
        frac = 0.0
        if axy.get("scale") == "symlog" and axy.get("linthresh"):
            mmax = _symlog_map(axy.get("max") or 1.0, axy["linthresh"]) or 1.0
            frac = _symlog_map(t, axy["linthresh"]) / mmax if mmax else 0.0
        elif (axy.get("max") or 0) > 0:
            frac = t / axy["max"]
        ypx = pt + (1.0 - frac) * geo["plot_h"]
        draw.line([(pl, ypx), (pr, ypx)], fill=_rgb(_C_GRID), width=1)
        draw.line([(pl - 4, ypx), (pl, ypx)], fill=_rgb(_C_AXIS), width=1)
        tw = _text_width(draw, lab, f_axis)
        text(pl - 8 - tw, ypx - 6, lab, f_axis, _C_MUTED)
    ylabel = axy.get("label", "")
    # rotated y label
    try:
        yl_img = Image.new("RGBA", (max(1, int(_text_width(draw, ylabel, f_sub)) + 4), 14), (0, 0, 0, 0))
        ImageDraw.Draw(yl_img).text((2, 0), ylabel, font=f_sub, fill=_rgb(_C_TEXT))
        yl_img = yl_img.rotate(90, expand=True)
        img.paste(yl_img, (8, int((pt + pb) / 2 - yl_img.size[1] / 2)), yl_img)
        draw = ImageDraw.Draw(img)
    except Exception:
        text(8, (pt + pb) // 2, ylabel, f_small, _C_TEXT)

    # Pareto guide
    if pareto.get("shown") and len(pareto.get("order") or []) >= 2:
        by_id = {p["measurement_id"]: p for p in points}
        coords = []
        for mid in pareto["order"]:
            p = by_id.get(mid)
            if p:
                px, py = _point_pixel(p, axes, geo)
                coords.append((min(max(px, pl), pr), min(max(py, pt), pb)))
        for i in range(len(coords) - 1):
            _dashed_line(draw, coords[i], coords[i + 1], _rgb(_C_PARETO), width=1, dash=5, gap=4)

    # points
    for p in points:
        px, py = _point_pixel(p, axes, geo)
        px = min(max(px, pl), pr)
        py = min(max(py, pt), pb)
        is_ctrl = p.get("is_control")
        fill = _rgb(_C_CONTROL if is_ctrl else _C_QUANT)
        if p.get("is_highlight"):
            draw.ellipse([px - 9, py - 9, px + 9, py + 9], outline=_rgb(_C_HIGHLIGHT), width=2)
        if is_ctrl:
            half = 6
            draw.polygon([(px, py - half), (px + half, py), (px, py + half), (px - half, py)],
                         fill=fill, outline=_rgb(_C_BG))
        else:
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=fill, outline=_rgb(_C_BG))
        nm = _truncate(p.get("label") or p.get("name"), 22)
        label_x = px - 9 - _text_width(draw, nm, f_small) if px > pr - geo["plot_w"] * 0.32 else px + 9
        # Keep right-edge labels inside the figure, matching the SVG placement.
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((label_x + ox, py - 6 + oy), nm, font=f_small, fill=_rgb(_C_BG))
        text(label_x, py - 6, nm, f_small, _C_TEXT)

    if not points:
        msg = "No plottable points - every member was excluded (see CSV)."
        tw = _text_width(draw, msg, f_sub)
        text((pl + pr) / 2 - tw / 2, (pt + pb) / 2 - 6, msg, f_sub, _C_MUTED)

    # footer
    fy = pb + 44
    text(pl, fy, _revision_label(payload), f_axis, _C_MUTED)
    for line in _footer_lines(payload):
        fy += 14
        text(pl, fy, line, f_small, _C_MUTED)

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


# --- CSV renderer -----------------------------------------------------------

def render_csv(payload):
    """Render a portable CSV of plot data and provenance from a plot payload."""
    if not isinstance(payload, dict):
        raise ValueError("render_csv expects a build_plot payload dict.")
    g = payload.get("group") or {}
    panel = payload.get("panel") or {}
    reference = payload.get("reference") or {}
    columns = [
        "measurement_id", "artifact_name", "classification", "is_control", "is_highlight",
        "plotted", "exclusion_reason", "size_bytes", "size_gib", "size_basis",
        "kl_nats", "kl_display", "y_unit", "scale", "scale_note",
        "artifact_revision", "source_url", "dataset_viewer_url",
        "group_id", "group_status", "ranking_allowed",
        "panel_id", "panel_name", "reference_id", "reference_name",
        "registry_revision", "registry_origin",
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)

    def row_for(item, plotted):
        return [
            item.get("measurement_id"), item.get("artifact"), item.get("classification"),
            item.get("is_control"), item.get("is_highlight"), plotted,
            item.get("exclusion_reason") or "",
            item.get("size_bytes") if plotted else "",
            _fmt(item.get("size_gib")) if plotted else "",
            item.get("size_basis") or "",
            _fmt(item.get("kl_nats")) if plotted else "",
            _fmt(item.get("kl_display")) if plotted else "",
            payload.get("y_unit"), payload.get("scale"), payload.get("scale_note") or "",
            item.get("revision") or "",
            item.get("source_url") or "",
            item.get("dataset_viewer_url") or "",
            g.get("id"), g.get("status"), g.get("ranking_allowed"),
            panel.get("id"), panel.get("name"), reference.get("id"), reference.get("name"),
            payload.get("registry_revision") or "",
            payload.get("registry_origin") or "",
        ]

    for p in payload.get("points") or []:
        writer.writerow(row_for(p, True))
    for e in payload.get("exclusions") or []:
        writer.writerow(row_for(e, False))
    return buf.getvalue().encode("utf-8")


# --- link snippets ----------------------------------------------------------

def plot_links(base, registry_revision, *, measurement_id=None, group_id=None, scale="auto"):
    """Build the canonical image, live and interactive URL snippets for a plot.

    Exactly one of ``measurement_id`` / ``group_id`` selects the lane. ``base``
    is validated as a canonical HTTPS hf.space host. ``registry_revision`` must
    be a full 40-character SHA for the immutable image/data endpoints; pass
    ``None`` for a bundled snapshot, in which case immutable endpoints are
    refused (not fabricated) while live aliases and the unpinned interactive
    view remain available.
    """
    if scale not in _SCALES:
        raise ValueError("Unknown plot scale %r; choose 'auto', 'linear' or 'symlog'." % (scale,))
    host = links.explorer_base(base)
    if measurement_id is None and group_id is None:
        raise ValueError("Specify exactly one of measurement_id or group_id for the plot links.")
    if measurement_id is not None and group_id is not None:
        raise ValueError("Specify measurement OR group for plot links, not both.")
    if measurement_id is not None and not links.MEASUREMENT.fullmatch(measurement_id):
        raise ValueError("measurement_id must be a QFS measurement id.")
    if group_id is not None:
        if not isinstance(group_id, str) or len(group_id) > 512:
            raise ValueError("group_id must be a comparability group id string.")
        # group ids are JSON arrays [key, lane]; validate structural shape only.
        try:
            parsed = json.loads(group_id)
        except (ValueError, TypeError):
            raise ValueError("group_id must be a comparability group id string.")
        if not (isinstance(parsed, list) and len(parsed) == 2):
            raise ValueError("group_id must be a comparability group id string.")
    bundled = registry_revision is None
    if not bundled and not (isinstance(registry_revision, str) and links.SHA.fullmatch(registry_revision)):
        raise ValueError("registry_revision must be a full lowercase 40-character commit SHA, or None for a bundled snapshot.")

    selector = {"measurement": measurement_id} if measurement_id is not None else {"group": group_id}
    live_query = dict(selector)
    live_query["scale"] = scale
    live = {
        "svg": host + "/plots/live.svg?" + urlencode(live_query),
        "png": host + "/plots/live.png?" + urlencode(live_query),
        "embed": host + "/plots/embed?" + urlencode({**live_query, "live": "1"}),
        "note": ("Live aliases resolve the current registry revision through the Explorer "
                 "transport with a bounded TTL; they are not pinned to an immutable SHA."),
    }

    if bundled:
        # A bundled snapshot has no pinned SHA, so immutable image/data endpoints
        # (which require one) are refused rather than fabricated. Live aliases and
        # the interactive view (unpinned) remain available.
        immutable = None
        immutable_records = None
        interactive_query = {"tab": "plots"}
        interactive_query.update(selector)
        interactive_query["scale"] = scale
        interactive = host + "/?" + urlencode(interactive_query)
        interactive_note = ("Interactive link opens the Explorer's current snapshot; it is "
                            "not pinned to an immutable registry revision.")
    else:
        immutable_query = dict(selector)
        immutable_query["registry_revision"] = registry_revision
        immutable_query["scale"] = scale
        immutable = {
            "svg": host + "/plots/render.svg?" + urlencode(immutable_query),
            "png": host + "/plots/render.png?" + urlencode(immutable_query),
            "csv": host + "/plots/data.csv?" + urlencode(immutable_query),
            "json": host + "/plots/data.json?" + urlencode(immutable_query),
            "embed": host + "/plots/embed?" + urlencode(immutable_query),
        }
        immutable_records = _immutable_records_url(registry_revision)
        interactive_query = {"tab": "plots"}
        interactive_query.update(selector)
        interactive_query["registry_revision"] = registry_revision
        interactive_query["scale"] = scale
        interactive = host + "/?" + urlencode(interactive_query)
        interactive_note = ""

    return {
        "interactive": interactive,
        "interactive_note": interactive_note,
        "immutable": immutable,
        "immutable_refused_reason": ("Bundled snapshot has no pinned registry SHA; immutable "
                                     "image/data endpoints require one and were not fabricated."
                                     if bundled else ""),
        "live": live,
        "immutable_registry_records": immutable_records,
        "registry_dataset": "https://huggingface.co/datasets/%s" % links.REGISTRY,
    }
