"""Public snapshot plot controls, independent of the application module."""
from __future__ import annotations

import html
import re

import gradio as gr

from . import links, plots


def _md(value):
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", html.escape(str(value), quote=False))


def build_plot_ui(registry_for, snapshot_state, base, initial_revision, initial_group):
    """Create a Plots tab and return outputs plus snapshot-aware view/clear callbacks."""
    def view(revision, group_id, measurement_id=None, scale="auto"):
        current = registry_for(revision)
        if not group_id and not measurement_id:
            return clear("Choose a comparison group to plot.", current.groups(""), scale)
        payload = plots.build_plot(current, group_id=group_id, measurement_id=measurement_id or None, scale=scale)
        gid = payload["group"]["id"]
        rows = current.group(gid)["rows"]
        choices = [(r.get("artifact", r["id"]) + " · " + r["id"], r["id"]) for r in rows]
        urls = plots.plot_links(base, payload["registry_revision"], measurement_id=measurement_id or None,
                                group_id=None if measurement_id else gid, scale=scale)
        immutable = urls["immutable"]
        status = "### " + _md(payload["title"]) + "\n\n" + _md(payload["caution"])
        status += "\n\n**Actual registry pin:** " + _md(payload["registry_revision"] or "Unpinned bundled snapshot — inspection only")
        status += "\n\n**Requested scale:** %s · **Rendered scale:** %s · **Axis unit:** %s. CSV and JSON retain KL in nats." % (scale, payload["scale"], payload["y_unit"])
        status += "\n\n" + "\n".join("- " + _md(n) for n in payload["registry_notes"] + payload["group"]["reasons"] + payload["group"]["caveats"])
        if payload["highlight"] and payload["highlight"]["excluded"]:
            status += "\n\n**Highlighted measurement is excluded:** " + _md(payload["highlight"]["exclusion_reason"])
        downloads = (" · ".join("[%s](%s&download=1)" % (kind.upper(), immutable[kind]) for kind in ("png", "svg", "csv", "json"))
                     if immutable else urls["immutable_refused_reason"])
        navigation = "[Open this plot](%s)" % urls["interactive"]
        if measurement_id and payload["registry_revision"]:
            navigation += " · [Inspect highlighted measurement and receipts](%s)" % links.measurement_url(base, measurement_id, payload["registry_revision"])
        live_markdown = "[![QFS size–KL plot — live cached](%s)](%s)" % (urls["live"]["png"], urls["interactive"])
        snapshot_markdown = "[![QFS size–KL plot — immutable snapshot](%s)](%s)" % (immutable["png"], urls["interactive"]) if immutable else ""
        def iframe(url):
            return '<iframe src="%s" title="QFS size–KL evidence plot" width="100%%" height="680" loading="lazy"></iframe>' % html.escape(url, quote=True)
        members = {p["measurement_id"]: p for p in payload["points"] + payload["exclusions"]}
        table = [[members[r["id"]][k] for k in ("name", "size_gib", "size_basis", "kl_nats", "classification", "is_control", "plotted", "measurement_id")] for r in rows]
        return (gr.Dropdown(choices=current.groups(""), value=gid, render=False), gr.Dropdown(choices=choices, value=measurement_id or None, render=False), scale,
                status, '<div class="qfs-plot">' + plots.render_svg(payload, base=base).decode("utf-8") + '</div>', table,
                payload["exclusions"], payload, downloads, navigation, snapshot_markdown, live_markdown,
                iframe(immutable["embed"]) if immutable else "", iframe(urls["live"]["embed"]))

    def clear(message="No linked plot loaded.", choices=None, scale="auto"):
        return (gr.Dropdown(choices=choices or [], value=None, render=False), gr.Dropdown(choices=[], value=None, render=False), scale,
                message, "", [], [], {}, "", "", "", "", "", "")

    initial = view(initial_revision, initial_group)
    with gr.Tab("Plots", id="plots"):
        gr.Markdown("## Size versus distribution fidelity\nChoose a full panel/reference/lane group, then optionally highlight one measurement. Highlighting never filters away its peers or controls. **Serialized GiB is not VRAM; lower KL is not task accuracy.** Whole-repository, weight-file and payload counts retain their declared basis. Mixed size bases and unknown groups stay inspection-only, without a ranking guide.")
        group = gr.Dropdown(choices=registry_for(initial_revision).groups(""), value=initial_group, label="Comparison group — complete same-lane context")
        with gr.Row():
            measurement = gr.Dropdown(choices=[(r.get("artifact", r["id"]) + " · " + r["id"], r["id"]) for r in registry_for(initial_revision).group(initial_group)["rows"]] if initial_group else [], value=None,
                                      label="Highlight a measurement (clear to show the group)", scale=4)
            scale = gr.Radio(choices=["auto", "linear", "symlog"], value="auto", label="KL display scale", scale=1)
        status = gr.Markdown(initial[3])
        chart = gr.HTML(initial[4])
        gr.Markdown("Click a plotted marker to open that measurement highlighted in the same pinned lane. Hover for size/KL values. Excluded members remain inspectable using the selector above.")
        navigation = gr.Markdown(initial[9])
        downloads = gr.Markdown(initial[8])
        members = gr.Dataframe(value=initial[5], headers=["Artifact", "Serialized GiB", "Size basis", "KL (nats)", "Evidence class", "Native control", "Plotted", "Measurement ID"],
                               interactive=False, label="All group members — no cross-group ranking", wrap=True)
        with gr.Accordion("Excluded members and reasons", open=False):
            exclusions = gr.JSON(value=initial[6])
        with gr.Accordion("Full plot data, predicate, provenance and registry pin", open=False):
            payload = gr.JSON(value=initial[7])
        with gr.Accordion("Share, download and embed", open=False):
            gr.Markdown("**Snapshot** links retain this exact public registry commit. **Live cached** images/iframes resolve the latest public registry, cached for up to five minutes; they can change. A live image's click-through opens this selected snapshot when pinned. Bundled data cannot be exported as a fabricated immutable snapshot. No private Job result appears on these public routes.")
            snapshot_markdown = gr.Code(value=initial[10], language="markdown", label="Immutable snapshot Markdown image", interactive=False)
            live_markdown = gr.Code(value=initial[11], language="markdown", label="Live cached Markdown image", interactive=False)
            snapshot_embed = gr.Code(value=initial[12], language="html", label="Immutable snapshot iframe", interactive=False)
            live_embed = gr.Code(value=initial[13], language="html", label="Live cached iframe", interactive=False)
    outputs = [group, measurement, scale, status, chart, members, exclusions, payload, downloads, navigation,
               snapshot_markdown, live_markdown, snapshot_embed, live_embed]
    group.input(lambda gid, rev, selected_scale: view(rev, gid, scale=selected_scale), [group, snapshot_state, scale], outputs, api_name=False, concurrency_limit=2)
    measurement.input(lambda mid, gid, rev, selected_scale: view(rev, gid, mid, selected_scale), [measurement, group, snapshot_state, scale], outputs, api_name=False, concurrency_limit=2)
    scale.input(lambda selected_scale, gid, mid, rev: view(rev, gid, mid, selected_scale), [scale, group, measurement, snapshot_state], outputs, api_name=False, concurrency_limit=2)
    return {"outputs": outputs, "view": view, "clear": clear, "scale": scale}
