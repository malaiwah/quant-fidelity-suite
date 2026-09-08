"""QFS Explorer: public evidence browsing and explicit caller-funded workflows."""
from __future__ import annotations

import html
from pathlib import Path
import os
import re
from functools import lru_cache
from urllib.parse import quote

import gradio as gr
from starlette.middleware import Middleware

from explorer.data import ExplorerRegistry
from explorer import costs, contribute, links, snippets
from explorer.job_ui import build_jobs_ui
from explorer.plot_ui import build_plot_ui
from explorer.transport import ExplorerTransport

SPACE_ID = os.environ.get("SPACE_ID", "malaiwah/qfs-explorer")
SOURCE_URL = "https://github.com/malaiwah/quant-fidelity-suite"
REGISTRY_URL = "https://huggingface.co/datasets/malaiwah/quant-fidelity-registry"
EXPLORER_BASE = links.explorer_base(os.environ.get("SPACE_HOST"))
CSS = """
.gradio-container {width: 100% !important; min-width: 0 !important; max-width: 1240px !important; box-sizing: border-box; margin: auto;}
#hero {padding: 30px 32px; border-radius: 18px; background: #102b36; color: #f5fafb; margin-bottom: 12px;}
#hero h1 {font-size: 38px; letter-spacing: -1.3px; color: #f5fafb; margin: 6px 0 10px;}
#hero p {color: #c9dce2; max-width: 780px; font-size: 16px; line-height: 1.6;}
.eyebrow {font-size: 11px; letter-spacing: 2px; font-weight: 700; color: #70ddc1;}
#stats {display: flex; gap: 12px; flex-wrap: wrap; margin: 14px 0 24px;}
.stat {flex: 1; min-width: 140px; padding: 16px 20px; border: 1px solid #b8cbd2; border-radius: 12px;}
.stat strong {display:block; font-size: 25px; color: #16826e;}
.stat span {font-size: 13px; opacity: .8;}
#comparison-status {border-left: 4px solid #16826e; padding-left: 18px; margin: 10px 0;}
#footer {font-size: 12px; opacity: .8; padding: 20px 0; border-top: 1px solid #b8cbd2;}
button.primary {font-weight: 650 !important;}
.qfs-plot svg {width: 100%; height: auto; display: block;}
@media(max-width: 640px) {#hero {padding: 22px 18px;} #hero h1 {font-size: 29px;} .stat {min-width: 100px;} [role="tab"] {padding-inline: 8px !important;}}
"""




def md(value):
    """Escape registry/user strings before placing them in Markdown."""
    text = html.escape(str(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def money(value):
    return "Not included / quote needed" if value is None else "$%.2f" % value


def row_table(rows):
    return [[r.get("artifact", ""), "%.11g" % r["kl"] if r.get("kl") is not None else "Not recorded",
             "%.4f" % (100 * r["top1"]) if r.get("top1") is not None else "Not recorded",
             r.get("classification", "unknown"),
             (r["revision"][:12] + "…") if r.get("revision") else "Not pinned"] for r in rows]


def receipt_choices(rows):
    return [("%s · %s" % (r.get("artifact", "Measurement"), r.get("id", "")), r["id"])
            for r in rows]


def evidence_summary(data):
    if not data:
        return "Select a measurement to inspect its evidence."
    measurement = data.get("measurement") or {}
    artifact = data.get("artifact") or {}
    metric = measurement.get("metric") or {}
    classification = (measurement.get("comparability") or {}).get("class", "unknown")
    text = "### %s\n**KL:** %s nats · **Evidence class:** %s\n\n" % (
        md(artifact.get("name", measurement.get("id", "Measurement"))),
        md(metric.get("value", "not recorded")), md(classification))
    text += "Read the scope and disclosures before interpreting this number.\n\n"
    disclosures = measurement.get("disclosures") or []
    if disclosures:
        text += "**Disclosures**\n\n"
        for item in disclosures:
            if isinstance(item, dict):
                text += "- **%s** (%s): %s\n" % (md(item.get("code", "Disclosure")), md(item.get("severity", "note")), md(item.get("detail", "")))
            else:
                text += "- " + md(item) + "\n"
    links = data.get("source_links") or []
    if links:
        text += "\n**Open the source evidence**\n\n"
        for index, url in enumerate(links, 1):
            safe_url = quote(url, safe="/:?&=%@+,-._~#")
            text += "- [Source %d · %s](%s)\n" % (index, md(url.rsplit("/", 1)[-1][:100]), safe_url)
    else:
        text += "\nNo public source link is recorded. Inspect the original provenance below."
    return text


def group_view(registry, group_id):
    if not group_id:
        return "Choose a comparison group to see its evidence.", [], gr.Dropdown(choices=[], value=None), {}, {}
    group = registry.group(group_id)
    status = group.get("status", "unknown")
    if not group.get("context", {}).get("ranking_allowed", False) and str(status).lower() == "true":
        status = "unknown"
    title = {"true": "Comparable within this group", "false": "Do not rank these measurements",
             "unknown": "Comparability is not established"}.get(str(status).lower(), "Comparability is not established")
    note = ("Lower KL means closer next-token distributions to this reference—not higher task accuracy."
            if str(status).lower() == "true" else
            "These results are shown for inspection, not as a quality ranking. Changing a filter does not remove this restriction.")
    reasons = "\n".join("- " + md(reason) for reason in (
        group.get("reasons", []) + group.get("context", {}).get("caveats", [])))
    rows = group.get("rows", [])
    choices = receipt_choices(rows)
    first = rows[0]["id"] if rows else None
    description = "### %s\n%s\n\n%s\n\n%s" % (title, md(group.get("title", "")), note, reasons)
    return description, row_table(rows), gr.Dropdown(choices=choices, value=first), registry.detail(first) if first else {}, group.get("context", {})


def snapshot_markup(overview):
    return '<div id="stats">%s</div>' % "".join(
        '<div class="stat"><strong>%s</strong><span>%s</span></div>' % (html.escape(str(value)), label)
        for value, label in [(overview["measurement_count"], "published measurements"),
                             (overview["model_count"], "model families"),
                             (overview["group_count"], "comparison groups"),
                             ("CPU", "Public evidence browsing")])


def snapshot_footer(overview):
    return ("**Registry snapshot:** %s · %s\n\n%s\n\n"
            "[Source code](%s) · [Registry dataset](%s) · [Annotation standard](%s/blob/main/docs/CARD-ANNOTATION-SPEC.md)\n\n"
            "Shared evidence links pin this snapshot. Dataset-viewer searches use live data. Prices are dated, not quotes."
            % (md(overview.get("revision") or overview["snapshot"]), md(overview["origin"]),
               md(" ".join(overview.get("notes", []))), SOURCE_URL, REGISTRY_URL, SOURCE_URL))


def create_app():
    registry = ExplorerRegistry()
    overview = registry.overview()
    initial_revision = overview.get("revision") or "bundled"

    @lru_cache(maxsize=8)
    def registry_for(revision):
        if revision == initial_revision:
            return registry
        if not isinstance(revision, str) or not links.SHA.fullmatch(revision):
            raise ValueError("No valid registry snapshot is selected. Open the Explorer without the invalid evidence link.")
        return ExplorerRegistry(revision=revision)

    @lru_cache(maxsize=8)
    def card_choices(revision):
        current = registry_for(revision)
        rows = {row["id"]: row for _, gid in current.groups("") for row in current.group(gid)["rows"]}
        return receipt_choices([rows[mid] for mid in sorted(rows)])
    models = registry.models()
    first_model = models[0][1] if models else ""
    initial_groups = registry.groups(first_model)
    first_group = initial_groups[0][1] if initial_groups else None
    initial = group_view(registry, first_group)
    offers = costs.catalog()
    offer_choices = [(o["label"], o["id"]) for o in offers]
    first_offer = next((o["id"] for o in offers if o["id"] == "hf-jobs-l4"), offers[0]["id"] if offers else None)

    def select_model(model_id, revision):
        registry = registry_for(revision)
        choices = registry.groups(model_id or "")
        selected = choices[0][1] if choices else None
        return (gr.Dropdown(choices=choices, value=selected), *group_view(registry, selected))

    def open_comparison(group_id, revision):
        registry = registry_for(revision)
        if not group_id:
            raise gr.Error("Check a model first, then choose one of its comparison groups.")
        rows = registry.group(group_id)["rows"]
        model_id = registry.detail(rows[0]["id"])["measurement"]["model_ref"] if rows else ""
        return (gr.Dropdown(value=model_id), gr.Dropdown(choices=registry.groups(model_id), value=group_id),
                *group_view(registry, group_id))

    def lookup(target, revision):
        registry = registry_for(revision)
        try:
            result = registry.lookup(target)
        except ValueError as exc:
            return "### Check the model link\n" + md(exc), [], gr.Dropdown(choices=[], value=None), {}
        rows = result.get("rows", [])
        warnings = "\n".join("- " + md(w) for w in result.get("warnings", []))
        text = "### %s\n%s\n\n%s" % (md(result.get("title", "Lookup result")), md(result.get("message", "")), warnings)
        ids = set(result.get("group_ids", []))
        choices = [(label, key) for label, key in registry.groups("") if key in ids]
        return text, row_table(rows), gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), result.get("target", {})

    def estimate(offer_id, hours, storage_gb, storage_days, override_hourly):
        try:
            result = costs.estimate(offer_id, hours, storage_gb, storage_days,
                                    override_hourly=str(override_hourly).strip() if override_hourly is not None and str(override_hourly).strip() else None)
        except (ValueError, TypeError) as exc:
            return "### Check your estimate\n" + md(exc), {}
        offer = result["offer"]
        heading = money(result.get("total_known_usd")) + " known subtotal"
        if result.get("compute_usd") is None:
            heading = "Compute quote required · " + heading
        text = ("### %s\n**Compute:** %s · **Storage:** %s\n\n"
                "This is a scenario calculation, not a quote or a measured QFS run duration."
                % (heading, money(result.get("compute_usd")), money(result.get("storage_usd"))))
        warnings = result.get("warnings", [])
        if warnings:
            text += "\n\n" + "\n".join("- " + md(w) for w in warnings)
        source = offer.get("source_url", "")
        if source.startswith("https://"):
            text += "\n\n[Provider pricing source](%s) · Checked %s" % (source, md(offer.get("checked_at", "")))
        return text, result

    def inspect_receipt(text):
        try:
            result = contribute.inspect_receipt(text)
        except ValueError as exc:
            return "Cannot inspect this receipt: " + str(exc), {}, ""
        summary = "%s\n\n%s" % (str(result.get("status", "unknown")).upper(), result.get("summary", ""))
        for label in ("errors", "warnings"):
            if result.get(label):
                summary += "\n\n" + label.capitalize() + ":\n" + "\n".join("- " + str(v) for v in result[label])
        return summary, result.get("details", {}), result.get("next_steps", "")

    def share_evidence(data):
        if not data:
            return "Select a measurement to inspect its evidence.", "", {}
        snapshot = data.get("snapshot") or {}
        mid = (data.get("measurement") or {}).get("id")
        urls = links.evidence_links(EXPLORER_BASE, mid, snapshot.get("revision"))
        return evidence_summary(data), urls.get("explorer", "No permanent link: this is an unpinned bundled snapshot."), urls

    def generate_card(measurement_ids, existing_card, revision):
        try:
            current = registry_for(revision)
            if current.overview().get("notes"):
                raise ValueError("Card generation requires a public snapshot without loading or integrity warnings.")
            data = current.registry_data()
            links.enrich_root_reference(data, measurement_ids or [])
            result = snippets.generate_snippet(data, measurement_ids or [], explorer_base=EXPLORER_BASE,
                                               registry_revision=current.overview().get("revision") or "",
                                               existing_card=existing_card or "")
        except ValueError as exc:
            return "Cannot generate this card: " + str(exc), "", "", "", {}
        validation = result.get("validation") or {}
        status = ("Ready to copy — existing QFS checks and local HF parser roundtrip passed."
                  if validation.get("ok") else "Not ready to paste — resolve these provenance or validation findings.")
        if result.get("model_repository"):
            status = "Target model: %s · Role: %s\n\n%s" % (result["model_repository"], result.get("role", "unknown"), status)
        findings = validation.get("errors") or []
        warnings = result.get("warnings") or []
        status += "\n\n" + "\n".join(str(item) for item in findings + warnings)
        return status, result.get("metadata_yaml", ""), result.get("markdown_snippet", ""), result.get("merged_card", ""), {
            "validation": validation, "links": result.get("links", {}),
            "role": result.get("role"), "model_repository": result.get("model_repository")}

    def use_selected_for_card(mid, revision):
        registry_for(revision).detail(mid)
        return gr.update(selected="cards"), gr.Dropdown(choices=card_choices(revision), value=[mid])

    def load_link(request: gr.Request):
        try:
            params = links.parse_query(request.query_params or {})
            if not params:
                return tuple(gr.skip() for _ in linked_outputs)
            revision = params["registry_revision"] if "registry_revision" in params else initial_revision
            current = registry_for(revision)
            mid, gid, model_id = params.get("measurement"), params.get("group"), params.get("model")
            if mid:
                record = current.detail(mid)
                if params.get("group") is not None and params["group"] != record["group_id"]:
                    raise ValueError("The highlighted measurement belongs to a different comparison group.")
                gid, model_id = record["group_id"], record["measurement"]["model_ref"]
            elif gid:
                rows = current.group(gid)["rows"]
                if not rows:
                    raise ValueError("This group contains no published evidence.")
                mid = rows[0]["id"]
                model_id = current.detail(mid)["measurement"]["model_ref"]
            else:
                model_id = model_id or current.models()[0][1]
                choices = current.groups(model_id)
                gid = choices[0][1] if choices else None
            view = group_view(current, gid)
            if not mid and view[3]:
                mid = view[3]["measurement"]["id"]
            if mid:
                view = (view[0], view[1], gr.Dropdown(choices=receipt_choices(current.group(gid)["rows"]), value=mid),
                        current.detail(mid), view[4])
            result = {
                snapshot_state: revision, tabs: gr.update(selected=params.get("tab", "explore")),
                model: gr.Dropdown(choices=current.models(), value=model_id),
                group: gr.Dropdown(choices=current.groups(model_id), value=gid),
                group_status: view[0], table: view[1], detail_id: view[2], detail: view[3], context: view[4],
                cards_measurements: gr.Dropdown(choices=card_choices(revision),
                    value=[mid] if mid and any(key in params for key in ("measurement", "model", "group")) else []),
                stats: snapshot_markup(current.overview()), footer: snapshot_footer(current.overview()),
                link_notice: "**Opened linked evidence** from registry `%s`. The model, group and receipt are selected below."
                             % md(current.overview().get("revision") or current.overview()["snapshot"]),
            }
            plot_mid = params.get("measurement") if params.get("tab") == "plots" else mid
            result.update(zip(plot_ui["outputs"], plot_ui["view"](revision, gid, plot_mid, params.get("scale", "auto"))))
            if params.get("target"):
                looked_up = lookup(params["target"], revision)
                result.update({target: params["target"], lookup_status: looked_up[0], lookup_rows: looked_up[1],
                               matched_group: looked_up[2], target_json: looked_up[3]})
            return result
        except (ValueError, KeyError) as exc:
            result = {
                snapshot_state: None, link_notice: "**Cannot open this evidence link.** " + md(exc) + "\n\nNo different snapshot or measurement was substituted.",
                model: gr.Dropdown(choices=[], value=None), group: gr.Dropdown(choices=[], value=None),
                group_status: "No linked evidence loaded.", table: [], detail_id: gr.Dropdown(choices=[], value=None),
                detail: {}, context: {}, cards_measurements: gr.Dropdown(choices=[], value=[]),
                stats: "<p>Requested evidence unavailable.</p>", footer: "Open the Explorer without query parameters to browse the current snapshot.",
                lookup_rows: [], matched_group: gr.Dropdown(choices=[], value=None), target_json: {},
            }
            result.update(zip(plot_ui["outputs"], plot_ui["clear"]()))
            return result

    with gr.Blocks(title="QFS Explorer") as demo:
        snapshot_state = gr.State(initial_revision)
        gr.HTML('<div id="hero"><div class="eyebrow">QUANT FIDELITY SUITE</div>'
                '<h1>Find the evidence behind a quant.</h1>'
                '<p>Check what has already been measured, compare only like-for-like results, '
                'and plan your next measurement—with receipts, not guesswork.</p></div>')
        stats = gr.HTML(snapshot_markup(overview))
        gr.Markdown("**Browse public evidence and plots without signing in or renting a GPU.** HF Jobs are optional, explicitly authorized and billed to your signed-in account. Publication and registry acceptance are separate confirmation steps.")
        link_notice = gr.Markdown("")
        with gr.Tabs() as tabs:
            with gr.Tab("Explore", id="explore"):
                gr.Markdown("## Is your quant already measured?\nPaste a Hugging Face model link or `owner/model`. We check the revision—not just the name.")
                with gr.Row():
                    target = gr.Textbox(label="Hugging Face model", placeholder="malaiwah/GLM-5.3-Flash-TR3-6bpw", scale=5)
                    search = gr.Button("Check this model", variant="primary", scale=1)
                gr.Examples(examples=[["malaiwah/GLM-5.3-Flash-TR3-6bpw"], ["zai-org/GLM-5.3-BF16"]], inputs=[target])
                lookup_status = gr.Markdown("An exact match can save you a new capture. A different revision is not the same artifact.")
                lookup_rows = gr.Dataframe(headers=["Artifact", "KL (nats)", "Top-1 (%)", "Evidence class", "Revision (short)"],
                                           interactive=False, label="Matching evidence—not a cross-group ranking", wrap=True)
                with gr.Row():
                    matched_group = gr.Dropdown(choices=[], label="Comparison groups containing these results", scale=4)
                    open_group = gr.Button("Explore selected group", scale=1)
                with gr.Accordion("Resolved target and revision", open=False):
                    target_json = gr.JSON()
                gr.Markdown("## Browse like-for-like evidence\n**1. Pick a model → 2. Pick a panel/lane group → 3. Inspect a result.** Different panels and execution paths answer different questions.")
                with gr.Row():
                    model = gr.Dropdown(choices=models, value=first_model, label="1 · Model family", scale=2)
                    group = gr.Dropdown(choices=initial_groups, value=first_group, label="2 · Comparison group", scale=5)
                group_status = gr.Markdown(initial[0], elem_id="comparison-status")
                table = gr.Dataframe(value=initial[1], headers=["Artifact", "KL (nats)", "Top-1 (%)", "Evidence class", "Revision (short)"],
                                     interactive=False, label="Group measurements", wrap=True)
                gr.Markdown("Display values are rounded. Select a measurement below for full precision, the complete revision and original receipts.")
                with gr.Accordion("What makes these results comparable?", open=False):
                    context = gr.JSON(value=initial[4], label="Panel, reference, lane and predicate")
                detail_id = gr.Dropdown(choices=receipt_choices(registry.group(first_group)["rows"]) if first_group else [],
                                        value=registry.group(first_group)["rows"][0]["id"] if first_group and registry.group(first_group)["rows"] else None,
                                        label="3 · Inspect a measurement and its receipts")
                evidence = gr.Markdown(evidence_summary(initial[3]))
                initial_share = share_evidence(initial[3])
                share_url = gr.Textbox(value=initial_share[1], label="Permanent link to this exact evidence", interactive=False, buttons=["copy"])
                card_selected = gr.Button("Create model-card snippet for this result", variant="primary")
                plot_selected = gr.Button("Plot this measurement with its same-lane peers")
                with gr.Accordion("Dataset links and immutable registry source", open=False):
                    share_sources = gr.JSON(value=initial_share[2])
                with gr.Accordion("Original records and machine-readable provenance", open=False):
                    detail = gr.JSON(value=initial[3], label="Complete evidence", open=False)
                with gr.Accordion("New to KL and QFS? Read this first", open=False):
                    gr.Markdown("**KL divergence** measures how much a candidate's next-token probabilities differ from a reference. Lower is closer, but only within a valid comparison.\n\n"
                                "**A panel** is a fixed set of contexts. **A lane** is the execution path. **Scope** says which weights or activations were quantized.\n\n"
                                "**Advisory** means there are limitations to disclose—not that the number is useless. **Strict** is not a universal quality endorsement.\n\n"
                                "A weights-only capture cannot tell you everything about a production serving kernel. A small preview cannot rank quantization rates. "
                                "[Read the measurement contract](%s/blob/main/WHAT-WE-MEASURE.md)." % SOURCE_URL)
                lookup_outputs = [lookup_status, lookup_rows, matched_group, target_json]
                search.click(lookup, [target, snapshot_state], lookup_outputs, api_name="lookup", concurrency_limit=2)
                target.submit(lookup, [target, snapshot_state], lookup_outputs, api_name=False, concurrency_limit=2)
                model.input(select_model, [model, snapshot_state], [group, group_status, table, detail_id, detail, context], api_name=False)
                group.input(lambda g, rev: group_view(registry_for(rev), g), [group, snapshot_state], [group_status, table, detail_id, detail, context], api_name="comparison_group")
                open_group.click(open_comparison, [matched_group, snapshot_state],
                                 [model, group, group_status, table, detail_id, detail, context], api_name=False)
                detail_id.change(lambda key, rev: registry_for(rev).detail(key) if key else {}, [detail_id, snapshot_state], [detail], api_name="measurement")
                detail.change(share_evidence, [detail], [evidence, share_url, share_sources], api_name=False)
            plot_ui = build_plot_ui(registry_for, snapshot_state, EXPLORER_BASE, initial_revision, first_group)
            def use_selected_for_plot(mid, revision, scale):
                if not mid:
                    raise gr.Error("Select a measurement first.")
                return (gr.update(selected="plots"), *plot_ui["view"](revision, None, mid, scale))
            plot_selected.click(use_selected_for_plot, [detail_id, snapshot_state, plot_ui["scale"]],
                                [tabs, *plot_ui["outputs"]], api_name=False, concurrency_limit=2)
            with gr.Tab("Costs", id="costs"):
                gr.Markdown("## Find a sensible place to run\nCompare **hardware cost**, then check model fit and QFS compatibility. A cheaper GPU-hour is not necessarily a cheaper finished measurement.")
                with gr.Row():
                    offer = gr.Dropdown(choices=offer_choices, value=first_offer, label="Hardware offer", scale=3)
                    hours = gr.Number(value=2, minimum=0, label="Your estimated billable hours", scale=1)
                with gr.Row():
                    storage_gb = gr.Number(value=0, minimum=0, label="Persistent storage (GB)")
                    storage_days = gr.Number(value=0, minimum=0, label="Storage retention (days)")
                    override = gr.Textbox(value="", placeholder="Leave blank to use the published rate", label="Optional live quote override (USD/hour)")
                estimate_button = gr.Button("Calculate scenario cost", variant="primary")
                estimate_result = gr.Markdown("Choose an offer and enter an estimated runtime. We do not predict model speed from GPU names.")
                with gr.Accordion("Calculation, assumptions and source", open=False):
                    estimate_json = gr.JSON()
                estimate_button.click(estimate, [offer, hours, storage_gb, storage_days, override], [estimate_result, estimate_json], api_name="estimate_cost")
                with gr.Accordion("Compare all provider hardware and prices", open=False):
                    gr.Dataframe(value=[[o.get("provider"), o.get("gpu"), o.get("gpu_count"), o.get("vram_gb"), o.get("hourly_usd"), o.get("price_basis"), o.get("checked_at", "")[:10]]
                                        for o in offers], headers=["Provider", "GPU", "GPU count", "VRAM / GPU (GB)", "USD / configuration / hour", "Price basis", "Checked"],
                                 interactive=False, wrap=True, label="Dated prices—not live availability")
                with gr.Accordion("Billing differences, QFS compatibility and pricing sources", open=False):
                    gr.Markdown(costs.guidance())
            with gr.Tab("Cards", id="cards"):
                gr.Markdown("## Put traceable fidelity evidence on your model card\n"
                            "**1. Select your measurements → 2. Generate → 3. Copy the YAML and evidence paragraph.**\n\n"
                            "Uses Hugging Face `model-index` plus QFS `x_fidelity`, not a new annotation format. "
                            "Select only measurements of the same artifact. Nothing is posted or changed on the Hub.")
                cards_measurements = gr.Dropdown(choices=card_choices(initial_revision), value=[],
                                                multiselect=True, label="Published measurements for this model card")
                with gr.Accordion("Optional: merge into your existing card", open=False):
                    gr.Markdown("Paste the complete README to preserve its existing metadata and body. Do not paste secrets or private material into this public app; use a private duplicate for private cards.")
                    existing_card = gr.Code(language="markdown", label="Existing model card (optional, maximum 128 KiB)", lines=8, max_lines=16)
                generate_button = gr.Button("Generate model-card snippets", variant="primary")
                card_status = gr.Textbox(label="Provenance and validation", interactive=False, lines=5, max_lines=12)
                metadata_yaml = gr.Code(language="yaml", label="YAML — merge these keys into your existing front matter", interactive=False, lines=12, max_lines=22)
                markdown_snippet = gr.Code(language="markdown", label="Evidence paragraph — paste into the card body", interactive=False, lines=8, max_lines=16)
                with gr.Accordion("Complete merged card", open=False):
                    merged_card = gr.Code(language="markdown", label="Preserved card with generated annotations", interactive=False, lines=10, max_lines=24)
                with gr.Accordion("Validation details and source links", open=False):
                    card_validation = gr.JSON()
                gr.Markdown("**About paper-style annotations:** HF extracts `arxiv:<id>` from real paper links. "
                            "QFS has receipt/specification links, not an invented paper identifier or HF verification badge. "
                            "The newer `.eval_results` format needs a registered evaluation task; this generator does not pretend QFS has one. "
                            "[Read the existing QFS annotation specification](%s/blob/main/docs/CARD-ANNOTATION-SPEC.md)." % SOURCE_URL)
                generate_button.click(generate_card, [cards_measurements, existing_card, snapshot_state],
                                      [card_status, metadata_yaml, markdown_snippet, merged_card, card_validation],
                                      api_name="generate_card_snippet", concurrency_limit=2)
                card_selected.click(use_selected_for_card, [detail_id, snapshot_state], [tabs, cards_measurements], api_name=False)
            with gr.Tab("Contribute", id="contribute"):
                gr.Markdown("## Your workspace, your budget\n**Browse first → run only with your consent → inspect saved results → choose publication and review separately.**\n\n"
                            "Use **HF Jobs** to launch supported captures and measurements in your signed-in account, with your deadline and cost ceiling. "
                            "Results remain private until you explicitly publish them. Public registry review does not automatically accept a claim.\n\n"
                            "A private copy is optional for isolating your workspace. CPU Basic copies have no hourly compute charge; paid Space hardware/storage is billed to the copy's owner. "
                            "Secrets are not copied. Upgrading the Explorer Space does not itself run a measurement.")
                with gr.Row():
                    gr.Button("Duplicate into my account", link="https://huggingface.co/spaces/%s?duplicate=true" % SPACE_ID, variant="primary")
                    gr.Button("Open registry discussions", link=REGISTRY_URL + "/discussions")
                with gr.Accordion("Step-by-step: private copies, measurements and submission", open=False):
                    gr.Markdown(contribute.guidance(SPACE_ID))
                gr.Markdown("## Check a receipt before you submit\nPaste a **measurement-receipt.json**. This performs an offline validation preview—not a model run, an upload, or registry acceptance. Do not paste credentials or private material.")
                receipt_text = gr.Code(language="json", label="Receipt JSON (maximum 1 MiB)", lines=12, max_lines=18)
                example_receipt = gr.Button("Try the published Dione Q4 example")
                inspect_button = gr.Button("Inspect receipt", variant="primary")
                inspection = gr.Textbox(label="Validation preview", lines=8, interactive=False)
                with gr.Accordion("Structured result / normalized draft", open=False):
                    inspection_json = gr.JSON()
                next_steps = gr.Textbox(label="Your next step", lines=8, interactive=False)
                inspect_button.click(inspect_receipt, [receipt_text], [inspection, inspection_json, next_steps], api_name="inspect_receipt", concurrency_limit=1)
                example_receipt.click(
                    lambda: ((Path(__file__).parent / "registry/docs/examples/dione-q4.submission.json").read_text(encoding="utf-8"),
                             "Published Dione Q4 example loaded. Click Inspect receipt to run the offline checks.", {}, ""),
                    outputs=[receipt_text, inspection, inspection_json, next_steps], api_name=False)
            build_jobs_ui()
        footer = gr.Markdown(snapshot_footer(overview), elem_id="footer")
        linked_outputs = [snapshot_state, tabs, model, group, group_status, table, detail_id, detail, context,
                          cards_measurements, stats, footer, link_notice, target, lookup_status, lookup_rows,
                          matched_group, target_json, *plot_ui["outputs"]]
        demo.load(load_link, outputs=linked_outputs, api_name=False)
    return demo


if __name__ == "__main__":
    create_app().queue(max_size=32, default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"), css=CSS,
        show_error=False, ssr_mode=False,
        max_file_size=0, allowed_paths=[],
        app_kwargs={"middleware": [Middleware(ExplorerTransport, base=EXPLORER_BASE)]},
    )
