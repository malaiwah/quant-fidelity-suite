"""QFS Explorer: a read-only, CPU-only interface; never provisions compute."""
from __future__ import annotations

import html
from pathlib import Path
import os
import re
from urllib.parse import quote

import gradio as gr
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

from explorer.data import ExplorerRegistry
from explorer import costs, contribute

SPACE_ID = os.environ.get("SPACE_ID", "malaiwah/qfs-explorer")
SOURCE_URL = "https://github.com/malaiwah/quant-fidelity-suite"
REGISTRY_URL = "https://huggingface.co/datasets/malaiwah/quant-fidelity-registry"
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
@media(max-width: 640px) {#hero {padding: 22px 18px;} #hero h1 {font-size: 29px;} .stat {min-width: 100px;}}
"""


class ReadOnlyTransport:
    """This text/JSON app has no upload, file download or remote-file proxy."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        endpoint = path.split("/gradio_api/", 1)[-1]
        if scope["type"] == "http" and "/gradio_api/" in path and endpoint.startswith(("upload", "file=", "file/", "stream/")):
            await PlainTextResponse("File transport is disabled in this read-only Explorer.", status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


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


def create_app():
    registry = ExplorerRegistry()
    overview = registry.overview()
    models = registry.models()
    first_model = models[0][1] if models else ""
    initial_groups = registry.groups(first_model)
    first_group = initial_groups[0][1] if initial_groups else None
    initial = group_view(registry, first_group)
    offers = costs.catalog()
    offer_choices = [(o["label"], o["id"]) for o in offers]
    first_offer = next((o["id"] for o in offers if o["id"] == "hf-jobs-l4"), offers[0]["id"] if offers else None)

    def select_model(model_id):
        choices = registry.groups(model_id or "")
        selected = choices[0][1] if choices else None
        return (gr.Dropdown(choices=choices, value=selected), *group_view(registry, selected))

    def open_comparison(group_id):
        if not group_id:
            raise gr.Error("Check a model first, then choose one of its comparison groups.")
        rows = registry.group(group_id)["rows"]
        model_id = registry.detail(rows[0]["id"])["measurement"]["model_ref"] if rows else ""
        return (gr.Dropdown(value=model_id), gr.Dropdown(choices=registry.groups(model_id), value=group_id),
                *group_view(registry, group_id))

    def lookup(target):
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

    with gr.Blocks(title="QFS Explorer") as demo:
        gr.HTML('<div id="hero"><div class="eyebrow">QUANT FIDELITY SUITE</div>'
                '<h1>Find the evidence behind a quant.</h1>'
                '<p>Check what has already been measured, compare only like-for-like results, '
                'and plan your next measurement—with receipts, not guesswork.</p></div>')
        gr.HTML('<div id="stats">%s</div>' % "".join(
            '<div class="stat"><strong>%s</strong><span>%s</span></div>' % (html.escape(str(value)), label)
            for value, label in [(overview["measurement_count"], "published measurements"),
                                 (overview["model_count"], "model families"),
                                 (overview["group_count"], "comparison groups"),
                                 ("$0", "Explorer GPU spend")]))
        gr.Markdown("**No GPU required. No credentials requested. This app never rents hardware or submits on your behalf.**")
        with gr.Tabs() as tabs:
            with gr.Tab("Find & explore", id="explore"):
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
                with gr.Accordion("Original records and machine-readable provenance", open=False):
                    detail = gr.JSON(value=initial[3], label="Complete evidence", open=False)
                with gr.Accordion("New to KL and QFS? Read this first", open=False):
                    gr.Markdown("**KL divergence** measures how much a candidate's next-token probabilities differ from a reference. Lower is closer, but only within a valid comparison.\n\n"
                                "**A panel** is a fixed set of contexts. **A lane** is the execution path. **Scope** says which weights or activations were quantized.\n\n"
                                "**Advisory** means there are limitations to disclose—not that the number is useless. **Strict** is not a universal quality endorsement.\n\n"
                                "A weights-only capture cannot tell you everything about a production serving kernel. A small preview cannot rank quantization rates. "
                                "[Read the measurement contract](%s/blob/main/WHAT-WE-MEASURE.md)." % SOURCE_URL)
                lookup_outputs = [lookup_status, lookup_rows, matched_group, target_json]
                search.click(lookup, [target], lookup_outputs, api_name="lookup", concurrency_limit=2)
                target.submit(lookup, [target], lookup_outputs, api_name=False, concurrency_limit=2)
                model.input(select_model, [model], [group, group_status, table, detail_id, detail, context], api_name=False)
                group.input(lambda g: group_view(registry, g), [group], [group_status, table, detail_id, detail, context], api_name="comparison_group")
                open_group.click(open_comparison, [matched_group],
                                 [model, group, group_status, table, detail_id, detail, context], api_name=False)
                detail_id.change(lambda key: registry.detail(key) if key else {}, [detail_id], [detail], api_name="measurement")
                detail.change(evidence_summary, [detail], [evidence], api_name=False)
            with gr.Tab("Cost planner", id="costs"):
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
            with gr.Tab("Contribute & own workspace", id="contribute"):
                gr.Markdown("## Your workspace, your budget\n**1. Make a private copy → 2. Measure with your own resources outside this app → 3. Bring back the receipt for review.**\n\n"
                            "CPU Basic copies have no hourly compute charge. Paid hardware/storage is billed to the copy's owner. Secrets are not copied. "
                            "**Duplicating or upgrading this Explorer does not turn it into a GPU runner.** HF Jobs execution is a future integration, not an enabled feature.")
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
        notes = " ".join(overview.get("notes", []))
        gr.Markdown("**Registry snapshot:** %s · %s\n\n%s\n\n"
                    "[Source code](%s) · [Registry dataset](%s) · [Measurement guide](%s/blob/main/docs/THIRD-PARTY-QUICKSTART.md)\n\n"
                    "Prices are dated reference information, not booking quotes. This CPU app does not execute measurements."
                    % (md(overview["snapshot"]), md(overview["origin"]), md(notes), SOURCE_URL, REGISTRY_URL, SOURCE_URL), elem_id="footer")
    return demo


if __name__ == "__main__":
    create_app().queue(max_size=32, default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"), css=CSS,
        show_error=False, ssr_mode=False,
        max_file_size=0, allowed_paths=[],
        app_kwargs={"middleware": [Middleware(ReadOnlyTransport)]},
    )
