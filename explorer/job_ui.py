"""Gradio Jobs/review UI; all authority is obtained from the current caller."""
from __future__ import annotations

import json
import gradio as gr

from . import jobs, review
from .auth import actor_from_request


def _error(exc):
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return str(exc)
    return "HF operation failed. Refresh its saved state before retrying; no alternate account was used."


def build_jobs_ui():
    prepared_state = gr.State(None)
    publication_state = gr.State(None)
    approval_state = gr.State(None)
    options = jobs.presets()
    choices = [(p["label"], p["id"]) for p in options]
    default = next((p["id"] for p in options if p["id"] == "root:glm_moe_dsa"), choices[0][1] if choices else None)

    def account(request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            actor = actor_from_request(request, oauth_profile, oauth_token)
            hardware = jobs.hardware(actor)
            return ("Signed in as **%s**. HF Jobs are billed to **your personal account**. Results remain private until you explicitly publish." % actor.username,
                    gr.Dropdown(choices=[("%s · $%s/hour · %s" % (h["pretty_name"], h["hourly_usd"], h["ram"]), h["name"]) for h in hardware], value="cpu-basic"),
                    actor.public())
        except Exception as exc:
            return "Sign in to use Jobs. " + _error(exc), gr.skip(), {}

    def inputs(preset, flavor, seconds, maximum, mode, model_repo, model_rev, ref_repo, ref_rev, cand_repo, cand_rev, panel_repo, panel_rev, panel_path, scope, codec, bits, output_repo, review_metadata):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or (isinstance(seconds, float) and not seconds.is_integer()):
            raise ValueError("The provider deadline must be a whole number of seconds.")
        spec = {"preset": preset, "flavor": flavor, "timeout_seconds": int(seconds), "max_compute_usd": str(maximum)}
        if review_metadata and review_metadata.strip():
            metadata = json.loads(review_metadata)
            if not isinstance(metadata, dict):
                raise ValueError("Review metadata must be a JSON object.")
            spec["review_metadata"] = metadata
        if preset == "custom":
            spec.pop("preset")
            spec.update(mode=mode, model_repository=model_repo, model_revision=model_rev,
                        reference_repository=ref_repo, reference_revision=ref_rev,
                        candidate_repository=cand_repo, candidate_revision=cand_rev,
                        panel_repository=panel_repo, panel_revision=panel_rev, panel_path=panel_path,
                        scope_json=scope, codec=codec, declared_bits=bits)
        if output_repo:spec["output_repository"] = output_repo
        return {k: v for k, v in spec.items() if v not in (None, "")}

    def prepare(preset, flavor, seconds, maximum, mode, model_repo, model_rev, ref_repo, ref_rev, cand_repo, cand_rev, panel_repo, panel_rev, panel_path, scope, codec, bits, output_repo, review_metadata,
                request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            actor = actor_from_request(request, oauth_profile, oauth_token)
            prepared = jobs.prepare(actor, inputs(preset, flavor, seconds, maximum, mode, model_repo, model_rev, ref_repo, ref_rev, cand_repo, cand_rev, panel_repo, panel_rev, panel_path, scope, codec, bits, output_repo, review_metadata))
            plan = prepared["plan"]
            text = ("**Ready to run as %s** · %s · deadline %s seconds · conservative compute estimate **$%s**.\n\n"
                    "No Job was created by this preview. Output: `%s` (private staging); no worker token."
                    % (actor.username, plan["hardware"]["flavor"], plan["hardware"]["timeout_seconds"], plan["hardware"]["estimated_max_compute_usd"], plan["output"]["dataset_repository"]))
            return prepared, text, plan
        except Exception as exc:
            return None, "**Cannot prepare:** " + _error(exc), {}

    def launch(prepared, consent, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            actor = actor_from_request(request, oauth_profile, oauth_token)
            result = jobs.launch(actor, prepared, confirm_compute=consent)
            return result["job_id"], "**Job submitted to your account.** " + result["url"], result
        except Exception as exc:
            return gr.skip(), "**Not launched / check saved state:** " + _error(exc), {}

    def one_click(preset, flavor, seconds, maximum, mode, model_repo, model_rev, ref_repo, ref_rev, cand_repo, cand_rev, panel_repo, panel_rev, panel_path, scope, codec, bits, output_repo, review_metadata, consent,
                  request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            if consent is not True:raise ValueError("Confirm billing to your account before running.")
            actor = actor_from_request(request, oauth_profile, oauth_token)
            prepared = jobs.prepare(actor, inputs(preset, flavor, seconds, maximum, mode, model_repo, model_rev, ref_repo, ref_rev, cand_repo, cand_rev, panel_repo, panel_rev, panel_path, scope, codec, bits, output_repo, review_metadata))
            result = jobs.launch(actor, prepared, confirm_compute=True)
            return prepared, result["job_id"], "**Job submitted.** " + result["url"], result
        except Exception as exc:
            return None, gr.skip(), "**Not launched / check saved state:** " + _error(exc), {}

    def refresh(job_id, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            actor = actor_from_request(request, oauth_profile, oauth_token)
            mine = jobs.list_runs(actor)
            choices = [("%s · %s · %s" % (j["job_id"], j["status"], j["flavor"]), j["job_id"]) for j in mine]
            chosen = job_id or (mine[0]["job_id"] if mine else None)
            info = jobs.inspect(actor, chosen) if chosen else {}
            log = jobs.logs(actor, chosen) if chosen else "No QFS Jobs found in your account."
            return gr.Dropdown(choices=choices, value=chosen, allow_custom_value=True), info, log
        except Exception as exc:return gr.skip(), {"error": _error(exc)}, ""

    def stop(job_id, consent, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:return jobs.cancel(actor_from_request(request, oauth_profile, oauth_token), job_id, confirm=consent)
        except Exception as exc:return {"error": _error(exc)}

    def recover(job_id, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            proof = jobs.fetch_result(actor_from_request(request, oauth_profile, oauth_token), job_id)
            result = proof["result"]
            return "**Persisted result verified.** Captures/receipts are recoverable. Public publication and registry submission are still separate actions.", {"mode": proof["plan"]["mode"], "workflow_id": result["workflow_id"], "outputs": result["outputs"], "result_sha256": result["result_sha256"], "qualified": bool(proof.get("qualification"))}
        except Exception as exc:return "**Result not verified:** " + _error(exc), {}

    def publish(job_id, visibility, consent, rights, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            result = jobs.publish_result(actor_from_request(request, oauth_profile, oauth_token), job_id, visibility=visibility, confirm_publish=consent, confirm_redistribution=rights)
            return result, "**Publication complete.** Check the exact visibility and immutable links below.", result
        except Exception as exc:return None, "**Publication refused:** " + _error(exc), {}

    def request_review(job_id, consent, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            result = jobs.request_review(actor_from_request(request, oauth_profile, oauth_token), job_id, confirm_public=consent)
            return "**Review requested.** This is not acceptance or independent verification. " + result["url"], result
        except Exception as exc:return "**Request refused:** " + _error(exc), {}

    def review_list(request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            result = review.list_requests(actor_from_request(request, oauth_profile, oauth_token))
            return gr.Dropdown(choices=[("#%s · %s · %s" % (r["discussion_id"], r["author"], r["title"]), r["discussion_id"]) for r in result["requests"]], value=None), result
        except Exception as exc:return gr.Dropdown(choices=[], value=None), {"error": _error(exc)}

    def inspect_review(discussion_id, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            result = review.inspect_request(actor_from_request(request, oauth_profile, oauth_token), int(discussion_id))
            return result["approval_ticket"], "**Validated preview, not yet accepted.** Review all warnings, provenance and changed records.", result
        except Exception as exc:return None, "**Review refused:** " + _error(exc), {}

    def accept_review(ticket, consent, request: gr.Request, oauth_profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
        try:
            result = review.accept_request(actor_from_request(request, oauth_profile, oauth_token), ticket, confirm_accept=consent)
            from .transport import invalidate_live_registry
            invalidate_live_registry()
            return None, "**Accepted at an immutable registry commit.** Acceptance is not independent reproduction. " + result["commit_url"], result
        except Exception as exc:return None, "**Not accepted:** " + _error(exc), {}

    with gr.Tab("HF Jobs", id="jobs"):
        gr.Markdown("## Capture and measure in your own HF account\n**Sign in → choose a workflow → set a deadline/cost ceiling → run.** Read-only plots need no login. Jobs use your namespace, never the Space owner's credentials. Results are private by default and persist in a tokenless bucket volume.")
        with gr.Row():
            gr.LoginButton()
            load_account = gr.Button("Check my account & hardware")
            gr.DuplicateButton(value="Make a private workspace")
        account_status = gr.Markdown("Not signed in. Browsing and plotting remain public; Jobs and publication require your account.")
        account_json = gr.JSON(visible=False)
        with gr.Row():
            preset = gr.Dropdown(choices=choices + [("Custom pinned model / existing datasets", "custom")], value=default, label="Workflow preset", scale=5)
            flavor = gr.Dropdown(choices=[("CPU Basic · live price checked before launch", "cpu-basic"), ("CPU Upgrade · more memory for Fruit", "cpu-upgrade")], value="cpu-basic", label="HF hardware", scale=3)
        with gr.Row():
            seconds = gr.Number(value=600, precision=0, minimum=60, maximum=7200, label="Provider deadline (seconds)")
            maximum = gr.Textbox(value="0.25", label="Maximum compute estimate (USD)")
            output_repo = gr.Textbox(label="Optional NEW capture dataset repository", placeholder="Leave blank for a unique repo in your account")
        gr.Markdown("**Cost boundary:** HF bills starting/running time by the minute. The preview includes the deadline plus two startup minutes; it is not an account-wide hard-dollar cap. Storage and other HF services are separate. CPU Basic Jobs are paid, unlike CPU Basic Space hosting. Fruit is ~10 GB of weights; choose CPU Upgrade and a longer deadline, not the smallest host.")
        with gr.Accordion("Custom immutable inputs and actual intervention scope", open=False):
            mode = gr.Radio([("Native root: two captures + control", "root"), ("Candidate: two captures + reference measurement", "candidate"), ("Compare existing fidelity datasets", "compare")], value="root", label="Custom workflow")
            with gr.Row():
                model_repo=gr.Textbox(label="Model repository");model_rev=gr.Textbox(label="Model commit (40 hex)")
            with gr.Row():
                ref_repo=gr.Textbox(label="Reference dataset repository");ref_rev=gr.Textbox(label="Reference dataset commit")
            with gr.Row():
                cand_repo=gr.Textbox(label="Candidate dataset (compare-only)");cand_rev=gr.Textbox(label="Candidate dataset commit")
            with gr.Row():
                panel_repo=gr.Textbox(label="Original token-panel dataset");panel_rev=gr.Textbox(label="Panel dataset commit");panel_path=gr.Textbox(label="Raw token-panel path")
            scope=gr.Code(language="json",label="Actual scope JSON (or use model's pinned scope.json)")
            with gr.Row():codec=gr.Textbox(label="Codec");bits=gr.Number(label="Declared nominal bits",value=None)
            gr.Markdown("Unknown custom code is not granted execution authority. Use native Transformers classes or the listed reviewed immutable runtime pins. A raw GGUF file without its required config/layout contract is not an admitted model.")
            review_metadata = gr.Code(language="json", label="Optional original model/panel attribution for a new root review", value="{}")
        controls=[preset,flavor,seconds,maximum,mode,model_repo,model_rev,ref_repo,ref_rev,cand_repo,cand_rev,panel_repo,panel_rev,panel_path,scope,codec,bits,output_repo,review_metadata]
        consent=gr.Checkbox(value=False,label="I authorize this Job in MY HF account, with the selected deadline and compute estimate ceiling.")
        with gr.Row():
            prepare_button=gr.Button("Preview exact plan")
            run_button=gr.Button("Run selected workflow",variant="primary")
            launch_button=gr.Button("Launch reviewed plan")
        status=gr.Markdown("Preparing is read-only. Only an explicitly confirmed launch creates compute.")
        with gr.Accordion("Exact plan / provider result",open=False):plan_json=gr.JSON()
        gr.Markdown("### My Jobs and durable results\nRefresh to recover Jobs after a page/Space restart. You can also paste your own QFS Job ID from a private duplicate. COMPLETED does not mean scientific verification or publication.")
        with gr.Row():
            job_id=gr.Dropdown(choices=[],allow_custom_value=True,label="Your HF Job ID",scale=5)
            refresh_button=gr.Button("Refresh my Jobs / logs")
        job_json=gr.JSON(label="Actual provider status",open=False)
        job_log=gr.Textbox(label="Bounded Job logs (private to your authenticated session)",lines=10,interactive=False)
        with gr.Row():
            cancel_consent=gr.Checkbox(value=False,label="Cancel this selected Job")
            cancel_button=gr.Button("Cancel Job",variant="stop")
            recover_button=gr.Button("Fetch & verify persisted results")
        result_status=gr.Markdown("")
        result_json=gr.JSON(label="Verified result / immutable publication",open=False)
        with gr.Row():
            visibility=gr.Radio([("Private (default)","private"),("Public, shareable evidence","public")],value="private",label="Publication visibility")
            publish_consent=gr.Checkbox(value=False,label="Save this verified result to my new HF repositories")
        rights=gr.Checkbox(value=False,label="For PUBLIC publication: I have the rights to redistribute these captures/head weights and accept public disclosure.")
        publish_button=gr.Button("Publish verified datasets/evidence")
        review_consent=gr.Checkbox(value=False,label="Post this PUBLIC immutable evidence as a registry review request (not automatic acceptance)")
        review_button=gr.Button("Request registry review")
        request_status=gr.Markdown("")
        request_json=gr.JSON(open=False)
        load_account.click(account,outputs=[account_status,flavor,account_json],api_name=False)
        prepare_button.click(prepare,controls,[prepared_state,status,plan_json],api_name=False)
        launch_button.click(launch,[prepared_state,consent],[job_id,status,plan_json],api_name=False,concurrency_limit=1)
        run_button.click(one_click,controls+[consent],[prepared_state,job_id,status,plan_json],api_name=False,concurrency_limit=1)
        refresh_button.click(refresh,[job_id],[job_id,job_json,job_log],api_name=False)
        cancel_button.click(stop,[job_id,cancel_consent],[job_json],api_name=False)
        recover_button.click(recover,[job_id],[result_status,result_json],api_name=False,concurrency_limit=2)
        publish_button.click(publish,[job_id,visibility,publish_consent,rights],[publication_state,result_status,result_json],api_name=False,concurrency_limit=1)
        review_button.click(request_review,[job_id,review_consent],[request_status,request_json],api_name=False,concurrency_limit=1)

    with gr.Tab("Registry review",id="review"):
        gr.Markdown("## Review claims without running contributor code\nAnyone can request review of public evidence. **Only the authenticated registry namespace owner can inspect an acceptance preview and commit it.** Validation is not independent reproduction; reported evidence stays labeled as such.")
        load_requests=gr.Button("Load public review requests")
        discussion=gr.Dropdown(choices=[],label="Registry discussion")
        inspect_button=gr.Button("Validate and preview changes",variant="primary")
        review_status=gr.Markdown("No review selected. No data is written by inspection.")
        review_json=gr.JSON(label="Evidence, warnings and exact registry changes",open=True)
        acceptance=gr.Checkbox(value=False,label="I reviewed these exact records/warnings and accept the claim into this registry; I am not asserting independent reproduction.")
        accept_button=gr.Button("Accept reviewed claim",variant="primary")
        load_requests.click(review_list,outputs=[discussion,review_json],api_name=False)
        inspect_button.click(inspect_review,[discussion],[approval_state,review_status,review_json],api_name=False,concurrency_limit=1)
        accept_button.click(accept_review,[approval_state,acceptance],[approval_state,review_status,review_json],api_name=False,concurrency_limit=1)

    return {"prepared":prepared_state,"job_id":job_id,"status":status}
