"""Generated public plots and caller-authenticated APIs; no filesystem proxy."""
from __future__ import annotations

import asyncio
from functools import lru_cache
import hashlib
import html
import json
import threading
import time
from urllib.parse import parse_qs, urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from . import links, plots
from .data import ExplorerRegistry
from .auth import actor_from_request, AuthError

_PLOT_LOCK = threading.BoundedSemaphore(2)
_LATEST_LOCK = threading.Lock()
_LATEST = {"expires": 0.0, "registry": None}
_MAX_BODY = 1024 * 1024


@lru_cache(maxsize=8)
def registry_at(revision):
    if not links.SHA.fullmatch(revision or ""):
        raise ValueError("Immutable plots need an exact registry revision.")
    return ExplorerRegistry(revision=revision)


def latest_registry():
    with _LATEST_LOCK:
        if _LATEST["registry"] is None or time.monotonic() >= _LATEST["expires"]:
            registry = ExplorerRegistry()
            if not registry.overview().get("revision"):
                raise ValueError("A live public registry snapshot is unavailable; no bundled revision was substituted.")
            _LATEST.update(registry=registry, expires=time.monotonic() + 300)
        return _LATEST["registry"]


def invalidate_live_registry():
    with _LATEST_LOCK:
        _LATEST["expires"] = 0


def _selectors(query, *, live=False):
    allowed = {"measurement", "group", "registry_revision", "scale", "download", "live"}
    if set(query) - allowed or any(len(v) != 1 for v in query.values()):
        raise ValueError("Unknown or repeated plot parameters.")
    args = {k: v[0] for k, v in query.items()}
    if bool(args.get("measurement")) == bool(args.get("group")):
        raise ValueError("Select exactly one measurement or comparison group.")
    if args.get("measurement") and not links.MEASUREMENT.fullmatch(args["measurement"]):
        raise ValueError("Invalid measurement identity.")
    if args.get("group") and len(args["group"]) > 512:
        raise ValueError("Comparison-group identity is too long.")
    if args.get("scale", "auto") not in ("auto", "linear", "symlog"):
        raise ValueError("Unknown plot scale.")
    if live:
        if args.get("registry_revision"):
            raise ValueError("A live plot cannot also claim a fixed revision.")
        revision = latest_registry().overview()["revision"]
    else:
        revision = args.get("registry_revision")
        if not links.SHA.fullmatch(revision or ""):
            raise ValueError("Use a full immutable registry revision, or the explicitly live endpoint.")
    return revision, args.get("measurement"), args.get("group"), args.get("scale", "auto")


@lru_cache(maxsize=64)
def plot_payload(revision, measurement, group, scale):
    return plots.build_plot(registry_at(revision), measurement_id=measurement, group_id=group, scale=scale)


@lru_cache(maxsize=64)
def plot_bytes(revision, measurement, group, scale, kind, base):
    payload = plot_payload(revision, measurement, group, scale)
    if kind == "svg":return plots.render_svg(payload, base=base)
    if kind == "png":return plots.render_png(payload)
    if kind == "csv":return plots.render_csv(payload)
    if kind == "json":return json.dumps(payload, indent=2, allow_nan=False).encode()
    raise ValueError("Unknown plot format.")


def plot_response(path, query, headers, base):
    live = path.startswith("/plots/live.") or (path == "/plots/embed" and query.get("live") == ["1"])
    revision, measurement, group, scale = _selectors(query, live=live)
    if not _PLOT_LOCK.acquire(blocking=False):
        return PlainTextResponse("Plot renderer is busy. Try the cached image again shortly.", status_code=429, headers={"Retry-After": "2"})
    try:
        payload = plot_payload(revision, measurement, group, scale)
        if path == "/plots/embed":
            urls = plots.plot_links(base, revision, measurement_id=measurement, group_id=group, scale=scale)
            svg = plots.render_svg(payload, base=base).decode()
            body = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QFS size–KL evidence plot</title><style>body{margin:0;font:14px system-ui,sans-serif;color:#111827;background:white}figure{margin:0}svg{width:100%;height:auto;display:block}footer{padding:8px 16px}a{margin-right:14px}</style></head><body><figure>' + svg + '</figure><footer><a target="_blank" rel="noopener" href="' + html.escape(urls["interactive"], quote=True) + '">Open evidence</a><a href="' + html.escape(urls["immutable"]["png"] + "&download=1", quote=True) + '">Download PNG</a><a href="' + html.escape(urls["immutable"]["csv"] + "&download=1", quote=True) + '">Data & exclusions</a>' + ("Live cached view; the resolved revision is printed in the figure." if live else "Immutable registry snapshot.") + '</footer></body></html>'
            response = HTMLResponse(body)
            response.headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors *; base-uri 'none'; form-action 'none'"
        else:
            kind = path.rsplit(".", 1)[-1]
            content = plot_bytes(revision, measurement, group, scale, kind, base)
            etag = '"' + hashlib.sha256(content).hexdigest() + '"'
            common = {"ETag": etag, "X-QFS-Registry-Revision": revision,
                      "Cache-Control": "public, max-age=300" if live else "public, max-age=31536000, immutable",
                      "X-Content-Type-Options": "nosniff", "Access-Control-Allow-Origin": "*"}
            if headers.get("if-none-match") == etag:return Response(status_code=304, headers=common)
            media = {"svg": "image/svg+xml", "png": "image/png", "csv": "text/csv; charset=utf-8", "json": "application/json"}[kind]
            if query.get("download") == ["1"] or kind in ("csv", "json"):
                common["Content-Disposition"] = 'attachment; filename="qfs-size-kl-' + revision[:12] + '.' + kind + '"'
            response = Response(content, media_type=media, headers=common)
        response.headers["Cache-Control"] = "public, max-age=300" if live else "public, max-age=31536000, immutable"
        response.headers["X-QFS-Registry-Revision"] = revision
        return response
    finally:
        _PLOT_LOCK.release()


async def _body(request):
    size = 0
    chunks = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > _MAX_BODY:raise ValueError("Request body exceeds 1 MiB.")
        chunks.append(chunk)
    raw = b"".join(chunks)
    value = json.loads(raw or b"{}")
    if not isinstance(value, dict):raise ValueError("Expected a JSON object.")
    return value


def _result_public(proof):
    result, plan = proof["result"], proof["plan"]
    return {"workflow_id": plan["workflow_id"], "status": "verified", "mode": plan["mode"],
            "plan_sha256": plan["plan_sha256"], "result_sha256": result["result_sha256"],
            "outputs": result["outputs"], "qualification": bool(proof.get("qualification")),
            "bucket": plan["output"]["bucket"], "prefix": plan["output"]["prefix"],
            "timings": proof.get("timings"),
            "notice": "Persisted bytes and scientific receipts verified; no automatic public publication or registry acceptance."}


async def api_response(request):
    from . import jobs, review, retention
    actor = await asyncio.to_thread(actor_from_request, request)
    path, method = request.url.path, request.method
    data = await _body(request) if method == "POST" else {}
    if path == "/qfs/api/account" and method == "GET":return actor.public()
    if path == "/qfs/api/jobs" and method == "GET":return {"jobs": await asyncio.to_thread(jobs.list_runs, actor)}
    if path == "/qfs/api/jobs/hardware" and method == "GET":return {"hardware": await asyncio.to_thread(jobs.hardware, actor)}
    if path == "/qfs/api/jobs/prepare" and method == "POST":return await asyncio.to_thread(jobs.prepare, actor, data)
    if path == "/qfs/api/jobs/launch" and method == "POST":return await asyncio.to_thread(jobs.launch, actor, data.get("prepared"), confirm_compute=data.get("confirm_compute") is True)
    if path.startswith("/qfs/api/jobs/"):
        parts = path[len("/qfs/api/jobs/"):].split("/")
        job_id = parts[0]
        action = parts[1] if len(parts) == 2 else "status"
        if action == "status" and method == "GET":return await asyncio.to_thread(jobs.inspect, actor, job_id)
        if action == "logs" and method == "GET":return {"logs": await asyncio.to_thread(jobs.logs, actor, job_id)}
        if action == "cancel" and method == "POST":return await asyncio.to_thread(jobs.cancel, actor, job_id, confirm=data.get("confirm") is True)
        if action == "results" and method == "POST":return _result_public(await asyncio.to_thread(jobs.fetch_result, actor, job_id))
        if action == "publish" and method == "POST":return await asyncio.to_thread(jobs.publish_result, actor, job_id, visibility=data.get("visibility", "private"), confirm_publish=data.get("confirm_publish") is True, confirm_redistribution=data.get("confirm_redistribution") is True)
        if action == "metadata" and method == "POST":return await asyncio.to_thread(jobs.update_publication_metadata, actor, job_id, data.get("metadata"), confirm_metadata=data.get("confirm_metadata") is True)
        if action == "request-review" and method == "POST":return await asyncio.to_thread(jobs.request_review, actor, job_id, confirm_public=data.get("confirm_public") is True)
        if action == "retention" and method == "GET":return await asyncio.to_thread(retention.preview, actor, job_id)
        if action == "delete-staging" and method == "POST":
            reviewed = data.get("reviewed")
            if not isinstance(reviewed, dict) or (reviewed.get("inventory") or {}).get("job_id") != job_id:
                raise ValueError("Preview this exact Job before deleting staging.")
            return await asyncio.to_thread(retention.delete, actor, reviewed, confirm_delete=data.get("confirm_delete") is True)
    if path == "/qfs/api/review" and method == "GET":return await asyncio.to_thread(review.list_requests, actor)
    if path == "/qfs/api/review/inspect" and method == "POST":return await asyncio.to_thread(review.inspect_request, actor, data.get("discussion_id"))
    if path == "/qfs/api/review/accept" and method == "POST":
        result = await asyncio.to_thread(review.accept_request, actor, data.get("approval_ticket"), confirm_accept=data.get("confirm_accept") is True)
        invalidate_live_registry()
        return result
    raise ValueError("Unknown workflow API route or method.")


class ExplorerTransport:
    def __init__(self, app, *, base):
        self.app, self.base = app, base

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send);return
        path = scope.get("path", "")
        try:
            query = parse_qs(scope.get("query_string", b"").decode("utf-8", "strict"), keep_blank_values=True, max_num_fields=64)
            if any(k in links.QUERY_FIELDS and len(v) != 1 for k, v in query.items()):raise ValueError("Repeated QFS evidence parameters are ambiguous.")
            if "deep_link" in query:raise ValueError("Use QFS measurement/snapshot links, not Gradio saved-session links.")
        except (ValueError, UnicodeError) as exc:
            await PlainTextResponse(str(exc), status_code=400)(scope, receive, send);return
        if path in ("/plots/render.svg", "/plots/render.png", "/plots/data.csv", "/plots/data.json", "/plots/live.svg", "/plots/live.png", "/plots/embed"):
            if scope["method"] not in ("GET", "HEAD"):
                await PlainTextResponse("Read-only plot endpoint.", status_code=405)(scope, receive, send);return
            try:
                headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
                response = await asyncio.to_thread(plot_response, path, query, headers, self.base)
            except (ValueError, KeyError) as exc:
                response = PlainTextResponse("Plot unavailable: " + str(exc), status_code=400)
            await response(scope, receive, send);return
        if path.startswith("/qfs/api/"):
            try:
                response = JSONResponse(await api_response(Request(scope, receive)), headers={"Cache-Control": "no-store"})
            except AuthError as exc:
                response = JSONResponse({"error": str(exc)}, status_code=401, headers={"Cache-Control": "no-store"})
            except (ValueError, KeyError, TypeError) as exc:
                response = JSONResponse({"error": str(exc)}, status_code=400, headers={"Cache-Control": "no-store"})
            except Exception:
                response = JSONResponse({"error": "The operation failed. Inspect its saved state; no alternate account or silent retry was used."}, status_code=502, headers={"Cache-Control": "no-store"})
            await response(scope, receive, send);return
        endpoint = path.split("/gradio_api/", 1)[-1]
        if "/gradio_api/" in path and endpoint.startswith(("upload", "file=", "file/", "stream/", "deep_link")):
            await PlainTextResponse("General file/saved-session transport is disabled. Use generated plot downloads or authenticated HF result links.", status_code=403)(scope, receive, send);return

        async def safe_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"content-security-policy", b"frame-ancestors 'self' https://huggingface.co"))
                headers.append((b"x-content-type-options", b"nosniff"))
                message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, safe_send)
