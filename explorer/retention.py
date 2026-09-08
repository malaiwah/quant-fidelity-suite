"""Caller-owned, inventory-bound deletion of terminal HF Job staging only."""
from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timezone

from . import jobs


def _inventory(actor, job_id):
    from huggingface_hub import BucketFile

    job = jobs._owned_job(actor, job_id)
    if jobs._stage(job) not in jobs._TERMINAL:
        raise jobs.JobsError("Stop the selected Job before deleting any of its staging files.")
    repository, head, ledger = jobs._ledger(actor)
    workflow = (job.labels or {}).get("qfs_workflow_id")
    if not jobs.UUID.fullmatch(workflow or ""):
        raise jobs.JobsError("The Job has no exact workflow identity.")
    saved = ledger["runs"].get(workflow)
    if not saved or saved.get("job_id") != job_id:
        raise jobs.JobsError("The saved workflow does not identify this exact Job.")
    plan = saved["plan"]
    jobs._verify_provider(actor, job, plan)
    bucket = actor.username + "/qfs-explorer-results"
    prefix = "runs/" + workflow + "/"
    if plan["output"]["bucket"] != bucket or plan["output"]["prefix"] + "/" != prefix:
        raise jobs.JobsError("The selected staging prefix is not this caller's exact workflow.")
    if actor.client().bucket_info(bucket).private is not True:
        raise jobs.JobsError("Retention only operates on the caller's private staging bucket.")
    files = []
    total = 0
    for item in actor.client().list_bucket_tree(bucket, prefix=prefix, recursive=True):
        if not isinstance(item, BucketFile):
            continue
        version = getattr(item, "xet_hash", None)
        if not isinstance(version, str) or not jobs.HEX.fullmatch(version):
            raise jobs.JobsError("Staging file lacks an exact content version; deletion cannot be bound safely.")
        if not item.path.startswith(prefix):
            raise jobs.JobsError("Bucket inventory escaped the selected workflow prefix.")
        jobs._relative(item.path[len(prefix):])
        size = getattr(item, "size", None)
        if type(size) is not int or size < 0 or len(files) >= jobs.MAX_FILES:
            raise jobs.JobsError("Staging inventory is unbounded or too large for interactive deletion.")
        total += size
        files.append({"path": item.path, "bytes": size, "xet_hash": version})
    files.sort(key=lambda row: row["path"])
    inventory = {"purpose": "qfs.private-staging-retention.v1", "owner": actor.username,
                 "job_id": job_id, "workflow_id": workflow, "bucket": bucket,
                 "prefix": prefix, "plan_sha256": plan["plan_sha256"],
                 "files": files, "file_count": len(files), "bytes": total}
    inventory["inventory_sha256"] = hashlib.sha256(jobs.canonical(inventory)).hexdigest()
    return inventory, repository, head, ledger, saved


def preview(actor, job_id):
    """Read-only preview. Nothing is deleted or automatically scheduled."""
    with jobs._LOCK:
        inventory, _, _, _, saved = _inventory(actor, job_id)
        ticket = {"purpose": "qfs.delete-private-staging.v1", "owner": actor.username,
                  "job_id": job_id, "inventory_sha256": inventory["inventory_sha256"],
                  "expires_at": int(time.time()) + 600}
        ticket["signature"] = hmac.new(jobs._signing_key(), jobs.canonical(ticket), hashlib.sha256).hexdigest()
        publications = [{"visibility": visibility, "repository": value["repository"], "revision": value["revision"]}
                        for visibility, value in (saved.get("publications") or {}).items()]
        summary = {key: value for key, value in inventory.items() if key != "files"}
        summary["file_sample"] = inventory["files"][:20]
        summary["sample_is_complete"] = inventory["file_count"] <= 20
        return {"inventory": summary, "ticket": ticket, "publications_preserved": publications,
                "previous_deletion": saved.get("staging_retention"),
                "notice": "Deletes only this terminal Job's private bucket inputs/outputs. Keeps the Job, audit ledger, published/private evidence datasets and registry records. Without a published copy, capture recovery from staging will be lost."}


def delete(actor, reviewed, *, confirm_delete=False):
    """CAS-journaled deletion, followed by a fresh empty-prefix observation."""
    if confirm_delete is not True or not isinstance(reviewed, dict):
        raise jobs.JobsError("Preview and explicitly confirm deletion of the selected private staging files.")
    ticket = reviewed.get("ticket")
    required = {"purpose", "owner", "job_id", "inventory_sha256", "expires_at", "signature"}
    if not isinstance(ticket, dict) or set(ticket) != required:
        raise jobs.JobsError("Invalid deletion ticket; preview again.")
    signed = {key: value for key, value in ticket.items() if key != "signature"}
    signature = hmac.new(jobs._signing_key(), jobs.canonical(signed), hashlib.sha256).hexdigest()
    if (ticket["purpose"] != "qfs.delete-private-staging.v1" or ticket["owner"] != actor.username
            or type(ticket["expires_at"]) is not int or not time.time() < ticket["expires_at"] <= time.time() + 600
            or not isinstance(ticket["signature"], str) or not hmac.compare_digest(signature, ticket["signature"])):
        raise jobs.JobsError("Deletion preview expired, changed, or belongs to another caller.")
    with jobs._LOCK:
        inventory, repository, head, ledger, saved = _inventory(actor, ticket["job_id"])
        if inventory["inventory_sha256"] != ticket["inventory_sha256"]:
            raise jobs.JobsError("Staging changed after preview; review the new inventory before deleting.")
        record = {"status": "DELETING", "job_id": ticket["job_id"], "prefix": inventory["prefix"],
                  "inventory_sha256": inventory["inventory_sha256"], "file_count": inventory["file_count"],
                  "bytes": inventory["bytes"], "requested_at": datetime.now(timezone.utc).isoformat()}
        saved["staging_retention"] = record
        head = jobs._save_ledger(actor, repository, head, ledger)
        try:
            paths = [entry["path"] for entry in inventory["files"]]
            for start in range(0, len(paths), 1000):
                actor.client().batch_bucket_files(inventory["bucket"], delete=paths[start:start + 1000])
            remaining, _, _, _, _ = _inventory(actor, ticket["job_id"])
            if remaining["file_count"]:
                raise jobs.JobsError("Some staging files remain; preview the remaining inventory before retrying.")
        except Exception:
            # The durable DELETING journal is intentional: bucket batches are not atomic.
            # A fresh preview permits safe reconciliation, never a silent success claim.
            raise
        record.update(status="DELETED", verified_empty_at=datetime.now(timezone.utc).isoformat())
        jobs._save_ledger(actor, repository, head, ledger)
        return {"job_id": ticket["job_id"], "staging_retention": record,
                "published_datasets_deleted": False, "registry_records_deleted": False,
                "notice": "Selected private staging is empty. The Job and audit/publication records remain."}
