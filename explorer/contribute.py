"""Read-only contribution guidance and bounded adapters to QFS's real validators.

Result strings may contain receipt data. Render them as text/JSON, never HTML or
unescaped Markdown. Only guidance() returns trusted Markdown.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading

_ROOT = Path(__file__).resolve().parents[1]
_MAX_BYTES = 1_048_576
_VALIDATION_SLOTS = threading.BoundedSemaphore(2)
_SUBMISSION = "quant-fidelity-registry/submission-receipt.v1"
_COMPARISON = "malaiwah.fidelity-comparison-receipt.v1"
_DISCUSSIONS = "https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions"
_SUITE = "https://github.com/malaiwah/quant-fidelity-suite/blob/main"
_LIMITATION = (
    "Offline receipt validation is not registry acceptance or independent verification. "
    "It checks declarations and their internal consistency, not downloaded model/tensor "
    "bytes, remote evidence hashes, authorship, or the scientific validity of a new experiment."
)
_SUBMISSION_NEXT = (
    "Keep the original sealed measurement-receipt.json unchanged. In your suite checkout run:\n"
    "bin/registry-submit /path/to/measurement-receipt.json\n"
    "If validation passes, open " + _DISCUSSIONS + " and create a discussion titled "
    "'submission: <repo> on <panel>'. Include the CONTRIBUTING section 2 template and "
    "the original receipt. The maintainer validates and derives the registry row; the "
    "review draft is not a submission receipt and must not be inserted into data/*.jsonl. "
    "The documented GitHub PR mirror is not live. Nothing is submitted by this app."
)
_COMPARISON_NEXT = (
    "A comparison-receipt.json is not a submission-receipt.v1; do not pass it to registry-submit. "
    "Keep your full run directory and receipts. On your own machine, preview the post:\n"
    "bin/fidelity-post render --result /path/to/out/result --out post.md\n"
    "Only when you intend to publish, run locally (this opens a discussion on the candidate model):\n"
    "bin/fidelity-post publish --result /path/to/out/result --token-file ~/.hf_token "
    "--receipt /path/to/out/post-receipt.json\n"
    "Then open " + _DISCUSSIONS + " with title 'submission: <repo> on <panel>'. "
    "Include that model discussion URL, the candidate dataset repo@revision if published, "
    "and attach result/receipts/reference-comparison/comparison-receipt.json and "
    "result/receipts/root-qualification.json; provide the full receipts directory to the "
    "maintainer. The maintainer validates and derives the row. Never paste your token here. "
    "See THIRD-PARTY-QUICKSTART sections 5–6; the GitHub PR mirror is not live."
)


def guidance(space_id: str = "malaiwah/qfs-explorer") -> str:
    """Return trusted Markdown; only a validated Space id enters its links."""
    if not isinstance(space_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", space_id
    ):
        raise ValueError("Use a Space id in owner/name form, not a URL.")
    return f"""## Your workspace, your budget

Public evidence browsing, plots, cost scenarios and pasted-receipt inspection require no login
or rented GPU. **HF Jobs** adds explicit caller-funded captures and measurements; publication
and registry review are separate actions. Sign in through HF OAuth for those workflows.
Do not paste secrets or confidential artifacts into the public receipt/card forms.

### Make a private copy

1. Open [this Space](https://huggingface.co/spaces/{space_id}) and use its menu → **Duplicate this Space**.
2. Choose your own account or organization as owner and **Private** visibility. Review the hardware
   choice before confirming; keep **CPU Basic** for the Explorer interface.
3. The **new owner is billed for any paid Space hardware/storage they select**, not this public Space's owner.
   Secrets are **not copied**. Public browsing needs no tokens; authenticated actions use the current
   caller's HF OAuth identity, never ambient Space-owner credentials.
4. A private CPU copy is a private Explorer interface. **HF Jobs is separate compute**, billed to
   the signed-in caller. Upgrading Space hardware does not itself start a measurement.
   Preserve result-bucket/repository links; the Space's temporary filesystem is not durable storage.

See [HF duplication](https://huggingface.co/docs/hub/spaces-overview#duplicating-a-space),
[Space secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables)
and [hardware billing](https://huggingface.co/docs/hub/spaces-gpus).

### Measure your own model with your own resources

**Implemented today, outside this app:** use the suite's local measurement tools on supported
hardware/surfaces, or its current **RunPod secure on-demand** candidate route from a machine you
control. Start with `bin/measure <hf-url> --plan-only` and the
[support matrix]({_SUITE}/README.md#before-you-rent-what-is-measurable-today).
Model, panel availability, storage surface, lane and profile must all be supported; a model URL
alone is not enough. Local lanes are narrower than the cloud lane and cannot execute the
third-party quant recipe described in the current contribution guide.

- Local planning: `bin/measure-local --help`, then the documented `--estimate-only` recipe.
- RunPod planning: follow the [third-party quickstart]({_SUITE}/docs/THIRD-PARTY-QUICKSTART.md),
  especially section 3b for a candidate against a published root. Use the exact `--dry-run`
  command before any paid run. It may perform metadata/account checks but provisions no pod;
  its all-in hard cap is a liability ceiling, **not expected spend**.
- Keep credentials in the owner-only files required by that CLI, on **your machine**. The
  reaper, budget caps, two cold runs and paid confirmation are real requirements, not options
  this Explorer bypasses. Read the [cloud contract]({_SUITE}/docs/CLOUD-RECIPES.md).

**HF Jobs in this app:** choose a supported workflow, inspect the pinned source and inputs,
set your deadline/cost ceiling, and explicitly authorize launch. You can refresh saved state,
cancel a run and recover persisted captures before deciding whether to publish. Jobs use your
account and compute quota; duplicating this Space transfers no credits or credentials.
Publication is a separate confirmation with visibility and rights checks. Public review requests
do not automatically write registry rows; only the authenticated registry owner can accept them.

### Bring back evidence, not a hand-written registry row

Keep the pinned artifact identity, sealed captures, qualification, comparisons, archive and
terminal/billing receipt together. Pasting JSON below does not upload those artifacts or verify
remote downloads. A valid seal only proves the receipt is internally unchanged, not that its
claims are true.

- **Legacy teacher-logits/local submission:** paste the runner's sealed
  `measurement-receipt.json` (`quant-fidelity-registry/submission-receipt.v1`). We run the
  existing schema/seal/adaptation gate and determinism checks without changing the registry.
  Any generated row preview is a **review draft**, never the receipt you submit.
- **Current candidate route:** paste `result/receipts/reference-comparison/comparison-receipt.json`
  (`malaiwah.fidelity-comparison-receipt.v1`). We use the existing comparison validator for
  schema, seal, self-comparison and bias rules. This format is **not a registry submission**.
  Keep `root-qualification.json` and all supporting receipts for maintainer review.
- Preserve advisory classes, disclosures and refusals. Equal comparability keys alone do not
  grant a ranking license. Outside measurements enter as advisory; independent reproduction
  is a separate claim, not awarded by this form.

The **live submission destination** is the [HF registry dataset discussions]({_DISCUSSIONS}).
Use [quickstart sections 5–6]({_SUITE}/docs/THIRD-PARTY-QUICKSTART.md#5-what-goes-to-the-registry-and-who-files-it)
and the [contribution contract]({_SUITE}/registry/CONTRIBUTING.md) for exact commands and attachments.
The GitHub PR mirror described there is **not live**. The receipt result below tells you which
route applies; nothing is published automatically.
"""


def _result(status, summary, *, errors=None, warnings=None, details=None, next_steps=None):
    return {
        "status": status, "summary": summary, "errors": errors or [],
        "warnings": [_LIMITATION] + (warnings or []), "details": details or {},
        "next_steps": next_steps or _SUBMISSION_NEXT,
    }


def _parse(text, *, max_nodes=32768):
    if type(max_nodes) is not int or not 1 <= max_nodes <= 65536:
        raise ValueError("Invalid bounded JSON node allowance.")
    if not isinstance(text, str):
        raise ValueError("Paste one complete receipt JSON object; no credentials.")
    if len(text) > _MAX_BYTES or len(text.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("Receipt exceeds the 1 MiB UTF-8 limit. Paste only the receipt, not tensors or an archive.")
    if not text.strip():
        raise ValueError("Paste one complete receipt JSON object; no credentials.")

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("Duplicate JSON keys are ambiguous. Use the original runner-emitted receipt.")
            obj[key] = value
        return obj

    def nonfinite(_):
        raise ValueError("NaN and Infinity are not valid receipt JSON numbers.")

    try:
        doc = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON at line {exc.lineno}, column {exc.colno}. Paste the complete JSON file without Markdown fences.") from None
    if not isinstance(doc, dict):
        raise ValueError("A receipt must be a JSON object, not an array or scalar.")
    stack = [(doc, 0)]
    count = 0
    while stack:
        value, depth = stack.pop()
        count += 1
        if depth > 48 or count > max_nodes:
            raise ValueError("Receipt is too deeply nested or complex for this public inspector. Use the local validator.")
        if isinstance(value, (dict, list)):
            if len(value) > 4096:
                raise ValueError("Receipt contains an oversized object or array. Use the local validator for bulk data.")
            children = list(value.keys()) + list(value.values()) if isinstance(value, dict) else value
            stack.extend((child, depth + 1) for child in children)
        elif isinstance(value, str):
            if len(value) > 65536:
                raise ValueError("A receipt string exceeds 64 KiB. Paste receipt metadata, not embedded files.")
            value.encode("utf-8")  # Reject escaped unpaired surrogates before canonical sealing.
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Receipt numbers must be finite; a numeric exponent overflowed.")
    return doc


def inspect_receipt(text: str) -> dict:
    """Validate only. Untrusted input never selects files, commands or network URLs.

    The upstream JSON Schema regexes were not designed for hostile public input.
    A short-lived isolated Python worker bounds those checks, instead of running
    potentially pathological regexes on a Gradio request thread indefinitely.
    """
    try:
        _parse(text)
    except (ValueError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError) else "Receipt encoding or nesting is invalid. Use UTF-8 JSON with ordinary nesting."
        return _result("refused", "Receipt could not be read safely.", errors=[message])
    if not _VALIDATION_SLOTS.acquire(blocking=False):
        return _result("refused", "Receipt validators are busy.", errors=["Try again shortly, or use the local validator; nothing has been submitted."])
    try:
        with tempfile.TemporaryDirectory(prefix="qfs-receipt-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--validate-receipt"],
                input=text, text=True, encoding="utf-8", capture_output=True,
                cwd=directory, timeout=8, check=False,
            )
        if completed.returncode != 0:
            raise ValueError("The bundled validator could not complete. Try the local command and report the failure to the Space maintainer.")
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return _result("refused", "Validation exceeded the public time limit.", errors=["This receipt could not be safely checked within 8 seconds. Use the original receipt with the local validator."])
    except (OSError, ValueError) as exc:
        message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else "The receipt validator is unavailable. Try the local command or contact the Space maintainer."
        return _result("refused", "Validation could not complete.", errors=[message])
    finally:
        _VALIDATION_SLOTS.release()


def _validate_document(doc, text):
    # Fixed trusted import roots only; no path from the pasted object is opened.
    sys.path.insert(0, str(_ROOT / "registry" / "tools"))
    sys.path.insert(0, str(_ROOT / "bin"))
    import _minischema
    import registry_add as add
    import registry_lib as lib
    import registry_validate as validate

    if doc.get("submission_schema") == _SUBMISSION:
        schemas = _minischema.Registry(str(_ROOT / "registry" / "schema"))
        errors = schemas.validate(doc, "submission.schema.json")
        if errors:
            return _result("refused", "Submission receipt schema failed.", errors=[str(e) for e in errors[:20]])
        # A fixed private location, never a measurer-supplied path. The logical
        # preview source below avoids exporting temporary server filesystem paths.
        receipt = Path("measurement-receipt.json")
        receipt.write_text(text, encoding="utf-8")
        registry = lib.load_registry(str(_ROOT / "registry" / "data"))
        try:
            sub, _, digest = add.load_submission(str(receipt))
            row, new = add.submission_to_records(
                sub, "receipts/measurement-receipt.json", digest, registry,
                maintainer_attribution=False,
            )
        except add.Refuse as exc:
            return _result("refused", "Existing submission gate refused the receipt.", errors=[str(exc)],
                           details={"validator_exit_code": exc.code},
                           next_steps=(("Resolve this finding: " + exc.remedy + "\n\n") if exc.remedy else "") + _SUBMISSION_NEXT)
        errors = schemas.validate(row, "measurement.schema.json")
        if errors:
            return _result("refused", "The generated measurement row failed schema validation.", errors=[str(e) for e in errors[:20]])
        for record in new:
            registry[lib.collection_of_id(record["id"])][record["id"]] = record
        registry["measurements"] = {row["id"]: row}
        report = validate.Report()
        validate.check_determinism(registry, report)
        warnings = [f"{d['code']}: {d.get('detail', '')}" for d in row["disclosures"]]
        warnings += [f"{f['check']}: {f['message']}" for f in report.warnings]
        if report.errors:
            return _result("refused", "The generated row failed existing determinism checks.",
                           errors=[f"{f['check']}: {f['message']}" for f in report.errors], warnings=warnings)
        draft = {"notice": "Review draft only; not accepted, published, or a submission receipt.",
                 "measurement": row, "new_records": new}
        return _result(
            row["comparability"]["class"], "Offline submission checks passed; the generated row remains advisory pending maintainer review.",
            warnings=warnings + ["Only the submission gate, generated measurement schema and determinism invariants ran. Full-registry, evidence-on-disk and maintainer acceptance checks did not run. Same-key rows are not automatically rankable."],
            details={"receipt_schema": _SUBMISSION, "receipt_sha256": sub["receipt_sha256"],
                     "class": row["comparability"]["class"], "metric": row["metric"],
                     "comparability": row["comparability"], "disclosures": row["disclosures"],
                     "checks": ["submission_schema", "seal", "scope_digest", "submission_to_records", "measurement_schema", "determinism"],
                     "draft_filename": "measurement-review-draft.json",
                     "normalized_draft_json": json.dumps(draft, ensure_ascii=False, allow_nan=False, indent=2)},
        )
    if doc.get("schema") == _COMPARISON:
        from fidelity import dsvalidate
        # validate_receipt also examines semantic fields after schema errors;
        # guard their shapes with its own schema checker before that phase.
        errors = dsvalidate.schema_errors(doc, "fidelity-comparison-receipt.schema.json")
        if errors:
            return _result("refused", "Comparison receipt schema failed.", errors=errors[:20], next_steps=_COMPARISON_NEXT)
        report = dsvalidate.validate_receipt(doc)
        declared_class = doc["comparability"]["class"]
        receipt_warnings = [
            f"{d['severity']} / {d['code']}: {d['detail']}" for d in doc["disclosures"]
        ]
        next_steps = _COMPARISON_NEXT
        if doc["comparison_kind"] != "measurement":
            receipt_warnings.append("This is a reproduction confirmation or run-to-run floor, not a quantization measurement. It cannot be submitted as a candidate measurement.")
            next_steps = (
                "Keep this as supporting evidence, not the measurement to submit. "
                "Find result/receipts/reference-comparison/comparison-receipt.json "
                "from a comparison of different candidate and reference artifacts. "
                "Do not relabel or reseal this receipt to turn it into a measurement.\n\n"
                + _COMPARISON_NEXT
            )
        return _result(
            "refused" if report.errors else ("advisory" if declared_class == "advisory" else "preview"),
            "Comparison receipt validation failed." if report.errors else "Comparison receipt checks passed; this is evidence for review, not a registry submission.",
            errors=[f"{f['rule']} / {f['code']}: {f['message']}" for f in report.errors],
            warnings=[f"{f['rule']} / {f['code']}: {f['message']}" for f in report.warnings]
                     + receipt_warnings
                     + ["The separate root qualification and downloaded dataset bytes were not supplied or checked. Receipt comparability remains a declaration until supporting evidence is reviewed."],
            details={"receipt_schema": _COMPARISON, "checks": report.checks, "declared_class": declared_class,
                     "receipt_sha256": doc.get("receipt_sha256"), "comparison_kind": doc.get("comparison_kind"),
                     "metric": doc.get("metric"), "comparability": doc.get("comparability"),
                     "gates": doc.get("gates"), "submission": doc.get("submission"),
                     "findings": doc.get("findings"), "disclosures": doc.get("disclosures")},
            next_steps=next_steps,
        )
    return _result(
        "refused", "This JSON is not a supported measurement receipt.",
        errors=["Expected submission_schema = " + _SUBMISSION + " or schema = " + _COMPARISON + ". A billing/terminal receipt, manifest, raw registry row or old report is not interchangeable with either."],
        details={"supported_receipt_schemas": [_SUBMISSION, _COMPARISON]},
        next_steps="Find the runner's measurement-receipt.json, or the candidate route's reference-comparison/comparison-receipt.json. See " + _SUITE + "/docs/THIRD-PARTY-QUICKSTART.md#5-what-goes-to-the-registry-and-who-files-it",
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--validate-receipt"]:
        raise SystemExit("This module is an Explorer validator, not a measurement runner.")
    try:
        _text = sys.stdin.read(_MAX_BYTES + 1)
        _output = _validate_document(_parse(_text), _text)
    except Exception:
        # Never return filesystem paths, traceback internals or environment data.
        _output = _result("refused", "The existing validator could not safely process this receipt.",
                          errors=["Check that this is the complete original receipt. Run the local validator for diagnostics; if it succeeds, report this incompatibility to the Space maintainer."])
    print(json.dumps(_output, ensure_ascii=True, allow_nan=False))
