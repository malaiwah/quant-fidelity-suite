"""Offline, sourced cost planning. No provider clients, credentials or execution."""
from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path


_PRICE_FILE = Path(__file__).with_name("pricing.json")


def catalog() -> list[dict]:
    """Return independent plain-data offers from the checked-in price snapshot.

    vram_gb is per device; hourly_usd is for the entire gpu_count configuration.
    A null price means no verified price, not zero. Text is data: render it as
    JSON/table cells, not interpolated Markdown or HTML.
    """
    try:
        return json.loads(_PRICE_FILE.read_text(encoding="utf-8"))["offers"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("The bundled pricing catalog could not be loaded. Ask the Space owner to restore explorer/pricing.json.") from exc


def _number(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise ValueError(f"{name} must be a finite number greater than or equal to zero.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number greater than or equal to zero.") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a finite number greater than or equal to zero.")
    # JSON/Gradio uses finite IEEE floats. Refuse extreme inputs before arithmetic.
    if result > Decimal("1e100"):
        raise ValueError(f"{name} is too large for a meaningful cost estimate. Enter a smaller value.")
    return result


def _amount(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("The estimated cost is too large. Reduce the duration, storage, or hourly rate.")
    return round(result, 8)


def estimate(
    offer_id: str,
    hours: float,
    storage_gb: float = 0,
    storage_days: float = 0,
    override_hourly: float | None = None,
) -> dict:
    """Estimate one configuration's billable duration, never workload runtime.

    override_hourly is a compute-only USD/hour quote for the WHOLE configuration,
    not a per-GPU rate for a multi-GPU row. Unknown compute/storage stay None;
    total_known_usd is a subtotal, not an assertion of a complete bill. Storage
    means additional retained storage, not included ephemeral/root disk. No
    storage is priced when the provider/category does not have an exact tariff.
    """
    duration = _number(hours, "Hours")
    capacity = _number(storage_gb, "Storage GB")
    retention = _number(storage_days, "Storage days")
    rate = None if override_hourly is None else _number(override_hourly, "Hourly override")
    selected = next((row for row in catalog() if row["id"] == offer_id), None)
    if selected is None:
        raise ValueError("Choose an offer from the pricing catalog, then estimate again.")
    warnings = [
        "Planning estimate only, not a quote, spending cap, reservation, or execution authorization.",
        "Equal GPU-hours do not mean equal workload runtime. Benchmark the exact model, engine, precision, panel and hardware; VRAM fit alone is not sufficient.",
        "Include billable startup, downloads, setup, measurement, validation, retrieval and idle time in Hours. This calculator does not predict those durations.",
        "Taxes, subscriptions, credits, bandwidth, extra ports, additional disks, retries and teardown overruns are excluded unless separately stated.",
        "Prices are a dated snapshot, not live inventory. Check the source and exact regional configuration before spending outside this app.",
    ]
    assumptions = {
        "hours_entered": _amount(duration),
        "storage_gb": _amount(capacity),
        "storage_days": _amount(retention),
        "gpu_count": selected["gpu_count"],
        "vram_gb_per_device": selected["vram_gb"],
        "aggregate_vram_gb": selected["aggregate_vram_gb"],
        "hourly_rate_scope": "Whole selected configuration; do not multiply by GPU count again.",
        "storage_scope": "Additional retained storage only; not included ephemeral/root disk. Constant GB over the entered retention period.",
        "credits_deducted_usd": 0,
        "checked_at": selected["checked_at"],
    }
    if selected["gpu_count"] > 1:
        warnings.append("This is a multi-GPU configuration: its hourly rate is aggregate, but VRAM is per device. Aggregate VRAM is not one contiguous GPU memory pool.")
    if rate is not None:
        assumptions["rate_source"] = "User-entered live quote override; not independently verified."
        warnings.append("Override must be compute-only USD/hour for this exact GPU count. Do not paste a per-GPU or storage-inclusive total into this field.")
    elif selected["hourly_usd"] is not None:
        rate = Decimal(str(selected["hourly_usd"]))
        assumptions["rate_source"] = selected["price_basis"] + " snapshot"
    else:
        assumptions["rate_source"] = "Unknown; live quote required"
        warnings.append("No verified hourly rate. Enter a current compute-only hourly quote to include compute; a missing price is not free.")
    if selected["price_basis"] == "marketplace":
        warnings.append("Marketplace quotes depend on host, region, reliability, rental type and availability. Storage and transfer charges remain separate and unknown.")
    if selected["price_basis"] == "from":
        warnings.append("A 'from' price is a starting rate, not a guaranteed quote for this configuration.")
    increment = Decimal(str(selected.get("billing_increment_seconds", 1)))
    billable_seconds = (duration * 3600 / increment).to_integral_value(rounding=ROUND_CEILING) * increment
    billable_hours = billable_seconds / 3600
    assumptions["billable_hours"] = _amount(billable_hours)
    assumptions["compute_formula"] = "ceil(hours * 3600 / billing_increment_seconds) * billing_increment_seconds / 3600 * whole_configuration_hourly_usd"
    assumptions["billing_increment_seconds"] = int(increment)
    assumptions["hourly_usd_used"] = None if rate is None else _amount(rate)
    compute = None if rate is None else rate * billable_hours
    storage = None
    tariff = selected.get("storage")
    if capacity == 0 or retention == 0:
        storage = Decimal(0)
        assumptions["storage_formula"] = "No positive GB-day quantity entered; no additional storage modeled."
        if (capacity == 0) != (retention == 0):
            warnings.append("One storage input is zero, so no storage cost was modeled. Enter both GB and retention days for retained storage.")
    elif tariff is not None:
        storage = capacity * retention / Decimal(str(tariff["month_days"])) * Decimal(str(tariff["usd_per_gb_month"]))
        assumptions["storage_formula"] = "GB * days / month_days * USD_per_GB_month"
        assumptions["storage_tariff"] = tariff
        warnings.append("Storage uses a stated 30-day planning month, not a verified invoice convention; provider rounding and actual retention can change the charge.")
    else:
        warnings.append("Additional storage is unpriced for this provider/category. total_known_usd excludes it; check storage type, unit, region, billing increments and retention after stopping.")
    warnings.extend(selected["notes"])
    known = (compute if compute is not None else Decimal(0)) + (storage if storage is not None else Decimal(0))
    return {
        "compute_usd": None if compute is None else _amount(compute),
        "storage_usd": None if storage is None else _amount(storage),
        "total_known_usd": _amount(known),
        "is_complete": compute is not None and storage is not None,
        "total_label": "Known compute + modeled storage only; other charges excluded",
        "warnings": warnings,
        "offer": selected,
        "assumptions": assumptions,
    }


def guidance() -> str:
    """Return authored static Markdown; never interpolate user or catalog text."""
    return """### What this planner does — and does not do

Compare a **dated public price snapshot checked 2026-09-07**, then estimate the
cost of a duration **you supply**. This public CPU Basic Explorer is read-only:
it does not rent hardware, run model code, accept tokens, or submit results.
A low price is not a scientific qualification or a promise of available capacity.

**Practical comparison.** A single L4 has 24 GB VRAM: the checked on-demand list
rates are HF Spaces/Jobs $0.80/hour, RunPod Secure Cloud $0.49/hour and JarvisLabs
$0.44/hour. Different host RAM, disk, startup and operating environments matter.
Vast.ai needs a live host quote, not a made-up static price. Lambda documents
A10 24 GB rather than L4, but does not publish that rate in its current price table.
For bigger models, compare **per-device** A100 80 GB, H100 80 GB or H200 141 GB,
not only aggregate VRAM. Lambda's listed 8× A100 80 GB rate is $2.79 per GPU-hour,
so the whole instance is **$22.32/hour**, not $2.79. Eight GPUs are not one 640 GB GPU.
Equal GPU-hours do not imply equal runtime or equal scientific output. Benchmark
the exact workload and budget for setup, downloads, verification and retrieval.

### Hugging Face: Spaces are not Jobs

- **Spaces** host apps. CPU Basic has $0/hour compute (16 GB RAM, 50 GB ephemeral
  disk), but current docs require a paid plan to create a new Gradio/Docker
  compute Space. L4 is $0.80/hour; A100 80 GB is $2.50/hour. H100 was removed from
  Spaces in December 2025; H200 is not listed in Spaces hardware docs.
- **Jobs** run bounded tasks, separately priced and billed: CPU Basic is
  **$0.01/hour, not free**; L4 $0.80/hour; A100 80 GB $2.50/hour; H200 141 GB
  $5/hour. Jobs require a positive credit balance. Default timeout is 30 minutes;
  a longer run needs an explicit timeout. Exposed ports add $0.01 per Job-hour.
  The pricing page specifies minute billing although the overview says seconds;
  this planner conservatively rounds to the pricing page's minute increment.
- Spaces bill while **Starting or Running**, even when idle; building is not
  billed. Paid Spaces stay awake indefinitely by default unless sleep is set.
  Sleeping or pausing stops hardware charges; a visitor can wake a sleeping app.
- PRO includes **$2/month of general compute credit**, usable across eligible
  HF compute services. This is not blanket free GPU access. The planner does
  not read balances or deduct credits, and does not include the PRO subscription.
- **Duplicating creates your own workspace under the selected owner.** Its
  hardware/storage usage is billed to that owner (your account or organization),
  not inherited as free compute from this public app. Choose CPU Basic, check
  the creation/plan requirements, and review billing before any later upgrade.
  A private duplicate of this Explorer is still only a read-only Explorer.
- Space local disk is ephemeral. Current HF docs recommend Storage Buckets for
  persistent volumes; Hub/bucket storage allowances and overage are distinct
  from GPU RAM and ephemeral disk. Additional HF storage is left unpriced here.

### Operating costs that an hourly GPU number hides

RunPod Pods use per-second compute billing; running idle Pods cost money. A
stopped Pod can retain chargeable volumes. Disk type changes the tariff:
container $0.10/GB-month, volume $0.10 running/$0.20 stopped, and standard network
storage below 1 TB $0.07/GB-month (network volume billing is hourly). Because this
planner has no disk-type/state selector, it leaves RunPod storage unpriced.

JarvisLabs uses per-minute on-demand billing and lists $0.10/GB-month storage;
this planner uses an explicit 30-day approximation. USD rates apply outside
India; GPU availability varies across India/EU regions. Template/VM startup
claims are vendor claims, not measured QFS setup times. Retained storage is
separate from compute, and pause does not mean all stored data is free.

Vast.ai is a host marketplace. On-demand, interruptible and prepaid reserved
quotes are different products. Verify GPU count/VRAM, host reliability, region,
CPU/RAM/disk, transfer charges and interruption terms. Storage continues while
an instance exists, including while stopped. The catalog intentionally has
null prices; enter a live **compute-only whole-configuration quote**.

Lambda list prices are per GPU, not per server. The catalog converts verified
8-GPU configurations to whole-instance prices. On-demand billing is by the
minute after launch health checks, including idle use, until termination.
Filesystems continue billing after instance deletion and must be in the same
region; the documentation's $0.20/GiB-month is **only an example**, so storage
is left unpriced. H200 marketing mentions are not a verified instance quote;
GH200 is a different 96 GB GPU with an ARM CPU, not H200 141 GB.

### QFS execution admission is a separate decision

The checked repository's `bin/fidelity/providers.py` admits the qualified
RunPod paid lifecycle, subject to the exact CLI preflight, identity, scientific,
cost/deadline, retrieval and reaper gates. That is **not** a rental facility in
this app and does not mean every RunPod/GPU offer is qualified. JarvisLabs,
Lambda and Vast.ai adapters retain explicit blockers for the publishable paid
measurement route; cheap capacity or a rehearsal is not an admitted receipt.
HF Spaces and HF Jobs are **not currently admitted QFS execution backends**.
Jobs hardware/flavor data is included for comparison and a possible future
qualified integration, not speculative provider execution. Never weaken strict,
advisory or refusal semantics merely to fit a cheaper machine.

### Primary sources

- [HF Spaces hardware, disk and billing](https://huggingface.co/docs/hub/spaces-gpus)
- [HF Jobs hardware, pricing and timeout](https://huggingface.co/docs/hub/jobs-pricing)
- [HF monthly compute credits](https://huggingface.co/docs/inference-providers/pricing)
- [HF persistent storage](https://huggingface.co/docs/hub/spaces-storage)
- [RunPod GPU list prices](https://www.runpod.io/pricing) · [Pod billing/storage](https://docs.runpod.io/pods/pricing)
- [JarvisLabs pricing and regions](https://jarvislabs.ai/pricing)
- [Vast.ai pricing](https://vast.ai/pricing) · [Marketplace billing](https://docs.vast.ai/guides/instances/pricing)
- [Lambda prices](https://lambda.ai/pricing) · [GPU counts/specifications](https://docs.lambda.ai/public-cloud/on-demand/) · [Billing](https://docs.lambda.ai/public-cloud/billing/)
"""
