# `docs/cards/` — the annotated K6 and K8 model cards

These are the **reference implementation** of
[`../CARD-ANNOTATION-SPEC.md`](../CARD-ANNOTATION-SPEC.md), applied to our own
two published models. Frontmatter is produced by `bin/fidelity-card annotate`
from registry rows; `annotate` does not rewrite body prose. **Correction,
2026-09-07:** the local bodies now include scientific corrections to the older
published cards. They are not asserted byte-identical to the live Hub bodies,
and this repository update does not publish them to the Hub.

**Plot publication, 2026-09-08:** the `QFS-SIZE-KL` sections and their
`assets/qfs-glm53-k{6,8}-size-kl-panel25.*` files are published on both Hub
cards: K6 commit `1e40b7ede0479fe0979a649155a2188c7819628d`, K8 commit
`feac953db533aaa9378162fd6a0d145b82d9371d`. They use registry snapshot
`598c441a2281963f1469ea4ec02d166081b3ac5a`, retain the streaming/panel25
inspection-only predicate, and include PNG, SVG, CSV and provenance JSON.
Both live card images were visually verified; all five uploaded files per
model were fetched at those commits and matched byte-for-byte. Publication
added only the plot section/assets to each existing remote card; it did not
replace the remote body with the separately corrected local body.

```
GLM-5.3-Flash-TR3-6bpw.README.md    malaiwah/GLM-5.3-Flash-TR3-6bpw
GLM-5.3-Flash-TR3-8bpw.README.md    malaiwah/GLM-5.3-Flash-TR3-8bpw
```

## Verification

Pushing a card is a separately permissioned act. The following are historical
checks, not fresh live Hub/rendering verification:

| axis | result |
|---|---|
| live Hub `POST /api/validate-yaml` (the same gate a `git push` runs) | clean, both cards |
| `huggingface_hub` YAML → `ModelCardData` → YAML round-trip | structurally identical, both cards |
| our XC-1..XC-7 cross-checks against `registry/data/measurements.jsonl` | clean, both cards |

What is **not** verified is how the eval widget *renders*, which needs one push
to a private scratch model repo. The shape is byte-for-byte the structure
`HuggingFaceH4/zephyr-7b-beta` uses in production, so confidence is high, but
the operator should authorize that one scratch push before annotating the real
repositories.

## Regenerating

The registry is a moving target — rows get added, scopes get split. Each card
records which registry state produced it, in
`x_fidelity.registry.snapshot.data_sha256`. Regeneration is one command:

```bash
bin/fidelity-card annotate \
  --card docs/cards/GLM-5.3-Flash-TR3-6bpw.README.md \
  --role quant --model-name GLM-5.3-Flash-TR3-6bpw \
  --artifact-id artifact--malaiwah.glm-5.3-flash-tr3-6bpw \
  --base-model zai-org/GLM-5.3-Flash-BF16 \
  --head-file-sha256 47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0 \
  --final-norm-file-sha256 c228a123dee3062c3ad0129094e9d98a264e33087ee88d79c8d6c5a6e60f2fed \
  --equality-receipt "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/head-equality-fp8.json" \
  --dataset brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
  --dataset malaiwah/GLM-5.3-Flash-fidelity-suite-v1 \
  --out docs/cards/GLM-5.3-Flash-TR3-6bpw.README.md --validate
```

`--artifact-id` resolves every **published** measurement for that artifact, so a
new row appears in the card automatically and XC-3 keeps the two layers in step.

`--reference-model` and `--reference-revision` are **no longer passed**: the
generator derives them by walking measurement → `reference_ref` →
`artifact_ref` → `huggingface.{repository, revision}` (GEN-11). Regenerate
from the **corrected local body**, not an older remote README, so annotation
cannot restore withdrawn scientific headlines. Exact output also depends on
the pinned registry snapshot and supplied metadata.

The one field nothing in the registry supplies —
`head.lm_head_tensor_content_sha256`, which is a *content* digest and the
published receipts record only the *file* digest (O-6) — is warned about by
name, with the flag that would supply it, instead of being written as a silent
null. It stays null, so `head.replay_permitted` is `false` and a comparator must
refuse to replay these artifacts' hidden states through anyone else's head
(HEAD-4).

## What the annotation actually says

Layer 1, `model-index` — one entry, one result per (panel-scope, lane) pair:

* **lane lives in `dataset.split`.** `huggingface_hub` merges results on
  `(task.type, dataset.type, dataset.config, dataset.split, dataset.revision)`,
  so a lane carried only in `dataset.args` is silently discarded and two lanes
  collapse into one — exactly the mixing **BIAS-006** forbids.
* **measurement scope lives in `dataset.config`.** The registry carries
  `panel25` (all 25 windows) and `clean17` (the 17 that survive a 13-gram
  calibration-overlap scan) rows for the same artifact and lane. They score
  different position counts and must not merge.
* the floor-subtracted number is a **second metric in the same result**, never a
  headline, carrying `floor_measurement_id`, `floor_lane` and the
  non-additivity caveat.

Layer 2, `x_fidelity` — what `model-index` structurally cannot hold: the
fidelity-dataset pointer, the registry ids, the scope digest, determinism
evidence, and head identity.

## The head digest is deliberately null

Both cards ship `x_fidelity.head.lm_head_tensor_content_sha256: null` and
`replay_permitted: false`.

That is not an omission. Our published receipts
(`head-extraction.json`, `head-equality-fp8.json`) record the head's **file**
digest `47eaf729…`, which is a container digest and never an identity. The
**tensor content** digest is `aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4`
— recomputed independently from the published `head/head.safetensors` by
`bin/fidelity/dsformat.py::tensor_content_sha256`. These legacy cards retain
their own unbound/null head metadata; newer published root datasets in the
registry do not retroactively bind these old cards. Adding a digest requires
the exact capture/head provenance, not merely finding a familiar tensor hash.
