<!--
SPLICE TARGET: registry/README.md
Insert as a top-level section, after the "What is in here" / collections section
and before the schema reference. Verbatim; do not summarize it further.
-->

## Measure your own quant

The supported measurement routes live in
[CONTRIBUTING.md](CONTRIBUTING.md) and the suite's
[third-party quickstart](https://github.com/malaiwah/quant-fidelity-suite/blob/main/docs/THIRD-PARTY-QUICKSTART.md).
Run commands from a clone of the suite, not from the registry dataset clone.

**Cloud candidate measurement:** use `bin/measure-cloud --provider runpod
--role candidate` with a pinned model revision, authored candidate scope,
codec and bits, matching `--panel-dir`, and a pinned `--reference-dataset`.
It also requires explicit spend limits and an output directory. Start with
the quickstart's complete `--dry-run` command: no resources are created.
There are no `--lane` or `--spot` flags on the current cloud CLI.

The candidate route produces a fidelity dataset plus
`<out>/result/receipts/reference-comparison/comparison-receipt.json`.
That is not a registry submission receipt. Submit the comparison evidence
and discussion through the maintainer workflow in CONTRIBUTING.

**Local planning:** `bin/measure-local --artifact <hf-repo> --panel <hf-dataset>
--vram-budget 30 --estimate-only` plans without downloading weights.
Local execution supports only its declared `packed` and `native-bf16`
surfaces and emits preview-class evidence; planning another surface does not
make it executable through `--execute` or promote a preview to publication.

**Legacy teacher-logits submission:** the admitted cloud `--role quant`
route produces `<out>/receipts/measurement-receipt.json`, schema
`quant-fidelity-registry/submission-receipt.v1`. Submit that sealed file at
<https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
with title `submission: <repo> on <panel>`. Do not edit the sealed receipt.

A registered panel is not necessarily public or fetchable. Check the route's
descriptor/reference requirements before running. For a new registry panel,
follow CONTRIBUTING §6 rather than relabelling an existing panel.
