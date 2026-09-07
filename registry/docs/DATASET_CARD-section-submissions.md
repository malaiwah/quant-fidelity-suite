<!--
SPLICE TARGET: the HF dataset card for malaiwah/quant-fidelity-registry
(README.md at the dataset root, below the YAML front matter).
Insert as a top-level section near the end, after the collections description.
Keep it self-contained: on HF this is often the only page a contributor reads.
-->

## Submit a measurement

**Discussions are the primary channel.** No git, no fork, no CI.

1. Choose the route in the suite's
   [third-party quickstart](https://github.com/malaiwah/quant-fidelity-suite/blob/main/docs/THIRD-PARTY-QUICKSTART.md).
   The admitted cloud `--role quant` route writes a sealed registry submission
   at `<out>/receipts/measurement-receipt.json`. The cloud `--role candidate`
   route instead writes a fidelity dataset and
   `<out>/result/receipts/reference-comparison/comparison-receipt.json`;
   send that comparison evidence through the maintainer workflow rather than
   renaming it to a submission receipt. Local outputs may be **preview** class
   and are not automatically eligible for registry publication.

2. For the legacy submission receipt, verify the seal. Four lines, no dependencies:

   ```python
   import json, hashlib
   d = json.load(open("measurement-receipt.json")); claimed = d["receipt_sha256"]; d["receipt_sha256"] = ""
   canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
   print(hashlib.sha256(canon.encode()).hexdigest() == claimed)   # must print True
   ```

3. Open a **[new discussion](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions)**
   titled `submission: <repo> on <panel>`, and paste:

   ````markdown
   ### Submission

   - **Artifact:** 0xSero/GLM-5.3-Flash-EXL3-Q4 @ 99cccdf0e8741715662c383828a9ea601990c125
   - **Panel:** panel--glm53.brandonmusic.final25
   - **Reference:** reference--brandonmusic.glm53-bf16-fp32-logits.final25
   - **Metric:** mean_of_run_means_tokenwise_kld = 0.027262784814670614 nats
   - **Lane:** sealed-ep8
   - **I am:** the measurer / also the quant's author? → measurer only
   - **Anything odd about this run:** none

   <details><summary>measurement-receipt.json</summary>

   ```json
   { ...paste the whole file... }
   ```

   </details>
   ````

For an eligible legacy submission, that is the whole file we need. For a
candidate-route comparison, attach the comparison evidence instead. We validate
the evidence, generate rows, and reply with the row id, key and pair-predicate
verdict — or the failing check. A key alone is not a ranking certificate.

**You are credited by HF handle.** The measurer of a number and the producer of
a quant are separate fields and neither is transferable: if you measured someone
else's quant, you get the number and they keep the quant. Your discussion URL is
attached to the row as its source, so anyone can find your submission and argue
with it.

**Prefer a pull request?** The GitHub mirror at
[malaiwah/quant-fidelity-registry](https://github.com/malaiwah/quant-fidelity-registry)
is designed to take the same receipt as one file under
`receipts/<your-handle>/`, with CI that checks the seal, the schema and every
registry invariant before a human looks — but it is **not live yet** (the URL
404s today). Until it is, the discussion above is the one working path.

Full rules — mandatory fields, what gets bounced, how to register a new panel:
**[CONTRIBUTING.md](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/blob/main/CONTRIBUTING.md)**.
A real sealed example:
**[docs/examples/dione-q4.submission.json](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/blob/main/docs/examples/dione-q4.submission.json)**.
