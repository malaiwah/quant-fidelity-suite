# When a family publishes no unquantized weights

**Historical scope, clarified 2026-09-07.** Repository counts, availability
and DeepSeek release descriptions below describe the inspected snapshot,
not a current Hub census. The designation rule remains reference-relative;
its former directional KL claim is withdrawn in §"What the number means".

Every measurement in this registry is a distance **from** something. Until now
that something was always the model's own unquantized release, and a family
that publishes none was simply unmeasurable: Stage A closed
`deepseek-ai/DeepSeek-V4-Flash-0731` as NO-GO for exactly this reason, with a
working engine and no anchor. **Every** `deepseek_v4` repo on the Hub ships a
`quantization_config` — the published root *is* FP8-E4M3 attention plus
FP4-E2M1 experts — and `mlx-community/DeepSeek-V4-Flash-bf16` is a 4-bit MLX
repo wearing a misleading name.

That is 100 quantized children and 4.58M downloads with no yardstick, because
of a missing artifact nobody is going to publish.

## The rule

> When no unquantized release of a model exists, the published artifact with
> the **most bits — actual, never extrapolated** — is DESIGNATED as that
> family's reference.

Three things it does **not** change, and one it does:

* It does not change any **capture**. The bytes read off a checkpoint are the
  same bytes either way.
* It does not change any **fidelity dataset**. A dataset records what a model
  computed on a panel; the designation is about what other numbers are compared
  against, which is a later and separate step.
* It does not change any **artifact record**. The proxy is still described as
  exactly the quantized thing it is.
* It changes only the **measures** — what the divergences are distances from.

And it is **superseded, never invalidated**. If true unquantized weights are
released later, rows measured against them carry a new `reference_id` and
therefore land in a new comparability group. The proxy-referenced rows remain
correct statements about what they actually measured.

## Why this is safe, and how that is enforced rather than promised

`COMPARABILITY_KEY_FIELDS` binds `reference_id`. So a row against a designated
proxy **cannot** share a comparability group with a row against unquantized
weights — not by convention, but because the key is a hash over inputs that
include the reference identity. Demonstrated rather than asserted:

```
BF16-referenced key : cmp--209383798edd8dc2
proxy-referenced key: cmp--092d2cc380b22bab
```

Identical in every other field. The registry could not rank them together if
it wanted to.

What the key cannot do is stop a **reader** from quoting a proxy-referenced
number as though it were divergence from the model. So **REFC-006** requires,
as errors:

1. `reference_kind = quantized_proxy` → the referenced artifact must have
   `kind` in `{quant, requantized}`. A proxy pointing at a base artifact is a
   `native_*` reference wearing the wrong label, and would escape (2) and (3).
2. Every measurement against it carries a `different_reference_kind`
   disclosure with `affects_comparability: true`.
3. Every such measurement is `comparability.class = advisory`. **A designated
   proxy is not a measured floor.**

Three selftest cases prove each clause fires:
`proxy-reference-undisclosed`, `proxy-reference-marked-strict`,
`proxy-reference-on-base-artifact`.

## What the number means, stated plainly

A row against a designated proxy answers *"how far is this candidate from the
chosen published reference?"* — not *"how far is it from an unavailable
unquantized model?"*. Highest declared precision is a selection rule, not a
proof of maximal fidelity.

**Correction, 2026-09-07.** The earlier "systematically smaller" claim,
including the quoted historical schema wording for `dequantized_from_quant`,
is withdrawn. KL has no triangle inequality or monotonicity under a change of
reference: a proxy-referenced KL can be larger or smaller than a true-teacher
KL. Its bias direction is unknown without a controlled measurement.
The reference self-compare's 0.0 is an origin by construction, not a measured
floor or evidence that the quantized reference loses nothing.

`quantized_proxy` is the sibling case where the quantized artifact is used
**directly** as the teacher rather than dequantized first. It was in the enum
and had no invariant; REFC-006 is that invariant.

## Choosing the proxy

"Most bits, actual, never extrapolated" means: read the rates the release
itself publishes, as `--scope-json` already requires and as the plan-time gate
already verifies against the artifact's own `quantization_config`. A nominal
bpw in a repo name is not evidence. Where two candidates are close, prefer the
one whose scope is **fully read** over one with `unknown` classes — an
unmeasurable recipe makes a poor origin.

For the inspected `deepseek_v4` family, the official quantized root is the
designated origin by this rule. The recorded 3,176/3,176 tensor reconstruction
and zero self-compare establish their stated tested scopes, not maximal
fidelity, native full-forward parity, or the present availability of every
Hub repository. A new unquantized release would require a new reference id.
