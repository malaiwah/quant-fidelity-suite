# The BF16 floor — what a quant's KLD actually costs

**Measured floor: 0.011505922619330299 nats** (mean KLD(teacher / student) where the
"student" is the UNQUANTIZED BF16 model, full 25-window / 51,175-position
sealed panel, fp64, streaming lane, **2 cold runs producing identical means**,
`bitwise_deterministic: true`).

## Why this measurement exists

Our K6 and K8 quants score 0.013715 and 0.012384 on this panel — only **1.11x
apart** — while K8's shipped payload store is **13.2x tighter** than K6's in
weight-space NMSE. These are different estimands and need not track one another.
Scoring BF16 weights against the original teacher still measures 0.011506 nats
on this panel. The original teacher used brandonmusic's EP4 runtime, unlike the
replay lane. Different arithmetic/topology is a plausible contributor, not a
proved additive decomposition. "Floor" is the historical name of this control,
not a guaranteed lower bound on quantized KLD.

## Excess over control

(Renamed from "quantization-attributable error" on 2026-08-31, peer-review
P1-05: the difference `D(P‖Q_quant) − D(P‖Q_control)` is an estimate of the
excess divergence over the unquantized control, not a causal attribution — it
is not itself a divergence and can be negative. Even with only weights changed,
subtracting KL values does not isolate a teacher-independent quantization loss.)

| | panel KLD | minus floor | = excess over control |
|---|---:|---:|---:|
| BF16 (floor) | 0.011506 | — | 0 |
| K6 (6 bpw, 254 GB) | 0.013715 | -0.011506 | **0.002209** |
| K8 (8 bpw, 331 GB) | 0.012384 | -0.011506 | **0.000878** |

K8's descriptive excess is smaller than K6's: 0.000878 versus 0.002209 nats.
The common control subtraction changes neither the paired raw difference nor
its ordering. **The once-published residual ratio ("2.52x") is withdrawn**:
small residuals amplify control error, with no uncertainty attached here.
Read excess and raw values side by side, not as separable causes.

## How to use this (and how not to)

- The floor is a property of THIS panel + THIS teacher + THIS lane. It is not a
  universal constant. Re-measure it whenever any of those change.
- Subtracting a floor measured on a DIFFERENT lane is invalid. Our official-FP8
  figure (0.020615) was captured cross-stack, and its matching cross-stack floor
  is 0.012712 — so FP8's excess over THAT control is ~0.0079, never computed
  against this one.
- The subtraction is an exact descriptive contrast of recorded means, not an
  approximation to an additive KL law. Small magnitudes do not create additivity.
- Equal KL to the teacher for a quant and its BF16 control does not imply their
  distributions are equal or that an error is below the panel's resolution.
  A same-lane teacher changes the reference; it is not a causal correction.

## Cost, honestly

The measurement cost **$18.90** (1x H200 spot at $1.99/h, 9h21m) — but most of
that was BUILDING the native-BF16 student mode in stream_score.py, not
measuring. A repeat costs ~2 runs x ~3h = ~$12, and a single run ~$6.
Receipts: native-bf16-kld.json (2 cold runs, identical means) and
native-bf16-kld-run1only.json.
