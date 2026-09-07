# exllamav3 glm5_next port — design bundle (pre-implementation)

Design + first-draft for running GLM-5.3-Flash natively on exllamav3 (target:
first K6 trellis quant, scored on the fidelity suite in this repo). Produced
2026-08-27 by a 7-agent workflow against exllamav3 v1.4.4, before any
implementation session. Status: **not yet implemented or GPU-tested** — this is
the blueprint the K6 session starts from.

The design reuses DeepSeek-V4 mHC, `glm_moe_dsa` MLA/indexer/MoE structure
and GDN cache machinery. Source-level similarity is not executed native parity.
Required modules and cache/kernel integration remain unimplemented in this bundle.

**2026-09-07 qualification:** the executable harness now fails native construction,
load/forward errors and missing requested coverage. CPU/reference-only operation
requires `--ref-only` and prints **UNQUALIFIED / NON-NATIVE**. mHC loaded values
must equal independently read checkpoint tensors before numerical comparisons.
The draft probes required installed FLA API parameters instead of a guessed
version floor. Historical embedded source in the design notes is not the current
executable; use `tests/glm5_layer_parity.py` and `glm5_next.py.draft`.
Even all requested layer rows passing would not qualify whole-model serving,
cache rewind, complete long-context behavior or native quant reconstruction.

| File | What it is |
|---|---|
| [BLUEPRINT.md](BLUEPRINT.md) | Full port blueprint: file-by-file plan, config asserts, tensor-key mapping, what to keep unquantized, hardest-3 risks |
| [glm5_next.py.draft](glm5_next.py.draft) | Syntax-checked first draft of `exllamav3/architecture/glm5_next.py` (PORT-CHECK tags mark unresolved deps) |
| [DRAFT-NOTES.md](DRAFT-NOTES.md) | What the draft assumes and which plan items must land first |
| [tests/glm5_layer_parity.py](tests/glm5_layer_parity.py) | Per-layer torch-oracle parity harness (KDA / NoPE-MLA / noaux_tc MoE / mHC), smoke-tested on a synthetic mini-checkpoint |
| [tests/mini_ckpt_test.py](tests/mini_ckpt_test.py) | Builds the synthetic mini-checkpoint the harness self-checks against |
| [PARITY.md](PARITY.md) | Harness design + measured self-check numbers |
| [REVIEW.md](REVIEW.md) | Historical adversarial review, with dated resolution notes for harness refusal and FLA feature checks; missing modules remain prerequisites |

Context: brandonmusic's 4bpw EXL3 (custom Transformers TP2 adapter +
exllamav3 kernels) proves the quant path works today without this native port;
this bundle is the road to a self-contained exllamav3 architecture and the
K6/K5 variants. See JOURNAL.md at repo root for the campaign log.
