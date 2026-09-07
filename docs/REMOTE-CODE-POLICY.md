# Measuring models whose modeling code ships in the repo

Approved by the maintainer, 2026-09-01.

**Implementation status, 2026-09-07:** this is an authorization policy, not a
shipped general remote-code loader. `hf_capture.py` resolves native
transformers configs/classes; it does not enable `trust_remote_code`.
Repository modeling code and `auto_map` alone therefore do not admit a target
to capture or the paid route. The requirements below must be implemented and
verified before enabling such execution.

## The situation this answers

Model vendors do not wait for `transformers`. At launch, an architecture is
served three ways that involve no transformers release at all: vLLM/SGLang
implement it natively in their own codebases; llama.cpp and MLX reimplement it
independently; and the repo itself ships `modeling_*.py` loaded via
`trust_remote_code=True`. Kimi-K3 is the live example: 1.5 TB, `kimi_k3`
absent from transformers 5.16.1, and a complete `modeling_kimi_k3.py` +
`auto_map` in the repo — loadable today by anyone willing to execute it.

Refusing all remote code forever would leave whole families unmeasurable for
weeks after launch, which is exactly when a fidelity root matters most (see
docs/RACE-MODE.md). Executing it casually would put arbitrary unaudited code in
the same process as our credentials and our claims.

## The policy

Remote code is acceptable **in the checkpoint lane only**, under four
conditions, all mandatory:

1. **Revision-pinned.** The model revision is a 40-hex commit, so the executed
   `.py` bytes are immutable and re-fetchable by anyone.
2. **Digested like our own code.** Every repo-shipped `.py` that can execute
   enters `harness.code_digests` with its sha256, alongside the suite's own
   estimator closure. The registry's whole claim is "we hashed what ran"; the
   origin of the code changes nothing about that obligation.
3. **Credential-isolated capture.** Fetch and capture must be separated, with
   no HF credential in the capture environment or accessible token files.
   Merely unsetting a variable or unlinking one file is not a sandbox:
   arbitrary code could still read other mounts, model bytes or credentials,
   communicate over the network, or alter results. The loader's eventual
   isolation boundary must cover those risks; this policy is not evidence
   that such isolation currently exists.
4. **Disclosed on every row.** A `remote_code` disclosure
   (`affects_comparability: true`) on every measurement it produces.

## Enforcement, not convention

**RC-001** (schema/invariants.json, enforced in `registry_validate.py`) refuses
a `remote_code` disclosure without a recorded harness and a
`remote_model_code` role. That is a necessary metadata gate, not proof that
every executable transitive file was hashed, that credentials were absent,
or that remote code ran safely. The complete closure and execution boundary
remain requirements of any future loader.

## What this does not change

- The **serving lane** is untouched — vLLM's native implementations are that
  lane's identity, not a shortcut.
- `hy_v4`-style repos that ship **no** modeling code remain blocked on a
  transformers release; there is nothing to pin or digest.
- The generation sanity probe runs regardless: remote code that loads but
  produces a degenerate distribution fails the capture, not the reader.
