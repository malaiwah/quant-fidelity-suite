# OnePassArchiveVerify — one streaming pass for archive verify + extract

## What changed

`bin/fidelity/resultsink.py` — `_verified_archive` and `extract_verified_archive`:

**Before (old code):** `_verified_archive` read the compressed tar ~9 times:
1. `_archive_source` → `sha256_file` (1 compressed pass)
2. `tarfile.open(mode="r:gz")` → gzip inflate + tar header walk (1 inflate pass)
3. `archive.extractfile(manifest)` → seek back to manifest member (partial re-inflate)
4. `archive.extractfile(job.json)` → seek back (partial re-inflate)
5. Per-member `archive.extractfile()` loop → seek back for each member (partial re-inflates)
6. `_archive_source` again → `sha256_file` (1 compressed pass, post-verify guard)

`extract_verified_archive` then did a **second full inflate** (tarfile.getmembers + extractfile loop) plus two more `_archive_source` sha256 calls = **3 more compressed passes + 1 more inflate**.

Total for extract: ~6 compressed reads + 2 full inflates of a 5 GB tar.

**After (new code):** `_verified_archive` reads the compressed file exactly **twice** (1 sha256 pass via `_archive_source`, 1 inflate pass) and inflates **once**. Each tar member is hashed and (optionally) written to the staging directory during the single inflate — no seeking back. The post-verify file-unchanged guard uses the hashing reader's running sha256 instead of re-reading the file.

`extract_verified_archive` passes `extract_to=destination` to `_verified_archive`, so verification and extraction happen in the **same single inflate pass**. Total for extract: 2 compressed reads (sha256 + inflate) + 1 inflate.

### Every existing check preserved (same refusal texts)

| Check | Error text (unchanged) |
|---|---|
| Transfer byte count mismatch | `"transferred archive byte count mismatch: expected %d, got %d"` |
| Transfer SHA-256 mismatch | `"transferred archive SHA-256 mismatch"` |
| Malformed expected SHA-256 | `"expected archive SHA-256 is malformed"` |
| Member safety limit | `"archive exceeds member safety limit"` |
| Duplicate member | `"duplicate archive member %s"` |
| Non-regular file | `"archive member %s is not a regular file"` |
| Negative size | `"archive member %s has negative size"` |
| Uncompressed-byte limit | `"archive exceeds uncompressed-byte safety limit"` |
| Retained member memory cap | `"retained archive member exceeds memory safety cap: %s"` |
| Retained metadata memory cap | `"retained archive metadata exceeds memory safety cap"` |
| Missing manifest | `"archive lacks %s"` |
| Cannot read manifest | `"cannot read %s"` |
| Manifest schema/seal invalid | `"result manifest schema or self-seal is invalid"` |
| Manifest files not array | `"result manifest files must be an array"` |
| Manifest entry not object | `"result manifest file entry is not an object"` |
| Duplicate/reserved manifest path | `"duplicate/reserved manifest path %s"` |
| Invalid size/SHA-256 in record | `"invalid size or SHA-256 for %s"` |
| Member set mismatch | `"archive member set mismatch (missing=%r extra=%r)"` |
| Missing job.json for caps | `"completed science archive lacks exact job.json"` |
| Cannot read job.json for caps | `"cannot read job.json for archive caps"` |
| job.json differs before caps | `"job.json differs before archive-cap enforcement"` |
| Member size mismatch | `"archive member size mismatch for %s"` |
| Cannot read member | `"cannot read archive member %s"` |
| Member digest mismatch | `"archive member digest mismatch for %s"` |
| Truncated/unreadable | `"result archive is truncated or unreadable: %s"` |
| Archive changed during verify | `"result archive changed during verification"` |
| Extraction dest exists | `"extraction destination already exists: %s"` |

### Atomic extraction guarantee

Extraction writes to a `tempfile.mkdtemp` staging dir, fsyncs, and `os.replace`s to the destination only after all checks pass. On any `ArchiveError` or `OSError/EOFError/tarfile.TarError`, the staging dir is `shutil.rmtree`'d — nothing partial left behind.

## Timing on the real 5 GB tar

Archive: `/home/mbelleau/code/fidelity-runs/glm52-root/result.tar.gz` (5.07 GB, X5570 CPU)

| | Old code | New code |
|---|---|---|
| Compressed reads | ~6 | 2 |
| Inflate passes | 2 | 1 |
| Wall time (extract_verified_archive) | 3669.7s (~61 min) | 1328.9s (~22 min) |
| Speedup | — | 2.76× (saved ~2341s / ~39 min) |

Both runs verified the same 5,067,535,143-byte archive (sha256
d7129ca1c034fbbc…) and extracted it successfully. Measured on an
Intel Xeon X5570 @ 2.93 GHz under normal load.

## Selftest rungs

### Existing rungs (all pass, 123 → 132 total)

```
T26: 132 passed, 0 failed
```

### New rungs (R95–R102b)

| Rung | What it tests | Parent (old code) | After (new code) |
|---|---|---|---|
| R95 | verify opens archive ≤ 2 times | **FAIL** (3 opens) | PASS |
| R96 | verify reads ≤ 2× compressed bytes | **FAIL** (3× bytes) | PASS |
| R97 | extract opens archive ≤ 2 times | **FAIL** (6 opens) | PASS |
| R98 | tampered member sha refused | PASS | PASS |
| R99 | missing manifest refused | PASS | PASS |
| R100 | truncated gzip refused | PASS | PASS |
| R101 | extra member refused | PASS | PASS |
| R102 | tampered archive refuses extraction | PASS | PASS |
| R102b | refused extraction leaves no partial dir | PASS | PASS |

### Parent red / after green proof

Copied the new `selftest_result_sink.py` into a worktree at `origin/main` (old `resultsink.py`):

```
FAIL  R95 one-pass verify opens the archive file at most twice  -- opened 3 times
FAIL  R96 one-pass verify reads <= 2x compressed bytes (sha + inflate)  -- 276564 bytes read vs 92188 archive bytes
FAIL  R97 one-pass extract opens the archive at most twice (sha + single inflate, no re-read for extraction)  -- opened 6 times
T26: 129 passed, 3 failed
```

After the fix (same test file, new `resultsink.py`):

```
PASS  R95 one-pass verify opens the archive file at most twice
PASS  R96 one-pass verify reads <= 2x compressed bytes (sha + inflate)
PASS  R97 one-pass extract opens the archive at most twice (sha + single inflate, no re-read for extraction)
T26: 132 passed, 0 failed
```

## Other suites

- `bin/selftest_contract_harness.py`: **25 passed, 0 failed** (unchanged)
- `bin/selftest_stage_measure.py`: **all passed** (1 skipped, unchanged)

## Branch and commit

- Branch: `OnePassArchiveVerify`
- Commit: `4514d5d` — `resultsink: one streaming pass for archive verify + extract` (rebased on 3fc1336)
- Files changed: `bin/fidelity/resultsink.py`, `bin/selftest_result_sink.py`
