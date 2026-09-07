#!/usr/bin/env python3
"""Secret files are 0600 from the first instant, at both ends, and cleaned up.

WHY THIS EXISTS
---------------
Peer review 2026-08-31, security chapter (High): the controller wrote the
local token file and only THEN chmod'd it to 0600 -- on a permissive umask
that is a window another local user can read, and the project's own
concurrency test once captured a full token through it.  The remote path
created `.secrets`, uploaded the token, and tightened modes only afterwards.
The container entrypoint opened the destination with a plain `open("w")`
(follows a planted symlink on the persistent bind mount) and never removed
the token when the run ended.

  S1  write_secret_file: 0600 file inside a 0700 directory, atomically.
  S2  a pre-existing loose-mode file is replaced, not inherited.
  S3  a planted symlink is removed, never written through.
  S4  shred_secret_file removes the file; a missing file is a no-op.
  S5  remote transport ORDER: refuse a pre-existing secret directory, create
      it as 0700 BEFORE upload, upload uniquely, chmod 600, atomic rename.
  S6  the token value never appears on any remote command line.
  S7  the local copy is shredded even when the upload raises.
  S8  container write_token uses the same exclusive/no-follow creation.
  S9  the container entrypoint removes the token in a finally -- a FAILED
      stage still leaves no token behind on the bind mount.

No network, no provider, no real token.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity.common import shred_secret_file, write_secret_file  # noqa: E402

FAILED = []
TOKEN = "hf_selftest_not_a_real_token_11111"


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:8]:
            print("        %s" % line)


def mode(path):
    return oct(os.lstat(str(path)).st_mode & 0o777)


def load_measure_cloud():
    spec = importlib.util.spec_from_file_location(
        "measure_cloud", str(ROOT / "bin" / "measure_cloud.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubJL:
    """Execute transport shell commands locally; never contact a provider."""

    def __init__(self, fail_upload=False, events=None):
        self.ops = []
        self.fail_upload = fail_upload
        self.events = events if events is not None else []
        self.upload_modes = []

    def exec(self, machine_id, command, **kw):
        self.ops.append(("exec", command))
        self.events.append(("exec", command))
        result = subprocess.run(
            ["sh", "-c", command], check=True, capture_output=True, text=True)
        return {"exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}

    def upload(self, machine_id, local, remote):
        self.ops.append(("upload", local, remote))
        self.events.append(("upload", remote))
        if self.fail_upload:
            raise RuntimeError("upload failed (stub)")
        self.upload_modes.append((
            mode(local), mode(Path(local).parent), mode(Path(remote).parent),
            not (Path(remote).parent / "hf_token").exists()))
        shutil.copyfile(local, remote)
        return {"ok": True}


class StubTD:
    fs_root = "/fs"
    machine_id = 7


def main():
    old_umask = os.umask(0o000)   # the hostile umask the review names
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            StubTD.fs_root = str(td / "remote")

            # S1
            dest = td / "sec" / "hf_token"
            write_secret_file(str(dest), TOKEN)
            check("S1 file is 0600 in a 0700 directory (umask 000)",
                  mode(dest) == "0o600" and mode(td / "sec") == "0o700",
                  "file=%s dir=%s" % (mode(dest), mode(td / "sec")))
            check("S1b content intact", dest.read_text() == TOKEN)

            # S2
            loose = td / "sec" / "loose"
            loose.write_text("old")
            os.chmod(str(loose), 0o644)
            write_secret_file(str(loose), TOKEN)
            check("S2 pre-existing 0644 file replaced with a fresh 0600 one",
                  mode(loose) == "0o600" and loose.read_text() == TOKEN)

            # S3
            victim = td / "victim"
            victim.write_text("victim-bytes")
            link = td / "sec" / "planted"
            link.symlink_to(victim)
            write_secret_file(str(link), TOKEN)
            check("S3 planted symlink is removed, never written through",
                  victim.read_text() == "victim-bytes"
                  and not link.is_symlink() and mode(link) == "0o600")

            # S4
            shred_secret_file(str(loose))

            # S4c: paid RunPod accepts only an explicit owned 0600 token file.
            mc = load_measure_cloud()
            download_token = td / "download-token"
            write_secret_file(str(download_token), TOKEN)
            check("S4c required download token reads the exact 0600 file",
                  mc._load_required_hf_download_token(
                      str(download_token)) == TOKEN)
            download_token.chmod(0o644)
            bad_mode_refused = False
            try:
                mc._load_required_hf_download_token(str(download_token))
            except mc.Refusal:
                bad_mode_refused = True
            missing_refused = False
            try:
                mc._load_required_hf_download_token(
                    str(td / "missing-download-token"))
            except mc.Refusal:
                missing_refused = True
            check("S4d loose or missing download token refuses before transport",
                  bad_mode_refused and missing_refused)

            # Omitted or duplicated publication credentials refuse before any
            # remote operation; a separate explicit read credential can fetch.
            publication_token = td / "publish-token"
            write_secret_file(str(publication_token), TOKEN)
            copied_token = td / "copied-publish-token"
            write_secret_file(str(copied_token), TOKEN)
            credential_args = mc.build_parser().parse_args([
                "--provider", "runpod", "--role", "root",
                "--hf-token-file", str(publication_token),
                "--publish-root-to", "selftest/new-root"])
            mc._apply_runpod_defaults(credential_args)
            check("S4e publication token is not a pod download default",
                  credential_args.hf_download_token_file is None)
            for token_path in (None, str(publication_token), str(copied_token)):
                credential_args.hf_download_token_file = token_path
                provider = StubJL()
                refused = False
                try:
                    remote_token = mc._load_runpod_download_token(credential_args)
                    mc._transport_hf_token(
                        provider, 7, str(td / "must-not-upload"),
                        td / "refused-upload", remote_token)
                except mc.Refusal as exc:
                    refused = TOKEN not in str(exc)
                check("S4f omitted/reused publication credential never reaches pod",
                      refused and not provider.ops)
            read_token = td / "read-token"
            read_value = "hf_selftest_distinct_read_credential_22222"
            write_secret_file(str(read_token), read_value)
            credential_args.hf_download_token_file = str(read_token)
            provider = StubJL()
            mc._transport_hf_token(
                provider, 7, str(td / "explicit-read"), td / "read-upload",
                mc._load_runpod_download_token(credential_args))
            check("S4g only the explicit separate credential reaches the pod",
                  (td / "explicit-read/.secrets/hf_token").read_text() == read_value
                  and publication_token.read_text() == TOKEN)

            check("S4 shred removes the file", not loose.exists())
            shred_secret_file(str(td / "never-existed"))

            # S5/S6: run transport commands against a local temporary filesystem.
            jl = StubJL()
            outdir = td / "out"
            outdir.mkdir()
            mc._transport_hf_token(
                jl, StubTD.machine_id, StubTD.fs_root, outdir, TOKEN)
            remote_token_path = Path(StubTD.fs_root) / ".secrets/hf_token"
            check("S5a upload starts with private staging and remote directory",
                  jl.upload_modes == [("0o600", "0o700", "0o700", True)])
            check("S5b transport leaves only the private complete token",
                  remote_token_path.read_text() == TOKEN
                  and mode(remote_token_path) == "0o600"
                  and list(remote_token_path.parent.iterdir()) == [remote_token_path])
            blocked = StubJL()
            try:
                mc._transport_hf_token(
                    blocked, 7, StubTD.fs_root, td / "occupied", TOKEN)
            except RuntimeError:
                pass
            else:
                raise AssertionError("occupied remote secret directory accepted")
            check("S5c occupied secret directory refuses without overwrite",
                  not blocked.upload_modes and remote_token_path.read_text() == TOKEN)
            check("S5e the local staging copy is gone afterwards",
                  not (outdir / ".secrets-local" / "hf_token").exists())
            check("S5f the local staging directory was 0700",
                  mode(outdir / ".secrets-local") == "0o700")
            check("S6 the token value never appears on a remote command line",
                  all(TOKEN not in " ".join(str(part) for part in op)
                      for op in jl.ops), jl.ops)

            # S7: shredded even when the upload raises.
            jl2 = StubJL(fail_upload=True)
            outdir2 = td / "out2"
            outdir2.mkdir()
            raised = False
            try:
                mc._transport_hf_token(
                    jl2, StubTD.machine_id, str(td / "upload-failure"), outdir2, TOKEN)
            except RuntimeError:
                raised = True
            check("S7 a failed upload still shreds the local copy (and "
                  "propagates)",
                  raised and not (outdir2 / ".secrets-local" / "hf_token").exists())

            # S7a/S7b: the RunPod target-fetch wrapper scopes credential use
            # to that one stage and cleans up on both success and failure.
            events = []
            runpod = StubJL(events=events)
            original_stage = mc._runpod_stage
            runpod_root = str(td / "runpod")
            secrets_dir = "%s/.secrets" % runpod_root
            fetch_credentials = []
            try:
                def fetch_stage(*_args, **_kwargs):
                    fetch_credentials.append(
                        (Path(secrets_dir) / "hf_token").read_text())
                mc._runpod_stage = fetch_stage
                mc._transport_hf_token(
                    runpod, StubTD.machine_id, runpod_root,
                    td / "runpod-success", TOKEN, secrets_dir=secrets_dir)
                cleanup = mc._paid_fetch_target_and_remove_token(
                    runpod, StubTD.machine_id, runpod_root, "/engine",
                    1.0, "image@sha256:" + "a" * 64, secrets_dir)
                check("S7a RunPod authenticates only fetch_target, then confirms "
                      "remote cleanup",
                      cleanup.get("confirmed") is True
                      and fetch_credentials == [TOKEN]
                      and not Path(secrets_dir).exists())

                events.clear()
                def fail_stage(*_args, **_kwargs):
                    fetch_stage()
                    raise RuntimeError("fetch failed")
                mc._runpod_stage = fail_stage
                failed = False
                try:
                    mc._transport_hf_token(
                        runpod, StubTD.machine_id, runpod_root,
                        td / "runpod-failure", TOKEN,
                        secrets_dir=secrets_dir)
                    mc._paid_fetch_target_and_remove_token(
                        runpod, StubTD.machine_id, runpod_root, "/engine",
                        1.0, "image@sha256:" + "a" * 64, secrets_dir)
                except RuntimeError as exc:
                    failed = str(exc) == "fetch failed"
                check("S7b failed RunPod fetch still removes the remote token",
                      failed and fetch_credentials == [TOKEN, TOKEN]
                      and not Path(secrets_dir).exists())
            finally:
                mc._runpod_stage = original_stage

            # S8: the container entrypoint's writer.
            spec = importlib.util.spec_from_file_location(
                "container_entry", str(ROOT / "bin" / "container_entry.py"))
            CE = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(CE)
            fs = td / "fs"
            fs.mkdir()
            victim2 = td / "victim2"
            victim2.write_text("victim-bytes")
            (fs / ".secrets").mkdir(mode=0o700)
            (fs / ".secrets" / "hf_token").symlink_to(victim2)
            src = td / "tokfile"
            src.write_text(TOKEN + "\n")
            wrote = CE.write_token(fs, str(src), lambda *_a, **_k: None)
            tokpath = fs / ".secrets" / "hf_token"
            check("S8 container write_token: symlink removed, 0600, 0700 dir, "
                  "no write-through",
                  wrote and victim2.read_text() == "victim-bytes"
                  and not tokpath.is_symlink() and mode(tokpath) == "0o600"
                  and mode(fs / ".secrets") == "0o700")

            # S9: drive the entrypoint's stage lifecycle with its science
            # validators stubbed at this unit boundary. Other selftests own the
            # strict job.v2 contract; this case owns the token-finally invariant.
            fs9 = td / "fs9"
            fs9.mkdir()
            fs9.joinpath("job.json").write_text("{}\n", encoding="utf-8")
            os.environ["HF_TOKEN"] = TOKEN
            os.environ.pop("FIDELITY_SUITE_ROOT", None)
            try:
                CE._prevalidate_stage_job = lambda *_a, **_k: {
                    "role": "quant",
                    "capture": {},
                    "target": {"surface": "native-bf16"},
                }
                CE.sync_suite = lambda *_a, **_k: 0
                CE.validate_job_document = lambda *_a, **_k: None
                CE.require_accelerator = lambda *_a, **_k: None
                CE.run_stage = lambda *_a, **_k: 1
                CE.RS.parse_sinks = lambda *_a, **_k: []
                CE.RS.build_summary = lambda *_a, **_k: {}
                CE.RS.deliver = lambda *_a, **_k: []
                rc = CE.main(["stage", "fetch_target", "--fs-root", str(fs9),
                              "--engine-root", str(td / "noengine")])
                token_left = (fs9 / ".secrets" / "hf_token").exists()
                check("S9 a failed stage leaves no token on the run root "
                      "(rc=%s)" % rc,
                      rc == CE.EXIT_FAILED and not token_left)
            finally:
                os.environ.pop("HF_TOKEN", None)
    finally:
        os.umask(old_umask)

    print()
    if FAILED:
        print("selftest_secret_files: %d FAILED" % len(FAILED))
        return 1
    print("selftest_secret_files: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
