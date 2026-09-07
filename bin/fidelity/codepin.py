"""Opt-in immutable Hub Python bundles; importing this module is stdlib-only.

This is provenance verification, NOT a sandbox or a source-code safety review.
The caller explicitly trusts the selected repository/commit. No local manifest
can substitute for the Hub commit's authenticated tree and blob identities.
The auto_map entry modules, package initializers and recursively discovered
relative imports form the closure (including imports inside functions/branches).
Dynamic imports outside that closure refuse. Only Python sources are downloaded,
except authenticated config.json mapping metadata for an explicit runtime fork
when the original model config has no auto_map. Fork dimensions are never used.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import types
from typing import Any

MAX_FILES = 256
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")


class CodePinError(RuntimeError):
    """A requested runtime cannot be bound to one verified immutable bundle."""


def raw_config(model_dir: str) -> dict[str, Any]:
    with open(os.path.join(model_dir, "config.json"), "rb") as source:
        payload = source.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise CodePinError("REFUSED: config.json exceeds the 2 MiB code-pin limit")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise CodePinError("REFUSED: config.json must be a JSON object")
    return value


def validate_options(trust_remote_code: bool, revision: str | None,
                     repository: str | None) -> None:
    if not trust_remote_code:
        if revision is not None or repository is not None:
            raise CodePinError("REFUSED: --code-revision/--code-repository require --trust-remote-code")
        return
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise CodePinError("REFUSED: --trust-remote-code requires --code-revision with a full 40-hex commit SHA")
    if repository is not None and not REPOSITORY.fullmatch(repository):
        raise CodePinError("REFUSED: --code-repository must be an exact owner/repo, not a path or URL")


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value.endswith(".py") or "\\" in value
            or str(path) != value or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
            or any(not part.isidentifier() for part in path.parts[:-1])
            or not path.stem.isidentifier()):
        raise CodePinError("REFUSED: unsafe Python code path %r" % value)
    return value


def _reference(value: str) -> tuple[str | None, str]:
    if not isinstance(value, str):
        raise CodePinError("REFUSED: auto_map class references must be strings")
    repository, separator, reference = value.partition("--")
    if not separator:
        reference, repository = value, None
    elif not REPOSITORY.fullmatch(repository):
        raise CodePinError("REFUSED: invalid auto_map code repository %r" % repository)
    parts = reference.split(".")
    if len(parts) < 2 or not all(part.isidentifier() for part in parts):
        raise CodePinError("REFUSED: unsafe auto_map class reference %r" % value)
    return repository, reference


class _VerifiedLoader(importlib.abc.Loader):
    def __init__(self, bundle, path):
        self.bundle, self.path = bundle, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Compile exactly the verified bytes, never a possibly stale .pyc file.
        payload = self.bundle.read_verified(self.path)
        filename = str(self.bundle.root / self.path)
        module.__file__ = filename
        exec(compile(payload, filename, "exec"), module.__dict__)


class _VerifiedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, bundle):
        self.bundle = bundle

    def find_spec(self, fullname, path=None, target=None):
        prefix = self.bundle.namespace + "."
        if not fullname.startswith(prefix):
            return None
        relative = fullname[len(prefix):].replace(".", "/")
        for candidate, package in ((relative + "/__init__.py", True), (relative + ".py", False)):
            if candidate in self.bundle.files:
                spec = importlib.util.spec_from_loader(
                    fullname, _VerifiedLoader(self.bundle, candidate), is_package=package)
                spec.origin = str(self.bundle.root / candidate)
                if package:
                    spec.submodule_search_locations = [str(self.bundle.root / relative)]
                return spec
        if any(name.startswith(relative + "/") for name in self.bundle.files):
            spec = importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(self.bundle.root / relative)]
            return spec
        # Never fall through to the filesystem for a missing bundle dependency.
        raise CodePinError("REFUSED: import outside verified Python closure: " + fullname)


class VerifiedCode:
    def __init__(self, repository, revision, payloads, identities, references,
                 mapping_source=None):
        self.repository, self.revision = repository, revision
        self.references = references
        self.files = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
        self.identities = identities
        self.mapping_source = mapping_source
        identity = {"repository": repository, "revision": revision, "files": self.files}
        if mapping_source is not None:
            identity["mapping_source"] = mapping_source
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.closure_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
        self._temporary = tempfile.TemporaryDirectory(prefix="qfs-verified-code-")
        self.root = Path(self._temporary.name).resolve()
        for name, data in payloads.items():
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o400)
        self.namespace = "qfs_verified_" + self.closure_sha256
        self.finder = _VerifiedFinder(self)
        self.resolved_classes = {}
        self._installed = False

    def read_verified(self, name):
        path = self.root / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise CodePinError("REFUSED: verified code path was replaced: " + name)
        with path.open("rb") as source:
            payload = source.read(MAX_FILE_BYTES + 1)
        if hashlib.sha256(payload).hexdigest() != self.files[name]:
            raise CodePinError("REFUSED: verified code bytes changed: " + name)
        return payload

    def _install(self):
        if self._installed:
            return
        if self.namespace in sys.modules:
            # Reusing another invocation's Python module cache would make this
            # invocation's verified directory a cosmetic rather than real pin.
            raise CodePinError("REFUSED: pinned code namespace already loaded; use a fresh capture process")
        for name in self.files:
            self.read_verified(name)
        sys.meta_path.insert(0, self.finder)
        package = types.ModuleType(self.namespace)
        package.__path__ = [str(self.root)]
        package.__package__ = self.namespace
        package.__spec__ = importlib.util.spec_from_loader(self.namespace, loader=None, is_package=True)
        sys.modules[self.namespace] = package
        self._installed = True
        if "__init__.py" in self.files:
            _VerifiedLoader(self, "__init__.py").exec_module(package)

    def load_class(self, reference: str):
        self._install()
        _, reference = _reference(reference)
        module_name, class_name = reference.rsplit(".", 1)
        module = importlib.import_module(self.namespace + "." + module_name)
        cls = getattr(module, class_name)
        if not isinstance(cls, type) or not cls.__module__.startswith(self.namespace + "."):
            raise CodePinError("REFUSED: auto_map did not resolve to a class defined by the verified bundle: " + reference)
        self.resolved_classes[reference] = cls.__module__ + "." + cls.__qualname__
        self.verify_imports()
        return cls

    def verify_imports(self):
        for name in self.files:
            self.read_verified(name)
        imported = {}
        for name, module in list(sys.modules.items()):
            if name != self.namespace and not name.startswith(self.namespace + "."):
                continue
            filename = getattr(module, "__file__", None)
            if filename is None:
                if getattr(module, "__path__", None) is None:
                    raise CodePinError("REFUSED: code module has no verifiable source: " + name)
                continue
            try:
                relative = Path(filename).relative_to(self.root).as_posix()
            except ValueError as exc:
                raise CodePinError("REFUSED: imported code escaped verified directory: " + name) from exc
            if relative not in self.files:
                raise CodePinError("REFUSED: imported source outside verified closure: " + relative)
            self.read_verified(relative)
            imported[relative] = self.files[relative]
        return imported

    def evidence(self):
        return {"schema": "malaiwah.verified-code.v1", "repository": self.repository,
                "revision": self.revision, "closure_sha256": self.closure_sha256,
                "closure_policy": "auto-map-static-relative-imports-and-package-initializers",
                "files": [{"path": name, "sha256": self.files[name], **self.identities[name]}
                          for name in sorted(self.files)],
                "imported_files": self.verify_imports(),
                "resolved_classes": dict(self.resolved_classes),
                "mapping_source": self.mapping_source,
                "verification": "Hub immutable tree plus Git blob/LFS SHA-256; private verified source loader",
                "security_scope": "explicitly trusted code; provenance verification is not a sandbox"}


def prepare(raw: dict[str, Any], revision: str, repository: str | None = None) -> VerifiedCode:
    validate_options(True, revision, repository)
    auto_map = raw.get("auto_map") or {}
    if not isinstance(auto_map, dict):
        raise CodePinError("REFUSED: auto_map must be an object")
    references = {}
    foreign = set()
    for key, value in auto_map.items():
        # Tokenizer entries may be [slow, fast]; bind their repository provenance
        # too, although this model loader never executes custom tokenizers.
        for entry in value if isinstance(value, (list, tuple)) else [value]:
            if entry is None:
                continue
            repo, reference = _reference(entry)
            if repo:
                foreign.add(repo)
            if isinstance(value, str):
                references[key] = reference
    if len(foreign) > 1 and repository is None:
        raise CodePinError("REFUSED: mixed foreign auto_map repositories require --code-repository naming one consolidated bundle")
    repository = repository or (next(iter(foreign)) if foreign else None)
    if repository is None:
        raise CodePinError("REFUSED: local custom-code weights require --code-repository owner/repo")
    # A runtime fork can supply only the missing dispatch metadata. Its config
    # values never replace the original checkpoint's config values.

    # Dependency imported ONLY after opt-in and all flag/reference checks.
    from huggingface_hub import HfApi, hf_hub_download

    revision = revision.lower()
    api = HfApi()
    info = api.model_info(repository, revision=revision)
    if info.sha != revision:
        raise CodePinError("REFUSED: Hub did not resolve the exact requested code commit")
    entries = {entry.path: entry for entry in api.list_repo_tree(
        repository, revision=revision, recursive=True)
        if hasattr(entry, "blob_id") and (entry.path.endswith(".py") or entry.path == "config.json")}

    def fetch_verified(name):
        entry = entries[name]
        if not isinstance(entry.size, int) or entry.size < 0 or entry.size > MAX_FILE_BYTES:
            raise CodePinError("REFUSED: code/metadata file exceeds 2 MiB limit: " + name)
        cached = hf_hub_download(repository, name, revision=revision)
        with open(cached, "rb") as source:
            payload = source.read(MAX_FILE_BYTES + 1)
        if len(payload) != entry.size:
            raise CodePinError("REFUSED: Hub code size mismatch: " + name)
        lfs = getattr(entry, "lfs", None)
        if lfs:
            expected = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            if not expected or hashlib.sha256(payload).hexdigest() != expected:
                raise CodePinError("REFUSED: Hub LFS code digest mismatch: " + name)
        else:
            digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if digest != entry.blob_id:
                raise CodePinError("REFUSED: Hub Git code blob mismatch: " + name)
        return payload

    mapping_source = None
    if not auto_map:
        if "config.json" not in entries:
            raise CodePinError("REFUSED: explicit runtime repository has no pinned config.json auto_map")
        payload = fetch_verified("config.json")
        mapping_config = json.loads(payload)
        auto_map = mapping_config.get("auto_map") if isinstance(mapping_config, dict) else None
        if not isinstance(auto_map, dict):
            raise CodePinError("REFUSED: pinned runtime config.json must supply an auto_map object")
        # All references explicitly resolve inside THIS selected bundle, never
        # execute a second foreign repository named by the fork's metadata.
        for key, value in auto_map.items():
            for entry in value if isinstance(value, (list, tuple)) else [value]:
                if entry is not None:
                    _, reference = _reference(entry)
                    if isinstance(value, str):
                        references[key] = reference
        entry = entries["config.json"]
        mapping_source = {"path": "config.json", "sha256": hashlib.sha256(payload).hexdigest(),
                          "git_blob_sha1": entry.blob_id, "size": entry.size,
                          "use": "auto_map only; original model configuration unchanged"}
    if not any(key in references for key in ("AutoModelForCausalLM", "AutoModelForImageTextToText")):
        raise CodePinError("REFUSED: pinned code requires auto_map AutoModelForCausalLM or AutoModelForImageTextToText")
    entries.pop("config.json", None)
    payloads, identities, pending = {}, {}, set()

    def add_module(parts, required=True):
        prefix = "/".join(parts)
        candidates = (prefix + "/__init__.py", prefix + ".py") if parts else ("__init__.py",)
        found = next((name for name in candidates if name in entries), None)
        namespace = any(name.startswith(prefix + "/") for name in entries) if parts else True
        if found:
            pending.add(found)
        elif required and not namespace:
            raise CodePinError("REFUSED: relative dependency missing from code commit: " + ".".join(parts))
        # Parent __init__ files execute before their children and belong to the
        # same verified closure even if the entry never mentions them directly.
        for index in range(len(parts)):
            initializer = "/".join(parts[:index] + ["__init__.py"])
            if initializer in entries:
                pending.add(initializer)

    for key in ("AutoConfig", "AutoModelForCausalLM", "AutoModelForImageTextToText"):
        if key in references:
            add_module(references[key].rsplit(".", 1)[0].split("."))
    local_roots = {name.split("/")[0].removesuffix(".py") for name in entries}
    total = mapping_source["size"] if mapping_source else 0
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in payloads:
            continue
        _safe_path(name)
        entry = entries[name]
        if not isinstance(entry.size, int) or entry.size < 0 or entry.size > MAX_FILE_BYTES:
            raise CodePinError("REFUSED: code file exceeds 2 MiB limit: " + name)
        total += entry.size
        if len(payloads) + int(mapping_source is not None) >= MAX_FILES or total > MAX_TOTAL_BYTES:
            raise CodePinError("REFUSED: code bundle exceeds 256 files / 16 MiB limits")
        payload = fetch_verified(name)
        tree = ast.parse(payload, filename=name)  # no execution
        payloads[name] = payload
        identities[name] = {"git_blob_sha1": entry.blob_id, "size": entry.size}
        for node in ast.walk(tree):
            absolute = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                        else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            if any(value and value.split(".")[0] in local_roots for value in absolute):
                raise CodePinError("REFUSED: bundle-local imports must be relative: " + name)
            if isinstance(node, ast.ImportFrom) and node.level:
                parent = name.split("/")[:-1]
                if node.level > len(parent) + 1:
                    raise CodePinError("REFUSED: relative import escapes code bundle: " + name)
                base = parent[:len(parent) - node.level + 1]
                target = base + (node.module.split(".") if node.module else [])
                add_module(target)
                for alias in node.names:
                    if alias.name != "*":
                        # `from .package import module` also executes module;
                        # an exported class/constant need not be a source file.
                        initializer = "/".join(target + ["__init__.py"])
                        required = node.module is None and initializer not in entries
                        add_module(target + [alias.name], required=required)
    if not payloads:
        raise CodePinError("REFUSED: code closure contains no Python sources")
    return VerifiedCode(repository, revision, payloads, identities, references, mapping_source)
