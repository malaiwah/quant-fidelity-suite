#!/usr/bin/env python3
"""An annotation must name something the module actually has.

    python3 bin/selftest_annotations.py

`from __future__ import annotations` makes annotations lazy strings.  That is
what the tree wants -- it is how `bin/` keeps PEP 604 unions off the stock
python3.9 floor (`bin/selftest_py39_floor.py`) -- and it has one consequence
nothing here measured until today: **a missing typing import is invisible to
every runtime check we own.**  `py_compile` passes.  The import passes.  The
whole battery passes.  The annotation is still a lie, and
`typing.get_type_hints()` raises `NameError` on it, so anything that resolves
hints at runtime breaks and every reader and tool in between is misinformed.

Measured today, both found by a `pyright` sweep filtered to
`reportUndefinedVariable`, both real:

  * `bin/measure_cloud.py` -- `_model_file_identity(..., allow_unindexed:
    Sequence[str] = ())` with no `Sequence` in the module's `typing` import.
  * `engines/tools/fetch_truncated_ckpt.py:270` -- `Sequence` in
    `filter_index_lists`, `typing` imported at line 98 without it.

Two instances of one shape, in code that had just passed every gate this
project has.  `pyright` is the better tool and it is NOT the tool that can run
in the push gate: it is an npm package, so putting it there would cost that
job the no-network, no-install property that makes a red push build mean "a
real defect" rather than "a registry hiccup".  So the rule set lives in
`pyproject.toml` for the nightly job and for anyone's editor, and this rung is
the stdlib-only, offline, milliseconds-per-tree half that runs on every push.

WHAT IT CHECKS, exactly: every name used in an annotation position -- argument
annotations, return annotations, `AnnAssign` annotations, and the contents of
a string forward reference -- must be BOUND SOMEWHERE IN THAT MODULE (import,
assignment, `def`, `class`, parameter, comprehension target, `except ... as`,
`global`/`nonlocal`) or be a builtin.

WHAT IT DELIBERATELY DOES NOT CHECK, so nobody reads silence as proof:

  * Whether the bound name is the RIGHT type.  That is a type checker's job
    and this repo has deliberately no type-check target (AGENTS.md).
  * Names outside annotation positions.  `reportUndefinedVariable` over
    executable code is pyright's job in the nightly; a stdlib re-implementation
    would have to model scoping rules and would produce false refusals.
  * Only the ROOT of a dotted name is checked: `typing.Sequence` asks whether
    `typing` is imported, not whether `typing` has that attribute.
  * Draft and historical trees (`port/`, `tools/`, `remote/`) are excluded,
    the same boundary `selftest_py39_floor.py` draws.

Offline, stock python3.9, no installs, no network, no GPU.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Every tree whose annotations are read by a human or a tool. `port/`,
# `tools/` and `remote/` are draft or historical (AGENTS.md "Where things
# live") and are not claimed here.
TREES = ("bin", "registry", "engines", "container")
SKIP_PARTS = frozenset({".venv", "__pycache__", "port", "tools", "remote"})

BUILTINS = frozenset(dir(builtins)) | frozenset({
    # Bound by the interpreter rather than by any statement in the module.
    "__file__", "__name__", "__doc__", "__spec__", "__package__",
    "__builtins__", "__debug__", "__loader__",
})


def annotation_nodes(tree: ast.AST) -> Iterable[Tuple[ast.AST, str]]:
    """Yield (annotation_node, label) for every annotation in the module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            every = list(getattr(args, "posonlyargs", [])) + list(args.args) \
                + list(args.kwonlyargs)
            if args.vararg is not None:
                every.append(args.vararg)
            if args.kwarg is not None:
                every.append(args.kwarg)
            for arg in every:
                if arg.annotation is not None:
                    yield (arg.annotation,
                           "argument %r of %s()" % (arg.arg, node.name))
            if node.returns is not None:
                yield node.returns, "return type of %s()" % node.name
        elif isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None) or "<attribute>"
            yield node.annotation, "annotation of %s" % target


def bound_names(tree: ast.AST) -> Set[str]:
    """Every name this module binds anywhere, at any scope depth.

    Deliberately permissive about WHERE the binding happens: a class defined
    inside a function, a `TYPE_CHECKING` import and a module-level alias are
    all legitimate targets for an annotation, and modelling python's scoping
    rules exactly would buy false refusals rather than findings. What this
    catches is the one shape that is never legitimate -- a name the module
    never binds at all.
    """
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def names_used(node: ast.AST) -> Set[str]:
    """Root names an annotation depends on, including inside forward refs."""
    used: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            used.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            root = sub
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            # A forward reference. If it does not parse as an expression it is
            # not a type, and guessing is worse than ignoring it.
            try:
                inner = ast.parse(sub.value, mode="eval")
            except SyntaxError:
                continue
            used |= names_used(inner)
    return used


def check_source(src: str, label: str) -> List[str]:
    """Annotation names this module uses and never binds."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return ["%s: does not parse: %s" % (label, exc)]
    bound = bound_names(tree) | BUILTINS
    problems: List[str] = []
    for node, where in annotation_nodes(tree):
        for name in sorted(names_used(node) - bound):
            problems.append(
                "%s:%d %s uses %r in an annotation and the module never "
                "binds it" % (label, getattr(node, "lineno", 0), where, name))
    return problems


def annotated_files() -> List[Path]:
    out: List[Path] = []
    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS & set(path.parts):
                continue
            out.append(path)
    return out


FAILURES: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print("PASS %s" % label)
    else:
        FAILURES.append(label)
        print("FAIL %s%s" % (label, (" -- " + detail) if detail else ""))


def main(argv: Optional[List[str]] = None) -> int:
    print("== every annotation names something its module has ==")
    files = annotated_files()
    check("the sweep covers a plausible number of modules",
          len(files) > 100, "found %d" % len(files))

    problems: List[str] = []
    for path in files:
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            problems.append("%s: unreadable: %s" % (path, exc))
            continue
        problems.extend(check_source(src, str(path.relative_to(ROOT))))

    for problem in problems:
        print("   %s" % problem)
    check("no annotation in %s names an unbound module name (%d modules "
          "checked)" % ("/".join(TREES), len(files)),
          not problems, "%d finding(s)" % len(problems))

    # The check must be able to go red, on its own input, in this run. A gate
    # nobody has watched fail is a gate nobody has tested.
    print()
    print("== the check can refuse, and does not refuse what is fine ==")
    missing_import = ("from __future__ import annotations\n"
                      "from typing import Optional\n"
                      "def f(x: Sequence[str] = ()) -> Optional[int]:\n"
                      "    return None\n")
    found = check_source(missing_import, "<fixture>")
    check("a typing name used in an annotation and never imported is "
          "REFUSED, even under `from __future__ import annotations`",
          len(found) == 1 and "'Sequence'" in found[0], "%s" % found)

    forward_ref = ("class Node:\n"
                   "    def parent(self) -> 'Nowhere':\n"
                   "        return None\n")
    check("a string forward reference to a name the module never binds is "
          "REFUSED", len(check_source(forward_ref, "<fixture>")) == 1,
          "%s" % check_source(forward_ref, "<fixture>"))

    fine = ("from __future__ import annotations\n"
            "import typing\n"
            "from typing import TYPE_CHECKING, List\n"
            "if TYPE_CHECKING:\n"
            "    from pathlib import Path\n"
            "Alias = List[int]\n"
            "class Holder:\n"
            "    field: 'Holder'\n"
            "    def go(self, p: Path, q: typing.Mapping[str, int],\n"
            "           r: Alias) -> List[Holder]:\n"
            "        return []\n"
            "def outer():\n"
            "    class Local: pass\n"
            "    def inner(v: Local) -> 'Local':\n"
            "        return v\n"
            "    return inner\n")
    check("a TYPE_CHECKING import, a dotted name, a local class and a "
          "module alias are all accepted",
          not check_source(fine, "<fixture>"),
          "%s" % check_source(fine, "<fixture>"))

    print()
    if FAILURES:
        print("selftest_annotations: %d FAILED" % len(FAILURES))
        for name in FAILURES:
            print("  - %s" % name)
        return 1
    print("selftest_annotations: all rungs passed (%d modules)" % len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
