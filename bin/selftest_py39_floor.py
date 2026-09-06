#!/usr/bin/env python3
"""Make the stock-python3.9 floor a MEASURED property, not a convention.

AGENTS.md: "`bin/` controller paths and all of `registry/` must run on stock
python3.9 with no installs."  Until today that was prose.  Nothing in the tree
checked it, and the battery runs under whatever interpreter the developer has
-- 3.14 on this workstation -- so a 3.10-only construct in a controller path
would pass every gate here and fail on a rented box after the money is spent.

TWO passes, because one is not enough and believing otherwise was very nearly
today's fourth "promise outran the check":

  1. SYNTAX.  `ast.parse(src, feature_version=(3, 9))` rejects `match`,
     `except*`, parenthesized context managers and friends.

  2. PEP 604 (`X | None`).  Pass 1 CANNOT see this: `int | None` is a
     perfectly valid BinOp at parse time on every version.  It raises
     TypeError at RUNTIME on 3.9, and only when the annotation is actually
     evaluated.  So `from __future__ import annotations` decides the verdict:

       * module WITHOUT the future import -- a union in any annotation
         position (argument, return, AnnAssign) is evaluated at def time and
         raises on 3.9.  REFUSED.
       * module WITH the future import -- annotations are lazy strings, so
         those are safe; but a union in a runtime position still raises.  The
         two that matter are `isinstance(x, int | str)` and an explicit
         `Foo: TypeAlias = int | None`.  REFUSED in both cases regardless of
         the future import.

     Credit JLParity, who caught that pass 1 alone would have gone green on
     exactly the `UP045` breakage the pyproject comments warn about.

WHAT THIS DELIBERATELY DOES NOT CHECK, so nobody mistakes silence for proof:

  * Plain assignments such as `MODE = stat.S_IRUSR | stat.S_IWUSR`.  A real
    bitwise-or over flags and a PEP 604 alias are the same AST node, and
    telling them apart needs type information this check does not have.
    Flagging them would produce false refusals in exactly the paths (file
    modes, os.open flags) where they are correct.
  * `dict[str, int]` / `list[int]` subscripts.  Builtin generics ARE
    subscriptable on 3.9, so these are fine.
  * Anything outside `bin/` and `registry/`.  `engines/` runs on the paid
    CUDA environment (python3.12) and `port/`, `tools/`, `remote/` are draft
    or historical.  The floor is a claim about controller paths only.

Offline, no installs, no network.  Runs in well under a second.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# The floor applies to these trees, and only these.  Anything added here is a
# claim that the code runs on stock python3.9 with no installs.
FLOOR_TREES = ("bin", "registry")

# Not part of the floor: virtualenvs, caches, and the draft/historical trees.
SKIP_PARTS = frozenset({".venv", "__pycache__", "port", "tools", "remote"})

FLOOR = (3, 9)


def _annotation_nodes(tree: ast.AST):
    """Yield (node, label) for every annotation position in the module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            every = list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)
            if args.vararg is not None:
                every.append(args.vararg)
            if args.kwarg is not None:
                every.append(args.kwarg)
            for arg in every:
                if arg.annotation is not None:
                    yield arg.annotation, "argument %r of %s()" % (arg.arg, node.name)
            if node.returns is not None:
                yield node.returns, "return type of %s()" % node.name
        elif isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None) or "<attr>"
            yield node.annotation, "annotation of %s" % target


def _has_bitor(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return True
    return False


def _is_type_alias(node: ast.AnnAssign) -> bool:
    ann = node.annotation
    name = getattr(ann, "id", None) or getattr(ann, "attr", None)
    return name == "TypeAlias"


def _future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def check_source(src: str, label: str) -> List[str]:
    """Return a list of floor violations for one module's source."""
    problems: List[str] = []

    # Pass 1 -- syntax the 3.9 parser rejects outright.
    try:
        tree = ast.parse(src, filename=label, feature_version=FLOOR)
    except SyntaxError as exc:
        return ["%s:%s: post-3.9 SYNTAX: %s"
                % (label, exc.lineno or 0, (exc.msg or "").strip())]

    # Pass 2 -- PEP 604 unions, which pass 1 cannot see.
    lazy = _future_annotations(tree)
    if not lazy:
        for node, where in _annotation_nodes(tree):
            if _has_bitor(node):
                problems.append(
                    "%s:%s: PEP 604 union in %s, and the module has no "
                    "`from __future__ import annotations`, so 3.9 evaluates "
                    "it at definition time and raises TypeError"
                    % (label, node.lineno, where))

    for node in ast.walk(tree):
        # `Foo: TypeAlias = int | None` is evaluated eagerly even under the
        # future import: the VALUE is not an annotation.
        if isinstance(node, ast.AnnAssign) and _is_type_alias(node):
            if node.value is not None and _has_bitor(node.value):
                problems.append(
                    "%s:%s: PEP 604 union in a TypeAlias VALUE, which 3.9 "
                    "evaluates eagerly even with lazy annotations"
                    % (label, node.lineno))
        # `isinstance(x, int | str)` is a runtime union, always eager.
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "id", None)
            if fname in ("isinstance", "issubclass") and len(node.args) == 2:
                if _has_bitor(node.args[1]):
                    problems.append(
                        "%s:%s: PEP 604 union as the second argument to %s(), "
                        "which is evaluated at runtime and raises on 3.9"
                        % (label, node.lineno, fname))

    return problems


def floor_files() -> List[Path]:
    out: List[Path] = []
    for tree in FLOOR_TREES:
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
    print("== the stock-python3.9 floor, measured ==")
    files = floor_files()
    check("the floor covers a plausible number of controller modules",
          len(files) > 100, "found %d" % len(files))

    violations: List[Tuple[Path, str]] = []
    for path in files:
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
            violations.append((path, "%s: unreadable: %s" % (path, exc)))
            continue
        for problem in check_source(src, str(path.relative_to(ROOT))):
            violations.append((path, problem))

    for _path, problem in violations:
        print("   %s" % problem)
    check("bin/ and registry/ contain no post-3.9 syntax and no eagerly "
          "evaluated PEP 604 union (%d modules checked)" % len(files),
          not violations, "%d violation(s)" % len(violations))

    # The check must be able to go red, and each pass must go red on its own
    # input.  A gate nobody has seen fail is a gate nobody has tested.
    print()
    print("== the check itself can refuse (both passes) ==")
    syntax_bad = "def f(x):\n    match x:\n        case 1: return 2\n"
    found = check_source(syntax_bad, "<syntax>")
    check("pass 1 refuses a match statement",
          any("post-3.9 SYNTAX" in p for p in found), repr(found))

    union_bad = "def f(x: int | None) -> str | None:\n    return None\n"
    found = check_source(union_bad, "<union>")
    check("pass 2 refuses an annotation union with no future import",
          any("PEP 604" in p for p in found), repr(found))

    union_lazy = ("from __future__ import annotations\n"
                  "def f(x: int | None) -> str | None:\n    return None\n")
    check("pass 2 ACCEPTS the same unions under lazy annotations",
          check_source(union_lazy, "<lazy>") == [],
          repr(check_source(union_lazy, "<lazy>")))

    alias_bad = ("from __future__ import annotations\n"
                 "from typing import TypeAlias\n"
                 "Maybe: TypeAlias = int | None\n")
    found = check_source(alias_bad, "<alias>")
    check("pass 2 refuses a TypeAlias union even under lazy annotations",
          any("TypeAlias VALUE" in p for p in found), repr(found))

    isinst_bad = ("from __future__ import annotations\n"
                  "def f(x):\n    return isinstance(x, int | str)\n")
    found = check_source(isinst_bad, "<isinstance>")
    check("pass 2 refuses a runtime isinstance union even under lazy "
          "annotations",
          any("second argument to isinstance" in p for p in found), repr(found))

    flags_ok = ("import stat\n"
                "MODE = stat.S_IRUSR | stat.S_IWUSR\n"
                "FLAGS = 0o1 | 0o2\n")
    check("a real bitwise-or over flags is NOT a floor violation",
          check_source(flags_ok, "<flags>") == [],
          repr(check_source(flags_ok, "<flags>")))

    generics_ok = ("def f(d):\n    return dict[str, int]()\n")
    check("builtin generic subscripts are fine on 3.9",
          check_source(generics_ok, "<generics>") == [],
          repr(check_source(generics_ok, "<generics>")))

    print()
    if FAILURES:
        print("selftest_py39_floor: %d FAILED" % len(FAILURES))
        return 1
    print("selftest_py39_floor: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
