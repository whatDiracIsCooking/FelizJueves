#!/usr/bin/env python3
"""Move modules of src/felizjueves, and every reference to them, in one run.

    devtools/move_modules.py [--dry-run] [--map FILE]... [OLD=NEW ...]

The map is old -> new dotted module names relative to `felizjueves`, as a JSON
object in a file (`--map`, repeatable) or as `OLD=NEW` pairs on the command
line. An OLD may be a package (`j_levers`): every module in it moves with it.
In one run the tool

- `git mv`s each module's file (a package file by file), creating every
  missing package `__init__.py` on the way, and turning a module file into a
  package (`ir.py` -> `ir/__init__.py`) when a move puts a module under it;
- rewrites, in every `.py` of `src/`, `test/` and `measure/`:
  * imports: `from felizjueves.X import ...`, `from felizjueves import X`
    (split per new parent, aliased when the last name changes so the local
    name stays), `import felizjueves.X`, and relative imports under `src/`;
  * every `felizjueves.X` / `felizjueves/X.py` mention in code, strings and
    comments (`monkeypatch.setattr("felizjueves....")`, `sys.modules[...]`);
  * `"X:attr"` strings inside a `MEMBERS = {...}` table (`engine.MEMBERS`);
    the same shape elsewhere (`"dispatch:wide"`) is reported, not rewritten;
  * in `test/test_layering.py`, and in a test that imports it (its
    `_graph()` speaks the same bare names), every string that is exactly a
    moved module (or `"=X"`): `RULE_OF` / `EDGES` / `GENERATORS` keys and
    values, `may` prefixes, `must_not["X"]` subscripts, the modules a test
    walks from. Rule names (`RULES`' keys and subscripts, `RULE_OF`'s
    values, what a `RULE_OF[...]` is compared with) are not modules and are
    kept. A `must_not` keyword that names a moved module becomes
    `**{"new.name": ...}`, the same prefix spelled as a string. In a test of
    the rename machinery itself (one that calls `apply_renames`), only
    subscripts are renamed and the rest is reported (hypothetical moves);
  * source-path reads: a `/` chain of strings after `"felizjueves"` or after
    a module-level name bound to one (`SRC / "kernel.py"`);
  * in a moved module, `Path(__file__)...parents[n]` / `.parent.parent...`
    re-counted for its new depth;
- rewrites `felizjueves.X`, `python -m felizjueves.X` and
  `src/felizjueves/X.py` in `docs/`, `.claude/` (not its worktrees),
  `devtools/`, `README.md`, and in the string VALUES of `data/**/*.json`
  (object keys and every other `data/` file are left alone);
- appends the exact module map to `test/layering_renames.json` when that file
  exists (#502's format: a JSON list, one `{old: new}` object per move, a
  package's `__init__` under the package's name), once.

It prints every file it moved, created or touched; what needs a hand (a new
package with no layering rule, a line the rename pushed past 88 characters);
and every mention it could not classify: a module-shaped name in a place it
does not rewrite (a file outside the scopes above, a JSON key, a dotted
`"ao.drive"` string in an ordinary test, a moved file's path outside a `/`
chain). It does not guess; it reports. Bare prose mentions (`ao/j.py`,
`` `kernel_k` ``) and bare file names a move keeps (`"kdup.py"`) are not
references it rewrites. Prefix rules that a move widens or narrows (`may:
("ao",)` once `ao.eri` leaves `ao/`) are #502's equivalence check's to catch.

RE-RUNNABLE. Applying a map that has already been applied changes nothing: a
module already at its new place is not moved again, every pattern matches
only the OLD spelling, and the renames entry is appended once. So a move PR
that conflicts on rebase is fixed by resetting the conflicted files to `main`
and re-running. A map whose NEW name is another entry's OLD name is refused,
because it would not be re-runnable.

After writing, the touched `.py` files get `ruff check --select I --fix`
(moved imports re-sort); `--no-ruff` skips it. Exit 0 on success (with or
without unclassified mentions; `--strict` makes those exit 1), 2 on a bad map
or a state it will not resolve (a module at both its old and new place).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PKG = "felizjueves"
CODE_DIRS = ("src", "test", "measure")
DOC_DIRS = ("docs", ".claude", "devtools")
DOC_FILES = ("README.md",)
LAYERING = "test/test_layering.py"
RENAMES = "test/layering_renames.json"
SELF_EXCLUDE = ("devtools/move_modules.py", "devtools/moves/",
                "test/test_move_modules.py", RENAMES,
                "test/layering_relation.json")
SKIP_PARTS = {".git", ".venv", "__pycache__", "worktrees", ".kernel-tier",
              ".logs", "node_modules", ".measure-cache"}
LINE_LENGTH = 88  # pyproject.toml's [tool.ruff] line-length
_IDENT = r"[A-Za-z_]\w*"
_DOTTED = re.compile(rf"^{_IDENT}(\.{_IDENT})*$")


class MapError(Exception):
    """A map or a tree state the tool refuses to act on."""


# ---------------------------------------------------------------------------
# The plan: which files go where, and the exact module map
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    root: Path
    moves: dict[str, str]                    # exact modules, old -> new
    file_moves: list[tuple[str, str]] = field(default_factory=list)
    new_inits: list[str] = field(default_factory=list)
    old_of_new_file: dict[str, str] = field(default_factory=dict)  # new rel -> old rel
    old_paths: dict[str, str] = field(default_factory=dict)  # old path form -> new
    packages: set[str] = field(default_factory=set)  # old names that are packages

    @property
    def src(self) -> Path:
        return self.root / "src" / PKG


def _mod_file(src: Path, mod: str) -> Path | None:
    """The file that holds module `mod` under `src`, or None."""
    parts = mod.split(".")
    pkg = src.joinpath(*parts) / "__init__.py"
    if pkg.is_file():
        return pkg
    f = src.joinpath(*parts[:-1]) / f"{parts[-1]}.py"
    return f if f.is_file() else None


def _modules_under(src: Path, pkg: str) -> dict[str, Path]:
    """Every module of package `pkg` (itself under its own name)."""
    base = src.joinpath(*pkg.split("."))
    out: dict[str, Path] = {}
    for p in sorted(base.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src).with_suffix("").parts
        name = ".".join(rel[:-1] if rel[-1] == "__init__" else rel)
        out[name] = p
    return out


def _is_pkg(src: Path, mod: str) -> bool:
    return (src.joinpath(*mod.split(".")) / "__init__.py").is_file()


def load_map(files: list[Path], pairs: list[str]) -> dict[str, str]:
    raw: dict[str, str] = {}
    for f in files:
        doc = json.loads(Path(f).read_text())
        if isinstance(doc, dict) and "moves" in doc and isinstance(doc["moves"], dict):
            doc = doc["moves"]
        if not isinstance(doc, dict):
            raise MapError(f"{f}: a map is a JSON object {{old: new}}")
        raw.update(doc)
    for p in pairs:
        if "=" not in p:
            raise MapError(f"{p!r}: a pair is OLD=NEW")
        old, new = p.split("=", 1)
        raw[old.strip()] = new.strip()
    for old, new in raw.items():
        for n in (old, new):
            if not isinstance(n, str) or not _DOTTED.match(n):
                raise MapError(f"{n!r} is not a dotted module name")
            if n == PKG or n.startswith(PKG + "."):
                raise MapError(f"{n!r}: names are relative to {PKG}")
    return raw


def plan_moves(root: Path, user_map: dict[str, str]) -> Plan:
    src = root / "src" / PKG
    moves: dict[str, str] = {}
    file_moves: list[tuple[str, str]] = []
    rel = lambda p: p.relative_to(root).as_posix()  # noqa: E731
    for old, new in user_map.items():
        if old == new:
            continue
        here, there = _mod_file(src, old), _mod_file(src, new)
        if here and there:
            raise MapError(f"{old} -> {new}: both exist; resolve by hand")
        if here is None and there is None:
            raise MapError(f"{old} -> {new}: neither exists")
        side = old if here else new
        if _is_pkg(src, side):
            for m, p in _modules_under(src, side).items():
                tail = m[len(side):]
                moves[old + tail] = new + tail
                if here:
                    sub = p.relative_to(src.joinpath(*old.split(".")))
                    file_moves.append(
                        (rel(p), rel(src.joinpath(*new.split(".")) / sub)))
        else:
            moves[old] = new
            if here:
                parts = new.split(".")
                file_moves.append(
                    (rel(here),
                     rel(src.joinpath(*parts[:-1]) / f"{parts[-1]}.py")))
    clash = sorted(set(moves) & set(moves.values()))
    if clash:
        raise MapError(f"{clash}: a new name is another move's old name; the "
                       "map would not be re-runnable")
    news = list(moves.values())
    dup = sorted({n for n in news if news.count(n) > 1})
    if dup:
        raise MapError(f"two modules land on {dup}")
    landing = set(news)
    plan = Plan(root, dict(sorted(moves.items())), file_moves)
    plan.packages = {o for o in moves if any(m.startswith(o + ".") for m in moves)}
    # Every ancestor package of a new name: create it, or turn a module file
    # into a package (`ir.py` -> `ir/__init__.py`).
    moved_out = {o for o, _ in file_moves}
    seen: set[str] = set()
    for new in sorted(landing):
        parts = new.split(".")
        for i in range(1, len(parts)):
            anc = ".".join(parts[:i])
            if anc in seen or anc in landing:
                continue
            seen.add(anc)
            d = src.joinpath(*parts[:i])
            if (d / "__init__.py").is_file():
                continue
            as_file = d.with_suffix(".py")
            if as_file.is_file() and rel(as_file) not in moved_out:
                plan.file_moves.append((rel(as_file), rel(d / "__init__.py")))
            else:
                plan.new_inits.append(rel(d / "__init__.py"))
    for o, n in plan.file_moves:
        plan.old_of_new_file[n] = o
    for o, n in plan.file_moves:
        po = o[len("src/"):]
        pn = n[len("src/"):]
        plan.old_paths[po] = pn
    # Whole package directories, for `felizjueves/j_levers/` style paths.
    for old, new in user_map.items():
        if old != new and old in plan.moves and _is_pkg(
                src, old if _mod_file(src, old) else new):
            plan.old_paths[f"{PKG}/{old.replace('.', '/')}/"] = (
                f"{PKG}/{new.replace('.', '/')}/")
    return plan


# ---------------------------------------------------------------------------
# Text rewriting
# ---------------------------------------------------------------------------


def rename(plan: Plan, mod: str) -> str:
    """`mod` (relative to felizjueves) after the moves, prefix-wise."""
    best = ""
    for old in plan.moves:
        if (mod == old or mod.startswith(old + ".")) and len(old) > len(best):
            best = old
    return plan.moves[best] + mod[len(best):] if best else mod


def _alt(names: list[str]) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


@dataclass
class Patterns:
    dotted: re.Pattern | None
    path: re.Pattern | None
    member: re.Pattern | None


def patterns(plan: Plan) -> Patterns:
    olds = list(plan.moves)
    dotted = member = path = None
    if olds:
        dotted = re.compile(rf"(?<![\w.]){PKG}\.({_alt(olds)})(?!\w)")
        member = re.compile(rf"^({_alt(olds)}):({_IDENT})$")
    if plan.old_paths:
        files = [p for p in plan.old_paths if not p.endswith("/")]
        dirs = [p for p in plan.old_paths if p.endswith("/")]
        alts = []
        if files:
            alts.append(rf"(?:{_alt(files)})(?!\w)")
        if dirs:
            alts.append(rf"(?:{_alt([d[:-1] for d in dirs])})(?!\w)")
        path = re.compile(rf"(?<!\w)(?<!include/)((?:{'|'.join(alts)}))")
    return Patterns(dotted, path, member)


def _sub_mentions(plan: Plan, pats: Patterns, text: str) -> str:
    if pats.dotted:
        text = pats.dotted.sub(
            lambda m: f"{PKG}.{plan.moves[m.group(1)]}", text)
    if pats.path:
        def path_new(m: re.Match) -> str:
            got = m.group(1)
            if got in plan.old_paths:
                return plan.old_paths[got]
            return plan.old_paths[got + "/"][:-1]
        text = pats.path.sub(path_new, text)
    return text


class Lines:
    """Map ast (lineno, byte col) to a character offset in `text`."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.starts = [0]
        for line in text.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.lines = text.splitlines(keepends=True)

    def off(self, lineno: int, col: int) -> int:
        line = self.lines[lineno - 1] if lineno - 1 < len(self.lines) else ""
        return self.starts[lineno - 1] + len(
            line.encode()[:col].decode(errors="replace"))

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (self.off(node.lineno, node.col_offset),
                self.off(node.end_lineno, node.end_col_offset))

    def lineno_of(self, offset: int) -> int:
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1


def _q(s: str) -> str:
    return json.dumps(s)


@dataclass
class Report:
    moved: list[tuple[str, str]] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    renames_appended: bool = False


def _module_of(rel_path: str) -> str | None:
    """`src/felizjueves/a/b.py` -> `a.b` (`__init__` for the root)."""
    pre = f"src/{PKG}/"
    if not rel_path.startswith(pre) or not rel_path.endswith(".py"):
        return None
    parts = rel_path[len(pre):-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


class PyRewriter:
    def __init__(self, plan: Plan, pats: Patterns, rel_now: str,
                 rel_final: str, text: str, report: Report) -> None:
        self.plan, self.pats, self.report = plan, pats, report
        self.rel_now, self.rel_final = rel_now, rel_final
        self.text = text
        self.L = Lines(text)
        self.edits: list[tuple[int, int, str]] = []
        self.claimed: list[tuple[int, int]] = []

    # -- helpers ----------------------------------------------------------
    def _edit(self, node: ast.AST, new: str) -> None:
        a, b = self.L.span(node)
        self.edits.append((a, b, new))

    def _claim(self, node: ast.AST) -> None:
        self.claimed.append(self.L.span(node))

    def _flag(self, node: ast.AST, why: str) -> None:
        self.report.unclassified.append(
            f"{self.rel_now}:{node.lineno}: {why}")

    def _old_mod(self, s: str) -> bool:
        """`s` is a moved module, or a name inside a moved package."""
        return s in self.plan.moves or any(
            s.startswith(o + ".") for o in self.plan.packages)

    def _claimed(self, node: ast.AST) -> bool:
        a, b = self.L.span(node)
        return any(x <= a and b <= y for x, y in self.claimed)

    # -- imports ----------------------------------------------------------
    def _pkg_of_file(self, rel: str) -> list[str] | None:
        mod = _module_of(rel)
        if mod is None:
            return None
        parts = mod.split(".") if mod else []
        return parts if rel.endswith("__init__.py") else parts[:-1]

    def imports(self, tree: ast.AST) -> None:
        old_rel = self.plan.old_of_new_file.get(self.rel_final, self.rel_now)
        old_pkg = self._pkg_of_file(old_rel)
        new_pkg = self._pkg_of_file(self.rel_final)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                if old_pkg is None or new_pkg is None:
                    continue
                base = old_pkg[:len(old_pkg) - (node.level - 1)]
                tail = node.module.split(".") if node.module else []
                abs_mod = ".".join(base + tail)
            elif node.module == PKG:
                abs_mod = ""
            elif node.module and node.module.startswith(PKG + "."):
                abs_mod = node.module[len(PKG) + 1:]
            else:
                continue
            groups: dict[str, list[str]] = {}
            for a in node.names:
                full = f"{abs_mod}.{a.name}" if abs_mod else a.name
                if a.name != "*" and full in self.plan.moves:
                    nf = self.plan.moves[full]
                    parent, _, last = nf.rpartition(".")
                    local = a.asname or a.name
                    text = last if local == last else f"{last} as {local}"
                    groups.setdefault(parent, []).append(text)
                else:
                    text = a.name if not a.asname else f"{a.name} as {a.asname}"
                    groups.setdefault(rename(self.plan, abs_mod), []).append(text)
            orig_names = [a.name if not a.asname else f"{a.name} as {a.asname}"
                          for a in node.names]
            rel_out = bool(node.level)

            def head(mod: str, rel_out: bool = rel_out,
                     new_pkg: list[str] | None = new_pkg) -> str:
                if not rel_out or new_pkg is None:
                    return f"{PKG}.{mod}" if mod else PKG
                tgt = mod.split(".") if mod else []
                common = 0
                while (common < len(new_pkg) and common < len(tgt)
                       and new_pkg[common] == tgt[common]):
                    common += 1
                dots = len(new_pkg) - common + 1
                return "." * dots + ".".join(tgt[common:])

            orig_head = "." * node.level + (node.module or "")
            if (len(groups) == 1 and head(next(iter(groups))) == orig_head
                    and next(iter(groups.values())) == orig_names):
                continue
            indent = " " * node.col_offset
            lines = [f"from {head(m)} import {', '.join(ns)}"
                     for m, ns in groups.items()]
            # Comments (a noqa directive) inside the statement, or after it on
            # its last line, go on every statement it becomes.
            a, b = self.L.span(node)
            inner = re.findall(r"#[^\n]*", self.text[a:b])
            rest = self.text[b:self.text.find("\n", b) if "\n" in self.text[b:]
                             else len(self.text)]
            trailing = re.match(r"\s*(#[^\n]*)", rest)
            if len(set(inner)) > 1:
                self._flag(node, f"comments {inner[1:]} inside a rewritten "
                                 "import were dropped")
            comment = inner[0] if inner else (trailing.group(1) if trailing
                                              else None)
            if comment:
                last = len(lines) - (0 if inner else 1)
                lines = [f"{ln}  {comment.strip()}" if i < last else ln
                         for i, ln in enumerate(lines)]
            self._edit(node, ("\n" + indent).join(lines))

    # -- strings ----------------------------------------------------------
    def members(self, tree: ast.AST) -> None:
        """`"X:attr"` strings inside a `MEMBERS = {...}` assignment (the
        shape `engine.MEMBERS` names its callables in) are renamed; the same
        shape anywhere else (`"dispatch:wide"` is a batch-count word, not a
        module) is reported, never rewritten."""
        if not self.pats.member:
            return
        inside: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                if any(isinstance(t, ast.Name) and t.id == "MEMBERS"
                       for t in targets) and node.value is not None:
                    inside.update(id(x) for x in ast.walk(node.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                m = self.pats.member.match(node.value)
                if not m:
                    continue
                if id(node) in inside:
                    self._edit(node, _q(
                        f"{rename(self.plan, m.group(1))}:{m.group(2)}"))
                else:
                    self._flag(node, f"{node.value!r} has the `module:attr` "
                                     "shape outside a MEMBERS table")
                self._claim(node)

    def source_paths(self, tree: ast.AST) -> None:
        roots: set[str] = set()
        for node in getattr(tree, "body", []):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                ops = _div_chain(node.value)
                if ops and _is_str(ops[-1], PKG):
                    roots.add(node.targets[0].id)
        done: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or id(node) in done:
                continue
            ops = _div_chain(node)
            if not ops:
                continue
            for sub in ast.walk(node):
                done.add(id(sub))
            start = None
            for i, op in enumerate(ops):
                if _is_str(op, PKG) or (i == 0 and isinstance(op, ast.Name)
                                        and op.id in roots):
                    start = i + 1
            if start is None:
                continue
            segs = []
            for op in ops[start:]:
                if isinstance(op, ast.Constant) and isinstance(op.value, str):
                    segs.append(op)
                else:
                    break
            if not segs:
                continue
            path = "/".join(s.value.strip("/") for s in segs)
            new = self._new_rel_path(path)
            if new is None:
                continue
            for s in segs:
                self._claim(s)
            if new != path:
                a = self.L.span(segs[0])[0]
                b = self.L.span(segs[-1])[1]
                self.edits.append(
                    (a, b, " / ".join(_q(p) for p in new.split("/"))))

    def _new_rel_path(self, path: str) -> str | None:
        """`kernel.py` -> `mmd/kernel.py`, or None if it names nothing moved."""
        key = f"{PKG}/{path}"
        if key in self.plan.old_paths:
            return self.plan.old_paths[key][len(PKG) + 1:]
        for old, new in self.plan.old_paths.items():
            if old.endswith("/") and (key + "/").startswith(old):
                return (new + (key + "/")[len(old):]).rstrip("/")[len(PKG) + 1:]
        return path if any(
            n == key or n.startswith(key + "/") or (n.endswith("/") and
                                                    key.startswith(n))
            for n in self.plan.old_paths.values()) else None

    def layering(self, tree: ast.AST) -> None:
        """`test_layering.py`: every module name, never a rule name.

        Rule names are `RULES`' keys and subscripts, `RULE_OF`'s values, and
        what a `RULE_OF[...]` / `RULE_OF.get(...)` is compared with. Every
        other string that is exactly a moved module (or `"=X"`) is a module
        name -- a `RULE_OF` / `EDGES` / `GENERATORS` key or value, a `may`
        prefix, a `must_not["X"]` subscript, a module a test walks from --
        and is renamed. A `must_not` keyword that names a moved module
        becomes `**{"new.name": ...}`, the same prefix spelled as a string.
        """
        rule_pos: set[int] = set()

        def strs(n: ast.AST | None) -> list[ast.Constant]:
            if n is None:
                return []
            return [x for x in ast.walk(n)
                    if isinstance(x, ast.Constant) and isinstance(x.value, str)]

        def is_name(n: ast.AST, *ids: str) -> bool:
            return isinstance(n, ast.Name) and n.id in ids

        def rule_lookup(n: ast.AST) -> bool:
            if isinstance(n, ast.Subscript) and is_name(n.value, "RULE_OF"):
                return True
            return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and is_name(n.func.value, "RULE_OF"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for t in targets:
                    if is_name(t, "RULES") and isinstance(node.value, ast.Dict):
                        rule_pos.update(id(k) for k in node.value.keys if k)
                    if is_name(t, "RULE_OF") and isinstance(node.value, ast.Dict):
                        for v in node.value.values:
                            rule_pos.update(id(c) for c in strs(v))
                    if isinstance(t, ast.Subscript) and is_name(t.value, "RULE_OF"):
                        rule_pos.update(id(c) for c in strs(node.value))
            if isinstance(node, ast.Subscript) and is_name(node.value, "RULES"):
                rule_pos.update(id(c) for c in strs(node.slice))
            if isinstance(node, ast.Compare) and (
                    rule_lookup(node.left)
                    or any(rule_lookup(c) for c in node.comparators)):
                for c in [node.left, *node.comparators]:
                    if not rule_lookup(c):
                        rule_pos.update(id(x) for x in strs(c))
            if isinstance(node, ast.Call) and is_name(node.func, "_rule", "Rule"):
                for kw in node.keywords:
                    if kw.arg and self._old_mod(kw.arg):
                        # Two edits around the value, so the strings inside
                        # it are still renamed by the scan below.
                        a = self.L.span(kw)[0]
                        va, vb = self.L.span(kw.value)
                        ind = " " * (kw.col_offset + 4)
                        self.edits.append(
                            (a, va, f"**{{{_q(rename(self.plan, kw.arg))}:"
                                    f"\n{ind}"))
                        self.edits.append((vb, vb, "}"))
        # A test of the rename machinery itself (it calls `apply_renames`)
        # spells HYPOTHETICAL moves in real names: only its subscripts
        # (`RULE_OF["x"]`, a relation's `base["x"]`) are module positions;
        # the rest is reported.
        hypo: set[int] = set()
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and any(
                    isinstance(c, ast.Call) and is_name(c.func, "apply_renames")
                    for c in ast.walk(fn)):
                hypo.update(id(c) for c in strs(fn))
                for x in ast.walk(fn):
                    if isinstance(x, ast.Subscript) and isinstance(
                            x.value, ast.Name) and x.value.id != "RULES":
                        hypo.difference_update(id(c) for c in strs(x.slice))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in rule_pos or self._claimed(node):
                continue
            exact = node.value.startswith("=")
            body = node.value[1:] if exact else node.value
            if id(node) in hypo:
                if self._old_mod(body):
                    self._flag(node, f"{node.value!r} in a test of the rename "
                                     "machinery (a hypothetical move?)")
                    self._claim(node)
                continue
            if self._old_mod(body):
                self._edit(node, _q(("=" if exact else "")
                                    + rename(self.plan, body)))
                self._claim(node)

    def suspicious(self, tree: ast.AST) -> None:
        """Strings that look like a moved module but were not rewritten.

        A bare file name (`"kdup.py"`) is flagged only when the move changes
        it (`ir.py` -> `__init__.py`): a basename the move keeps is still
        right. A relative path (`"ao/j.py"`) is flagged always, as is an
        exact moved module name with a dot in it (`"ao.drive"`)."""
        paths: dict[str, str] = {}
        for o, n in self.plan.old_paths.items():
            if o.endswith("/"):
                continue
            rel = o[len(PKG) + 1:]
            if "/" in rel:
                paths[rel] = n
            if rel.rsplit("/", 1)[-1] != n.rsplit("/", 1)[-1]:
                paths[rel.rsplit("/", 1)[-1]] = n
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if self._claimed(node):
                continue
            s = node.value
            if s in paths:
                self._flag(node, f"{s!r} names a moved module's file outside a "
                                 f"`/` chain after {PKG!r}")
            elif "." in s and self._old_mod(s):
                self._flag(node, f"{s!r} looks like a moved module name")

    # -- depth ------------------------------------------------------------
    def depth(self, text: str) -> str:
        old = self.plan.old_of_new_file.get(self.rel_final)
        if old is None or old == self.rel_final or self.rel_now != old:
            return text
        delta = self.rel_final.count("/") - old.count("/")
        if not delta:
            return text

        def parents(m: re.Match) -> str:
            n = int(m.group(2)) + delta
            if n < 0:
                raise MapError(f"{self.rel_now}: parents[{m.group(2)}] + {delta} < 0")
            return f"{m.group(1)}.parents[{n}]"

        def chain(m: re.Match) -> str:
            n = m.group(2).count(".parent") + delta
            if n <= 0:
                raise MapError(f"{self.rel_now}: .parent chain + {delta} <= 0")
            return m.group(1) + ".parent" * n

        head = r"(Path\(__file__\)(?:\.resolve\(\))?)"
        text = re.sub(head + r"\.parents\[(\d+)\]", parents, text)
        text = re.sub(head + r"((?:\.parent)+)(?![\w\[])", chain, text)
        return text

    # -- driver -----------------------------------------------------------
    def run(self) -> str:
        try:
            tree = ast.parse(self.text)
        except SyntaxError as e:
            self.report.unclassified.append(
                f"{self.rel_now}:{e.lineno}: not parsed ({e.msg}); text "
                "mentions rewritten only")
            return self.depth(_sub_mentions(self.plan, self.pats, self.text))
        self.imports(tree)
        self.members(tree)
        self.source_paths(tree)
        if LAYERING in (self.rel_now, self.rel_final) or _uses_layering(tree):
            self.layering(tree)
        # Apply the edits, and the mention patterns to the gaps between them.
        edits = sorted(set(self.edits))
        for (_a1, b1, _), (a2, _b2, _) in zip(edits, edits[1:], strict=False):
            if a2 < b1:
                raise MapError(f"{self.rel_now}: overlapping edits at "
                               f"line {self.L.lineno_of(a2)}")
        out, pos = [], 0
        for a, b, new in edits:
            out.append(_sub_mentions(self.plan, self.pats, self.text[pos:a]))
            out.append(new)
            pos = b
        out.append(_sub_mentions(self.plan, self.pats, self.text[pos:]))
        new_text = "".join(out)
        # Flag what is left only once the rewritten text is known, so a
        # string the mention patterns fixed is not reported.
        self.suspicious(tree)
        return self.depth(new_text)


def _uses_layering(tree: ast.AST) -> bool:
    """A test that imports `test_layering` (its `_graph`) names modules the
    way `test_layering.py` does, as bare strings."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "test_layering":
            return True
        if isinstance(node, ast.Import) and any(
                a.name == "test_layering" for a in node.names):
            return True
    return False


def _div_chain(node: ast.AST) -> list[ast.AST] | None:
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return None
    out: list[ast.AST] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        out.append(node.right)
        node = node.left
    out.append(node)
    return out[::-1]


def _is_str(node: ast.AST, s: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == s


def rewrite_json(plan: Plan, pats: Patterns, rel: str, text: str,
                 report: Report) -> str:
    """Mentions rewritten inside string VALUES only; keys are left alone."""
    out, i, n = [], 0, len(text)
    stack: list[list] = []  # ["{", expecting_key] or ["["]
    L = None
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tok = text[i:j + 1]
            is_key = bool(stack) and stack[-1][0] == "{" and stack[-1][1]
            new = _sub_mentions(plan, pats, tok)
            if new != tok and is_key:
                L = L or Lines(text)
                report.unclassified.append(
                    f"{rel}:{L.lineno_of(i)}: a module mention in a JSON key")
                new = tok
            out.append(new)
            if is_key:
                stack[-1][1] = False
            i = j + 1
            continue
        if c == "{":
            stack.append(["{", True])
        elif c == "[":
            stack.append(["["])
        elif c in "}]":
            stack.pop()
        elif c == "," and stack and stack[-1][0] == "{":
            stack[-1][1] = True
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str, check: bool = False,
         capture: bool = True) -> subprocess.CompletedProcess:
    """`git -C root ...` with the repository taken from `root` alone.

    Inside a git hook (the pre-push fast tier runs the tests) git exports
    GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE, and a `git -C <tree>` would
    then act on the OUTER repository; they are dropped."""
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION}
    return subprocess.run(["git", "-C", str(root), *args], check=check,
                          capture_output=capture, text=True, env=env)


_GIT_LOCATION = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_COMMON_DIR", "GIT_PREFIX", "GIT_NAMESPACE")


def _files(root: Path) -> list[str]:
    try:
        got = _git(root, "ls-files", "-co", "--exclude-standard",
                   check=True).stdout.split("\n")
        files = [f for f in got if f and (root / f).is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [p.relative_to(root).as_posix() for p in root.rglob("*")
                 if p.is_file()]
    return sorted(f for f in set(files)
                  if not SKIP_PARTS & set(Path(f).parts[:-1])
                  and not any(f == x or f.startswith(x) for x in SELF_EXCLUDE))


def _in_git(root: Path) -> bool:
    return _git(root, "rev-parse", "--git-dir").returncode == 0


def _scope(rel: str) -> str:
    top = rel.split("/", 1)[0]
    if rel.endswith(".py") and top in CODE_DIRS:
        return "py"
    if top == "data":
        return "json" if rel.endswith(".json") else "other"
    if top in DOC_DIRS or rel in DOC_FILES:
        return "doc"
    return "other"


def run(root: Path, user_map: dict[str, str], dry_run: bool = False,
        use_ruff: bool = True) -> Report:
    root = root.resolve()
    plan = plan_moves(root, user_map)
    pats = patterns(plan)
    report = Report()
    final_of = dict(plan.file_moves)
    new_texts: dict[str, tuple[str, str]] = {}  # final rel -> (text, now rel)
    for rel in _files(root):
        p = root / rel
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        final = final_of.get(rel, rel)
        kind = _scope(rel)
        if kind == "py":
            new = PyRewriter(plan, pats, rel, final, text, report).run()
            before = set(text.splitlines())
            for i, line in enumerate(new.splitlines(), 1):
                if (len(line) > LINE_LENGTH and line not in before
                        and not line.lstrip().startswith(("from ", "import "))):
                    report.hand.append(f"{final}:{i}: {len(line)} > "
                                       f"{LINE_LENGTH} characters after the "
                                       "rename; re-wrap it")
        elif kind == "doc":
            new = _sub_mentions(plan, pats, text)
        elif kind == "json":
            new = rewrite_json(plan, pats, rel, text, report)
        else:
            if _sub_mentions(plan, pats, text) != text:
                L = Lines(text)
                for pat in (pats.dotted, pats.path):
                    for m in (pat.finditer(text) if pat else ()):
                        report.unclassified.append(
                            f"{rel}:{L.lineno_of(m.start())}: {m.group(0)!r} "
                            "outside the rewritten scopes")
            new = text
        if new != text or final != rel:
            new_texts[final] = (new, rel)
            if new != text:
                report.touched.append(final)
    report.moved = list(plan.file_moves)
    report.created = list(plan.new_inits)
    for init in plan.new_inits:
        mod = _module_of(init)
        report.hand.append(f"new package {mod!r} ({init}) needs a RULE_OF and "
                           f"an EDGES entry in {LAYERING}")
    renames = root / RENAMES
    entry = {o: n for o, n in plan.moves.items()}
    if renames.is_file() and entry:
        maps = json.loads(renames.read_text())
        if not isinstance(maps, list):
            raise MapError(f"{RENAMES} is not a JSON list")
        if entry not in maps:
            report.renames_appended = True
            if not dry_run:
                maps.append(entry)
                renames.write_text(json.dumps(maps, indent=2, sort_keys=True)
                                   + "\n")
            report.touched.append(RENAMES)
    if dry_run:
        return report
    git = _in_git(root)
    for init in plan.new_inits:
        p = root / init
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("")
            if git:
                _git(root, "add", init, check=True)
    for old, new in plan.file_moves:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        tracked = git and _git(root, "ls-files", "--error-unmatch",
                               old).returncode == 0
        if tracked:
            _git(root, "mv", old, new, check=True)
        else:
            shutil.move(root / old, root / new)
        d = (root / old).parent
        while d != root and d.is_dir() and not any(
                x for x in d.iterdir() if x.name != "__pycache__"):
            shutil.rmtree(d)
            d = d.parent
    for final, (text, _now) in new_texts.items():
        p = root / final
        if p.read_text() != text:
            p.write_text(text)
    py = [f for f in report.touched if f.endswith(".py")]
    if use_ruff and py:
        subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "I",
                        "--fix", "--quiet", "--exit-zero", *py], cwd=root,
                       capture_output=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", metavar="OLD=NEW")
    ap.add_argument("--map", action="append", default=[], type=Path)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ruff", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any mention is unclassified")
    args = ap.parse_args(argv)
    try:
        user_map = load_map(args.map, args.pairs)
        if not user_map:
            ap.error("no moves given")
        rep = run(args.root, user_map, args.dry_run, not args.no_ruff)
    except MapError as e:
        print(f"move_modules: {e}", file=sys.stderr)
        return 2
    verb = "would " if args.dry_run else ""
    for o, n in rep.moved:
        print(f"{verb}move    {o} -> {n}")
    for c in rep.created:
        print(f"{verb}create  {c}")
    for t in sorted(set(rep.touched)):
        print(f"{verb}touch   {t}")
    for h in rep.hand:
        print(f"by hand  {h}")
    for u in rep.unclassified:
        print(f"UNCLASSIFIED  {u}")
    print(f"{len(rep.moved)} moved, {len(rep.created)} created, "
          f"{len(set(rep.touched))} touched, {len(rep.unclassified)} "
          f"unclassified" + (" (dry run)" if args.dry_run else ""))
    return 1 if args.strict and rep.unclassified else 0


if __name__ == "__main__":
    sys.exit(main())
