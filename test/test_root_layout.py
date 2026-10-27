"""The package root stays flat-free: new modules go into a package (#515).

Milestone 18 (#501-#515) moved every module out of `src/felizjueves/` into a
domain package. What is left at the root is the key every subsystem joins on
(`spec.py`), the one module that names both engines (`engine.py`), the lever
vocabulary (`levers.py`) and the package marker. This test refuses anything
else there -- a new module, a new package nobody declared, or a stray file --
so the next module lands in the package that owns it, and a new package is a
decision written down here.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "felizjueves"

MODULES = frozenset({"__init__.py", "spec.py", "engine.py", "levers.py"})
PACKAGES = frozenset({
    "ao", "mmd", "hgp", "plan", "ir", "emit", "eri", "one_electron",
    "performance", "literature", "reference", "gradient",
})
# Interpreter and tool droppings, never tracked.
IGNORED = frozenset({"__pycache__"})


def _entries() -> tuple[set[str], set[str]]:
    files, dirs = set(), set()
    for p in ROOT.iterdir():
        if p.name in IGNORED or p.name.startswith("."):
            continue
        (dirs if p.is_dir() else files).add(p.name)
    return files, dirs


def test_the_root_holds_only_the_four_modules():
    files, _ = _entries()
    assert files == MODULES, (
        f"unexpected at src/felizjueves/: {sorted(files - MODULES)}; "
        f"missing: {sorted(MODULES - files)} -- put a new module in the "
        "package that owns it")


def test_the_root_holds_only_the_declared_packages():
    _, dirs = _entries()
    assert dirs == PACKAGES, (
        f"undeclared at src/felizjueves/: {sorted(dirs - PACKAGES)}; "
        f"missing: {sorted(PACKAGES - dirs)} -- a new top-level package is "
        "declared here first")


def test_every_declared_package_is_a_package():
    for name in sorted(PACKAGES):
        assert (ROOT / name / "__init__.py").is_file(), name
