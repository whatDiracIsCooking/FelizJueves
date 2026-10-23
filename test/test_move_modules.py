"""devtools/move_modules.py (#501): the module-move codemod, on a synthetic tree.

A throwaway map is applied to a small copy of the repo's shape -- a package
under `src/felizjueves`, a `test_layering.py`, a test that reads source by
path, a docs file and a data JSON -- and each rewrite the issue names is
checked, then a second run is checked to change nothing.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "devtools" / "move_modules.py"


def _load():
    spec = importlib.util.spec_from_file_location("move_modules", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


mm = _load()

TREE = {
    "src/felizjueves/__init__.py": "",
    "src/felizjueves/util.py": "X = 1\n",
    "src/felizjueves/kern.py": (
        "from pathlib import Path\n"
        "\n"
        "from felizjueves import util\n"
        "\n"
        "REPO = Path(__file__).resolve().parents[2]\n"
        "DATA = (Path(__file__).resolve().parent.parent.parent\n"
        "        / \"data\")\n"
        "\n"
        "\n"
        "def build():\n"
        "    return util.X\n"),
    "src/felizjueves/engine.py": (
        "MEMBERS = {\n"
        "    \"k\": (\"kern:build\", \"util:X\"),\n"
        "}\n"
        "WORD = \"kern:wide\"\n"),
    "src/felizjueves/pkg/__init__.py": "",
    "src/felizjueves/pkg/a.py": "A = 2\n",
    "src/felizjueves/irx.py": "from . import util\n",
    "src/felizjueves/lower.py": "from .irx import util\n",
    "src/felizjueves/user.py": (
        "from felizjueves import kern, util\n"
        "from felizjueves import pkg as P\n"
        "from felizjueves.kern import build  # noqa: F401\n"
        "from felizjueves.pkg import a\n"
        "import felizjueves.kern as K\n"
        "\n"
        "from .kern import build as b2\n"
        "\n"
        "TARGET = \"felizjueves.kern.build\"\n"
        "BOTH = (kern, util, P, a, K, b2, felizjueves.kern.build)\n"),
    "test/test_layering.py": (
        "RULES = {\n"
        "    \"util\": _rule(\"leaf\", ()),\n"
        "    \"kern\": _rule(\"builder\", (\"util\", \"=kern\"),\n"
        "                  kern=\"not itself\",\n"
        "                  util=\"fine\"),\n"
        "}\n"
        "RULE_OF = {\"kern\": \"kern\", \"util\": \"util\"}\n"
        "RULE_OF[\"pkg.a\"] = \"util\"\n"
        "EDGES = {k: frozenset(v) for k, v in {\n"
        "    \"kern\": (\"util\",),\n"
        "    \"user\": (\"kern\", \"pkg\", \"pkg.a\", \"util\"),\n"
        "}.items()}\n"
        "_BUILDERS = (\"kern\",)\n"
        "EDGES[\"pkg\"] = EDGES[\"util\"] | frozenset(_BUILDERS)\n"
        "GENERATORS = {\"user\": frozenset((\"kern\",))}\n"
        "\n"
        "\n"
        "def test_walk():\n"
        "    todo = [\"kern\"]\n"
        "    assert RULES[\"kern\"].must_not[\"kern\"]\n"
        "    return todo\n"),
    "test/test_reads.py": (
        "from pathlib import Path\n"
        "\n"
        "SRC = Path(__file__).resolve().parents[1] / \"src\" / \"felizjueves\"\n"
        "\n"
        "\n"
        "def test_reads():\n"
        "    a = (SRC / \"kern.py\").read_text()\n"
        "    b = (Path(__file__).resolve().parents[1] / \"src\" / \"felizjueves\"\n"
        "         / \"pkg\" / \"a.py\").read_text()\n"
        "    c = SRC / \"util.py\"\n"
        "    return a, b, c, \"pkg.a\"\n"),
    "docs/notes.md": (
        "Run `python -m felizjueves.kern`; it lives in `src/felizjueves/kern.py`\n"
        "and reads `felizjueves.pkg.a` (`src/felizjueves/pkg/`).\n"
        "The header `include/felizjueves/kern.py` and `felizjueves.kernel_x`\n"
        "and `felizjueves.util` stay; so does a bare `kern.py`.\n"
        "`src/felizjueves/irx.py` becomes a package.\n"),
    "data/store.json": json.dumps({
        "about": "written by python -m felizjueves.kern",
        "felizjueves.kern": "a key is left alone",
        "rows": [{"evidence": ["src/felizjueves/kern.py:3"], "n": 1}],
    }, indent=1) + "\n",
    "data/README.md": "made by felizjueves.kern\n",
    "test/layering_renames.json": "[]\n",
}

MAP = {"kern": "core.kern", "pkg": "deep.pkg2", "lower": "irx.lower"}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for rel, text in TREE.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    if shutil.which("git"):
        def git(*a: str) -> None:
            # mm._git drops GIT_DIR & co.: under the pre-push hook they name
            # the OUTER repository, and `git init` there would make it bare.
            mm._git(tmp_path, *a, check=True)
        git("init", "-q")
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t")
    return tmp_path


def _snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def test_a_map_is_applied_everywhere_and_a_second_run_is_a_no_op(tree):
    rep = mm.run(tree, MAP, use_ruff=False)
    src = tree / "src" / "felizjueves"

    # Files moved; missing packages created; a module file became a package.
    assert not (src / "kern.py").exists() and (src / "core" / "kern.py").exists()
    assert (src / "core" / "__init__.py").read_text() == ""
    assert (src / "deep" / "pkg2" / "a.py").exists() and not (src / "pkg").exists()
    assert (src / "deep" / "__init__.py").exists()
    assert (src / "irx" / "__init__.py").exists() and not (src / "irx.py").exists()
    assert (src / "irx" / "lower.py").exists()
    assert ("src/felizjueves/irx.py", "src/felizjueves/irx/__init__.py") in rep.moved

    # Imports, every form; an alias keeps the local name; a comment follows.
    user = (src / "user.py").read_text()
    assert "from felizjueves import util\n" in user
    assert "from felizjueves.core import kern\n" in user
    assert "from felizjueves.deep import pkg2 as P\n" in user
    assert "from felizjueves.core.kern import build  # noqa: F401\n" in user
    assert "from felizjueves.deep.pkg2 import a\n" in user
    assert "import felizjueves.core.kern as K\n" in user
    assert "from .core.kern import build as b2\n" in user
    assert 'TARGET = "felizjueves.core.kern.build"' in user
    assert "felizjueves.core.kern.build)" in user
    # Relative imports re-counted from the moved file's new package.
    assert (src / "irx" / "__init__.py").read_text() == "from .. import util\n"
    assert (src / "irx" / "lower.py").read_text() == "from . import util\n"

    # MEMBERS-style strings, and only inside MEMBERS.
    engine = (src / "engine.py").read_text()
    assert '"core.kern:build", "util:X"' in engine
    assert 'WORD = "kern:wide"' in engine
    assert any("'kern:wide'" in u for u in rep.unclassified)

    # parents[n] and .parent chains re-counted one level deeper.
    kern = (src / "core" / "kern.py").read_text()
    assert "Path(__file__).resolve().parents[3]" in kern
    assert "Path(__file__).resolve().parent.parent.parent.parent\n" in kern

    # test_layering: module names renamed, rule names kept, must_not keyword.
    lay = (tree / "test" / "test_layering.py").read_text()
    assert '"core.kern": "kern"' in lay  # RULE_OF key renamed, value is a rule
    assert '"kern": _rule("builder", ("util", "=core.kern")' in lay
    assert '**{"core.kern":\n' in lay and 'util="fine"' in lay
    assert 'RULE_OF["deep.pkg2.a"] = "util"' in lay
    assert '"user": ("core.kern", "deep.pkg2", "deep.pkg2.a", "util")' in lay
    assert '_BUILDERS = ("core.kern",)' in lay
    assert 'EDGES["deep.pkg2"] = EDGES["util"]' in lay
    assert 'GENERATORS = {"user": frozenset(("core.kern",))}' in lay
    assert 'todo = ["core.kern"]' in lay
    assert 'RULES["kern"].must_not["core.kern"]' in lay
    compile(lay, "test_layering.py", "exec")

    # Source-path reads.
    reads = (tree / "test" / "test_reads.py").read_text()
    assert '(SRC / "core" / "kern.py")' in reads
    assert '/ "deep" / "pkg2" / "a.py"' in reads
    assert 'SRC / "util.py"' in reads
    assert any("'pkg.a' looks like a moved module name" in u
               for u in rep.unclassified)

    # Docs: the three forms rewritten, everything else left.
    doc = (tree / "docs" / "notes.md").read_text()
    assert "`python -m felizjueves.core.kern`" in doc
    assert "`src/felizjueves/core/kern.py`" in doc
    assert "`felizjueves.deep.pkg2.a` (`src/felizjueves/deep/pkg2/`)" in doc
    assert "`include/felizjueves/kern.py`" in doc
    assert "`felizjueves.kernel_x`" in doc and "`felizjueves.util`" in doc
    assert "a bare `kern.py`" in doc
    assert "`src/felizjueves/irx/__init__.py` becomes a package" in doc

    # data JSON: string values rewritten, keys and other files left alone.
    store = json.loads((tree / "data" / "store.json").read_text())
    assert store["about"] == "written by python -m felizjueves.core.kern"
    assert store["felizjueves.kern"] == "a key is left alone"
    assert store["rows"][0]["evidence"] == ["src/felizjueves/core/kern.py:3"]
    assert (tree / "data" / "README.md").read_text() == "made by felizjueves.kern\n"
    assert any(u.startswith("data/store.json:") and "JSON key" in u
               for u in rep.unclassified)
    assert any(u.startswith("data/README.md:") for u in rep.unclassified)

    # The renames file gets one exact-module map, the package's own
    # `__init__` under the package's name.
    renames = json.loads((tree / "test" / "layering_renames.json").read_text())
    assert renames == [{"kern": "core.kern", "lower": "irx.lower",
                        "pkg": "deep.pkg2", "pkg.a": "deep.pkg2.a"}]
    assert any("'core'" in h for h in rep.hand)  # new packages need a rule

    # Every file it touched is reported.
    for rel in ("src/felizjueves/user.py", "test/test_layering.py",
                "docs/notes.md", "data/store.json", "test/layering_renames.json"):
        assert rel in rep.touched

    # A second run changes nothing.
    before = _snapshot(tree)
    again = mm.run(tree, MAP, use_ruff=False)
    assert _snapshot(tree) == before
    assert not again.moved and not again.created and not again.touched
    assert not again.renames_appended


def test_a_dry_run_reports_and_writes_nothing(tree):
    before = _snapshot(tree)
    rep = mm.run(tree, MAP, dry_run=True, use_ruff=False)
    assert _snapshot(tree) == before
    assert ("src/felizjueves/kern.py", "src/felizjueves/core/kern.py") in rep.moved
    assert "docs/notes.md" in rep.touched and rep.renames_appended


def test_the_cli_takes_pairs_and_a_map_file(tree, capsys):
    path = tree / "m.json"
    path.write_text(json.dumps({"pkg": "deep.pkg2"}))
    assert mm.main(["--root", str(tree), "--dry-run", "--map", str(path),
                    "kern=core.kern"]) == 0
    out = capsys.readouterr().out
    assert "would move    src/felizjueves/kern.py -> src/felizjueves/core/kern.py" \
        in out
    assert "would touch   docs/notes.md" in out
    assert "UNCLASSIFIED  " in out
    assert mm.main(["--root", str(tree), "--dry-run", "--strict",
                    "kern=core.kern"]) == 1


@pytest.mark.parametrize("bad, why", [
    ({"kern": "util"}, "both exist"),
    ({"nope": "x.nope"}, "neither exists"),
    ({"kern": "x.k", "util": "x.k"}, "two modules land"),
    ({"felizjueves.kern": "x"}, "relative to"),
])
def test_a_bad_map_is_refused(tree, bad, why):
    with pytest.raises(mm.MapError, match=why):
        mm.run(tree, mm.load_map([], [f"{k}={v}" for k, v in bad.items()]),
               use_ruff=False)
    assert mm.main(["--root", str(tree), "--dry-run",
                    *[f"{k}={v}" for k, v in bad.items()]]) == 2
