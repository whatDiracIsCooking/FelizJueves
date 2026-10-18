"""The measure/ tree's fixtures: test/conftest.py's, loaded by path.

WHY A SECOND TREE. test/ holds CORRECTNESS -- a kernel against a reference,
a builder against its spec, an analysis rule against synthetic rows -- and
nothing in it times, counts or reads back a measured store. What does lives
here: the opt-in sweeps that time or profile a kernel and write a store, the
probes that bound a card by its vendor's published peak, the recompiles that
compare today's toolchain with the stored rows, and the checks that the
tracked reports in data/measured/ are what their rows recompute to. pyproject's
`testpaths` names test/ only, so a bare `pytest`, the push gate and
devtools/kernel-tier.sh never collect a file here; devtools/measure.sh is the
one way in, one sweep per subcommand (`devtools/measure.sh list`).

ONE SET OF FIXTURES, NOT TWO. A sweep times the very binary a correctness
test checks, through the same compiler fixtures, the same card lock (#91)
and the same VRAM probe, so this conftest does not restate any of them: it
loads test/conftest.py under another module name and re-exports everything
it defines -- the fixtures, the `pytest_configure` / `pytest_unconfigure`
hooks that take the card's lock in an xdist controller, and the plain names
(`Toolchain`, `VRAM_PROBE`, `check_vram`, ...) that test modules import with
`from conftest import ...`. pytest registers THIS file as the session's
`conftest`, so such an import -- from a measure file, or from a test module
a measure file imports (`import test_kernel_device`) -- resolves here and
finds the same objects. test/ is put on sys.path first, which is what lets a
measure file import the correctness modules and their drivers
(`fock_driver`, `plan_driver`, `tiers`, ...) by bare name, as test/ does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent.parent / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

_spec = importlib.util.spec_from_file_location("fj_test_conftest",
                                               TEST_DIR / "conftest.py")
assert _spec is not None and _spec.loader is not None
_harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _harness
_spec.loader.exec_module(_harness)

globals().update({k: v for k, v in vars(_harness).items()
                  if not (k.startswith("__") and k.endswith("__"))})
