"""The oracle: tracked fixtures from a program with different ancestry.

PySCF, over libcint, shares no code and no derivation with this repo's Hermite
path, which is what makes it a witness rather than a second copy of our own
arithmetic. The rule this package establishes, and later tests rely on:

    A NUMERIC COMPARISON AGAINST A TRACKED FIXTURE IS FAST-TIER;
    PySCF IS NEEDED ONLY TO REGENERATE ONE.

- `manifest` is the fixture schema, its loader and the provenance record. It
  imports numpy and the standard library and nothing else, so every fast-tier
  test can load a fixture on a box with no PySCF (`uv sync` never installs it).
- `generate` is the PySCF-driven writer, run as
  `PYTHONPATH=test uv run --group oracle python -m oracle.generate --regenerate`
  -- the shape `felizjueves.performance.measured --remeasure` set. Its
  `oracle`-marked test rebuilds each fixture and asserts it is what is tracked.
- `numeric` is the numerical golden store (#655, `test/golden/numeric/`):
  plain `.npy` arrays keyed by registry case, their manifest, and `compare`,
  the one helper a test calls against them. NumPy only, like `manifest`;
  `generate --numeric` writes the arrays.

A fixture is DATA WITH PROVENANCE, NOT THE RIGHT ANSWER: it records what
libcint computed, with which versions, on which host and date. A drift between
a fixture and a live PySCF is news about PySCF until proven otherwise.

This directory is a package (this file) on purpose: pytest's default `prepend`
import mode puts `test/` on sys.path, and a package keeps these module
basenames from colliding with a test module's.
"""
