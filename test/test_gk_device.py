"""The exchange-energy gradient `gk_` ON A CARD, both forms, every unit
compiled alone under the user's #618 budget (generator test/gk_driver.py).

H2O/STO-3G (the kernel tier's case, every K class s / p, L <= 4): each
class in its DEFAULT form (unrolled below `LOWSTACK_MIN_L`) and in its
ROLLED form (`lowstack=True`, named `..._rolled`) on the same upload, one
program. Held to `gradient.host.exchange_gradient` -- the whole `dE_K`
(the sum of the default launches) and, per class, the rolled launch
against the default one (THE DIFFERENTIAL), each within `DEV_TOL
max|G|`. This was test_grad_device.py's `k` case (moved here, #618: that
harness puts every unit in one translation unit).

CO/cc-pVTZ (`slow`, `high_l`: L + 1 = 13 at (ff|ff)), `co_ccpvtz_j`'s own
recorded density, f on both atoms: all 256 K classes in their DEFAULT
forms, one program per card, against the tracked LIBCINT witness
(`manifest.k_gradient`: `int2e_ip1_cart` contracted per class and whole)
within `FIXTURE_TOL max|G|`, per class and whole -- no pure-Python host
reference on the card's input. Plus THE DIFFERENTIAL: every class with
every `l <= 2` whose unrolled form compiled within half the budget on
this arch in data/measured/gk.json (its `unrolled` rows) is launched
in its other form too, named `..._unrolled` / `..._rolled`, and the two
`G` are held to each other within `DEV_TOL max|G|`.

THE BOUNDS, stated before any run: `DEV_TOL = 1e-13` of `max|G|` (the
interpreted forms agree to ~1 eps of their round-off scale; on the card
the atomics add in hardware order and the vendor's `exp` / Boys differ by
a few ulps: test_grad_device.py's bound); `FIXTURE_TOL = 1e-12` of the
whole `max|G|` against libcint (the issue's).

EVERY COMPILE is one unit, killed at `gk_driver.UNIT_BUDGET_S = 10` s
(a unit over it fails the test; it is never retried with more).
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from gk_driver import UNIT_BUDGET_S, G, compile_gk, gk_class_launches, gk_steps
from harness_driver import Ref, sequence_spec
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.gradient import host
from felizjueves.gradient.kernel_gk import UNROLLED_L_MAX, gk_lowstack, gk_name
from felizjueves.performance import measured
from felizjueves.reference.pq import one_centre

DEV_TOL = 1e-13  # of max|G| (module docstring)
FIXTURE_TOL = 1e-12  # of the whole max|G|, against libcint
STORE = Path(__file__).resolve().parent.parent / "data/measured/gk.json"
H2O, CO = "h2o_sto3g_j", "co_ccpvtz_j"


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, np.asarray(fx.density)


def _classes(basis) -> list[tuple[int, int, int, int]]:
    ls = sorted({s.l for s in basis.shells})
    return [(a, c, b, d) for a in ls for c in ls for b in ls for d in ls]


def _other(basis, nuclei, D, classes):
    """Each class's other form, named so it shares a program with its
    default: `..._rolled` / `..._unrolled`."""
    out = []
    for am in classes:
        low = not gk_lowstack(am)
        (x,) = gk_class_launches(basis, nuclei, D, [am], lowstack=low)
        name = f"{gk_name(am)}_{'rolled' if low else 'unrolled'}"
        out.append(replace(x, kernel=replace(x.kernel, name=name)))
    return out


def _compiled_unrolled(arch: str) -> set[tuple[int, int, int, int]]:
    """The classes whose unrolled unit compiled, alone, within HALF the
    budget on `arch` (data/measured/gk.json): the margin for the four
    compiles the driver runs at once."""
    rows = json.loads(STORE.read_text())["compile"]
    return {tuple(r["am"]) for r in rows if r["arch"] == arch
            and r["form"] == "unrolled" and not r["over_budget"]
            and r["compile_s"] <= UNIT_BUDGET_S / 2}


def _report(tag: str, errs, got) -> None:
    print(f"\n{tag}: worst {max(errs):.3e} of max|G|; run-to-run "
          f"{float(np.abs(got[0] - got[1]).max()):.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")


def _h2o(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    basis, nuclei, D = _case(H2O)
    classes = _classes(basis)
    assert all(sum(c) < 5 and max(c) <= 1 for c in classes)
    launches = (gk_class_launches(basis, nuclei, D, classes)
                + _other(basis, nuclei, D, classes))
    b = compile_gk(launches, len(nuclei), backend, tc, tmp_path)
    got = b.run()
    n = len(classes)
    want = np.array(host.exchange_gradient(basis, D, nuclei,
                                           one_centre=one_centre))
    scale = float(np.abs(want).max())
    errs = [float(np.abs(g[:n].sum(axis=0) - want).max() / scale)
            for g in got]
    assert max(errs) <= DEV_TOL, (backend, errs)
    diff = max(float(np.abs(g[:n] - g[n:]).max() / scale) for g in got)
    assert diff <= DEV_TOL, (backend, diff)
    _report(f"{backend} h2o_sto3g gk (whole vs host; rolled vs default "
            f"{diff:.2e})", errs, got)


def _co(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    basis, nuclei, D = _case(CO)
    wit = manifest.k_gradient()
    classes = _classes(basis)
    assert len(classes) == 256 and set(wit.classes) == set(classes)
    both = sorted(c for c in _compiled_unrolled(
        measured.BACKENDS[backend].arch.name)
        if max(c) <= UNROLLED_L_MAX)
    assert both, "no unrolled compile rows for this arch"
    launches = (gk_class_launches(basis, nuclei, D, classes)
                + _other(basis, nuclei, D, both))
    b = compile_gk(launches, len(nuclei), backend, tc, tmp_path)
    slowest = max(b.compile_s, key=b.compile_s.get)
    print(f"\n{backend}: {len(b.compile_s)} units, slowest {slowest} "
          f"{b.compile_s[slowest]:.2f} s")
    got = b.run()
    n = len(classes)
    scale = float(np.abs(wit.whole).max())
    errs = []
    for g in got:
        per = max(float(np.abs(g[i] - wit.classes[c]).max())
                  for i, c in enumerate(classes)) / scale
        whole = float(np.abs(g[:n].sum(axis=0) - wit.whole).max()) / scale
        errs.append(max(per, whole))
        assert per <= FIXTURE_TOL and whole <= FIXTURE_TOL, (per, whole)
    at = {c: i for i, c in enumerate(classes)}
    diff = max(float(np.abs(g[at[c]] - g[n + j]).max()) / scale
               for g in got for j, c in enumerate(both))
    assert diff <= DEV_TOL, (backend, diff)
    _report(f"{backend} co_ccpvtz gk, 256 classes against libcint "
            f"(differential over {len(both)} classes {diff:.2e})", errs, got)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_gk_h2o_sto3g_both_forms(request, run_device_binary,
                                      tmp_path) -> None:
    _h2o("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_gk_h2o_sto3g_both_forms(request, run_device_binary,
                                     tmp_path) -> None:
    _h2o("hip", request, run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
def test_cuda_gk_co_ccpvtz_against_libcint(request, run_device_binary,
                                           tmp_path) -> None:
    _co("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
def test_hip_gk_co_ccpvtz_against_libcint(request, run_device_binary,
                                          tmp_path) -> None:
    _co("hip", request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`): the H2O program
# ---------------------------------------------------------------------------


def _h2o_compiles(backend: str, tc, tmp_path: Path) -> None:
    """Every H2O/STO-3G unit in its default form and the shared runner's
    `main()`, each its own `-c` under `UNIT_BUDGET_S`, linked -- no card.
    One compile at a time (`jobs=1`): the compile tier already runs its
    tests in parallel, and the budget is per unit, not per box."""
    basis, nuclei, D = _case(H2O)
    launches = gk_class_launches(basis, nuclei, D, _classes(basis))
    b = compile_gk(launches, len(nuclei), backend, tc, tmp_path, jobs=1)
    assert b.binary.exists() and "gk_main" in b.compile_s
    slowest = max(b.compile_s, key=b.compile_s.get)
    print(f"\n{backend}: {len(b.compile_s)} units, slowest {slowest} "
          f"{b.compile_s[slowest]:.2f} s (budget {UNIT_BUDGET_S:g} s)")


@pytest.mark.cuda
def test_cuda_gk_h2o_sto3g_compiles(nvcc, tmp_path) -> None:
    _h2o_compiles("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_gk_h2o_sto3g_compiles(hipcc, tmp_path) -> None:
    _h2o_compiles("hip", hipcc, tmp_path)


def test_the_sequence_hands_each_launch_its_arrays() -> None:
    """Each launch of the runner's sequence names its own arrays (the
    density one shared buffer) holding exactly the launch's, and its own
    `G` slice; `sequence_spec` accepts every launch (no card, no
    compiler)."""
    basis, nuclei, D = _case(H2O)
    classes = _classes(basis)
    launches = (gk_class_launches(basis, nuclei, D, classes)
                + _other(basis, nuclei, D, classes))
    n_atom = len(nuclei)
    buffers, steps = gk_steps(launches, n_atom)
    assert buffers[G].shape == (len(launches), n_atom, 3)
    for c, (x, st) in enumerate(zip(launches, steps, strict=True)):
        assert st.args[G] == Ref(G, c * n_atom * 3)
        assert (st.args["n_bra"], st.args["n_ket"]) == (x.n_bra, x.n_ket)
        for name, a in x.arrays.items():
            buf = st.args[name]
            assert buf == ("M" if name == "M" else f"c{c}.{name}")
            assert np.array_equal(buffers[buf], np.ravel(a))
    sequence_spec(buffers, steps, outs=[G], runs=2, witness="each")
