"""The d1 gradient kernels ON A CARD: `G = SUM_mn M_mn dX_mn/dR` for `S`,
`T` and `V` (Hellmann-Feynman included), `dE_J/dR` from `gjab_` and
`dE_K/dR` from `gk_`, against `gradient.host` (generator
`test/grad_driver.py`).

PER MOLECULE, H2O/STO-3G and H2O/cc-pVDZ (`slow`, for its input size and
its (dd) classes, L + 1 = 5), each fixture's own density as `M` (the
recorded symmetric Hilbert matrix; `S`'s `M` is the same matrix -- the
kernel contracts any `M`). One program per (molecule, target), every pair
class of the basis in it, `RUNS` runs into a zeroed `G`.

THE BOUND, stated before any run: `|G_dev - G_host| <= DEV_TOL max|G|`,
`DEV_TOL = 1e-13`. The interpreted kernels are within `kernel_g1e`'s /
`kernel_gjab`'s 64 eps of `max|G|` of the host (test_kernel_g1e.py,
test_kernel_gjab.py); on the card the atomics add the records of an atom
in an order the hardware chooses, and the vendor's `exp` and Boys differ
from the interpreter's by a few ulps -- `1e-13` (~450 eps) holds that with
room and would still catch any wrong term (a gradient term is >= 1e-6 of
`max|G|` on these inputs). The runs are compared with each other too.

`gjab`'s `Jp` is the host's at `L + 1` (grad_driver.py's docstring).
`gk` (H2O/STO-3G here until #618) is test_gk_device.py's now: its units
are compiled one at a time there (test/gk_driver.py), in both forms.

RUN ON BOTH CARDS (2026-10-27), every case passed, cc-pVDZ included:
worst 3.20e-14 of `max|G|` on the RTX 3080 and 2.36e-14 on the RX 9060
XT (`gt` on cc-pVDZ, the atomics' run-to-run spread 2e-14 of it); `gk`
4.1e-15 / 4.0e-15.

THROUGH (ff) (#614; `slow`, L + 1 = 7 at (ff), under the device cap):
`gs`, `gt`, `gv` on CO/cc-pVTZ (`co_ccpvtz_j`, its own density as `M`)
with a BARE PROTON added as a third nucleus (`EXTRA_NUCLEI`, no shell on
it). CO/cc-pVTZ puts an f shell on both atoms, so all 16 ordered classes
(ss)-(ff) occur with `i <= j` (asserted), the off-centre (df) and (ff)
pairs move both of a pair's atoms, and `gv`'s Hellmann-Feynman term lands
on a nucleus that carries no basis function. Every class in its default
form: `gv` (df), (fd), (ff) in `kernel_g1e`'s LOW-STACK form, the rest
unrolled. Each program's compile is capped at `COMPILE_TIMEOUT_S = 300`.
The same `DEV_TOL`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from grad_driver import (
    compile_grad,
    gjab_launches,
    gk_launches,
    grad_runners,
    grad_steps,
    one_electron_launches,
)
from harness_driver import build_chain, chain_specs
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.gradient import host
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

DEV_TOL = 1e-13  # of max|G| (module docstring)
MOLECULES = [pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
             pytest.param("h2o_ccpvdz_j", id="h2o_ccpvdz",
                          marks=(pytest.mark.slow,))]
TARGETS = ("s", "t", "v", "j")
# `k` moved to test_gk_device.py (#618): its units are compiled one at a
# time there, under the user's per-unit budget, in both forms
CASES = [pytest.param(m.values[0], w, id=f"{m.id}-{w}", marks=m.marks)
         for m in MOLECULES for w in TARGETS]
# #614: every one-electron class through (ff), f on two centres, a third
# nucleus with no shell (module docstring)
FF = "co_ccpvtz_j"
EXTRA_NUCLEI = {FF: ((1.0, (-1.1, 0.9, -0.4)),)}
FF_CASES = [pytest.param(FF, w, id=f"co_ccpvtz_proton-{w}",
                         marks=(pytest.mark.slow,)) for w in ("s", "t", "v")]
COMPILE_TIMEOUT_S = 300.0  # the user's per-compile budget


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    nuclei += EXTRA_NUCLEI.get(name, ())
    return ob.from_fixture(fx), nuclei, np.asarray(fx.arrays["density"])


def _host(name: str, which: str) -> np.ndarray:
    basis, nuclei, D = _case(name)
    if which == "s":
        return np.array(host.overlap_gradient(basis, D, nuclei))
    if which == "t":
        return np.array(host.kinetic_gradient(basis, D, nuclei))
    if which == "v":
        return np.array(host.potential_gradient(basis, D, nuclei,
                                                one_centre=one_centre))
    if which == "k":
        return np.array(host.exchange_gradient(basis, D, nuclei,
                                               one_centre=one_centre))
    return np.array(host.coulomb_gradient(basis, D, nuclei,
                                          one_centre=one_centre))


def _launches(name: str, which: str):
    basis, nuclei, D = _case(name)
    if which == "j":
        return gjab_launches(basis, nuclei, D, one_centre=one_centre)
    if which == "k":
        return gk_launches(basis, nuclei, D)
    target = {"s": Target.S, "t": Target.T, "v": Target.V}[which]
    return one_electron_launches(basis, nuclei, target, D)


def _check(backend: str, name: str, which: str, request,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    _, nuclei, _ = _case(name)
    launches = _launches(name, which)
    if name == FF:
        names = {x.kernel.name[3:] for x in launches}
        assert names == {f"{a}{b}" for a in range(4) for b in range(4)}, names
    b = compile_grad(launches, len(nuclei), backend, tc, tmp_path,
                     timeout=COMPILE_TIMEOUT_S)
    got = b.run()
    want = _host(name, which)
    scale = np.abs(want).max()
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= DEV_TOL, (
        f"{backend} {name} {which}: |G - host| / max|G| = {errs} > "
        f"{DEV_TOL:g}")
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} {name} grad {which}: |G - host| / max|G| per run "
          f"{errs}; run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "which"), CASES + FF_CASES)
def test_cuda_gradient_matches_the_host(name, which, request, tmp_path) -> None:
    _check("cuda", name, which, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "which"), CASES + FF_CASES)
def test_hip_gradient_matches_the_host(name, which, request, tmp_path) -> None:
    _check("hip", name, which, request, tmp_path)


# --- No card: the runners compile, and every launch's spec is the kernel's ---


def _runners_compile(backend: str, tc, name: str, which: str,
                     tmp_path: Path) -> None:
    launches = _launches(name, which)
    bins = build_chain(tc, grad_runners(launches, backend, tmp_path), tmp_path,
                       timeout=COMPILE_TIMEOUT_S)
    assert set(bins) == {x.kernel.name for x in launches}


@pytest.mark.cuda
@pytest.mark.parametrize(("name", "which"), CASES)  # the registry's compile tier
def test_cuda_gradient_runners_compile(name, which, nvcc, tmp_path) -> None:
    _runners_compile("cuda", nvcc, name, which, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("name", "which"), CASES)  # the registry's compile tier
def test_hip_gradient_runners_compile(name, which, hipcc, tmp_path) -> None:
    _runners_compile("hip", hipcc, name, which, tmp_path)


@pytest.mark.parametrize(("name", "which"), CASES + FF_CASES)
def test_every_launch_names_the_kernels_arguments(name, which) -> None:
    """Every launch's runner spec, built host-side: each parameter named
    once, of its own dtype, `G` the one output (harness_driver.launch_spec
    refuses anything else)."""
    _, nuclei, _ = _case(name)
    launches = _launches(name, which)
    specs = chain_specs(grad_steps(launches), "G", np.zeros((len(nuclei), 3)))
    for x, spec in zip(launches, specs, strict=True):
        assert [a["name"] for a in spec["args"]] == [
            p.name for p in x.kernel.params]
        assert [a["name"] for a in spec["args"] if "out" in a] == ["G"]
