"""The DF Hessian kernels ON A CARD (#694): `d2metric_` and `d2eri3_`
against `hessian.host_df.metric_hessian` and `hessian.host_eri3.eri3_hessian`
(generator `test/hess_driver.py`'s program, `d2metric_launches` /
`d2eri3_launches`; the arrays are `df_grad_launch`'s, the interpreted
tests').

PART 1, H2O/cc-pVDZ AGAINST cc-pVDZ-JKFIT (`slow`): the fixture's AO basis
and density, the JKFIT set read from data/aux_basis/ (`ao.bse`, Cartesian,
s to f), and the DF-J weights of test_grad_df_device.py one derivative order
up: `d = (P|Q)^{-1} (P|mn) D`, `Gam = D (x) d` for `d2eri3_` and `W = -1/2
d d^T` for `d2metric_`, `d` from the host's `(P|Q)` and `(mn|P)`. One
program per target, every class of it (`d2eri3` through (dd|f), `d2metric`
through (f|f)), `RUNS` runs into a zeroed `H`.

PART 2, THE LADDER THROUGH (ff|i) AND (i|i) (`slow` + `high_l`): the d1 DF
ladder's aux-i system (`manifest.df_gradient`: `prim_spd_f`, He s, p, d and
a contracted Ne f, against l = 0..6 aux shells on both atoms), which reaches
every class of the ladder, each in its DEFAULT form, with that witness's
closed-form `Gam` and `W`. One program per aux `l` (`lP` for d2eri3, `lQ`
for d2metric), each compile within the user's 300 s budget
(`COMPILE_BUDGET_S`, a hard timeout). There is no libcint witness at d2;
the host is the one witness, itself held to finite differences of the d1
host (test_hessian_host_df.py, test_hessian_host_eri3.py).

THE BOUND, stated before any run: `|H_dev - H_host| <= DEV_TOL max(twin)`,
`DEV_TOL = 1e-14`, `twin` the host's `absolute=True` sum, test_hess_device.py's
bound: `max|H|` is no scale here (the tower cancels at high L, and the
interpreted kernels are within 0.4 eps of the twin and up to ~7,000 eps of
`max|H|`, test_kernel_d2_df_ladder.py). The card's atomics reorder the adds
and its `exp` and Boys differ from the interpreter's by ulps, each rounding
bounded by the same twin. The runs are compared with each other too.

NOT RUN ON ANY CARD YET: the session that wrote this file had nvcc and no
card (#694). Its host-only tests (the staging covers every class once) ran.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from harness_driver import chain_specs
from hess_driver import (
    compile_hess,
    d2eri3_launches,
    d2metric_launches,
    hess_runners,
    hess_steps,
)
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.ao import bse
from felizjueves.eri.density_fit.eri3 import eri3_tensor
from felizjueves.eri.density_fit.metric import metric_matrix
from felizjueves.hessian import host_df, host_eri3
from felizjueves.reference.pq import pq

DEV_TOL = 1e-14  # of max(twin) (module docstring)
COMPILE_BUDGET_S = 300.0  # the user's per-compile budget (#613, #694)
FIXTURE = "h2o_ccpvdz_j"
AUX_SET = "cc-pVDZ-JKFIT"


@cache
def _case():
    fx = manifest.load(manifest.FIXTURES / f"{FIXTURE}.json")
    basis = ob.from_fixture(fx)
    atoms = [(int(a.charge), tuple(a.coord)) for a in fx.atoms]
    aux = ob.from_bse(bse.named(AUX_SET, bse.AUXILIARY), atoms)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    D = np.asarray(fx.arrays["density"])
    T = np.array(eri3_tensor(basis, aux, pq=pq))
    d = np.linalg.solve(np.array(metric_matrix(aux, pq=pq)),
                        np.einsum("mnp,mn->p", T, D))
    return basis, aux, nuclei, D[:, :, None] * d[None, None, :], \
        -0.5 * np.outer(d, d)


def _launches(which: str):
    basis, aux, nuclei, Gam, W = _case()
    return (d2metric_launches(aux, nuclei, W) if which == "metric"
            else d2eri3_launches(basis, aux, nuclei, Gam))


@cache
def _host(which: str) -> tuple[np.ndarray, float]:
    """The host's `H` and the max of its `absolute=True` twin."""
    basis, aux, nuclei, Gam, W = _case()
    if which == "metric":
        H, twin = (np.array(host_df.metric_hessian(
            aux, W, nuclei, pq=pq, absolute=a)) for a in (False, True))
    else:
        H, twin = (np.array(host_eri3.eri3_hessian(
            basis, aux, Gam, nuclei, pq=pq, absolute=a))
            for a in (False, True))
    return H, float(np.abs(twin).max())


def _compare(backend: str, what: str, got: np.ndarray, want: np.ndarray,
             scale: float) -> None:
    errs = [float(np.abs(h - want).max() / scale) for h in got]
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} {what}: |H - host| / max(twin) per run {errs}; "
          f"run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")
    assert max(errs) <= DEV_TOL, (backend, what, errs)


def _check(backend: str, which: str, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    nuclei = _case()[2]
    got = compile_hess(_launches(which), len(nuclei), backend, tc,
                       tmp_path, timeout=COMPILE_BUDGET_S).run()
    want, scale = _host(which)
    _compare(backend, f"H2O/cc-pVDZ {AUX_SET} {which}", got, want, scale)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("which", ("metric", "eri3"))
def test_cuda_df_hessian_matches_the_host(which, request, tmp_path) -> None:
    _check("cuda", which, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("which", ("metric", "eri3"))
def test_hip_df_hessian_matches_the_host(which, request, tmp_path) -> None:
    _check("hip", which, request, tmp_path)


@pytest.mark.slow
def test_the_staged_launches_cover_every_class_once(tmp_path) -> None:
    # Host-only: the program's input is the interpreted tests' staging, and
    # the classes the card runs are every class of the two bases.
    basis, aux, *_ = _case()
    assert max(s.l for s in basis.shells) == 2
    assert max(s.l for s in aux.shells) == 3
    eri3 = [x.kernel.name for x in _launches("eri3")]
    metric = [x.kernel.name for x in _launches("metric")]
    assert len(set(eri3)) == len(eri3) == 6 * 4
    assert len(set(metric)) == len(metric) == 10
    for which in ("metric", "eri3"):
        xs = _launches(which)
        assert all(x.n_rec >= 1 for x in xs)
        specs = chain_specs(hess_steps(xs), "H", np.zeros((3, 3, 3, 3)))
        assert [s["kernel"] for s in specs] == [x.kernel.name for x in xs]
        runners = hess_runners(xs, "cuda", tmp_path / which)
        assert set(runners) == {x.kernel.name for x in xs}


# --- Part 2: the ladder through (ff|i) and (i|i) --------------------------------


@cache
def _auxi():
    wit = manifest.df_gradient()
    ts = wit.three_center
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return wit, basis, aux, nuclei


def _ladder_launches(which: str, l: int):  # noqa: E741
    """Every class of the aux-i system with aux `l` (`lP` / `lQ`)."""
    wit, basis, aux, nuclei = _auxi()
    if which == "metric":
        return d2metric_launches(aux, nuclei, wit.w(), lQ=l)
    return d2eri3_launches(basis, aux, nuclei, wit.gam(), lP=l)


@cache
def _ladder_host(which: str, l: int) -> tuple[np.ndarray, float]:  # noqa: E741
    wit, basis, aux, nuclei = _auxi()
    sh, ash = basis.shells, aux.shells
    if which == "metric":
        pairs = [(i, j) for i in range(len(ash)) for j in range(i, len(ash))
                 if max(ash[i].l, ash[j].l) == l]
        H, twin = (np.array(host_df.metric_hessian(
            aux, wit.w(), nuclei, pq=pq, pairs=pairs, absolute=a))
            for a in (False, True))
    else:
        cells = [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
                 for n in range(len(ash)) if ash[n].l == l]
        H, twin = (np.array(host_eri3.eri3_hessian(
            basis, aux, wit.gam(), nuclei, pq=pq, cells=cells, absolute=a))
            for a in (False, True))
    return H, float(np.abs(twin).max())


def _ladder(backend: str, which: str, l: int, request,  # noqa: E741
            tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    nuclei = _auxi()[3]
    got = compile_hess(_ladder_launches(which, l), len(nuclei), backend, tc,
                       tmp_path, timeout=COMPILE_BUDGET_S).run()
    want, scale = _ladder_host(which, l)
    _compare(backend, f"aux-i {which} l={l}", got, want, scale)


LADDER = [("metric", lq) for lq in range(7)] + [("eri3", lp)
                                                for lp in range(7)]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("which,l", LADDER, ids=lambda x: str(x))
def test_cuda_df_hessian_ladder_through_i(which, l, request,  # noqa: E741
                                          tmp_path):
    _ladder("cuda", which, l, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("which,l", LADDER, ids=lambda x: str(x))
def test_hip_df_hessian_ladder_through_i(which, l, request,  # noqa: E741
                                         tmp_path):
    _ladder("hip", which, l, request, tmp_path)


@pytest.mark.slow  # builds every class through L = 12
def test_the_ladder_parts_cover_every_class_of_the_ladder() -> None:
    """Host-only: the aux-i system's classes are the whole ladder, split
    over the parts by aux l, each once."""
    seen: dict[str, list[str]] = {"metric": [], "eri3": []}
    for which, l in LADDER:  # noqa: E741
        seen[which] += [x.kernel.name for x in _ladder_launches(which, l)]
    assert sorted(seen["metric"]) == sorted(
        f"d2metric_{p}{q}" for p in range(7) for q in range(p, 7))
    assert sorted(seen["eri3"]) == sorted(
        f"d2eri3_{a}{b}{p}" for a in range(4) for b in range(a, 4)
        for p in range(7))
