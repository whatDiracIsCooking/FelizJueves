"""The DF d1 gradient kernels ON A CARD: `gmetric_` and `geri3_` building
the DF-J energy gradient of H2O/STO-3G against `gradient.host_df`
(generator `test/grad_driver.py`'s program, which takes any `GradLaunch`;
the arrays are `df_grad_launch`'s, the interpreted tests').

THE INPUT. H2O/STO-3G's AO basis and density against the cc-pVTZ-JKFIT aux
basis's s, p and d shells (every class at L <= 4, the kernels' validated
cap: `(d|d)` and `(pp|d)` the highest), with the DF-J weights of
test_gradient_host_df.py: `d = (P|Q)^{-1} (P|mn) D`, `Gam = D (x) d` for
`geri3_` and `W = -1/2 d d^T` for `gmetric_`, `d` from the host's `(P|Q)`
and `(mn|P)`. One program per target, every class of it, `RUNS` runs into
a zeroed `G`; the sum of the two is the whole DF-J gradient and is checked
against the host's sum as well.

THE BOUND, stated before any run: `|G_dev - G_host| <= DEV_TOL max|G|`,
`DEV_TOL = 1e-13`, test_grad_device.py's: the interpreted kernels are
within 5 eps of their scale of the host (test_kernel_gmetric.py,
test_kernel_geri3.py); the card's atomics reorder the adds and its `exp`
and Boys differ by ulps. The runs are compared with each other too.

RUN ON BOTH CARDS (2026-10-27): worst 6.58e-15 of `max|G|` on the RTX
3080 and 4.87e-15 on the RX 9060 XT (`gmetric`).

PART 2, THE LADDER THROUGH (ff|i) AND (i|i) (#613; `slow` + `high_l`): the
aux-i witness's system (`manifest.df_gradient`: `prim_spd_f`, He s, p, d
and a contracted Ne f, against l = 0..6 aux shells on both atoms), which
reaches every class FreeBreakfast emits (geri3 `la <= lb <= 3`, `lP <= 6`;
gmetric `lP <= lQ <= 6`), each in its DEFAULT form (the low-stack one from
`kernel_geri3.LOWSTACK_MIN_L` / `kernel_gmetric.LOWSTACK_MIN_L`), with the
witness's closed-form `Gam` and `W`. One program per aux `l` (`lP` for
geri3, `lQ` for gmetric), so each compile stays within the user's 300 s
budget (`COMPILE_BUDGET_S`, a hard timeout). Against the host over the
program's classes at `DEV_TOL` of their `max|G|`, and against libcint's
derivative integrals (the witness, per class) at `LIBCINT_TOL` of the
witness's largest class `max|G|`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from df_grad_launch import eri3_arrays, metric_arrays
from grad_driver import GradLaunch, compile_grad, grad_program
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.eri3 import eri3_classes, eri3_tensor
from felizjueves.eri.density_fit.metric import metric_classes, metric_matrix
from felizjueves.gradient import host_df
from felizjueves.gradient.kernel_geri3 import build_geri3_kernel
from felizjueves.gradient.kernel_gmetric import build_gmetric_kernel
from felizjueves.reference.pq import pq

DEV_TOL = 1e-13  # of max|G| (module docstring)
LIBCINT_TOL = 1e-12  # of the witness's max|G| (Part 2)
COMPILE_BUDGET_S = 300.0  # the user's per-compile budget (#613)
SET = "h2o_sto3g_jkfit_3c"


@cache
def _case():
    ts = manifest.three_center_set(SET)
    basis = ob.from_fixture(ts.parent)
    aux = Basis(tuple(s for s in ob.from_fixture(ts).shells if s.l <= 2))
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    D = np.asarray(ts.parent.arrays["density"])
    T = np.array(eri3_tensor(basis, aux, pq=pq))
    d = np.linalg.solve(np.array(metric_matrix(aux, pq=pq)),
                        np.einsum("mnp,mn->p", T, D))
    return basis, aux, nuclei, D[:, :, None] * d[None, None, :], \
        -0.5 * np.outer(d, d)


def _launches(which: str) -> list[GradLaunch]:
    basis, aux, nuclei, Gam, W = _case()
    out = []
    if which == "metric":
        for c in metric_classes(aux):
            arrs = metric_arrays(c, aux, nuclei) | {"M": W}
            out.append(GradLaunch(build_gmetric_kernel((c.lP, c.lQ)),
                                  c.n_pair, arrs, {
                "n_pair": c.n_pair, "n_rec": c.n_rec, "n_aux": aux.n_ao,
                "n_atom": len(nuclei)}))
        return out
    for c in eri3_classes(basis, aux):
        arrs = eri3_arrays(c, basis, aux, nuclei) | {"M": Gam}
        out.append(GradLaunch(build_geri3_kernel((c.la, c.lb, c.lP)),
                              c.n_cell, arrs, {
            "n_cell": c.n_cell, "n_bra_rec": c.n_bra_rec,
            "n_aux_rec": c.n_aux_rec, "n_ao": basis.n_ao,
            "n_aux": aux.n_ao, "n_atom": len(nuclei)}))
    return out


@cache
def _host(which: str) -> np.ndarray:
    basis, aux, nuclei, Gam, W = _case()
    if which == "metric":
        return np.array(host_df.metric_gradient(aux, W, nuclei, pq=pq))
    return np.array(host_df.eri3_gradient(basis, aux, Gam, nuclei, pq=pq))


def _check(backend: str, which: str, request, run_device_binary,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    nuclei = _case()[2]
    got = compile_grad(_launches(which), len(nuclei), backend, tc,
                       tmp_path).run(run_device_binary)
    want = _host(which)
    scale = np.abs(want).max()
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= DEV_TOL, (
        f"{backend} {which}: |G - host| / max|G| = {errs} > {DEV_TOL:g}")
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} H2O/STO-3G grad {which}: |G - host| / max|G| per "
          f"run {errs}; run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("which", ("metric", "eri3"))
def test_cuda_df_gradient_matches_the_host(which, request, run_device_binary,
                                           tmp_path) -> None:
    _check("cuda", which, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("which", ("metric", "eri3"))
def test_hip_df_gradient_matches_the_host(which, request, run_device_binary,
                                          tmp_path) -> None:
    _check("hip", which, request, run_device_binary, tmp_path)


@pytest.mark.slow
def test_the_staged_launches_cover_every_class_once(tmp_path) -> None:
    # Host-only: the program's input is the interpreted tests' staging, and
    # the classes the card runs are every class of the two bases at L <= 4.
    basis, aux, *_ = _case()
    eri3 = {(c.la, c.lb, c.lP) for c in eri3_classes(basis, aux)}
    assert max(sum(c) for c in eri3) == 4
    assert {(c.lP, c.lQ) for c in metric_classes(aux)} == {
        (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    for which in ("metric", "eri3"):
        xs = _launches(which)
        assert all(x.n_rec >= 1 for x in xs)
        prog = grad_program(xs, 3, "cuda", tmp_path / which)
        assert prog.source.count("<<<") == len(xs)
    # DF-J: the sum of the two host terms has zero net force.
    G = _host("metric") + _host("eri3")
    assert np.abs(G.sum(axis=0)).max() <= 1e-12 * np.abs(G).max()


# --- Part 2: the ladder through (ff|i) and (i|i) (#613) ------------------------


@cache
def _auxi():
    wit = manifest.df_gradient()
    ts = wit.three_center
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return wit, basis, aux, nuclei


def _ladder_launches(which: str, l: int) -> tuple[list[GradLaunch], list]:  # noqa: E741
    """Every class of the aux-i system with aux `l` (`lP` / `lQ`)."""
    wit, basis, aux, nuclei = _auxi()
    out, classes = [], []
    if which == "metric":
        for c in metric_classes(aux):
            if c.lQ != l:
                continue
            arrs = metric_arrays(c, aux, nuclei) | {"M": wit.w()}
            out.append(GradLaunch(build_gmetric_kernel((c.lP, c.lQ)),
                                  c.n_pair, arrs, {
                "n_pair": c.n_pair, "n_rec": c.n_rec, "n_aux": aux.n_ao,
                "n_atom": len(nuclei)}))
            classes.append((c.lP, c.lQ))
        return out, classes
    for c in eri3_classes(basis, aux):
        if c.lP != l:
            continue
        arrs = eri3_arrays(c, basis, aux, nuclei) | {"M": wit.gam()}
        out.append(GradLaunch(build_geri3_kernel((c.la, c.lb, c.lP)),
                              c.n_cell, arrs, {
            "n_cell": c.n_cell, "n_bra_rec": c.n_bra_rec,
            "n_aux_rec": c.n_aux_rec, "n_ao": basis.n_ao,
            "n_aux": aux.n_ao, "n_atom": len(nuclei)}))
        classes.append((c.la, c.lb, c.lP))
    return out, classes


@cache
def _ladder_host(which: str, l: int) -> np.ndarray:  # noqa: E741
    wit, basis, aux, nuclei = _auxi()
    sh, ash = basis.shells, aux.shells
    if which == "metric":
        pairs = [(i, j) for i in range(len(ash)) for j in range(i, len(ash))
                 if max(ash[i].l, ash[j].l) == l]
        return np.array(host_df.metric_gradient(aux, wit.w(), nuclei, pq=pq,
                                                pairs=pairs))
    cells = [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
             for n in range(len(ash)) if ash[n].l == l]
    return np.array(host_df.eri3_gradient(basis, aux, wit.gam(), nuclei,
                                          pq=pq, cells=cells))


def _ladder(backend: str, which: str, l: int, request,  # noqa: E741
            run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    wit, _, _, nuclei = _auxi()
    xs, classes = _ladder_launches(which, l)
    got = compile_grad(xs, len(nuclei), backend, tc, tmp_path,
                       timeout=COMPILE_BUDGET_S).run(run_device_binary)
    want = _ladder_host(which, l)
    scale = np.abs(want).max()
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    ref = (sum(wit.gmetric[c] for c in classes) if which == "metric"
           else sum(wit.eri3[c] for c in classes))
    big = max(np.abs(g).max() for g in (wit.gmetric if which == "metric"
                                        else wit.eri3).values())
    lib = float(np.abs(got[0] - ref).max() / big)
    print(f"\n{backend} aux-i {which} l={l}: {len(classes)} classes, "
          f"|G - host| / max|G| {max(errs):.2e}, |G - libcint| / max {lib:.2e}"
          f", bit-identical {bool(np.array_equal(got[0], got[1]))}")
    assert max(errs) <= DEV_TOL, (backend, which, l, errs)
    assert lib <= LIBCINT_TOL, (backend, which, l, lib)


LADDER = [("metric", lq) for lq in range(7)] + [("eri3", lp)
                                                for lp in range(7)]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("which,l", LADDER, ids=lambda x: str(x))
def test_cuda_df_gradient_ladder_through_i(which, l, request,  # noqa: E741
                                           run_device_binary, tmp_path):
    _ladder("cuda", which, l, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("which,l", LADDER, ids=lambda x: str(x))
def test_hip_df_gradient_ladder_through_i(which, l, request,  # noqa: E741
                                          run_device_binary, tmp_path):
    _ladder("hip", which, l, request, run_device_binary, tmp_path)


@pytest.mark.slow  # builds every class through L = 12
def test_the_ladder_parts_cover_every_class_of_the_ladder() -> None:
    """Host-only: the aux-i system's classes are the whole ladder, split
    over the parts by aux l, each once."""
    seen = {"metric": [], "eri3": []}
    for which, l in LADDER:  # noqa: E741
        seen[which] += _ladder_launches(which, l)[1]
    assert sorted(seen["metric"]) == [(p, q) for p in range(7)
                                      for q in range(p, 7)]
    assert sorted(seen["eri3"]) == [(a, b, p) for a in range(4)
                                    for b in range(a, 4) for p in range(7)]
