"""The skeleton Coulomb Hessian through the chain ON THE CARD (#690): `DQ`
and `hdq`, then the `[p|q]` kernels at both raised classes, then `hjab`,
with no `J_p` from the host, on the RTX 3080 and the RX 9060 XT.

THE CHAIN (test/hess_driver.py, "the Coulomb Hessian's chain on the
card"), every arithmetic step a generated kernel:

  1. DQ: every ket pair class's `Target.DQ` kernel on the shared runner,
     `D_q` / `Dmax` read back -- the J chain's round-trip -- and the J
     chain's plan keyed and sorted on them, tau 0 (test_j_chain_ondevice's
     `_plan_from`, one batch);
  2. hdq: every ket pair class's `hdq_` on the card, `D1` read back, and
     the HOST GATHER of the derivative kets per atom, axis and ket Hermite
     class (`hess_driver.gather_dkets`; the design's step 3 is the host's,
     as the J chain's sort is);
  3. ONE runner process (`plan_driver.driver`, `tail=hess_driver.
     hjab_tail`): the plan with every bra class raised by two
     (`grad_driver.raised_plan(plan, 2)`, `hjab_pq2_spec`'s classes) into
     `J<lp>` -- `Jp2` --, then the `[p|q]` kernels of `hjab_pq1_spec`'s
     classes over the derivative kets into each `(C, l)` slab of
     `Jp1[n_atom, 3, n_rows, n_p1]` -- the tail's first launches --, then
     one `hjab_` per bra pair class reading both where they lie, into one
     zeroed `H`, read back per run.

THE BOUND, stated before any run: `|H_dev - H_ref| <= DEV_TOL max(twin)`,
`DEV_TOL = 1e-14` (~45 eps), `twin` the reference's ABSOLUTE TWIN -- the
same sums with every factor in `|.|`, the size of what cancels: a d2 E_J's
same-atom blocks carry `(2 alpha)^2` and cancel to 1/650 (cc-pVDZ) and
1/1800 (STO-3G) of libcint's twin on H2O, ~1/5000 of the host's at
STO-3G, so `max|H|` is not its rounding scale (libcint's own
translational residual is 1.1e-12 of `max|H|` on H2O/STO-3G, 6e-16 of its
twin).
test_hess_j_device.py's per-class bound, not loosened. The reference is:

  - H2/STO-3G and `SF` (below): `hessian.host.coulomb_hessian` live, its
    `absolute=True` twin;
  - H2O over data/basis/'s STO-3G and cc-pVDZ: LIBCINT, the numeric
    golden `hjab_h2o_*_bse` (`oracle.numeric`'s `jab.d2`: `int2e_ipip1` /
    `_ipvip1` / `_ip1ip2` `_cart` contracted with `D (x) D` in J's pairing,
    the `_j` fixtures' densities) and its twin `hjab_h2o_*_bse_abs`, the
    same assembly in `|.|`. The host is held to the same golden at the
    same bound (H2O/STO-3G, `slow`, host-only), which ties the per-class
    reference to libcint.

The runs are compared with each other too (the atomics' spread).

THE CELLS, each class in its default form (`hjab` rolled from `la + lb =
3`, `hdq` from `lc + ld = 5`):

  - H2/STO-3G: s only, `[p|q]` at L = 2 -- the PR gate's cell;
  - H2O/STO-3G (BSE): s, p on O, `[p|q]` to L = 6 (`slow`);
  - H2O/cc-pVDZ (BSE, the file's general contraction): s, p, d, `[p|q]`
    to L = 10 (`slow`, `high_l`: run by hand, `pytest -m "gpu and
    <backend> and high_l" -k hjab_chain`);
  - `SF`: two atoms, each an s shell of two primitives and an f shell --
    the classes (ss), (sf), (fs), (ff), `hjab_33` / `hdq_33` rolled,
    `[p|q]` to L = 14 ((8, 6) and (7, 7)) (`slow`, `high_l`).

EVERY COMPILE: every unit alone is within the 120 s per-.cu budget
(data/measured/hjab.json); the driver, one runner over every unit of a
cell, is capped at `COMPILE_TIMEOUT_S` as a safety net, its wall-clock
printed.
"""

from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from grad_driver import raised_plan
from hess_driver import gather_dkets, hdq_on_card, hjab_tail, ket_pair_classes
from oracle import basis as ob
from oracle import manifest
from oracle import numeric as golden
from plan_driver import (
    RUNS,
    compile_checked,
    compile_dq,
    dq_classes,
    driver,
    read_back_tail,
    run,
    stage,
)
from test_j_chain_ondevice import _ket_data, _plan_from, _rows_of
from test_kernel_device import BACKEND, _toolchain
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.hessian import host
from felizjueves.reference.pq import one_centre

DEV_TOL = 1e-14  # of max(twin) (module docstring)
COMPILE_TIMEOUT_S = 1800.0  # the driver's safety net (module docstring)

_A, _B = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5)
SF = (Basis((Shell(0, _A, (1.6, 0.4), (0.6, 0.5)),
             Shell(3, _A, (0.7,), (1.0,)),
             Shell(0, _B, (1.3, 0.35), (0.55, 0.6)),
             Shell(3, _B, (0.6,), (1.0,)))),
      ((2.0, _A), (1.0, _B)))

GOLDEN = {"h2o_sto3g": "hjab_h2o_sto3g_bse", "h2o_ccpvdz": "hjab_h2o_ccpvdz_bse"}
# (cell, the top [p|q] L, the d0 plan's top L)
CELLS = (("h2_sto3g_j", 2, 0), ("h2o_sto3g", 6, 4), ("h2o_ccpvdz", 10, 8),
         ("sf", 14, 12))
PARAMS = [by_l(name, L, id=name) for name, L, _ in CELLS]
L_MAX_PLAN = {name: top for name, _, top in CELLS}


@cache
def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(7).standard_normal((n, n))
    return X @ X.T / n


@cache
def _case(name: str):
    """`(basis, nuclei, D)`."""
    if name == "sf":
        basis, nuclei = SF
        return basis, nuclei, _density(basis.n_ao)
    if name in GOLDEN:
        g = GOLDEN[name]
        nuclei = tuple((float(z), tuple(c)) for z, c in golden.atoms(g))
        return golden.basis(g), nuclei, np.asarray(golden.density(g))
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, np.asarray(fx.arrays["density"])


@cache
def _reference(name: str) -> tuple[np.ndarray, float]:
    """`(H, max(twin))`: libcint's for the H2O cells, the host's live."""
    if name in GOLDEN:
        g = GOLDEN[name]
        return (np.array(golden.want(g)),
                float(np.abs(golden.want(g + "_abs")).max()))
    basis, nuclei, D = _case(name)
    n = len(nuclei)
    H, tw = (np.array(host.coulomb_hessian(
        basis, D, nuclei, one_centre=one_centre, absolute=a)).reshape(
            n, 3, n, 3) for a in (False, True))
    return H, float(np.abs(tw).max())


def _chain(backend: str, name: str, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    arch = BACKEND[backend][1]
    basis, nuclei, D = _case(name)
    n_atom = len(nuclei)

    # 1. DQ on the card; the round-trip, and the plan keyed on its Dmax
    classes = dq_classes(basis, D)
    dq_out, _ = compile_dq(classes, backend, tc, tmp_path / "dq",
                           timeout=COMPILE_TIMEOUT_S).run()
    dq_rows = _rows_of(classes, dq_out)
    plan = _plan_from(basis, dq_rows, 1, L_MAX_PLAN[name])
    assert plan.tau == 0.0
    plan2 = raised_plan(plan, 2)

    # 2. hdq on the card; the host's gather
    d1s = hdq_on_card(basis, nuclei, D, backend, tc, tmp_path / "hdq",
                      timeout=COMPILE_TIMEOUT_S)
    assert sorted(d1s) == ket_pair_classes(basis)
    dkets = gather_dkets(d1s)

    # 3. Jp2, Jp1 and hjab, one process
    tail = hjab_tail(plan2, basis, nuclei, D, dkets, backend)
    drv = driver(plan2, backend, tmp_path / "gen", tail=tail)
    t0 = time.perf_counter()
    binary = compile_checked(tc, drv, tmp_path / "driver",
                             timeout=COMPILE_TIMEOUT_S)
    compile_s = time.perf_counter() - t0
    ran = run(drv, binary, stage(plan2, basis, D, drv,
                                 ket_data=_ket_data(basis, dq_rows)),
              tmp_path / "run", runs=RUNS)
    got = np.array(read_back_tail(drv, ran))
    assert got.shape == (RUNS, n_atom, 3, n_atom, 3)
    assert np.all(np.isfinite(got))

    want, twin = _reference(name)
    top = float(np.abs(want).max())
    errs = [float(np.abs(g - want).max() / twin) for g in got]
    of_max = [float(np.abs(g - want).max() / top) for g in got]
    spread = float(np.abs(got[0] - got[1]).max() / twin)
    hjabs = sorted({s.kernel.name for s in tail.steps
                    if s.kernel.name.startswith("hjab_")})
    pqs = sorted({(c.lp, c.lq) for c in plan2})
    n_pq1 = sum(1 for s in tail.steps if not s.kernel.name.startswith("hjab_"))
    print(f"\n{backend} {arch.name} {name} hjab chain: Jp2 [p|q] {pqs}, "
          f"{n_pq1} Jp1 launches, {hjabs}; driver compile {compile_s:.1f} s; "
          f"|H - ref| / max(twin) per run {errs} (of max|H| {of_max}); "
          f"run-to-run {spread:.2e}; twin / max|H| {twin / top:.0f}")
    assert max(errs) <= DEV_TOL, (
        f"{backend} {name} hjab chain: |H - ref| / max(twin) = {errs} > "
        f"{DEV_TOL:g}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", PARAMS)
def test_cuda_hjab_chain_matches_the_reference(name, request,
                                               tmp_path) -> None:
    _chain("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", PARAMS)
def test_hip_hjab_chain_matches_the_reference(name, request,
                                              tmp_path) -> None:
    _chain("hip", name, request, tmp_path)


# --- no card ------------------------------------------------------------------


@pytest.mark.slow
def test_the_host_is_the_libcint_golden_on_h2o_sto3g() -> None:
    """The host's whole `d2 E_J` on H2O/STO-3G (~40 s) against libcint, at
    the cells' bound: what ties the per-class reference to libcint."""
    name = "h2o_sto3g"
    basis, nuclei, D = _case(name)
    n = len(nuclei)
    H = np.array(host.coulomb_hessian(basis, D, nuclei,
                                      one_centre=one_centre)).reshape(
                                          n, 3, n, 3)
    want, twin = _reference(name)
    err = float(np.abs(H - want).max() / twin)
    print(f"\nhost vs libcint, H2O/STO-3G: {err:.2e} of max(twin), "
          f"{np.abs(H - want).max() / np.abs(want).max():.2e} of max|H|")
    assert err <= DEV_TOL


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_the_goldens_are_on_the_cells_inputs(name) -> None:
    """The golden and its twin are on the cell's basis and nuclei, and the
    twin bounds the golden."""
    basis, nuclei, D = _case(name)
    want, twin = _reference(name)
    assert want.shape == (len(nuclei), 3, len(nuclei), 3)
    assert D.shape == (basis.n_ao, basis.n_ao)
    assert twin >= float(np.abs(want).max())
    M = want.reshape(3 * len(nuclei), -1)
    assert np.abs(M - M.T).max() <= 1e-12 * np.abs(M).max()


def test_the_tail_stages_both_raised_classes() -> None:
    """No card: on H2O/STO-3G (the host's `D_q`, `D1`), the twice-raised
    plan keeps every bra record, the tail's `Jp1` launches cover every
    bra Hermite class against every derivative-ket group, and its `hjab`
    launches every bra record once."""
    from test_j_chain_ondevice import _host_dq_rows

    from felizjueves.gradient import host as gh
    from felizjueves.hessian.kernel_hdq import n_q1

    name = "h2o_sto3g"
    basis, nuclei, D = _case(name)
    classes = dq_classes(basis, D)
    plan = _plan_from(basis, _host_dq_rows(basis, classes), 1,
                      L_MAX_PLAN[name])
    plan2 = raised_plan(plan, 2)
    assert [(c.lp, c.lq) for c in plan2] == [(c.lp + 2, c.lq) for c in plan]
    d1s = {}
    for lc, ld in ket_pair_classes(basis):
        rec = gh.records(basis, nuclei, lc, ld)
        D1 = np.array(host.ket_derivative_records(basis, D, lc, ld))
        assert D1.shape == (6, len(rec["s_ab"]), n_q1(lc, ld))
        d1s[lc, ld] = (rec, D1)
    dkets = gather_dkets(d1s)
    assert {(g.C, g.l) for g in dkets} == {(C, l) for C in range(3)
                                           for l in range(3)}  # noqa: E741
    tail = hjab_tail(plan2, basis, nuclei, D, dkets, "cuda")
    hj = [s for s in tail.steps if s.kernel.name.startswith("hjab_")]
    pq = [s for s in tail.steps if not s.kernel.name.startswith("hjab_")]
    n_bra_L = len({c.lp for c in plan2})
    assert len(pq) == n_bra_L * len(dkets)
    assert sum(s.args["n_rec"] for s in hj) == sum(
        len(c.bras) for c in {c.lp: c for c in plan2}.values())
    assert tail.steps[0].zero[0] == "H" and tail.out == "H"
    n_ket = sum(len(g.eta) for g in dkets)
    assert n_ket == 2 * 3 * sum(len(r["s_ab"]) for r, _ in d1s.values())
