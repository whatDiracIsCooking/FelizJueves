"""The whole RHF nuclear gradient ON A CARD (#566): `gradient.rhf.
rhf_gradient` with every electronic term's kernels launched on the card,
the terms reduced on the host.

THE RUNNERS INJECTED (`rhf.Runners`), each term its own program(s), every
kernel in its default form:

- `overlap`, `kinetic`, `potential`: `gs_` / `gt_` / `gv_` per pair class
  (`grad_driver.one_electron_launches`, #562's), `M = W` for `S`, `D` for
  `T` and `V`, `gv_` with the Hellmann-Feynman term;
- `coulomb`: THE CHAIN ON THE CARD (#619, test_gjab_chain_ondevice.py's):
  `DQ` on the card, the J chain's plan keyed on its `Dmax`, every bra class
  raised by one (`grad_driver.raised_plan`), the raised `[p|q]` launched,
  then `gjab_` reading `J_p` where it lies (`grad_driver.gjab_tail`) -- no
  `J_p` from the host;
- `exchange`: every `gk_` class (`gk_driver`, #618's, every unit compiled
  alone under its 10 s budget), one `G` per launch, the classes' `G`
  summed on the host in launch order.

`V_nn^x` is the host's whichever runners ran (`rhf.rhf_gradient`). Each
program runs `RUNS` times; run `r` of every term is assembled into
`rhf_gradient` once per `r`, so each assembly is one whole device gradient.

THE WITNESSES are #550's gradient fixtures, read as test_rhf_gradient.py
reads them (`_witness`: the converged `D` and `W` as inputs, each term from
libcint at that density, `grad:h` split into `T` and `V`, the total
`nuc_grad_method().kernel()`'s), on H2O/STO-3G and H2O/6-31G: s and p
shells, so every K class reaches `(pp|pp)` and the raised `[p|q]` L = 5,
the milestone's one class above 4 (`slow`, by `tiers.by_l`).

THE BOUNDS, stated before any run (the issue's acceptance):

- every term and the total within `ABS_TOL = 1e-10` absolute of PySCF;
- the total, and every term, within `ABS_TOL` of the INTERPRETED
  `rhf_gradient` (`gradient.interpreted.interpreted_runners`, computed
  before the card is taken);
- RUN TO RUN: the two runs' totals, and every term, within `SPREAD_TOL =
  1e-13` absolute of each other. The issue asked for bit-identity, but the
  d1 kernels `atomicAdd` into `G` in an order the hardware chooses
  (docs/implementation/gradient.md, the #552 / #557 departure: no partials,
  no fixed-order reduction), so it is not available -- a first hip run
  differed by 2.8e-17 (STO-3G) and 1.8e-15 (6-31G) -- and the user
  restated the criterion as this bound. Whether the totals came out
  bit-identical is reported beside it.

The measured figures go to stdout and to the test's `user_properties` (the
kernel tier's pytest.xml).

Each compile is capped at `COMPILE_TIMEOUT_S = 300`, the user's per-compile
budget; `gk_` units at `gk_driver.UNIT_BUDGET_S`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from gk_driver import compile_gk, gk_class_launches
from grad_driver import (
    compile_grad,
    gjab_read_back,
    gjab_tail,
    one_electron_launches,
    raised_plan,
)
from plan_driver import (
    RUNS,
    compile_checked,
    compile_dq,
    dq_classes,
    driver,
    run,
    stage,
)
from test_j_chain_ondevice import _ket_data, _plan_from, _rows_of
from test_kernel_device import BACKEND, _toolchain
from test_rhf_gradient import _witness
from tiers import by_l

from felizjueves.gradient import rhf
from felizjueves.gradient.interpreted import interpreted_runners
from felizjueves.reference.boys import boys
from felizjueves.spec import Target

ABS_TOL = 1e-10  # absolute, against PySCF and the interpreted gradient
# Run to run, absolute, on every term and the total: the d1 kernels
# atomicAdd into G (docs/implementation/gradient.md, the #552/#557
# departure: no partials, no fixed-order reduction), so bit-identity across
# launches is not available; the user restated #566's criterion as this.
SPREAD_TOL = 1e-13
COMPILE_TIMEOUT_S = 300.0  # the user's per-compile budget
WITNESSES = ("h2o_sto3g_grad", "h2o_631g_grad")
ONE_E = {"overlap": Target.S, "kinetic": Target.T, "potential": Target.V}


def _lmax(name: str) -> int:
    return max(s.l for s in _witness(name)[1].shells)


# gk_ at (pp|pp) + 1 and the raised [p|q] at L = 4 lmax + 1: both s, p
PARAMS = [by_l(n, 4 * _lmax(n) + 1, id=n) for n in WITNESSES]


@cache
def _interpreted(name: str) -> rhf.RhfGradient:
    w, basis, nuclei = _witness(name)
    return rhf.rhf_gradient(basis, nuclei, w.density, w.weighted,
                            interpreted_runners(boys=boys))


def _one_electron(term: str, basis, nuclei, M, backend, tc,
                  workdir: Path) -> np.ndarray:
    launches = one_electron_launches(basis, nuclei, ONE_E[term], M)
    return compile_grad(launches, len(nuclei), backend, tc, workdir,
                        timeout=COMPILE_TIMEOUT_S).run()


def _coulomb(basis, nuclei, D, backend, tc, workdir: Path) -> np.ndarray:
    """`dE_J/dR` per run: DQ -> raised `[p|q]` -> `gjab_`, on the card."""
    classes = dq_classes(basis, D)
    dq_out, _ = compile_dq(classes, backend, tc, workdir / "dq",
                           timeout=COMPILE_TIMEOUT_S).run()
    dq_rows = _rows_of(classes, dq_out)
    plan = raised_plan(_plan_from(basis, dq_rows, 1,
                                  4 * max(s.l for s in basis.shells)))
    tail = gjab_tail(plan, basis, nuclei, D)
    drv = driver(plan, backend, workdir / "gen", tail=tail)
    binary = compile_checked(tc, drv, workdir / "driver",
                             timeout=COMPILE_TIMEOUT_S)
    ran = run(drv, binary, stage(plan, basis, D, drv,
                                 ket_data=_ket_data(basis, dq_rows)),
              workdir / "run", runs=RUNS)
    _, got = gjab_read_back(drv, ran, len(nuclei))
    return got


def _exchange(basis, nuclei, D, backend, tc, workdir: Path) -> np.ndarray:
    """`dE_K/dR` per run: every K class's `gk_` `G`, summed in launch
    order."""
    ls = sorted({s.l for s in basis.shells})
    classes = [(a, c, b, d) for a in ls for c in ls for b in ls for d in ls]
    got = compile_gk(gk_class_launches(basis, nuclei, D, classes),
                     len(nuclei), backend, tc, workdir).run()
    out = np.zeros((got.shape[0], len(nuclei), 3))
    for r in range(got.shape[0]):
        for g in got[r]:
            out[r] += g
    return out


def _device_terms(backend: str, name: str, tc, tmp_path: Path
                  ) -> dict[str, np.ndarray]:
    """Every electronic term's `G` per run, `[run][n_atom][3]`, unsigned
    (the runners' contract)."""
    w, basis, nuclei = _witness(name)
    D, W = w.density, w.weighted
    out = {t: _one_electron(t, basis, nuclei, W if t == "overlap" else D,
                            backend, tc, tmp_path / t) for t in ONE_E}
    out["coulomb"] = _coulomb(basis, nuclei, D, backend, tc,
                              tmp_path / "coulomb")
    out["exchange"] = _exchange(basis, nuclei, D, backend, tc,
                                tmp_path / "exchange")
    for t, g in out.items():
        assert g.shape == (RUNS, len(nuclei), 3), (t, g.shape)
        assert np.all(np.isfinite(g)), t
    return out


def _runners(terms: dict[str, np.ndarray], r: int, D, W) -> rhf.Runners:
    """Run `r` of every device term as `rhf.Runners`, each checking it is
    handed the matrix its program was launched with."""
    def runner(t: str):
        want = W if t == "overlap" else D

        def go(basis, nuclei, M):
            assert np.array_equal(np.asarray(M), want), t
            return terms[t][r]
        return go
    return rhf.Runners(**{t: runner(t) for t in rhf.TERMS[:-1]})


def _err(got, want) -> float:
    return float(np.abs(np.asarray(got) - np.asarray(want)).max())


def _check(backend: str, name: str, request, tmp_path: Path) -> None:
    w, basis, nuclei = _witness(name)
    interp = _interpreted(name)  # before the card is taken
    tc = _toolchain(backend, request)
    terms = _device_terms(backend, name, tc, tmp_path)
    runs = [rhf.rhf_gradient(basis, nuclei, w.density, w.weighted,
                             _runners(terms, r, w.density, w.weighted))
            for r in range(RUNS)]
    names = (*rhf.TERMS, "total")
    vs_pyscf = {t: max(_err(g.term(t), w.total if t == "total"
                            else w.terms[t]) for g in runs) for t in names}
    vs_interp = {t: max(_err(g.term(t), interp.term(t)) for g in runs)
                 for t in names}
    spread = {t: max(_err(runs[0].term(t), runs[r].term(t))
                     for r in range(1, RUNS)) for t in names}
    line = (f"{backend} {BACKEND[backend][1].name} {name} RHF gradient: "
            + "; ".join(f"{t} vs PySCF {vs_pyscf[t]:.2e}, vs interpreted "
                        f"{vs_interp[t]:.2e}, run-to-run {spread[t]:.2e}"
                        for t in names)
            + "; totals bit-identical "
            f"{bool(np.array_equal(runs[0].total, runs[1].total))}")
    request.node.user_properties.append(("rhf_gradient", line))
    print("\n" + line)
    for t in names:
        assert vs_pyscf[t] <= ABS_TOL, (
            f"{backend} {name} {t}: {vs_pyscf[t]:.2e} from PySCF, over "
            f"{ABS_TOL:g}")
        assert vs_interp[t] <= ABS_TOL, (
            f"{backend} {name} {t}: {vs_interp[t]:.2e} from the interpreted "
            f"rhf_gradient, over {ABS_TOL:g}")
        assert spread[t] <= SPREAD_TOL, (
            f"{backend} {name} {t}: the runs differ by {spread[t]:.2e}, over "
            f"{SPREAD_TOL:g}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", PARAMS)
def test_cuda_rhf_gradient_on_the_card(name, request, tmp_path) -> None:
    _check("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", PARAMS)
def test_hip_rhf_gradient_on_the_card(name, request, tmp_path) -> None:
    _check("hip", name, request, tmp_path)
