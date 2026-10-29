"""The Coulomb-energy gradient through the chain ON THE CARD (#619):
`DQ`, then the `[p|q]` kernels with the bra raised by one, then `gjab`,
with no `J_p` from the host, on the RTX 3080 and the RX 9060 XT.

THE CHAIN (test/grad_driver.py, "THE CHAIN ON THE CARD"), every arithmetic
step a generated kernel:

  1. DQ: every ket pair class's `Target.DQ` kernel (`plan_driver.dq_program`),
     `D_q` / `Dmax` read back -- the J chain's one round-trip -- and the J
     chain's plan keyed and sorted on them (test_j_chain_ondevice.py's
     `_plan_from`, one batch);
  2. `[p|q]`: that plan with every bra class raised by one
     (`grad_driver.raised_plan`, `kernel_gjab.gjab_pq_spec`'s classes) on
     the J chain's driver (`plan_driver.driver`), staged from the DQ
     kernel's `D_q` / `Dmax`;
  3. gjab: in the same process, on the same stream, one `gjab_` launch per
     bra pair class reading its raised `J_p` buffer where it lies
     (`grad_driver.gjab_tail`), into one zeroed `G`, read back per run.

THE BOUND, stated before any run: `|G_dev - G_host| <= DEV_TOL max|G|`,
`DEV_TOL = 1e-13`, against `gradient.host.coulomb_gradient` (held to finite
differences of the libcint-witnessed `coulomb_matrix`) -- test_grad_device.py's
bound for `gjab` over the host's `J_p`, not loosened. What the chain adds
is the card's `J_p`: the plan's MILD recurrence, the vendor's Boys and
`exp`, and the `[p|q]` atomics, each a few ulps of a `J_p` entry; the
interpreted chain is within 64 eps of the split-centre scale
(test_kernel_gjab.py), far inside 1e-13 of `max|G|`. The two runs are
compared with each other too, as a witness of the atomics' spread.

THE CELLS, each class in its default form (`kernel_gjab.gjab_lowstack`) and
the small ones again in the low-stack form forced:

  - H2/STO-3G: s only, raised `[p|q]` at L = 1 -- the PR gate's cell;
  - H2O/STO-3G: s, p on O, raised `[p|q]` to L = 5 (`slow`);
  - CO/cc-pVTZ (`co_ccpvtz_j`): f on both atoms, all 16 ordered classes
    (ss)-(ff) as the plan's shell pairs `i >= j` order them (asserted),
    (df), (fd), (ff) in the low-stack form, the rest unrolled; raised
    `[p|q]` to L = 13, so `slow` and
    `high_l` (tiers.py: off both kernel-tier selections, run by hand,
    `pytest -m "gpu and <backend> and high_l" -k gjab_chain`).

Every compile (the DQ program and the driver) is capped at
`COMPILE_TIMEOUT_S = 300`, the user's per-compile budget.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from grad_driver import gjab_read_back, gjab_tail, raised_plan
from oracle import basis as ob
from oracle import manifest
from plan_driver import (
    RUNS,
    compile_checked,
    counts,
    dq_classes,
    dq_program,
    dq_read_back,
    dq_stage,
    driver,
    stage,
)
from test_j_chain_ondevice import _host_dq_rows, _ket_data, _plan_from, _rows_of
from test_j_launched_ondevice import _check_counts
from test_kernel_device import BACKEND, _toolchain
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient.kernel_gjab import gjab_lowstack
from felizjueves.reference.pq import one_centre

DEV_TOL = 1e-13  # of max|G| (module docstring)
COMPILE_TIMEOUT_S = 300.0  # the user's per-compile budget
FF = "co_ccpvtz_j"
L_MAX_PLAN = {"h2_sto3g_j": 0, "h2o_sto3g_j": 4, FF: 12}  # the d0 plan's
# (name, the raised plan's top [p|q] L, forms)
CELLS = (("h2_sto3g_j", 1, ("default", "lowstack")),
         ("h2o_sto3g_j", 5, ("default", "lowstack")),
         (FF, 13, ("default",)))
PARAMS = [by_l((name, form), L, id=f"{name}-{form}")
          for name, L, forms in CELLS for form in forms]


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, np.asarray(fx.arrays["density"])


@cache
def _host(name: str) -> np.ndarray:
    basis, nuclei, D = _case(name)
    return np.array(host.coulomb_gradient(basis, D, nuclei,
                                          one_centre=one_centre))


def _chain(backend: str, name: str, form: str, request, run_device_binary,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    arch = BACKEND[backend][1]
    basis, nuclei, D = _case(name)
    lowstack = True if form == "lowstack" else None

    # 1. DQ on the card; the round-trip, and the plan keyed on its Dmax
    classes = dq_classes(basis, D)
    dqp = dq_program(classes, backend, tmp_path / "dq")
    dq_bin = compile_checked(tc, dqp, tmp_path / "dq_bin",
                             timeout=COMPILE_TIMEOUT_S)
    src, dst = tmp_path / "dq.in", tmp_path / "dq.out"
    dq_stage(classes).tofile(src)
    out = run_device_binary(dq_bin, str(src), str(dst), vram=backend)
    assert (f"fj-dq classes {len(classes)} launches {len(classes) * RUNS} "
            f"runs {RUNS}") in out, out
    dq_rows = _rows_of(classes, dq_read_back(
        classes, np.fromfile(dst, dtype=np.float64))[0])
    plan = raised_plan(_plan_from(basis, dq_rows, 1, L_MAX_PLAN[name]))

    # 2 and 3: the raised [p|q] and gjab, one process
    tail = gjab_tail(plan, basis, nuclei, D, lowstack=lowstack)
    gjabs = {kern.name for _, kern in tail.kernels}
    if name == FF:
        assert gjabs == {f"gjab_{a}{b}" for a in range(4)
                         for b in range(4)}, gjabs
    drv = driver(plan, backend, tmp_path / "gen", tail=tail)
    binary = compile_checked(tc, drv, tmp_path / "driver",
                             timeout=COMPILE_TIMEOUT_S)
    src, dst = tmp_path / "in.bin", tmp_path / "out.bin"
    stage(plan, basis, D, drv, ket_data=_ket_data(basis, dq_rows)).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    _check_counts(counts(out), drv, plan, warmup=0, reps=RUNS)
    ng = len(tail.kernels)
    assert (f"fj-gjab classes {ng} uploads_gjab {ng} gjab_launches "
            f"{ng * RUNS} g_memsets {RUNS} g_readbacks {RUNS}") in out, out
    jps, got = gjab_read_back(drv, np.fromfile(dst, dtype=np.float64),
                              len(nuclei))
    for r in range(RUNS):
        assert all(np.all(np.isfinite(jp)) for jp in jps[r].values())

    want = _host(name)
    scale = float(np.abs(want).max())
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    low = sorted(k.name for _, k in tail.kernels
                 if lowstack or gjab_lowstack((int(k.name[5]), int(k.name[6]))))
    print(f"\n{backend} {arch.name} {name} gjab chain ({form}): [p|q] "
          f"{' '.join(f'({c.lp},{c.lq})' for c in plan)} (L <= "
          f"{max(c.L for c in plan)}); {ng} gjab, low-stack {low}; "
          f"|G - host| / max|G| per run {errs}; run-to-run {spread:.2e}")
    assert max(errs) <= DEV_TOL, (
        f"{backend} {name} gjab chain: |G - host| / max|G| = {errs} > "
        f"{DEV_TOL:g}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cell", PARAMS)
def test_cuda_gjab_chain_matches_the_host(cell, request, run_device_binary,
                                          tmp_path) -> None:
    _chain("cuda", *cell, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cell", PARAMS)
def test_hip_gjab_chain_matches_the_host(cell, request, run_device_binary,
                                         tmp_path) -> None:
    _chain("hip", *cell, request, run_device_binary, tmp_path)


@pytest.mark.parametrize("cell", [by_l(n, L, id=n) for n, L, _ in CELLS[:2]])
def test_the_raised_plan_is_the_plans_bras_one_up(cell) -> None:
    """No card: `raised_plan` keeps every record and block and raises `lp`,
    and `gjab_tail` stages every record of every bra pair."""
    basis, nuclei, D = _case(cell)
    classes = dq_classes(basis, D)
    plan = _plan_from(basis, _host_dq_rows(basis, classes), 1,
                      L_MAX_PLAN[cell])
    up = raised_plan(plan)
    assert [(c.lp, c.lq) for c in up] == [(c.lp + 1, c.lq) for c in plan]
    for a, b in zip(plan, up, strict=True):
        assert a.kets == b.kets and a.bra_blocks == b.bra_blocks
        assert [(r.i, r.j, r.prim) for r in a.bras] == [
            (r.i, r.j, r.prim) for r in b.bras]
    tail = gjab_tail(up, basis, nuclei, D)
    rows = {}
    for cl in up:
        rows.setdefault(cl.lp, len(cl.bras))
    n_rec = sum(rows.values())
    assert tail.n_output == 3 * len(nuclei)
    assert tail.n_input == 15 * n_rec + basis.n_ao ** 2
