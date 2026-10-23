"""The whole molecule on the card (#144): `J_mn` from the LAUNCH PLAN, one
`J_p` buffer per bra class, on the RTX 3080 and the RX 9060 XT.

WHAT RUNS WHERE. `launch.launches` plans the molecule (#138); a driver
`main()` generated FROM THAT PLAN (`plan_driver.py`, harness) uploads each
bra class and each ket class ONCE, zeroes one `J_p` per bra class once per
run, launches every batch kernel of every `(lp, lq)` class into its bra
class's buffer (fp64 atomics across ket classes and batches), and reads each
`J_p` back. Eq. 20 and the scatter run host-side (`ao.staged._transform`),
as `jgen` does. The screen is REAL: `Ip`, `Iq`, `Dmax` are the plan's, `tau`
is the plan's, and the kets arrive in the plan's sorted order -- the inputs
are `ao.staged`'s own buffers, whose keys it checks bit for bit.

TWO ANCESTRIES, BOTH IN THE ASSERTION (#144's hazard: comparing the device
build only with the Python driver that generated its launch order would be
comparing a thing with itself at one remove):

  launch oracle  `launch.walk` over the SAME plan (#142): the plan walked in
                 Python with `reference.pq` and `fsum` -- equal to the host
                 sweep to 0. `|J_dev - J_oracle| <= 1e-13 max|J_oracle|`:
                 test_ao_j_ondevice.py's INTERP_TOL, inherited (the card
                 differs from `reference.pq` by FMA contraction and libm
                 `exp`/`sqrt` -- a different rounding, far inside it).
  libcint        the tracked `get_jk` fixture, `|J_dev - J_pyscf| <= 1e-12
                 scale_J` element by element (test_ao_j.py's JK_TOL, via
                 test_ao_j_ondevice.py). With `tau > 0` the comparison carries
                 `ao.j`'s stated screening claim on top, `|J(tau) - J(0)| <=
                 tau * Screened.scale` (#88) -- the screen's own error bound,
                 inherited, not a loosening: `JK_TOL scale_J + tau scale`.

THE fp64-ATOMIC RUN-TO-RUN TOLERANCE, stated where it is asserted
(`_run`). The driver runs the whole plan RUNS = 2 times in one process.
Each element of `J_p[lp]` receives exactly K = (kernels writing into that
bra class) atomic adds, one per kernel from lane 0, each adding a partial
`s_k` that the kernel's own (deterministic) arithmetic produced. fp64 atomics
are UNORDERED in general, so `J_p` is not promised bit-reproducible; however
the K partials are ordered, the sum is within `gamma_{K-1} SUM_k |s_k|` of
their exact sum, `gamma_n = n u / (1 - n u)`, and `SUM_k |s_k| <= A_p =
SUM_ket SUM_q |[p|q]| |D_q|` (every partial is a sub-sum of those terms).
So two runs differ by at most

    |J_p(run 1) - J_p(run 2)| <= 2 gamma_{K-1} A_p,   element by element,

with `A_p` computed on the host (`_abs_jp`, over EVERY ket, which bounds a
screened run's partials too). Whether the runs came out bit-identical is
printed -- an observation (stream-ordered launches make the order fixed
here), not the assertion.

THE CELLS (per card; each one driver, compiled and launched):

  h2_sto3g_j   tau 0, 1 batch   one class `(0, 0)`;
  h2o_sto3g_j  tau 0, 1 batch   nine classes, lp, lq in {0, 1, 2}: s and p
                                SHELLS, Hermite pair momenta up to 2;
  h2o_sto3g_j  tau 0, 3 batches eight split classes (atomics across batches
                                AND ket classes), `(0, 0)` refused -> 1;
  h2o_sto3g_j  tau 1e-3, 3      the screen firing (the oracle's `n_skipped`
                                > 0), the break on the card;
  h2o_ccpvdz_j tau 0, DISPATCHED  `dispatch.Resolver(arch, "wide")` injected
                                into `launches(resolve=)` (#145's deferred
                                bullet). 25 classes, lp, lq in 0..4 (d shells).

THE DISPATCHED CELL, honestly. On H2O/STO-3G (pair momenta <= 2, L <= 4) no
class has a measured row, so the resolver falls back to 1 everywhere and
would exercise nothing but its fallback. The smallest tracked J fixture that
reaches a MEASURED class is H2O/cc-pVDZ: its `(lp, lq) = (4, 3)` is
`pq_spec(7)`'s split, a SUBGRID class, and the resolver reads its row. On
this box's two cards the wide-T rows pick 1 batch at L = 7 (and so do the
near-T ones), so the dispatched count is 1 for every class -- 24 by
fallback, `(4, 3)` by measurement. The test asserts that each class runs at
`dispatch.choose(...)`'s count and that `(4, 3)` is measured; it prints the
choice. No tracked J fixture reaches a class whose measured choice is > 1
(that needs L >= 13: pair momenta of 7, beyond any fixture's shells).

COMPILE-ONLY COMPANION. `test_*_driver_compiles` (markers `cuda` / `hip`,
no `gpu`) generates and compiles every cell's driver where no card is: with
devtools/config.sh's NVCC_FLAGS / HIPCC_FLAGS, and a warning LOCATED in
generated text -- the driver or an emitted kernel unit -- fails it (#92),
as it fails the kernel tier's compile phase. The device tests compile the
same way, so a warning fails them too.

THE BUILD WALL-CLOCK (#146) and #147's sort-key A/B are measurement and
live in measure/measure_j_launched_ondevice.py (`devtools/measure.sh builds`
and `devtools/measure.sh sort-key`); each times a driver `_check` below has
checked first, so the timed binary is the checked binary.

THE CHAIN (#147). `measured_store.CHAIN_INPUT` is test_ao_screening.py's
two-H2 chain, the extended input on which a decay screen bites and whose
every T stays inside the device Boys' witnessed [0, 500] (measured_store's
docstring, THE SORT KEY, says why not four). It has no libcint fixture, so
its outside witness is the host sweep's J(0) (`ao.j.coulomb_matrix`, the
code held to libcint on every fixture) to `ORACLE_TOL max|J0| + tau scale`;
`test_<backend>_the_chain_under_both_keys` checks it, both keys, untimed.

WHAT SKIPS: the device tests are `gpu` plus `cuda` or `hip` and take the
compiler and device fixtures, so on a box without the card they SKIP -- the
run is green and empty, and devtools/kernel-tier.sh exits 2. The VRAM probe
(`run_device_binary(..., vram=)`) runs first and fails with both figures
rather than launching into an out-of-memory error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
import test_ao_j_ondevice
from oracle import basis as ob
from oracle import manifest
from plan_driver import (
    RUNS,
    Driver,
    compile_checked,
    counts,
    driver,
    generated_warnings,
    read_back,
    stage,
    transform,
)
from test_ao_screening import _h2_chain, _hilbert
from test_kernel_device import BACKEND, _toolchain

from felizjueves import dispatch, measured_store
from felizjueves.ao.density import checked_density
from felizjueves.ao.j import (
    coulomb_matrix,
    coulomb_matrix_and_scale,
    pair_bound_pq,
)
from felizjueves.ao.staged import (
    _bra_buffers,
    _dglobal,
    _ket_buffers,
    _kets_checked,
)
from felizjueves.kernel import bra_functions, ket_functions
from felizjueves.launch import LaunchPlan, class_spec, launches, walk
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.pq import fundamental
from felizjueves.reference.pq import pq as ref_pq

JK_TOL = test_ao_j_ondevice.JK_TOL  # 1e-12 scale_J vs libcint, inherited
ORACLE_TOL = test_ao_j_ondevice.INTERP_TOL  # 1e-13 max|J| vs reference.pq
U = 2.0**-53
BOUND = partial(pair_bound_pq, pq=ref_pq)  # the host sweep's bound, exactly
DISPATCH_INPUT = "wide"
DISPATCH = "dispatch"

# (fixture, tau, n_batches or DISPATCH)
CELLS = (
    ("h2_sto3g_j", 0.0, 1),
    ("h2o_sto3g_j", 0.0, 1),
    ("h2o_sto3g_j", 0.0, 3),
    ("h2o_sto3g_j", 1e-3, 3),
    ("h2o_ccpvdz_j", 0.0, DISPATCH),
)
# H2O/cc-pVDZ dispatched is ~90 s on the RTX 3080, so it is `slow`, off the
# PR gate (kernel-tier.sh --full runs it); the STO-3G cells stay on it.
PARAMS = [pytest.param(*c, id=f"{c[0]}-tau{c[1]:g}-"
                       + (c[2] if c[2] == DISPATCH else f"b{c[2]}"),
                       marks=(pytest.mark.slow,)
                       if c[0] == "h2o_ccpvdz_j" else ())
          for c in CELLS]


@cache
def _fixture(name: str):
    """(fixture or None, basis, D). `measured_store.CHAIN_INPUT` is the
    two-H2 chain of test_ao_screening.py, which has no libcint fixture
    (None): `_check` holds it to the host sweep's J(0) instead."""
    if name == measured_store.CHAIN_INPUT:
        basis = _h2_chain(2)
        return None, basis, _hilbert(basis.n_ao)
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx), np.array(fx.density)


@cache
def _scale_j(name: str) -> np.ndarray:
    """`scale_J` of `ao.j.coulomb_matrix_and_scale` (the JK_TOL's scale)."""
    _, basis, D = _fixture(name)
    return np.array(coulomb_matrix_and_scale(basis, D, pq=ref_pq)[1])


def _plan(backend: str, name: str, tau: float, n,
          density_sort: bool = True) -> LaunchPlan:
    _, basis, D = _fixture(name)
    if n == DISPATCH:
        arch = BACKEND[backend][1]
        return launches(basis, D, bound=BOUND, n_batches=1, tau=tau,
                        density_sort=density_sort, strict=False,
                        resolve=dispatch.Resolver(arch, DISPATCH_INPUT))
    return launches(basis, D, bound=BOUND, n_batches=n, tau=tau,
                    density_sort=density_sort, strict=False)


def _r_all(L: int, PQ: tuple[float, float, float], theta: float):
    """Every `[r](0)` with |r| <= L for one pair, by `reference.pq`'s own
    recurrence and fundamentals, memoised over the pair (for `_abs_jp`)."""
    memo: dict = {}

    def node(r, m):
        key = (r, m)
        if key in memo:
            return memo[key]
        for d in range(3):
            if r[d]:
                down = r[:d] + (r[d] - 1,) + r[d + 1:]
                v = PQ[d] * node(down, m + 1)
                if r[d] >= 2:
                    v += (r[d] - 1) * node(r[:d] + (r[d] - 2,) + r[d + 1:],
                                           m + 1)
                break
        else:
            v = fundamental(m, PQ, theta)
        memo[key] = v
        return v

    return lambda r: node(r, 0)


def _abs_jp(plan: LaunchPlan, name: str, *, basis=None, D=None,
            ket_data=None) -> dict[int, np.ndarray]:
    """`A_p = SUM_ket SUM_q |[p|q]| |D_q|` per bra record and `p`, per bra
    class, over EVERY ket of every class (module docstring). `ket_data=`
    (#157) takes `D_q` from `ao.staged`'s hook -- the DQ kernel's, staged
    exactly as the driver's input -- instead of the host pre-contraction;
    `basis` / `D` then give the molecule, and `name` is only a label."""
    if basis is None:
        _, basis, D = _fixture(name)
    rows = checked_density(basis, D)
    # None under the product key, and with ket_data (the chain's own Dmax)
    dglobal = None if ket_data is not None else _dglobal(plan, basis, rows)
    out: dict[int, np.ndarray] = {}
    for cl in plan:
        spec = class_spec(cl.lp, cl.lq, cl.n_batches)
        ps, qs = bra_functions(spec), ket_functions(spec)
        b = _bra_buffers(basis, cl.bras)
        k = (_ket_buffers(basis, rows, cl.kets, len(qs), dglobal)
             if ket_data is None
             else _kets_checked(cl.kets, len(qs), ket_data(cl.kets, len(qs))))
        acc = out.setdefault(cl.lp, np.zeros((len(cl.bras), len(ps))))
        absD = np.abs(k.D)
        for i in range(len(cl.bras)):
            for j in range(len(cl.kets)):
                PQ = tuple(float(x) for x in b.P[i] - k.Q[j])
                th = math.sqrt(b.zeta[i] * k.eta[j] / (b.zeta[i] + k.eta[j]))
                R = _r_all(cl.L, PQ, th)
                for ip, p in enumerate(ps):
                    acc[i, ip] += sum(
                        abs(R((p[0] + q[0], p[1] + q[1], p[2] + q[2]))) * d
                        for q, d in zip(qs, absD[j], strict=True))
    return out


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


def _dispatch_report(backend: str, plan: LaunchPlan) -> str:
    """Assert every class ran at `dispatch.choose`'s count; describe it."""
    arch = BACKEND[backend][1]
    measured = []
    for cl in plan:
        ch = dispatch.choose(cl.lp, cl.lq, arch, DISPATCH_INPUT)
        assert cl.n_batches == ch.n_batches, (cl.lp, cl.lq, ch)
        if ch.measured:
            measured.append(ch)
    assert [(c.lp, c.lq) for c in measured] == [(4, 3)], measured
    return "; ".join(
        f"dispatch {DISPATCH_INPUT} ({c.lp},{c.lq}) -> {c.n_batches} measured "
        f"(median {c.median_ns} ns, runner-up {c.runner_up} by "
        f"{c.margin_ns} ns)" for c in measured) + (
        f"; {len(plan) - len(measured)} classes by fallback -> "
        f"{dispatch.FALLBACK}")


# ---------------------------------------------------------------------------
# The device run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checked:
    """One cell's driver, compiled, run in its correctness mode and checked
    (`_check`): what a timing of the SAME binary is compared against."""

    plan: LaunchPlan
    drv: Driver
    binary: Path
    src: Path  # its staged input
    jp: dict[int, np.ndarray]  # the checked run's J_p, per bra class
    bound: dict[int, np.ndarray]  # 2 gamma_{K-1} A_p, per bra class
    n_pairs: int
    n_skipped: int


def _built(backend: str, tc, name: str, tau: float, n, density_sort: bool,
           workdir: Path) -> tuple[LaunchPlan, Driver, Path, Path]:
    """Plan, generate, compile (#92's filter) and stage one cell."""
    _, basis, D = _fixture(name)
    plan = _plan(backend, name, tau, n, density_sort)
    drv = driver(plan, backend, workdir / "gen")
    binary = compile_checked(tc, drv, workdir / "driver")
    src = workdir / "in.bin"
    stage(plan, basis, D, drv).tofile(src)
    return plan, drv, binary, src


def _check_counts(c: dict[str, int], drv: Driver, plan: LaunchPlan, *,
                  warmup: int, reps: int) -> None:
    """The plan's shape, as the driver counted it, over `warmup + reps`
    builds in one process."""
    runs = warmup + reps
    nb, nk, nl = len(drv.bra), len(drv.ket), len(drv.launches)
    c = {k: v for k, v in c.items() if k != "upload_ns"}
    assert c == dict(runs=runs, bra_classes=nb, ket_classes=nk,
                     uploads_bra=nb, uploads_ket=nk, jp_buffers=nb,
                     memsets=nb * runs, launches=nl * runs,
                     readbacks=nb * runs, plan_launches=nl, warmup=warmup,
                     reps=reps), c
    assert nl == sum(len(cl.kernels) for cl in plan)


def _run(backend: str, name: str, tau: float, n, request,
         run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    _check(backend, name, tau, n, True, tc, run_device_binary, tmp_path)


def _check(backend: str, name: str, tau: float, n, density_sort: bool, tc,
           run_device_binary, workdir: Path) -> Checked:
    """Build one cell and run it in its correctness mode: the counts, the
    run-to-run bound, the launch oracle and libcint (module docstring)."""
    fx, basis, D = _fixture(name)
    plan, drv, binary, src = _built(backend, tc, name, tau, n, density_sort,
                                    workdir)
    dst = workdir / "out.bin"
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    _check_counts(counts(out), drv, plan, warmup=0, reps=RUNS)
    nb, nk, nl = len(drv.bra), len(drv.ket), len(drv.launches)

    runs = read_back(drv, np.fromfile(dst, dtype=np.float64))
    for r in runs:
        for lp, jp in r.items():
            assert np.all(np.isfinite(jp)), f"non-finite J_p[{lp}]"

    # the fp64-atomic run-to-run tolerance (module docstring):
    #   |J_p(1) - J_p(2)| <= 2 gamma_{K-1} A_p, K = kernels into J_p[lp]
    A = _abs_jp(plan, name)
    bounds = {lp: 2.0 * _gamma(K - 1) * A[lp]
              for lp, K in drv.writers.items()}
    bitwise = True
    for lp, K in drv.writers.items():
        diff = np.abs(runs[0][lp] - runs[1][lp])
        bitwise &= bool(np.all(diff == 0.0))
        assert np.all(diff <= bounds[lp]), (
            f"J_p[{lp}] run to run: worst {float(np.max(diff)):.3e} over "
            f"the {K}-atomic bound")

    J = transform(basis, plan, runs[0])
    oracle = walk(plan, basis, D, pq=ref_pq)
    Jo = np.array(oracle.J)
    if fx is not None:
        ref = fx.arrays[manifest.J_ARRAY]
        scale = _scale_j(name)
        allowed = JK_TOL * scale + tau * np.array(oracle.scale)
        outside = "get_jk"
    else:
        # No libcint fixture (the chain): the host sweep's unscreened J(0),
        # through `ao.j.coulomb_matrix` -- the code witnessed against libcint
        # on every fixture -- with the oracle's bound and the screen's own
        # error claim on top, both inherited: ORACLE_TOL max|J0| + tau scale.
        ref = np.array(coulomb_matrix(basis, D, pq=ref_pq))
        scale = np.full_like(ref, np.max(np.abs(ref)))
        allowed = ORACLE_TOL * scale + tau * np.array(oracle.scale)
        outside = "host J(0)"
    rel = float(np.max(np.abs(J - Jo)) / np.max(np.abs(Jo)))
    jk = float(np.max(np.abs(J - ref) / scale))

    arch = BACKEND[backend][1]
    ran = " ".join(f"({cl.lp},{cl.lq})b{cl.n_batches}" for cl in plan)
    extra = ("; " + _dispatch_report(backend, plan)) if n == DISPATCH else ""
    print(f"\n{backend} {arch.name} {name} tau={tau:g} n_batches={n} "
          f"density_sort={density_sort}: "
          f"{len(plan)} classes [{ran}], refused {list(plan.refused)}; "
          f"{nl} kernels per run, {nb} J_p buffers, {nk} ket uploads; "
          f"oracle skipped {oracle.n_skipped} of {oracle.n_pairs}; "
          f"max|J - {outside}|/scale = {jk:.2e} (tol "
          f"{JK_TOL if fx is not None else ORACLE_TOL:g}"
          f"{' + tau scale' if tau else ''}); "
          f"max|J - oracle|/max|J| = {rel:.2e} (tol {ORACLE_TOL:g}); "
          f"runs bit-identical: {bitwise}{extra}")

    if tau > 0.0 and density_sort:
        assert oracle.n_skipped > 0, "the screen never fired"
    elif tau == 0.0:
        assert oracle.n_skipped == 0
    assert rel <= ORACLE_TOL, (
        f"{name} on {arch.name}: J vs the launch oracle, worst {rel:.2e}")
    assert np.all(np.abs(J - ref) <= allowed), (
        f"{name} on {arch.name}: J vs {outside}, worst err/scale {jk:.2e}")
    return Checked(plan, drv, binary, src, runs[0], bounds, oracle.n_pairs,
                   oracle.n_skipped)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), PARAMS)
def test_cuda_j_from_the_launch_plan(name, tau, n, request, run_device_binary,
                                     tmp_path) -> None:
    """The whole molecule's `J` from the plan on the RTX 3080."""
    _run("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), PARAMS)
def test_hip_j_from_the_launch_plan(name, tau, n, request, run_device_binary,
                                    tmp_path) -> None:
    """The whole molecule's `J` from the plan on the RX 9060 XT."""
    _run("hip", name, tau, n, request, run_device_binary, tmp_path)


# #147's extended input under both keys, in the order the A/B times them
# (measure/measure_j_launched_ondevice.py: product first on even cells,
# global first on odd ones), checked here, never timed.
CHAIN_ARMS = [pytest.param(tau, ds, id=f"tau{tau:g}-sort{int(ds)}")
              for i, (fixture, tau, _) in enumerate(
                  measured_store.SORT_KEY_CELLS)
              if fixture == measured_store.CHAIN_INPUT
              for ds in ((True, False) if i % 2 == 0 else (False, True))]


def _chain_arm(backend, tau, density_sort, request, run_device_binary,
               tmp_path) -> None:
    tc = _toolchain(backend, request)
    ck = _check(backend, measured_store.CHAIN_INPUT, tau, 1, density_sort, tc,
                run_device_binary, tmp_path)
    if not density_sort:
        assert ck.n_skipped == 0  # the global key screens nothing here


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("tau", "density_sort"), CHAIN_ARMS)
def test_cuda_the_chain_under_both_keys(tau, density_sort, request,
                                        run_device_binary, tmp_path) -> None:
    """#147's extended input, both arms, checked (not timed) on the 3080."""
    _chain_arm("cuda", tau, density_sort, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("tau", "density_sort"), CHAIN_ARMS)
def test_hip_the_chain_under_both_keys(tau, density_sort, request,
                                       run_device_binary, tmp_path) -> None:
    """#147's extended input, both arms, checked (not timed) on the 9060 XT."""
    _chain_arm("hip", tau, density_sort, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companion (`cuda` / `hip`, no `gpu`): no card needed
# ---------------------------------------------------------------------------


def _compile_only(backend: str, tc, name: str, tau: float, n,
                  tmp_path: Path) -> None:
    plan = _plan(backend, name, tau, n)
    drv = driver(plan, backend, tmp_path / "gen")
    compile_checked(tc, drv, tmp_path / "driver")


@pytest.mark.cuda
@pytest.mark.parametrize(("name", "tau", "n"), PARAMS)
def test_cuda_driver_compiles(name, tau, n, nvcc, tmp_path) -> None:
    _compile_only("cuda", nvcc, name, tau, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("name", "tau", "n"), PARAMS)
def test_hip_driver_compiles(name, tau, n, hipcc, tmp_path) -> None:
    _compile_only("hip", hipcc, name, tau, n, tmp_path)


def _planted_warning_fails(backend: str, tc, tmp_path: Path) -> None:
    """A warning planted in an emitted unit fails `compile_checked`, naming
    the unit: the #92 ratchet is live on this path, not merely parsed."""
    plan = _plan(backend, "h2_sto3g_j", 0.0, 1)
    drv = driver(plan, backend, tmp_path / "gen")
    unit = drv.units[0]
    unit.write_text(unit.read_text() + "\n__device__ int fj_planted() "
                    "{ int fj_unused; return 0; }\n")
    with pytest.raises(AssertionError, match=r"warning\(s\) in generated text"
                       ) as e:
        compile_checked(tc, drv, tmp_path / "driver")
    assert unit.name in str(e.value)


@pytest.mark.cuda
def test_cuda_a_warning_in_generated_text_fails(nvcc, tmp_path) -> None:
    _planted_warning_fails("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_a_warning_in_generated_text_fails(hipcc, tmp_path) -> None:
    _planted_warning_fails("hip", hipcc, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the driver's text and data, nothing compiled or launched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("n", [1, 3])
def test_the_driver_is_the_plans_shape(backend, n, tmp_path):
    """One `J_p` per bra class, one upload per ket class, every batch kernel
    of every class launched once per run into its bra class's buffer, each
    one ADDING, and each unit `emit`'s text of that kernel."""
    plan = _plan(backend, "h2o_sto3g_j", 0.0, n)
    drv = driver(plan, backend, tmp_path)
    emit, arch, _ = BACKEND[backend]
    assert [lp for lp, _, _ in drv.bra] == sorted({cl.lp for cl in plan})
    assert [lq for lq, _, _ in drv.ket] == sorted({cl.lq for cl in plan})
    assert [(x.lp, x.lq) for x in drv.launches] == [
        (cl.lp, cl.lq) for cl in plan for _ in cl.kernels]
    for x, unit in zip(drv.launches, drv.units, strict=True):
        assert unit.read_text() == emit(x.kernel, arch)
        assert "atomicAdd" in unit.read_text(), f"{x.kernel.name} does not add"
        assert f"{x.ns}::{x.kernel.name}<<<" in drv.source
    assert drv.source.count("<<<") == len(drv.launches)
    assert f"static const int NB = {len(drv.bra)}, NK = {len(drv.ket)}," in (
        drv.source)
    # a one-batch class's kernel is atomized HERE, never in the builder
    if n == 1:
        built = {k.name: k for cl in plan for k in cl.kernels}
        assert all("atomicAdd" not in emit(k, arch) for k in built.values())


def test_the_staged_input_is_the_plans():
    """The packed input carries the plan's tau, the bra records in pair
    order and the kets in the plan's sorted order, with keys bit-exact."""
    _, basis, D = _fixture("h2o_sto3g_j")
    plan = launches(basis, D, bound=BOUND, n_batches=1, tau=1e-3,
                    strict=False)
    rows = checked_density(basis, D)
    first = {}
    for cl in plan:
        first.setdefault(cl.lq, cl)
    for cl in first.values():
        k = _ket_buffers(basis, rows, cl.kets, len(ket_functions(
            class_spec(cl.lp, cl.lq))), None)
        assert list(k.Iq * k.Dmax) == [r.key for r in cl.kets]
        assert all(a >= b for a, b in zip(k.Iq * k.Dmax, (k.Iq * k.Dmax)[1:],
                                          strict=False))


def test_stage_packs_every_class_once(tmp_path):
    _, basis, D = _fixture("h2o_sto3g_j")
    plan = launches(basis, D, bound=BOUND, n_batches=3, tau=1e-3,
                    strict=False)
    drv = driver(plan, "cuda", tmp_path)
    data = stage(plan, basis, D, drv)
    assert data.size == drv.n_input and data[0] == 1e-3
    n_bra = {cl.lp: len(cl.bras) for cl in plan}
    n_ket = {cl.lq: len(cl.kets) for cl in plan}
    assert [(lp, n) for lp, n, _ in drv.bra] == sorted(n_bra.items())
    assert [(lq, n) for lq, n, _ in drv.ket] == sorted(n_ket.items())


def test_transform_of_the_oracles_jp_is_the_oracle():
    """Eq. 20 + scatter (`transform`) on `J_p` from `reference.pq` gives the
    oracle's `J`: the host half of the device path, checked without a card."""
    _, basis, D = _fixture("h2_sto3g_j")
    plan = launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)
    (cl,) = plan.classes
    b = _bra_buffers(basis, cl.bras)
    k = _ket_buffers(basis, checked_density(basis, D), cl.kets, 1, None)
    jp = np.zeros((len(cl.bras), 1))
    for i in range(len(cl.bras)):
        for j in range(len(cl.kets)):
            PQ = tuple(float(x) for x in b.P[i] - k.Q[j])
            th = math.sqrt(b.zeta[i] * k.eta[j] / (b.zeta[i] + k.eta[j]))
            jp[i, 0] += ref_pq((0, 0, 0), (0, 0, 0), PQ, th) * k.D[j, 0]
    J = transform(basis, plan, {0: jp})
    Jo = np.array(walk(plan, basis, D, pq=ref_pq).J)
    assert float(np.max(np.abs(J - Jo)) / np.max(np.abs(Jo))) <= ORACLE_TOL


def test_abs_jp_bounds_the_signed_jp():
    """`A_p` (the run-to-run tolerance's scale) bounds `|J_p|` and is the
    `reference.pq` recurrence's: on H2 it equals `SUM |[0|0] D|` directly."""
    _, basis, D = _fixture("h2_sto3g_j")
    plan = launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)
    (cl,) = plan.classes
    A = _abs_jp(plan, "h2_sto3g_j")[0]
    b = _bra_buffers(basis, cl.bras)
    k = _ket_buffers(basis, checked_density(basis, D), cl.kets, 1, None)
    want = np.zeros_like(A)
    for i in range(len(cl.bras)):
        for j in range(len(cl.kets)):
            PQ = tuple(float(x) for x in b.P[i] - k.Q[j])
            th = math.sqrt(b.zeta[i] * k.eta[j] / (b.zeta[i] + k.eta[j]))
            want[i, 0] += abs(ref_pq((0, 0, 0), (0, 0, 0), PQ, th)) * abs(
                k.D[j, 0])
    assert np.allclose(A, want, rtol=1e-14, atol=0.0)


def test_the_recurrence_matches_reference_pq():
    """`_r_all` is `reference.pq.one_centre`'s recurrence, memoised per pair."""
    from felizjueves.reference.pq import one_centre
    PQ, th = (0.3, -0.7, 1.1), 0.9
    R = _r_all(4, PQ, th)
    for r in hermite_targets(4):
        assert R(r) == one_centre(r, 0, PQ, th)


def test_the_warning_filter_reads_both_spellings(tmp_path):
    """#92's filter: a located warning in the driver or a kernel unit counts,
    in nvcc's and clang's spelling; one in a vendor header does not."""
    plan = launches(*_fixture("h2_sto3g_j")[1:], bound=BOUND, n_batches=1,
                    tau=0.0)
    drv = driver(plan, "cuda", tmp_path)
    unit = drv.units[0]
    diag = "\n".join([
        f'{drv.path}(12): warning #177-D: variable "x" was declared but never '
        "referenced",
        f"{unit.name}:40:5: warning: unused variable 'y' [-Wunused-variable]",
        "/usr/local/cuda/include/cuda_runtime.h(99): warning #1: vendor",
    ])
    got = generated_warnings(drv, diag)
    assert got == [
        f'{drv.path.name}, line 12: #177-D: variable "x" was declared but '
        "never referenced",
        f"{unit.name}, line 40: unused variable 'y' [-Wunused-variable]",
    ], got
    assert generated_warnings(drv, diag.splitlines()[2]) == []
