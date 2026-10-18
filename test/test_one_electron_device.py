"""S, T and V on the card (#242): every pair class through (dd) against the
host, and S, T and V of H2O/STO-3G and H2O/cc-pVDZ against libcint, on the
RTX 3080 and the RX 9060 XT; and, `slow` (#244, the deferred tier), every
pair class through (ff) against the host -- the predecessor's
`sgen`/`tgen`/`vgen` parity bar.

THE BUILD is `one_electron_driver.build_on_card`: one program per target of
one launch (`launch.one_electron_launch`, #241), every pair class's kernel
(`kernel_s` / `kernel_t` / `kernel_v`, #237-#239) launched once into one
zeroed matrix, `RUNS` times. Under devtools/kernel-tier.sh a class whose
unit the generate phase wrote (`--one-e-l 2`: `la <= lb <= 2`) includes
THAT file, asserted byte-identical to `emit(...)` and already through the
tier's compile phase and #92's filter; a class `la > lb` (a basis whose p or
d shells precede its s shells reaches them: H2O's O p before H s) is
emitted by the driver, the same text. Every program is compiled with
devtools/config.sh's flags and FAILS on a warning located in generated text.

PART 1, PER PAIR CLASS, THROUGH (dd). HF/cc-pVTZ's launch, its classes
restricted to `la, lb <= 2` -- all nine ordered classes `(0..2, 0..2)` are
reached (F's d before H's s, p and d), each over every record of every shell
pair of that class -- against `ao.overlap` / `ao.kinetic` / `ao.potential`
with `pairs=` those shell pairs. Every element is owned by exactly one shell
pair, so each class is checked on its own elements, and every element no
kept pair owns must come back exactly 0.0. Its nuclei are the fixture's
atoms; V's Boys argument reaches T ~ 1.25e5 there (F 1s against H), inside
the wide `fj_boys` the V units carry (#266) and inside `reference.boys`.

PART 1 THROUGH (ff), `slow` (#244). The same launch, unrestricted: all
sixteen ordered classes `(0..3, 0..3)` (F's f against H's s, p and d, and
F's own `(ff)` diagonal pair), the same check under the same bound. `V` at
`(ff)` is Hermite order 6 and `T` at `(ff)` reads axis tables at `lb + 2 =
5`: nothing above `L = 8` is compiled. Under the tier with `--one-e-l 3`
the `la <= lb` units are the generate phase's; without it the driver emits
them, the same text. Its compile-only companions are `slow` too.

PART 2, PER MOLECULE. H2O/STO-3G and H2O/cc-pVDZ, every class at once, S, T
and V against the tracked libcint fixture arrays (`int1e_ovlp`, `int1e_kin`,
`int1e_nuc`, #240) at test_ao_overlap.py's stated tolerance, unchanged:
`|M_dev - M_libcint| <= ONE_E_TOL one_e_scale(E_M)` (1e-12, the absolute
value scale `test_ao_{overlap,kinetic,potential}.py` state), and against the
host under the bound below.

THE DEVICE-VERSUS-HOST BOUND, per element, stated before any run, with
`u = 2^-53`, `gamma_n = n u / (1 - n u)`, `L = la + lb` and `K` the number
of records of the shell pair that owns the element (each adds once, in an
order the fp64 atomics choose):

    |M_dev - M_host| <= (2 gamma_{N + K} + delta) Mabs,

`Mabs` the sum over the owning pair's records of every term's ABSOLUTE
VALUE, each factor from its absolute-value recurrence: `Eabs` is the `E`
recurrence of `ao.hermite` from the SAME seed `exp(zeta X_PA X_PB)` raised
over `|X_PA|`, `|X_PB|` (test_j_chain_ondevice.py's `_e_abs`, and
`|E| <= Eabs` there), and for V `Rabs` is `[t]^(0)`'s recurrence over
`|P - C|` from `(2/sqrt(pi)) theta (2 theta^2)^m F_m` -- a polynomial with
non-negative coefficients in those, so the same on any recurrence path
(MILD's or `one_centre`'s) and `|[t]| <= Rabs` on each. Each side computes
each term within `gamma_N` of exact relative to its absolute value and adds
the `K` terms within `gamma_K` more, so the two sides differ by at most
twice that; FMA contraction only removes roundings. `N`, per target:

  S  `N_E(L) + 1`: `E`'s `N_E = 4L + 8` (test_j_chain_ondevice.py: above
     the `3L + 8` roundings of the recurrence, the 3D product and an `exp`
     within an ulp on each vendor's libm and the host's), times `s`.
  T  `N_E(L + 2) + 13`: the axis tables are built to `lb + 2`; the kinetic
     axis rounds at most 8 times (`eb^2`, two scalings, three products, two
     adds), the three-term product rule 4 more, and `s` 1.
  V  `N_E(L) + 3L + 12 + n_t + n_nuc + 3`: the fundamental `[0]^(m)` rounds
     `T`, `theta`, the power and the prefactor (12 is above those), each of
     at most `L` recurrence levels a product and an add and a scaling (3L),
     the `n_t`-term contraction over `t`, the `n_nuc`-term nucleus sum, and
     the products by `Z` and `s`;
     `delta = WIDE_BOYS_REL` (1e-13, test_kernel_device.py): the device
     `fj_boys` is held within it of `reference.boys` at every point, and
     every `[t]^(0)` is a non-negative combination of `F_m` in `Rabs`, so an
     `F` difference moves an element by at most `delta Mabs`. S and T have
     no Boys: `delta = 0`.

THE RUN-TO-RUN BOUND: two runs of one binary compute every term
identically, and only the order of the `K` atomic adds into an element
differs, so `|M(run 1) - M(run 2)| <= 2 gamma_K Mabs`; a pair with one
record must repeat bit for bit.

NOT A MEASUREMENT. The test prints each class's worst ratio to its bound
and nothing is written to `data/measured/`: no register, spill, wall-clock
or counter row (the issue: an unmeasured row is worse than none). V's
registers and spills per class are read from the compilers in the PR.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program both
parts use, so a vendor spelling problem shows without the card. The device
tests are `gpu` plus `cuda` or `hip`, behind the card lock and the VRAM
probe: on a box without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from one_electron_driver import (
    build_on_card,
    kernel_of,
    one_e_program,
    one_e_stage,
    tier_units,
)
from oracle import basis as ob
from oracle import manifest
from plan_driver import compile_checked
from test_ao_kinetic import kinetic_scale_block
from test_ao_overlap import ONE_E_TOL, _overlap_scale_block, one_e_scale
from test_ao_potential import potential_scale_block
from test_kernel_device import BACKEND, WIDE_BOYS_REL, _toolchain
from tiers import by_l, is_fast

from felizjueves.ao.hermite import _raise, e_axis, hermite_order
from felizjueves.ao.kinetic import kinetic_matrix
from felizjueves.ao.one_electron import matrix
from felizjueves.ao.overlap import overlap_matrix
from felizjueves.ao.pairs import overlap_bound
from felizjueves.ao.potential import potential_matrix
from felizjueves.ao.shell import cartesians
from felizjueves.emit import __main__ as emit_cli
from felizjueves.interp import Interpreter, Launch
from felizjueves.launch import OneELaunch, one_electron_launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

U = 2.0 ** -53
TARGETS = (Target.S, Target.T, Target.V)
TARGET_PARAMS = [pytest.param(t, id=t.name) for t in TARGETS]
PART1 = "hf_ccpvtz"
LMAX = 2  # through (dd): the default tier's ceiling (the issue)
LMAX_DEFERRED = 3  # through (ff): #244's deferred tier, `slow`
MOLECULES = ("h2o_sto3g_j", "h2o_ccpvdz_j")
# H2O/cc-pVDZ is `slow`: the PR gate holds S, T and V to libcint on
# H2O/STO-3G, and the d shells to the host in part 1.
MOLECULE_CELLS = [pytest.param(m, t, id=f"{m}-{t.name}",
                               marks=(pytest.mark.slow,)
                               if m == "h2o_ccpvdz_j" else ())
                  for m in MOLECULES for t in TARGETS]
FIXTURE_ARRAY = {Target.S: manifest.OVLP_ARRAY, Target.T: manifest.KIN_ARRAY,
                 Target.V: manifest.NUC_ARRAY}


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


def _n_e(L: int) -> int:
    return 4 * L + 8  # test_j_chain_ondevice.py's N_E


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@cache
def _case(name: str):
    """(fixture, basis, nuclei) of a tracked fixture; nuclei its atoms."""
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx), tuple((float(a.charge), a.coord)
                                          for a in fx.atoms)


@cache
def _launch(name: str, lmax: int | None = None) -> OneELaunch:
    """The fixture's one-electron launch at tau = 0 (nothing dropped), its
    classes restricted to `la, lb <= lmax` when given."""
    _, basis, nuclei = _case(name)
    full = one_electron_launch(basis, nuclei, tau=0.0, bound=overlap_bound)
    assert not full.dropped
    if lmax is None:
        return full
    classes = tuple(c for c in full if c.la <= lmax and c.lb <= lmax)
    kept = tuple((i, j) for c in classes for i, j, _, _ in c.blocks)
    return OneELaunch(classes, full.nuclei, full.tau, tuple(sorted(kept)), ())


def _host(name: str, target: Target, pairs) -> np.ndarray:
    _, basis, nuclei = _case(name)
    if target is Target.S:
        return np.array(overlap_matrix(basis, pairs=pairs))
    if target is Target.T:
        return np.array(kinetic_matrix(basis, pairs=pairs))
    return np.array(potential_matrix(basis, nuclei, one_centre=one_centre,
                                     pairs=pairs))


@cache
def _host_cached(name: str, target: Target, lmax: int | None) -> np.ndarray:
    return _host(name, target, _launch(name, lmax).kept)


# ---------------------------------------------------------------------------
# The bound's scale: Mabs and K per element, from the launch's own records
# ---------------------------------------------------------------------------


def _axis_abs(i: int, j: int, zeta: float, xa: float, xb: float) -> list[float]:
    """`E^{ij}_t` over `|X_PA|`, `|X_PB|` from the true seed (module
    docstring): `|E^{ij}_t| <= Eabs^{ij}_t`."""
    h = 0.5 / zeta
    e = [math.exp(zeta * xa * xb)]
    for _ in range(i):
        e = _raise(e, abs(xa), h)
    for _ in range(j):
        e = _raise(e, abs(xb), h)
    return e


def _r_abs(L: int, pc, theta: float) -> dict[tuple[int, int, int], float]:
    """`Rabs[t]` for `|t| <= L`: `[t]^(0)`'s recurrence over `|P - C|` from
    `(2/sqrt(pi)) theta (2 theta^2)^m F_m(theta^2 |P - C|^2)`."""
    a = [abs(x) for x in pc]
    T = theta * theta * (pc[0] ** 2 + pc[1] ** 2 + pc[2] ** 2)
    base = [2.0 / math.sqrt(math.pi) * theta * (2.0 * theta * theta) ** m
            * boys(m, T) for m in range(L + 1)]
    memo: dict = {}

    def node(r, m):
        if (r, m) not in memo:
            for d in range(3):
                if r[d]:
                    down = r[:d] + (r[d] - 1,) + r[d + 1:]
                    v = a[d] * node(down, m + 1)
                    if r[d] >= 2:
                        v += (r[d] - 1) * node(r[:d] + (r[d] - 2,) + r[d + 1:],
                                               m + 1)
                    memo[(r, m)] = v
                    break
            else:
                memo[(r, m)] = base[m]
        return memo[(r, m)]

    return {t: node(t, 0) for t in hermite_order(L)}


def _k_abs(srow: list[float], j: int, eb: float) -> float:
    k = 2.0 * eb * eb * srow[j + 2] + eb * (2 * j + 1) * srow[j]
    if j >= 2:
        k += 0.5 * j * (j - 1) * srow[j - 2]
    return k


def _rec_abs(cls, r, target: Target, nuclei) -> list[float]:
    """Each Cartesian pair's term magnitude for one record, row-major
    `cartesians(la) x cartesians(lb)` (the kernels' `v` order)."""
    la, lb = cls.la, cls.lb
    carts = [(a, b) for a in cartesians(la) for b in cartesians(lb)]
    s = abs(r.s)
    if target is Target.T:
        S = [[[_axis_abs(i, j, r.zeta, r.XPA[d], r.XPB[d])[0]
               for j in range(lb + 3)] for i in range(la + 1)] for d in range(3)]
        out = []
        for a, b in carts:
            sv = [S[d][a[d]][b[d]] for d in range(3)]
            kv = [_k_abs(S[d][a[d]], b[d], r.eb) for d in range(3)]
            out.append(s * (kv[0] * sv[1] * sv[2] + sv[0] * kv[1] * sv[2]
                            + sv[0] * sv[1] * kv[2]))
        return out
    axes = [{(i, j): _axis_abs(i, j, r.zeta, r.XPA[d], r.XPB[d])
             for i in range(la + 1) for j in range(lb + 1)} for d in range(3)]
    if target is Target.S:
        return [s * axes[0][a[0], b[0]][0] * axes[1][a[1], b[1]][0]
                * axes[2][a[2], b[2]][0] for a, b in carts]
    ts = hermite_order(la + lb)
    R = [0.0] * len(ts)
    for Z, C in nuclei:
        ra = _r_abs(la + lb, [r.P[d] - C[d] for d in range(3)],
                    math.sqrt(r.zeta))
        for n, t in enumerate(ts):
            R[n] += abs(Z) * ra[t]
    out = []
    for a, b in carts:
        v = 0.0
        for n, t in enumerate(ts):
            if all(t[d] <= a[d] + b[d] for d in range(3)):
                v += (axes[0][a[0], b[0]][t[0]] * axes[1][a[1], b[1]][t[1]]
                      * axes[2][a[2], b[2]][t[2]]) * R[n]
        out.append(s * v)
    return out


def _place(M: np.ndarray, r, na: int, nb: int, vals) -> None:
    """kernel_jab's scatter rule, ADDING: the block and its mirror; a
    diagonal shell pair's lower triangle, mirrored once."""
    oa, o_b = r.off_a, r.off_b
    for i in range(na):
        for j in range(nb):
            if oa == o_b and j > i:
                continue
            v = vals[i * nb + j]
            M[oa + i, o_b + j] += v
            if not (oa == o_b and i == j):
                M[o_b + j, oa + i] += v


@cache
def _scale(name: str, target: Target, lmax: int | None):
    """`(Mabs, K, N, cls_of)` per element of the launch's matrix: `N` the
    per-term rounding count of the owning class, `cls_of` its class."""
    _, basis, nuclei = _case(name)
    n = basis.n_ao
    M, K, N = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    cls_of = np.full((n, n), "", dtype=object)
    for cls in _launch(name, lmax):
        na, nb, L = len(cartesians(cls.la)), len(cartesians(cls.lb)), cls.L
        if target is Target.S:
            nt = _n_e(L) + 1
        elif target is Target.T:
            nt = _n_e(L + 2) + 13
        else:
            nt = _n_e(L) + 3 * L + 12 + len(hermite_order(L)) + len(nuclei) + 3
        mine = np.zeros((n, n))
        for r in cls.recs:
            _place(M, r, na, nb, _rec_abs(cls, r, target, nuclei))
            _place(K, r, na, nb, [1.0] * (na * nb))
            _place(mine, r, na, nb, [1.0] * (na * nb))
        N[mine > 0] = nt
        cls_of[mine > 0] = f"{cls.la}{cls.lb}"
    return M, K, N, cls_of


def _bound(name: str, target: Target, lmax: int | None):
    M, K, N, cls_of = _scale(name, target, lmax)
    g = np.vectorize(_gamma)
    delta = WIDE_BOYS_REL if target is Target.V else 0.0
    tol = (2.0 * g((N + K).astype(int)) + delta) * M
    r2r = 2.0 * g(K.astype(int)) * M
    return tol, r2r, M, K, cls_of


def _ratio(err: np.ndarray, tol: np.ndarray) -> float:
    pos = tol > 0
    return float(np.max(err[pos] / tol[pos])) if np.any(pos) else 0.0


def _check_host(backend: str, name: str, target: Target, lmax: int | None,
                Ms: np.ndarray) -> dict[str, float]:
    """Both runs finite, the unowned elements exactly 0, run to run inside
    the atomic bound, run 0 inside the device-vs-host bound; returns each
    class's worst `|M_dev - M_host| / bound`."""
    tol, r2r, Mabs, K, cls_of = _bound(name, target, lmax)
    host = _host_cached(name, target, lmax)
    owned = K > 0
    assert np.all(np.isfinite(Ms)), f"{target.name}: a non-finite element"
    assert np.all(Ms[:, ~owned] == 0.0), (
        f"{target.name}: a write outside the launch's shell pairs")
    d = np.abs(Ms[0] - Ms[1])
    assert np.all(d <= r2r), (
        f"{target.name} run to run: worst {_ratio(d, r2r):.2e} of the "
        "fp64-atomic bound")
    assert np.all(d[K == 1] == 0.0), (
        f"{target.name}: a one-record element differs between runs")
    err = np.abs(Ms[0] - host)
    worst = {c: _ratio(np.where(cls_of == c, err, 0.0), tol)
             for c in sorted(set(cls_of[owned]))}
    bad = owned & (err > tol)
    assert not np.any(bad), (
        f"{backend} {name} {target.name}: |M_dev - M_host| over the bound in "
        f"classes {sorted(set(cls_of[bad]))}; worst ratio per class {worst}")
    return worst


# ---------------------------------------------------------------------------
# Part 1: every pair class through (dd), against the host
# ---------------------------------------------------------------------------


def _units(backend: str, launch: OneELaunch, target: Target):
    return tier_units(backend, [kernel_of(c, target) for c in launch])


_THROUGH = {LMAX: "(dd)", LMAX_DEFERRED: "(ff)"}


def _part1(backend: str, target: Target, request, run_device_binary,
           tmp_path: Path, lmax: int = LMAX) -> None:
    tc = _toolchain(backend, request)
    launch = _launch(PART1, lmax)
    _, basis, _ = _case(PART1)
    units = _units(backend, launch, target)
    Ms = build_on_card(launch, target, basis.n_ao, backend, tc, tmp_path,
                       run_device_binary, units=units)
    worst = _check_host(backend, PART1, target, lmax, Ms)
    recs = {f"{c.la}{c.lb}": len(c.recs) for c in launch}
    print(f"\n{backend} {BACKEND[backend][1].name} {target.name} {PART1} "
          f"classes through {_THROUGH[lmax]} ({len(units)} of {len(launch)} "
          "units the "
          f"tier's): records {recs}; worst |M - M_host| / bound per class "
          + " ".join(f"{c}:{w:.2e}" for c, w in worst.items())
          + f"; runs bit-identical {bool(np.array_equal(Ms[0], Ms[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("target", TARGET_PARAMS)
def test_cuda_every_class_through_dd_matches_the_host(target, request,
                                                      run_device_binary,
                                                      tmp_path) -> None:
    """Part 1 on the RTX 3080."""
    _part1("cuda", target, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("target", TARGET_PARAMS)
def test_hip_every_class_through_dd_matches_the_host(target, request,
                                                     run_device_binary,
                                                     tmp_path) -> None:
    """Part 1 on the RX 9060 XT."""
    _part1("hip", target, request, run_device_binary, tmp_path)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("target", TARGET_PARAMS)
def test_cuda_every_class_through_ff_matches_the_host(target, request,
                                                      run_device_binary,
                                                      tmp_path) -> None:
    """Part 1 through (ff) on the RTX 3080 (#244, deferred)."""
    _part1("cuda", target, request, run_device_binary, tmp_path,
           LMAX_DEFERRED)


@pytest.mark.slow
@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("target", TARGET_PARAMS)
def test_hip_every_class_through_ff_matches_the_host(target, request,
                                                     run_device_binary,
                                                     tmp_path) -> None:
    """Part 1 through (ff) on the RX 9060 XT (#244, deferred)."""
    _part1("hip", target, request, run_device_binary, tmp_path,
           LMAX_DEFERRED)


# ---------------------------------------------------------------------------
# Part 2: H2O/STO-3G and H2O/cc-pVDZ, against libcint and the host
# ---------------------------------------------------------------------------


@cache
def _libcint_scale(name: str, target: Target) -> np.ndarray:
    """test_ao_{overlap,kinetic,potential}.py's `one_e_scale(E_M)`."""
    _, basis, nuclei = _case(name)
    if target is Target.S:
        block = _overlap_scale_block
    elif target is Target.T:
        block = kinetic_scale_block
    else:
        def block(sa, sb):
            return potential_scale_block(sa, sb, nuclei)
    return one_e_scale(np.array(matrix(basis, block)))


def _part2(backend: str, name: str, target: Target, request,
           run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    fx, basis, _ = _case(name)
    launch = _launch(name)
    assert max(max(c.la, c.lb) for c in launch) <= LMAX
    units = _units(backend, launch, target)
    Ms = build_on_card(launch, target, basis.n_ao, backend, tc, tmp_path,
                       run_device_binary, units=units)
    worst = _check_host(backend, name, target, None, Ms)
    ref = fx.arrays[FIXTURE_ARRAY[target]]
    scale = _libcint_scale(name, target)
    assert ref.shape == Ms[0].shape and np.all(scale > 0)
    err = np.abs(Ms[0] - ref)
    lc = float(np.max(err / scale))
    print(f"\n{backend} {BACKEND[backend][1].name} {target.name} {name}: "
          f"{len(launch)} classes {[f'{c.la}{c.lb}' for c in launch]} "
          f"({len(units)} the tier's); max|M - {FIXTURE_ARRAY[target]}| = "
          f"{err.max():.2e}, / scale {lc:.2e} (tol {ONE_E_TOL:g}); worst "
          f"|M - M_host| / bound {max(worst.values()):.2e}; runs "
          f"bit-identical {bool(np.array_equal(Ms[0], Ms[1]))}")
    assert np.all(err <= ONE_E_TOL * scale), (
        f"{backend} {name} {target.name} against libcint: worst err/scale "
        f"{lc:.2e} over {ONE_E_TOL:g}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "target"), MOLECULE_CELLS)
def test_cuda_one_electron_matrices_match_libcint(name, target, request,
                                                  run_device_binary,
                                                  tmp_path) -> None:
    """Part 2 on the RTX 3080."""
    _part2("cuda", name, target, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "target"), MOLECULE_CELLS)
def test_hip_one_electron_matrices_match_libcint(name, target, request,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """Part 2 on the RX 9060 XT."""
    _part2("hip", name, target, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`): no card needed
# ---------------------------------------------------------------------------

COMPILE_CELLS = [pytest.param(n, t, LMAX, id=f"{n}-{t.name}")
                 for n in (PART1, *MOLECULES) for t in TARGETS] + [
    pytest.param(PART1, t, LMAX_DEFERRED, id=f"{PART1}-ff-{t.name}",
                 marks=pytest.mark.slow) for t in TARGETS]


def _compiles(backend: str, tc, name: str, target: Target, tmp_path: Path,
              lmax: int = LMAX):
    _, basis, _ = _case(name)
    launch = _launch(name, lmax if name == PART1 else None)
    prog = one_e_program(launch, target, basis.n_ao, backend, tmp_path,
                         units=_units(backend, launch, target))
    compile_checked(tc, prog, tmp_path / "bin")


@pytest.mark.cuda
@pytest.mark.parametrize(("name", "target", "lmax"), COMPILE_CELLS)
def test_cuda_one_electron_programs_compile(name, target, lmax, nvcc,
                                            tmp_path):
    _compiles("cuda", nvcc, name, target, tmp_path, lmax)


@pytest.mark.hip
@pytest.mark.parametrize(("name", "target", "lmax"), COMPILE_CELLS)
def test_hip_one_electron_programs_compile(name, target, lmax, hipcc,
                                           tmp_path):
    _compiles("hip", hipcc, name, target, tmp_path, lmax)


# ---------------------------------------------------------------------------
# Fast tier: the programs' text and data, and the bound's pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [by_l(lm, 2 * lm, id=str(lm))
                                  for lm in (LMAX, LMAX_DEFERRED)])
def test_part1_reaches_every_ordered_class(lmax: int) -> None:
    """Through (dd) restricted, and through (ff) the whole launch: every
    ordered class `(la, lb)`, none above."""
    launch = _launch(PART1, lmax)
    got = {(c.la, c.lb) for c in launch}
    assert got == {(a, b) for a in range(lmax + 1) for b in range(lmax + 1)}
    if lmax == LMAX_DEFERRED:
        assert launch.kept == _launch(PART1).kept


@pytest.mark.parametrize("cell", [by_l((lm, n), 2 * lm, id=f"{lm}-{n}")
                                  for lm, n in ((LMAX, 18),
                                                (LMAX_DEFERRED, 30))])
def test_the_tier_writes_the_ladders_classes_and_no_other(cell,
                                                          tmp_path) -> None:
    """KERNEL_GEN_CMD's `--one-e-l 2` (and #244's `--one-e-l 3`) writes every
    `la <= lb` class's three units through (dd) (and (ff)), `emit`'s text;
    the `la > lb` classes the launch also reaches are the driver's to
    emit."""
    lmax, n_units = cell
    emit, arch, ext = BACKEND["cuda"]
    written = emit_cli.generate_one_e("cuda", arch, tmp_path, lmax)
    assert len(written) == n_units
    for c in _launch(PART1, lmax):
        for t in TARGETS:
            k = kernel_of(c, t)
            path = tmp_path / f"{k.name}.{ext}"
            assert path.exists() == (c.la <= c.lb), k.name
            if path.exists():
                assert path.read_text() == emit(k, arch)


@pytest.mark.parametrize("target", TARGET_PARAMS)
def test_the_program_reads_the_kernels_parameters(target, tmp_path) -> None:
    """Every class's launch passes one argument per kernel parameter, the
    staged input is the size the program's own check asks for."""
    launch = _launch("h2o_sto3g_j")
    _, basis, _ = _case("h2o_sto3g_j")
    prog = one_e_program(launch, target, basis.n_ao, "hip", tmp_path)
    for c in launch:
        k = kernel_of(c, target)
        at = prog.source.index(">>>(", prog.source.index(f"::{k.name}<<<"))
        call = prog.source[at:prog.source.index(");", at)]
        assert call.count(",") == len(k.params) - 1, k.name
    data = one_e_stage(launch, target)
    n_nuc = len(launch.nuclei) if target is Target.V else 0
    nd = {Target.S: 8, Target.T: 9, Target.V: 11}[target]
    assert data.size == 4 * n_nuc + sum((nd + 2) * len(c.recs) for c in launch)


@pytest.mark.parametrize("target", TARGET_PARAMS)
def test_the_staged_data_through_the_interpreter_is_the_host(target) -> None:
    """The arrays the card receives, sliced as the program slices them and
    run through the INTERPRETED kernels into one matrix, give the host's
    matrix within the device-vs-host bound: the staging, checked without a
    card."""
    name = "h2o_sto3g_j"
    launch = _launch(name)
    _, basis, _ = _case(name)
    n = basis.n_ao
    data = one_e_stage(launch, target)
    at = 0
    kw: dict = {}
    if target is Target.V:
        nn = len(launch.nuclei)
        kw = dict(n_nuc=nn, Z=data[:nn].copy(),
                  Cnuc=data[nn:4 * nn].reshape(nn, 3).copy())
        at = 4 * nn
    out = np.zeros((n, n))
    for c in launch:
        k = kernel_of(c, target)
        nr = len(c.recs)
        mat = k.params[-1].name  # the OUT matrix, last
        args = {}
        for p in k.params:
            if not p.array or p.name in kw or p.name in ("off_a", "off_b", mat):
                continue
            args[p.name] = data[at:at + nr].copy()
            at += nr
        args["off_a"] = data[at:at + nr].astype(np.int32)
        args["off_b"] = data[at + nr:at + 2 * nr].astype(np.int32)
        at += 2 * nr
        grid, block = (-(-nr // 8), (8, 1))
        out = Interpreter(k, launch=Launch(grid, block, 1), boys=boys).run(
            n_rec=nr, n_ao=n, **kw, **args, **{mat: out})[mat]
    assert at == data.size
    tol, _, _, _, _ = _bound(name, target, None)
    err = np.abs(out - _host_cached(name, target, None))
    assert np.all(err <= tol), _ratio(err, tol)


def _r_abs_and_e_abs_bound(fast: bool) -> None:
    """Over the orders on one side of the cut (`L`, and `i + j` for `E`),
    each drawing what it drew when one test walked them all."""
    rng = np.random.default_rng(242)
    for L in range(2 * LMAX_DEFERRED + 1):
        pc = rng.uniform(-2, 2, 3)
        theta = float(rng.uniform(0.3, 2.0))
        if is_fast(L) != fast:
            continue
        ra = _r_abs(L, pc, theta)
        for t in hermite_order(L):
            assert abs(one_centre(t, 0, pc, theta)) <= ra[t] * (1 + 1e-12)
    for i in range(LMAX_DEFERRED + 1):
        for j in range(LMAX_DEFERRED + 3):
            zeta = float(rng.uniform(0.2, 3.0))
            xa, xb = rng.uniform(-1.5, 1.5, 2)
            if is_fast(i + j) != fast:
                continue
            e = e_axis(i, j, zeta, xa, xb)
            ea = _axis_abs(i, j, zeta, xa, xb)
            assert all(abs(x) <= y * (1 + 1e-12) for x, y in zip(e, ea,
                                                                  strict=True))


def test_r_abs_bounds_one_centre_and_e_abs_bounds_e() -> None:
    _r_abs_and_e_abs_bound(fast=True)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_r_abs_bounds_one_centre_and_e_abs_bounds_e_above_l4() -> None:
    _r_abs_and_e_abs_bound(fast=False)
