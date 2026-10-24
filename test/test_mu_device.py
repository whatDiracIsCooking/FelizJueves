"""The dipole `MU = <a| q (r - C) |b>` on the card (#523): every ordered pair
class through (dd) against the host and libcint, whole-H2O `MU` against
libcint's `int1e_r_cart` at both origins, the origin-shift identity against
the card's own `S`, and `q = -1` traced with the density against PySCF's
`dip_moment` -- on the RTX 3080 and the RX 9060 XT.

MILESTONE 19'S CAP: nothing above L = 4, every shell `l <= 2`. The driver
(`mu_driver.py`) refuses a class above (dd), and nothing here builds,
emits, launches or checks one.

THE BUILD is `mu_driver.compile_mu`: one program per launch
(`launch.one_electron_launch`, no nuclei, tau = 0), every pair class's
`kernel_mu` kernel (#521) launched once into one zeroed `MU`, `RUNS` times.
The origin and the charge are read from the input, so ONE binary runs at
every `(C, q)` a test asks for: the origin shift and `q = -1` exercise the
kernel's runtime scalars, not a second compilation. A class whose unit the
tier's generate phase wrote (`--mu-l`) includes THAT file, asserted
byte-identical to `emit(...)`; any other class -- `la > lb`, which no ladder
writes, or every class when the tier's generator did not ask for `--mu-l`
-- is emitted by the driver, the same text. Every program is compiled with
devtools/config.sh's flags and FAILS on a warning located in generated
text.

PART 1, PER PAIR CLASS, THROUGH (dd): `hf_ccpvtz_spd_dipole` (#520,
HF/cc-pVTZ without f), whose launch reaches all nine ordered classes
`(0..2, 0..2)`, each between two centres. At the fixture's origin and at 0,
`q = 1`:

- against the HOST, `one_electron.multipole.multipole_block` placed by
  `kernel_jab`'s rule (test_kernel_mu.py's `_host`, the interpreted
  kernel's bitwise witness), per element under the bound below, reported
  per class;
- against LIBCINT, `int1e_r_cart:origin` / `:zero`, per class's elements,
  within `LIBCINT_TOL = 1e-12` of `max|r|` (test_kernel_mu.py's bound);
- `q = -1` from the same binary: the negation of `q = 1` within the
  run-to-run bound (only the atomics' order may differ), bitwise where an
  element has one record.

PART 2, PER MOLECULE: H2O/STO-3G and 6-31G (cc-pVDZ `slow`, for its input
size only) at both origins against libcint's `int1e_r_cart` (1e-12 of
`max|r|`) and the host (the bound below); and at `q = -1`, the electronic
dipole `Tr(D MU)` plus `SUM_A Z_A (R_A - C)` against PySCF's RHF
`dip_moment`, within 1e-12 of `max|nuclear| + max|electronic|`
(test_ao_multipole.py's scale, unchanged).

PART 3, THE ORIGIN SHIFT ON THE CARD: on the spd basis,
`MU(C') = MU(C) + q (C - C') S` with `MU` at two origins and `S` all three
built on the card (`S` by `kernel_s`, `one_electron_driver.build_on_card`),
at `q = 1` and `q = -1`, within `SHIFT_TOL = 1e-14` of `max|MU|`
(test_kernel_mu.py's and test_ao_multipole.py's bound, unchanged); and the
shift from the fixture's origin to 0 against libcint's `int1e_ovlp`, within
1e-12 of `max|MU|`.

THE DEVICE-VERSUS-HOST BOUND, per element of component `k`, stated before
any run, with `u = 2^-53`, `gamma_n = n u / (1 - n u)`, `L = la + lb` and
`K` the records of the owning shell pair (each adds once, in an order the
fp64 atomics choose):

    |MU_dev - MU_host| <= 2 gamma_{N + K} MUabs,   N = N_E(L) + 8,

`MUabs` the sum over the owning pair's records of
`|q s| PROD_d F_d`, `F_d = Eabs_0` on an axis `d != k` and
`F_k = Eabs_0 |X_PC| + Eabs_1` (the `Eabs_1` term only where the axis has
one, `i + j >= 1`), `Eabs` the `E` recurrence from the same seed over
`|X_PA|`, `|X_PB|` (test_one_electron_device.py's `_axis_abs`). `N_E =
4L + 8` is test_j_chain_ondevice.py's per-entry count (the recurrence and
an `exp` within an ulp on each libm); the 8 is above the rest: `X_PC`, the
axis product and add (3), the 3D product (2), `q s` (1) and `(q s) v` (1).
Each side computes each term within `gamma_N` of exact relative to its
absolute value and adds `K` terms within `gamma_K` more; FMA contraction
only removes roundings. No Boys, so no `delta`.

THE RUN-TO-RUN BOUND: `|MU(run 1) - MU(run 2)| <= 2 gamma_K MUabs`, and an
element with one record repeats bit for bit.

NOT A MEASUREMENT: worst ratios are printed, nothing is written to
`data/measured/`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program, so
a vendor spelling problem shows without the card. The device tests are
`gpu` plus `cuda` or `hip`, behind the card lock and the VRAM probe.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from mu_driver import (
    MU_LMAX,
    compile_mu,
    kernel_of,
    mu_program,
    mu_stage,
    units_of,
)
from one_electron_driver import build_on_card
from one_electron_driver import kernel_of as one_e_kernel_of
from one_electron_driver import tier_units as one_e_tier_units
from oracle import basis as ob
from oracle import manifest
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_mu import _host as _host_jab
from test_one_electron_device import _axis_abs, _place

from felizjueves.ao.pairs import overlap_bound
from felizjueves.ao.shell import cartesians
from felizjueves.eri.j_coulomb.launch import OneELaunch, one_electron_launch
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron.kernel_mu import MU, launch_shape
from felizjueves.spec import Target

U = 2.0 ** -53
LIBCINT_TOL = 1e-12  # of max|r| (int1e_r_cart) and of the dip_moment scale
SHIFT_TOL = 1e-14  # of max|MU|: test_kernel_mu.py's / test_ao_multipole.py's
ZERO = (0.0, 0.0, 0.0)
C2 = (-1.1, 0.25, 0.6)  # the second origin of the shift (test_kernel_mu.py's)
SPD = manifest.SPD_DIPOLE
MOLECULES = ("h2o_sto3g", "h2o_631g", "h2o_ccpvdz")
MOLECULE_PARAMS = [pytest.param(m, id=m, marks=(pytest.mark.slow,)
                                if m == "h2o_ccpvdz" else ())
                   for m in MOLECULES]


def _gamma(n):
    n = np.asarray(n, dtype=float)
    return n * U / (1.0 - n * U)


def _n_e(L: int) -> int:
    return 4 * L + 8  # test_j_chain_ondevice.py's N_E


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@cache
def _case(name: str):
    """`(basis, origin, r, r0, extra)`: the spd fixture, or an H2O pair
    (`extra` its `DipolePair`, else its `DipoleIntegrals`)."""
    if name == SPD:
        d = manifest.dipole_integrals()
        return ob.from_fixture(d.fixture), d.origin, d.r, d.r0, d
    p = manifest.dipole_pair(name)
    return ob.from_fixture(p.dipole), p.origin, p.r, p.r0, p


@cache
def _launch(name: str) -> OneELaunch:
    basis = _case(name)[0]
    assert max(s.l for s in basis.shells) <= MU_LMAX
    launch = one_electron_launch(basis, (), tau=0.0, bound=overlap_bound)
    assert not launch.dropped
    return launch


@cache
def _host(name: str, C: tuple[float, ...], q: float) -> np.ndarray:
    """`multipole_block` of every kept shell pair, placed by kernel_jab's
    rule (test_kernel_mu.py's `_host`)."""
    return _host_jab(_case(name)[0], _launch(name).kept, C, q)


# ---------------------------------------------------------------------------
# The bound's scale: MUabs and K per element, from the launch's own records
# ---------------------------------------------------------------------------


def _rec_abs(cls, r, C, q: float) -> list[list[float]]:
    """Per component, each Cartesian pair's term magnitude for one record,
    row-major `cartesians(la) x cartesians(lb)`."""
    la, lb = cls.la, cls.lb
    s = abs(q * r.s)
    xpc = [abs(r.P[d] - C[d]) for d in range(3)]
    ax = [{(i, j): _axis_abs(i, j, r.zeta, r.XPA[d], r.XPB[d])
           for i in range(la + 1) for j in range(lb + 1)} for d in range(3)]
    out: list[list[float]] = [[], [], []]
    for a in cartesians(la):
        for b in cartesians(lb):
            f0, f1 = [], []
            for d in range(3):
                e = ax[d][a[d], b[d]]
                f0.append(e[0])
                f1.append(e[0] * xpc[d] + (e[1] if len(e) > 1 else 0.0))
            for k in range(3):
                v = s
                for d in range(3):
                    v *= f1[d] if d == k else f0[d]
                out[k].append(v)
    return out


@cache
def _scale(name: str, C: tuple[float, ...], q: float):
    """`(MUabs[3], K, N, cls_of)` per element of the launch's `MU`."""
    n = _case(name)[0].n_ao
    M = np.zeros((3, n, n))
    K, N = np.zeros((n, n)), np.zeros((n, n))
    cls_of = np.full((n, n), "", dtype=object)
    for cls in _launch(name):
        na, nb = len(cartesians(cls.la)), len(cartesians(cls.lb))
        mine = np.zeros((n, n))
        for r in cls.recs:
            vals = _rec_abs(cls, r, C, q)
            for k in range(3):
                _place(M[k], r, na, nb, vals[k])
            _place(K, r, na, nb, [1.0] * (na * nb))
            _place(mine, r, na, nb, [1.0] * (na * nb))
        N[mine > 0] = _n_e(cls.L) + 8
        cls_of[mine > 0] = f"{cls.la}{cls.lb}"
    return M, K, N, cls_of


def _bound(name: str, C, q: float):
    M, K, N, cls_of = _scale(name, tuple(float(x) for x in C), float(q))
    tol = 2.0 * _gamma(N + K)[None] * M
    r2r = 2.0 * _gamma(K)[None] * M
    return tol, r2r, K, cls_of


def _ratio(err: np.ndarray, tol: np.ndarray) -> float:
    pos = tol > 0
    return float(np.max(err[pos] / tol[pos])) if np.any(pos) else 0.0


def _check_host(backend: str, name: str, C, q: float,
                Ms: np.ndarray) -> dict[str, float]:
    """Both runs finite, the unowned elements exactly 0, run to run inside
    the atomic bound, run 0 inside the device-vs-host bound; returns each
    class's worst `|MU_dev - MU_host| / bound`."""
    C = tuple(float(x) for x in C)
    tol, r2r, K, cls_of = _bound(name, C, q)
    host = _host(name, C, float(q))
    owned = K > 0
    assert np.all(np.isfinite(Ms)), "a non-finite element"
    assert np.all(Ms[:, :, ~owned] == 0.0), (
        "a write outside the launch's shell pairs")
    d = np.abs(Ms[0] - Ms[1])
    assert np.all(d <= r2r), (
        f"run to run: worst {_ratio(d, r2r):.2e} of the fp64-atomic bound")
    assert np.all(d[:, K == 1] == 0.0), "a one-record element differs"
    err = np.abs(Ms[0] - host)
    worst = {c: _ratio(np.where((cls_of == c)[None], err, 0.0), tol)
             for c in sorted(set(cls_of[owned]))}
    bad = owned[None] & (err > tol)
    assert not np.any(bad), (
        f"{backend} {name} C={C} q={q}: |MU_dev - MU_host| over the bound "
        f"in classes {sorted(set(cls_of[bad.any(axis=0)]))}; worst ratio per "
        f"class {worst}")
    return worst


def _host_rel(name: str, C, Ms: np.ndarray) -> float:
    """`max|MU_dev - MU_host| / max|MU_host|` at `q = 1`, printed beside
    the per-element bound (which is what is asserted)."""
    host = _host(name, tuple(float(x) for x in C), 1.0)
    return float(np.max(np.abs(Ms[0] - host)) / np.max(np.abs(host)))


def _libcint_rel(got: np.ndarray, want: np.ndarray,
                 mask: np.ndarray | None = None) -> float:
    err = np.abs(got - want)
    if mask is not None:
        err = np.where(mask[None], err, 0.0)
    return float(err.max() / np.max(np.abs(want)))


def _binary(backend: str, name: str, request, tmp_path: Path):
    tc = _toolchain(backend, request)
    launch = _launch(name)
    units = units_of(backend, launch)
    b = compile_mu(launch, _case(name)[0].n_ao, backend, tc, tmp_path,
                   units=units)
    return b, len(units)


def _tag(backend: str) -> str:
    return f"{backend} {BACKEND[backend][1].name}"


# ---------------------------------------------------------------------------
# Part 1: every ordered pair class through (dd), host and libcint
# ---------------------------------------------------------------------------


def _part1(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    b, n_tier = _binary(backend, SPD, request, tmp_path)
    launch = _launch(SPD)
    _, origin, r, r0, _ = _case(SPD)
    _, _, K, cls_of = _bound(SPD, origin, 1.0)
    classes = sorted(set(cls_of[K > 0]))
    lines = []
    for C, want, tag in ((origin, r, "C"), (ZERO, r0, "0")):
        plus = b.run(C, 1.0, run_device_binary)
        worst = _check_host(backend, SPD, C, 1.0, plus)
        lc = {c: _libcint_rel(plus[0], want, cls_of == c) for c in classes}
        bad = {c: v for c, v in lc.items() if v > LIBCINT_TOL}
        assert not bad, (f"{backend} {SPD} at {tag}: per-class |MU - "
                         f"int1e_r_cart| / max|r| over {LIBCINT_TOL:g}: {bad}")
        minus = b.run(C, -1.0, run_device_binary)
        _check_host(backend, SPD, C, -1.0, minus)
        _, r2r, _, _ = _bound(SPD, C, 1.0)
        d = np.abs(minus[0] + plus[0])
        assert np.all(d <= r2r), (
            f"q = -1 is not the negation of q = 1: {_ratio(d, r2r):.2e} of "
            "the run-to-run bound")
        assert np.all(d[:, K == 1] == 0.0), "q = -1: a one-record element"
        lines.append(f"  origin {tag}: worst |MU - host| / bound per class "
                     + " ".join(f"{c}:{w:.2e}" for c, w in worst.items())
                     + "; |MU - int1e_r_cart| / max|r| per class "
                     + " ".join(f"{c}:{v:.2e}" for c, v in lc.items())
                     + f" (tol {LIBCINT_TOL:g}); |MU - host| / max|MU| "
                     f"{_host_rel(SPD, C, plus):.2e}; runs bit-identical "
                     f"{bool(np.array_equal(plus[0], plus[1]))}; q=-1 exact "
                     f"negation {bool(np.array_equal(minus[0], -plus[0]))}")
    assert classes == [f"{a}{c}" for a in range(3) for c in range(3)]
    recs = {f"{c.la}{c.lb}": len(c.recs) for c in launch}
    print(f"\n{_tag(backend)} MU {SPD}: classes through (dd) ({n_tier} of "
          f"{len(launch)} units the tier's), records {recs}\n"
          + "\n".join(lines))


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_mu_every_class_through_dd_matches_host_and_libcint(
        request, run_device_binary, tmp_path) -> None:
    """Part 1 on the RTX 3080."""
    _part1("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_mu_every_class_through_dd_matches_host_and_libcint(
        request, run_device_binary, tmp_path) -> None:
    """Part 1 on the RX 9060 XT."""
    _part1("hip", request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: whole H2O, libcint at both origins and dip_moment at q = -1
# ---------------------------------------------------------------------------


def _part2(backend: str, name: str, request, run_device_binary,
           tmp_path: Path) -> None:
    b, n_tier = _binary(backend, name, request, tmp_path)
    basis, origin, r, r0, p = _case(name)
    D = p.density
    Z = p.dipole.arrays[manifest.DIP_CHARGES_ARRAY]
    R = p.dipole.arrays[manifest.DIP_COORDS_ARRAY]
    lines = []
    for C, want, dip, tag in ((origin, r, p.dip, "C"),
                              (ZERO, r0, p.dip0, "0")):
        Ms = b.run(C, 1.0, run_device_binary)
        worst = _check_host(backend, name, C, 1.0, Ms)
        lc = _libcint_rel(Ms[0], want)
        assert lc <= LIBCINT_TOL, (
            f"{backend} {name} at {tag}: |MU - int1e_r_cart| / max|r| = "
            f"{lc:.2e} over {LIBCINT_TOL:g}")
        neg = b.run(C, -1.0, run_device_binary)
        wneg = _check_host(backend, name, C, -1.0, neg)
        el = np.einsum("xij,ji->x", neg[0], D)  # q = -1: the electrons'
        nuc = Z @ (R - np.asarray(C, dtype=float)[None, :])
        scale = np.max(np.abs(nuc)) + np.max(np.abs(el))
        dm = float(np.max(np.abs((nuc + el) - dip)) / scale)
        assert dm <= LIBCINT_TOL, (
            f"{backend} {name} at {tag}: |Tr(D MU_q=-1) + nuc - dip_moment| "
            f"/ scale = {dm:.2e} over {LIBCINT_TOL:g}")
        lines.append(f"  origin {tag}: |MU - int1e_r_cart| / max|r| "
                     f"{lc:.2e}; |dip - dip_moment| / scale {dm:.2e} (tol "
                     f"{LIBCINT_TOL:g}); |MU - host| / max|MU| "
                     f"{_host_rel(name, C, Ms):.2e}; worst |MU - host| / bound "
                     f"{max(max(worst.values()), max(wneg.values())):.2e}; "
                     "runs bit-identical "
                     f"{bool(np.array_equal(Ms[0], Ms[1]))}")
    launch = _launch(name)
    print(f"\n{_tag(backend)} MU {name}: {len(launch)} classes "
          f"{[f'{c.la}{c.lb}' for c in launch]} ({n_tier} the tier's)\n"
          + "\n".join(lines))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULE_PARAMS)
def test_cuda_mu_h2o_matches_libcint_and_dip_moment(
        name, request, run_device_binary, tmp_path) -> None:
    """Part 2 on the RTX 3080."""
    _part2("cuda", name, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULE_PARAMS)
def test_hip_mu_h2o_matches_libcint_and_dip_moment(
        name, request, run_device_binary, tmp_path) -> None:
    """Part 2 on the RX 9060 XT."""
    _part2("hip", name, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 3: the origin shift, against the card's S
# ---------------------------------------------------------------------------


def _part3(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    b, _ = _binary(backend, SPD, request, tmp_path / "mu")
    basis, origin, _, _, d = _case(SPD)
    launch = _launch(SPD)
    tc = _toolchain(backend, request)
    units = one_e_tier_units(backend,
                             [one_e_kernel_of(c, Target.S) for c in launch])
    S = build_on_card(launch, Target.S, basis.n_ao, backend, tc,
                      tmp_path / "s", run_device_binary, units=units)[0]
    C, C2v = np.asarray(origin, dtype=float), np.asarray(C2)
    shift = {}
    for q in (1.0, -1.0):
        at_c = b.run(C, q, run_device_binary)[0]
        at_c2 = b.run(C2v, q, run_device_binary)[0]
        want = at_c + q * (C - C2v)[:, None, None] * S[None]
        shift[q] = float(np.max(np.abs(at_c2 - want)) / np.max(np.abs(want)))
        assert shift[q] <= SHIFT_TOL, (
            f"{backend} q={q}: MU(C') - MU(C) - q (C - C') S_card = "
            f"{shift[q]:.2e} of max|MU|, over {SHIFT_TOL:g}")
    at_c = b.run(C, 1.0, run_device_binary)[0]
    mu0 = b.run(ZERO, 1.0, run_device_binary)[0]
    want = mu0 - C[:, None, None] * d.ovlp[None]
    lc = float(np.max(np.abs(at_c - want)) / np.max(np.abs(want)))
    assert lc <= LIBCINT_TOL, (
        f"{backend}: MU(C) - MU(0) + C S_libcint = {lc:.2e} of max|MU|")
    print(f"\n{_tag(backend)} MU {SPD} origin shift: |MU(C') - MU(C) - "
          f"q (C - C') S_card| / max|MU| q=+1 {shift[1.0]:.2e}, q=-1 "
          f"{shift[-1.0]:.2e} (tol {SHIFT_TOL:g}); against int1e_ovlp "
          f"{lc:.2e} (tol {LIBCINT_TOL:g})")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_mu_origin_shift_against_the_cards_s(
        request, run_device_binary, tmp_path) -> None:
    """Part 3 on the RTX 3080."""
    _part3("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_mu_origin_shift_against_the_cards_s(
        request, run_device_binary, tmp_path) -> None:
    """Part 3 on the RX 9060 XT."""
    _part3("hip", request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`): no card needed
# ---------------------------------------------------------------------------

COMPILE_CELLS = [pytest.param(SPD, id=SPD)] + MOLECULE_PARAMS


def _compiles(backend: str, tc, name: str, tmp_path: Path) -> None:
    launch = _launch(name)
    prog = mu_program(launch, _case(name)[0].n_ao, backend, tmp_path,
                      units=units_of(backend, launch))
    compile_checked(tc, prog, tmp_path / "bin")


@pytest.mark.cuda
@pytest.mark.parametrize("name", COMPILE_CELLS)
def test_cuda_mu_programs_compile(name, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc, name, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("name", COMPILE_CELLS)
def test_hip_mu_programs_compile(name, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc, name, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the programs' text and data, and the cap
# ---------------------------------------------------------------------------


def test_mu_spd_launch_reaches_every_ordered_class_and_none_above_dd(
        ) -> None:
    got = {(c.la, c.lb) for c in _launch(SPD)}
    assert got == {(a, b) for a in range(3) for b in range(3)}
    for name in MOLECULES:
        assert all(max(c.la, c.lb) <= MU_LMAX for c in _launch(name))


def test_mu_driver_refuses_a_class_above_dd() -> None:
    class _F:
        la, lb = 0, 3
    with pytest.raises(ValueError, match="above"):
        kernel_of(_F())  # type: ignore[arg-type]


def test_mu_program_reads_the_kernels_parameters(tmp_path) -> None:
    """Every class's launch passes one argument per kernel parameter; the
    staged input is the size the program's own check asks for."""
    name = "h2o_sto3g"
    launch = _launch(name)
    prog = mu_program(launch, _case(name)[0].n_ao, "hip", tmp_path)
    for c in launch:
        k = kernel_of(c)
        at = prog.source.index(">>>(", prog.source.index(f"::{k.name}<<<"))
        call = prog.source[at:prog.source.index(");", at)]
        assert call.count(",") == len(k.params) - 1, k.name
        assert "Cx, Cy, Cz, Q, dM" in call, k.name
    data = mu_stage(launch, (0.5, -0.25, 1.0), -1.0)
    assert list(data[:4]) == [0.5, -0.25, 1.0, -1.0]
    assert data.size == 4 + sum((11 + 2) * len(c.recs) for c in launch)


@pytest.mark.parametrize("q", [1.0, -1.0])
def test_mu_staged_data_through_the_interpreter_is_the_host(q) -> None:
    """The arrays the card receives, sliced as the program slices them, run
    through the INTERPRETED kernels into one `MU`, give the host within the
    device-vs-host bound: the staging, checked without a card."""
    name = "h2o_sto3g"
    launch = _launch(name)
    basis, origin = _case(name)[:2]
    n = basis.n_ao
    data = mu_stage(launch, origin, q)
    C = data[:3]
    at = 4
    out = np.zeros((3, n, n))
    for c in launch:
        k = kernel_of(c)
        nr = len(c.recs)
        args = {}
        for p in k.params:
            if not p.array or p.name in ("off_a", "off_b", MU):
                continue
            args[p.name] = data[at:at + nr].copy()
            at += nr
        args["off_a"] = data[at:at + nr].astype(np.int32)
        args["off_b"] = data[at + nr:at + 2 * nr].astype(np.int32)
        at += 2 * nr
        grid, block = launch_shape(nr, 8)
        out = Interpreter(k, launch=Launch(grid, block, 1)).run(
            n_rec=nr, n_ao=n, C_x=float(C[0]), C_y=float(C[1]),
            C_z=float(C[2]), q=float(data[3]), **args, MU=out)[MU]
    assert at == data.size
    tol, _, _, _ = _bound(name, origin, q)
    err = np.abs(out - _host(name, tuple(float(x) for x in origin), q))
    assert np.all(err <= tol), _ratio(err, tol)
