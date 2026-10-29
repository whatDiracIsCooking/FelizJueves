"""The quadrupole `<a| q (r - C)_d (r - C)_e |b>` on the card: every ordered
pair class through (dd) against the host and libcint's `int1e_rr_cart` at
both origins, and `q = -1` from the same binary -- on the RTX 3080 and the
RX 9060 XT.

THE CAP is the dipole's: nothing above L = 4, every shell `l <= 2`; the
driver (`mu_driver.py`, `target=Target.QUADRUPOLE`) refuses a class above
(dd).

THE BUILD is `mu_driver.compile_mu(..., target=Target.QUADRUPOLE)`: one
program over `launch.one_electron_launch`'s plan of `hf_ccpvtz_spd_dipole`
(HF/cc-pVTZ without f: all nine ordered classes `(0..2, 0..2)`, each between
two centres), every class's `quad_{la}{lb}` kernel launched once into one
zeroed `6 x n_ao x n_ao` `MU`, `RUNS` times; the origin and charge read
from the input, so ONE binary serves every `(C, q)`. A unit the tier's
generate phase wrote (`--quad-l`) is that file, asserted byte-identical; any
other is emitted by the driver, the same text. Compiled with
devtools/config.sh's flags, FAILING on a warning located in generated text.

AGAINST:

- LIBCINT, `manifest.quadrupole_integrals()`'s `int1e_rr_cart` at the
  fixture's origin and at 0, per class's elements, within
  `LIBCINT_TOL = 1e-12` of `max|rr|` (test_kernel_quad.py's bound; the
  interpreted kernel measured 1.1e-15);
- the HOST, `multipole_block(e=2)` placed by `kernel_jab`'s rule
  (test_kernel_quad.py's `_host`, the interpreted kernel's bitwise
  witness), per element under the bound below;
- `q = -1`: the negation of `q = 1` within the run-to-run bound, bitwise
  where an element has one record.

THE DEVICE-VERSUS-HOST BOUND, per element of component `k = (d, e)`, stated
before any run, with `u = 2^-53`, `gamma_n = n u / (1 - n u)`, `L = la + lb`
and `K` the records of the owning shell pair:

    |Q_dev - Q_host| <= 2 gamma_{N + K} Qabs,   N = N_E(L) + 16,

`Qabs` the sum over the owning pair's records of `|q s| PROD_axis F`, each
axis factor `F = SUM_{t <= min(n, i+j)} Eabs_t Mabs^n_t` at that axis's
order `n` in `k`, `Eabs` the `E` recurrence over `|X_PA|`, `|X_PB|`
(test_one_electron_device.py's `_axis_abs`) and `Mabs` the Hermite moments
from `|X_PC|` (`hermite_moments`, every term then non-negative). `N_E = 4L +
8` is test_mu_device.py's per-entry count; the 16 covers `X_PC`, the moments
(at most two roundings each), the three-term axis sum (five), the 3D
product (2), `q s` and `(q s) v` (2), with room. No Boys.

THE RUN-TO-RUN BOUND: `|Q(run 1) - Q(run 2)| <= 2 gamma_K Qabs`, and an
element with one record repeats bit for bit.

NOT RUN YET: this session had no card and neither vendor compiler (the
units passed clang 18's CUDA front end, a proxy). NOT A MEASUREMENT: worst
ratios are printed, nothing is written to `data/measured/`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from mu_driver import QUAD_LMAX, compile_mu, kernel_of, mu_program, mu_stage, units_of
from oracle import basis as ob
from oracle import manifest
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_quad import _host as _host_jab
from test_one_electron_device import _axis_abs, _place

from felizjueves.ao.pairs import overlap_bound
from felizjueves.ao.shell import cartesians
from felizjueves.eri.j_coulomb.launch import OneELaunch, one_electron_launch
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron.kernel_mu import MU, launch_shape
from felizjueves.one_electron.multipole import hermite_moments
from felizjueves.spec import Target

QUAD = Target.QUADRUPOLE
NCOMP = 6
U = 2.0 ** -53
LIBCINT_TOL = 1e-12  # of max|rr|
ZERO = (0.0, 0.0, 0.0)
COMPS = cartesians(2)  # xx xy xz yy yz zz, as orders per axis


def _gamma(n):
    n = np.asarray(n, dtype=float)
    return n * U / (1.0 - n * U)


@cache
def _case():
    """`(basis, origin, rr, rr0)` of the quadrupole witness."""
    w = manifest.quadrupole_integrals()
    return ob.from_fixture(w.parent.fixture), w.origin, w.rr, w.rr0


@cache
def _launch() -> OneELaunch:
    basis = _case()[0]
    assert max(s.l for s in basis.shells) <= QUAD_LMAX
    launch = one_electron_launch(basis, (), tau=0.0, bound=overlap_bound)
    assert not launch.dropped
    return launch


@cache
def _host(C: tuple[float, ...], q: float) -> np.ndarray:
    return _host_jab(_case()[0], _launch().kept, C, q)


def _rec_abs(cls, r, C, q: float) -> list[list[float]]:
    """Per component, each Cartesian pair's term magnitude for one record."""
    s = abs(q * r.s)
    mom = [[hermite_moments(n, abs(r.P[d] - C[d]), r.zeta) for n in range(3)]
           for d in range(3)]
    out: list[list[float]] = [[] for _ in COMPS]
    for a in cartesians(cls.la):
        for b in cartesians(cls.lb):
            ax = [_axis_abs(a[d], b[d], r.zeta, r.XPA[d], r.XPB[d])
                  for d in range(3)]
            for k, ek in enumerate(COMPS):
                v = s
                for d in range(3):
                    m = mom[d][ek[d]]
                    v *= sum(ax[d][t] * m[t]
                             for t in range(min(len(ax[d]), len(m))))
                out[k].append(v)
    return out


@cache
def _scale(C: tuple[float, ...], q: float):
    """`(Qabs[6], K, N, cls_of)` per element of the launch's `MU`."""
    n = _case()[0].n_ao
    M = np.zeros((NCOMP, n, n))
    K, N = np.zeros((n, n)), np.zeros((n, n))
    cls_of = np.full((n, n), "", dtype=object)
    for cls in _launch():
        na, nb = len(cartesians(cls.la)), len(cartesians(cls.lb))
        mine = np.zeros((n, n))
        for r in cls.recs:
            vals = _rec_abs(cls, r, C, q)
            for k in range(NCOMP):
                _place(M[k], r, na, nb, vals[k])
            _place(K, r, na, nb, [1.0] * (na * nb))
            _place(mine, r, na, nb, [1.0] * (na * nb))
        N[mine > 0] = 4 * cls.L + 8 + 16
        cls_of[mine > 0] = f"{cls.la}{cls.lb}"
    return M, K, N, cls_of


def _bound(C, q: float):
    M, K, N, cls_of = _scale(tuple(float(x) for x in C), float(q))
    return 2.0 * _gamma(N + K)[None] * M, 2.0 * _gamma(K)[None] * M, K, cls_of


def _ratio(err: np.ndarray, tol: np.ndarray) -> float:
    pos = tol > 0
    return float(np.max(err[pos] / tol[pos])) if np.any(pos) else 0.0


def _check_host(backend: str, C, q: float, Ms: np.ndarray) -> dict:
    C = tuple(float(x) for x in C)
    tol, r2r, K, cls_of = _bound(C, q)
    owned = K > 0
    assert np.all(np.isfinite(Ms)), "a non-finite element"
    assert np.all(Ms[:, :, ~owned] == 0.0), "a write outside the launch"
    d = np.abs(Ms[0] - Ms[1])
    assert np.all(d <= r2r), f"run to run: {_ratio(d, r2r):.2e} of the bound"
    assert np.all(d[:, K == 1] == 0.0), "a one-record element differs"
    err = np.abs(Ms[0] - _host(C, float(q)))
    worst = {c: _ratio(np.where((cls_of == c)[None], err, 0.0), tol)
             for c in sorted(set(cls_of[owned]))}
    bad = owned[None] & (err > tol)
    assert not np.any(bad), (
        f"{backend} C={C} q={q}: |Q_dev - Q_host| over the bound in classes "
        f"{sorted(set(cls_of[bad.any(axis=0)]))}; worst per class {worst}")
    return worst


def _part1(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    basis, origin, rr, rr0 = _case()
    launch = _launch()
    units = units_of(backend, launch, QUAD)
    b = compile_mu(launch, basis.n_ao, backend, tc, tmp_path, units=units,
                   target=QUAD)
    _, _, K, cls_of = _bound(origin, 1.0)
    classes = sorted(set(cls_of[K > 0]))
    assert classes == [f"{a}{c}" for a in range(3) for c in range(3)]
    lines = []
    for C, want, tag in ((origin, rr, "C"), (ZERO, rr0, "0")):
        plus = b.run(C, 1.0, run_device_binary)
        worst = _check_host(backend, C, 1.0, plus)
        scale = float(np.max(np.abs(want)))
        lc = {c: float(np.max(np.where((cls_of == c)[None],
                                       np.abs(plus[0] - want), 0.0)) / scale)
              for c in classes}
        bad = {c: v for c, v in lc.items() if v > LIBCINT_TOL}
        assert not bad, (f"{backend} at {tag}: per-class |Q - int1e_rr_cart| "
                         f"/ max|rr| over {LIBCINT_TOL:g}: {bad}")
        minus = b.run(C, -1.0, run_device_binary)
        _check_host(backend, C, -1.0, minus)
        _, r2r, _, _ = _bound(C, 1.0)
        d = np.abs(minus[0] + plus[0])
        assert np.all(d <= r2r), "q = -1 is not the negation of q = 1"
        assert np.all(d[:, K == 1] == 0.0), "q = -1: a one-record element"
        lines.append(f"  origin {tag}: worst |Q - host| / bound per class "
                     + " ".join(f"{c}:{w:.2e}" for c, w in worst.items())
                     + "; |Q - int1e_rr_cart| / max|rr| per class "
                     + " ".join(f"{c}:{v:.2e}" for c, v in lc.items())
                     + f"; runs bit-identical "
                     f"{bool(np.array_equal(plus[0], plus[1]))}")
    print(f"\n{backend} {BACKEND[backend][1].name} quadrupole "
          f"{manifest.SPD_DIPOLE}: {len(units)} of {len(launch)} units the "
          "tier's\n" + "\n".join(lines))


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_quad_every_class_through_dd_matches_host_and_libcint(
        request, run_device_binary, tmp_path) -> None:
    """On the RTX 3080 (the card fixture taken by `_toolchain`)."""
    _part1("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_quad_every_class_through_dd_matches_host_and_libcint(
        request, run_device_binary, tmp_path) -> None:
    """On the RX 9060 XT (the card fixture taken by `_toolchain`)."""
    _part1("hip", request, run_device_binary, tmp_path)


# --- compile-only companions (`cuda` / `hip`, no `gpu`) ------------------------


def _compiles(backend: str, tc, tmp_path: Path) -> None:
    launch = _launch()
    prog = mu_program(launch, _case()[0].n_ao, backend, tmp_path,
                      units=units_of(backend, launch, QUAD), target=QUAD)
    compile_checked(tc, prog, tmp_path / "bin")


@pytest.mark.cuda
def test_cuda_quad_program_compiles(nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_quad_program_compiles(hipcc, tmp_path) -> None:
    _compiles("hip", hipcc, tmp_path)


# --- fast tier: the program's text and data -----------------------------------


def test_quad_program_launches_the_quad_units_into_six_components(
        tmp_path) -> None:
    launch = _launch()
    prog = mu_program(launch, _case()[0].n_ao, "hip", tmp_path, target=QUAD)
    assert "NCOMP = 6" in prog.source
    for c in launch:
        k = kernel_of(c, QUAD)
        assert k.name == f"quad_{c.la}{c.lb}"
        at = prog.source.index(">>>(", prog.source.index(f"::{k.name}<<<"))
        call = prog.source[at:prog.source.index(");", at)]
        assert call.count(",") == len(k.params) - 1, k.name
        assert "Cx, Cy, Cz, Q, dM" in call, k.name
    assert "mu_" not in "".join(p.name for p in prog.units)


def test_the_driver_refuses_a_non_multipole_target() -> None:
    with pytest.raises(ValueError, match="not a multipole target"):
        kernel_of(next(iter(_launch())), Target.S)


@pytest.mark.parametrize("q", [1.0, -1.0])
def test_quad_staged_data_through_the_interpreter_is_the_host(q) -> None:
    """The arrays the card receives, sliced as the program slices them, run
    through the INTERPRETED kernels: the staging, checked without a card."""
    launch = _launch()
    basis, origin = _case()[:2]
    n = basis.n_ao
    data = mu_stage(launch, origin, q, QUAD)
    C = data[:3]
    at = 4
    out = np.zeros((NCOMP, n, n))
    for c in launch:
        k = kernel_of(c, QUAD)
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
    tol, _, _, _ = _bound(origin, q)
    err = np.abs(out - _host(tuple(float(x) for x in origin), q))
    assert np.all(err <= tol), _ratio(err, tol)
