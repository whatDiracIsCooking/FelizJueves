"""The exchange-energy Hessian `hkx_` ON A CARD (#691), against
`hessian.host.exchange_hessian` (generator `test/hess_driver.py`'s
`hkx_launches`, one warp per bra record, the kets grouped by atom pair).

PER CLASS, every K class through (ff|ff) (256), each its own program in
its DEFAULT form (unrolled below `kernel_hkx.LOWSTACK_MIN_L`, rolled from
it), on `TRI_SPDF`: three atoms, a single-primitive s, p, d and f shell on
each. Each class launches ONE bra shell pair `(l_a on A, l_c on B)`
against TWO ket shell pairs `(l_b on B, l_d on C)` and `(l_b on C, l_d on
A)` -- two segments, three centres, every term of `H` live -- so the host
reference is three quartets (~6 s at (ff|ff)), not a class. Marked by
`tiers.by_l` on the raised `L + 2`: `slow` above (ss|ps)'s, `high_l` above
`DEVICE_L_MAX` (L + 2 > 8, i.e. L >= 7). This is the issue's "every class
launches on the RTX 3080" and, at f, its witness classes.

THE DIFFERENTIAL (`slow`): every s, p class from `LOWSTACK_MIN_L` launched
in its UNROLLED form beside its default rolled one on the same input
(named `..._unrolled`), the two held to each other.

PER MOLECULE (`slow`), AGAINST LIBCINT: H2O over data/basis/'s STO-3G and
cc-pVDZ (the numeric golden store's `hkx_h2o_*_bse`, `oracle.numeric`:
libcint's `int2e_ipip1` / `_ipvip1` / `_ip1ip2` contracted with `D (x) D`,
the `_j` fixtures' densities), every K class of the basis in one program,
the whole skeleton `d2 E_K` within the fp64 tier, 1e-12 of `max|H|`.
Host-only beside it (`slow`, ~40 s): the host sweep on H2O/STO-3G against
the same golden, which ties the per-class reference above to libcint.

THE BOUND per class, stated before any run: `|H_dev - H_host| <= DEV_TOL
max(twin)`, `DEV_TOL = 1e-14`, `twin` the host's `absolute=True` sum (the
interpreted kernels are within ~0.4 eps of it, test_kernel_hkx.py); the
card's atomics reorder the adds and its `exp` and Boys differ from the
interpreter's by ulps, each rounding bounded by the same twin
(test_hess_device.py's bound). The runs are compared with each other too.

EVERY COMPILE is one unit alone, killed at `COMPILE_BUDGET_S` = 120 s, the
user's per-.cu budget (data/measured/hkx.json has every unit's figures:
every default unit under 5 s of nvcc and 8 s of hipcc).

THE FRAME ON THE CARD. Under xdist the card is shared by every worker (the
controller holds its lock), and a launch reserves its stack for every
resident thread: at (ff|ff), ~20 KB of `sm_86` stack, ~2 GB of the RTX
3080's 10 GB. With six workers on the card `hkx_33_32` and `hkx_33_33`
failed their launch ("cuda: launch: out of memory", 2026-10-31) and passed
alone; run the `high_l` cases with at most two workers. The RX 9060 XT
ran them at six.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hess_driver import compile_hess, hkx_launches
from oracle import numeric as golden
from test_kernel_device import _toolchain
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.hessian import host
from felizjueves.hessian.kernel_hkx import (
    LOWSTACK_MIN_L,
    UNROLLED_L_MAX,
    hkx_name,
)
from felizjueves.reference.pq import one_centre

DEV_TOL = 1e-14  # of max(twin) (module docstring)
COMPILE_BUDGET_S = 120.0  # the user's per-.cu compile budget

_A, _B, _C = (0.0, 0.1, -0.2), (1.3, -0.4, 0.5), (-0.5, 1.2, 0.7)
_EXP = {0: 0.9, 1: 0.7, 2: 0.6, 3: 0.5}
# shell index 4 * atom + l, single primitives
TRI_SPDF = (Basis(tuple(Shell(l, X, (_EXP[l] * (1.0 + 0.15 * n),), (1.0,))
                        for n, X in enumerate((_A, _B, _C))
                        for l in range(4))),  # noqa: E741
            ((2.0, _A), (1.0, _B), (3.0, _C)))


def _shell(atom: int, l: int) -> int:  # noqa: E741
    return 4 * atom + l


def _pairs(am):
    la, lc, lb, ld = am
    bras = [(_shell(0, la), _shell(1, lc))]
    kets = [(_shell(1, lb), _shell(2, ld)), (_shell(2, lb), _shell(0, ld))]
    return bras, kets


@cache
def _tri_density() -> np.ndarray:
    n = TRI_SPDF[0].n_ao
    X = np.random.default_rng(7).standard_normal((n, n))
    return X @ X.T / n


def _classes(lmax: int):
    r = range(lmax + 1)
    return [(a, c, b, d) for a in r for c in r for b in r for d in r]


@cache
def _tri_host(am) -> tuple[np.ndarray, float]:
    basis, nuclei = TRI_SPDF
    bras, kets = _pairs(am)
    q = [(i, k, j, m) for i, k in bras for j, m in kets]
    n = len(nuclei)
    H, twin = (np.array(host.exchange_hessian(
        basis, _tri_density(), nuclei, one_centre=one_centre, quartets=q,
        absolute=a)).reshape(n, 3, n, 3) for a in (False, True))
    return H, float(np.abs(twin).max())


def _compare(backend: str, what: str, got: np.ndarray, want: np.ndarray,
             scale: float) -> None:
    errs = [float(np.abs(h - want).max() / scale) for h in got]
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} {what}: |H - host| / max(twin) per run {errs}; "
          f"max|H| / max(twin) {float(np.abs(want).max()) / scale:.2e}; "
          f"run-to-run {spread:.2e}")
    assert max(errs) <= DEV_TOL, (backend, what, errs)


def _run(backend, launches, n_atom, request, tmp_path) -> np.ndarray:
    tc = _toolchain(backend, request)
    return compile_hess(launches, n_atom, backend, tc, tmp_path,
                        timeout=COMPILE_BUDGET_S).run()


def _class_case(backend: str, am, request, tmp_path: Path) -> None:
    basis, nuclei = TRI_SPDF
    xs = hkx_launches(basis, nuclei, _tri_density(), [am],
                      pairs={am: _pairs(am)})
    assert [x.n_rec for x in xs] == [1], am
    got = _run(backend, xs, len(nuclei), request, tmp_path)
    want, scale = _tri_host(am)
    # a live witness: H is no round-off of its twin (at (ff|ff) the twin
    # is ~3e3 max|H|, the rule's terms cancelling)
    assert float(np.abs(want).max()) > 1e-6 * scale, am
    _compare(backend, f"tri_spdf {hkx_name(am)}", got, want, scale)


CLASSES = [by_l(am, sum(am) + 2, id=hkx_name(am)) for am in _classes(3)]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("am", CLASSES)
def test_cuda_hkx_class_matches_the_host(am, request, tmp_path) -> None:
    _class_case("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("am", CLASSES)
def test_hip_hkx_class_matches_the_host(am, request, tmp_path) -> None:
    _class_case("hip", am, request, tmp_path)


# --- the differential: rolled against unrolled, s and p -----------------------

DIFF = [am for am in _classes(UNROLLED_L_MAX) if sum(am) >= LOWSTACK_MIN_L]


def _differential(backend: str, am, request, tmp_path: Path) -> None:
    basis, nuclei = TRI_SPDF
    kw = {"pairs": {am: _pairs(am)}}
    (rolled,) = hkx_launches(basis, nuclei, _tri_density(), [am], **kw)
    (unrolled,) = hkx_launches(basis, nuclei, _tri_density(), [am],
                               lowstack=False, **kw)
    unrolled = replace(unrolled, kernel=replace(
        unrolled.kernel, name=f"{hkx_name(am)}_unrolled"))
    n = len(nuclei)
    a = _run(backend, [rolled], n, request, tmp_path / "rolled")
    b = _run(backend, [unrolled], n, request, tmp_path / "unrolled")
    _, scale = _tri_host(am)
    _compare(backend, f"tri_spdf {hkx_name(am)} rolled vs unrolled", a,
             b[0], scale)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("am", DIFF, ids=hkx_name)
def test_cuda_hkx_rolled_is_the_unrolled(am, request, tmp_path) -> None:
    _differential("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("am", DIFF, ids=hkx_name)
def test_hip_hkx_rolled_is_the_unrolled(am, request, tmp_path) -> None:
    _differential("hip", am, request, tmp_path)


# --- the whole d2 E_K of a molecule, against libcint ---------------------------

MOLECULES = ("hkx_h2o_sto3g_bse", "hkx_h2o_ccpvdz_bse")


@cache
def _molecule(name: str):
    nuclei = tuple((float(z), tuple(c)) for z, c in golden.atoms(name))
    return golden.basis(name), nuclei, np.asarray(golden.density(name))


def _molecule_case(backend: str, name: str, request, tmp_path: Path) -> None:
    basis, nuclei, D = _molecule(name)
    got = _run(backend, hkx_launches(basis, nuclei, D), len(nuclei), request,
               tmp_path)
    errs = [golden.compare(name, h) for h in got]  # 1e-12 of max|H|
    spread = float(np.abs(got[0] - got[1]).max()
                   / np.abs(golden.want(name)).max())
    print(f"\n{backend} {name}: |H - libcint| / max|H| per run {errs}; "
          f"run-to-run {spread:.2e}")
    M = got[0].reshape(3 * len(nuclei), -1)
    assert np.abs(M - M.T).max() <= 1e-13 * np.abs(M).max()


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", MOLECULES,
                         ids=lambda n: n.removeprefix("hkx_").removesuffix(
                             "_bse"))
def test_cuda_hkx_molecule_matches_pyscf(name, request, tmp_path) -> None:
    _molecule_case("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", MOLECULES,
                         ids=lambda n: n.removeprefix("hkx_").removesuffix(
                             "_bse"))
def test_hip_hkx_molecule_matches_pyscf(name, request, tmp_path) -> None:
    _molecule_case("hip", name, request, tmp_path)


@pytest.mark.slow
def test_the_host_is_the_libcint_golden_on_h2o_sto3g() -> None:
    # Host-only (~40 s): the per-class reference above, on a whole molecule,
    # against libcint's assembled second-derivative integrals.
    name = MOLECULES[0]
    basis, nuclei, D = _molecule(name)
    n = len(nuclei)
    H = np.array(host.exchange_hessian(basis, D, nuclei,
                                       one_centre=one_centre)
                 ).reshape(n, 3, n, 3)
    err = golden.compare(name, H)  # 1e-12 of max|H|
    print(f"\nhost {name}: {err:.2e} of max|H|")


def test_every_class_has_one_bra_record_and_two_segments() -> None:
    # Host-only: the staging the card runs is the witness's quartets.
    basis, nuclei = TRI_SPDF
    for am in [(0, 0, 0, 0), (3, 1, 2, 0), (3, 3, 3, 3)]:
        bras, kets = _pairs(am)
        bra, ket, seg = host.exchange_hessian_records(
            basis, nuclei, am, bra_pairs=bras, ket_pairs=kets)
        assert len(bra["s_ac"]) == 1 and len(ket["s_bd"]) == 2
        assert seg["seg_start"] == [0, 1, 2]
        assert sorted(zip(seg["seg_b"], seg["seg_d"], strict=True)) == [
            (1, 2), (2, 0)]
