"""The exchange derivative matrix `kx1_` / `kxc_` ON A CARD (#689), against
`hessian.host_kx` and PySCF (generator `test/hess_driver.py`'s
`kx_launches`, one warp per bra record, the one OUT `X[3, n_ao, n_ao]`).

PER CLASS AND FORM, every K class through (ff|ff) (256 a form), each its
own program in its one (rolled) form, on test_hkx_device.py's `TRI_SPDF`:
three atoms, a single-primitive s, p, d and f shell on each. Each class
launches ONE bra shell pair `(l_a on A, l_c on B)` against TWO ket shell
pairs `(l_b on B, l_d on C)` and `(l_b on C, l_d on A)`, so every centre
of the rule is off the others and the host reference is two quartets.
`kx1_` is one launch (every atom's); `kxc_` is the launch of atom B, the
one its bra's `c` sits on (on A and C it has no record). Marked by
`tiers.by_l` on the raised `L + 1`: `slow` above `FAST_L_MAX`, `high_l`
above `DEVICE_L_MAX` (L + 1 > 8, i.e. L >= 8). This is the issue's
"every class launches on the RTX 3080 ... run on both cards against the
host", and with the interpreted tests (test_kernel_kx.py, through (dd|dd)
and three f witnesses) its "every class through (ff|ff)": the rolled
loops are generic over the class, f adds no code path, and the card runs
every f class against the host.

PER MOLECULE (`slow`), AGAINST PySCF: the three gradient fixtures of
`oracle.manifest.KX_PARENTS` (H2O over STO-3G, STO-3G plus a d on O, and
cc-pVDZ), every K class of the basis: `kx1_` into one `K1`, held to the
fixture's `vk1` (libcint's `int2e_ip1` contracted `ijkl,jk->il`), then
`kxc_` once per atom into its own `KC_X`, the whole `K^X = P_X K1 + (P_X
K1)^T + KC_X + KC_X^T` held to the tracked witness of PySCF's
`hessian.rhf.make_h1` exchange part, and `SUM_X K^X = 0`.

THE BOUNDS, stated before any run. Per class `|X_dev - X_host| <= DEV_TOL
max(twin)`, `DEV_TOL = 1e-14` (~45 eps), `twin` the host's
`absolute=True` sum: test_hkx_device.py's bound for the same reason (the
interpreted kernels are within 16 eps of the twin; the card's atomics
reorder the adds and its `exp` and Boys differ from the interpreter's by
ulps). Against PySCF, `REL_TOL = 1e-12` of `max|PySCF|`, `oracle.numeric`'s
fp64 tier (the host is within 1.2e-13 of it, test_hessian_host_kx.py);
`SUM_X K^X` within `1e-13 max|K1|` (test_kernel_kx.py's). The two runs
are compared with each other too.

EVERY COMPILE is one unit alone, killed at `COMPILE_BUDGET_S` = 120 s, the
user's per-.cu budget (data/measured/kx.json: every unit under 6 s on
either compiler, at most 9,824 B of `sm_86` stack and 10,192 B of
`gfx1200` scratch at `kx1_33_33`). A launch reserves its stack for every
resident thread (~1 GB of the RTX 3080's 10 GB at (ff|ff)); test_hkx_
device.py's `hkx_33_3x` failed their launch with six xdist workers on the
card at twice that frame, so run the `high_l` cases with at most two.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hess_driver import compile_hess, kx_launches
from oracle import basis as ob
from oracle import manifest
from test_hkx_device import TRI_SPDF, _pairs, _tri_density
from test_kernel_device import _toolchain
from tiers import by_l

from felizjueves.gradient import host as ghost
from felizjueves.hessian import host_kx
from felizjueves.hessian.kernel_kx import kx_name, kx_spec
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

DEV_TOL = 1e-14  # of max(twin) (module docstring)
REL_TOL = 1e-12  # of max|PySCF| (module docstring)
SUM_TOL = 1e-13  # of max|K1|
COMPILE_BUDGET_S = 120.0  # the user's per-.cu compile budget
KX1, KXC = Target.KX1, Target.KXC
B = 1  # the atom the witness bra's c sits on: kxc_'s launch


def _classes(lmax: int):
    r = range(lmax + 1)
    return [(a, c, b, d) for a in r for c in r for b in r for d in r]


def _run(backend, launches, n_ao, request, tmp_path, binary=None):
    """`[run][3][n_ao][n_ao]`; `binary` reuses compiled runners (a `kxc_`
    launch list per atom over one compile)."""
    if binary is None:
        tc = _toolchain(backend, request)
        binary = compile_hess(launches, 0, backend, tc, tmp_path,
                              n_out=3 * n_ao * n_ao,
                              timeout=COMPILE_BUDGET_S)
    else:
        binary = replace(binary, launches=tuple(launches),
                         workdir=tmp_path)
    got = binary.run()
    return got.reshape(len(got), 3, n_ao, n_ao), binary


@cache
def _tri_host(am, form: Target) -> tuple[np.ndarray, float]:
    basis, nuclei = TRI_SPDF
    bras, kets = _pairs(am)
    q = [(i, k, j, m) for i, k in bras for j, m in kets]
    D = _tri_density()
    if form is KX1:
        f = host_kx.exchange_derivative_ip1
        kw = {}
    else:
        f, kw = host_kx.exchange_derivative_contracted, {"atom": B}
    want, twin = (np.array(f(basis, D, nuclei, one_centre=one_centre,
                             quartets=q, absolute=a, **kw))
                  for a in (False, True))
    return want, float(np.abs(twin).max())


def _compare(label: str, got: np.ndarray, want: np.ndarray, scale: float,
             tol: float) -> None:
    errs = [float(np.abs(x - want).max() / scale) for x in got]
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{label}: |X - want| / scale per run {errs}; "
          f"max|want| / scale {float(np.abs(want).max()) / scale:.2e}; "
          f"run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")
    assert max(errs) <= tol, (label, errs, tol)


def _class_case(backend: str, spec, request, tmp_path: Path) -> None:
    basis, nuclei = TRI_SPDF
    am, form = spec.am, spec.target
    atom = None if form is KX1 else B
    xs = kx_launches(basis, nuclei, _tri_density(), [am], form=form,
                     atom=atom, pairs={am: _pairs(am)})
    assert [x.n_rec for x in xs] == [1], am
    assert [x.arrays["s_bd"].size for x in xs] == [2], am
    got, _ = _run(backend, xs, basis.n_ao, request, tmp_path)
    want, scale = _tri_host(am, form)
    assert float(np.abs(want).max()) > 1e-6 * scale, am  # a live witness
    _compare(f"{backend} tri_spdf {kx_name(spec)}", got, want, scale,
             DEV_TOL)


CLASSES = [by_l(kx_spec(*am, form=f), sum(am) + 1,
                id=kx_name(am, form=f))
           for f in (KX1, KXC) for am in _classes(3)]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", CLASSES)
def test_cuda_kx_class_matches_the_host(spec, request, tmp_path) -> None:
    _class_case("cuda", spec, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", CLASSES)
def test_hip_kx_class_matches_the_host(spec, request, tmp_path) -> None:
    _class_case("hip", spec, request, tmp_path)


# --- the whole K^X of a molecule, against PySCF --------------------------------

MOLECULES = [pytest.param(p, id=p.removesuffix("_grad"),
                          marks=(pytest.mark.slow,))
             for p in manifest.KX_PARENTS]


@cache
def _molecule(parent: str):
    fx = manifest.gradient_fixture(parent)
    nuclei = tuple((float(x.charge), x.coord) for x in fx.atoms)
    return fx, ob.from_fixture(fx), nuclei


def _on(basis, nuclei, atom: int) -> np.ndarray:
    at = ghost.shell_atoms(basis, nuclei)
    on = np.zeros(basis.n_ao, dtype=bool)
    for s, o in enumerate(basis.offsets):
        on[o:o + basis.shells[s].n_cart] = at[s] == atom
    return on


def _molecule_case(backend: str, parent: str, request,
                   tmp_path: Path) -> None:
    fx, basis, nuclei = _molecule(parent)
    D, n = np.asarray(fx.density), basis.n_ao
    k1s, _ = _run(backend, kx_launches(basis, nuclei, D, form=KX1), n,
                  request, tmp_path / "kx1")
    vk1 = np.asarray(fx.arrays["vk1"])
    _compare(f"{backend} {parent} K1 vs PySCF vk1", k1s, vk1,
             float(np.abs(vk1).max()), REL_TOL)
    want = manifest.kx_witness(parent).kx
    per_atom = [kx_launches(basis, nuclei, D, form=KXC, atom=atom)
                for atom in range(len(nuclei))]
    tc = _toolchain(backend, request)  # every atom's kxc_ units, once
    binary = compile_hess([x for xs in per_atom for x in xs], 0, backend,
                          tc, tmp_path / "kxc", n_out=3 * n * n,
                          timeout=COMPILE_BUDGET_S)
    kcs = [_run(backend, xs, n, request, tmp_path / f"kxc{atom}",
                binary)[0] for atom, xs in enumerate(per_atom)]
    scale = float(np.abs(want).max())
    for r in range(len(k1s)):
        total = np.zeros((3, n, n))
        for atom in range(len(nuclei)):
            p = np.where(_on(basis, nuclei, atom)[None, :, None],
                         k1s[r], 0.0)
            kc = kcs[atom][r]
            got = p + p.transpose(0, 2, 1) + kc + kc.transpose(0, 2, 1)
            err = float(np.abs(got - want[atom]).max() / scale)
            print(f"{backend} {parent} run {r} atom {atom}: |K^X - PySCF|"
                  f" / max|PySCF| {err:.2e}")
            assert err <= REL_TOL, (parent, r, atom, err)
            total += got
        s = float(np.abs(total).max() / np.abs(k1s[r]).max())
        print(f"{backend} {parent} run {r}: max|SUM_X K^X| / max|K1| "
              f"{s:.2e}")
        assert s <= SUM_TOL, (parent, r, s)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("parent", MOLECULES)
def test_cuda_kx_molecule_matches_pyscf(parent, request, tmp_path) -> None:
    _molecule_case("cuda", parent, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("parent", MOLECULES)
def test_hip_kx_molecule_matches_pyscf(parent, request, tmp_path) -> None:
    _molecule_case("hip", parent, request, tmp_path)


# --- host-only: the staging the card runs --------------------------------------


def test_every_class_launches_one_bra_record_against_two_kets() -> None:
    basis, nuclei = TRI_SPDF
    D = _tri_density()
    for am in [(0, 0, 0, 0), (3, 1, 2, 0), (3, 3, 3, 3)]:
        pairs = {am: _pairs(am)}
        for form, atom in ((KX1, None), (KXC, B)):
            (x,) = kx_launches(basis, nuclei, D, [am], form=form,
                               atom=atom, pairs=pairs)
            assert x.n_rec == 1 and x.arrays["s_bd"].size == 2
            assert x.out == "X" and x.warp_per_record
        for atom in (0, 2):  # no c of the witness bra sits there
            assert kx_launches(basis, nuclei, D, [am], form=KXC, atom=atom,
                               pairs=pairs) == []


def test_a_form_and_an_atom_that_disagree_are_refused() -> None:
    basis, nuclei = TRI_SPDF
    D = _tri_density()
    with pytest.raises(ValueError, match="per atom"):
        kx_launches(basis, nuclei, D, [(0, 0, 0, 0)], form=KX1, atom=0)
    with pytest.raises(ValueError, match="per atom"):
        kx_launches(basis, nuclei, D, [(0, 0, 0, 0)], form=KXC)
