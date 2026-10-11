"""`ao.j`: the Coulomb matrix `J_mn`, by two sweeps that must agree, against
PySCF's `get_jk`.

THE TWO WITNESSES.

  internal  `coulomb_matrix_reference` (every ORDERED bra and ket shell pair
            at weight 1, no symmetry anywhere) against `coulomb_matrix`
            (unique pairs, ket off-diagonal weight 2). The off-diagonal
            weight is the order-unity error this design is most likely to
            ship; the reference sweep exists for that and nothing else, and
            `test_the_weight_is_exercised` shows the cases actually feel it.
  external  a tracked fixture: PySCF/libcint's `scf.hf.get_jk` J for the
            density STORED IN THE FIXTURE (`density`, the hilbert matrix
            1/(i+j+1): symmetric, no SCF, bit-reproducible), so both programs
            see identical input. `generate.j_array` cross-checks `get_jk`
            against `einsum(int2e, dm)` before writing it.

TAU = 0.0 THROUGHOUT, and that is the reason screening cannot hide an error
here: at `tau = 0` `ao.j` evaluates no bound and takes no screening branch
(its module docstring), and the bound product is >= 0 anyway, so every
(bra, ket) record pair is summed. The screen's own tests are at the bottom,
and #88 is the tau ladder.

THE CASES. Fast tier: H2/STO-3G, H2O/STO-3G, H2O/6-31G -- s and p only.
`slow`: H2O/cc-pVDZ, by #85's rule (a `d` function puts a case there). One
H2O geometry, no atom on an axis, for every H2O basis.

THE TOLERANCES, stated and not tuned.

  sweeps    |J - J_ref| <= 1e-14 max|J_ref|, element by element (the issue's
            1e-14 relative, relative to the matrix's largest element).
  symmetry  |J - J^T|   <= 1e-15 max|J|, on BOTH sweeps. For the unique
            sweep it holds by construction (it mirrors); on the reference
            sweep `J[a][b]` and `J[b][a]` are computed independently, so
            that is where the 1e-15 says something.
  get_jk    |J - J_pyscf| <= 1e-12 scale_J, element by element, DERIVED:
            `scale_J` (`ao.j.coulomb_matrix_and_scale`) is the same sum with
            every factor in absolute value, which equals
            sum_{ls} scale_eri(mn|ls) |D_ls| for #85's per-element ERI scale.
            #85 holds every ERI to 1e-12 scale_eri against libcint, so by the
            triangle inequality J inherits 1e-12 scale_J -- the same bound,
            not a new one. (1e-12 is ~4500 ulps of the scale; the J sums'
            own rounding is far below it.)

Measured worst on this box (2026-10-11, pyscf 2.14.0 / libcint 6.1.3), all
printed with `-s`:

    case          sweeps      symmetry(ref)  get_jk err/scale
    h2_sto3g_j    1.6e-16     0              3.7e-15
    h2o_sto3g_j   4.6e-16     1.6e-17        5.2e-15
    h2o_631g_j    5.1e-16     2.7e-17        8.6e-15
    h2o_ccpvdz_j  1.1e-15     8.2e-17        1.3e-14

A failure is a defect to find, not a tolerance to widen.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.j import (
    coulomb_matrix,
    coulomb_matrix_and_scale,
    coulomb_matrix_reference,
    jp_from_pq,
    pair_bound_pq,
)
from felizjueves.ao.pairs import pair_bound
from felizjueves.reference.pq import one_centre, pq

TAU = 0.0  # no screening: see the module docstring
SWEEP_TOL = 1e-14
SYM_TOL = 1e-15
JK_TOL = 1e-12

FAST = ("h2_sto3g_j", "h2o_sto3g_j", "h2o_631g_j")
SLOW = ("h2o_ccpvdz_j",)
CASES = FAST + SLOW

PARAMS = ([pytest.param(n, id=n) for n in FAST]
          + [pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW])


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx)


def test_the_cases_are_what_the_issue_names():
    """Fast: s/p only; slow: the case with d functions -- #85's rule."""
    for name in CASES:
        fx, basis = _case(name)
        assert manifest.J_ARRAY in fx.arrays
        lmax = max(s.l for s in basis.shells)
        assert (lmax >= 2) == (name in SLOW), name
    assert {_case(n)[0].provenance.basis for n in CASES} == {
        "sto-3g", "6-31g", "cc-pvdz"}


@pytest.mark.parametrize("name", PARAMS)
def test_coulomb_matrix(name):
    """Both sweeps agree; J is symmetric; J matches get_jk. One test per case
    because each sweep is the cost, and is computed once."""
    fx, basis = _case(name)
    D = fx.density
    J, scale = (np.array(x) for x in
                coulomb_matrix_and_scale(basis, D, pq=pq))
    J0 = np.array(coulomb_matrix(basis, D, pq=pq, tau=TAU))
    R = np.array(coulomb_matrix_reference(basis, D, pq=pq, tau=TAU))
    ref = fx.arrays[manifest.J_ARRAY]

    # coulomb_matrix_and_scale is coulomb_matrix at tau = 0, the same sums.
    assert np.array_equal(J, J0)

    sweeps = float(np.max(np.abs(J - R)) / np.max(np.abs(R)))
    sym_ref = float(np.max(np.abs(R - R.T)) / np.max(np.abs(R)))
    sym = float(np.max(np.abs(J - J.T)) / np.max(np.abs(J)))
    err = np.abs(J - ref)
    assert np.all(scale > 0)
    worst = float(np.max(err / scale))
    r = ref.ravel()
    ratio = float(J.ravel() @ r / (r @ r))  # a uniform constant shows here
    print(f"\n{name}: sweeps {sweeps:.2e}, symmetry ref {sym_ref:.2e} / "
          f"unique {sym:.2e}, max|J - get_jk| = {err.max():.2e}, "
          f"max err/scale = {worst:.2e}, J/get_jk ratio = {ratio:.15g}")

    assert np.all(np.abs(J - R) <= SWEEP_TOL * np.max(np.abs(R))), (
        f"{name}: the two sweeps disagree, worst {sweeps:.2e} of max|J| -- "
        "an off-diagonal weight error shows here first")
    assert sym_ref <= SYM_TOL, f"{name}: reference J asymmetric, {sym_ref:.2e}"
    assert sym <= SYM_TOL, f"{name}: J asymmetric, {sym:.2e}"
    bad = err > JK_TOL * scale
    assert not bad.any(), (
        f"{name}: {int(bad.sum())} of {err.size} elements exceed 1e-12 * "
        f"scale; worst err/scale {worst:.2e}; J/get_jk ratio {ratio:.6g}")


def test_the_weight_is_exercised():
    """The sweeps' agreement means something only if the off-diagonal ket
    shell pairs carry weight. Split D into its diagonal-shell-pair blocks and
    the rest: by linearity a unique sweep with weight 1 instead of 2 would be
    off by exactly J(D_off)/2, which must be far above the 1e-14 the sweeps
    are held to."""
    fx, basis = _case("h2o_sto3g_j")
    D = np.array(fx.density)
    mask = np.zeros_like(D, dtype=bool)
    for s, sh in enumerate(basis.shells):
        o = basis.offsets[s]
        mask[o:o + sh.n_cart, o:o + sh.n_cart] = True
    J = np.array(coulomb_matrix(basis, D, pq=pq))
    j_off = np.array(coulomb_matrix(basis, np.where(mask, 0.0, D), pq=pq))
    j_diag = np.array(coulomb_matrix(basis, np.where(mask, D, 0.0), pq=pq))
    # Measured: 0.53 against max|J| = 6.8, i.e. 7.7%. Any floor eleven orders
    # above SWEEP_TOL makes the point; 1e-3 is that floor.
    assert np.max(np.abs(j_off / 2)) > 1e-3 * np.max(np.abs(J))
    assert np.allclose(j_off + j_diag, J, rtol=0, atol=1e-14 * np.abs(J).max())


# --- the jp seam (#87) ----------------------------------------------------


@pytest.mark.parametrize("name", FAST)
def test_the_jp_seam_is_the_pq_path(name):
    """`jp=jp_from_pq(pq)` is the same J as the `pq` path: only the order of
    the ket sum (grouped by class) differs, so the sweeps' 1e-14 holds."""
    fx, basis = _case(name)
    J = np.array(coulomb_matrix(basis, fx.density, pq=pq))
    S = np.array(coulomb_matrix(basis, fx.density, pq=pq, jp=jp_from_pq(pq)))
    assert np.all(np.abs(S - J) <= SWEEP_TOL * np.max(np.abs(J)))
    assert np.array_equal(S, S.T)


def test_the_jp_seam_refuses_a_screen_and_a_wrong_shape():
    fx, basis = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="unscreened"):
        coulomb_matrix(basis, fx.density, pq=pq, tau=1e-10,
                       jp=jp_from_pq(pq))
    with pytest.raises(ValueError, match="wrong shape"):
        coulomb_matrix(basis, fx.density, pq=pq,
                       jp=lambda b, k, bras, kets: [[0.0, 0.0] for _ in bras])


# --- the density contract -------------------------------------------------


@pytest.mark.parametrize("sweep", [coulomb_matrix, coulomb_matrix_reference])
def test_an_asymmetric_density_is_rejected_not_symmetrised(sweep):
    fx, basis = _case("h2_sto3g_j")
    D = np.array(fx.density)
    D[0, 1] += 1e-12
    with pytest.raises(ValueError, match="not symmetric"):
        sweep(basis, D, pq=pq)


@pytest.mark.parametrize("bad", [np.eye(3), [[1.0, 0.0]], [[1.0, np.nan],
                                                            [np.nan, 1.0]]])
def test_a_malformed_density_is_rejected(bad):
    _, basis = _case("h2_sto3g_j")  # nao = 2
    with pytest.raises(ValueError):
        coulomb_matrix(basis, bad, pq=pq)


def test_a_plain_nested_list_density_is_accepted():
    fx, basis = _case("h2_sto3g_j")
    a = coulomb_matrix(basis, fx.density, pq=pq)
    b = coulomb_matrix(basis, fx.density.tolist(), pq=pq)
    assert a == b


# --- the screen (tau > 0 only; #88 is the ladder) -------------------------


def test_the_bound_through_pq_is_pairs_bound():
    """`I_p` via `[p|p](0)` is `pairs.pair_bound`'s radicand with the sign
    already applied: the two must be the same number."""
    for zeta in (0.05, 0.7, 3.0, 41.0):
        for L in range(5):
            a = pair_bound_pq(zeta, L, pq=pq)
            b = pair_bound(zeta, L, one_centre=one_centre)
            assert a == b, (zeta, L, a, b)


@pytest.mark.parametrize("tau", [-1.0, float("nan"), float("inf")])
def test_a_bad_tau_is_rejected(tau):
    fx, basis = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="tau"):
        coulomb_matrix(basis, fx.density, pq=pq, tau=tau)


@pytest.mark.parametrize("sweep,name", [
    (coulomb_matrix, "h2o_sto3g_j"),
    (coulomb_matrix_reference, "h2_sto3g_j"),  # the costly sweep: small case
])
def test_a_negligible_tau_skips_nothing_and_a_huge_one_everything(sweep, name):
    fx, basis = _case(name)
    J = sweep(basis, fx.density, pq=pq)
    assert sweep(basis, fx.density, pq=pq, tau=1e-300) == J
    zero = sweep(basis, fx.density, pq=pq, tau=1e300)
    assert all(x == 0.0 for row in zero for x in row)


# --- the drift check: needs PySCF -----------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", CASES)
def test_j_fixture_is_what_pyscf_computes_today(name):
    """Regenerate and compare, bytes and all. A failure is news about
    PySCF/libcint (versions and date in the message), not proof the fixture
    was wrong."""
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate

    msg = generate.drift(name)
    assert msg is None, msg
