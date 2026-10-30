"""`kernel_kx`: the exchange derivative matrix kernels (`Target.KX1`,
`Target.KXC` at d1, #689), interpreted.

THE WITNESS is `hessian.host_kx` (held to PySCF's `int2e_ip1` through the
tracked fixtures, test/test_hessian_host_kx.py). The two share the records
and the rule and nothing of the arithmetic's order: the host
differentiates `E^{ac}` before its fold and runs `reference.pq`'s
`[r]^(0)`; the kernel folds rolled sums over compact tables one index up
and runs the plan's MILD recurrence one bra order up (`kx_pq_spec`).

- PER CLASS AND FORM, one launch over every ordered bra record of `(la,
  lc)` (`kxc`: once per atom, the bra records whose `c` sits on it)
  against every ordered ket record of `(lb, ld)` (`host_kx.kx_records`),
  against the host over `gradient.host.class_quartets`: every K class with
  every shell s or p on test_gradient_host.py's two-atom s, p basis, and
  every class with a d shell through `(dd|dd)` on test_kernel_gk.py's
  two-atom s, p, d basis (classes above `FAST_L_MAX` `slow`, by
  `tiers.by_l` on the raised `L + 1`).
- THE f WITNESSES (`slow`): `(ff|ff)`, `(fs|fs)` and `(fd|pd)` on
  test_kernel_gk.py's s, p, d, f basis, `(ff|ff)` over one off-centre bra
  pair against the two off-centre ket pairs (an interpreted `(ff|ff)`
  quartet takes minutes), as test_kernel_gk.py's are.
- THE WHOLE `K^X` of the s, p basis: the 16 class launches of each form
  assembled (`P_X K1 + (P_X K1)^T + KC_X + KC_X^T`) against the host's,
  and `SUM_X K^X = 0` (a translation of every centre leaves `K` fixed).
- LANES: one lane, four and 32 per warp give the same `X` to round-off.

ROUND-OFF: `|X_kernel - X_host| <= TOL max(scale)`, `TOL = 16 eps`,
`scale` the host's `absolute=True` twin (every factor in `|.|`, both terms
of the derivative rule added).
"""

from __future__ import annotations

import numpy as np
import pytest
from test_gradient_host import PAIR_SP
from test_kernel_gk import SPDF, _case, _density, _only
from tiers import by_l

from felizjueves.gradient import host as ghost
from felizjueves.hessian import host_kx
from felizjueves.hessian.kernel_kx import (
    FORMS,
    build_kx_kernel,
    kx_name,
    kx_pq_spec,
    kx_spec,
    launch_shape,
    params,
)
from felizjueves.ir import AtomicAdd, Intent, WarpReduce, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.kernel import split
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS
KX1, KXC = Target.KX1, Target.KXC


def _atoms(name: str) -> list[int | None]:
    return list(range(len(_case(name)[1])))


def _launch(name: str, am, form: Target, X: np.ndarray, *,
            atom: int | None = None, lanes: int = 4, bra_pairs=None,
            ket_pairs=None) -> np.ndarray:
    """One launch over the class's records (`kxc`: those whose `c` sits on
    `atom`), or only those of the shell pairs `bra_pairs` / `ket_pairs`."""
    basis, nuclei = _case(name)
    bra, ket = host_kx.kx_records(basis, nuclei, am, atom=atom)
    del bra["ec_bra" if form is KX1 else "ea_bra"]
    off = basis.offsets
    if bra_pairs is not None:
        bra = _only(bra, ("off_a", "off_c"),
                    {(off[i], off[k]) for i, k in bra_pairs})
    if ket_pairs is not None:
        ket = _only(ket, ("off_b", "off_d"),
                    {(off[j], off[m]) for j, m in ket_pairs})
    n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
    if n_bra == 0:
        return X
    arrays = {k: np.array(v, dtype=np.int32 if k.startswith("off")
                          else np.float64) for k, v in {**bra, **ket}.items()}
    grid, block = launch_shape(n_bra, lanes)
    return Interpreter(build_kx_kernel(am, form=form),
                       launch=Launch(grid, block, lanes), boys=boys).run(
        n_bra=n_bra, n_ket=n_ket, n_ao=basis.n_ao, M=_density(name), X=X,
        **arrays)["X"]


def _host(name: str, form: Target, quartets=None, *, atom=None,
          absolute: bool = False) -> np.ndarray:
    basis, nuclei = _case(name)
    if form is KX1:
        return np.array(host_kx.exchange_derivative_ip1(
            basis, _density(name), nuclei, one_centre=one_centre,
            quartets=quartets, absolute=absolute))
    return np.array(host_kx.exchange_derivative_contracted(
        basis, _density(name), nuclei, atom, one_centre=one_centre,
        quartets=quartets, absolute=absolute))


def _zeros(name: str) -> np.ndarray:
    n = _case(name)[0].n_ao
    return np.zeros((3, n, n))


def _classes(lmax: int, *, need: int = 0):
    r = range(lmax + 1)
    return [c for c in ((a, b, c, d) for a in r for b in r for c in r
                        for d in r) if max(c) >= need]


def _param(name: str, am, form: Target):
    return by_l((name, am, form), sum(am) + 1,
                id=f"{name}-{kx_name(am, form=form)}")


CASES = [*(_param("pair_sp", am, f) for am in _classes(1) for f in FORMS),
         *(_param("spd", am, f) for am in _classes(2, need=2)
           for f in FORMS)]


def _check(name, am, form, *, atom, bras=None, kets=None) -> None:
    basis = _case(name)[0]
    pairs = ghost.exchange_pairs
    bras = bras or pairs(basis, am[0], am[1])
    kets = kets or pairs(basis, am[2], am[3])
    q = [(i, k, j, m) for i, k in bras for j, m in kets]
    X = _launch(name, am, form, _zeros(name), atom=atom, bra_pairs=bras,
                ket_pairs=kets)
    want = _host(name, form, q, atom=atom)
    sc = float(_host(name, form, q, atom=atom, absolute=True).max())
    err = float(np.abs(X - want).max())
    assert err <= TOL * max(sc, np.finfo(float).tiny), (
        f"{kx_name(am, form=form)} atom {atom}: "
        f"{err / max(sc, 1e-300) / EPS:.2f} eps of scale")


@pytest.mark.parametrize("case", CASES)
def test_one_launch_over_the_class_is_the_host_matrix(case) -> None:
    name, am, form = case
    assert ghost.class_quartets(_case(name)[0], am), am
    for atom in ([None] if form is KX1 else _atoms(name)):
        _check(name, am, form, atom=atom)


# (class, bra shell pairs, ket shell pairs) on SPDF: shells 0 s_A, 1 d_A,
# 2 f_A, 3 p_B, 4 d_B, 5 f_B (module docstring, THE f WITNESSES)
F_WITNESSES = [
    ((3, 3, 3, 3), [(2, 5)], [(2, 5), (5, 2)]),
    ((3, 0, 3, 0), None, None),
    ((3, 2, 1, 2), None, None),
]


@pytest.mark.slow
@pytest.mark.parametrize("form", FORMS, ids=lambda f: f.value)
@pytest.mark.parametrize(("am", "bras", "kets"), F_WITNESSES,
                         ids=lambda v: "".join(map(str, v))
                         if isinstance(v, tuple) else None)
def test_an_f_witness_is_the_host_matrix(am, bras, kets, form) -> None:
    basis, nuclei = SPDF
    atoms = [None] if form is KX1 else range(len(nuclei))
    for atom in atoms:
        _check("spdf", am, form, atom=atom, bras=bras, kets=kets)


def _whole(form: Target, atom=None) -> np.ndarray:
    X = _zeros("pair_sp")
    for am in _classes(1):
        X = _launch("pair_sp", am, form, X, atom=atom)
    return X


def _assemble(k1: np.ndarray, kc: np.ndarray, atom: int) -> np.ndarray:
    basis, nuclei = PAIR_SP
    at = ghost.shell_atoms(basis, nuclei)
    on = np.zeros(basis.n_ao, dtype=bool)
    for s, o in enumerate(basis.offsets):
        on[o:o + basis.shells[s].n_cart] = at[s] == atom
    p = np.where(on[None, :, None], k1, 0.0)
    return p + p.transpose(0, 2, 1) + kc + kc.transpose(0, 2, 1)


@pytest.mark.slow
def test_the_classes_assemble_the_whole_k_x_and_it_sums_to_zero() -> None:
    basis, nuclei = PAIR_SP
    D = _density("pair_sp")
    k1 = _whole(KX1)
    sc = float(_host("pair_sp", KX1, absolute=True).max())
    assert np.abs(k1 - _host("pair_sp", KX1)).max() <= TOL * sc
    total = np.zeros_like(k1)
    for atom in range(len(nuclei)):
        got = _assemble(k1, _whole(KXC, atom), atom)
        want = np.array(host_kx.exchange_derivative(
            basis, D, nuclei, atom, one_centre=one_centre, k1=k1.tolist()))
        assert np.abs(got - want).max() <= 4 * TOL * sc, atom
        total += got
    assert np.abs(total).max() <= 1e-13 * np.abs(k1).max()


@pytest.mark.parametrize("lanes", [1, 32])
@pytest.mark.parametrize("form", FORMS, ids=lambda f: f.value)
def test_the_lane_count_is_a_reassociation(lanes: int, form) -> None:
    am = (0, 0, 0, 1)
    atom = None if form is KX1 else 1
    ref = _launch("pair_sp", am, form, _zeros("pair_sp"), atom=atom)
    got = _launch("pair_sp", am, form, _zeros("pair_sp"), atom=atom,
                  lanes=lanes)
    q = ghost.class_quartets(PAIR_SP[0], am)
    sc = float(_host("pair_sp", form, q, atom=atom, absolute=True).max())
    assert np.abs(got - ref).max() <= TOL * sc


# --- the kernel's shape -------------------------------------------------------


def test_the_keys_the_names_and_the_plan_class() -> None:
    assert kx_spec(2, 1, 1, 0).key() == "coulomb/d1/(2,1|1,0)/b1/kx1/literal"
    assert kx_spec(2, 1, 1, 0, form=KXC).key() == (
        "coulomb/d1/(2,1|1,0)/b1/kxc/literal")
    assert kx_name((2, 1, 1, 0)) == build_kx_kernel((2, 1, 1, 0)).name \
        == "kx1_21_10"
    assert kx_name(kx_spec(2, 1, 1, 0, form=KXC)) == "kxc_21_10"
    pqs = kx_pq_spec(2, 1, 1, 0)
    assert pqs.target is Target.K and pqs.derivative_order == 0
    assert split(pqs) == (4, 1)


def test_the_prefixes_collide_with_no_other_unit() -> None:
    # every other kernel family's prefix is a name ending in "_"; none of
    # them is, or starts, "kx1_" / "kxc_" (gk_, hk_, hkx_, ghk_, k_, kb_..)
    for other in ("gk_", "ghk_", "hk_", "hkx_", "k_", "kb_", "kbr_", "kbrc_",
                  "kbd_", "kbrt_", "kbs_", "rk_", "f_", "gf_"):
        for mine in ("kx1_", "kxc_"):
            assert not mine.startswith(other) and not other.startswith(mine)


@pytest.mark.parametrize("spec", [
    Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K),
    Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.KX1, n_batches=3),
], ids=["gk", "batched"])
def test_what_is_not_a_one_batch_kx_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="KX1 / KXC key|one batch only"):
        build_kx_kernel(spec)


def test_a_form_that_contradicts_the_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="not its target"):
        build_kx_kernel(kx_spec(1, 0, 0, 0), form=KXC)


@pytest.mark.parametrize("order", [0, 2])
def test_the_matrix_targets_exist_at_d1_only(order: int) -> None:
    for t in FORMS:
        with pytest.raises(ValueError, match="derivative order 1 only"):
            Spec(Operator.COULOMB, order, (1, 0, 0, 0), t)


def test_the_parameters_the_one_output_and_the_scatter() -> None:
    for form, expo in ((KX1, "ea_bra"), (KXC, "ec_bra")):
        k = build_kx_kernel((1, 0, 1, 1), form=form)
        assert tuple(p.name for p in k.params) == params(form)
        assert expo in params(form)
        out = [p for p in k.params if p.intent is Intent.OUT]
        assert [p.name for p in out] == ["X"]
        stmts = list(walk(k.body))
        adds = [s for s in stmts if isinstance(s, AtomicAdd)]
        assert len(adds) == 3 and all(a.dst.array == "X" for a in adds)
        assert not any(isinstance(s, WarpReduce) for s in stmts)


def test_each_form_holds_its_moved_side_one_index_up() -> None:
    # kx1 raises a only: tables (la + 1, lc); kxc raises c: (la, lc + 1)
    a = repr(build_kx_kernel((1, 2, 0, 0), form=KX1).body)
    c = repr(build_kx_kernel((1, 2, 0, 0), form=KXC).body)
    assert "i <= 2, j <= 2" in a and "i <= 1, j <= 3" in c
