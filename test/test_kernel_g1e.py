"""`kernel_g1e`: the `S`, `T` and `V` gradient kernels (d1), interpreted.

THE WITNESS is `gradient.host` (itself held to finite differences of the
libcint-witnessed integrals, test/test_gradient_host.py): per class, one
launch over every record of every shell pair `i <= j` of that class, into
a zeroed `G`, against `host.*_gradient(..., pairs=)` of the same pairs.

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ (the
  one fixture whose shells reach f, both orientations occurring with
  `i <= j`), `M` non-symmetric (the weight rule is exact for any `M`).
- THE WHOLE GRADIENT, one launch per class into one `G`, on H2O/STO-3G.

ROUND-OFF, NOT BITWISE: `|G_kernel - G_host| <= TOL max(scale)` with
`TOL = 16 eps`. The kernel forms each component as one expression and `V`
regroups by Hermite index before its nucleus sum, against the host's row
by row sums and `reference.pq`'s fixed recursion. `scale` is the sum, over
the class's shell pairs, of `|G|` of each pair alone with EVERY SHELL ON
A ROW OF ITS OWN (`host`'s `atoms=`), folded back onto the atoms: the
size of the terms before they cancel. `max|G|` is not a scale -- on a
one-centre pair (F's own s-f, d-f) `dA` and `dB` cancel exactly on one
atom and `V`'s row is round-off plus its Hellmann-Feynman term (81 eps of
`max|G|` at (sf) when tried). Measured: `gs` and `gt` bitwise, `gv` worst
2.2 eps. An order-unity error is ~1 of the scale.

`L` for the tier cut is `la + lb + 1`, the Hermite order a gradient
reaches (test/tiers.py): (sd), (pp) and below are fast, the rest `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient.kernel_g1e import (
    ATOMIC_REASON,
    LOWSTACK_MIN_LSUM,
    ZEROED_G,
    build_g1e_kernel,
    g1e_lowstack,
    g1e_name,
    g1e_spec,
    launch_shape,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
TARGETS = (Target.S, Target.T, Target.V)
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _m(n: int) -> np.ndarray:
    return np.random.default_rng(3).standard_normal((n, n))


def _launch(target, basis, nuclei, M, la, lb, G, threads=7, lowstack=None):
    rec = host.records(basis, nuclei, la, lb)
    kern = build_g1e_kernel((target, la, lb), lowstack=lowstack)
    want = {p.name for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if k.startswith(("off", "atom"))
                        else float) for k, v in rec.items() if k in want}
    if target is Target.V:
        arrs["Z"] = np.array([z for z, _ in nuclei])
        arrs["Cnuc"] = np.array([c for _, c in nuclei])
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1), boys=boys).run(
        n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), M=M, G=G,
        **arrs)["G"]


def _host(target, basis, nuclei, M, pairs=None, atoms=None) -> np.ndarray:
    kw = dict(pairs=pairs, atoms=atoms)
    if target is Target.S:
        g = host.overlap_gradient(basis, M, nuclei, **kw)
    elif target is Target.T:
        g = host.kinetic_gradient(basis, M, nuclei, **kw)
    else:
        g = host.potential_gradient(basis, M, nuclei, one_centre=one_centre,
                                    **kw)
    return np.array(g)


def _scale(target, basis, nuclei, M, pairs) -> float:
    """The module docstring's `scale`, its largest element."""
    nat = len(nuclei)
    real = host.shell_atoms(basis, nuclei)
    split = tuple(nat + i for i in range(len(basis.shells)))
    sc = np.zeros((nat, 3))
    for p in pairs:
        g = np.abs(_host(target, basis, nuclei, M, [p], atoms=split))
        sc += g[:nat]
        for i, row in enumerate(split):
            sc[real[i]] += g[row]
    return float(sc.max())


def _close(got, want, scale: float, what: str) -> None:
    if scale == 0.0:  # every term zero by parity (a one-centre p-f overlap)
        assert np.all(got == 0.0) and np.all(want == 0.0), what
        return
    err = np.abs(got - want).max()
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls, target) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = _launch(target, basis, nuclei, M, *cls, np.zeros((len(nuclei), 3)))
    _close(G, _host(target, basis, nuclei, M, pairs),
           _scale(target, basis, nuclei, M, pairs), f"{target} {cls}")


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_gv_in_its_other_form_is_the_host_gradient(cls) -> None:
    """THE LOW-STACK gv (#614): every class in the form its default is not
    -- forced low-stack below the floor, forced unrolled above it -- held
    to the same bound."""
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    M = _m(basis.n_ao)
    other = not g1e_lowstack((Target.V, *cls))
    G = _launch(Target.V, basis, nuclei, M, *cls,
                np.zeros((len(nuclei), 3)), lowstack=other)
    _close(G, _host(Target.V, basis, nuclei, M, pairs),
           _scale(Target.V, basis, nuclei, M, pairs), f"gv {cls} other form")


def test_the_low_stack_rule_and_its_refusals() -> None:
    gated = {c for c in CLASSES if g1e_lowstack((Target.V, *c))}
    assert LOWSTACK_MIN_LSUM == 5
    assert gated == {(2, 3), (3, 2), (3, 3)}
    for t in (Target.S, Target.T):
        assert not any(g1e_lowstack((t, *c)) for c in CLASSES)
        with pytest.raises(ValueError, match="only gv has a low-stack"):
            build_g1e_kernel((t, 3, 3), lowstack=True)
    # one name, one parameter list, in both forms
    for c in ((1, 1), (3, 3)):
        a = build_g1e_kernel((Target.V, *c), lowstack=True)
        b = build_g1e_kernel((Target.V, *c), lowstack=False)
        assert a.name == b.name and a.params == b.params


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_whole_gradient_of_h2o_is_the_host_gradient(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G = np.zeros((len(nuclei), 3))
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        G = _launch(target, basis, nuclei, M, la, lb, G)
    pairs = [(i, j) for i in range(len(basis.shells))
             for j in range(i, len(basis.shells))]
    _close(G, _host(target, basis, nuclei, M),
           _scale(target, basis, nuclei, M, pairs), f"{target} H2O/STO-3G")
    assert np.abs(G.sum(axis=0)).max() <= 1e-13 * np.abs(G).max()


def test_g_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G0 = np.arange(9.0).reshape(3, 3)
    G = _launch(Target.S, basis, nuclei, M, 0, 1, G0.copy())
    want = _launch(Target.S, basis, nuclei, M, 0, 1, np.zeros((3, 3)))
    # The records add into G0's values one by one: equal to G0 + want up to
    # the reassociation of those adds.
    assert np.abs(G - (G0 + want)).max() <= 4 * EPS * np.abs(G0).max()
    assert not np.array_equal(G, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    for t, prefix in zip(TARGETS, ("gs", "gt", "gv"), strict=True):
        assert g1e_spec(t, 2, 1).key().split("/")[1] == "d1"
        assert g1e_name((t, 2, 1)) == f"{prefix}_21"
        assert build_g1e_kernel(g1e_spec(t, 2, 1)).name == f"{prefix}_21"


def test_a_d0_key_or_another_target_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d1 S, T or V key"):
        build_g1e_kernel(Spec(Operator.OVERLAP, 0, (1, 0), Target.S))
    with pytest.raises(ValueError, match="not a d1 S, T or V key"):
        build_g1e_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.JAB))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_parameters_and_the_one_output(target) -> None:
    k = build_g1e_kernel((target, 1, 1))
    names = [p.name for p in k.params]
    head = ["n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y", "pa_z",
            "pb_x", "pb_y", "pb_z"]
    p = ["p_x", "p_y", "p_z"] if target is Target.V else []
    tail = ["s_ab", "ea_bra", "eb_ket", "off_a", "off_b", "atom_a", "atom_b",
            "M"]
    nuc = ["Z", "Cnuc"] if target is Target.V else []
    assert names == [*head, *p, *tail, *nuc, "G"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["G"]


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_every_write_is_an_atomic_add_into_g(target) -> None:
    k = build_g1e_kernel((target, 1, 2))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert adds and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)
    # V adds the nucleus rows inside its loop: 6 centre adds + 3 per nucleus.
    assert len(adds) == (9 if target is Target.V else 6)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
