"""`ao.schwarz`: the Schwarz matrix `sqrt((mn|mn))`, host-side, against libcint.

THREE WITNESSES, each of a different claim:

1. LIBCINT'S `(mn|mn)`, the `int2e_diag` array of the three `_schwarz`
   fixtures (H2/STO-3G, H2O/STO-3G, H2O/6-31G: s and p only, so every class
   is at L = 2 (la + lb) <= 4). Element by element, `|ours - libcint| <=
   1e-12 * scale`, `scale` being `ao.eri`'s absolute-value twin -- test_ao_eri's
   tolerance and its reason, unchanged.
2. THE INEQUALITY ITSELF, against libcint's own off-diagonal blocks: for every
   tracked `int2e` block whose two pair classes are at L <= 4,
   `|(ab|cd)| <= Q_ab Q_cd`. One-sided -- it cannot show `Q` right, only not
   too small -- but it is the property the matrix exists for, and it needs no
   new data. The factor `1 + 1e-12` is the rounding of a computed product,
   stated, not fitted.
3. THE STAGING (`schwarz_classes`): every shell pair `i <= j` staged exactly
   once, `la <= lb`, its records `ao.density.epair`'s, the ranges contiguous.

Fast tier: no compiler, no card, no PySCF (the drift test is `oracle`).
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.density import epair
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.schwarz import (
    diagonal_and_scale,
    diagonal_matrix,
    scale_matrix,
    schwarz_classes,
    schwarz_matrix,
)
from felizjueves.reference.pq import pq

TOL = 1e-12  # x scale: test_ao_eri.py's tolerance, for the same arithmetic
FIXTURES = ("h2_sto3g_schwarz", "h2o_sto3g_schwarz", "h2o_631g_schwarz")
# the tracked libcint ERI blocks the inequality is read against
ERI_FIXTURES = ("h2o_631g_eri", "hf_ccpvtz_eri")
L_MAX = 4  # every class tested has L = 2 (la + lb) <= 4


@cache
def _fixture(name: str) -> manifest.Fixture:
    return manifest.load(manifest.FIXTURES / f"{name}.json")


@cache
def _basis(name: str) -> Basis:
    return ob.from_fixture(_fixture(name))


@pytest.mark.parametrize("name", FIXTURES)
def test_the_fixture_is_s_and_p_only_and_its_inputs_are_the_j_case(name):
    fx = _fixture(name)
    assert max(s.l for s in _basis(name).shells) <= 1
    j = _fixture(name.replace("_schwarz", "_j"))
    assert fx.converted == j.converted and fx.nao == j.nao
    assert fx.provenance.mol == j.provenance.mol


@pytest.mark.parametrize("name", FIXTURES)
def test_the_diagonal_matches_libcint(name):
    basis = _basis(name)
    ours = np.array(diagonal_matrix(basis, pq=pq))
    scale = np.array(scale_matrix(basis, pq=pq))
    ref = _fixture(name).arrays[manifest.SCHWARZ_ARRAY]
    err = np.abs(ours - ref)
    worst = float(np.max(err / scale))
    print(f"\n{name}: max |(mn|mn) - libcint| = {err.max():.2e}, worst "
          f"{worst:.2e} of scale")
    assert np.all(scale > 0.0)
    assert np.all(err <= TOL * scale), worst
    assert np.array_equal(ours, ours.T)  # the placement mirrors one value


@pytest.mark.parametrize("name", FIXTURES)
def test_the_schwarz_matrix_is_the_root_of_the_diagonal(name):
    basis = _basis(name)
    q = np.array(schwarz_matrix(basis, pq=pq))
    d = np.array(diagonal_matrix(basis, pq=pq))
    assert np.array_equal(q, np.sqrt(np.maximum(d, 0.0)))
    # ... and the diagonal of the matrix is sqrt((mm|mm)): every one positive
    assert np.all(np.diag(q) > 0.0)


def test_the_clamp_takes_a_rounded_negative_to_zero(monkeypatch):
    import felizjueves.eri.schwarz as hs

    basis = Basis((Shell(0, (0.0, 0.0, 0.0), (1.0,), (1.0,)),))
    monkeypatch.setattr(hs, "diagonal_and_scale",
                        lambda sa, sb, *, pq: ((-1e-30,), (1.0,)))
    assert hs.schwarz_matrix(basis, pq=pq) == [[0.0]]


# ---------------------------------------------------------------------------
# The inequality, against libcint's own blocks
# ---------------------------------------------------------------------------


def _inequality_cases():
    out = []
    for name in ERI_FIXTURES:
        sh = _basis(name).shells
        for q in sorted(manifest.eri_blocks(_fixture(name))):
            ls = [sh[s].l for s in q]
            if 2 * (ls[0] + ls[1]) <= L_MAX and 2 * (ls[2] + ls[3]) <= L_MAX:
                out.append(pytest.param(name, q, id=f"{name}-{q}"))
    return out


INEQUALITY = _inequality_cases()


def test_the_inequality_reaches_every_class_at_or_under_l4():
    cls = set()
    for p in INEQUALITY:
        name, q = p.values
        sh = _basis(name).shells
        cls.add(tuple(sh[s].l for s in q))
    assert len(INEQUALITY) >= 10
    assert {(1, 1, 1, 1), (1, 0, 1, 0), (0, 0, 0, 0)} <= cls, cls


@pytest.mark.parametrize("name,q", INEQUALITY)
def test_every_libcint_element_is_under_its_schwarz_product(name, q):
    sh = _basis(name).shells
    blk = manifest.eri_blocks(_fixture(name))[q]
    sa, sb, sc, sd = (sh[s] for s in q)
    qab = np.sqrt(np.maximum(diagonal_and_scale(sa, sb, pq=pq)[0], 0.0))
    qcd = np.sqrt(np.maximum(diagonal_and_scale(sc, sd, pq=pq)[0], 0.0))
    bound = np.outer(qab, qcd).reshape(blk.shape)
    ratio = float(np.max(np.abs(blk) / bound))
    print(f"\n{name} {q}: max |(ab|cd)| / (Q_ab Q_cd) = {ratio:.4f}")
    assert np.all(np.abs(blk) <= bound * (1.0 + TOL)), ratio


# ---------------------------------------------------------------------------
# The staging
# ---------------------------------------------------------------------------


MIXED = Basis((
    Shell(2, (0.0, 0.0, 0.0), (1.1, 0.3), (0.5, 0.6)),
    Shell(1, (0.0, 0.0, 0.0), (1.9, 0.45), (0.38, 0.71)),
    Shell(0, (0.3, -0.7, 1.1), (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(1, (-1.2, 0.4, 0.25), (0.9,), (1.0,)),
))


def test_every_shell_pair_is_staged_once_with_la_at_most_lb():
    sh, off = MIXED.shells, MIXED.offsets
    seen = []
    for c in schwarz_classes(MIXED):
        assert c.la <= c.lb
        assert c.rec0[0] == 0 and c.rec1[-1] == c.n_rec
        for k, (a, b) in enumerate(c.pairs):
            assert (sh[a].l, sh[b].l) == (c.la, c.lb)
            assert (c.off_a[k], c.off_b[k]) == (off[a], off[b])
            if k:
                assert c.rec0[k] == c.rec1[k - 1]  # contiguous
            recs = epair(sh[a], sh[b]).recs
            assert c.rec1[k] - c.rec0[k] == len(recs)
            for r, (P, zeta, s) in zip(range(c.rec0[k], c.rec1[k]), recs,
                                       strict=True):
                assert (c.zeta_ab[r], c.P[r], c.s_ab[r]) == (zeta, tuple(P), s)
                assert all(math.isclose(c.XPA[r][d], P[d] - sh[a].centre[d])
                           for d in range(3))
            seen.append(tuple(sorted((a, b))))
    n = len(sh)
    assert sorted(seen) == [(i, j) for i in range(n) for j in range(i, n)]
    # the d shell precedes the s and p shells in basis order: swapped
    assert any(c.la < c.lb and c.pairs[0][0] > c.pairs[0][1]
               for c in schwarz_classes(MIXED))


@pytest.mark.oracle
@pytest.mark.parametrize("name", FIXTURES)
def test_the_schwarz_fixture_is_what_pyscf_computes_today(name):
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate

    msg = generate.drift(name)
    assert msg is None, msg
