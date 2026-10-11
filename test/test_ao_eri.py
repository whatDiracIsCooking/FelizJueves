"""`ao.eri`: `(ab|cd)` from the Hermite path, against libcint's `int2e`, ONE
ANGULAR-MOMENTUM CLASS AT A TIME.

Why per class, and per block: a wrong `common_fac`, a double normalisation or
a misplaced pair factor is a clean constant per class (#85 found exactly
that -- `ao.pairs` normalised the primitives a second time, a factor
N(la,a)N(lb,b)N(lc,c)N(ld,d) per primitive quartet), so a block comparison
names the culprit. A J-level mismatch would be one number with no address.

THE WITNESS. For each `(molecule, basis)` case, `test/oracle/generate.py`
stores libcint's Cartesian `int2e` for one representative shell quartet of
every canonical class (`eri_quartets`: most atoms, most distinct shells,
most primitives), the whole block, under the converted basis
`oracle.basis` builds -- whose conventions #84 pinned by the overlap. The
comparison is against that tracked fixture, so it is fast-tier for s/p and
`slow` for any class containing d or above; PySCF only regenerates
(`oracle`-marked drift test, bottom).

    h2o_631g_eri   H2O / 6-31G: contraction depth (O 1s, six primitives) at
                   low l -- what exercises the dressing
    hf_ccpvtz_eri  HF / cc-pVTZ: s to f, a general contraction, H off-axis;
                   the classes with la+lb+lc+ld <= 8 (46 of 55). A class's
                   block size is fixed by its l's, and all 55 are ~1.2 MB,
                   over the repo's 512 KB large-file hook; the nine left out
                   are L = 9..12: (ff|fs) (ff|dp) (fd|fp) (fd|dd)
                   (ff|fp) (ff|dd) (fd|fd) (ff|fd) (ff|ff).

THE TOLERANCE, stated and not tuned: element by element,

    |ours - libcint| <= 1e-12 * scale,
    scale = sum_{prims} sum_pq |E^{ab}_p| |E^{cd}_q| |[p|q]|

(`ao.eri.eri_block_and_scale`, the dressed E): the size of what was added,
so the bound is about rounding in either program, not about how much a sum
cancelled. 1e-12 is ~4500 ulps of the scale. Measured worst on this box
(2026-10-11, pyscf 2.14.0 / libcint 6.1.3): 1.3e-13 of scale, (fs|ps) on
HF/cc-pVTZ; every class is printed with `-s`. (Before the cap the same
code met all 55 classes, (ff|ff) included, at that same worst 1.3e-13.)
A failure here is a defect to find, not a tolerance to widen -- that
needs a `decimal` recomputation first, the way test_kernel_device.py
justified its 5e-13.

THE SYMMETRY. Our own block for all eight orderings of the quartet must be
the same tensor, transposed: `(ab|cd) = (ba|cd) = (ab|dc) = (cd|ab) = ...`.
Internal and free, and it catches an index transposition the oracle
comparison could absorb (a representative quartet can be symmetric in a way
that hides one).
"""

from __future__ import annotations

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from oracle.generate import CASES as CASE_OF
from oracle.generate import eri_quartets

from felizjueves.ao.eri import eri_block, eri_block_and_scale
from felizjueves.reference.pq import pq

TOL = 1e-12
CASES = ("h2o_631g_eri", "hf_ccpvtz_eri")

# The eight orderings of (i j | k l), as positions into the quartet.
PERMS = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
         (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))


def _fixture(name: str) -> manifest.Fixture:
    return manifest.load(manifest.FIXTURES / f"{name}.json")


_FX = {name: _fixture(name) for name in CASES}
_SHELLS = {name: ob.from_fixture(fx).shells for name, fx in _FX.items()}


def _params():
    out = []
    for name, fx in _FX.items():
        for q in sorted(manifest.eri_blocks(fx)):
            ls = tuple(_SHELLS[name][s].l for s in q)
            cls = "({}{}|{}{})".format(*("spdfghi"[x] for x in ls))
            marks = [pytest.mark.slow] if max(ls) >= 2 else []
            out.append(pytest.param(name, q, id=f"{name}-{cls}-{q}",
                                    marks=marks))
    return out


PARAMS = _params()


def _ours(name, q, *, scale=False):
    sh = _SHELLS[name]
    bra, ket = (sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]])
    shape = tuple(sh[s].n_cart for s in q)
    if not scale:
        return np.array(eri_block(bra, ket, pq=pq)).reshape(shape)
    blk, sc = eri_block_and_scale(bra, ket, pq=pq)
    return np.array(blk).reshape(shape), np.array(sc).reshape(shape)


def test_every_canonical_class_has_its_representative():
    """The fixture's quartets are exactly `eri_quartets` of its basis: a
    class cannot drop out of the comparison silently."""
    for name, fx in _FX.items():
        recs = [{"atom": s.atom, "l": s.l, "exponents": s.exponents}
                for s in fx.converted.shells]
        want = eri_quartets(recs, CASE_OF[name].max_eri_L)
        assert sorted(manifest.eri_blocks(fx)) == sorted(want)
    # The cap is the 512 KB fixture hook's, and it still reaches f: every
    # class with L <= 8 of HF/cc-pVTZ, 46 of 55.
    assert len(manifest.eri_blocks(_FX["hf_ccpvtz_eri"])) == 46


def test_the_fast_tier_holds_the_sp_classes_and_both_cases():
    fast = [p for p in PARAMS if not p.marks]
    assert {p.values[0] for p in fast} == set(CASES)
    assert len(fast) == 6 + 6  # (ss|ss) .. (pp|pp), in each case


@pytest.mark.claim("jengine/prefactor-factorisation")
@pytest.mark.parametrize("name,q", PARAMS)
def test_block_matches_libcint(name, q):
    """Also the witness of `jengine/prefactor-factorisation` (#90): a pair
    factor missing, doubled or at the wrong power is a per-class constant
    here. It sees the PRODUCT lambda only; the placement is closed by the
    in-repo absolute pins of each half (the record names them)."""
    ref = manifest.eri_blocks(_FX[name])[q]
    ours, scale = _ours(name, q, scale=True)
    err = np.abs(ours - ref)
    assert np.all(scale > 0) or np.all(err[scale == 0] == 0.0)
    worst = float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                   where=scale > 0)))
    print(f"\n{name} {q}: max|ours - libcint| = {err.max():.2e}, "
          f"max err/scale = {worst:.2e}")
    bad = err > TOL * scale
    r = ref.ravel()
    ratio = float(ours.ravel() @ r / (r @ r))  # a per-class constant shows here
    assert not bad.any(), (
        f"{name} {q}: {int(bad.sum())} of {err.size} elements exceed "
        f"1e-12 * scale; worst err/scale {worst:.2e}; block ratio "
        f"ours/libcint = {ratio:.6g}")


@pytest.mark.parametrize("name,q", PARAMS)
def test_block_has_eightfold_symmetry(name, q):
    x, scale = _ours(name, q, scale=True)
    for perm in PERMS[1:]:
        y = _ours(name, tuple(q[p] for p in perm))
        sub = "".join("abcd"[p] for p in perm)
        z = np.einsum(f"{sub}->abcd", y)  # back to (i j | k l) order
        assert np.all(np.abs(z - x) <= TOL * scale), (name, q, perm)


# --- the drift check: needs PySCF ------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", CASES)
def test_eri_fixture_is_what_pyscf_computes_today(name):
    """Regenerate and compare every int2e block as doubles. A failure is news
    about PySCF/libcint (versions and date in the message), not proof the
    fixture was wrong."""
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate

    msg = generate.drift(name)
    assert msg is None, msg
    fresh = generate.build(name)
    tracked = manifest.eri_blocks(_FX[name])
    names = [n for n in fresh["arrays"] if n.startswith(manifest.ERI_PREFIX)]
    assert len(names) == len(tracked)
    for n in names:
        a = fresh["arrays"][n]
        q = tuple(int(x) for x in n[len(manifest.ERI_PREFIX):].split(","))
        got = np.array(a["values"]).reshape(a["shape"])
        assert np.array_equal(got, tracked[q]), n
