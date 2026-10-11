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
                   the classes with la+lb+lc+ld <= 8 (46 of 55)
    hf_ccpvtz_eri_l9 .. _l12
                   the same molecule and basis, one file per L = 9..12
                   (#121): (ff|fs) (ff|dp) (fd|fp) (fd|dd) | (ff|fp) (ff|dd)
                   (fd|fd) | (ff|fd) | (ff|ff). A class's block size is fixed
                   by its l's, and all 55 in one file are ~1.2 MB, over the
                   repo's 512 KB large-file hook; split by L, every file is
                   under it and the five together hold all 55 classes of
                   HF/cc-pVTZ, each exactly once -- no reach to record.

THE TOLERANCE, stated and not tuned: element by element,

    |ours - libcint| <= 1e-12 * scale,
    scale = sum_{prims} sum_pq |E^{ab}_p| |E^{cd}_q| |[p|q]|

(`ao.eri.eri_block_and_scale`, the dressed E): the size of what was added,
so the bound is about rounding in either program, not about how much a sum
cancelled. 1e-12 is ~4500 ulps of the scale. Measured worst on this box
(2026-10-11, pyscf 2.14.0 / libcint 6.1.3): 1.3e-13 of scale, (fs|ps) on
HF/cc-pVTZ; every class is printed with `-s`, all 55 tracked, (ff|ff)
included.
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
HF_SPLIT = ("hf_ccpvtz_eri", "hf_ccpvtz_eri_l9", "hf_ccpvtz_eri_l10",
            "hf_ccpvtz_eri_l11", "hf_ccpvtz_eri_l12")
CASES = ("h2o_631g_eri", *HF_SPLIT)
HOOK_BYTES = 512 * 1024  # .pre-commit-config.yaml: check-added-large-files 512

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
        case = CASE_OF[name]
        want = eri_quartets(recs, case.max_eri_L, case.min_eri_L)
        assert sorted(manifest.eri_blocks(fx)) == sorted(want)
    # hf_ccpvtz_eri is capped at L <= 8 by the 512 KB hook: 46 of 55.
    assert len(manifest.eri_blocks(_FX["hf_ccpvtz_eri"])) == 46


def test_the_split_covers_every_class_of_hf_ccpvtz_once():
    """#121: the five HF/cc-pVTZ files are one basis (same converted shells)
    and together hold every canonical class -- all 55 -- in exactly one file.
    So `jengine/prefactor-factorisation`'s witness has no class boundary."""
    shells = _FX["hf_ccpvtz_eri"].converted.shells
    recs = [{"atom": s.atom, "l": s.l, "exponents": s.exponents}
            for s in shells]
    seen: list[tuple[int, int, int, int]] = []
    for name in HF_SPLIT:
        assert _FX[name].converted.shells == shells, name
        seen += manifest.eri_blocks(_FX[name])
    assert len(seen) == len(set(seen)) == 55
    assert sorted(seen) == sorted(eri_quartets(recs))
    by_L = {}
    for name in HF_SPLIT:
        by_L[name] = sorted({sum(shells[s].l for s in q)
                             for q in manifest.eri_blocks(_FX[name])})
    assert by_L == {"hf_ccpvtz_eri": list(range(9)),
                    "hf_ccpvtz_eri_l9": [9], "hf_ccpvtz_eri_l10": [10],
                    "hf_ccpvtz_eri_l11": [11], "hf_ccpvtz_eri_l12": [12]}


@pytest.mark.parametrize("name", CASES)
def test_eri_fixture_is_under_the_large_file_hook(name):
    """The split dodges the cap; it must not quietly creep back over it."""
    assert (manifest.FIXTURES / f"{name}.json").stat().st_size < HOOK_BYTES


def test_the_fast_tier_holds_the_sp_classes_and_both_cases():
    fast = [p for p in PARAMS if not p.marks]
    # The L = 9..12 files are f-only (slow): s/p lives in the two base cases.
    assert {p.values[0] for p in fast} == {"h2o_631g_eri", "hf_ccpvtz_eri"}
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


def _worst(name, q):
    """max err/scale of our block against libcint's, as the test above."""
    ref = manifest.eri_blocks(_FX[name])[q]
    ours, scale = _ours(name, q, scale=True)
    err = np.abs(ours - ref)
    return float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                  where=scale > 0)))


@pytest.mark.claim("jengine/m-weight-minus-two-theta-squared")
def test_the_m_weight_is_minus_two_theta_squared(monkeypatch):
    """[JEngine, v2, eq. 13] prints (-2 theta)^m; `reference.pq` forms
    (-2 theta^2)^m, and the libcint comparison above is what holds it -- shown
    here by perturbing the BASE alone, every other factor of the fundamental
    untouched (#120). Over the fast-tier classes: the true weight passes; the
    printed -2 theta, and -2 theta^2 scaled by 1 + 1e-9, fail every class
    above (ss|ss); (ss|ss) reaches only m = 0 and is bit-for-bit unchanged,
    so the perturbation reaches the weight and nothing else."""
    import felizjueves.reference.pq as pqmod

    fast = [p.values for p in PARAMS if not p.marks]
    true = {(n, q): _worst(n, q) for n, q in fast}
    assert all(w <= TOL for w in true.values()), true

    exact = pqmod.fundamental
    bases = {
        "printed (-2 theta)^m": lambda t: -2.0 * t,
        "(-2 theta^2)^m * (1 + 1e-9)^m": lambda t: -2.0 * t * t * (1 + 1e-9),
    }
    for label, base in bases.items():
        def wrong(m, pqv, theta, base=base):
            return exact(m, pqv, theta) * (base(theta) / (-2.0 * theta * theta)) ** m

        monkeypatch.setattr(pqmod, "fundamental", wrong)
        for n, q in fast:
            w = _worst(n, q)
            L = sum(_SHELLS[n][s].l for s in q)
            print(f"\n{label}: {n} {q} L={L}: max err/scale = {w:.2e}")
            if L == 0:
                assert w == true[(n, q)], (label, n, q)
            else:
                assert w > TOL, (label, n, q, w)
        monkeypatch.undo()
    assert pqmod.fundamental is exact


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
