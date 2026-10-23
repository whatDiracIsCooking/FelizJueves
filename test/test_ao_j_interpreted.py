"""`ao.drive`: `J_mn` with `J_p` from the INTERPRETED GENERATED KERNEL (#87).

`ao.j.coulomb_matrix(basis, D, pq=pq, jp=KernelJP(boys=boys))` runs the
`[p|q]` kernel `kernel.build_pq_kernels` builds for each REAL class
`(la, lb | lc, ld)` the basis reaches, in `interp.py`, on the real primitive
pair records of `ao.pairs`; the ket pre-contraction, the bra transform and the
scatter are #86's, already witnessed. So this is the first externally
witnessed number produced by the generator rather than by `reference/`.

THE TWO WITNESSES, both at TAU = 0 (the seam is unscreened; `ao.j` refuses
`jp` with `tau > 0`, so screening cannot hide an error here, and the
sorted-ket precondition -- which only makes the `break` correct -- is moot):

  reference  `J` from `reference.pq` through `ao.j`'s own `pq` path:
             |J_kernel - J_pq| <= 1e-13 max|J_pq|, element by element
             (the issue's 1e-13 relative, relative to the matrix's largest
             element, test_ao_j.py's convention).
  libcint    the tracked `get_jk` fixture, at test_ao_j.py's DERIVED
             tolerance |J_kernel - J_pyscf| <= 1e-12 scale_J
             (`ao.j.coulomb_matrix_and_scale`; see test_ao_j.py for the
             derivation). Not re-derived and not loosened.

THE BATCH AXIS. For each `n_batches` in {1, 3, 6, 9} one full `J` is built
with every class at that batch count when the plan admits it (#64), and at
one batch -- recorded in `KernelJP.refused` -- when it does not. Each test
asserts exactly which `(class, n_batches)` cells ran and which were refused,
and `test_the_cells_are_what_the_plan_admits` that the union over the four
runs is every admitted cell of every class the case reaches.

THE CASES, by cost (interpreter, this host):

  fast   H2/STO-3G full J (one class, (ss|ss), which admits only n = 1);
         H2O/STO-3G at the `J_p` level on a sampled subset of its real
         records, every s/p class and every admitted n_batches.
  slow   H2O/STO-3G, H2O/6-31G and H2O/cc-pVDZ full J, every admitted cell
         (measured ~2-4 s, ~8-14 s and ~45-50 s per J respectively).
  slow   H2O/cc-pVDZ at the `J_p` level: for EVERY class its basis reaches
         (all 81 `(la, lb | lc, ld)` with l <= 2) and every admitted
         n_batches, the kernel-backed `jp` against `ao.j.jp_from_pq(pq)` on a
         deterministic sample of that class's REAL records -- SAMPLE_BRA bra
         records drawn from all its bra shell pairs and SAMPLE_KET ket
         records of the class, chosen by a seeded generator. This isolates
         each class: in a full `J` a wrong class is one term among 81, while
         here it is the whole comparison, at the issue's 1e-13 relative
         (relative to max|J_p| of the sample). Sampled because the question
         is per class and a few real records answer it cheaply.

Nothing compiles and nothing launches on a card. Boys is injected
(`reference.boys.boys`); `ao.drive` itself does not import `reference/`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_layering import _graph, _under

from felizjueves.ao.drive import (
    BATCH_COUNTS,
    KernelJP,
    admitted_batches,
    class_spec,
)
from felizjueves.ao.j import (
    coulomb_matrix,
    coulomb_matrix_and_scale,
    jp_from_pq,
)
from felizjueves.kernel import split
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

INTERP_TOL = 1e-13  # vs reference/pq, relative to the largest element
JK_TOL = 1e-12  # vs libcint, times scale_J: test_ao_j.py's, inherited
SAMPLE_BRA = 3
SAMPLE_KET = 12


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx)


@cache
def _pq_j(name: str) -> tuple[np.ndarray, np.ndarray]:
    fx, basis = _case(name)
    J, scale = coulomb_matrix_and_scale(basis, fx.density, pq=pq)
    return np.array(J), np.array(scale)


def _shell_ams(basis) -> set[tuple[int, int]]:
    sh = basis.shells
    return {(sh[a].l, sh[b].l) for a in range(len(sh)) for b in range(a + 1)}


def _classes(basis) -> set[tuple[int, int, int, int]]:
    """Every (la, lb | lc, ld) the unique sweep launches for this basis."""
    ams = _shell_ams(basis)
    return {(*b, *k) for b in ams for k in ams}


def _expected(basis, n: int):
    """(ran, refused) for one full J at batch count n."""
    ran, refused = set(), set()
    for am in _classes(basis):
        if n in admitted_batches(am):
            ran.add((am, n))
        else:
            refused.add((am, n))
            ran.add((am, 1))
    return ran, refused


# ---------------------------------------------------------------------------
# Full J: reference/pq and libcint
# ---------------------------------------------------------------------------

FULL_SLOW = ("h2o_sto3g_j", "h2o_631g_j", "h2o_ccpvdz_j")
FULL = [pytest.param("h2_sto3g_j", n, id=f"h2_sto3g_j-b{n}")
        for n in BATCH_COUNTS]
FULL += [pytest.param(name, n, id=f"{name}-b{n}", marks=pytest.mark.slow)
         for name in FULL_SLOW for n in BATCH_COUNTS]


def _check_full(name: str, n: int) -> None:
    fx, basis = _case(name)
    k = KernelJP(boys=boys, n_batches=n, strict=False)
    J = np.array(coulomb_matrix(basis, fx.density, pq=pq, jp=k))
    ran, refused = _expected(basis, n)
    assert k.ran == ran and k.refused == refused

    R, scale = _pq_j(name)
    ref = fx.arrays[manifest.J_ARRAY]
    rel = float(np.max(np.abs(J - R)) / np.max(np.abs(R)))
    jk = float(np.max(np.abs(J - ref) / scale))
    print(f"\n{name} n_batches={n}: {len(ran)} cells ran, {len(refused)} "
          f"refused; max|J - J_pq|/max|J_pq| = {rel:.2e}; "
          f"max|J - get_jk|/scale = {jk:.2e}")
    assert np.all(np.abs(J - R) <= INTERP_TOL * np.max(np.abs(R))), (
        f"{name} b{n}: interpreted J vs reference/pq J, worst {rel:.2e}")
    assert np.all(np.abs(J - ref) <= JK_TOL * scale), (
        f"{name} b{n}: interpreted J vs libcint, worst err/scale {jk:.2e}")


@pytest.mark.claim("jengine/theta-undefined")
@pytest.mark.parametrize(("name", "n"), FULL)
def test_interpreted_j_matches_reference_and_libcint(name: str, n: int):
    """Also the OUTSIDE witness of `jengine/theta-undefined` (#90): the
    kernel's prelude forms theta, and libcint shares no ancestry with it.
    theta perturbed consistently at every site that forms it, so the
    reference/pq comparison still agrees to 0, fails the libcint comparison
    (8.85e-10 scale_J on H2 at a relative theta error of 1e-9). Why this
    test and not the on-device one: it runs without a card, so the fast tier
    exercises it (H2, theta through F_0) and the slow tier the H2O cases
    (m > 0); test_ao_j_ondevice.py corroborates on both cards but skips,
    green and empty, wherever there is none."""
    _check_full(name, n)


@pytest.mark.claim("jengine/alg1-line10-factor-four")
def test_j_p_is_written_unscaled_and_a_factor_four_is_caught():
    """`[JEngine, v2, Alg. 1]` line 10 writes `J[p] = 4 Jshm[p]`; this repo's
    kernel writes `J_p` unscaled and puts the only pair weight host-side
    (`ao/j.py`'s unique sweep), and that `J_mn` matches libcint's `get_jk`
    at test_ao_j.py's derived tolerance. The control is the claim made false:
    the same kernel's `J_p` multiplied by 4 on its way out -- Alg. 1's write,
    with the host weights unchanged -- misses libcint by orders of magnitude
    more than the tolerance, so the comparison sees the factor. H2/STO-3G,
    one batch: fast tier, no card."""
    name = "h2_sto3g_j"
    fx, basis = _case(name)
    ref = fx.arrays[manifest.J_ARRAY]
    _, scale = _pq_j(name)

    def worst(jp) -> float:
        J = np.array(coulomb_matrix(basis, fx.density, pq=pq, jp=jp))
        return float(np.max(np.abs(J - ref) / scale))

    k = KernelJP(boys=boys, n_batches=1, strict=False)
    assert worst(k) <= JK_TOL

    def written_times_four(bra_am, ket_am, bras, kets):
        return [[4.0 * v for v in row] for row in k(bra_am, ket_am, bras, kets)]

    assert worst(written_times_four) > 1e6 * JK_TOL


def test_the_cells_are_what_the_plan_admits():
    """Over the four full-J runs, every admitted (class, n_batches) of every
    full-J case ran, and nothing the plan refuses was built."""
    for name in ("h2_sto3g_j", *FULL_SLOW):
        _, basis = _case(name)
        ran: set = set()
        for n in BATCH_COUNTS:
            ran |= _expected(basis, n)[0]
        want = {(am, n) for am in _classes(basis)
                for n in admitted_batches(am)}
        assert ran == want
        assert all(n in admitted_batches(am) for am, n in ran)
    # The plan does refuse some: (ss|ss) is L = 0 and cannot split.
    assert admitted_batches((0, 0, 0, 0)) == (1,)


# ---------------------------------------------------------------------------
# J_p per class, on sampled real records
# ---------------------------------------------------------------------------


@cache
def _records(name: str):
    """The real (bra_am, ket_am) -> (bras, kets) launches of the unique sweep,
    captured through the seam (it returns zeros, so this is cheap): every bra
    record of every bra shell pair of `bra_am`, and the class's ket records."""
    fx, basis = _case(name)
    got: dict = {}

    def capture(bra_am, ket_am, bras, kets):
        b, _ = got.setdefault((bra_am, ket_am), ([], list(kets)))
        b.extend(bras)  # every bra shell pair of this am; kets are per class
        return [[0.0] * _n_p(bra_am) for _ in bras]

    coulomb_matrix(basis, fx.density, pq=pq, jp=capture)
    return got


def _n_p(bra_am) -> int:
    L = sum(bra_am)
    return (L + 1) * (L + 2) * (L + 3) // 6


def _sample(name: str, am):
    bras, kets = _records(name)[(am[:2], am[2:])]
    rng = np.random.default_rng(sum((i + 1) * x for i, x in enumerate(am)))
    bi = sorted(rng.choice(len(bras), min(SAMPLE_BRA, len(bras)), replace=False))
    ki = sorted(rng.choice(len(kets), min(SAMPLE_KET, len(kets)), replace=False))
    return [bras[i] for i in bi], [kets[i] for i in ki]


def _check_class(name: str, am) -> None:
    bras, kets = _sample(name, am)
    want = np.array(jp_from_pq(pq)(am[:2], am[2:], bras, kets))
    assert np.max(np.abs(want)) > 0, f"{am}: the sample is all zeros"
    cells = admitted_batches(am)
    assert cells and cells[0] == 1
    for n in cells:
        k = KernelJP(boys=boys, n_batches=n)
        got = k(am[:2], am[2:], bras, kets)
        assert k.ran == {(am, n)}
        rel = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
        assert rel <= INTERP_TOL, f"{name} {am} n_batches={n}: {rel:.2e}"


def _class_params(name: str, marks=()):
    _, basis = _case(name)
    return [pytest.param(name, am, id=f"{name}-{''.join(map(str, am))}",
                         marks=marks) for am in sorted(_classes(basis))]


@pytest.mark.parametrize(("name", "am"), _class_params("h2o_sto3g_j"))
def test_kernel_jp_per_class_sto3g(name, am):
    _check_class(name, am)


@pytest.mark.parametrize(("name", "am"),
                         _class_params("h2o_ccpvdz_j", (pytest.mark.slow,)))
def test_kernel_jp_per_class_ccpvdz(name, am):
    _check_class(name, am)


def test_ccpvdz_reaches_every_class_up_to_d():
    _, basis = _case("h2o_ccpvdz_j")
    classes = _classes(basis)
    assert len(classes) == 81
    assert max(sum(split(class_spec(am))) for am in classes) == 8


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_bra_and_ket_orders_are_different_kernels():
    """(p,p|s,s) and (s,s|p,p): different n_p / n_q, so different kernels."""
    a, b = class_spec((1, 1, 0, 0)), class_spec((0, 0, 1, 1))
    assert split(a) == (2, 0) and split(b) == (0, 2)
    bras, kets = _sample("h2o_sto3g_j", (1, 1, 0, 0))
    assert KernelJP(boys=boys)((1, 1), (0, 0), bras, kets).shape == (
        len(bras), 10)


def test_a_refused_batch_count_raises_when_strict():
    bras, kets = _sample("h2o_sto3g_j", (0, 0, 0, 0))
    with pytest.raises(ValueError, match="refuses"):
        KernelJP(boys=boys, n_batches=3)((0, 0), (0, 0), bras, kets)


def test_the_seam_is_unscreened():
    fx, basis = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="unscreened"):
        coulomb_matrix(basis, fx.density, pq=pq, tau=1e-10,
                       jp=KernelJP(boys=boys))


def test_the_adapter_does_not_reach_the_reference():
    """Boys is injected; nothing ao.drive imports, transitively, is
    reference/ (test_layering.py holds the rule; this names the claim)."""
    graph = _graph()
    seen, todo = set(), ["ao.drive"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "ir.interp" in seen and "kernel" in seen
    assert not [e for e in seen if _under(e, "reference")]
