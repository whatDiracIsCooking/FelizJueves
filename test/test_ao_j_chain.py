"""`eri.j_coulomb.drive.coulomb_matrix_chain`: `J_mn` with every step generated (#155).

The whole J chain composed and INTERPRETED -- `Target.DQ` (#153) for the ket
pre-contraction and `Dmax`, the `[p|q]` kernels for `J_p`, `Target.JAB`
(#154) for eq. 20 and the scatter -- over milestone 6's class-grouped
staging (`eri.j_coulomb.staged`, #143). No compiler, no card, no PySCF.

THE WITNESSES, at tolerances inherited and not loosened here:

  libcint     the tracked `get_jk` fixtures at `tau = 0`:
              `|J - J_pyscf| <= 1e-12 scale_J` (test_ao_j.py's `JK_TOL`,
              with `scale_J` from `eri.j_coulomb.j.coulomb_matrix_and_scale`);
  host        `eri.j_coulomb.j`'s pure host path, `coulomb_matrix` (unique sweep) and
              `coulomb_matrix_reference` (every ordered pair, weight 1):
              `|J - J_host| <= 1e-13 max|J_host|` (test_launch_staged.py's
              `HOST_TOL`);
  staged      `eri.j_coulomb.staged.coulomb_matrix_staged` over `launch.launches`' plan,
              whose `D_q`, `Dmax` and eq. 20 are the HOST's: the chain's plan
              (keyed on the DQ kernel's read-back `Dmax`) is `==` to
              `launches`', and its `J` is `==` to the staged `J`. Bit for bit,
              because the DQ and JAB kernels are each `==` to their host step
              interpreted (#153, #154) and the `[p|q]` launches are the same.

THE CASES are every tracked `J` fixture, by #85's rule (a `d` function makes
a case `slow`): H2/STO-3G and H2O/STO-3G fast, H2O/6-31G and H2O/cc-pVDZ
`slow` (6-31G has no `d`; it is slow by cost, ~7 s interpreted, as the issue
places it). The reach is `d`: no `f` `J` fixture exists.

"EVERY ARITHMETIC STEP" IS ASSERTED, not stated: the chain's report says
which generated kernel ran which step for which classes, and the test holds
it equal to the classes the basis reaches, records counted. A host step
replaced by a tripwire (the host pre-contraction, the host transform, the
host launch plan) is never reached. A gap -- a class a kernel did not
cover -- raises.
"""

from __future__ import annotations

from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.pairs import primitive_pairs
from felizjueves.eri.j_coulomb import drive
from felizjueves.eri.j_coulomb import j as aoj
from felizjueves.eri.j_coulomb import staged as aostaged
from felizjueves.eri.j_coulomb.drive import coulomb_matrix_chain
from felizjueves.eri.j_coulomb.launch import launches
from felizjueves.eri.j_coulomb.staged import coulomb_matrix_staged
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

BOUND = partial(aoj.pair_bound_pq, pq=pq)  # the host sweep's bound, exactly
HOST_TOL = 1e-13  # vs the host path, relative to its largest element
JK_TOL = 1e-12  # vs libcint, times scale_J: test_ao_j.py's, inherited

FAST = ("h2_sto3g_j", "h2o_sto3g_j")
SLOW = ("h2o_631g_j", "h2o_ccpvdz_j")
# #486's H10 chain (50 AOs, 30 shells) is tracked for its decaying RHF
# density, a consumer's input; it is NOT run through the chain, by cost:
# 30 shells to H2O/cc-pVDZ's 12, already the slowest case, so the
# interpreted chain grows ~(30/12)^4 ~ 40x. Its vj is libcint's,
# cross-checked against einsum over the full int2e when written
# (oracle.generate).
# #570's CO/cc-pVTZ (20 shells, f on both atoms, [p|q] up to L = 12) is
# not run interpreted either, by cost: its chain on the card is
# test_j_chain_ondevice.py's, and the host J against its vj test_ao_j.py's.
NOT_RUN = ("h10_chain_ccpvdz_j", "co_ccpvtz_j")
CASES = [pytest.param(n, id=n) for n in FAST] + [
    pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW]


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx), np.array(fx.density)


@cache
def _chain(name: str, n: int = 1, strict: bool = True):
    _, basis, D = _fixture(name)
    return coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys,
                                n_batches=n, strict=strict)


def _rel(J, H) -> float:
    J, H = np.asarray(J), np.asarray(H)
    return float(np.max(np.abs(J - H)) / np.max(np.abs(H)))


def _pair_classes(basis) -> set[tuple[int, int]]:
    sh = basis.shells
    return {(sh[a].l, sh[b].l) for a in range(len(sh)) for b in range(a + 1)}


def _n_records(basis) -> int:
    sh = basis.shells
    return sum(len(primitive_pairs(sh[a], sh[b]))
               for a in range(len(sh)) for b in range(a + 1))


def test_the_cases_are_every_tracked_j_fixture():
    tracked = {p.stem for p in manifest.FIXTURES.glob("*.json")
               if manifest.J_ARRAY in manifest.load(p).arrays}
    assert tracked == set(FAST) | set(SLOW) | set(NOT_RUN)


# --- the answer, and the report ---------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_j_matches_libcint_and_the_host_path(name):
    fx, basis, D = _fixture(name)
    c = _chain(name)
    J, scale = aoj.coulomb_matrix_and_scale(basis, D, pq=pq)
    ref = fx.arrays[manifest.J_ARRAY]
    jk = float(np.max(np.abs(np.array(c.J) - ref) / np.array(scale)))
    R = aoj.coulomb_matrix_reference(basis, D, pq=pq)
    print(f"\n{name}: {c.steps()}\n  max|J - get_jk|/scale_J = {jk:.2e}; "
          f"vs host {_rel(c.J, J):.2e}; vs reference sweep {_rel(c.J, R):.2e}")
    assert jk <= JK_TOL, jk
    assert _rel(c.J, J) <= HOST_TOL
    assert _rel(c.J, R) <= HOST_TOL


@pytest.mark.parametrize("name", CASES)
def test_every_step_is_a_generated_kernel_for_every_class(name):
    """The acceptance, checkable: each step's generated kernel ran for
    exactly the classes the basis reaches, over every record."""
    _, basis, _ = _fixture(name)
    c = _chain(name)
    pair_classes = _pair_classes(basis)
    lps = {la + lb for la, lb in pair_classes}
    assert set(c.dq) == pair_classes and len(c.dq) == len(pair_classes)
    assert set(c.jab) == pair_classes and len(c.jab) == len(pair_classes)
    assert {(lp, lq) for lp, lq, _ in c.pq} == {
        (lp, lq) for lp in lps for lq in lps}
    assert len(c.pq) == len(lps) ** 2
    assert c.n_dq_records == c.n_jab_records == _n_records(basis)
    assert c.steps() == {"DQ": c.dq, "[p|q]": c.pq, "JAB": c.jab}


def test_the_report_names_the_classes():
    """Spelled out once, for H2O/STO-3G: s and p shells, so four ordered
    pair classes (a >= b by shell index reaches both (0, 1) and (1, 0)) and
    every (lp, lq) in 0..2."""
    c = _chain("h2o_sto3g_j")
    assert c.dq == c.jab == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert c.pq == tuple((lp, lq, 1) for lp in range(3) for lq in range(3))
    assert dict(c.staged.builds) == {"bra": 3, "J_p": 3, "ket": 3}


def test_the_chain_is_the_staged_host_path_bit_for_bit():
    """The kernel's read-back `Dmax` keys the same plan `launch.launches`
    builds from the host's, and the generated DQ and JAB steps give the
    staged host path's `J` exactly."""
    _, basis, D = _fixture("h2o_sto3g_j")
    c = _chain("h2o_sto3g_j")
    pl = launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)
    assert c.plan.classes == pl.classes
    assert c.J == coulomb_matrix_staged(pl, basis, D, boys=boys).J


def test_a_split_class_runs_through_the_chain():
    """`n_batches = 3` where the plan admits it, one batch (recorded) where
    it refuses: the split kernels' atomics into one `J_p` per bra class."""
    fx, basis, D = _fixture("h2o_sto3g_j")
    c = _chain("h2o_sto3g_j", 3, False)
    assert any(n == 3 for _, _, n in c.pq)
    assert {(lp, lq) for lp, lq, n in c.pq if n == 1} == {
        (lp, lq) for lp, lq, _ in c.plan.refused}
    _, scale = aoj.coulomb_matrix_and_scale(basis, D, pq=pq)
    jk = np.max(np.abs(np.array(c.J) - fx.arrays[manifest.J_ARRAY])
                / np.array(scale))
    assert jk <= JK_TOL


def test_the_screen_is_the_kernels_and_equals_the_host_screen():
    """At `tau > 0` the `[p|q]` kernel's `break` screens on the chain's
    keys; the host's sorted break at the same `tau` agrees, and so does the
    staged host path, exactly."""
    _, basis, D = _fixture("h2o_sto3g_j")
    pl = launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)
    prods = sorted(b.key * k.key for cl in pl for b in cl.bras
                   for k in cl.kets)
    tau = prods[len(prods) // 2]
    c = coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys, tau=tau)
    H = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=tau)
    assert H.n_skipped > 0
    assert _rel(c.J, H.J) <= HOST_TOL
    pl_tau = launches(basis, D, bound=BOUND, n_batches=1, tau=tau)
    assert c.J == coulomb_matrix_staged(pl_tau, basis, D, boys=boys).J


# --- no host step, and a gap raises -----------------------------------------


def _tripwire(name):
    def boom(*a, **k):
        raise AssertionError(f"the chain reached the host step {name}")
    return boom


def test_no_host_arithmetic_step_is_reached(monkeypatch):
    """The host pre-contraction, the host eq. 20 and scatter, the seams'
    host references and the host-keyed launch plan are tripwires: the chain
    reaches none of them and still gives the same `J`."""
    want = _chain("h2o_sto3g_j").J
    for mod, names in ((aoj, ("make_kets", "dq_from_host", "jab_from_host",
                              "_record_jp", "jp_from_pq")),
                       (aostaged, ("make_kets", "_ket_buffers", "_ket_vectors",
                                   "_transform", "_dglobal")),
                       (drive, ("KernelJP",))):
        for n in names:
            monkeypatch.setattr(mod, n, _tripwire(f"{mod.__name__}.{n}"))
    _, basis, D = _fixture("h2o_sto3g_j")
    got = coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys)
    assert got.J == want


def test_a_class_with_no_dq_kernel_raises(monkeypatch):
    real = drive._dq_kernel

    def missing(lc, ld):
        if (lc, ld) == (1, 1):
            raise NotImplementedError("no DQ kernel for (1, 1)")
        return real(lc, ld)

    monkeypatch.setattr(drive, "_dq_kernel", missing)
    _, basis, D = _fixture("h2o_sto3g_j")
    with pytest.raises(NotImplementedError, match=r"\(1, 1\)"):
        coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys)


def test_a_bra_class_the_jab_step_did_not_cover_raises(monkeypatch):
    """A JAB pass that silently leaves one bra class out (its blocks never
    transformed) is a gap: the chain raises rather than return that `J`."""
    real = drive._jab_launches

    def drops_one(basis, eps, blocks, jps, threads):
        sh = basis.shells
        kept = {lp: tuple(b for b in blks if (sh[b[0]].l, sh[b[1]].l)
                          != (0, 1)) for lp, blks in blocks.items()}
        return real(basis, eps, kept, jps, threads)

    monkeypatch.setattr(drive, "_jab_launches", drops_one)
    _, basis, D = _fixture("h2o_sto3g_j")
    with pytest.raises(ValueError, match="gap"):
        coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys)


def test_a_dmax_that_is_not_the_sorted_key_is_refused(monkeypatch):
    """The staged buffers check every `Iq * Dmax` against the plan key, so
    a DQ result that differs from what the plan sorted by cannot run."""
    real_kc = aostaged._kets_checked  # the plan is keyed; perturb the gather

    def perturbed(recs, n_q, kd):
        return real_kc(recs, n_q, aostaged.KetData(
            kd.Q, kd.eta, kd.D, np.asarray(kd.Dmax) * (1.0 + 2.0 ** -40)))

    monkeypatch.setattr(aostaged, "_kets_checked", perturbed)
    _, basis, D = _fixture("h2o_sto3g_j")
    with pytest.raises(ValueError, match="plan key"):
        coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys)


def test_a_refused_batch_count_raises_when_strict():
    _, basis, D = _fixture("h2o_sto3g_j")
    with pytest.raises(ValueError, match="refuses"):
        coulomb_matrix_chain(basis, D, bound=BOUND, boys=boys, n_batches=3)


def test_the_round_trip_is_named_in_the_adapter():
    doc = coulomb_matrix_chain.__doc__
    assert "THE ROUND-TRIP" in doc and "launch program" in doc
    assert "Dmax" in doc and "READS `Dmax`" in doc
