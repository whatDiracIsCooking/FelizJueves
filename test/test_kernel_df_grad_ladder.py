"""The DF gradient kernels through (ff|i) and (i|i) (#613), interpreted.

THE LADDER is FreeBreakfast's: `geri3` `la <= lb <= 3`, `lP <= 6` (70
classes) and `gmetric` `lP <= lQ <= 6` (28). test_kernel_geri3.py and
test_kernel_gmetric.py hold every class at `L <= 4`; this file the rest.

THE GATE. Both builders have a LOW-STACK form (`kernel_geri3` /
`kernel_gmetric`, THE LOW-STACK FORM) picked from an `L` floor set from
measured rows (data/measured/df_grad_lowstack.json, `devtools/measure.sh
df-grad-lowstack-compile`, its rule declared before any row): geri3 from
`L = 5` (the unrolled (sf|d), (pd|d), (pf|p), (dd|p) and (df|s) are over
14,000 B of `sm_86` stack), gmetric from `L = 10` ((g|i) and (h|h) are). No class at
`L <= 4` is gated, so every golden is unchanged.

THE WITNESSES. The host, `gradient.host_df` (another route on every
single-center side, itself held to finite differences): `|G - G_host| <=
16 eps` of the TOWER-TWIN split-centre scale -- test_kernel_g1e.py's
split-centre scale (every shell on a row of its own, folded back) of the
host run on `|Gam|` / `|W|` with `[t|tau]` replaced by its absolute-value
tower (`test_kernel_metric.pq_tower`, #606's yardstick for the d0 kernels
at high L): at L = 9 to 13 the tower's own cancellation is what both
routes round, and the signed scale is blind to it (the unrolled (h|h)
reads 17.5 eps of it, and the low-stack (i|i) 52 though it is libcint's
to 7.8e-16 of max) -- every class of
the ladder in its default form on a small basis that reaches it with a
swapped or diagonal pair, and every `L <= 4` class in the low-stack form
FORCED (the form is exact arithmetic of another association, so not
bitwise). And libcint: `int3c2e_ip1_cart` / `int3c2e_ip2_cart` and
`int2c2e_ip1_cart`, contracted per class into `G` on the aux-i sets'
system (`oracle.manifest.df_gradient`; Cartesian, as the kernels are), the
host against it for every class and the kernel at the top classes, within
`1e-12` of the witness's largest `max|G|`. Above `FAST_L_MAX` everything
is `slow` (an interpreted (ff|i) cell is ~40 s per record pair).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_eri3, run_metric
from oracle import basis as ob
from oracle import manifest
from test_kernel_metric import pq_tower
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host_df
from felizjueves.gradient import kernel_geri3 as kg
from felizjueves.gradient import kernel_gmetric as km
from felizjueves.gradient.host import shell_atoms
from felizjueves.ir import Decl, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

EPS = 2.0 ** -52
TOL = 16 * EPS  # of the split-centre scale
LIBCINT_TOL = 1e-12  # of the witness's largest class max|G|
ERI3_LADDER = [(a, b, p) for a in range(4) for b in range(a, 4)
               for p in range(7)]
METRIC_LADDER = [(p, q) for p in range(7) for q in range(p, 7)]

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
NUCLEI = ((8.0, A), (1.0, B), (1.0, C))


def _eid(c) -> str:
    return kg.geri3_name(c)


def _mid(c) -> str:
    return km.gmetric_name(c)


# --- the gate ------------------------------------------------------------------


def test_the_gate_is_an_l_floor_at_5_and_10():
    assert kg.LOWSTACK_MIN_L == 5 and km.LOWSTACK_MIN_L == 10
    assert ({c for c in ERI3_LADDER if kg.geri3_lowstack(c)}
            == {c for c in ERI3_LADDER if sum(c) >= 5})
    assert ({c for c in METRIC_LADDER if km.gmetric_lowstack(c)}
            == {(4, 6), (5, 5), (5, 6), (6, 6)})
    assert len([c for c in ERI3_LADDER if kg.geri3_lowstack(c)]) == 49


def test_no_class_at_or_below_l4_is_gated():
    """So the L <= 4 goldens (test_emit_grad_df.py) are the unrolled text."""
    assert not any(kg.geri3_lowstack(c) for c in ERI3_LADDER if sum(c) <= 4)
    assert not any(km.gmetric_lowstack(c) for c in METRIC_LADDER
                   if sum(c) <= 4)


def test_the_forms_share_name_parameters_and_launch():
    for cls in ((0, 1, 1), (1, 1, 2)):
        a = kg.build_geri3_kernel(cls, lowstack=True)
        b = kg.build_geri3_kernel(cls, lowstack=False)
        assert (a.name, a.params) == (b.name, b.params)
    for cls in ((1, 2), (2, 2)):
        a = km.build_gmetric_kernel(cls, lowstack=True)
        b = km.build_gmetric_kernel(cls, lowstack=False)
        assert (a.name, a.params) == (b.name, b.params)


def _array_bytes(kern) -> int:
    return sum(8 * int(np.prod([e.value for e in s.shape]))
               for s in walk(kern.body) if isinstance(s, Decl) and s.shape)


@pytest.mark.slow  # builds every gated class through L = 12
def test_every_low_stack_class_declares_at_most_13_kb_of_arrays():
    """R, U and the tables: the frame's floor (the measured frames, with
    the compiler's spill, are at most 13,792 B on sm_86)."""
    worst = max(_array_bytes(kg.build_geri3_kernel(c, lowstack=True))
                for c in ERI3_LADDER)
    assert worst == _array_bytes(kg.build_geri3_kernel((3, 3, 6),
                                                       lowstack=True))
    assert worst <= 13_000, worst
    assert max(_array_bytes(km.build_gmetric_kernel(c, lowstack=True))
               for c in METRIC_LADDER) <= 8_000


@pytest.mark.slow  # builds the unrolled (ff|i), ~20 s
def test_the_unrolled_ff_i_builds_at_the_default_recursion_limit():
    """#613: a 3,360-term fold overflowed Python's default recursion limit
    in the IR checker; the contraction folds are chunked
    (`kernel_gmetric.chunked_fold`), bitwise the same left fold."""
    import sys
    assert sys.getrecursionlimit() <= 1000
    k = kg.build_geri3_kernel((3, 3, 6), lowstack=False)
    links = [s for s in walk(k.body)
             if isinstance(s, Decl) and "_c" in s.name and s.name[0] == "s"]
    assert len(links) >= 9 * 10, len(links)  # nine folds, each chunked


def test_a_chunked_fold_is_the_same_left_fold():
    from felizjueves.ir import Call, Lit, Var
    terms = [Lit(float(i) * 0.1) for i in range(300)]
    links, last = km.chunked_fold(terms, "s")
    assert [d.name for d in links] == ["s_c0", "s_c1"]
    want = terms[0].value
    for t in terms[1:]:
        want = want + t.value
    env: dict[str, float] = {}

    def ev(e) -> float:
        if isinstance(e, Lit):
            return e.value
        if isinstance(e, Var):
            return env[e.name]
        assert isinstance(e, Call)
        a, b = (ev(x) for x in e.args)
        return a + b
    for d in links:
        env[d.name] = ev(d.init)
    assert ev(last) == want  # bitwise: the same additions in the same order
    assert km.chunked_fold(terms[:128], "s")[0] == []


# --- the host, every class of the ladder ----------------------------------------


def _eri3_bases(cls):
    """`la` at A, `lb` at B, the higher listed first (so `la < lb` is
    staged swapped); `la == lb`: two shells, so a diagonal and an
    off-diagonal pair; the aux shell at C. One primitive each."""
    la, lb, lP = cls
    sh = [Shell(lb, B, (1.1,), (1.0,)), Shell(la, A, (0.8,), (1.0,))]
    return Basis(tuple(sh)), Basis((Shell(lP, C, (1.6,), (1.0,)),))


def _metric_basis(cls):
    lP, lQ = cls
    return Basis((Shell(lQ, B, (1.3,), (1.0,)), Shell(lP, A, (0.9,), (1.0,))))


def _gam(basis, aux):
    return np.random.default_rng(5).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))


def _eri3_scale(basis, aux, Gam, cells) -> float:
    nat, ns = len(NUCLEI), len(basis.shells)
    real, areal = shell_atoms(basis, NUCLEI), shell_atoms(aux, NUCLEI)
    split = tuple(nat + i for i in range(ns))
    asplit = tuple(nat + ns + i for i in range(len(aux.shells)))
    sc = np.zeros((nat, 3))
    for c in cells:
        g = np.abs(np.array(host_df.eri3_gradient(
            basis, aux, np.abs(Gam), NUCLEI, pq=pq_tower, cells=[c],
            atoms=split, aux_atoms=asplit)))
        for i, row in enumerate(split):
            sc[real[i]] += g[row]
        for i, row in enumerate(asplit):
            sc[areal[i]] += g[row]
    return float(sc.max())


def _metric_scale(aux, W, pairs) -> float:
    nat = len(NUCLEI)
    real = shell_atoms(aux, NUCLEI)
    split = tuple(nat + i for i in range(len(aux.shells)))
    sc = np.zeros((nat, 3))
    for p in pairs:
        g = np.abs(np.array(host_df.metric_gradient(
            aux, np.abs(W), NUCLEI, pq=pq_tower, pairs=[p], atoms=split)))
        for i, row in enumerate(split):
            sc[real[i]] += g[row]
    return float(sc.max())


def _check_eri3(cls, lowstack):
    basis, aux = _eri3_bases(cls)
    Gam = _gam(basis, aux)
    G, ran = run_eri3(basis, aux, NUCLEI, Gam, boys=boys, classes={cls},
                      lowstack=lowstack)
    assert ran == (cls,)
    sh = basis.shells
    cells = [(i, j, 0) for i in range(len(sh)) for j in range(i, len(sh))
             if tuple(sorted((sh[i].l, sh[j].l))) == cls[:2]]
    want = np.array(host_df.eri3_gradient(basis, aux, Gam, NUCLEI, pq=pq,
                                          cells=cells))
    scale = _eri3_scale(basis, aux, Gam, cells)
    err = float(np.abs(G - want).max())
    print(f"\n{kg.geri3_name(cls)}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale
    assert np.abs(G.sum(axis=0)).max() <= TOL * scale


def _check_metric(cls, lowstack):
    aux = _metric_basis(cls)
    W = np.random.default_rng(5).standard_normal((aux.n_ao, aux.n_ao))
    G, ran = run_metric(aux, NUCLEI, W, boys=boys, classes={cls},
                        lowstack=lowstack)
    assert ran == (cls,)
    pairs = [(0, 1)] if cls[0] != cls[1] else [(0, 0), (0, 1), (1, 1)]
    want = np.array(host_df.metric_gradient(aux, W, NUCLEI, pq=pq,
                                            pairs=pairs))
    scale = _metric_scale(aux, W, pairs)
    err = float(np.abs(G - want).max())
    print(f"\n{km.gmetric_name(cls)}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_eid(c))
                                 for c in ERI3_LADDER if sum(c) >= 5])
def test_every_geri3_class_above_l4_is_the_host_gradient(cls):
    _check_eri3(cls, None)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_mid(c))
                                 for c in METRIC_LADDER if sum(c) >= 5])
def test_every_gmetric_class_above_l4_is_the_host_gradient(cls):
    _check_metric(cls, None)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_eid(c))
                                 for c in ERI3_LADDER if sum(c) <= 4])
def test_the_forced_low_stack_geri3_is_the_host_gradient(cls):
    _check_eri3(cls, True)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_mid(c))
                                 for c in METRIC_LADDER if sum(c) <= 4])
def test_the_forced_low_stack_gmetric_is_the_host_gradient(cls):
    _check_metric(cls, True)


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(0, 0, 5), (1, 2, 2)], ids=_eid)
def test_the_unrolled_geri3_above_the_floor_is_the_host_gradient(cls):
    """Forced unrolled above the floor (where it is not kept, for its
    stack), so both forms are held to the host where both build."""
    _check_eri3(cls, False)


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(4, 6), (5, 5)], ids=_mid)
def test_the_unrolled_gmetric_above_the_floor_is_the_host_gradient(cls):
    _check_metric(cls, False)


# --- libcint: the aux-i witness (#613) -------------------------------------------


@cache
def _witness():
    wit = manifest.df_gradient()
    ts = wit.three_center
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return wit, basis, aux, nuclei


def _big(grads) -> float:
    return max(float(np.abs(g).max()) for g in grads.values())


def test_the_witness_reaches_every_ladder_class_on_the_sets():
    wit, *_ = _witness()
    assert sorted(wit.eri3) == ERI3_LADDER
    assert sorted(wit.gmetric) == METRIC_LADDER
    assert wit.three_center.name == "prim_spd_f_auxi_3c"
    assert wit.metric.name == "prim_spd_f_auxi_metric"
    g = wit.gam()
    assert g.shape == (20, 20, 168) and not np.array_equal(
        g, g.transpose(1, 0, 2))
    assert np.array_equal(wit.w(), manifest.df_gradient_w(168))


@pytest.mark.slow  # the host over every class of the system, ~2 min
def test_the_host_is_libcints_at_every_ladder_class():
    wit, basis, aux, nuclei = _witness()
    sh, ash = basis.shells, aux.shells
    worst = 0.0
    big = _big(wit.eri3)
    for cls in ERI3_LADDER:
        cells = [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
                 for n in range(len(ash))
                 if (*sorted((sh[i].l, sh[j].l)), ash[n].l) == cls]
        g = np.array(host_df.eri3_gradient(basis, aux, wit.gam(), nuclei,
                                           pq=pq, cells=cells))
        worst = max(worst, float(np.abs(g - wit.eri3[cls]).max()) / big)
    bigm = _big(wit.gmetric)
    worstm = 0.0
    for cls in METRIC_LADDER:
        pairs = [(i, j) for i in range(len(ash)) for j in range(i, len(ash))
                 if tuple(sorted((ash[i].l, ash[j].l))) == cls]
        g = np.array(host_df.metric_gradient(aux, wit.w(), nuclei, pq=pq,
                                             pairs=pairs))
        worstm = max(worstm, float(np.abs(g - wit.gmetric[cls]).max()) / bigm)
    print(f"\nhost vs libcint: geri3 {worst:.2e}, gmetric {worstm:.2e} of max")
    assert worst <= LIBCINT_TOL and worstm <= LIBCINT_TOL


@pytest.mark.slow  # interpreted at L = 11-12: minutes a class
@pytest.mark.parametrize("cls", [(2, 3, 6), (3, 3, 5), (3, 3, 6)], ids=_eid)
def test_a_top_geri3_class_is_libcints(cls):
    wit, basis, aux, nuclei = _witness()
    G, ran = run_eri3(basis, aux, nuclei, wit.gam(), boys=boys,
                      classes={cls})
    assert ran == (cls,)
    w = float(np.abs(G - wit.eri3[cls]).max()) / _big(wit.eri3)
    print(f"\n{kg.geri3_name(cls)} vs libcint: {w:.2e} of max")
    assert w <= LIBCINT_TOL


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(5, 6), (6, 6)], ids=_mid)
def test_a_top_gmetric_class_is_libcints(cls):
    wit, _, aux, nuclei = _witness()
    G, ran = run_metric(aux, nuclei, wit.w(), boys=boys, classes={cls})
    assert ran == (cls,)
    w = float(np.abs(G - wit.gmetric[cls]).max()) / _big(wit.gmetric)
    print(f"\n{km.gmetric_name(cls)} vs libcint: {w:.2e} of max")
    assert w <= LIBCINT_TOL


def test_a_witness_on_other_inputs_is_refused(tmp_path):
    import json
    src = manifest.DF_GRADIENT_DIR / f"{manifest.DF_GRADIENT}.json"
    raw = json.loads(src.read_text())
    raw["provenance"]["mol"] = {**raw["provenance"]["mol"],
                                "atom": "He 0 0 0; Ne 0.3 -0.5 1.3"}
    (tmp_path / src.name).write_text(json.dumps(raw))
    with pytest.raises(manifest.FixtureError, match="provenance.mol"):
        manifest.df_gradient(where=tmp_path)


@pytest.mark.oracle
def test_the_witness_is_what_pyscf_computes_today():
    """A drift is news about PySCF/libcint until proven otherwise."""
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    import json

    from oracle import generate
    fresh = generate.df_gradient_build()
    tracked = json.loads((manifest.DF_GRADIENT_DIR
                          / f"{manifest.DF_GRADIENT}.json").read_text())
    assert generate.comparable(fresh) == generate.comparable(tracked)
