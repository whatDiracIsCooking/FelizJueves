"""`f_levers`: the fused F kernel's lever seams, and the frozen path behind them (#368).

Three things are asserted, the issue's acceptance in order (#367's shape, for
`kernel_f`):

1. THE LITERAL KERNEL IS BYTE-IDENTICAL. The emitted CUDA and HIP text of
   every F class through L = 4 (milestone 13's cap) -- each built three ways:
   shipped, with the `eris=True` tap, and with the degeneracy factor off (a
   `Weights` the tests substitute) -- digested, against a digest computed
   from the tree BEFORE #368 (`main` at 83e29b1, the same enumeration run
   over its `src/`), pinned at L <= 2 and at L <= 4. The 21 goldens per
   backend and `fock_digests.json` are test_emit_fock.py's, unchanged.
2. EVERY HOOK HAS A DEFAULT, AND THE DEFAULT IS THE FROZEN PATH: the
   resolution of a literal spec is `DEFAULT_HOOKS`; each hook, wrapped by a
   spy that delegates to its default, is CALLED by the builder and changes
   nothing; and each, replaced, changes the kernel (it is on the path).
3. THE HOOKS ARE SUFFICIENT: a sketch of #350's batched VRR -- the boundary
   staged in a thread-private `Space.LOCAL` array, cut into two groups, the
   primitive nest run once per group -- written against `boundary_init`,
   `nest`, `accumulate` and `boundary_read` only and plugged in exactly as a
   lever module would be (a `levers.VOCABULARY` entry legal on `"f"` and an
   `f_levers.REGISTRY` line, both monkeypatched), builds a kernel whose `F`
   and tap `g` are the literal kernel's BIT FOR BIT, interpreted, at every
   class with L <= 2. A second sketch, #353's shape, shows every packed
   pair-record read passes through `field`. Neither ships.

Fast tier: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
import test_kernel_f as tk

from felizjueves import f_levers, levers
from felizjueves.ao.shell import Basis
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import seams
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FContext, FHooks
from felizjueves.hgp import vrr
from felizjueves.hgp.recurrence import vrr_name
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    For,
    Index,
    Lit,
    Note,
    PrecisionError,
    Scalar,
    Space,
    Var,
    walk,
)
from felizjueves.kernel_f import Weights, build_f_kernel, f_class_spec

# ---------------------------------------------------------------------------
# 1. Byte-identity of the literal kernel, against the pre-#368 tree
# ---------------------------------------------------------------------------

BUILDS = ({}, {"eris": True}, {"weights": Weights(1.0, -0.5, False)})


def digest(lmax: int) -> tuple[int, str]:
    """`(kernels, sha256)` over every class of `vrr.CLASSES` with L <=
    `lmax`, in order, each built per `BUILDS`, each kernel's `emit.cuda`
    (sm_86) then `emit.hip` (gfx1200) text."""
    h, n = hashlib.sha256(), 0
    for cls in vrr.CLASSES:
        if sum(cls) > lmax:
            continue
        for kw in BUILDS:
            kern = build_f_kernel(cls, **kw)
            h.update(cuda.emit(kern, SM_86).encode())
            h.update(hip.emit(kern, GFX1200).encode())
            n += 1
    return n, h.hexdigest()


# Computed by `digest` over the pre-#368 tree's src/ (main at 83e29b1).
# Re-taken at #301, the one deliberate text change since: every unit's Boys
# became one `fj_boys<M>` fill (the counts are unchanged).
LADDER_PIN_2 = (15, "877c180635231b24c5804d6cc4b2eaa8b23ada3fba1111a14b61d3ae3f567277")
LADDER_PIN_4 = (39, "1b9245d4121db3ca573ec35243cafed01d78b6bb414e27368838269b7bcf042c")


def test_the_literal_ladder_is_byte_identical_to_before_the_seams() -> None:
    assert digest(2) == LADDER_PIN_2


def test_the_literal_ladder_through_l4_is_byte_identical() -> None:
    assert digest(4) == LADDER_PIN_4


# ---------------------------------------------------------------------------
# 2. Every hook has a default, and the default is the frozen path
# ---------------------------------------------------------------------------

HOOKS = tuple(f.name for f in dataclasses.fields(FHooks))
DEFAULT_OF = {"field": seams.field_read,
              **{h: getattr(seams, h) for h in HOOKS if h != "field"}}


def test_every_hook_defaults_to_the_frozen_function() -> None:
    assert HOOKS == (
        "shared_decl", "per_bra", "bra_setup", "stage", "ket_loop",
        "per_ket", "screen", "work", "boundary_init", "nest", "ket_prim",
        "quartet_scalars", "vrr", "accumulate", "boundary_read", "hrr",
        "contracted", "eri", "tap", "digest", "digest_setup",
        "zero_buffers", "predigest", "predigest_eri", "flush_ket",
        "flush_bra", "field", "real", "cast", "params", "finish")
    for h in HOOKS:
        assert getattr(DEFAULT_HOOKS, h) is DEFAULT_OF[h], h
        assert DEFAULT_HOOKS.is_default(h)
    assert FHooks() == DEFAULT_HOOKS


def test_a_literal_spec_resolves_to_the_defaults_and_nothing_is_registered(
) -> None:
    assert all("f" in levers.VOCABULARY[n].legal for n in f_levers.REGISTRY)
    for am in ((0, 1, 0, 1), (1, 1, 1, 1)):
        spec = f_class_spec(*am)
        assert f_levers.hooks_for(spec) is DEFAULT_HOOKS
        assert build_f_kernel(spec) == build_f_kernel(spec,
                                                      hooks=DEFAULT_HOOKS)


# (0,1|0,1): the bra and ket HRRs both run (AB, CD read) and P = Q is
# decided at run time; (1,1|0,1): a = b at run time. With the tap, so `tap`
# and `eri` write; without it, so the shipped kernel is spied too.
SPY_CASES = [((0, 1, 0, 1), True), ((1, 1, 0, 1), False)]


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("am,eris", SPY_CASES, ids=["0101-tap", "1101"])
def test_each_hook_is_on_the_path_and_its_default_is_the_literal(
        hook: str, am, eris: bool) -> None:
    calls: list[object] = []
    default = DEFAULT_OF[hook]

    def spy(*args):
        calls.append(args)
        return default(*args)

    built = build_f_kernel(am, eris=eris, hooks=dataclasses.replace(
        DEFAULT_HOOKS, **{hook: spy}))
    assert calls, f"{hook} was never called"
    assert built == build_f_kernel(am, eris=eris)


def _mark(fn):
    """`fn`'s statements with a `Note` appended."""
    return lambda *a: (*fn(*a), Note("MARK"))


# What a replaced hook returns, per hook, to show the builder uses its value.
REPLACEMENTS = {
    **{h: _mark(DEFAULT_OF[h]) for h in HOOKS
       if h not in ("boundary_read", "eri", "field", "real", "params",
                    "finish", "vrr", "hrr")},
    # the recurrences' statements are read through ctx.vbody / ctx.hbody
    "vrr": _mark(seams.vrr),
    "hrr": _mark(seams.hrr),
    "boundary_read": lambda ctx, e, f: Lit(0.0),
    "eri": lambda ctx, t: Lit(1.0),
    "field": lambda ctx, arr, idx: Index(arr, idx[::-1]),
    "real": lambda ctx, region: (Scalar.F32 if region == "vrr"
                                 else Scalar.F64),
    "cast": lambda ctx, e, region: (Call(Fn.ADD, (e, Lit(0.0)))
                                    if region == "fock" else e),
    "params": lambda ctx: seams.params(ctx)[::-1],
    "finish": lambda ctx, kern: dataclasses.replace(kern, name="renamed"),
}


@pytest.mark.parametrize("hook", HOOKS)
def test_each_hook_replaced_changes_the_kernel(hook: str) -> None:
    assert set(REPLACEMENTS) == set(HOOKS)
    am = (0, 1, 0, 1)
    hooks = dataclasses.replace(DEFAULT_HOOKS, **{hook: REPLACEMENTS[hook]})
    if hook == "real":
        # The type reaches the region's first Decl, and the IR's precision
        # hygiene refuses it there: `real` moved without its `cast` (a
        # narrowing lever moves both, and retypes `vrr`) cannot build.
        with pytest.raises(PrecisionError, match="Decl 'Qk0'.*F64 into F32"):
            build_f_kernel(am, eris=True, hooks=hooks)
        return
    built = build_f_kernel(am, eris=True, hooks=hooks)
    assert built != build_f_kernel(am, eris=True)


def test_the_builder_refuses_a_hooks_that_is_not_fhooks() -> None:
    with pytest.raises(TypeError, match="FHooks"):
        build_f_kernel((0, 0, 0, 0), hooks=object())  # type: ignore[arg-type]


def _vocab(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setitem(levers.VOCABULARY, name, levers.Lever.of(
        name, 368, "a probe: moves F hooks (a test sketch)", {"f"}))


def _f_lever(monkeypatch: pytest.MonkeyPatch, name: str, apply) -> None:
    _vocab(monkeypatch, name)
    monkeypatch.setitem(f_levers.REGISTRY, name, apply)


def test_hooks_for_composes_registered_levers_in_key_order(
        monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    for name in ("zz-probe", "aa-probe"):
        def apply(h: FHooks, spec, name: str = name) -> FHooks:
            order.append(name)
            return h
        _f_lever(monkeypatch, name, apply)
    spec = f_class_spec(0, 1, 0, 1, ("aa-probe", "zz-probe"))
    assert f_levers.hooks_for(spec) is DEFAULT_HOOKS
    assert order == ["aa-probe", "zz-probe"]  # the key's (sorted) order

    monkeypatch.setitem(f_levers.REGISTRY, "aa-probe", lambda h, s: None)
    with pytest.raises(TypeError, match="not FHooks"):
        f_levers.hooks_for(spec)


def test_the_builder_applies_a_registered_lever(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry line reaches `build_f_kernel` with no edit to it; an
    unregistered kernel-body lever stays a `Note` only (#340)."""
    _f_lever(monkeypatch, "rename-probe", lambda h, s: dataclasses.replace(
        h, finish=lambda ctx, kern: dataclasses.replace(kern, name="moved")))
    _vocab(monkeypatch, "note-probe")
    moved = build_f_kernel(f_class_spec(0, 1, 0, 1, ("rename-probe",)))
    assert moved.name == "moved"
    noted = build_f_kernel(f_class_spec(0, 1, 0, 1, ("note-probe",)))
    literal = build_f_kernel((0, 1, 0, 1))
    assert noted.name == literal.name
    assert [s for s in noted.body if not isinstance(s, Note)] == [
        s for s in literal.body if not isinstance(s, Note)]


# ---------------------------------------------------------------------------
# 3. Sufficiency: #350's batched VRR, sketched on four hooks
# ---------------------------------------------------------------------------

BATCHED = "batched-vrr-sketch"
STAGED = "boundary"  # the thread-private staged boundary, Space.LOCAL
N_GROUPS = 2


def _slot(ctx: FContext) -> dict:
    return {ef: i for i, ef in enumerate(ctx.bnd)}


def _groups(ctx: FContext) -> list[tuple]:
    """The boundary cut into `N_GROUPS` contiguous runs of its order."""
    n = len(ctx.bnd)
    cut = [round(g * n / N_GROUPS) for g in range(N_GROUPS + 1)]
    return [ctx.bnd[a:b] for a, b in zip(cut, cut[1:], strict=False) if b > a]


def _cell(ctx: FContext, ef) -> Index:
    return Index(STAGED, (Lit(_slot(ctx)[ef]),))


def staged_init(ctx: FContext) -> tuple:
    return (Note(f"#350 sketch: the {len(ctx.bnd)} boundary values staged "
                 "in a thread-private array, zeroed per ket"),
            Decl(STAGED, ctx.hooks.real(ctx, "boundary"),
                 (Lit(len(ctx.bnd)),), Space.LOCAL),
            *(Assign(_cell(ctx, ef), Lit(0.0)) for ef in ctx.bnd))


def staged_accumulate(ctx: FContext, group=None) -> tuple:
    return tuple(Assign(_cell(ctx, (e, f)), Call(Fn.ADD, (
        _cell(ctx, (e, f)), Var(vrr_name((e, f, 0))))))
        for e, f in (ctx.bnd if group is None else group))


def grouped_nest(ctx: FContext) -> tuple:
    """The primitive nest once per group, each accumulating only its run:
    the recompute-for-registers trade, with the literal fold order kept."""
    h, out = ctx.hooks, []
    for group in _groups(ctx):
        quartet = (*h.quartet_scalars(ctx), *ctx.vbody,
                   *staged_accumulate(ctx, group))
        out.append(For(seams.KCD, Lit(0), Var("k_cd"), Lit(1), (
            *h.ket_prim(ctx),
            For(seams.KAB, Lit(0), Var("k_ab"), Lit(1), quartet))))
    return tuple(out)


def batched_apply(hooks: FHooks, spec) -> FHooks:
    """#350, sketched: four hooks, no edit to `kernel_f`."""
    return dataclasses.replace(
        hooks, boundary_init=staged_init, accumulate=staged_accumulate,
        nest=grouped_nest,
        boundary_read=lambda ctx, e, f: _cell(ctx, (e, f)))


@pytest.fixture
def batched_lever(monkeypatch: pytest.MonkeyPatch) -> str:
    _f_lever(monkeypatch, BATCHED, batched_apply)
    return BATCHED


def _launch(kern, cls, monkeypatch: pytest.MonkeyPatch):
    """`F` (unclosed) and the tap `g` of `kern` over two bras by two kets,
    `test_kernel_f`'s shells and density, two threads."""
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    geoms = [tk.CENTRES, tk._moved((0.3, -0.2, 0.1))]
    basis = Basis(tuple(s for g in geoms for s in tk._shells(cls, g)))
    bras = [tk.records(basis, 0, 1), tk.records(basis, 4, 5)]
    kets = [tk.records(basis, 2, 3), tk.records(basis, 6, 7)]
    F, g, _ = tk.fused(cls, basis, tk._density(basis.n_ao), bras, kets,
                       threads=2, eris=True)
    return F, g


L2 = [c for c in vrr.CLASSES if sum(c) <= 2]


@pytest.mark.parametrize("cls", L2, ids=str)
def test_the_batched_vrr_sketch_is_the_literal_kernel_bitwise(
        cls, batched_lever: str, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = f_class_spec(*cls, (batched_lever,))
    kern = build_f_kernel(spec, eris=True)
    literal = build_f_kernel(cls, eris=True)
    assert kern != literal
    staged = [s for s in walk(kern.body)
              if isinstance(s, Decl) and s.name == STAGED]
    assert len(staged) == 1 and staged[0].space is Space.LOCAL
    assert not any(isinstance(s, Decl) and s.name.startswith("C")
                   and s.name not in ("CD_x", "CD_y", "CD_z")
                   for s in walk(kern.body))  # no register accumulator left
    nests = [s for s in walk(kern.body)
             if isinstance(s, For) and s.var == seams.KCD]
    assert len(nests) == min(N_GROUPS, len(vrr.targets(cls)))
    F, g = _launch(kern, cls, monkeypatch)
    F0, g0 = _launch(literal, cls, monkeypatch)
    assert np.array_equal(g, g0)
    assert np.array_equal(F, F0)


RECORDS = ("bra_prim", seams.AB_SH, "ket_prim", "AB", "CD", "G_ab", "G_cd",
           "bra_sh", "bra_ao", "ket_sh", "ket_ao")


def _reads(kern) -> set[str]:
    """Every array an `Index` reads in `kern` (assignment targets aside)."""
    from felizjueves.ir import AtomicAdd, _exprs_of, _subexprs

    out: set[str] = set()
    for s in walk(kern.body):
        if isinstance(s, (Assign, AtomicAdd)):  # a target is not a read
            at = s.dst.at if isinstance(s.dst, Index) else ()
            exprs = (s.value, *at)
        else:
            exprs = _exprs_of(s)
        for e in exprs:
            if e is None:
                continue
            for x in _subexprs(e):
                if isinstance(x, Index):
                    out.add(x.array)
    return out


@pytest.mark.parametrize("cls", [(0, 1, 0, 1), (1, 1, 1, 1)], ids=str)
def test_every_packed_record_read_goes_through_field(cls) -> None:
    """#353's shape: a `field` that retargets every record read (here to
    `<arr>_t`, a typed view's stand-in) leaves no read of a packed record
    behind; `D`, `Dsh` and the `ab_sh` writes are not record reads."""
    hooks = dataclasses.replace(
        DEFAULT_HOOKS, field=lambda ctx, arr, idx: Index(f"{arr}_t", idx))
    literal = _reads(build_f_kernel(cls, eris=True))
    assert set(RECORDS) & literal == set(RECORDS) - (
        set() if cls[1] and cls[3] else {"AB", "CD"})
    moved = _reads(build_f_kernel(cls, eris=True, hooks=hooks))
    assert not set(RECORDS) & moved, sorted(set(RECORDS) & moved)
    assert {"D", "Dsh"} <= moved
