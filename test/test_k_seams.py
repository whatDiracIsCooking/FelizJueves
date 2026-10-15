"""`k_levers`: the K kernel's lever seams, and the frozen path behind them (#367).

Three things are asserted, the issue's acceptance in order:

1. THE LITERAL KERNEL IS BYTE-IDENTICAL. The emitted CUDA and HIP text of
   every K kernel the builders make through L = 4 -- every class at one
   batch, and every class at 3 and 6 batches where the plan admits the
   split -- digested, against a digest computed from the tree BEFORE #367
   (`main` at 6a57a32, the same enumeration run over its `src/`). The K
   subgrid cells through L = 10 are the same check, `slow`.
2. EVERY HOOK HAS A DEFAULT, AND THE DEFAULT IS THE FROZEN PATH: the
   resolution of a literal spec is `DEFAULT_HOOKS`; each hook, wrapped by a
   spy that delegates to its default, is CALLED by the builder and changes
   nothing; and each, replaced, changes the kernel (it is on the path).
3. THE HOOKS ARE SUFFICIENT: a sketch of #341's per-element split lever,
   written here against `keep` and `fan_out` only and plugged in exactly as a
   lever module would be (a `levers.VOCABULARY` entry and a
   `k_levers.REGISTRY` line, both monkeypatched), builds `na * nb` kernels
   per batch whose union is the literal kernel's `K` BIT FOR BIT,
   interpreted, at every class with L <= 2 -- one batch and three. A second
   sketch, #342's shape (`E^ac` bound under another name and read through
   `e_ac`), shows the binding and its reads move together. Neither ships.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import e_fragment as ef
from felizjueves import k_levers, levers
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Decl, Index, Inline, Kernel, Lit, Note, Var, walk
from felizjueves.k_levers import seams
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KContext, KHooks
from felizjueves.kernel_k import (
    KSH,
    build_k_kernel,
    build_k_kernels,
    k_class_spec,
    launch_shape,
)
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# 1. Byte-identity of the literal kernel, against the pre-#367 tree
# ---------------------------------------------------------------------------

LADDER: list[Am] = [am for am in itertools.product(range(5), repeat=4)
                    if sum(am) <= 4]


def _ladder_specs() -> list[Spec]:
    """Every K class through L = 4 at one batch, then at 3 and 6 batches
    where the plan admits the split (an empty batch is refused by `plan`)."""
    out = [k_class_spec(*am) for am in LADDER]
    for am in LADDER:
        for n in (3, 6):
            spec = Spec(Operator.COULOMB, 0, am, Target.K, n_batches=n)
            try:
                build_k_kernels(spec)
            except ValueError:
                continue
            out.append(spec)
    return out


def digest(specs: list[Spec]) -> tuple[int, str]:
    """`(kernels, sha256)` over each spec's `build_k_kernels`, in order, each
    kernel's `emit.cuda` (sm_86) then `emit.hip` (gfx1200) text."""
    h, n = hashlib.sha256(), 0
    for spec in specs:
        for kern in build_k_kernels(spec):
            h.update(cuda.emit(kern, SM_86).encode())
            h.update(hip.emit(kern, GFX1200).encode())
            n += 1
    return n, h.hexdigest()


# Computed by `digest` over the pre-#367 tree's src/ (main at 6a57a32).
LADDER_PIN = (607, "ae2c63aa770d4257a81382e81fde53dcd8d5a8ff0609871eab9d4e9299e274f0")
SUBGRID_PIN = (62, "7fff10a0d78d89ab4f4805dbda7878ed12c2f6c619952bb92370c09504e31cef")


def test_the_literal_ladder_is_byte_identical_to_before_the_seams() -> None:
    specs = _ladder_specs()
    assert len([s for s in specs if s.n_batches == 1]) == 70
    assert digest(specs) == LADDER_PIN


@pytest.mark.slow
def test_the_k_subgrid_through_l10_is_byte_identical() -> None:
    from felizjueves.measured_store import K_SUBGRID

    specs = [s for s in K_SUBGRID if sum(s.am) <= 10]
    assert len(specs) == 10
    assert digest(specs) == SUBGRID_PIN


# ---------------------------------------------------------------------------
# 2. Every hook has a default, and the default is the frozen path
# ---------------------------------------------------------------------------

HOOKS = tuple(f.name for f in dataclasses.fields(KHooks))
DEFAULT_OF = {"keep": seams.keep_all, "fan_out": seams.one_kernel,
              **{h: getattr(seams, h) for h in HOOKS
                 if h not in ("keep", "fan_out")}}


def test_every_hook_defaults_to_the_frozen_function() -> None:
    assert HOOKS == ("keep", "fan_out", "shared_decl", "bind_e_ac", "e_ac",
                     "bind_e_bd", "e_bd", "contract", "ket_stage",
                     "bra_digest", "init_acc", "write_back", "params",
                     "finish")
    for h in HOOKS:
        assert getattr(DEFAULT_HOOKS, h) is DEFAULT_OF[h], h
        assert DEFAULT_HOOKS.is_default(h)
    assert KHooks() == DEFAULT_HOOKS


def test_a_literal_spec_resolves_to_the_defaults_and_nothing_is_registered(
) -> None:
    assert k_levers.REGISTRY == {}
    for spec in (k_class_spec(1, 0, 1, 0),
                 Spec(Operator.COULOMB, 0, (1, 1, 0, 0), Target.K,
                      n_batches=3)):
        assert k_levers.hooks_for(spec) is DEFAULT_HOOKS
        assert build_k_kernels(spec) == build_k_kernels(spec,
                                                        hooks=DEFAULT_HOOKS)


SPY_SPECS = [k_class_spec(1, 1, 1, 0),
             Spec(Operator.COULOMB, 0, (1, 1, 1, 0), Target.K, n_batches=3)]


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("spec", SPY_SPECS, ids=["b1", "b3"])
def test_each_hook_is_on_the_path_and_its_default_is_the_literal(
        hook: str, spec: Spec) -> None:
    calls: list[object] = []
    default = DEFAULT_OF[hook]

    def spy(*args):
        calls.append(args)
        return default(*args)

    built = build_k_kernels(spec, hooks=dataclasses.replace(
        DEFAULT_HOOKS, **{hook: spy}))
    assert calls, f"{hook} was never called"
    assert built == build_k_kernels(spec)


def _marked(ctx: KContext, *_: object) -> tuple:
    return (Note("MARK"),)


# What a replaced hook returns, per hook, to show the builder uses its value.
REPLACEMENTS = {
    "keep": lambda spec, ia, ib: (ia, ib) == (0, 0),
    "fan_out": lambda spec, h: (("_x", h), ("_y", h)),
    "shared_decl": lambda ctx: dataclasses.replace(
        seams.shared_decl(ctx), shape=(Lit(ctx.n_shared + 1),
                                       seams.shared_decl(ctx).shape[1])),
    "bind_e_ac": lambda ctx: (*seams.bind_e_ac(ctx), Note("MARK")),
    "e_ac": lambda ctx, rac, i: Index("Eac", (Lit(rac), Lit(0))),
    "bind_e_bd": lambda ctx: (*seams.bind_e_bd(ctx), Note("MARK")),
    "e_bd": lambda ctx, rbd, j: Index("Ebd", (Lit(rbd), Lit(0))),
    "contract": lambda ctx: (*seams.contract(ctx), Note("MARK")),
    "ket_stage": lambda ctx, rbd: (*seams.ket_stage(ctx, rbd), Note("MARK")),
    "bra_digest": lambda ctx, rbd: (*seams.bra_digest(ctx, rbd),
                                    Note("MARK")),
    "init_acc": lambda ctx: (*seams.init_acc(ctx), Note("MARK")),
    "write_back": lambda ctx: (Note("MARK"), *seams.write_back(ctx)),
    "params": lambda ctx: seams.params(ctx)[::-1],
    "finish": lambda ctx, kern: dataclasses.replace(kern, name="renamed"),
}


@pytest.mark.parametrize("hook", HOOKS)
def test_each_hook_replaced_changes_the_kernel(hook: str) -> None:
    assert set(REPLACEMENTS) == set(HOOKS)
    spec = SPY_SPECS[0]
    built = build_k_kernels(spec, hooks=dataclasses.replace(
        DEFAULT_HOOKS, **{hook: REPLACEMENTS[hook]}))
    assert built != build_k_kernels(spec)


def test_a_fan_out_is_refused_by_the_one_kernel_builder() -> None:
    fan = dataclasses.replace(DEFAULT_HOOKS,
                              fan_out=REPLACEMENTS["fan_out"])
    with pytest.raises(ValueError, match="fans out into 2 kernels"):
        build_k_kernel(SPY_SPECS[0], hooks=fan)
    names = [k.name for k in build_k_kernels(SPY_SPECS[0], hooks=fan)]
    assert names == ["k_11_10_x", "k_11_10_y"]
    with pytest.raises(TypeError, match="KHooks"):
        build_k_kernel(SPY_SPECS[0], hooks=object())  # type: ignore[arg-type]


def test_hooks_for_composes_registered_levers_in_key_order(
        monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    for name in ("zz-probe", "aa-probe"):
        monkeypatch.setitem(levers.VOCABULARY, name, levers.Lever.of(
            name, 367, "a probe: moves one K hook", {"k"}))

        def apply(h: KHooks, spec: Spec, name: str = name) -> KHooks:
            order.append(name)
            return h
        monkeypatch.setitem(k_levers.REGISTRY, name, apply)
    spec = Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K,
                ("aa-probe", "zz-probe"))
    assert k_levers.hooks_for(spec) is DEFAULT_HOOKS
    assert order == ["aa-probe", "zz-probe"]  # the key's (sorted) order

    monkeypatch.setitem(k_levers.REGISTRY, "aa-probe", lambda h, s: None)
    with pytest.raises(TypeError, match="not KHooks"):
        k_levers.hooks_for(spec)


# ---------------------------------------------------------------------------
# 3. Sufficiency: #341's per-element split, sketched on `keep` + `fan_out`
# ---------------------------------------------------------------------------

SPLIT = "per-element-sketch"


def _element(ia: int, ib: int):
    return lambda spec, a, b: (a, b) == (ia, ib)


def split_apply(hooks: KHooks, spec: Spec) -> KHooks:
    """#341, sketched: one kernel per `K[a_i, b_j]`, each carrying only that
    element's accumulator; its name suffixed `_e{i}_{j}`. Two hooks, no
    edit to `kernel_k`."""
    la, _, lb, _ = spec.am

    def fan_out(spec: Spec, h: KHooks) -> tuple[tuple[str, KHooks], ...]:
        return tuple((f"_e{ia}_{ib}", dataclasses.replace(
            h, keep=_element(ia, ib), fan_out=seams.one_kernel))
            for ia in range(seams.ncart(la)) for ib in range(seams.ncart(lb)))

    return dataclasses.replace(hooks, fan_out=fan_out)


@pytest.fixture
def split_lever(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(levers.VOCABULARY, SPLIT, levers.Lever.of(
        SPLIT, 341, "per-element split K kernel (a test sketch, #367)",
        {"k"}))
    monkeypatch.setitem(k_levers.REGISTRY, SPLIT, split_apply)
    return SPLIT


def _run(kernels: tuple[Kernel, ...], am: Am, *, zero: bool) -> np.ndarray:
    """The kernels in order onto one `K`, one lane, the test_kernel_k shells."""
    bra, ket = tk._pairs(am)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3])])
    if zero:
        args["K"] = np.zeros_like(args["K"])
    grid, block = launch_shape(args["n_bra"], 1)
    for kern in kernels:
        args["K"] = Interpreter(kern, launch=Launch(grid, block, 1),
                                boys=boys).run(**args)["K"]
    return args["K"]


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a))
                                for a in tk.FAST])
def test_the_split_sketch_is_the_literal_block_bitwise(
        am: Am, split_lever: str) -> None:
    spec = Spec(Operator.COULOMB, 0, am, Target.K, (split_lever,))
    kernels = build_k_kernels(spec)
    na, nb = seams.ncart(am[0]), seams.ncart(am[2])
    assert len(kernels) == na * nb
    literal = build_k_kernel(am)
    for kern in kernels:
        (sh,) = [s for s in walk(kern.body)
                 if isinstance(s, Decl) and s.name == KSH]
        # one accumulator slot after the I column, not na * nb
        assert sh.shape[0].value <= tk.shared_doubles(am) - na * nb + 1
        assert any(isinstance(s, Note) and "#341" in s.prov
                   for s in kern.body)
    K = _run(kernels, am, zero=False)  # NaN-filled: every entry is written
    assert not np.isnan(K).any()
    assert K.tolist() == _run((literal,), am, zero=False).tolist()


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a))
                                for a in tk.FAST if sum(a) > 0])
def test_the_split_sketch_composes_with_the_batch_split(
        am: Am, split_lever: str) -> None:
    """Three batches x `na nb` elements, each atomized: onto a zeroed `K`,
    each element sees its batches' partials in the literal order, so the
    result is the literal batched kernels' bit for bit."""
    spec = Spec(Operator.COULOMB, 0, am, Target.K, (split_lever,), 3)
    literal = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K, (), 3))
    kernels = build_k_kernels(spec)
    assert len(kernels) <= 3 * seams.ncart(am[0]) * seams.ncart(am[2])
    assert (_run(kernels, am, zero=True).tolist()
            == _run(literal, am, zero=True).tolist())


def test_an_e_ac_rebinding_moves_the_binding_and_its_reads_together() -> None:
    """#342's shape on the seams: `E^ac` bound under another name, every
    read through `e_ac` -- the kernel computes the same `K`, bitwise."""
    def bind(ctx: KContext) -> tuple:
        out = seams.bind_e_ac(ctx)
        return tuple(dataclasses.replace(s, name="Eac2")
                     if isinstance(s, Decl) and s.name == "Eac"
                     else _rebind(s) for s in out)

    def _rebind(s):
        if isinstance(s, Inline):
            return Inline(s.fragment, {**dict(s.bindings),
                                       ef.E_ARRAY: Var("Eac2")})
        return s

    hooks = dataclasses.replace(
        DEFAULT_HOOKS, bind_e_ac=bind,
        e_ac=lambda ctx, rac, i: Index("Eac2", (Lit(rac), Lit(i))))
    for am in [(1, 1, 1, 0), (2, 0, 0, 0), (0, 1, 1, 0)]:
        kern = build_k_kernel(am, hooks=hooks)
        assert kern != build_k_kernel(am)
        assert not any(isinstance(s, Decl) and s.name == "Eac"
                       for s in walk(kern.body))
        assert (_run((kern,), am, zero=False).tolist()
                == _run((build_k_kernel(am),), am, zero=False).tolist())
