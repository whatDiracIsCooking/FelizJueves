"""The kernel IR: construction, validation, structural equality, the Raw census.

Fast tier: builds dataclasses, compiles nothing, launches nothing.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from felizjueves import ir
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Boys,
    Break,
    Call,
    Decl,
    Factor,
    Fn,
    For,
    Fragment,
    If,
    Index,
    Inline,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Raw,
    Scalar,
    Space,
    Term,
    Var,
    WarpReduce,
    Weighted,
    walk,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def count_raw_in_src(root: Path = SRC) -> int:
    """Every `Raw(...)` constructed anywhere under src/ (ir.py defines, not uses)."""
    n = 0
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "Raw") or (
                isinstance(f, ast.Attribute) and f.attr == "Raw"
            ):
                n += 1
    return n


def test_the_raw_census() -> None:
    """Raw is a pressure valve, not a design. Metastasis has to be visible."""
    assert count_raw_in_src() == 0  # raise this number only with a reason


def test_the_census_sees_both_spellings(tmp_path: Path) -> None:
    # The census is only worth its number if it can count: check it on a
    # synthetic tree with both call spellings.
    (tmp_path / "m.py").write_text(
        "from felizjueves import ir\n"
        "from felizjueves.ir import Raw\n"
        "a = Raw('x')\n"
        "b = ir.Raw('y')\n"
        "class Raw: ...\n"
    )
    assert count_raw_in_src(tmp_path) == 2


# ---------------------------------------------------------------------------
# The recurrence instruction and its closed vocabulary
# ---------------------------------------------------------------------------


def _unit_and_general(d: int, r_d: int) -> Weighted:
    """[r]^(m) = PQ_d [r-1d]^(m+1) + (r_d - 1) [r-2d]^(m+1)."""
    terms = [Term(1, Factor.pq(d), Var("parent"))]
    if r_d >= 2:
        terms.append(Term(r_d - 1, Factor.ONE, Var("grandparent")))
    return Weighted(terms)


def test_the_unit_component_is_one_term() -> None:
    # The cheapness of r_d = 1 falls out of the form: the second term vanishes.
    assert len(_unit_and_general(0, 1).terms) == 1
    w = _unit_and_general(2, 4)
    assert [(t.c, t.g) for t in w.terms] == [(1, Factor.PQ_Z), (3, Factor.ONE)]


def test_the_factor_vocabulary_is_closed_and_named() -> None:
    assert [f.name for f in Factor] == [
        "ONE", "PQ_X", "PQ_Y", "PQ_Z",
        # #152: the E recurrence's g_k (test_e_fragment uses each one)
        "HALF_INV_ZETA", "PA_X", "PA_Y", "PA_Z", "PB_X", "PB_Y", "PB_Z",
        # #277: the HGP recurrences, [Fock] eqs 25-26 and 28
        "WP_X", "WP_Y", "WP_Z", "WQ_X", "WQ_Y", "WQ_Z", "QC_X", "QC_Y", "QC_Z",
        "INV2P", "INV2Q", "INV2PQ", "INV2P_ALPHA_OVER_P", "INV2Q_ALPHA_OVER_Q",
        "AB_X", "AB_Y", "AB_Z", "CD_X", "CD_Y", "CD_Z"]
    assert [Factor.pq(d) for d in range(3)] == [Factor.PQ_X, Factor.PQ_Y, Factor.PQ_Z]
    assert [Factor.pa(d) for d in range(3)] == [Factor.PA_X, Factor.PA_Y, Factor.PA_Z]
    assert [Factor.pb(d) for d in range(3)] == [Factor.PB_X, Factor.PB_Y, Factor.PB_Z]
    assert [Factor.wp(d) for d in range(3)] == [Factor.WP_X, Factor.WP_Y, Factor.WP_Z]
    assert [Factor.wq(d) for d in range(3)] == [Factor.WQ_X, Factor.WQ_Y, Factor.WQ_Z]
    assert [Factor.qc(d) for d in range(3)] == [Factor.QC_X, Factor.QC_Y, Factor.QC_Z]
    assert [Factor.ab(d) for d in range(3)] == [Factor.AB_X, Factor.AB_Y, Factor.AB_Z]
    assert [Factor.cd(d) for d in range(3)] == [Factor.CD_X, Factor.CD_Y, Factor.CD_Z]
    assert Factor.ONE.name_in_kernel is None
    assert Factor.PQ_Y.name_in_kernel == "PQ_y"
    assert Factor.HALF_INV_ZETA.name_in_kernel == "half_inv_zeta"
    assert Factor.INV2P_ALPHA_OVER_P.name_in_kernel == "inv2p_alpha_over_p"
    # Bound names are distinct: two members never claim one kernel scalar.
    names = [f.value for f in Factor]
    assert len(set(names)) == len(names)
    # Every bound name is an identifier: it is spelled as a C name.
    for f in Factor:
        if f.name_in_kernel is not None:
            ir._ident(f.name_in_kernel, "Factor name")


@pytest.mark.parametrize("g", ["PQ_x", "1", "theta", 1, None])
def test_a_factor_outside_the_vocabulary_raises(g: object) -> None:
    # A string that *spells* a member is still not one: that is Raw in disguise.
    with pytest.raises(ValueError, match="Factor vocabulary"):
        Term(1, g, Var("a"))  # type: ignore[arg-type]


@pytest.mark.parametrize("c", [0, 1.0, -1.0, True, False])
def test_a_coefficient_is_a_nonzero_int(c: object) -> None:
    # c = 0 is still refused now that c is signed (#277): a zero term is
    # dropped, not emitted -- the unit-component saving.
    with pytest.raises(ValueError, match="nonzero int"):
        Term(c, Factor.ONE, Var("a"))  # type: ignore[arg-type]


@pytest.mark.parametrize("c", [-1, -2, 1, 3])
def test_a_coefficient_is_signed(c: int) -> None:
    # eq 25's (m+1) lowering term is -e_d * inv2p_alpha_over_p (#277).
    t = Term(c, Factor.INV2P_ALPHA_OVER_P, Var("a"))
    assert t.c == c


# ---------------------------------------------------------------------------
# #277: the vocabulary read off the built HGP DAGs, and the registry
# ---------------------------------------------------------------------------

# hgp.vrr's coefficient names, per reduction axis d -> the Factor that is its
# g_k. The product names are the DERIVED scalars (Factor docstring, decision
# 2): one bound name each, never a two-factor Term.
_VRR_FACTOR = {
    "PA": Factor.pa, "WP": Factor.wp, "QC": Factor.qc, "WQ": Factor.wq,
    "inv2p": lambda d: Factor.INV2P,
    "inv2q": lambda d: Factor.INV2Q,
    "inv2pq": lambda d: Factor.INV2PQ,
    "inv2p*a_over_p": lambda d: Factor.INV2P_ALPHA_OVER_P,
    "inv2q*alpha_over_q": lambda d: Factor.INV2Q_ALPHA_OVER_Q,
}
_HRR_FACTOR = {"1": Factor.ONE} | {
    f"{side}_{ax}": getattr(Factor, f"{side}_{ax.upper()}")
    for side in ("AB", "CD") for ax in "xyz"}

HGP_FACTORS = frozenset(Factor) - {
    Factor.ONE, Factor.PQ_X, Factor.PQ_Y, Factor.PQ_Z, Factor.HALF_INV_ZETA,
    Factor.PA_X, Factor.PA_Y, Factor.PA_Z, Factor.PB_X, Factor.PB_Y,
    Factor.PB_Z}


def _vrr_axis(node) -> int:
    """`hgp.vrr.reduce`'s axis: the first nonzero of f, else of e."""
    e, f, _ = node
    return next(i for i, c in enumerate(f if any(f) else e) if c)


def _hgp_terms():
    """Every term of every built HGP DAG over the 21-class ladder, as an IR
    `Term` -- which raises if the vocabulary cannot say it."""
    from felizjueves.hgp import hrr, vrr

    for cls in vrr.CLASSES:
        for node, ts in vrr.build(cls).terms.items():
            for t in ts:
                yield "vrr", t.coef, Term(t.mult, _VRR_FACTOR[t.coef](
                    _vrr_axis(node)), Var("src"))
        for first in hrr.SIDES:
            for ts in hrr.build(cls, first).terms.values():
                for t in ts:
                    yield "hrr", t.coef, Term(1, _HRR_FACTOR[t.coef], Var("src"))


def test_every_hgp_dag_term_is_a_term_of_the_vocabulary() -> None:
    from felizjueves.hgp import vrr

    # The maps cover exactly the names the DAG modules declare.
    assert set(_VRR_FACTOR) == set(vrr.BRA) | set(vrr.KET)
    from felizjueves.hgp import hrr
    assert set(_HRR_FACTOR) == set(hrr.COEFS)

    used: set[Factor] = set()
    signs: dict[Factor, set[bool]] = {}
    for _, _, t in _hgp_terms():
        used.add(t.g)
        signs.setdefault(t.g, set()).add(t.c < 0)
    # The ladder's DAGs use every new member (no "for later" vocabulary),
    # plus ONE (HRR's transfer term) and PA, reused from the E fragment.
    assert used == HGP_FACTORS | {Factor.ONE, Factor.PA_X, Factor.PA_Y,
                                  Factor.PA_Z}
    # The sign lives in c, and only on the derived (m+1) lowering scalars.
    negative = {g for g, s in signs.items() if True in s}
    assert negative == {Factor.INV2P_ALPHA_OVER_P, Factor.INV2Q_ALPHA_OVER_Q}
    assert all(s == {True} for g, s in signs.items() if g in negative)


def _built_kernels() -> list[Kernel]:
    """One kernel of every family the generator builds today, at a size that
    exercises every factor it can use."""
    from felizjueves.eri.fock.kernel_f import build_f_kernel
    from felizjueves.hgp.recurrence import build_hrr_kernel, build_vrr_kernel
    from felizjueves.kernel_dq import build_dq_kernel
    from felizjueves.kernel_jab import build_jab_kernel
    from felizjueves.kernel_k import build_k_kernel
    from felizjueves.mmd.kernel import build_pq_kernel
    from felizjueves.one_electron.kernel_s import build_s_kernel
    from felizjueves.one_electron.kernel_t import build_t_kernel
    from felizjueves.one_electron.kernel_v import build_v_kernel
    from felizjueves.spec import Operator, Spec, Target

    pppp = Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.F)
    return [build_pq_kernel(2), build_dq_kernel((1, 1)),
            build_jab_kernel((1, 1)), build_s_kernel((1, 1)),
            build_t_kernel((1, 1)), build_v_kernel((1, 1)),
            build_k_kernel((1, 0, 0, 0)),
            # #279: the standalone HGP VRR and HRR kernels, which bind every
            # HGP member as a scalar Param; (pp|pp) reaches each of them.
            build_vrr_kernel(pppp), build_hrr_kernel(pppp),
            # #282: the fused F kernel, which binds the same members as
            # per-quartet / per-pair Decls instead of Params.
            build_f_kernel(pppp)]


# Members no built kernel binds yet, each with the reason. Empty since #279:
# the HGP members (#277) were registered here "awaiting #282", and the
# standalone VRR/HRR kernels of `hgp/recurrence.py` bind them all before the
# fused F kernel exists. A member listed here that a kernel does bind fails
# the test below, so a new unbound member must be added with its reason.
UNBOUND: dict[Factor, str] = {}


def test_every_factor_is_bound_by_a_built_kernel_or_registered_unbound() -> None:
    bound: set[Factor] = set()
    for k in _built_kernels():
        names = ({p.name for p in k.params}
                 | {s.name for s in walk(k.body) if isinstance(s, Decl)}
                 | {b for s in walk(k.body) if isinstance(s, Inline)
                    for b in s.fragment.binds})
        used = {t.g for s in walk(k.body) for root in ir._exprs_of(s)
                if root is not None for e in ir._subexprs(root)
                if isinstance(e, Weighted) for t in e.terms}
        # A factor counts as bound only where a Weighted uses it and a
        # Param, Decl or fragment bind of its name exists.
        bound |= {f for f in used
                  if f.name_in_kernel is None or f.name_in_kernel in names}
    assert bound.isdisjoint(UNBOUND), sorted(f.name for f in bound & set(UNBOUND))
    assert bound | set(UNBOUND) == set(Factor), sorted(
        f.name for f in set(Factor) - bound - set(UNBOUND))


def test_weighted_rejects_empty_and_foreign_terms() -> None:
    with pytest.raises(ValueError):
        Weighted(())
    with pytest.raises(TypeError):
        Weighted((Var("a"),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Term(1, Factor.ONE, Lit(1.0))  # type: ignore[arg-type]


def test_the_scalar_vocabulary_and_the_literal_rule() -> None:
    # #338: F32 joins; a value's precision is declared, never a backend flag.
    assert {s.name for s in Scalar} == {"F64", "F32", "I32"}
    # A literal's type is its Python type, never its context's: a float is
    # F64 even beside an F32, and an F32 constant is a TO_F32 of one.
    assert Lit(1.5).type is Scalar.F64
    assert Lit(3).type is Scalar.I32
    with pytest.raises(TypeError):
        Lit(True)
    assert ir.expr_type(Call(Fn.TO_F32, (Lit(0.5),)), _none) is Scalar.F32
    # ... and the constant is exactly the float32 written.
    with pytest.raises(ir.PrecisionError, match="not a float32"):
        Call(Fn.TO_F32, (Lit(0.1),))
    with pytest.raises(ir.PrecisionError, match="not a float32"):
        Call(Fn.TO_F32, (Lit(1e300),))
    Call(Fn.TO_F32, (Lit(3),))  # an I32 converts, as C converts it


# ---------------------------------------------------------------------------
# Precision hygiene (#338): which operands may mix, and the cast rule
# ---------------------------------------------------------------------------


def _none(name: str) -> None:
    return None


_TYPES = {"f": Scalar.F32, "g": Scalar.F32, "PQ_x": Scalar.F32,
          "D": Scalar.F64, "E": Scalar.F64, "PQ_y": Scalar.F64, "i": Scalar.I32}


def _ty(e: ir.Expr) -> Scalar | None:
    return ir.expr_type(e, lambda n: _TYPES.get(n))


def _bin(fn: Fn, a: str, b: str) -> Call:
    return Call(fn, (Var(a), Var(b)))


@pytest.mark.parametrize("fn", [Fn.ADD, Fn.SUB, Fn.MUL, Fn.DIV, Fn.LT])
def test_arithmetic_types(fn: Fn) -> None:
    out = Scalar.I32 if fn is Fn.LT else None
    assert _ty(_bin(fn, "f", "g")) is (out or Scalar.F32)
    assert _ty(_bin(fn, "D", "E")) is (out or Scalar.F64)
    assert _ty(_bin(fn, "i", "i")) is Scalar.I32
    assert _ty(_bin(fn, "f", "i")) is (out or Scalar.F32)  # the int converts
    assert _ty(_bin(fn, "i", "D")) is (out or Scalar.F64)
    for a, b in (("f", "D"), ("D", "f")):
        with pytest.raises(ir.PrecisionError, match="mixes F32 with F64"):
            _ty(_bin(fn, a, b))
    # a float literal is F64, so it does not mix with an F32 silently either
    with pytest.raises(ir.PrecisionError):
        _ty(Call(fn, (Var("f"), Lit(2.0))))
    assert _ty(Call(fn, (Var("f"), Lit(2)))) is (out or Scalar.F32)


def test_the_cast_rule() -> None:
    to32 = Call(Fn.TO_F32, (Var("D"),))
    to64 = Call(Fn.TO_F64, (Var("f"),))
    assert _ty(to32) is Scalar.F32 and _ty(to64) is Scalar.F64
    assert _ty(Call(Fn.ADD, (Var("f"), to32))) is Scalar.F32
    assert _ty(Call(Fn.ADD, (Var("D"), to64))) is Scalar.F64
    assert _ty(Call(Fn.TO_F64, (Var("i"),))) is Scalar.F64
    # a cast says a precision changes; a cast to the value's own type is noise
    with pytest.raises(ir.PrecisionError, match="already F32"):
        _ty(Call(Fn.TO_F32, (Var("f"),)))
    with pytest.raises(ir.PrecisionError, match="already F64"):
        _ty(Call(Fn.TO_F64, (Var("D"),)))
    # an unknown name hides the answer; no rule fires on it
    assert _ty(Call(Fn.TO_F32, (Var("raw_name"),))) is Scalar.F32
    assert _ty(Call(Fn.ADD, (Var("f"), Var("raw_name")))) is None


def test_special_functions_keep_or_refuse_the_type() -> None:
    for fn in (Fn.SQRT, Fn.EXP):
        assert _ty(Call(fn, (Var("f"),))) is Scalar.F32
        assert _ty(Call(fn, (Var("D"),))) is Scalar.F64
        assert _ty(Call(fn, (Var("i"),))) is Scalar.F64
    # Boys is a statement now (#301), fp64 only: ir.Kernel refuses an F32 T
    out = Param("out", Scalar.F64, array=True, intent=Intent.OUT)
    for t, ok in ((Scalar.F64, True), (Scalar.F32, False)):
        body = (Decl("boys", Scalar.F64, shape=(Lit(3),)),
                Boys("boys", 2, Var("T")),
                Assign(Index("out", (Lit(0),)), Index("boys", (Lit(2),))))
        params = (Param("T", t), out)
        if ok:
            Kernel("k", params, body)
        else:
            with pytest.raises(ir.PrecisionError, match="no fp32 Boys"):
                Kernel("k", params, body)


def test_a_weighted_sum_has_one_float_type() -> None:
    f32 = Weighted((Term(2, Factor.PQ_X, Var("f")), Term(1, Factor.ONE, Var("g"))))
    f64 = Weighted((Term(2, Factor.PQ_Y, Var("D")), Term(1, Factor.ONE, Var("E"))))
    assert _ty(f32) is Scalar.F32 and _ty(f64) is Scalar.F64
    for terms in ((Term(1, Factor.PQ_X, Var("D")),),          # F32 factor, F64 src
                  (Term(1, Factor.PQ_Y, Var("f")),),          # F64 factor, F32 src
                  (Term(1, Factor.ONE, Var("f")), Term(1, Factor.ONE, Var("D")))):
        with pytest.raises(ir.PrecisionError, match="Weighted"):
            _ty(Weighted(terms))


def _k(*body: ir.Stmt) -> Kernel:
    return Kernel("k", (Param("f", Scalar.F32), Param("D", Scalar.F64),
                        Param("i", Scalar.I32),
                        Param("o32", Scalar.F32, array=True, intent=Intent.OUT),
                        Param("o64", Scalar.F64, array=True, intent=Intent.OUT)),
                  body)


O32, O64 = Index("o32", (Lit(0),)), Index("o64", (Lit(0),))


def test_a_kernel_refuses_every_silent_store() -> None:
    # stores that keep or widen-from-int are fine
    _k(Assign(O32, Var("f")), Assign(O32, Var("i")), Assign(O64, Var("D")),
       Assign(O64, Var("i")), Assign(O64, Call(Fn.TO_F64, (Var("f"),))),
       Decl("x", Scalar.F32, init=Call(Fn.TO_F32, (Lit(1.5),))),
       AtomicAdd(O32, Var("f")), WarpReduce(O64, Var("D")))
    bad = [
        Assign(O64, Var("f")),                       # F32 -> F64, no cast
        Assign(O32, Var("D")),                       # F64 -> F32, no cast
        Assign(O32, Lit(1.5)),                       # a float literal is F64
        Decl("x", Scalar.F32, init=Var("D")),
        Decl("x", Scalar.F64, init=Var("f")),
        Decl("x", Scalar.I32, init=Var("f")),        # F32 -> I32
        AtomicAdd(O64, Var("f")),
        WarpReduce(O32, Var("D")),
        Assign(O32, Call(Fn.ADD, (Var("f"), Var("D")))),  # mixed operands
        For("q", Lit(0), Var("f"), Lit(1), ()),      # an F32 loop bound
        If(Call(Fn.LT, (Var("f"), Var("D"))), ()),
    ]
    for s in bad:
        with pytest.raises(ir.PrecisionError):
            _k(s)


def test_scopes_and_fragments_are_typed() -> None:
    # A Decl's type holds in its scope and shadows outside it.
    _k(Decl("x", Scalar.F32, init=Var("f")),
       If(Lit(1), (Decl("x", Scalar.F64, init=Var("D")), Assign(O64, Var("x")))),
       Assign(O32, Var("x")))
    with pytest.raises(ir.PrecisionError):
        _k(Decl("x", Scalar.F32, init=Var("f")),
           If(Lit(1), (Decl("y", Scalar.F64, init=Var("D")),)),
           Assign(O64, Var("x")))
    # A fragment's scalar bind takes its binding's type; checked alone, a bind
    # is unknown and only the fragment's own locals are held to the rules.
    frag = Fragment("fr", ("u", "out"), (
        Decl("w", Scalar.F32, init=Var("u")),
        Assign(Index("out", (Lit(0),)), Var("w"))))
    _k(Inline(frag, {"u": Var("f"), "out": Var("o32")}))
    with pytest.raises(ir.PrecisionError):
        _k(Inline(frag, {"u": Var("D"), "out": Var("o32")}))   # F64 bind
    with pytest.raises(ir.PrecisionError):
        _k(Inline(frag, {"u": Var("f"), "out": Var("o64")}))   # F64 array
    with pytest.raises(ir.PrecisionError):
        Fragment("bad", ("u",), (Decl("w", Scalar.F32, init=Var("u")),
                                 Decl("z", Scalar.F64, init=Var("w"))))


def test_no_built_fp64_kernel_declares_f32() -> None:
    # Every refusal involves F32, so the hygiene check can refuse no kernel
    # that declares none -- and none of the built kernels declares one.
    for k in _built_kernels():
        declared = {p.type for p in k.params} | {
            s.type for s in walk(k.body) if isinstance(s, Decl)}
        assert Scalar.F32 not in declared, k.name


def test_call_vocabulary_and_arity() -> None:
    assert Call(Fn.THREAD_X).args == ()
    Call(Fn.EXP, (Var("x"),))  # #152: E^{00}_0's seed
    with pytest.raises(AttributeError):
        Fn.BOYS  # noqa: B018 -- a statement since #301, never a Call
    with pytest.raises(ValueError, match="takes 1 args"):
        Call(Fn.EXP, (Var("x"), Var("y")))
    with pytest.raises(ValueError, match="takes 2 args"):
        Call(Fn.ADD, (Lit(1),))
    with pytest.raises(ValueError, match="Fn vocabulary"):
        Call("pow", (Lit(1), Lit(2)))  # type: ignore[arg-type]


def test_for_bounds_are_exprs() -> None:
    f = For("q", Call(Fn.THREAD_Y), Var("n"), Call(Fn.BLOCK_DIM_Y), [])
    assert isinstance(f.start, Call) and f.body == ()
    with pytest.raises(TypeError, match="For stop must be an Expr"):
        For("q", Lit(0), 10, Lit(1), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="For start must be an Expr"):
        For("q", "threadIdx.y", Lit(10), Lit(1), ())  # type: ignore[arg-type]


def test_identifiers_are_checked() -> None:
    for bad in ("threadIdx.x", "a b", "", "for", "1a"):
        with pytest.raises(ValueError):
            Var(bad)


def test_bodies_hold_only_stmts() -> None:
    with pytest.raises(TypeError, match="not a Stmt"):
        If(Lit(1), (Var("x"),))  # type: ignore[arg-type]


def test_out_params_are_arrays() -> None:
    with pytest.raises(ValueError, match="must be an array"):
        Param("J", Scalar.F64, intent=Intent.OUT)


def test_array_decls_take_no_init() -> None:
    with pytest.raises(ValueError):
        Decl("s", Scalar.F64, (Lit(32),), Space.SHARED, init=Lit(0.0))


# ---------------------------------------------------------------------------
# The [p|q] contract (#25) is expressible
# ---------------------------------------------------------------------------


def _add(a: ir.Expr, b: ir.Expr) -> Call:
    return Call(Fn.ADD, (a, b))


def _mul(a: ir.Expr, b: ir.Expr) -> Call:
    return Call(Fn.MUL, (a, b))


def contract_shaped_kernel() -> Kernel:
    """Every construct `[JEngine, v2, Alg. 1]` needs, in one (toy) kernel.

    Not the #25 kernel -- the recurrence here is one instruction -- but every
    node that kernel needs appears, and nothing else does.
    """
    bra = _add(Call(Fn.THREAD_X), _mul(Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))
    ket_loop = For(
        "q",
        Call(Fn.THREAD_Y),
        Var("n_kets"),
        Call(Fn.BLOCK_DIM_Y),
        (
            Note("kets sorted by I_q * D_max descending: the break is exact"),
            Decl(
                "bound",
                Scalar.F64,
                init=_mul(Var("I_p"), Index("I_q", (Var("q"),))),
            ),
            If(Call(Fn.LT, (Var("bound"), Var("tau"))), (Break(),)),
            Decl("PQ_x", Scalar.F64, init=Index("PQx", (Var("q"),))),
            Decl(
                "T",
                Scalar.F64,
                init=_mul(Var("PQ_x"), Var("PQ_x")),
            ),
            Decl("boys", Scalar.F64, shape=(Lit(2),)),
            Boys("boys", 1, Var("T")),
            Decl("F1", Scalar.F64, init=Index("boys", (Lit(1),))),
            Note("[JEngine, v2, eq. 11]"),
            Decl(
                "pq",
                Scalar.F64,
                init=Weighted((Term(1, Factor.PQ_X, Var("F1")),)),
            ),
            Assign(
                Var("acc"),
                _add(Var("acc"), _mul(Var("pq"), Index("D", (Var("q"),)))),
            ),
        ),
    )
    return Kernel(
        "pq_contract",
        (
            Param("n_kets", Scalar.I32),
            Param("tau", Scalar.F64),
            Param("I_p", Scalar.F64),
            Param("I_q", Scalar.F64, array=True),
            Param("PQx", Scalar.F64, array=True),
            Param("D", Scalar.F64, array=True),
            Param("J", Scalar.F64, array=True, intent=Intent.OUT),
        ),
        (
            Note("[JEngine, v2, Alg. 1]"),
            Decl("p", Scalar.I32, init=bra),
            Decl("Js", Scalar.F64, (Lit(32),), Space.SHARED),
            Decl("acc", Scalar.F64, init=Lit(0.0)),
            ket_loop,
            Assign(Index("Js", (Call(Fn.THREAD_Y),)), Var("acc")),
            WarpReduce(Var("acc"), Index("Js", (Call(Fn.THREAD_Y),))),
            Note("plain write: single batch; [JEngine, v2, §5] adds the atomic"),
            If(
                Call(Fn.LT, (Call(Fn.THREAD_Y), Lit(1))),
                (Assign(Index("J", (Var("p"),)), Var("acc")),),
            ),
        ),
    )


def test_every_contract_construct_is_expressible() -> None:
    k = contract_shaped_kernel()
    kinds = {type(s) for s in walk(k.body)}
    # Everything but AtomicAdd (milestone 3) and Raw (counted, and zero).
    assert kinds == {Decl, Assign, For, If, Break, WarpReduce, Note, Boys}
    shared = [
        s for s in walk(k.body) if isinstance(s, Decl) and s.space is Space.SHARED
    ]
    assert len(shared) == 1


def test_raw_and_note_are_marked_and_nonempty() -> None:
    # Raw is legal -- it is counted, not banned -- but never empty or silent.
    assert Kernel("k", (), (Raw("__nanosleep(1);"),)).body[0].text
    for node in (Raw, Note):
        with pytest.raises(ValueError):
            node("")


def test_atomic_add_targets_an_element() -> None:
    AtomicAdd(Index("J", (Var("p"),)), Var("acc"))
    with pytest.raises(TypeError):
        AtomicAdd(Var("J"), Var("acc"))  # type: ignore[arg-type]


def test_the_ir_knows_no_vendor() -> None:
    # The wavefront width and vendor spellings live on the target (#37) and in
    # the backends; no vocabulary member or node here names them.
    names = {
        m.name.lower()
        for enum in (Scalar, Factor, Fn, Space, Intent)
        for m in enum  # type: ignore[attr-defined]
    }
    names |= {f.name for t in ir.STMT_TYPES for f in dataclasses.fields(t)}
    for word in ("warp_size", "wavefront", "lane", "cuda", "hip", "shfl", "sm_", "gfx"):
        assert not any(word in n for n in names), word


# ---------------------------------------------------------------------------
# Kernel validation
# ---------------------------------------------------------------------------


def test_break_outside_a_loop_raises() -> None:
    with pytest.raises(ValueError, match="Break outside a For"):
        Kernel("k", (), (If(Lit(1), (Break(),)),))


def test_an_unbound_factor_raises() -> None:
    body = (
        Decl("a", Scalar.F64, init=Lit(1.0)),
        Decl("b", Scalar.F64, init=Weighted((Term(1, Factor.PQ_Y, Var("a")),))),
    )
    with pytest.raises(ValueError, match="'PQ_y'"):
        Kernel("k", (), body)
    Kernel("k", (Param("PQ_y", Scalar.F64),), body)  # bound by a Param: fine


def test_duplicate_params_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Kernel("k", (Param("a", Scalar.F64), Param("a", Scalar.I32)), ())


# ---------------------------------------------------------------------------
# Frozen, hashable, structurally comparable
# ---------------------------------------------------------------------------


def test_a_kernel_is_frozen_and_hashable() -> None:
    k = contract_shaped_kernel()
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.name = "other"  # type: ignore[misc]
    assert hash(k) == hash(contract_shaped_kernel())
    assert {k: 1}[contract_shaped_kernel()] == 1


def test_structural_equality_is_the_check_for_a_pass() -> None:
    a, b = contract_shaped_kernel(), contract_shaped_kernel()
    assert a == b and a is not b
    # Lists are frozen to tuples, so spelling does not change identity.
    assert For("i", Lit(0), Lit(1), Lit(1), [Break()]) == For(
        "i", Lit(0), Lit(1), Lit(1), (Break(),)
    )
    # A Kernel -> Kernel identity round-trips to an equal kernel...
    assert dataclasses.replace(a, body=tuple(a.body)) == a
    # ...and dropping a statement (even an inert Note) is a different kernel.
    stripped = dataclasses.replace(
        a, body=tuple(s for s in a.body if not isinstance(s, Note))
    )
    assert stripped != a
    assert Lit(1) != Lit(1.0)  # an I32 is not an F64
    assert Term(1, Factor.ONE, Var("x")) != Term(2, Factor.ONE, Var("x"))


# ---------------------------------------------------------------------------
# Fragment and Inline (#149): a device-code block with a checked contract
# ---------------------------------------------------------------------------


def synthetic_fragment() -> Fragment:
    """`dst[i] = PQ_x * (src[0] + ... + src[i])` for i < n.

    Synthetic on purpose (no `E` yet): one scalar bind read as a `Factor`
    (`PQ_x`), one read as a loop bound (`n`), an array read (`src`), an array
    written (`dst`, the fragment's only way out), a local, and a loop var.
    """
    return Fragment("prefix_scale", ("PQ_x", "n", "src", "dst"), (
        Note("synthetic: dst[i] = PQ_x * (src[0] + ... + src[i])"),
        Decl("acc", Scalar.F64, init=Lit(0.0)),
        For("i", Lit(0), Var("n"), Lit(1), (
            Assign(Var("acc"), Weighted((
                Term(1, Factor.ONE, Var("acc")),
                Term(1, Factor.ONE, Index("src", (Var("i"),))),
            ))),
            Assign(Index("dst", (Var("i"),)),
                   Weighted((Term(1, Factor.PQ_X, Var("acc")),))),
        )),
    ))


def synthetic_inline_kernel(fragment: Fragment | None = None) -> Kernel:
    """A host kernel that splices `synthetic_fragment` once.

    `n` is bound to the host's own `n` (the one admitted name-sharing, and so
    no `const` line); `PQ_x` to an expression; the arrays to the host's. The
    host writes `out[0]` again after the splice, so the splice is not last.
    """
    frag = synthetic_fragment() if fragment is None else fragment
    n = Var("n")
    return Kernel("inline_probe", (
        Param("n", Scalar.I32),
        Param("scale", Scalar.F64),
        Param("xs", Scalar.F64, array=True, shape=(n,)),
        Param("out", Scalar.F64, array=True, intent=Intent.OUT, shape=(n,)),
    ), (
        Decl("half", Scalar.F64, init=_mul(Var("scale"), Lit(0.5))),
        Inline(frag, {"PQ_x": _add(Var("half"), Var("half")), "n": n,
                      "src": Var("xs"), "dst": Var("out")}),
        Assign(Index("out", (Lit(0),)),
               _add(Index("out", (Lit(0),)), Var("half"))),
    ))


def _inline(frag: Fragment, **over: ir.Expr) -> Inline:
    b = {"PQ_x": Var("scale"), "n": Var("n"), "src": Var("xs"),
         "dst": Var("out")}
    b.update(over)
    return Inline(frag, b)


def test_the_synthetic_kernel_builds() -> None:
    k = synthetic_inline_kernel()
    frag = synthetic_fragment()
    assert frag.array_binds == {"src", "dst"}
    assert frag.declared == {"acc", "i"}
    assert k == synthetic_inline_kernel() and hash(k) == hash(
        synthetic_inline_kernel())


def test_a_free_name_raises() -> None:
    with pytest.raises(ValueError, match=r"reads \['b'\] without"):
        Fragment("f", ("a",), (
            Decl("x", Scalar.F64, init=_add(Var("a"), Var("b"))),))
    # A Factor's scalar is a read too, and an indexed array.
    with pytest.raises(ValueError, match=r"reads \['PQ_y'\]"):
        Fragment("f", ("a",), (Decl("x", Scalar.F64, init=Weighted((
            Term(1, Factor.PQ_Y, Var("a")),))),))
    with pytest.raises(ValueError, match=r"reads \['v'\]"):
        Fragment("f", ("a",), (Decl("x", Scalar.F64, init=_add(
            Var("a"), Index("v", (Lit(0),)))),))


def test_a_name_used_before_its_decl_is_free() -> None:
    # Scope is source order: the read precedes the Decl, so `x` is free here.
    with pytest.raises(ValueError, match=r"reads \['x'\]"):
        Fragment("f", ("a",), (
            Decl("y", Scalar.F64, init=_add(Var("a"), Var("x"))),
            Decl("x", Scalar.F64, init=Lit(1.0)),
        ))
    # And a For var is not visible after its loop.
    with pytest.raises(ValueError, match=r"reads \['i'\]"):
        Fragment("f", ("a",), (
            For("i", Lit(0), Var("a"), Lit(1), ()),
            Decl("y", Scalar.I32, init=Var("i")),
        ))


def test_an_unused_bind_raises() -> None:
    with pytest.raises(ValueError, match="stale contract"):
        Fragment("f", ("a", "unused"), (Decl("x", Scalar.F64, init=Var("a")),))


def test_a_scalar_bind_is_read_only() -> None:
    with pytest.raises(ValueError, match="read-only"):
        Fragment("f", ("a",), (Assign(Var("a"), _add(Var("a"), Lit(1.0))),))


def test_binds_are_unique_identifiers() -> None:
    with pytest.raises(ValueError, match="duplicate binds"):
        Fragment("f", ("a", "a"), (Decl("x", Scalar.F64, init=Var("a")),))
    with pytest.raises(ValueError, match="not an identifier"):
        Fragment("f", ("a.b",), ())
    with pytest.raises(ValueError, match="both as a scalar and as an array"):
        Fragment("f", ("a",), (Decl("x", Scalar.F64, init=_add(
            Var("a"), Index("a", (Lit(0),)))),))
    with pytest.raises(ValueError, match="also binds"):
        Fragment("f", ("a",), (Decl("x", Scalar.F64, init=Var("a")),
                               Decl("a", Scalar.F64, init=Lit(1.0))))


def test_one_level_names_only() -> None:
    inner = Fragment("g", ("a",), (Decl("x", Scalar.F64, init=Var("a")),))
    with pytest.raises(ValueError, match="cannot hold an Inline"):
        Fragment("f", ("b",), (Inline(inner, {"a": Var("b")}),))
    with pytest.raises(ValueError, match="cannot hold a Raw"):
        Fragment("f", (), (Raw("__nanosleep(1);"),))
    with pytest.raises(ValueError, match="Break outside a For"):
        Fragment("f", (), (Break(),))


def test_an_inline_with_a_missing_or_extra_binding_raises() -> None:
    frag = synthetic_fragment()
    full = {"PQ_x": Var("scale"), "n": Var("n"), "src": Var("xs"),
            "dst": Var("out")}
    assert Inline(frag, full) == Inline(frag, list(reversed(full.items())))
    with pytest.raises(ValueError, match=r"missing bindings \['n'\]"):
        Inline(frag, {k: v for k, v in full.items() if k != "n"})
    with pytest.raises(ValueError, match=r"extra \['zeta'\]"):
        Inline(frag, {**full, "zeta": Lit(1.0)})
    with pytest.raises(ValueError, match="twice"):
        Inline(frag, [*full.items(), ("n", Lit(3))])
    with pytest.raises(TypeError, match="must be an Expr"):
        Inline(frag, {**full, "n": 3})
    with pytest.raises(TypeError, match="array bind 'src'"):
        Inline(frag, {**full, "src": Index("xs", (Lit(0),))})


def _host(*body: ir.Stmt, extra: tuple[Param, ...] = ()) -> Kernel:
    return Kernel("k", (
        Param("n", Scalar.I32),
        Param("scale", Scalar.F64),
        Param("xs", Scalar.F64, array=True),
        Param("out", Scalar.F64, array=True, intent=Intent.OUT),
        *extra,
    ), body)


def test_a_colliding_name_raises_and_nothing_is_renamed() -> None:
    frag = synthetic_fragment()
    _host(_inline(frag))  # no collision: fine
    with pytest.raises(ValueError, match=r"declares \['acc'\].*not a macro"):
        _host(Decl("acc", Scalar.F64, init=Lit(0.0)), _inline(frag))
    with pytest.raises(ValueError, match=r"declares \['i'\]"):
        _host(For("i", Lit(0), Lit(1), Lit(1), ()), _inline(frag))
    with pytest.raises(ValueError, match=r"declares \['acc'\]"):
        _host(_inline(frag), extra=(Param("acc", Scalar.F64),))
    # A collision after the splice, or nested in the host, is still one.
    with pytest.raises(ValueError, match=r"declares \['acc'\]"):
        _host(_inline(frag), If(Lit(1), (Decl("acc", Scalar.I32, init=Lit(0)),)))


def test_a_bind_may_share_a_host_name_only_as_itself() -> None:
    frag = synthetic_fragment()
    _host(_inline(frag, n=Var("n")))  # the host's own n: admitted
    with pytest.raises(ValueError, match=r"binds 'n', a host name"):
        _host(_inline(frag, n=_add(Var("n"), Lit(1))))
    with pytest.raises(ValueError, match=r"binds 'src', a host name"):
        _host(_inline(frag, n=Var("n")), extra=(
            Param("src", Scalar.F64, array=True),))


def test_array_and_scalar_binds_match_the_host() -> None:
    frag = synthetic_fragment()
    with pytest.raises(ValueError, match="not a host array"):
        _host(_inline(frag, src=Var("scale")))
    with pytest.raises(ValueError, match="bound to the host array"):
        _host(_inline(frag, PQ_x=Var("xs")))
    # A host array Decl is a host array.
    _host(Decl("buf", Scalar.F64, (Lit(4),)), _inline(frag, dst=Var("buf")))


def test_a_factor_bound_by_a_bind_satisfies_the_kernel() -> None:
    # PQ_x is neither a Param nor a Decl of the host: the bind binds it.
    k = synthetic_inline_kernel()
    assert "PQ_x" not in {p.name for p in k.params}


def test_walk_sees_through_the_splice() -> None:
    frag = synthetic_fragment()
    k = synthetic_inline_kernel(frag)
    inside = set(map(id, walk(frag.body)))
    seen = list(walk(k.body))
    assert any(isinstance(s, Inline) for s in seen)
    assert inside <= set(map(id, seen))  # the very statements, not copies
    assert any(isinstance(s, Decl) and s.name == "acc" for s in seen)
    hidden = list(walk(k.body, into_inline=False))
    assert not inside & set(map(id, hidden))
    # The expression census sees the bindings.
    exprs = [e for s in seen for root in ir._exprs_of(s) if root is not None
             for e in ir._subexprs(root)]
    assert Var("half") in exprs


def test_the_raw_census_is_unchanged_by_fragments() -> None:
    # Fragment and Inline add no escape hatch: a Fragment refuses Raw, and
    # src/ constructs none (the census above is still zero).
    assert count_raw_in_src() == 0
    assert not any(isinstance(s, Raw) for s in walk(synthetic_inline_kernel().body))


def test_the_raw_census_is_unchanged_by_the_k_kernel() -> None:
    # #198: the exchange kernel inlines two fragments and moves the census by
    # zero -- src/ still constructs no Raw, and its built Kernel (at L <= 2,
    # milestone 10's rule) holds none, through either splice.
    from felizjueves.kernel_k import build_k_kernel

    assert (SRC / "felizjueves" / "kernel_k.py").exists()
    assert count_raw_in_src() == 0
    for am in [(0, 0, 0, 0), (1, 0, 0, 1), (1, 1, 0, 0)]:
        assert not any(isinstance(s, Raw)
                       for s in walk(build_k_kernel(am).body))


# ---------------------------------------------------------------------------
# LaneBroadcast (#349): the indexed shuffle, validated as WarpReduce is
# ---------------------------------------------------------------------------


def _bcast_kernel(*body, extra=()) -> Kernel:
    return Kernel("b", (Param("out", Scalar.F64, array=True, intent=Intent.OUT),
                        *extra), tuple(body))


def test_lane_broadcast_validates_its_operands() -> None:
    LaneBroadcast = ir.LaneBroadcast
    s = LaneBroadcast(Var("x"), Var("v"), Lit(0))
    assert (s.dst, s.value, s.source) == (Var("x"), Var("v"), Lit(0))
    assert LaneBroadcast in ir.STMT_TYPES
    with pytest.raises(TypeError, match="LaneBroadcast dst"):
        LaneBroadcast(Lit(1.0), Var("v"), Lit(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LaneBroadcast value"):
        LaneBroadcast(Var("x"), "v", Lit(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LaneBroadcast source"):
        LaneBroadcast(Var("x"), Var("v"), 0)  # type: ignore[arg-type]
    # a statement like any other: frozen, hashable, a Stmt a body may hold
    assert hash(s) == hash(LaneBroadcast(Var("x"), Var("v"), Lit(0)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.source = Lit(1)  # type: ignore[misc]


def test_lane_broadcast_obeys_the_precision_rules() -> None:
    LaneBroadcast = ir.LaneBroadcast
    ty = Call(Fn.THREAD_Y, ())
    ok = _bcast_kernel(Decl("x", Scalar.F64),
                       LaneBroadcast(Var("x"), Lit(1.5), ty),
                       Assign(Index("out", (ty,)), Var("x")))
    assert any(isinstance(s, LaneBroadcast) for s in walk(ok.body))
    # the source is a lane index: I32, never a float
    with pytest.raises(ir.PrecisionError, match="its lane is F64"):
        _bcast_kernel(Decl("x", Scalar.F64),
                      LaneBroadcast(Var("x"), Lit(1.5), Lit(0.0)))
    # the value moves unchanged: F32 into an F64 dst is refused, as a store
    with pytest.raises(ir.PrecisionError, match="LaneBroadcast to"):
        _bcast_kernel(Decl("x", Scalar.F64), LaneBroadcast(
            Var("x"), Call(Fn.TO_F32, (Lit(1.5),)), Lit(0)))


def test_lane_broadcast_reads_its_source_in_a_fragment() -> None:
    LaneBroadcast = ir.LaneBroadcast
    body = (Decl("x", Scalar.F64),
            LaneBroadcast(Var("x"), Var("v"), Var("src")))
    frag = Fragment("bc", ("v", "src"), body)
    assert set(frag.binds) == {"v", "src"}
    with pytest.raises(ValueError, match="src"):  # the source is a read
        Fragment("bc", ("v",), body)
    # the source expression is one of the statement's expressions
    s = LaneBroadcast(Var("x"), Var("v"), Var("src"))
    assert ir._exprs_of(s) == (Var("x"), Var("v"), Var("src"))
