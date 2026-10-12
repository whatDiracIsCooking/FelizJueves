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
        "HALF_INV_ZETA", "PA_X", "PA_Y", "PA_Z", "PB_X", "PB_Y", "PB_Z"]
    assert [Factor.pq(d) for d in range(3)] == [Factor.PQ_X, Factor.PQ_Y, Factor.PQ_Z]
    assert [Factor.pa(d) for d in range(3)] == [Factor.PA_X, Factor.PA_Y, Factor.PA_Z]
    assert [Factor.pb(d) for d in range(3)] == [Factor.PB_X, Factor.PB_Y, Factor.PB_Z]
    assert Factor.ONE.name_in_kernel is None
    assert Factor.PQ_Y.name_in_kernel == "PQ_y"
    assert Factor.HALF_INV_ZETA.name_in_kernel == "half_inv_zeta"
    # Every bound name is an identifier: it is spelled as a C name.
    for f in Factor:
        if f.name_in_kernel is not None:
            ir._ident(f.name_in_kernel, "Factor name")


@pytest.mark.parametrize("g", ["PQ_x", "1", "theta", 1, None])
def test_a_factor_outside_the_vocabulary_raises(g: object) -> None:
    # A string that *spells* a member is still not one: that is Raw in disguise.
    with pytest.raises(ValueError, match="Factor vocabulary"):
        Term(1, g, Var("a"))  # type: ignore[arg-type]


@pytest.mark.parametrize("c", [0, -1, 1.0, True])
def test_a_coefficient_is_a_positive_int(c: object) -> None:
    with pytest.raises(ValueError, match="positive int"):
        Term(c, Factor.ONE, Var("a"))  # type: ignore[arg-type]


def test_weighted_rejects_empty_and_foreign_terms() -> None:
    with pytest.raises(ValueError):
        Weighted(())
    with pytest.raises(TypeError):
        Weighted((Var("a"),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Term(1, Factor.ONE, Lit(1.0))  # type: ignore[arg-type]


def test_the_scalar_type_is_fp64() -> None:
    # fp64 is an IR fact, not a backend flag; there is no fp32 member.
    assert {s.name for s in Scalar} == {"F64", "I32"}
    assert Lit(1.5).type is Scalar.F64
    assert Lit(3).type is Scalar.I32
    with pytest.raises(TypeError):
        Lit(True)


def test_call_vocabulary_and_arity() -> None:
    assert Call(Fn.THREAD_X).args == ()
    Call(Fn.BOYS, (Lit(3), Var("T")))
    Call(Fn.EXP, (Var("x"),))  # #152: E^{00}_0's seed
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
            Decl("F1", Scalar.F64, init=Call(Fn.BOYS, (Lit(1), Var("T")))),
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
    assert kinds == {Decl, Assign, For, If, Break, WarpReduce, Note}
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
