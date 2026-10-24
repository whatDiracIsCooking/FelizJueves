"""Struct-typed pair-record reads, `eri/fock/levers/struct_reads.py` (#353), host side.

The lever reads every packed pair record of the fused F kernel through a
struct view of its row (`ir.Member` over an `ir.Record`) instead of a
positional index into the flat array. What is asserted, the issue's
acceptance in order:

1. THE DECLARATION: `struct-reads` is legal on `Target.F` only (its own
   reasons on J and K), registered in `f_levers.REGISTRY`, moving the
   `field` hook only, refusing to compose over another lever's `field`.
2. THE LAYOUT IS DECLARED ONCE AND HELD TO THE PACKING: `struct_reads.VIEWS`
   against `kernel_f.pack` (the F side's packing; `eri/j_coulomb/staged.py` packs the
   J launch plan's per-field arrays and has no F record), member by member
   and value by value, the member names derived independently from
   `kernel_f.PrimRec` / `PairRec`'s fields -- and a drifted declaration (two
   members swapped, a member dropped) FAILS that check, so a drift is a red
   test, never a corrupted kernel. The writer's `static_assert` holds the C
   side to the same row (part 5).
3. THE IR IS THE LITERAL KERNEL'S, RETYPED: with every `Member` turned back
   into the `Index` it is, the lever's kernel equals the literal one (but
   for its provenance `Note`), at every class with L <= 4, with and without
   the tap; every record read is a `Member` (no positional read of a viewed
   array is left), and each member is the one the literal offset named.
4. PARITY, interpreted ("typed and untyped builds parity on the host
   shim"): the lever's `F` and tap `g` are the literal kernel's BIT FOR BIT
   at every class with L <= 4, one launch against `eri.fock.fock`'s sweep to
   `test_kernel_f.TOL`, and every degeneracy case within the cap.
5. THE SPELLING (`emit/_c.py`, shared by both backends; additive): one
   `struct fj_<record>` per record a kernel reads, in first-use order, each
   with its size `static_assert`; the read spelled through the view; the
   writer's refusals (a store through a view, a width or a type that is not
   the array's); and no literal F kernel through L <= 4 carries any of it
   ("untyped byte-identical": test_f_seams.py's digests and
   test_emit_fock.py's goldens are the pins, unchanged).
6. THE GOLDENS: `(sp|sp)` with the lever, one unit per backend.

Classes above L = 4 are `slow` (milestone 13's cap) and not run under it.
Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_f as tk
from tiers import split_by_l

from felizjueves import levers
from felizjueves.ao.shell import Basis
from felizjueves.emit import cuda, hip
from felizjueves.emit._c import EmitError, record_struct
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import levers as f_levers
from felizjueves.eri.fock.kernel_f import build_f_kernel, f_class_spec
from felizjueves.eri.fock.levers import seams
from felizjueves.eri.fock.levers import struct_reads as sr
from felizjueves.eri.fock.levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import vrr
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Index,
    Intent,
    Kernel,
    Lit,
    Member,
    Note,
    Param,
    Record,
    Scalar,
    Var,
    Weighted,
    _exprs_of,
    _subexprs,
    walk,
)
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Operator, Spec, Target

NAME = sr.NAME
CAP = 4  # milestone 13: every test at L <= 4

LADDER = split_by_l(vrr.CLASSES, sum, ids=lambda c: "".join(map(str, c)))
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]


def lever_spec(cls) -> Spec:
    return f_class_spec(*cls, (NAME,))


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_struct_reads_is_declared_legal_on_f_only_with_its_own_reasons(
) -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 353,
                                                      frozenset({"f"}))
    assert "[p|q]" in lever.refusals["j"] and "field hook" in lever.refusals[
        "j"]
    assert "K kernel" in lever.refusals["k"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="per-field record arrays"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="k_levers.seams"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def test_struct_reads_is_registered_keyed_and_moves_field_only() -> None:
    assert f_levers.REGISTRY[NAME] is sr.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/struct-reads"
    assert spec.kernel_body_levers() == (NAME,)
    moved = {f.name for f in dataclasses.fields(FHooks)
             if not f_levers.hooks_for(spec).is_default(f.name)}
    assert moved == {"field"}
    assert f_levers.hooks_for(f_class_spec(0, 1, 0, 1)) is DEFAULT_HOOKS
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    assert levers.provenance(NAME) in [s.prov for s in walk(kern.body)
                                       if isinstance(s, Note)]


def test_it_refuses_to_compose_over_another_field() -> None:
    other = dataclasses.replace(
        DEFAULT_HOOKS, field=lambda ctx, arr, idx: Index(arr, idx))
    with pytest.raises(ValueError, match="two readers"):
        sr.apply(other, lever_spec((0, 1, 0, 1)))


def test_an_undeclared_record_read_is_refused_not_read_untyped() -> None:
    with pytest.raises(ValueError, match="no declared view"):
        sr.typed_field(None, "bra_extra", (Var("bra"), Lit(0)))


# ---------------------------------------------------------------------------
# 2. The layout, declared once, held to kernel_f.pack
# ---------------------------------------------------------------------------

_AXES = ("x", "y", "z")


def _expand(obj) -> list[tuple[str, object]]:
    """`(name, value)` per scalar of a frozen dataclass, a 3-vector field
    `P` spelled `P_x P_y P_z`, in field order -- the packing's row order
    (`PrimRec.row`), derived from the dataclass, not from `seams.REC`."""
    out = []
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, tuple):
            out += [(f"{f.name}_{a}", x) for a, x in zip(_AXES, v,
                                                        strict=True)]
        else:
            out.append((f.name, v))
    return out


def _pairs() -> list[KF.PairRec]:
    """Two pairs of degree 3, every scalar distinct."""
    rng = np.random.default_rng(353)

    def vec():
        return tuple(float(x) for x in rng.uniform(-2.0, 2.0, 3))

    out = []
    for n in range(2):
        prims = tuple(KF.PrimRec(vec(), vec(), float(rng.uniform(0.1, 9.0)),
                                 float(rng.uniform(0.1, 9.0)))
                      for _ in range(3))
        out.append(KF.PairRec(vec(), float(rng.uniform(0.1, 1.0)), prims,
                              (2 * n + 1, 2 * n + 4), (7 * n + 3, 7 * n + 9)))
    return out


def layout_mismatches(views) -> list[str]:
    """Every way `views` disagrees with `kernel_f.pack` over `_pairs()`:
    a member that does not hold, in the packed row, the value of the
    source field it names. Empty when the declaration is the packing."""
    pairs, bad = _pairs(), []
    packed = KF.pack(pairs)
    prim_rows = [r for p in pairs for r in p.prims]
    sources = {  # array -> (flat packed array, the row sources, a namer)
        "prim": (packed.prim, prim_rows, _expand),
        "sep": (packed.sep, [p.sep for p in pairs],
                lambda v: list(zip(_AXES, v, strict=True))),
        "sh": (packed.sh, [p.sh for p in pairs],
               lambda v: list(zip(("a", "b"), v, strict=True))),
        "ao": (packed.ao, [p.ao for p in pairs],
               lambda v: list(zip(("a", "b"), v, strict=True))),
    }
    of_array = {"bra_prim": "prim", "ket_prim": "prim", seams.AB_SH: "prim",
                "AB": "sep", "CD": "sep", "bra_sh": "sh", "ket_sh": "sh",
                "bra_ao": "ao", "ket_ao": "ao"}
    if set(views) != set(of_array):
        bad.append(f"views cover {sorted(views)}, the packed records "
                   f"{sorted(of_array)}")
    for arr, rec in views.items():
        flat, rows, namer = sources[of_array[arr]]
        width = len(namer(rows[0]))
        if len(rec.members) != width:
            bad.append(f"{arr}: {rec.name} has {len(rec.members)} members, "
                       f"the packed row {width}")
            continue
        if len(flat) != width * len(rows):
            bad.append(f"{arr}: {len(flat)} packed values for {len(rows)} "
                       f"rows of {width}")
        for i, row in enumerate(rows):
            for name, want in namer(row):
                if name not in rec.members:
                    bad.append(f"{arr}: {rec.name} has no member {name}")
                    continue
                got = flat[i * width + rec.offset(name)]
                if got != want:
                    bad.append(f"{arr} row {i}: {rec.name}.{name} reads "
                               f"{got}, the packing holds {want}")
    return bad


def test_the_declared_views_are_the_packing_member_by_member() -> None:
    assert layout_mismatches(sr.VIEWS) == []
    # the seams' positional offsets name the same members
    assert sr.PRIM_PAIR.members == seams.REC == KF.REC
    assert [sr.PRIM_PAIR.offset(m) for m in ("P_x", "X_x", "zeta", "U")] == [
        seams._P, seams._X, seams._ZETA, seams._U]
    # one scalar per record, the packed arrays' (test_kernel_f.fused's)
    assert {r.scalar for r in (sr.PRIM_PAIR, sr.SEP)} == {Scalar.F64}
    assert sr.INDEX_PAIR.scalar is Scalar.I32


def _drift(rec: Record, members) -> dict:
    return {a: (dataclasses.replace(r, members=tuple(members)) if r == rec
                else r) for a, r in sr.VIEWS.items()}


@pytest.mark.parametrize("drift", [
    pytest.param(lambda: _drift(sr.PRIM_PAIR, ("P_x", "P_y", "P_z", "X_x",
                                               "X_y", "X_z", "U", "zeta")),
                 id="prim-zeta-U-swapped"),
    pytest.param(lambda: _drift(sr.PRIM_PAIR, sr.PRIM_PAIR.members[:-1]),
                 id="prim-U-dropped"),
    pytest.param(lambda: _drift(sr.SEP, ("y", "x", "z")), id="sep-swapped"),
    pytest.param(lambda: _drift(sr.INDEX_PAIR, ("b", "a")),
                 id="index-pair-swapped"),
    pytest.param(lambda: {a: r for a, r in sr.VIEWS.items() if a != "CD"},
                 id="CD-unviewed"),
])
def test_a_drifted_declaration_fails_the_layout_check(drift) -> None:
    assert layout_mismatches(drift())


def test_a_drifted_view_also_changes_the_spelled_struct() -> None:
    """The C side: the struct the writer spells is the declaration, so a
    drift reaches the emitted member order (and the static_assert keeps the
    width), never a silent reinterpretation of the packed row."""
    text = record_struct(sr.PRIM_PAIR)
    assert text.index("double zeta;") < text.index("double U;")
    assert ('static_assert(sizeof(fj_prim_pair) == 8 * sizeof(double)'
            in text)


# ---------------------------------------------------------------------------
# 3. The IR is the literal kernel's, retyped
# ---------------------------------------------------------------------------


def _untype_expr(e):
    if isinstance(e, Member):
        return Index(e.array, tuple(_untype_expr(i) for i in e.at))
    if isinstance(e, Index):
        return Index(e.array, tuple(_untype_expr(i) for i in e.at))
    if isinstance(e, Call):
        return Call(e.fn, tuple(_untype_expr(a) for a in e.args))
    if isinstance(e, Weighted):
        return dataclasses.replace(e, terms=tuple(
            dataclasses.replace(t, src=_untype_expr(t.src)) for t in e.terms))
    return e


def _untype(stmts):
    """`stmts` with every `Member` the `Index` it is (the statement types
    kernel_f builds: Decl, Assign, AtomicAdd, For, If, Note)."""
    out = []
    for s in stmts:
        if isinstance(s, Decl):
            s = dataclasses.replace(
                s, shape=tuple(_untype_expr(d) for d in s.shape),
                init=None if s.init is None else _untype_expr(s.init))
        elif isinstance(s, (Assign, AtomicAdd)):
            s = dataclasses.replace(s, dst=_untype_expr(s.dst),
                                    value=_untype_expr(s.value))
        elif hasattr(s, "body") and hasattr(s, "var"):  # For
            s = dataclasses.replace(s, start=_untype_expr(s.start),
                                    stop=_untype_expr(s.stop),
                                    step=_untype_expr(s.step),
                                    body=_untype(s.body))
        elif hasattr(s, "then"):  # If
            s = dataclasses.replace(s, cond=_untype_expr(s.cond),
                                    then=_untype(s.then),
                                    orelse=_untype(s.orelse))
        out.append(s)
    return tuple(out)


def _reads(kern) -> list:
    """Every `Index` (Members included) READ in `kern`."""
    out = []
    for s in walk(kern.body):
        if isinstance(s, (Assign, AtomicAdd)):
            at = s.dst.at if isinstance(s.dst, Index) else ()
            exprs = (s.value, *at)
        else:
            exprs = _exprs_of(s)
        for e in exprs:
            if e is not None:
                out += [x for x in _subexprs(e) if isinstance(x, Index)]
    return out


@pytest.mark.parametrize("eris", [False, True], ids=["shipped", "tap"])
@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_ir_is_the_literal_kernel_with_its_record_reads_typed(
        cls, eris: bool) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=eris)
    literal = build_f_kernel(cls, eris=eris)
    assert kern != literal
    assert kern.params == literal.params
    prov = Note(levers.provenance(NAME))
    assert tuple(s for s in _untype(kern.body) if s != prov) == literal.body

    reads = _reads(kern)
    members = [e for e in reads if isinstance(e, Member)]
    assert members and all(e.record is sr.VIEWS[e.array] for e in members)
    untyped = {e.array for e in reads if not isinstance(e, Member)}
    assert not untyped & set(sr.VIEWS), sorted(untyped & set(sr.VIEWS))
    assert untyped <= {"G_ab", "G_cd", "D", "Dsh", "boys"}  # boys: a local
    # the member named is the one the literal offset meant
    for e in members:
        assert e.record.members[e.at[-1].value] == e.member
    # same multiset of reads, retyped
    lit = sorted(map(repr, _reads(literal)))
    assert sorted(repr(_untype_expr(e)) for e in reads) == lit


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_literal_kernel_reads_nothing_typed(cls) -> None:
    assert not any(isinstance(e, Member)
                   for e in _reads(build_f_kernel(cls, eris=True)))


# ---------------------------------------------------------------------------
# 4. Parity, interpreted
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("cls", LADDER)
def test_the_typed_build_is_the_untyped_build_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    F, g = _launch(build_f_kernel(lever_spec(cls), eris=True), cls,
                   monkeypatch)
    F0, g0 = _launch(build_f_kernel(cls, eris=True), cls, monkeypatch)
    assert np.any(g0 != 0) and np.any(F0 != 0)  # not a comparison of zeros
    assert np.array_equal(g, g0)
    assert np.array_equal(F, F0)


def _against_sweep(cls, basis, D, bra_ix, ket_ix, diag, monkeypatch):
    kern = build_f_kernel(lever_spec(cls))
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    worst, *_ = tk._check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag)
    return worst


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    worst = _against_sweep(cls, basis, tk._density(basis.n_ao), [(0, 1)],
                           [(2, 3)], 0, monkeypatch)
    assert worst <= tk.TOL, worst


@pytest.mark.parametrize("case", [c for c in tk.COINCIDENCE
                                  if sum(tk.COINCIDENCE[c][0]) <= CAP])
def test_every_degeneracy_case_within_the_cap_equals_the_sweep(
        case, monkeypatch: pytest.MonkeyPatch) -> None:
    cls, n, bra_ix, ket_ix, diag = tk.COINCIDENCE[case]
    basis = Basis(tuple(tk._same_l_shells(cls[0], n)))
    worst = _against_sweep(cls, basis, tk._density(basis.n_ao), bra_ix,
                           ket_ix, diag, monkeypatch)
    assert worst <= tk.TOL, (case, worst)


# ---------------------------------------------------------------------------
# 5. The spelling
# ---------------------------------------------------------------------------

EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_each_record_is_spelled_once_ahead_of_the_kernel(backend) -> None:
    emit, arch, _ = EMITTERS[backend]
    text = emit(build_f_kernel(lever_spec((0, 1, 0, 1))), arch)
    head, body = text.split('extern "C"', 1)
    order = [head.index(f"struct fj_{r}") for r in ("sep", "index_pair",
                                                    "prim_pair")]
    assert order == sorted(order)  # first use: AB, bra_sh, bra_prim
    for r in (sr.SEP, sr.INDEX_PAIR, sr.PRIM_PAIR):
        assert head.count(record_struct(r)) == 1
        assert f"struct fj_{r.name}" not in body
    assert ("reinterpret_cast<const fj_prim_pair*>(bra_prim)"
            "[(bra * k_ab + i)].zeta" in body)
    assert ("reinterpret_cast<const fj_prim_pair*>(ket_prim)"
            "[(k * k_cd + kcd)].U" in body)
    assert "reinterpret_cast<const fj_prim_pair*>(ab_sh)[kab].P_x" in body
    assert "reinterpret_cast<const fj_sep*>(CD)[k].z" in body
    assert "reinterpret_cast<const fj_index_pair*>(ket_ao)[k].b" in body
    assert "ab_sh[i][6] = " in body  # the staging writes stay positional
    assert "G_ab[bra]" in body


@pytest.mark.parametrize("backend", sorted(EMITTERS))
@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_no_literal_f_kernel_carries_a_view(cls, backend) -> None:
    emit, arch, _ = EMITTERS[backend]
    for kw in ({}, {"eris": True}):
        text = emit(build_f_kernel(cls, **kw), arch)
        assert "struct fj_" not in text and "reinterpret_cast" not in text


_REC = Record("row3", Scalar.F64, ("u", "v", "w"))


def _tiny(shape, at, *, dst=None, scalar=Scalar.F64) -> Kernel:
    params = (Param("n", Scalar.I32),
              Param("a", scalar, array=True, shape=shape),
              Param("o", Scalar.F64, array=True, intent=Intent.OUT,
                    shape=(Var("n"),)))
    read = Member("a", at, _REC)
    if dst is not None:
        return Kernel("t", params, (Assign(dst, Lit(1.0)),))
    return Kernel("t", params, (Assign(Index("o", (Lit(0),)), read),))


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_writer_spells_and_refuses_a_view(backend) -> None:
    emit, arch, _ = EMITTERS[backend]
    ok = emit(_tiny((Var("n"), Lit(4), Lit(3)), (Lit(1), Lit(2), Lit(0))),
              arch)
    assert "reinterpret_cast<const fj_row3*>(a)[(1 * 4 + 2)].u" in ok
    with pytest.raises(EmitError, match="last extent is the literal 3"):
        emit(_tiny((Var("n"), Lit(4)), (Lit(1), Lit(0))), arch)
    with pytest.raises(EmitError, match="one subscript per extent"):
        emit(_tiny((Var("n"), Lit(4), Lit(3)), (Lit(1), Lit(0))), arch)
    with pytest.raises(EmitError, match="F64"):
        emit(_tiny((Var("n"), Lit(3)), (Lit(1), Lit(0)), scalar=Scalar.I32),
             arch)
    with pytest.raises(EmitError, match="never a store target"):
        emit(_tiny((Var("n"), Lit(3)), (Lit(0), Lit(0)),
                   dst=Member("o", (Lit(0), Lit(0)), _REC)), arch)


def test_a_member_is_validated_on_construction() -> None:
    with pytest.raises(ValueError, match="literal offset"):
        Member("a", (Lit(0), Lit(3)), _REC)
    with pytest.raises(ValueError, match="literal offset"):
        Member("a", (Lit(0), Var("i")), _REC)
    with pytest.raises(ValueError, match="a row index"):
        Member("a", (Lit(0),), _REC)
    with pytest.raises(TypeError, match="not a Record"):
        Member("a", (Lit(0), Lit(0)), "row3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        Record("r", Scalar.F64, ("x", "x"))
    m = Member("a", (Lit(0), Lit(2)), _REC)
    assert isinstance(m, Index) and m.member == "w"
    assert m != Index("a", (Lit(0), Lit(2)))  # a typed read is not untyped


# ---------------------------------------------------------------------------
# 6. The goldens: (sp|sp) with the lever, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.eri.fock.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, ('struct-reads',))); \
#   Path(f'test/golden/cuda/f_struct_reads/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_struct_reads/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_struct_reads_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_struct_reads"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert text.count("struct fj_") == 3
