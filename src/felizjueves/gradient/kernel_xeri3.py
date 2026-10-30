"""The UNCONTRACTED three-center d1 kernel: `Target.ERI3` at
`derivative_order = 1` with the kernel-body lever `uncontracted` (#692,
#693's lever), keyed ``coulomb/d1/(la,lb|lP)/b1/eri3/uncontracted``.

    build_xeri3_kernel(spec | (la, lb, lP)) -> Kernel        one per class
    xeri3_name(spec | (la, lb, lP))         -> str           xeri3_{la}{lb}{lP}
    xeri3_spec(la, lb, lP)                  -> Spec          the class's key
    launch_shape(n_cell, threads)           -> (grid, block)

The derivative integrals THEMSELVES, in libcint's `int3c2e_ip1` form, over
one AUX BLOCK of the caller's choosing:

    X[k, m, n, P - aux0] = (nabla_k m n | P) = -d(mn|P) / dA_k

`A` the centre of `m`, `X` `[3, n_ao, n_ao, n_blk]`: `eri3`'s `(m, n, P)`
layout with the Cartesian axis outermost, over the block's aux functions
`aux0 .. aux0 + n_blk - 1` (`gradient.host_df.eri3_ip1` is the reference,
on the full aux range). A DF-RHF Hessian contracts it many ways (DF-J's
`SUM_P X d_P` and `SUM_mn X D_mn`, DF-K's `SUM_lP X Gam_lnP`, the cross
terms' `(ij|P)^x`), each a gemm, and resolves atoms by slicing -- what
PySCF's `df.hessian.rhf` does with `int3c2e_ip1` in aux blocks. No fused
kernel covers all of them, so this one writes the tensor.

THE AUX CENTRE NEEDS NO SECOND KERNEL. Each integral is translationally
invariant, `d/dA + d/dB + d/dC = 0`, and `nabla = -d/dR` on each centre,
so

    (mn | nabla_k P) = -X[k, m, n, P] - X[k, n, m, P]

(libcint's `int3c2e_ip2`), which the caller forms. THE SIGN CHECK: with
`ip1 = -dA`, `ip1^T = -dB` (the `(n, m)` slot is `n`'s own centre), and
`ip2 = -dC = dA + dB = -ip1 - ip1^T`. test_kernel_xeri3.py holds the host
to libcint's `int3c2e_ip1_cart` and the identity to `int3c2e_ip2_cart`.

THE BLOCK. The full tensor is `3 n_ao^2 n_aux` doubles (9 GB at 500 AOs and
1500 aux functions), so the caller launches over a SUBSET of the cells,
as PySCF batches over aux shells: every cell whose aux shell lies in the
block, staged as `kernel_eri3` stages them (`eri.density_fit.eri3.
eri3_classes`, then the per-cell arrays restricted to the block's cells).
The kernel assumes nothing about the cells it is not given: it writes only
the slots of the cells it is given, and reads nothing but its inputs.
`off_p` stays the aux function's index in the whole aux basis; the column
written is `off_p - aux0`, so every listed cell's aux shell must lie in
`[aux0, aux0 + n_blk)`. The linear index of `X` is I32: keep `3 n_ao^2
n_blk < 2^31`.

THE SCATTER. A cell is an AO shell pair `i <= j` crossed with one aux
shell. It owns the slots `(m, n)` and `(n, m)` for `m` in shell `i`, `n` in
shell `j`, every aux function of its shell, and no other cell writes them,
so the stores are plain (no atomic). Per element `(a, b, q)` of a staged
cell it writes `(m, n)` = `(oa + a, ob + b)` with the `dA` rule and `(n,
m)` with the `dB` rule; a cell staged SWAPPED needs no flag, the rows
follow the staged order. A DIAGONAL shell pair (`oa == ob`, `la == lb`)
writes the `dA` slot alone: over every `(a, b)` of the shell that is every
slot, and the `dB` one would be the same number by another route.

THE ROUTE is `kernel_geri3`'s low-stack form (#613) with the weights and
the atom reduction removed: the bra's per-axis `E` tables one index up
(`e_fragment.axis_tables_fragment(la + 1, lb + 1)`), the Cartesian rule
`dA E^{ab}_t = 2 alpha E^{a+1_k,b}_t - a_k E^{a-1_k,b}_t` (and `dB`) as
rolled sums over them, `[t + tau]^(0)` by the plan's tower for the quartet
`(la + 1, lb | lP, 0)` at `L + 1` stored into `R`, and `U[t] = SUM_tau
[t + tau]^(0) E^q_tau` per aux row. One form at every class: no weight and
no density is held, so the stack is `kernel_geri3`'s low-stack one less the
weights. Per cell:

    X slots of the cell = 0
    per bra record:   Ex Ey Ez = axis_tables_fragment(la + 1, lb + 1)
      per aux record: R = the tower; Cx Cy Cz = center_tables_fragment(lP)
        per aux row q:  U[t] = SUM_tau R[t + tau] E^q_tau
          per (a, b):   X[k, m, n, q] -+= w (2 alpha SUM E^{a+1_k,b} U
                                            - a_k SUM E^{a-1_k,b} U)
                        X[k, n, m, q] likewise on b (not on a diagonal pair)

`w = s_ab s_aux`, the sign `(-1)^{lP}` of every live `tau` then the `-` of
`nabla`. The slots accumulate in global memory over the records (each is one
thread's own), so the kernel ZEROES THEM ITSELF first; the caller need not.

ROUND-OFF, NOT BITWISE, against `host_df.eri3_ip1` (the same rule on the AO
side, the plan's MILD DAG against `reference.pq`).

    params  n_cell n_bra_rec n_aux_rec n_ao n_blk aux0 (I32)
            zeta_ab[n_bra_rec]  P XPA XPB[n_bra_rec, 3]  s_ab  ea_bra  eb_ket
            A[n_aux_rec, 3]  alpha[n_aux_rec]  s_aux[n_aux_rec]
            ab_rec0 ab_rec1 p_rec0 p_rec1 off_a off_b off_p [n_cell]  (I32)
            X[3, n_ao, n_ao, n_blk]  (OUT, the cells' slots overwritten)

The arrays are `Eri3Class.arrays()`' (`s` as `s_aux`) plus
`host_df.eri3_records`' exponents; `kernel_geri3`'s, less `M`, `G`, the
atoms and `n_aux` / `n_atom`, plus `n_blk` and `aux0`.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host_df`
(its witness) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_metric import S_AUX
from felizjueves.gradient.kernel_geri3 import (
    AUX_TABLES,
    R_ARRAY,
    U_ARRAY,
    geri3_plan,
)
from felizjueves.gradient.kernel_gmetric import center_cols, launch_shape, tower
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    For,
    If,
    Index,
    Inline,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    node_name,
    store_targets,
)
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target, lever_provenance

__all__ = [
    "build_xeri3_kernel",
    "launch_shape",
    "xeri3_name",
    "xeri3_spec",
]

UNCONTRACTED = "uncontracted"  # the kernel-body lever (levers.VOCABULARY)
CELL = "cell"  # one thread per (AO shell pair, aux shell) of the block
BRA = "ib"  # the bra record, serial
KET = "ik"  # the aux record, serial inside the bra

OWNED = ("a cell owns its (m, n) and (n, m) slots over its aux shell's "
         "columns and no other cell writes them: plain stores, no atomic; "
         "they accumulate over the records in X itself, zeroed here first")
DIAGONAL = ("a diagonal shell pair (oa == ob) writes the dA slot alone: over "
            "every (a, b) of the shell that is every slot")


def xeri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` at d1, `Target.ERI3`, uncontracted."""
    return Spec(Operator.COULOMB, 1, (la, lb, lP), Target.ERI3,
                (UNCONTRACTED,))


def _class(cls: Spec | tuple[int, int, int]) -> tuple[int, int, int]:
    spec = cls if isinstance(cls, Spec) else xeri3_spec(*cls)
    if spec.target is not Target.ERI3 or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.ERI3 key")
    if spec.levers != (UNCONTRACTED,):
        raise ValueError(f"{spec.key()}: the uncontracted three-center "
                         f"derivative is the key with exactly the lever "
                         f"{UNCONTRACTED!r} (the contraction is "
                         "kernel_geri3's)")
    la, lb, lP = spec.am
    return la, lb, lP


def xeri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """`xeri3_` and the three `l`s, `_`-separated once any is two digits
    (`kernel_eri3.eri3_name`'s rule)."""
    am = _class(cls)
    return "xeri3_" + ("_" if max(am) > 9 else "").join(map(str, am))


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _plus(*vs: tuple[int, ...]) -> tuple[int, int, int]:
    return tuple(sum(v[d] for v in vs) for d in range(3))  # type: ignore[return-value]


def build_xeri3_kernel(cls: Spec | tuple[int, int, int], *,
                       name: str | None = None) -> Kernel:
    """The uncontracted three-center d1 kernel of one class (module
    docstring). `cls` is an uncontracted d1 `Target.ERI3` `Spec` or `(la,
    lb, lP)`."""
    la, lb, lP = _class(cls)
    Lab = la + lb
    L1 = Lab + lP + 1  # the tower's order
    dag, order = geri3_plan((la, lb, lP))
    fB = ef.axis_tables_fragment(la + 1, lb + 1)
    fP = ef.center_tables_fragment(lP)
    nA, nB, nP = (len(ef.cartesians(x)) for x in (la, lb, lP))
    # X = -dA, and dA carries (-1)^{lP}: an odd class adds, an even one
    # subtracts.
    acc = Fn.ADD if lP % 2 == 1 else Fn.SUB
    cell, ib, ik = Var(CELL), Var(BRA), Var(KET)
    targets = frozenset(_plus(t, tau) for t in hermite_targets(Lab + 1)
                        for q in ef.cartesians(lP)
                        for _, tau in center_cols(q, lP))
    tw = store_targets(tower(order, dag, {node_name((r, 0)) for r in targets}),
                       targets, L1, R_ARRAY)

    def u_read(h: Expr) -> Expr:
        return Index(U_ARRAY, (h,))

    def x_at(k: int, m: Expr, n: Expr, col: Expr) -> Index:
        return Index("X", (Lit(k), m, n, col))

    def per_q(q: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def u_of(t: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def r_at(tau: tuple[Expr, Expr, Expr]) -> Expr:
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(t[d], tau[d]) for d in range(3))),))
            return (*ef.rolled_center_sum(lP, q, r_at, "u", tag="c",
                                          arrays=AUX_TABLES),
                    Assign(Index(U_ARRAY, (ef.hermite_index(t),)), Var("u")))

        def row(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def elem(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
                m, n = _iadd(Var("oa"), Var("ra")), _iadd(Var("ob"), Var("rb"))
                col = _iadd(Var("pc"), Var("rq"))
                out: list[Stmt] = [
                    Decl("rb", Scalar.I32, init=ef.cartesian_index("b"))]
                for side, ab, two in (("A", a, "ea2"), ("B", b, "eb2")):
                    side_out: list[Stmt] = []
                    for k in range(3):
                        def moved(by: int, ab=ab, k=k) -> tuple[Expr, ...]:
                            step = (_iadd(ab[k], Lit(1)) if by > 0
                                    else _c(Fn.SUB, ab[k], Lit(1)))
                            return tuple(step if d == k else ab[d]
                                         for d in range(3))
                        up, lo = moved(1), moved(-1)
                        ra_, rb_ = (up, b) if side == "A" else (a, up)
                        la_, lb_ = (lo, b) if side == "A" else (a, lo)
                        hi, dn = f"h{side}{k}", f"l{side}{k}"
                        dst = (x_at(k, m, n, col) if side == "A"
                               else x_at(k, n, m, col))
                        side_out += [
                            *ef.rolled_e_sum(la + 1, lb + 1, ra_, rb_,
                                             u_read, hi, tag=f"p{side}{k}"),
                            Decl(dn, Scalar.F64, init=Lit(0.0)),
                            If(_c(Fn.LT, Lit(0), ab[k]), (
                                *ef.rolled_e_sum(la + 1, lb + 1, la_, lb_,
                                                 u_read, f"m{side}{k}",
                                                 tag=f"q{side}{k}"),
                                Assign(Var(dn), _mul(ab[k],
                                                     Var(f"m{side}{k}"))),
                            )),
                            Assign(dst, _c(acc, dst, _mul(Var("w"), _c(
                                Fn.SUB, _mul(Var(two), Var(hi)),
                                Var(dn))))),
                        ]
                    if side == "B" and la == lb:
                        out.append(If(Var("differ"), tuple(side_out)))
                    else:
                        out += side_out
                return tuple(out)
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lb, ("bx", "by", "bz"), elem, tag="b"))

        return (
            Decl("rq", Scalar.I32, init=ef.cartesian_index("q")),
            Note("U[t] = SUM_tau [t+tau]^(0) E^q_tau over the live tau of "
                 "this aux row, for every |t| <= la + lb + 1: once per (t, q)"),
            ef.hermite_loops(Lab + 1, u_of, tag="h"),
            Note(f"per element (a, b, q): X[k, m, n] {'+' if lP % 2 else '-'}"
                 "= w SUM_t (dA_k E^{ab})_t U[t] (the Cartesian rule, "
                 "2 alpha E^{a+1_k,b} - a_k E^{a-1_k,b}), X[k, n, m] likewise "
                 "on b; X = -dA (nabla), (-1)^|tau| = "
                 f"(-1)^{lP} on every live tau"),
            ef.cartesian_loops(la, ("ax", "ay", "az"), row, tag="a"),
        )

    aux_binds: dict[str, Expr] = {n: Var(n) for n in AUX_TABLES}
    if ef.HALF_INV_ZETA in fP.binds:
        aux_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("aq"))
    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("A", (ik, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ze"), Var("aq")),
            _c(Fn.ADD, Var("ze"), Var("aq"))))),
        Note(f"[t+tau]^(0) by the plan's DAG for the quartet ({la + 1},{lb}|"
             f"{lP},0) at L + 1 = {L1}, the targets no U reads dropped, each "
             "live one stored into R as it forms"),
        *tw,
        Note("the aux record's single-center E, per axis: "
             "e_fragment.center_tables_fragment, inlined"),
        *(Decl(n, Scalar.F64, (Lit(ef.center_table_extent(lP)),))
          for n in AUX_TABLES),
        Inline(fP, aux_binds),
        Decl("w", Scalar.F64, init=_mul(Var("sb"), Index(S_AUX, (ik,)))),
        ef.cartesian_loops(lP, ("qx", "qy", "qz"), per_q, tag="q"),
    )
    bra_binds: dict[str, Expr] = {ef.ZETA: Var("ze"),
                                  **{n: Var(n) for n in ef.AXIS_ARRAYS}}
    if ef.HALF_INV_ZETA in fB.binds:
        bra_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("ze"))
    for d in range(3):
        bra_binds[Factor.pa(d).name_in_kernel] = Index("XPA", (ib, Lit(d)))
        bra_binds[Factor.pb(d).name_in_kernel] = Index("XPB", (ib, Lit(d)))
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (ib,)))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (ib,)))),
        Note(f"the undressed per-axis E^{{ij}}_t, i <= {la + 1}, j <= "
             f"{lb + 1}: one index above the class on each side, for the "
             "derivative rule (e_fragment.axis_tables_fragment)"),
        *(Decl(n, Scalar.F64, (Lit(ef.axis_table_extent(la + 1, lb + 1)),))
          for n in ef.AXIS_ARRAYS),
        Inline(fB, bra_binds),
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )
    zi, zj, zq = Var("za"), Var("zb"), Var("zq")
    zm, zn = _iadd(Var("oa"), zi), _iadd(Var("ob"), zj)
    zcol = _iadd(Var("pc"), zq)
    zero = For("za", Lit(0), Lit(nA), Lit(1), (
        For("zb", Lit(0), Lit(nB), Lit(1), (
            For("zq", Lit(0), Lit(nP), Lit(1), tuple(
                Assign(x_at(k, m_, n_, zcol), Lit(0.0))
                for k in range(3) for m_, n_ in ((zm, zn), (zn, zm)))),)),))
    differ: tuple[Stmt, ...] = ()
    if la == lb:
        differ = (Note(DIAGONAL), Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))),)
    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Note("the aux shell's first column in the block: off_p - aux0"),
        Decl("pc", Scalar.I32, init=_c(Fn.SUB, Index("off_p", (cell,)),
                                       Var("aux0"))),
        *differ,
        Note(OWNED),
        zero,
        *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
          for n, x in ((R_ARRAY, L1), (U_ARRAY, Lab + 1))),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
    )
    body: tuple[Stmt, ...] = (
        Note(f"the uncontracted d1 three-center integrals, class (la, lb, "
             f"lP) = ({la}, {lb}, {lP}): X[k, m, n, P - aux0] = (nabla_k m n"
             "|P), libcint's int3c2e_ip1, over the cells of one aux block "
             "(felizjueves.gradient.kernel_xeri3)"),
        Note(lever_provenance(UNCONTRACTED)),
        Note("the aux centre is the caller's: (mn|nabla_k P) = -X[k, m, n, "
             "P] - X[k, n, m, P] (translational invariance)"),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or xeri3_name((la, lb, lP)), _params(), body)


def _params() -> tuple[Param, ...]:
    """The parameter list (module docstring)."""

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, nblk = Var("n_ao"), Var("n_blk")
    return (
        Param("n_cell", Scalar.I32),
        Param("n_bra_rec", Scalar.I32),
        Param("n_aux_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_blk", Scalar.I32),
        Param("aux0", Scalar.I32),
        f64("zeta_ab", (nbra,)),
        f64("P", (nbra, Lit(3))),
        f64("XPA", (nbra, Lit(3))),
        f64("XPB", (nbra, Lit(3))),
        f64("s_ab", (nbra,)),
        f64("ea_bra", (nbra,)),
        f64("eb_ket", (nbra,)),
        f64("A", (naux_rec, Lit(3))),
        f64("alpha", (naux_rec,)),
        f64(S_AUX, (naux_rec,)),
        *(i32(n, (ncell,)) for n in ("ab_rec0", "ab_rec1", "p_rec0",
                                     "p_rec1", "off_a", "off_b", "off_p")),
        Param("X", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(3), nao, nao, nblk)),
    )
