"""The exchange-energy Hessian kernel: `Target.K` at `derivative_order =
2`, the K kernel differentiated twice.

    build_hkx_kernel(spec | (la, lc, lb, ld)) -> Kernel   one per K class
    hkx_name(spec | (la, lc, lb, ld))         -> str      hkx_{la}{lc}_{lb}{ld}
    hkx_spec(la, lc, lb, ld)                  -> Spec     the class's d2 key
    hkx_pq_spec(la, lc, lb, ld)               -> Spec     the d0 K class
                                                          whose plan it runs
    launch_shape(n_bra, lanes)                -> (grid, (1, lanes))

THE ENERGY AND THE SPLIT. `E_K = 1/2 SUM_abcd D_ab D_cd (ac|bd)` (K's
pairing: bra record `(a, c)` on `A`, `C`; ket record `(b, d)` on `B`,
`D`; `am = (la, lc, lb, ld)`). For a SYMMETRIC `D`, renaming `(ac) <->
(bd)` turns the ket-ket term into the bra-bra one and the ket-bra term
into the bra-ket one, so the half cancels both twos (`hessian.host`,
THE EXCHANGE HESSIAN):

    d2 E_K / dx dy = SUM_{ordered quartets} D_ab D_cd
                     [d_bra^x d_bra^y + d_bra^x d_ket^y] (ac|bd)

Unlike `E_J`'s, `D_ab D_cd` straddles bra and ket, so the bra-ket term has
no ket derivative DENSITY to precompute (`kernel_hdq`): both derivatives
are taken inside one (bra, ket) pair, and this one kernel writes the
whole skeleton `d2 E_K` of its class. The RHF exchange energy is
`-1/2 E_K`: the factor is the caller's.

THE PLAN IS K's, TWO ORDERS UP. The bra-bra term reaches `E^{a+2,c}`
(`|p| <= lp + 2`) and the bra-ket term `E^{a+1,c}` against `E^{b+1,d}`
(`|p| <= lp + 1`, `|q| <= lq + 1`): every `[r]^(0)` either reads has
`|r| <= lp + lq + 2`, exactly the targets of the d0 K class with the bra
raised by two, `hkx_pq_spec = k_class_spec(la + 2, lc, lb, ld)` (split
`(lp + 2, lq)`), its one DAG in `schedule`'s order (`gk_pq_spec`'s move,
one order further).

THE BRA, AS RAISED ROWS (`kernel_gk`'s scheme, twice). On one axis the
rule `T_k(a) = 2 alpha [a + 1_k] - a_k [a - 1_k]` composes into a short
sum of shifted rows, each with an integer coefficient and a power of
`2 alpha` (`2 gamma`): `dA_k dA_k` is `(2 alpha)^2 [a + 2_k] - (2 alpha)
(2 a_k + 1) [a] + a_k (a_k - 1) [a - 2_k]`, `dA_k dC_l` four terms, and so
on (`_derivative_terms`, by composition, never by hand). Each shifted row
`(a', c')` is folded ONCE per `(b,d)` row,

    V(a', c') = SUM_p E^{a'c'}_p I[p]

and digested into every accumulator it serves, `acc += c W V`, `W = D_ab
D_cd`; the powers of `2 alpha`, `2 gamma` (per-bra constants) are kept
apart, one accumulator per power, and multiplied in after the ket loop.
The `E^{a'c'}` entries are products of the per-axis tables
`e_fragment.axis_tables_fragment(la + 2, lc + 2)`, bound to `Gx`, `Gy`,
`Gz`.

THE KET. Per ket record, the per-axis tables at `(lb + 1, ld + 1)`
(bound to `Fx`, `Fy`, `Fz`) give, per `(b,d)` row, the plain `s_bd
E^{bd}_q` (`|q| <= lq`) and the six first derivatives `s_bd (dY_l
E^{bd})_q` (`Y = B`, `D`; `|q| <= lq + 1`, `2 beta`, `2 delta` the
ket's), as straight-line locals; then

    I[p]      = SUM_q (-1)^|q| [p+q]^(0) s_bd E^{bd}_q          |p| <= lp + 2
    J{Y}{l}[p] = SUM_q (-1)^|q| [p+q]^(0) s_bd (dY_l E^{bd})_q   |p| <= lp + 1

and the bra-bra term digests `V` of `I`, the bra-ket term `V` of each
`J{Y}{l}` over the once-raised rows.

THE THREAD MAPPING IS `kernel_gk`'s: one warp per bra record, blockDim
`(1, lanes)`, the kets strip-mined over `THREAD_Y`. The bra-bra term's
atoms are the bra's, so it is reduced once per bra record: 21 `WarpReduce`s
(`AA`, `CC` with `k <= l`, `AC`) and 36 `AtomicAdd`s (the `AC` block also
on `[C l][A k]`). The bra-ket term's other atom is the KET's, which varies
from lane to lane; so the host groups the kets by their atom pair
(`hessian.host.exchange_hessian_records`) and the kernel walks those
SEGMENTS serially (a loop uniform across the warp, as `hjab`'s atoms):
per segment the kets are strip-mined, then 36 `WarpReduce`s and lane 0's
36 `AtomicAdd`s into `H[atom_X, k, seg_Y, l]`, `X` = `A`, `C` and `Y` =
`B`, `D`. Never mirrored: summed over every quartet the bra-ket term is
symmetric by itself. No shared memory.

NO SCREEN, as `gk`: K's bound bounds the integral, not its second
derivative. A launch is one class, every bra record of
`gradient.host.exchange_pairs(la, lc)` against every ket record of
`exchange_pairs(lb, ld)`, so the class launches over a basis sum to the
whole `d2 E_K`.

    params  n_bra n_ket n_seg n_ao n_atom (I32)
            P[n_bra, 3]  zeta_ac  XPA[n_bra, 3]  XPC[n_bra, 3]  s_ac
                ea_bra  ec_bra  [n_bra]                          bra side
            off_a  off_c  atom_a  atom_c  [n_bra]  (I32)
            Q[n_ket, 3]  eta  XQB[n_ket, 3]  XQD[n_ket, 3]  s_bd
                eb_ket  ed_ket  [n_ket]                          ket side
            off_b  off_d  [n_ket]  (I32)
            seg_start[n_seg + 1]  seg_b  seg_d  [n_seg]  (I32)   the segments
            M[n_ao, n_ao]               the density D, read at AO offsets
            H[n_atom, 3, n_atom, 3]     (OUT, read and AtomicAdd-ed)

`H` is ZEROED BY THE CALLER; every launch only adds.

ROUND-OFF, NOT BITWISE, against `hessian.host.exchange_hessian`, which
differentiates before the fold and runs `reference.pq`'s `[r]`; the
stated bound is in test/test_kernel_hkx.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor `hessian.host`
nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from collections import defaultdict

from felizjueves.eri.k_exchange.kernel_k import k_class_spec
from felizjueves.eri.k_exchange.kernel_k import launch_shape as _k_launch_shape
from felizjueves.eri.k_exchange.levers.seams import KET, absent, bindings
from felizjueves.hessian.kernel_hjab import ZEROED_H
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Expr,
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
    WarpReduce,
)
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.kernel import builder_plan
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "BB_KINDS",
    "PARAMS",
    "build_hkx_kernel",
    "hkx_name",
    "hkx_pq_spec",
    "hkx_spec",
    "launch_shape",
]

PARAMS = ("n_bra", "n_ket", "n_seg", "n_ao", "n_atom",
          "P", "zeta_ac", "XPA", "XPC", "s_ac", "ea_bra", "ec_bra",
          "off_a", "off_c", "atom_a", "atom_c",
          "Q", "eta", "XQB", "XQD", "s_bd", "eb_ket", "ed_ket",
          "off_b", "off_d", "seg_start", "seg_b", "seg_d",
          "M", "H")

# The bra-bra kinds: both derivatives on A, one on each, both on C.
BB_KINDS = ("AA", "AC", "CC")

BRA_ARRAYS = ("Gx", "Gy", "Gz")  # the bra's per-axis tables
KET_ARRAYS = ("Fx", "Fy", "Fz")  # the ket's per-axis tables
SEG = "sg"  # the segment loop's variable

NO_SCREEN = (
    "no screen: K's bound I_ac I_bd max|D| bounds the integral, not its "
    "second derivative, so every ordered quartet of the class is visited")

Am = tuple[int, int, int, int]
Vec = tuple[int, int, int]
# A shifted row's coefficient: (shift of a, shift of c, power of 2 alpha,
# power of 2 gamma) -> integer.
Terms = dict[tuple[Vec, Vec, int, int], int]


def hkx_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The K class `(la, lc | lb, ld)` at derivative order 2."""
    return Spec(Operator.COULOMB, 2, (la, lc, lb, ld), Target.K)


def hkx_pq_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The d0 K class whose plan the Hessian of `(la, lc | lb, ld)` runs:
    the bra raised by two, split `(la + lc + 2, lb + ld)`."""
    return k_class_spec(la + 2, lc, lb, ld)


def _class(cls: Spec | Am) -> Am:
    spec = cls if isinstance(cls, Spec) else hkx_spec(*cls)
    if spec.target is not Target.K or spec.derivative_order != 2:
        raise ValueError(f"{spec.key()} is not a d2 Target.K key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Hessian kernels take no levers")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: the exchange Hessian is built at "
                         "one batch only")
    la, lc, lb, ld = spec.am
    return la, lc, lb, ld


def hkx_name(cls: Spec | Am) -> str:
    """The name `build_hkx_kernel(cls)` gives its kernel, unbuilt."""
    la, lc, lb, ld = _class(cls)
    return f"hkx_{la}{lc}_{lb}{ld}"


def launch_shape(n_bra: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (1, lanes))`: one warp per bra record, `kernel_k`'s."""
    return _k_launch_shape(n_bra, lanes)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _add(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def _vadd(v: Vec, w: Vec) -> Vec:
    return (v[0] + w[0], v[1] + w[1], v[2] + w[2])


def _unit(k: int, by: int) -> Vec:
    return tuple(by * (d == k) for d in range(3))  # type: ignore[return-value]


ZERO: Vec = (0, 0, 0)


def _apply(terms: Terms, a: Vec, c: Vec, side: str, k: int) -> Terms:
    """One first derivative on `side` ("A" or "C"), axis `k`, applied to a
    sum of shifted rows of the row `(a, c)`: `T_k(x) = (2 e) [x + 1_k] -
    x_k [x - 1_k]` on that side's index, the `- x_k` term absent where
    `x_k = 0`."""
    out: Terms = defaultdict(int)
    for (sa, sc, i, j), cf in terms.items():
        x = _vadd(a, sa) if side == "A" else _vadd(c, sc)
        for by in (1, -1):
            if by == -1 and x[k] == 0:
                continue
            m = 1 if by == 1 else -x[k]
            if side == "A":
                key = (_vadd(sa, _unit(k, by)), sc, i + (by == 1), j)
            else:
                key = (sa, _vadd(sc, _unit(k, by)), i, j + (by == 1))
            out[key] += m * cf
    return {key: v for key, v in out.items() if v}


def _derivative_terms(a: Vec, c: Vec, ops: tuple[tuple[str, int], ...]
                      ) -> Terms:
    """`ops` (each `(side, axis)`) composed on the row `(a, c)`."""
    terms: Terms = {(ZERO, ZERO, 0, 0): 1}
    for side, k in ops:
        terms = _apply(terms, a, c, side, k)
    return terms


def _bb_components() -> list[tuple[str, int, int]]:
    """`(kind, k, l)` of the bra-bra term that are formed: `k <= l` where
    the kind is symmetric in them."""
    return [(kind, k, l) for kind in BB_KINDS for k in range(3)
            for l in range(3) if kind == "AC" or k <= l]  # noqa: E741


def _digests(la: int, lc: int):
    """Every shifted row the bra reads, `{(a', c'): [(acc, (ia, ic), cf,
    ket)]}`: `acc` the accumulator, `(ia, ic)` the original row (its `W`),
    `cf` the integer coefficient, `ket` None for the bra-bra term (folded
    against `I`) or `(Y, l)` for the bra-ket one (folded against
    `J{Y}{l}`). Also the accumulators' `{name: (component, i, j)}`."""
    ca, cc = ef.cartesians(la), ef.cartesians(lc)
    rows: dict[tuple[Vec, Vec], list] = defaultdict(list)
    accs: dict[str, tuple] = {}
    for ia, a in enumerate(ca):
        for ic, c in enumerate(cc):
            for kind, k, l in _bb_components():  # noqa: E741
                ops = ((kind[0], k), (kind[1], l))
                for (sa, sc, i, j), cf in sorted(
                        _derivative_terms(a, c, ops).items()):
                    name = f"bb{kind}{k}{l}_{i}{j}"
                    accs[name] = ((kind, k, l), i, j)
                    rows[_vadd(a, sa), _vadd(c, sc)].append(
                        (name, (ia, ic), cf, None))
            for side in "AC":
                for k in range(3):
                    for (sa, sc, i, j), cf in sorted(
                            _derivative_terms(a, c, ((side, k),)).items()):
                        for Y in "BD":
                            for l in range(3):  # noqa: E741
                                name = f"bk{side}{k}{Y}{l}_{i + j}"
                                accs[name] = ((side, k, Y, l), i, j)
                                rows[_vadd(a, sa), _vadd(c, sc)].append(
                                    (name, (ia, ic), cf, (Y, l)))
    return dict(sorted(rows.items())), accs


class _Straight:
    """Straight-line F64 locals reassigned per `(b,d)` row: a `Decl` on a
    name's first write, an `Assign` after (`kernel_gk`'s)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def set(self, name: str, value: Expr) -> Stmt:
        if name in self._seen:
            return Assign(Var(name), value)
        self._seen.add(name)
        return Decl(name, Scalar.F64, init=value)


def _table_reader(arrays: tuple[str, str, str], imax: int, jmax: int):
    """`E^{ij}_t` on axis `d` from per-axis tables
    `axis_tables_fragment(imax, jmax)` bound to `arrays`, at a constant
    index (`kernel_g1e.axis_reader`, under other names: the bra's and the
    ket's tables live in one kernel)."""
    L = imax + jmax

    def read(d: int, i: int, j: int, t: int) -> Expr:
        if not (0 <= i <= imax and 0 <= j <= jmax and 0 <= t <= i + j):
            raise ValueError(f"E^{{{i}{j}}}_{t} is not in the ({imax}, "
                             f"{jmax}) tables")
        return Index(arrays[d], (ef.axis_table_index(i, j, t, jmax, L),))
    return read


def _ket_first(F, b: Vec, d: Vec, Y: str, l: int, q: Vec  # noqa: E741
               ) -> Expr | None:
    """`(dY_l E^{bd})_q` from the ket tables, None where structurally
    zero: `2 e E^{x + 1_l} - x_l E^{x - 1_l}` on axis `l`."""
    for x in range(3):
        top = b[x] + d[x] + (x == l)
        if q[x] > top:
            return None
    f: list[Expr] = []
    for x in range(3):
        if x != l:
            f.append(F(x, b[x], d[x], q[x]))
            continue
        up = (b[x] + 1, d[x]) if Y == "B" else (b[x], d[x] + 1)
        out: Expr = _mul(Var("eb2" if Y == "B" else "ed2"), F(x, *up, q[x]))
        n = b[x] if Y == "B" else d[x]
        lo = (b[x] - 1, d[x]) if Y == "B" else (b[x], d[x] - 1)
        if n and q[x] <= lo[0] + lo[1]:
            out = _c(Fn.SUB, out, _mul(Lit(float(n)), F(x, *lo, q[x])))
        f.append(out)
    return _mul(_mul(f[0], f[1]), f[2])


def _fold_pq(ps, qs, cols: dict[int, str]) -> list[Expr]:
    """`SUM_q (-1)^|q| [p+q]^(0) col_q` per `p`, over the `q` in `cols`."""
    out = []
    for p in ps:
        acc: Expr = Lit(0.0)
        for j, nm in cols.items():
            q = qs[j]
            term = _mul(Var(node_name(((p[0] + q[0], p[1] + q[1],
                                        p[2] + q[2]), 0))), Var(nm))
            acc = _c(Fn.SUB if sum(q) % 2 else Fn.ADD, acc, term)
        out.append(acc)
    return out


def _powers(i: int, j: int) -> Expr | None:
    """`(2 alpha)^i (2 gamma)^j`, None when both are 0."""
    f: list[Expr] = [Var("ea2")] * i + [Var("ec2")] * j
    if not f:
        return None
    out = f[0]
    for x in f[1:]:
        out = _mul(out, x)
    return out


def _combine(names: list[tuple[str, int, int]]) -> Expr:
    """`SUM (2 alpha)^i (2 gamma)^j acc_ij` over one component's
    accumulators."""
    out: Expr | None = None
    for nm, i, j in names:
        p = _powers(i, j)
        t: Expr = Var(nm) if p is None else _mul(p, Var(nm))
        out = t if out is None else _add(out, t)
    assert out is not None
    return out


def build_hkx_kernel(cls: Spec | Am, *, name: str | None = None) -> Kernel:
    """The exchange-energy Hessian kernel of one K class (module
    docstring). `cls` is a d2 `Target.K` `Spec` or `(la, lc, lb, ld)`,
    the Hermite pairing, bra first."""
    la, lc, lb, ld = _class(cls)
    lp, lq = la + lc, lb + ld
    (dag,) = builder_plan(hkx_pq_spec(la, lc, lb, ld)).dags
    order = tuple(schedule(dag))
    p2, p1 = hermite_targets(lp + 2), hermite_targets(lp + 1)
    q0, q1 = hermite_targets(lq), hermite_targets(lq + 1)
    rows_bd = ef.block_rows(lb, ld)
    nb, nd = len(ef.cartesians(lb)), len(ef.cartesians(ld))
    na, nc = len(ef.cartesians(la)), len(ef.cartesians(lc))
    bra, ket, seg = Var("bra"), Var(KET), Var(SEG)
    E = _table_reader(BRA_ARRAYS, la + 2, lc + 2)
    F = _table_reader(KET_ARRAYS, lb + 1, ld + 1)
    shifted, accs = _digests(la, lc)
    bb_accs = [n for n in accs if n.startswith("bb")]
    bk_accs = [n for n in accs if n.startswith("bk")]

    # --- the bra record: scalars, the per-axis E tables, the accumulators ----
    frag_ac = ef.axis_tables_fragment(la + 2, lc + 2)
    binds = {k: v for k, v in bindings(frag_ac, Var("ze"), "XPA", "XPC",
                                       bra).items() if k in frag_ac.binds}
    for arr, bar in zip(ef.AXIS_ARRAYS, BRA_ARRAYS, strict=True):
        binds[arr] = Var(bar)
    frag_bd = ef.axis_tables_fragment(lb + 1, ld + 1)
    kbinds = {k: v for k, v in bindings(frag_bd, Var("et"), "XQB", "XQD",
                                        ket).items() if k in frag_bd.binds}
    for arr, kar in zip(ef.AXIS_ARRAYS, KET_ARRAYS, strict=True):
        kbinds[arr] = Var(kar)
    head: list[Stmt] = [
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (bra, Lit(d))))
          for d in range(3)),
        Decl("ze", Scalar.F64, init=Index("zeta_ac", (bra,))),
        Decl("sb", Scalar.F64, init=Index("s_ac", (bra,))),
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (bra,)))),
        Decl("ec2", Scalar.F64, init=_mul(Lit(2.0), Index("ec_bra", (bra,)))),
        *(Decl(v, Scalar.I32, init=Index(p, (bra,)))
          for v, p in (("oa", "off_a"), ("oc", "off_c"), ("aa", "atom_a"),
                       ("ac", "atom_c"))),
        Note(f"the undressed per-axis E^{{ac}}_t, i <= {la + 2}, "
             f"j <= {lc + 2}: two indices above the class on each side, for "
             "the shifted rows"),
        *(Decl(arr, Scalar.F64, (Lit(ef.axis_table_extent(la + 2, lc + 2)),))
          for arr in BRA_ARRAYS),
        Inline(frag_ac, binds),
        Note("the ket's per-axis tables, refilled per ket record"),
        *(Decl(arr, Scalar.F64, (Lit(ef.axis_table_extent(lb + 1, ld + 1)),))
          for arr in KET_ARRAYS),
        Note("the bra-bra accumulators, one per component and power of "
             "(2 alpha, 2 gamma), the power taken out"),
        *(Decl(n, Scalar.F64, init=Lit(0.0)) for n in bb_accs),
    ]

    # --- one (bra, ket) pair -------------------------------------------------
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    local = _Straight()
    pair: list[Stmt] = [
        *(Decl(n, Scalar.F64,
               init=_c(Fn.SUB, Var(f"P{d}"), Index("Q", (ket, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Decl("et", Scalar.F64, init=Index("eta", (ket,))),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ze"), Var("et")),
            _add(Var("ze"), Var("et"))))),
        *prelude,
        *recurrence_body(dag, list(order), fundamental),
        Note(f"the undressed per-axis E^bd_t of this ket record, i <= "
             f"{lb + 1}, j <= {ld + 1}: e_fragment.axis_tables_fragment, "
             "inlined into Fx, Fy, Fz"),
        Inline(frag_bd, kbinds),
        Decl("sq", Scalar.F64, init=Index("s_bd", (ket,))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (ket,)))),
        Decl("ed2", Scalar.F64, init=_mul(Lit(2.0), Index("ed_ket", (ket,)))),
        Decl("ob", Scalar.I32, init=Index("off_b", (ket,))),
        Decl("od", Scalar.I32, init=Index("off_d", (ket,))),
        Note("D_ab and D_cd straddle the records: read once per ket record "
             "at AO offsets"),
        *(Decl(f"Dab{i}_{j}", Scalar.F64, init=Index("M", (
            _add(Var("oa"), Lit(i)), _add(Var("ob"), Lit(j)))))
          for i in range(na) for j in range(nb)),
        *(Decl(f"Dcd{i}_{j}", Scalar.F64, init=Index("M", (
            _add(Var("oc"), Lit(i)), _add(Var("od"), Lit(j)))))
          for i in range(nc) for j in range(nd)),
        Note(f"per (b,d) row: s_bd E^bd_q and s_bd (dY_l E^bd)_q; I[p], "
             f"|p| <= {lp + 2}, and J{{Y}}{{l}}[p], |p| <= {lp + 1}; W = "
             "D_ab D_cd; every shifted row V(a', c') folded once and "
             "digested into the accumulators it serves"),
    ]
    for rbd, (b, d) in enumerate(rows_bd):
        ib, id_ = divmod(rbd, nd)
        plain: dict[int, str] = {}
        for j, q in enumerate(q0):
            if absent(q, b, d):
                continue
            nm = f"e{j}"
            plain[j] = nm
            pair.append(local.set(nm, _mul(Var("sq"), _mul(_mul(
                F(0, b[0], d[0], q[0]), F(1, b[1], d[1], q[1])),
                F(2, b[2], d[2], q[2])))))
        for n, v in enumerate(_fold_pq(p2, q0, plain)):
            pair.append(local.set(f"I{n}", v))
        for Y in "BD":
            for l in range(3):  # noqa: E741
                cols: dict[int, str] = {}
                for j, q in enumerate(q1):
                    v = _ket_first(F, b, d, Y, l, q)
                    if v is None:
                        continue
                    nm = f"d{Y}{l}_{j}"
                    cols[j] = nm
                    pair.append(local.set(nm, _mul(Var("sq"), v)))
                for n, v in enumerate(_fold_pq(p1, q1, cols)):
                    pair.append(local.set(f"J{Y}{l}_{n}", v))
        for ia in range(na):
            for ic in range(nc):
                pair.append(local.set(f"W{ia}_{ic}", _mul(
                    Var(f"Dab{ia}_{ib}"), Var(f"Dcd{ic}_{id_}"))))
        for (a_row, c_row), digests in shifted.items():
            for kt in sorted({dg[3] for dg in digests},
                             key=lambda x: (x is not None, x)):
                ps, src = (p2, "I") if kt is None else (p1, f"J{kt[0]}{kt[1]}_")
                v: Expr = Lit(0.0)
                for n, p in enumerate(ps):
                    if absent(p, a_row, c_row):
                        continue
                    f = [E(x, a_row[x], c_row[x], p[x]) for x in range(3)]
                    v = _add(v, _mul(_mul(_mul(f[0], f[1]), f[2]),
                                     Var(f"{src}{n}")))
                pair.append(local.set("v", v))
                for acc_name, (ia, ic), cf, dkt in digests:
                    if dkt != kt:
                        continue
                    wv: Expr = _mul(Var(f"W{ia}_{ic}"), Var("v"))
                    if abs(cf) != 1:
                        wv = _mul(Lit(float(abs(cf))), wv)
                    dst = Var(acc_name)
                    pair.append(Assign(dst, _c(
                        Fn.ADD if cf > 0 else Fn.SUB, dst, wv)))

    # --- the segments: the bra-ket term, reduced per ket atom pair -----------
    by_comp: dict[tuple, list[tuple[str, int, int]]] = defaultdict(list)
    for nm, (comp, i, j) in accs.items():
        by_comp[comp].append((nm, i, j))
    bk_tail: list[Stmt] = [Note("the bra-ket values: SUM (2 alpha)^i "
                                "(2 gamma)^j acc, reduced across the warp")]
    bk_adds: list[Stmt] = []
    for side, at in (("A", "aa"), ("C", "ac")):
        for k in range(3):
            for Y, yat in (("B", "yb"), ("D", "yd")):
                for l in range(3):  # noqa: E741
                    comp = (side, k, Y, l)
                    r = f"r{side}{k}{Y}{l}"
                    bk_tail += [Decl(r, Scalar.F64),
                                WarpReduce(Var(r), _combine(by_comp[comp]))]
                    bk_adds.append(AtomicAdd(
                        Index("H", (Var(at), Lit(k), Var(yat), Lit(l))),
                        _mul(Var("sb"), Var(r))))
    seg_body: tuple[Stmt, ...] = (
        Decl("yb", Scalar.I32, init=Index("seg_b", (seg,))),
        Decl("yd", Scalar.I32, init=Index("seg_d", (seg,))),
        Note("the bra-ket accumulators of this segment, one per component "
             "and power of 2 alpha (2 gamma)"),
        *(Decl(n, Scalar.F64, init=Lit(0.0)) for n in bk_accs),
        Note("this segment's kets strip-mined over threadIdx.y, stride "
             "blockDim.y"),
        For(KET, _add(Index("seg_start", (seg,)), Call(Fn.THREAD_Y)),
            Index("seg_start", (_add(seg, Lit(1)),)), Call(Fn.BLOCK_DIM_Y),
            tuple(pair)),
        *bk_tail,
        Note("AtomicAdd by lane 0 into the bra's and the segment's atoms"),
        If(_c(Fn.LT, Call(Fn.THREAD_Y), Lit(1)), tuple(bk_adds)),
    )

    # --- after every segment: the bra-bra term -------------------------------
    bb_tail: list[Stmt] = [Note("the bra-bra values, reduced across the "
                                "warp")]
    for kind, k, l in _bb_components():  # noqa: E741
        r = f"r{kind}{k}{l}"
        bb_tail += [Decl(r, Scalar.F64),
                    WarpReduce(Var(r), _combine(by_comp[kind, k, l]))]

    def h(x: str, k: int, y: str, l: int) -> Index:  # noqa: E741
        return Index("H", (Var(x), Lit(k), Var(y), Lit(l)))

    bb_adds: list[Stmt] = []
    for kind, atom in (("AA", "aa"), ("CC", "ac")):
        for k in range(3):
            for l in range(3):  # noqa: E741
                lo, hi = min(k, l), max(k, l)
                bb_adds.append(AtomicAdd(h(atom, k, atom, l), _mul(
                    Var("sb"), Var(f"r{kind}{lo}{hi}"))))
    for k in range(3):
        for l in range(3):  # noqa: E741
            v = _mul(Var("sb"), Var(f"rAC{k}{l}"))
            bb_adds += [AtomicAdd(h("aa", k, "ac", l), v),
                        AtomicAdd(h("ac", l, "aa", k), v)]
    bb_tail += [
        Note("AtomicAdd by lane 0: the AC block also on [C l][A k]"),
        If(_c(Fn.LT, Call(Fn.THREAD_Y), Lit(1)), tuple(bb_adds)),
    ]

    per_bra: list[Stmt] = [
        *head,
        Note("the ket segments, serially: each one atom pair (atom_b, "
             "atom_d), uniform across the warp"),
        For(SEG, Lit(0), Var("n_seg"), Lit(1), seg_body),
        *bb_tail,
    ]
    body: tuple[Stmt, ...] = (
        Note(f"Hessian of E_K = 1/2 SUM D_ab D_cd (ac|bd), K class "
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}): H[X, k, Y, l] += s_ac "
             "SUM D_ab D_cd [d_bra d_bra + d_bra d_ket] (ac|bd), [r] at "
             f"L = {lp + lq + 2} (felizjueves.hessian)"),
        Note(ZEROED_H),
        Note("PRECONDITION: D is symmetric (the ket-ket and ket-bra terms "
             "are the bra-bra and bra-ket ones only then)"),
        Note("PRECONDITION: the kets are grouped by (atom_b, atom_d), "
             "seg_start[n_seg] = n_ket (hessian.host."
             "exchange_hessian_records)"),
        Note(NO_SCREEN),
        Note("launch: blockDim = (1, warp width), one warp per bra record"),
        Decl("bra", Scalar.I32, init=_add(Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    return Kernel(name or hkx_name((la, lc, lb, ld)), _params(), body)


def _params() -> tuple[Param, ...]:
    nbr, nk, ns, nao, nat = (Var("n_bra"), Var("n_ket"), Var("n_seg"),
                             Var("n_ao"), Var("n_atom"))

    def f64(n: str, *shape: Expr) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, ext: Expr) -> Param:
        return Param(n, Scalar.I32, array=True, shape=(ext,))

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("n_seg", Scalar.I32), Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        f64("P", nbr, Lit(3)), f64("zeta_ac", nbr),
        f64("XPA", nbr, Lit(3)), f64("XPC", nbr, Lit(3)), f64("s_ac", nbr),
        f64("ea_bra", nbr), f64("ec_bra", nbr),
        i32("off_a", nbr), i32("off_c", nbr), i32("atom_a", nbr),
        i32("atom_c", nbr),
        f64("Q", nk, Lit(3)), f64("eta", nk), f64("XQB", nk, Lit(3)),
        f64("XQD", nk, Lit(3)), f64("s_bd", nk), f64("eb_ket", nk),
        f64("ed_ket", nk),
        i32("off_b", nk), i32("off_d", nk),
        i32("seg_start", _add(ns, Lit(1))), i32("seg_b", ns),
        i32("seg_d", ns),
        f64("M", nao, nao),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3), nat, Lit(3))),
    )
    assert tuple(p.name for p in out) == PARAMS
    return out
