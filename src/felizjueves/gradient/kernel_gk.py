"""The exchange-energy gradient kernel: `Target.K` at `derivative_order =
1`, the K kernel's bra derivative.

    build_gk_kernel(spec | (la, lc, lb, ld)) -> Kernel   one per K class
    gk_name(spec | (la, lc, lb, ld))         -> str      gk_{la}{lc}_{lb}{ld}
    gk_spec(la, lc, lb, ld)                  -> Spec     the class's d1 key
    gk_pq_spec(la, lc, lb, ld)               -> Spec     the d0 K class whose
                                                         plan it runs
    launch_shape(n_bra, lanes)               -> (grid, (1, lanes))
    gk_lowstack(spec | am)                   -> bool     THE ROLLED FORM's gate
    build_gk_kernel(..., lowstack=True|False)            either form, forced

THE ENERGY. `E_K = 1/2 SUM_abcd D_ab D_cd (ac|bd)` (`K_ab = SUM_cd D_cd
(ac|bd)`, the K kernel's pairing: bra record `(a, c)`, ket record `(b, d)`,
`am = (la, lc, lb, ld)`). For a SYMMETRIC `D` its ket derivative equals its
bra derivative (`gradient.host`'s module docstring), so

    dE_K / dR_{X,k} = SUM_{ordered bra, ket records} s_ac
                      SUM_abcd D_ab D_cd SUM_p (dX_k E^{ac})_p I_bd[p]
    I_bd[p] = SUM_q (-1)^|q| [p+q]^(0) s_bd E^{bd}_q,   |p| <= la + lc + 1

over the bra's two centres `X = A, C`. The RHF exchange energy is
`-1/2 E_K`: the factor is the caller's, as `gjab`'s half is.

THE PLAN IS K's, ONE ORDER UP. `dA E^{ac}` reaches `E^{a+1,c}`, so `I[p]`
runs to Hermite order `lp + 1`, and the `[r]^(0)` the kernel needs are
exactly those of the d0 K class with the bra raised by one,
`gk_pq_spec(la, lc, lb, ld) = k_class_spec(la + 1, lc, lb, ld)` (split
`(lp + 1, lq)`): `recurrence.recurrence_body` over its plan's one DAG, in
`schedule`'s order, as `kernel_k.ket_body` runs it. Nothing in the plan is
new (`gjab_pq_spec`'s move, on K).

THE DERIVATIVE, AS RAISED ROWS. On axis `k`,
`(dA_k E)^{ac} = 2 alpha E^{a+1_k, c} - a_k E^{a-1_k, c}`, so with the
ordinary stage-2 fold of a RAISED row

    V(a', c) = SUM_p E^{a'c}_p I[p]          a' in cartesians(la +- 1)

the A side is `2 alpha SUM W V(a+1_k, c) - SUM a_k W V(a-1_k, c)`,
`W = D_ab D_cd`, and the C side the same with `c` moved and `gamma`. Each
raised row is folded ONCE per `(b,d)` row and digested into every axis
it serves (up to three), into per-lane accumulators `uA_k` (the `2 alpha`
terms, the factor taken out) and `lA_k` (the `- a_k` terms), and `uC_k`,
`lC_k`; after the ket loop `gA_k = 2 alpha uA_k - lA_k`. The `E^{a'c}`
entries are products of the per-axis tables
`e_fragment.axis_tables_fragment(la + 1, lc + 1)` (#586's low-stack
tables, read at constant indices): no `E^{ac}` block, no derivative block.

THE THREAD MAPPING IS `kernel_k`'s: one warp per bra record, `bra =
THREAD_X + BLOCK_X * BLOCK_DIM_X` with blockDim `(1, lanes)`, the kets
strip-mined over `THREAD_Y`. The ket record's `E^bd` is
`block_fragment(lb, ld)` inlined into a local `Ebd`, as `kernel_k`'s
default `bind_e_bd`; stage 1 is its fold (`s_bd` per term, `-` for odd
`|q|`), into straight-line locals `I{n}`. Then one `WarpReduce` per
component, and lane 0 adds `s_ac gA_k` into `G[atom_a, k]` and
`s_ac gC_k` into `G[atom_c, k]`: six `AtomicAdd`s per bra record, as
`gjab`'s. No shared memory.

NO SCREEN. The K bound `I_ac I_bd max|D|` bounds the integral, not its
derivative; every ordered quartet of the launch is visited. A launch is
one class: every bra record of `exchange_pairs(la, lc)` against every ket
record of `exchange_pairs(lb, ld)` (`gradient.host.exchange_records`), so
the class's launches over a basis sum to the whole `dE_K`.

    params  n_bra n_ket n_ao n_atom (I32)
            P[n_bra, 3]  zeta_ac  XPA[n_bra, 3]  XPC[n_bra, 3]  s_ac
                ea_bra  ec_bra  [n_bra]                          bra side
            off_a  off_c  atom_a  atom_c  [n_bra]  (I32)
            Q[n_ket, 3]  eta  XQB[n_ket, 3]  XQD[n_ket, 3]  s_bd  [n_ket]
            off_b  off_d  [n_ket]  (I32)                         ket side
            M[n_ao, n_ao]         the density D, read at AO offsets
            G[n_atom, 3]          (OUT, read and AtomicAdd-ed)

`ea_bra` / `ec_bra` are the bra's two primitive exponents (`alpha` on `A`,
`gamma` on `C`). `D_ab` straddles the records (`a` on the bra, `b` on the
ket) as `D_cd` does, so both are read from `M` at the records' AO offsets
(`kernel_kblock`'s write addressing, as a read): `M[oa + i, ob + j]`,
`M[oc + k, od + l]`, once per ket record into locals. `G` is ZEROED BY
THE CALLER; every launch only adds.

THE ROLLED FORM (#618), the default from `L = la + lc + lb + ld >=
LOWSTACK_MIN_L = 5` and for every class with an f shell (`gk_lowstack`;
`lowstack=` forces either form; the name, parameters, launch and `G`
contract the same). The unrolled form above grows as `n_ac n_bd n_p n_q`
statements (4.0 MB of text at `(dd|dd)`) and is over #613's 14,000 B of
`sm_86` stack from `L = 5` (`(dd|ps)` 14,880 B); its f classes are never
built by default (`UNROLLED_L_MAX`). The rolled form holds no `E` block
and no straight-line fold:

    per bra:    Ex Ey Ez = the bra's per-axis E, (la + 1, lc + 1), COMPACT
                gA gC = 0;  R[n_t(L + 1)], I[n_t(lp + 1)], Fx Fy Fz  local
    per ket:    the recurrence, each [r]^(0) stored into R as it forms
                (`recurrence.store_targets`); Fx Fy Fz = the ket's per-axis
                E, (lb, ld), COMPACT, every odd-t entry negated: (-1)^|q|
                for every (b, d) row:
                  I[p] = s_bd SUM_q E^bd_q R[p + q]          |p| <= lp + 1
                  for every (a, c) row:  W = M[a, b] M[c, d]
                    gA_k += W (2 alpha SUM E^{a+1_k,c} I - a_k SUM E^{a-1_k,c} I)
                    gC_k likewise on c

each `SUM` a loop nest over the tables, factored per axis
(`e_fragment.rolled_e_sum`'s shape), every row loop a
`cartesian_loops` nest and the `p` loop a `hermite_loops` one, so the text
is the recurrence plus a fixed set of loops. A COMPACT table stores only
the reachable `E^{ij}_t`, `t <= i + j`, at `compact_offset(i, j, l2) + t`
(`tables_fragment`: `axis_tables_fragment`'s recurrence, its stores
re-indexed): 125 doubles a table at `(4, 4)` against the dense 225, which
takes `(ff|ff)` from 15,008 B of `sm_86` stack to 11,616 B (measured,
data/measured/gk.json). Each derivative term is folded per `(a, c)` row
(the unrolled form folds each raised row once and digests it into up to
three axes), so the two forms agree to round-off, not bitwise
(test/test_kernel_gk.py's differential).

ROUND-OFF, NOT BITWISE, against `gradient.host.exchange_gradient`, which
differentiates `E^{ac}` before the fold and runs `reference.pq`'s `[r]`
(the kernel: raised rows, the plan's MILD recurrence); the stated bound is
in test/test_kernel_gk.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor
`gradient.host` nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

from felizjueves.eri.k_exchange.kernel_k import k_class_spec
from felizjueves.eri.k_exchange.kernel_k import launch_shape as _k_launch_shape
from felizjueves.eri.k_exchange.levers.seams import KET, absent, bindings
from felizjueves.gradient.kernel_g1e import ZEROED_G, axis_reader
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Expr,
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
    store_targets,
)
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "LOWSTACK_MIN_L",
    "PARAMS",
    "build_gk_kernel",
    "gk_lowstack",
    "gk_name",
    "gk_pq_spec",
    "gk_spec",
    "launch_shape",
]

PARAMS = ("n_bra", "n_ket", "n_ao", "n_atom",
          "P", "zeta_ac", "XPA", "XPC", "s_ac", "ea_bra", "ec_bra",
          "off_a", "off_c", "atom_a", "atom_c",
          "Q", "eta", "XQB", "XQD", "s_bd", "off_b", "off_d",
          "M", "G")

NO_SCREEN = (
    "no screen: K's bound I_ac I_bd max|D| bounds the integral, not its "
    "derivative, so every ordered quartet of the class is visited")

WRITE_NOTE = (
    "AtomicAdd by lane 0: every bra record on an atom adds into that atom's "
    "row of G")

Am = tuple[int, int, int, int]


def gk_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The K class `(la, lc | lb, ld)` at derivative order 1."""
    return Spec(Operator.COULOMB, 1, (la, lc, lb, ld), Target.K)


def gk_pq_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The d0 K class whose plan the gradient of `(la, lc | lb, ld)` runs:
    the bra raised by one, split `(la + lc + 1, lb + ld)`."""
    return k_class_spec(la + 1, lc, lb, ld)


def _class(cls: Spec | Am) -> Am:
    spec = cls if isinstance(cls, Spec) else gk_spec(*cls)
    if spec.target is not Target.K or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.K key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: the exchange gradient is built at "
                         "one batch only")
    la, lc, lb, ld = spec.am
    return la, lc, lb, ld


def gk_name(cls: Spec | Am) -> str:
    """The name `build_gk_kernel(cls)` gives its kernel, unbuilt."""
    la, lc, lb, ld = _class(cls)
    return f"gk_{la}{lc}_{lb}{ld}"


def launch_shape(n_bra: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (1, lanes))`: one warp per bra record, `kernel_k`'s."""
    return _k_launch_shape(n_bra, lanes)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _add(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def _shift(v: tuple[int, int, int], k: int, by: int) -> tuple[int, int, int]:
    return tuple(c + by * (d == k) for d, c in enumerate(v))  # type: ignore[return-value]


class _Straight:
    """Straight-line F64 locals reassigned per `(b,d)` row: a `Decl` on a
    name's first write, an `Assign` after, so the ket loop's body declares
    each once."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def set(self, name: str, value: Expr) -> Stmt:
        if name in self._seen:
            return Assign(Var(name), value)
        self._seen.add(name)
        return Decl(name, Scalar.F64, init=value)


def _raised_rows(la: int, lc: int):
    """Every raised row the derivative reads, as `(a', c', digests)` --
    `a'` or `c'` one shell above or below the class -- with its digests
    `(acc, k, (ia, ic), n)`: `acc` the accumulator (`uA`, `lA`, `uC`,
    `lC`), `k` the axis, `(ia, ic)` the original `(a, c)` row's indices,
    `n` the moved component of the original row (the `- a_k` term's
    integer; read for a lowered row only)."""
    ca, cc = ef.cartesians(la), ef.cartesians(lc)
    out = []
    for side, l_moved in (("A", la), ("C", lc)):
        for up, lm in ((True, l_moved + 1), (False, l_moved - 1)):
            if lm < 0:
                continue
            for xm in ef.cartesians(lm):
                for other_i, other in enumerate(cc if side == "A" else ca):
                    digests = []
                    for k in range(3):
                        x = _shift(xm, k, -1 if up else 1)
                        if min(x) < 0:
                            continue
                        n = x[k]
                        ix = (ca if side == "A" else cc).index(x)
                        rac = ((ix, other_i) if side == "A"
                               else (other_i, ix))
                        acc = ("u" if up else "l") + side
                        digests.append((acc, k, rac, n))
                    a_row, c_row = ((xm, other) if side == "A"
                                    else (other, xm))
                    out.append((a_row, c_row, digests))
    return out


def build_gk_kernel(cls: Spec | Am, *, name: str | None = None,
                    lowstack: bool | None = None) -> Kernel:
    """The exchange-energy gradient kernel of one K class (module
    docstring). `cls` is a d1 `Target.K` `Spec` or `(la, lc, lb, ld)`,
    the Hermite pairing, bra first. `lowstack=None` takes
    `gk_lowstack(cls)`'s choice; `True` / `False` force THE ROLLED FORM /
    the unrolled one at any class (the name, parameters, launch and `G`
    contract the same)."""
    la, lc, lb, ld = _class(cls)
    if lowstack is None:
        lowstack = gk_lowstack((la, lc, lb, ld))
    if lowstack:
        return _build_lowstack((la, lc, lb, ld), name)
    lp, lq = la + lc, lb + ld
    (dag,) = builder_plan(gk_pq_spec(la, lc, lb, ld)).dags
    order = tuple(schedule(dag))
    ps, qs = hermite_targets(lp + 1), hermite_targets(lq)
    rows_bd = ef.block_rows(lb, ld)
    nb, nd = len(ef.cartesians(lb)), len(ef.cartesians(ld))
    nc = len(ef.cartesians(lc))
    na = len(ef.cartesians(la))
    bra, ket = Var("bra"), Var(KET)
    E = axis_reader(la + 1, lc + 1)
    raised = _raised_rows(la, lc)
    accs = [a for a in ("uA", "lA", "uC", "lC")
            if any(d[0] == a for *_, ds in raised for d in ds)]

    # --- the bra record: scalars and the per-axis E tables -------------------
    frag_ac = ef.axis_tables_fragment(la + 1, lc + 1)
    binds = {k: v for k, v in bindings(frag_ac, Var("ze"), "XPA", "XPC",
                                       bra).items() if k in frag_ac.binds}
    for arr in ef.AXIS_ARRAYS:
        binds[arr] = Var(arr)
    n_tab = ef.axis_table_extent(la + 1, lc + 1)
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
        Note(f"the undressed per-axis E^{{ac}}_t, i <= {la + 1}, "
             f"j <= {lc + 1}: one index above the class on each side, for "
             "the raised rows"),
        *(Decl(arr, Scalar.F64, (Lit(n_tab),)) for arr in ef.AXIS_ARRAYS),
        Inline(frag_ac, binds),
        Note("per-lane accumulators: u = the 2 alpha (2 gamma) terms, the "
             "factor taken out; l = the - a_k (- c_k) terms"),
        *(Decl(f"{a}{k}", Scalar.F64, init=Lit(0.0))
          for a in accs for k in range(3)),
    ]

    # --- one (bra, ket) pair: the recurrence, E^bd, stages 1 and 2 -----------
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    frag_bd = ef.block_fragment(lb, ld)
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
        Note("the undressed E^bd block of this ket record: "
             "e_fragment.block_fragment(lb, ld), inlined; s_bd is the host's"),
        Decl("Ebd", Scalar.F64, (Lit(len(rows_bd)), Lit(len(qs)))),
        Inline(frag_bd, {**bindings(frag_bd, Var("et"), "XQB", "XQD", ket),
                         ef.E_ARRAY: Var("Ebd")}),
        Decl("sq", Scalar.F64, init=Index("s_bd", (ket,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (ket,))),
        Decl("od", Scalar.I32, init=Index("off_d", (ket,))),
        Note("D_ab and D_cd straddle the records (a, c on the bra; b, d on "
             "the ket): read once per ket record at AO offsets"),
        *(Decl(f"Dab{i}_{j}", Scalar.F64, init=Index("M", (
            _add(Var("oa"), Lit(i)), _add(Var("ob"), Lit(j)))))
          for i in range(na) for j in range(nb)),
        *(Decl(f"Dcd{i}_{j}", Scalar.F64, init=Index("M", (
            _add(Var("oc"), Lit(i)), _add(Var("od"), Lit(j)))))
          for i in range(nc) for j in range(nd)),
        Note(f"per (b,d) row: stage 1, I[p] = SUM_q [p|q] s_bd E^bd_q, "
             f"|p| <= {lp + 1}; W = D_ab D_cd; then every raised row "
             "V(a', c) = SUM_p E^{a'c}_p I[p] folded once and digested into "
             "the axes it serves"),
    ]
    for rbd, (b, d) in enumerate(rows_bd):
        ib, id_ = divmod(rbd, nd)
        for n, p in enumerate(ps):
            acc: Expr = Lit(0.0)
            for j, q in enumerate(qs):
                if absent(q, b, d):
                    continue
                term = _mul(Var(node_name(((p[0] + q[0], p[1] + q[1],
                                            p[2] + q[2]), 0))),
                            _mul(Var("sq"), Index("Ebd", (Lit(rbd), Lit(j)))))
                acc = _c(Fn.SUB if sum(q) % 2 else Fn.ADD, acc, term)
            pair.append(local.set(f"I{n}", acc))
        for ia in range(na):
            for ic in range(nc):
                pair.append(local.set(f"W{ia}_{ic}", _mul(
                    Var(f"Dab{ia}_{ib}"), Var(f"Dcd{ic}_{id_}"))))
        for a_row, c_row, digests in raised:
            v: Expr = Lit(0.0)
            for n, p in enumerate(ps):
                if absent(p, a_row, c_row):
                    continue
                f = [E(x, a_row[x], c_row[x], p[x]) for x in range(3)]
                v = _add(v, _mul(_mul(_mul(f[0], f[1]), f[2]), Var(f"I{n}")))
            pair.append(local.set("v", v))
            for acc_name, k, (ia, ic), nk in digests:
                wv: Expr = _mul(Var(f"W{ia}_{ic}"), Var("v"))
                if acc_name[0] == "l" and nk != 1:
                    wv = _mul(Lit(float(nk)), wv)
                dst = Var(f"{acc_name}{k}")
                pair.append(Assign(dst, _add(dst, wv)))

    # --- after the ket loop: combine, reduce, scatter -------------------------
    tail: list[Stmt] = [Note("gA_k = 2 alpha uA_k - lA_k, gC_k likewise")]
    for side, two in (("A", "ea2"), ("C", "ec2")):
        for k in range(3):
            g: Expr = _mul(Var(two), Var(f"u{side}{k}"))
            if f"l{side}" in accs:
                g = _c(Fn.SUB, g, Var(f"l{side}{k}"))
            tail.append(Decl(f"g{side}{k}", Scalar.F64, init=g))
    tail.append(Note("reduce across the warp: every lane sees the sum"))
    for side in "AC":
        for k in range(3):
            tail += [Decl(f"r{side}{k}", Scalar.F64),
                     WarpReduce(Var(f"r{side}{k}"), Var(f"g{side}{k}"))]
    tail += [
        Note(WRITE_NOTE),
        If(_c(Fn.LT, Call(Fn.THREAD_Y), Lit(1)), tuple(
            AtomicAdd(Index("G", (Var(atom), Lit(k))),
                      _mul(Var("sb"), Var(f"r{side}{k}")))
            for side, atom in (("A", "aa"), ("C", "ac")) for k in range(3))),
    ]

    per_bra: list[Stmt] = [
        *head,
        Note("kets strip-mined over threadIdx.y, stride blockDim.y"),
        For(KET, Call(Fn.THREAD_Y), Var("n_ket"), Call(Fn.BLOCK_DIM_Y),
            tuple(pair)),
        *tail,
    ]
    body: tuple[Stmt, ...] = (
        Note(f"gradient of E_K = 1/2 SUM D_ab D_cd (ac|bd), K class "
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}): G[X, k] += s_ac SUM "
             "D_ab D_cd SUM_p (dX_k E^ac)_p SUM_q [p|q] s_bd E^bd_q on the "
             f"bra's two centres, [r] at L = {lp + lq + 1} "
             "(felizjueves.gradient)"),
        Note(ZEROED_G),
        Note("PRECONDITION: D is symmetric (the ket derivative is the bra "
             "derivative only then)"),
        Note(NO_SCREEN),
        Note("launch: blockDim = (1, warp width), one warp per bra record"),
        Decl("bra", Scalar.I32, init=_add(Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    return Kernel(name or gk_name((la, lc, lb, ld)), _params(), body)


def _params() -> tuple[Param, ...]:
    nbr, nk, nao, nat = (Var("n_bra"), Var("n_ket"), Var("n_ao"),
                         Var("n_atom"))

    def f64(n: str, *shape: Expr) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, ext: Expr) -> Param:
        return Param(n, Scalar.I32, array=True, shape=(ext,))

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("n_ao", Scalar.I32), Param("n_atom", Scalar.I32),
        f64("P", nbr, Lit(3)), f64("zeta_ac", nbr),
        f64("XPA", nbr, Lit(3)), f64("XPC", nbr, Lit(3)), f64("s_ac", nbr),
        f64("ea_bra", nbr), f64("ec_bra", nbr),
        i32("off_a", nbr), i32("off_c", nbr), i32("atom_a", nbr),
        i32("atom_c", nbr),
        f64("Q", nk, Lit(3)), f64("eta", nk), f64("XQB", nk, Lit(3)),
        f64("XQD", nk, Lit(3)), f64("s_bd", nk),
        i32("off_b", nk), i32("off_d", nk),
        f64("M", nao, nao),
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    )
    assert tuple(p.name for p in out) == PARAMS
    return out


# ---------------------------------------------------------------------------
# The rolled form (#618)
# ---------------------------------------------------------------------------

# every class from L = la + lc + lb + ld = LOWSTACK_MIN_L (#618): the
# smallest L with an inadmissible unrolled class (data/measured/gk.json)
LOWSTACK_MIN_L = 5
# the largest shell l the unrolled form is the default for: an f class's
# unrolled form is not attempted (#618, by decision), so it is not admissible
UNROLLED_L_MAX = 2
R_ARRAY = "R"  # [r]^(0) of one (bra, ket) pair, hermite_targets(L + 1) order
I_ARRAY = "I"  # stage 1 of one (b,d) row, hermite_targets(lp + 1) order
KET_TABLES = ("Fx", "Fy", "Fz")  # the ket record's per-axis E, sign folded in

LOWSTACK_NOTE = (
    "ROLLED (#618): the bra's per-axis E tables one index up and the ket's "
    "per-axis E tables (the ket sign (-1)^q folded into their odd-t "
    "entries), [r]^(0) into a local R as it forms, stage 1 into a local "
    "I[p] per (b,d) row, and each (a,c) row's six derivative terms as "
    "rolled sums over the tables, W = D_ab D_cd read from M where it is "
    "used: no E block, no straight-line fold, no per-row local is held")


def gk_lowstack(cls: Spec | Am) -> bool:
    """THE SELECTION RULE (#618): True when `la + lc + lb + ld >=
    LOWSTACK_MIN_L`, or when the class has an f (or higher) shell, whose
    unrolled form is never built by default (UNROLLED_L_MAX); there
    `build_gk_kernel` emits the rolled form by default. False below, where
    the unrolled text is unchanged."""
    am = _class(cls)
    return sum(am) >= LOWSTACK_MIN_L or max(am) > UNROLLED_L_MAX


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def compact_extent(l1: int, l2: int) -> int:
    """Doubles in one COMPACT per-axis table of `(l1, l2)`: only the
    reachable `E^{ij}_t`, `t <= i + j`, `SUM_{i <= l1, j <= l2} (i + j +
    1)` (`axis_table_extent`'s dense `(l1+1)(l2+1)(l1+l2+1)` less its
    structural zeros: 125 against 225 doubles at `(4, 4)`)."""
    return sum(i + j + 1 for i in range(l1 + 1) for j in range(l2 + 1))


def compact_offset(i: int, j: int, l2: int) -> int:
    """`E^{ij}_0`'s slot in a compact table: the rows `(i', j')` before
    `(i, j)`, row-major (`j` fastest), each `i' + j' + 1` long."""
    return ((l2 + 1) * i * (i + 1) // 2 + i * l2 * (l2 + 1) // 2
            + j * (i + 1) + j * (j - 1) // 2)


def _offset_expr(i: Expr, j: Expr, l2: int) -> Expr:
    """`compact_offset` as an I32 expression (each division exact)."""
    tri_i = _c(Fn.DIV, _mul(i, _iadd(i, Lit(1))), Lit(2))
    tri_j = _c(Fn.DIV, _mul(j, _c(Fn.SUB, j, Lit(1))), Lit(2))
    return _iadd(_iadd(_iadd(_mul(Lit(l2 + 1), tri_i),
                             _mul(i, Lit(l2 * (l2 + 1) // 2))),
                       _mul(j, _iadd(i, Lit(1)))), tri_j)


def _restored(node, at):
    """`node` (an IR dataclass, or a tuple of them) with every `Index`
    passed through `at` (which returns it, or its replacement)."""
    if isinstance(node, tuple):
        return tuple(_restored(x, at) for x in node)
    if not is_dataclass(node):
        return node
    out = replace(node, **{f.name: _restored(getattr(node, f.name), at)
                           for f in fields(node)})
    return at(out) if isinstance(out, Index) else out


def tables_fragment(l1: int, l2: int,
                    arrays: tuple[str, str, str]) -> Fragment:
    """`e_fragment.axis_tables_fragment(l1, l2)`, its recurrence statement
    for statement, every entry stored COMPACT (`compact_offset(i, j, l2) +
    t`) into `arrays`: the ket's copy names its own (`ir`'s hygiene refuses
    binding the host's `Ex Ey Ez` to another name)."""
    frag = ef.axis_tables_fragment(l1, l2)
    names = dict(zip(ef.AXIS_ARRAYS, arrays, strict=True))
    L = l1 + l2

    def at(ix: Index) -> Index:
        if ix.array not in names:
            return ix
        (n,) = ix.at
        assert isinstance(n, Lit), n
        ij, t = divmod(n.value, L + 1)
        i, j = divmod(ij, l2 + 1)
        return Index(names[ix.array], (Lit(compact_offset(i, j, l2) + t),))

    return Fragment(f"{frag.name}_compact_{arrays[0]}",
                    tuple(names.get(b, b) for b in frag.binds),
                    _restored(frag.body, at))


def _table_sum(arrays: tuple[str, str, str], l2: int,
               x: tuple[Expr, Expr, Expr], y: tuple[Expr, Expr, Expr],
               read, dst: str, tag: str) -> tuple[Stmt, ...]:
    """`dst = SUM_t E^{xy}_t X(t)` over the reachable `t` of one row, rolled
    over compact tables `arrays` and factored per axis as
    `e_fragment.rolled_e_sum` is,

        SUM_tx Ex[x y tx] (SUM_ty Ey[x y ty] (SUM_tz Ez[x y tz] X(t)))

    each fold from `+0.0`; `read(t)` the consumer's read at the three I32
    components. Declares `dst` and the locals `{tag}_*`."""
    t = (Var(f"{tag}_tx"), Var(f"{tag}_ty"), Var(f"{tag}_tz"))
    base = [f"{tag}_b{a}" for a in "xyz"]
    sy, sz = f"{tag}_sy", f"{tag}_sz"

    def e(k: int) -> Index:
        return Index(arrays[k], (_iadd(Var(base[k]), t[k]),))

    def acc(nm: str, value: Expr) -> Assign:
        return Assign(Var(nm), _add(Var(nm), value))

    def top(k: int) -> Expr:
        return _iadd(_iadd(x[k], y[k]), Lit(1))

    z_loop = For(t[2].name, Lit(0), top(2), Lit(1), (
        acc(sz, _mul(e(2), read(t))),))
    y_loop = For(t[1].name, Lit(0), top(1), Lit(1), (
        Decl(sz, Scalar.F64, init=Lit(0.0)), z_loop,
        acc(sy, _mul(e(1), Var(sz)))))
    x_loop = For(t[0].name, Lit(0), top(0), Lit(1), (
        Decl(sy, Scalar.F64, init=Lit(0.0)), y_loop,
        acc(dst, _mul(e(0), Var(sy)))))
    return (
        *(Decl(base[k], Scalar.I32, init=_offset_expr(x[k], y[k], l2))
          for k in range(3)),
        Decl(dst, Scalar.F64, init=Lit(0.0)),
        x_loop,
    )


def _ket_signs(lb: int, ld: int) -> For:
    """`F_k[i, j, t] = -F_k[i, j, t]` for every odd `t`, on each axis: the
    ket sign `(-1)^|q| = PROD_k (-1)^{q_k}` folded into the compact tables
    once per ket record (a negation, exact)."""
    i, j, t = Var("si"), Var("sj"), Var("st")
    at = _iadd(_offset_expr(i, j, ld), t)
    flips = tuple(
        Assign(Index(KET_TABLES[k], (at,)),
               _mul(Lit(-1.0), Index(KET_TABLES[k], (at,))))
        for k in range(3))
    return For("si", Lit(0), Lit(lb + 1), Lit(1), (
        For("sj", Lit(0), Lit(ld + 1), Lit(1), (
            For("st", Lit(1), _iadd(_iadd(i, j), Lit(1)), Lit(2), flips),)),))


def _build_lowstack(am: Am, name: str | None) -> Kernel:
    """`build_gk_kernel`'s kernel ROLLED (module docstring, THE ROLLED
    FORM): the same parameters, launch and `G` contract."""
    la, lc, lb, ld = am
    lp, lq = la + lc, lb + ld
    L1 = lp + lq + 1
    (dag,) = builder_plan(gk_pq_spec(la, lc, lb, ld)).dags
    order = tuple(schedule(dag))
    targets = frozenset(hermite_targets(L1))
    held = {r for (r, m) in order if m == 0}
    assert targets <= held, "the recurrence never forms every [r]^(0)"
    bra, ket = Var("bra"), Var(KET)
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    rec = store_targets((*prelude, *recurrence_body(dag, list(order),
                                                    fundamental)),
                        targets, L1, R_ARRAY)

    frag_ac = tables_fragment(la + 1, lc + 1, ef.AXIS_ARRAYS)
    binds_ac = {k: v for k, v in bindings(frag_ac, Var("ze"), "XPA", "XPC",
                                          bra).items() if k in frag_ac.binds}
    for arr in ef.AXIS_ARRAYS:
        binds_ac[arr] = Var(arr)
    frag_bd = tables_fragment(lb, ld, KET_TABLES)
    binds_bd = {k: v for k, v in bindings(frag_bd, Var("et"), "XQB", "XQD",
                                          ket).items() if k in frag_bd.binds}
    for arr in KET_TABLES:
        binds_bd[arr] = Var(arr)

    def i_read(t: tuple[Expr, Expr, Expr]) -> Expr:
        return Index(I_ARRAY, (ef.hermite_index(t),))

    def moved(v: tuple[Expr, Expr, Expr], k: int, by: int):
        step = _iadd(v[k], Lit(1)) if by > 0 else _c(Fn.SUB, v[k], Lit(1))
        return tuple(step if d == k else v[d] for d in range(3))

    def stage1(b, d):
        def per_p(p):
            def r_at(q):
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(p[x], q[x]) for x in range(3))),))
            return (*_table_sum(KET_TABLES, ld, b, d, r_at, "s1", "k"),
                    Assign(Index(I_ARRAY, (ef.hermite_index(p),)),
                           _mul(Var("sq"), Var("s1"))))
        return ef.hermite_loops(lp + 1, per_p, tag="h")

    def row_ac(b, d):
        def row_a(a):
            def elem(c):
                out: list[Stmt] = [
                    Decl("rc", Scalar.I32, init=ef.cartesian_index("c")),
                    Decl("wt", Scalar.F64, init=_mul(
                        Index("M", (_iadd(Var("oa"), Var("ra")),
                                    _iadd(Var("ob"), Var("rb")))),
                        Index("M", (_iadd(Var("oc"), Var("rc")),
                                    _iadd(Var("od"), Var("rd")))))),
                ]
                for side, two in (("A", "ea2"), ("C", "ec2")):
                    x = a if side == "A" else c
                    for k in range(3):
                        up, lo = moved(x, k, 1), moved(x, k, -1)
                        ua, uc = (up, c) if side == "A" else (a, up)
                        wa, wc = (lo, c) if side == "A" else (a, lo)
                        hi, dn = f"h{side}{k}", f"l{side}{k}"
                        out += [
                            *_table_sum(ef.AXIS_ARRAYS, lc + 1, ua, uc,
                                        i_read, hi, f"p{side}{k}"),
                            Decl(dn, Scalar.F64, init=Lit(0.0)),
                            If(_c(Fn.LT, Lit(0), x[k]), (
                                *_table_sum(ef.AXIS_ARRAYS, lc + 1, wa, wc,
                                            i_read, f"m{side}{k}",
                                            f"q{side}{k}"),
                                Assign(Var(dn), _mul(x[k],
                                                     Var(f"m{side}{k}"))),
                            )),
                            Assign(Var(f"g{side}{k}"), _add(
                                Var(f"g{side}{k}"), _mul(Var("wt"), _c(
                                    Fn.SUB, _mul(Var(two), Var(hi)),
                                    Var(dn))))),
                        ]
                return tuple(out)
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lc, ("cx", "cy", "cz"), elem, tag="c"))
        return ef.cartesian_loops(la, ("ax", "ay", "az"), row_a, tag="a")

    def row_b(b):
        def row_d(d):
            return (
                Decl("rd", Scalar.I32, init=ef.cartesian_index("d")),
                Note(f"stage 1: I[p] = s_bd SUM_q (-1)^|q| E^bd_q [p+q], "
                     f"|p| <= {lp + 1}, the sign in the ket tables"),
                stage1(b, d),
                Note("per (a,c) row, W = D_ab D_cd from M: gA_k += W (2 "
                     "alpha SUM E^{a+1_k,c} I - a_k SUM E^{a-1_k,c} I), gC "
                     "likewise on c, each a rolled sum over the bra tables"),
                row_ac(b, d),
            )
        return (Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
                ef.cartesian_loops(ld, ("dx", "dy", "dz"), row_d, tag="d"))

    pair: list[Stmt] = [
        *(Decl(n, Scalar.F64,
               init=_c(Fn.SUB, Var(f"P{d}"), Index("Q", (ket, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Decl("et", Scalar.F64, init=Index("eta", (ket,))),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ze"), Var("et")),
            _add(Var("ze"), Var("et"))))),
        Note(f"[r]^(0) by the plan's DAG for the d0 K class ({la + 1},{lc}|"
             f"{lb},{ld}) at L + 1 = {L1}, each stored into R as it forms"),
        *rec,
        Note("the undressed per-axis E^bd of this ket record: "
             "e_fragment.axis_tables_fragment(lb, ld), inlined into Fx Fy Fz"),
        Inline(frag_bd, binds_bd),
        Note("the ket sign (-1)^|q|: every odd-t entry negated, per axis"),
        _ket_signs(lb, ld),
        Decl("sq", Scalar.F64, init=Index("s_bd", (ket,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (ket,))),
        Decl("od", Scalar.I32, init=Index("off_d", (ket,))),
        ef.cartesian_loops(lb, ("bx", "by", "bz"), row_b, tag="b"),
    ]
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
        Note(LOWSTACK_NOTE),
        Note(f"the undressed per-axis E^{{ac}}_t, i <= {la + 1}, "
             f"j <= {lc + 1}: one index above the class on each side"),
        *(Decl(arr, Scalar.F64, (Lit(compact_extent(la + 1, lc + 1)),))
          for arr in ef.AXIS_ARRAYS),
        Inline(frag_ac, binds_ac),
        *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
          for s in "AC" for k in range(3)),
        Decl(R_ARRAY, Scalar.F64, (Lit(len(targets)),)),
        Decl(I_ARRAY, Scalar.F64, (Lit(len(hermite_targets(lp + 1))),)),
        *(Decl(arr, Scalar.F64, (Lit(compact_extent(lb, ld)),))
          for arr in KET_TABLES),
    ]
    tail: list[Stmt] = [Note("reduce across the warp: every lane sees the "
                             "sum")]
    for side in "AC":
        for k in range(3):
            tail += [Decl(f"r{side}{k}", Scalar.F64),
                     WarpReduce(Var(f"r{side}{k}"), Var(f"g{side}{k}"))]
    tail += [
        Note(WRITE_NOTE),
        If(_c(Fn.LT, Call(Fn.THREAD_Y), Lit(1)), tuple(
            AtomicAdd(Index("G", (Var(atom), Lit(k))),
                      _mul(Var("sb"), Var(f"r{side}{k}")))
            for side, atom in (("A", "aa"), ("C", "ac")) for k in range(3))),
    ]
    per_bra: list[Stmt] = [
        *head,
        Note("kets strip-mined over threadIdx.y, stride blockDim.y"),
        For(KET, Call(Fn.THREAD_Y), Var("n_ket"), Call(Fn.BLOCK_DIM_Y),
            tuple(pair)),
        *tail,
    ]
    body: tuple[Stmt, ...] = (
        Note(f"gradient of E_K = 1/2 SUM D_ab D_cd (ac|bd), K class "
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}), ROLLED (#618): G[X, k] "
             "+= s_ac SUM D_ab D_cd SUM_p (dX_k E^ac)_p SUM_q [p|q] s_bd "
             f"E^bd_q on the bra's two centres, [r] at L = {L1} "
             "(felizjueves.gradient)"),
        Note(ZEROED_G),
        Note("PRECONDITION: D is symmetric (the ket derivative is the bra "
             "derivative only then)"),
        Note(NO_SCREEN),
        Note("launch: blockDim = (1, warp width), one warp per bra record"),
        Decl("bra", Scalar.I32, init=_add(Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    return Kernel(name or gk_name(am), _params(), body)
