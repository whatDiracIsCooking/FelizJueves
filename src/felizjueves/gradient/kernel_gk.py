"""The exchange-energy gradient kernel: `Target.K` at `derivative_order =
1`, the K kernel's bra derivative.

    build_gk_kernel(spec | (la, lc, lb, ld)) -> Kernel   one per K class
    gk_name(spec | (la, lc, lb, ld))         -> str      gk_{la}{lc}_{lb}{ld}
    gk_spec(la, lc, lb, ld)                  -> Spec     the class's d1 key
    gk_pq_spec(la, lc, lb, ld)               -> Spec     the d0 K class whose
                                                         plan it runs
    launch_shape(n_bra, lanes)               -> (grid, (1, lanes))

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

ROUND-OFF, NOT BITWISE, against `gradient.host.exchange_gradient`, which
differentiates `E^{ac}` before the fold and runs `reference.pq`'s `[r]`
(the kernel: raised rows, the plan's MILD recurrence); the stated bound is
in test/test_kernel_gk.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor
`gradient.host` nor the interpreter nor `reference/`.
"""

from __future__ import annotations

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
    "PARAMS",
    "build_gk_kernel",
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


def build_gk_kernel(cls: Spec | Am, *, name: str | None = None) -> Kernel:
    """The exchange-energy gradient kernel of one K class (module
    docstring). `cls` is a d1 `Target.K` `Spec` or `(la, lc, lb, ld)`,
    the Hermite pairing, bra first."""
    la, lc, lb, ld = _class(cls)
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

