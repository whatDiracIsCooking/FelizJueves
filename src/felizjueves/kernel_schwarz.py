"""The Schwarz kernel, `Target.SCHWARZ`: `sqrt((mn|mn))` for every AO pair.

    build_schwarz_kernel(spec | (la, lb)) -> Kernel     one per pair class
    schwarz_spec(la, lb)                  -> Spec       the class's key
    schwarz_plan(spec | (la, lb))         -> (dag, order)  the plan's answer
    launch_shape(n_pair, threads)         -> (grid, block)

`(mn|mn)` is the component diagonal of the quartet `(la lb|la lb)` of one shell
pair against itself, by McMurchie & Davidson's Hermite path (`ao/eri.py`'s
module docstring, its witness):

    (mn|mn) = SUM_{bra rec i} SUM_{ket rec k} s_i s_k
              SUM_p E^{mn}_p(i) SUM_q (-1)^{|q|} [p+q]^(0)(P_i - P_k) E^{mn}_q(k)

with BOTH record sums over the SAME shell pair's primitive pairs. Then
`Q[m, n] = sqrt(max((mn|mn), 0))` (`ao/schwarz.py`, "THE CLAMP").

A SCHWARZ CLASS IS A PAIR THAT RUNS A QUARTET (`spec.py`). The key is the pair
`(la, lb)`; the recurrence is the `Target.J` quartet `(la, lb | la, lb)`'s, at
`L = 2 (la + lb)`: `schwarz_plan` asks `pipeline.plan` for THAT spec, and
`recurrence.fundamental_prelude` / `recurrence_body` spell its one DAG in
`schedule`'s order, unchanged -- the statements `kernel.build_pq_kernel` emits
for the same quartet, minus the dead ones (below).

ONLY THE DIAGONAL IS READ, SO MOST OF THE TOWER IS DEAD. Row `(a, b)` reaches
`p` (and `q`) only where `p_d <= a_d + b_d` on every axis -- the `E` block's
structural zeros -- so it reads `[p+q]` only with `(p+q)_d <= 2 (a_d + b_d)`.
`(sp|sp)` never reads `[1,1,0]`, say, which `(sx|sy)` would. The DAG builds
every target of the quartet; the ones no row reads, and every intermediate,
fundamental and Boys weight only they read, are DROPPED by one backward
liveness pass (`_live`, `e_fragment._live`'s rule), because a declared but
unread local is a `#177-D` on nvcc and an `-Wunused-variable` on hipcc (#92's
filter). Dropping a dead `Decl` changes no value that is kept.

THE THREAD MAPPING: ONE THREAD PER CONTRACTED SHELL PAIR, `pair = THREAD_X +
BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`. The thread runs the double
loop over its pair's record range `[rec0, rec1)` serially -- bra outer, the bra
`E` block formed once per bra record; ket inner, the ket `E` block once per
(bra, ket) -- and accumulates one `acc` per row in registers. No reduction and
no atomics: each element of the matrix belongs to exactly one shell pair, so it
is written ONCE, by a plain store, and the result is bit-reproducible run to run.

`E` IS `e_fragment.block_fragment(la, lb)`, INLINED TWICE -- into `Ebra` and
`Eket`, the two local arrays (kernel_k's pattern: a fragment's locals in the
ket loop shadow the bra's, and only the arrays are read). Every `Ebra[r, p]`
and `Eket[r, q]` read is a structurally nonzero entry, which the fragment
writes.

THE CONTRACTION, per (bra, ket) and row `r = (a, b)`:

    v_r = SUM_p Ebra[r, p] * (SUM_q +-[p+q]^(0) Eket[r, q])     p, q reachable
    acc_r = acc_r + (s_i s_k) * v_r

each sum a left fold in `hermite_order` order from its first term, a term with
odd `|q|` SUBTRACTED. NO PREFACTOR: `(pi/zeta)^{3/2}` and the contraction
coefficients are inside the host's `s` (`ao.pairs.dress`), `(2/sqrt(pi))
theta` inside `[0]^(m)` (the prelude's `W0`), and their product is McMurchie
& Davidson's `lambda`.

ROUND-OFF, NOT BITWISE, against `ao.schwarz`: the host's `[p|q]` is
`reference.pq`'s fixed recursion, the kernel's the plan's MILD DAG, and the
host dresses each `E` entry where the kernel multiplies `s_i s_k` once.

THE SCATTER, `kernel_jab`'s rule with a plain store: an off-diagonal shell
pair (`off_a != off_b`) writes row `(a, b)` to `Q[off_a + a, off_b + b]` and
its mirror; a diagonal one writes its lower triangle `a >= b`, mirrored. A
staged pair has `la <= lb` (`ao.schwarz.schwarz_classes`), so the ladder needs
no `la > lb` class.

THE WIDE BOYS (#266). `T = theta^2 |P_i - P_k|^2` reaches past the narrow
series' overflow near 709 whenever two tight primitive pairs of one shell pair
sit apart (`alpha = 1000` on A against `beta = 1` and `beta = 1000` on B two
bohr away is `T ~ 2000`), so the kernel sets `Kernel.wide_boys`, as `V` does.

    params  n_pair n_rec n_ao (I32)
            zeta_ab  s_ab  XP YP ZP  XPA YPA ZPA  XPB YPB ZPB
                                                  [n_rec]  per record
            rec0  rec1  off_a  off_b  [n_pair]  (I32)  per shell pair
            schwarz_bound[n_ao, n_ao]  (OUT, written once per element)

THE RECORDS ARE `PrimPairSoA` (#422, `prim_pair.py`): one flat array per
scalar and per axis of every vector, carried across the call by value in
the SAME struct `S`, `T`, `V` and `SX` take, so the signature is

    schwarz_11(int n_pair, int n_rec, int n_ao, PrimPairSoA pp,
               const int* rec0, ..., double* schwarz_bound)

The fields bound are `zeta` (as `zeta_ab`), `s` (as `s_ab`, `V`'s name, for
`V`'s reason: the prelude declares a local `s`) and `XP`..`ZPB`; `eb` is
unread. `off_a` / `off_b` are NOT bound, though the struct has fields of
those names: there they are PER RECORD (`n_rec` long, a one-electron thread
owns one record), here PER SHELL PAIR (`n_pair` long, indexed by `pair`,
beside `rec0` / `rec1`, which the struct does not carry). Binding them would
put an `n_pair` array in an `n_rec` field, so one struct value could no
longer serve every family. Whether the split helps this kernel's loads --
one thread per shell pair walking its records serially, not one per record
-- is unmeasured; nothing about the kernel is.

`schwarz_bound` is symmetric and every element of the pairs launched is
written, so `schwarz_bound[m + n * n_ao]` and `schwarz_bound[n + m * n_ao]` are
the same value. Elements of pairs NOT launched are untouched: a caller that
launches a subset zeroes first.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves import e_fragment as ef
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
    Weighted,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.prim_pair import AXES, bundled
from felizjueves.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "BRA",
    "KET",
    "OUT",
    "PAIR",
    "build_schwarz_kernel",
    "launch_shape",
    "quartet_spec",
    "schwarz_plan",
    "schwarz_spec",
]

PAIR = "pair"  # one thread per contracted shell pair
BRA = "ib"  # the bra record, serial
KET = "ik"  # the ket record, serial inside the bra
OUT = "schwarz_bound"

# The record arrays arrive in one `PrimPairSoA` (prim_pair.py); the per-pair
# offsets do not (the parameter block's note).
BUNDLED = bundled(*(f"{a}P{c}" for c in ("", "A", "B") for a in AXES),
                  zeta="zeta_ab", s="s_ab")

_DIAGONAL = (
    "diagonal shell pair (off_a == off_b): the lower triangle a >= b only, "
    "each mirrored once -- kernel_jab's diagonal rule, so (mn|mn) and (nm|nm) "
    "are one value")


def schwarz_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.SCHWARZ`."""
    return Spec(Operator.COULOMB, 0, (la, lb), Target.SCHWARZ)


def quartet_spec(cls: Spec | tuple[int, int]) -> Spec:
    """The `Target.J` quartet `(la, lb | la, lb)` whose recurrence the class
    runs -- the key the plan layer is asked for (module docstring)."""
    la, lb = _spec(cls).am
    return Spec(Operator.COULOMB, 0, (la, lb, la, lb), Target.J)


def _spec(cls: Spec | tuple[int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else schwarz_spec(*cls)
    if spec.target is not Target.SCHWARZ:
        raise ValueError(f"{spec.key()} is not a Target.SCHWARZ key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Schwarz kernel takes no levers")
    return spec


def schwarz_plan(cls: Spec | tuple[int, int]) -> tuple[BatchDag, Order]:
    """`pipeline.plan(quartet_spec(cls))`'s one DAG and `schedule(dag)`."""
    (dag,) = plan(quartet_spec(cls)).dags
    return dag, schedule(dag)


def launch_shape(n_pair: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per shell pair,
    `ceil(n_pair/threads)` blocks; a partial last block is legal."""
    if type(n_pair) is not int or n_pair < 1:
        raise ValueError(f"n_pair = {n_pair!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pair // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _reads(e: object) -> set[str]:
    """Every name an expression reads: `Var`s, arrays, `Factor` scalars."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Index):
        return {e.array}.union(*(_reads(i) for i in e.at))
    if isinstance(e, Call):
        return set().union(*(_reads(a) for a in e.args))
    if isinstance(e, Weighted):
        out: set[str] = set()
        for t in e.terms:
            out |= _reads(t.src)
            if t.g.name_in_kernel is not None:
                out.add(t.g.name_in_kernel)
        return out
    return set()  # Lit


def _live(body: tuple[Stmt, ...], wanted: set[str]) -> tuple[Stmt, ...]:
    """`body` (`Decl`s and `Note`s) without the `Decl`s that neither `wanted`
    nor a kept later `Decl` reads: one backward pass."""
    kept: list[Stmt] = []
    live = set(wanted)
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue
            live |= _reads(st.init)
        elif not isinstance(st, Note):
            raise TypeError(f"_live takes Decls and Notes, got "
                            f"{type(st).__name__}")
        kept.append(st)
    return tuple(reversed(kept))


def _plus(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, int, int]:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def _binds(frag, zeta: Expr, at: Expr, array: str) -> dict[str, Expr]:
    """One inlined `E` block's binds: `zeta`, `1/2zeta`, `PA_d`, `PB_d`."""
    out: dict[str, Expr] = {ef.ZETA: zeta, ef.E_ARRAY: Var(array)}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), zeta)
    for d, a in enumerate(AXES):
        out[Factor.pa(d).name_in_kernel] = Index(f"{a}PA", (at,))
        out[Factor.pb(d).name_in_kernel] = Index(f"{a}PB", (at,))
    return out


def build_schwarz_kernel(cls: Spec | tuple[int, int], *,
                         name: str | None = None) -> Kernel:
    """The Schwarz kernel of one pair class (module docstring).

    `cls` is a `Target.SCHWARZ` `Spec` or `(la, lb)`."""
    spec = _spec(cls)
    la, lb = spec.am
    dag, order = schwarz_plan(spec)
    frag = ef.block_fragment(la, lb)
    rows = ef.block_rows(la, lb)
    ts = hermite_targets(la + lb)
    nb = len(ef.cartesians(lb))
    pair, ib, ik = Var(PAIR), Var(BRA), Var(KET)

    # --- the contraction of one (bra, ket) record pair, per row ------------
    contract: list[Stmt] = []
    for r, (a, b) in enumerate(rows):
        reach = [(j, t) for j, t in enumerate(ts)
                 if all(t[d] <= a[d] + b[d] for d in range(3))]
        v: Expr | None = None
        for i, p in reach:
            inner: Expr | None = None
            for j, q in reach:
                term = _c(Fn.MUL, Var(node_name((_plus(p, q), 0))),
                          Index("Eket", (Lit(r), Lit(j))))
                if inner is None:  # q = (0, 0, 0), reachable from every row
                    inner = term
                else:
                    inner = _c(Fn.SUB if sum(q) % 2 else Fn.ADD, inner, term)
            term = _c(Fn.MUL, Index("Ebra", (Lit(r), Lit(i))), inner)
            v = term if v is None else _c(Fn.ADD, v, term)
        contract.append(Assign(Var(f"acc{r}"), _c(
            Fn.ADD, Var(f"acc{r}"), _c(Fn.MUL, Var("w"), v))))
    wanted: set[str] = set()
    for st in contract:
        wanted |= _reads(st.value)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = _live((*prelude, *recurrence_body(dag, order, fundamental)),
                  wanted)

    ket_body: tuple[Stmt, ...] = (
        Decl("et", Scalar.F64, init=Index("zeta_ab", (ik,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index(f"{a}P", (ik,))))
          for d, (n, a) in enumerate(zip(PQ_PARAMS, AXES, strict=True))),
        Note("theta^2 = zeta eta / (zeta + eta), j_engine.md 'Two defects in "
             "the source'"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("et")),
            _c(Fn.ADD, Var("ze"), Var("et"))))),
        Note("[p+q]^(0) by the plan's DAG for the quartet "
             f"({la},{lb}|{la},{lb}), the targets no diagonal row reads "
             "dropped with everything only they read"),
        *tower,
        Note("the undressed E block of this ket record: "
             "e_fragment.block_fragment, inlined"),
        Decl("Eket", Scalar.F64, (Lit(len(rows)), Lit(len(ts)))),
        Inline(frag, _binds(frag, Var("et"), ik, "Eket")),
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                      Index("s_ab", (ik,)))),
        Note("per row (a, b): acc += s_i s_k SUM_p Ebra_p SUM_q +-[p+q] "
             "Eket_q over the structurally nonzero p and q"),
        *contract,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"{a}P", (ib,)))
          for d, a in enumerate(AXES)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the undressed E block of this bra record, once: "
             "e_fragment.block_fragment, inlined"),
        Decl("Ebra", Scalar.F64, (Lit(len(rows)), Lit(len(ts)))),
        Inline(frag, _binds(frag, Var("ze"), ib, "Ebra")),
        Note("the ket records: the SAME shell pair's"),
        For(KET, Var("r0"), Var("r1"), Lit(1), ket_body),
    )

    # --- the clamp, the root, the scatter ----------------------------------
    finish: list[Stmt] = [
        Note("(mn|mn) >= 0 exactly; a rounded one may not be, so clamp "
             "before the root (ao/schwarz.py, THE CLAMP)"),
    ]
    for r in range(len(rows)):
        acc = Var(f"acc{r}")
        finish += [If(_c(Fn.LT, acc, Lit(0.0)), (Assign(acc, Lit(0.0)),)),
                   Decl(f"q{r}", Scalar.F64, init=_c(Fn.SQRT, acc))]

    def at(row: Expr, col: Expr) -> Index:
        return Index(OUT, (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off_diag: list[Stmt] = []
    for i in range(len(ef.cartesians(la))):
        for j in range(nb):
            q = Var(f"q{i * nb + j}")
            off_diag += [Assign(at(plus("oa", i), plus("ob", j)), q),
                         Assign(at(plus("ob", j), plus("oa", i)), q)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    scatter: Stmt
    if la == lb:
        diag: list[Stmt] = [Note(_DIAGONAL)]
        for i in range(nb):
            for j in range(i + 1):
                q = Var(f"q{i * nb + j}")
                diag.append(Assign(at(plus("oa", i), plus("oa", j)), q))
                if i != j:
                    diag.append(Assign(at(plus("oa", j), plus("oa", i)), q))
        scatter = If(differ, tuple(off_diag), tuple(diag))
    else:
        scatter = If(differ, tuple(off_diag))

    per_pair: tuple[Stmt, ...] = (
        Decl("r0", Scalar.I32, init=Index("rec0", (pair,))),
        Decl("r1", Scalar.I32, init=Index("rec1", (pair,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (pair,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (pair,))),
        *(Decl(f"acc{r}", Scalar.F64, init=Lit(0.0)) for r in range(len(rows))),
        Note("the bra records of the pair, serially"),
        For(BRA, Var("r0"), Var("r1"), Lit(1), bra_body),
        *finish,
        Note("scatter: one plain store per element and its mirror, no "
             "atomics -- each element belongs to one shell pair"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"Schwarz factors, class (la, lb) = ({la}, {lb}): "
             "Q[m, n] = sqrt((mn|mn)), the diagonal of the quartet "
             f"({la},{lb}|{la},{lb}) at L = {2 * (la + lb)} "
             "(ao/schwarz.py, ao/eri.py)"),
        Note("launch: one thread per contracted shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    npair, nrec, nao = Var("n_pair"), Var("n_rec"), Var("n_ao")
    params = (
        Param("n_pair", Scalar.I32),
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        f64("zeta_ab", (nrec,)),
        f64("s_ab", (nrec,)),
        *(f64(f"{a}P{c}", (nrec,)) for c in ("", "A", "B") for a in AXES),
        i32("rec0", (npair,)),
        i32("rec1", (npair,)),
        i32("off_a", (npair,)),
        i32("off_b", (npair,)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nao, nao)),
    )
    return Kernel(name or f"schwarz_{la}{lb}", params, body, wide_boys=True,
                  bundled=BUNDLED)
