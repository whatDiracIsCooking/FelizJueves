"""The exchange-energy Hessian kernel: `Target.K` at `derivative_order =
2`, the K kernel differentiated twice.

    build_hkx_kernel(spec | (la, lc, lb, ld)) -> Kernel   one per K class
    hkx_name(spec | (la, lc, lb, ld))         -> str      hkx_{la}{lc}_{lb}{ld}
    hkx_spec(la, lc, lb, ld)                  -> Spec     the class's d2 key
    hkx_pq_spec(la, lc, lb, ld)               -> Spec     the d0 K class
                                                          whose plan it runs
    launch_shape(n_bra, lanes)                -> (grid, (1, lanes))
    hkx_lowstack(spec | am)                   -> bool     THE ROLLED FORM's gate
    build_hkx_kernel(..., lowstack=True|False)            either form, forced

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

THE ROLLED FORM (#691), the default from `L = la + lc + lb + ld >=
LOWSTACK_MIN_L = 3` and for every class with a d or f shell
(`hkx_lowstack`; `lowstack=` forces either form, the unrolled one only
with every `l <= UNROLLED_L_MAX = 1`; the name, parameters, launch and
`H` contract the same). The unrolled form is 1.6 MB of text at `(pp|pp)`
and over #613's 14,000 B of `sm_86` stack from `L = 3` (`(pp|ps)` 16,936
B, `(pp|pp)` 33,952 B; data/measured/hkx.json). The rolled form is
`kernel_gk`'s #618 shape two orders up, and holds no shifted-row digest:

    per bra:   Ex Ey Ez = the bra's per-axis E, (la + 2, lc + 2), COMPACT
               BB[27] = 0  (kind AA, AC, CC x k x l)
    per segment:  BK[36] = 0  (X x k x Y x l)
      per ket:    [r]^(0), |r| <= L + 2, stored into R as it forms;
                  Fx Fy Fz = the ket's per-axis E, (lb + 1, ld + 1), COMPACT,
                  every odd-t entry negated: (-1)^|q|
        for every (b, d) row:
          I[p] = s_bd SUM_q E^bd_q R[p + q]                   |p| <= lp + 2
          for every (a, c) row:  W = M[a, b] M[c, d]
            for kind; for k, l (run time; l >= k for AA, CC):
              BB[kind k l] += W SUM_{4 rows} coef SUM E^{a'c'} I
          for Y in B, D; for l (run time):
            I[p] = s_bd SUM_q (dY_l E^bd)_q R[p + q]          |p| <= lp + 1
            for every (a, c) row; for k (run time); X in A, C:
              BK[X k Y l] += W (2e SUM E^{x+1_k} I - x_k SUM E^{x-1_k} I)

each `SUM` a loop nest over the compact tables (`kernel_gk._table_sum`),
each row loop a `cartesian_loops` nest, the axes `k`, `l` loop variables
whose unit vectors `1_k`, `1_l` are I32 locals (`uk`, `ul`) the shifted
rows add. A bra-bra component is the rule applied twice at run time, `dX_k`
then `dX'_l`: four shifted rows, the second rule reading the component of
the row the first one moved (`(x + 1_k)_l`), each lowered row guarded by
the component it lowers; at `dX_k dX_k` that is `(2e)^2 [x + 2_k] -
2e (x_k + 1) [x] - 2e x_k [x] + x_k (x_k - 1) [x - 2_k]`, four folds where
the unrolled form digests three. The powers of `2 alpha`, `2 gamma` are
multiplied in per term (the unrolled form keeps one accumulator per
power), so the two forms agree to round-off, not bitwise. The reductions
and atomics are the unrolled form's. So the text is the recurrence plus
25 rolled sums, whatever the class (147 KB at `(pp|ps)`, 361 KB at
`(ff|ff)`, the recurrence at `L + 2 = 14` most of it), and a unit
compiles in seconds (a first draft with the components unrolled in
Python, ~160 rolled sums, took 75-90 s of nvcc a unit). Its frame is
RECORDED, NOT GATED (as #694's d2eri3): `R` at `L + 2` and the bra tables
two indices up are ~11 KB alone at `(ff|ff)`, ~20 KB of `sm_86` stack in
all (data/measured/hkx.json); every default unit launches on both cards
(test/test_hkx_device.py).

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
from felizjueves.gradient.kernel_gk import (
    KET_TABLES,
    _ket_signs,
    _table_sum,
    compact_extent,
    tables_fragment,
)
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
    store_targets,
)
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "BB_KINDS",
    "LOWSTACK_MIN_L",
    "PARAMS",
    "UNROLLED_L_MAX",
    "build_hkx_kernel",
    "hkx_lowstack",
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


def build_hkx_kernel(cls: Spec | Am, *, name: str | None = None,
                     lowstack: bool | None = None) -> Kernel:
    """The exchange-energy Hessian kernel of one K class (module
    docstring). `cls` is a d2 `Target.K` `Spec` or `(la, lc, lb, ld)`,
    the Hermite pairing, bra first. `lowstack=None` takes
    `hkx_lowstack(cls)`'s choice; `True` / `False` force THE ROLLED FORM /
    the unrolled one (the name, parameters, launch and `H` contract the
    same; the unrolled one is built only with every `l <=
    UNROLLED_L_MAX`)."""
    la, lc, lb, ld = _class(cls)
    if lowstack is None:
        lowstack = hkx_lowstack((la, lc, lb, ld))
    if lowstack:
        return _build_lowstack((la, lc, lb, ld), name)
    if max(la, lc, lb, ld) > UNROLLED_L_MAX:
        raise ValueError(f"hkx_{la}{lc}_{lb}{ld}: the unrolled form is "
                         f"built only with every l <= {UNROLLED_L_MAX}")
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


# ---------------------------------------------------------------------------
# The rolled form (#691)
# ---------------------------------------------------------------------------

# the rolled form is the default from L = la + lc + lb + ld = LOWSTACK_MIN_L
# (#691): the smallest L with an unrolled class over 14,000 B of sm_86
# stack (data/measured/hkx.json: hkx_11_10, 16,936 B)
LOWSTACK_MIN_L = 3
# the largest shell l the unrolled form is built for: its text is 1.6 MB at
# (pp|pp) already, so a d class is never attempted unrolled
UNROLLED_L_MAX = 1
R_ARRAY = "R"  # [r]^(0) of one (bra, ket) pair, hermite_targets(L + 2)
I_ARRAY = "I"  # I[p], |p| <= lp + 2, then each J{Y}{l}[p], |p| <= lp + 1

LOWSTACK_NOTE = (
    "ROLLED (#691): the bra's per-axis E tables two indices up and the "
    "ket's one index up, COMPACT (the ket sign (-1)^q folded into their "
    "odd-t entries), [r]^(0) into a local R as it forms, per (b,d) row "
    "I[p] into a local, then each J{Y}{l}[p] into the same local in turn, "
    "and every (a,c) row's derivative terms as rolled sums over the bra "
    "tables, the derivative axes k, l loops at run time and the "
    "accumulators local arrays BB, BK, W = D_ab D_cd read from M where it "
    "is used: no E block, no straight-line fold, no shifted-row digest")


def hkx_lowstack(cls: Spec | Am) -> bool:
    """THE SELECTION RULE (#691): True when `la + lc + lb + ld >=
    LOWSTACK_MIN_L` or a shell is above `UNROLLED_L_MAX` (whose unrolled
    form is never built); there `build_hkx_kernel` emits the rolled form by
    default. False below, where the unrolled text is unchanged."""
    am = _class(cls)
    return sum(am) >= LOWSTACK_MIN_L or max(am) > UNROLLED_L_MAX


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _moved(v: tuple[Expr, Expr, Expr], k: int, by: int):
    """`v + by 1_k` as I32 expressions."""
    if by == 0:
        return v
    step = (_iadd(v[k], Lit(by)) if by > 0
            else _c(Fn.SUB, v[k], Lit(-by)))
    return tuple(step if d == k else v[d] for d in range(3))


def _build_lowstack(am: Am, name: str | None) -> Kernel:
    """`build_hkx_kernel`'s kernel ROLLED (LOWSTACK_NOTE): the same
    parameters, launch and `H` contract as the unrolled form."""
    la, lc, lb, ld = am
    lp, lq = la + lc, lb + ld
    L2 = lp + lq + 2
    (dag,) = builder_plan(hkx_pq_spec(la, lc, lb, ld)).dags
    order = tuple(schedule(dag))
    targets = frozenset(hermite_targets(L2))
    held = {r for (r, m) in order if m == 0}
    assert targets <= held, "the recurrence never forms every [r]^(0)"
    bra, ket, seg = Var("bra"), Var(KET), Var(SEG)
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    rec = store_targets((*prelude, *recurrence_body(dag, list(order),
                                                    fundamental)),
                        targets, L2, R_ARRAY)
    t1, t2 = la + 2, lc + 2  # the bra tables
    u1, u2 = lb + 1, ld + 1  # the ket tables

    frag_ac = tables_fragment(t1, t2, ef.AXIS_ARRAYS)
    binds_ac = {k: v for k, v in bindings(frag_ac, Var("ze"), "XPA", "XPC",
                                          bra).items() if k in frag_ac.binds}
    for arr in ef.AXIS_ARRAYS:
        binds_ac[arr] = Var(arr)
    frag_bd = tables_fragment(u1, u2, KET_TABLES)
    binds_bd = {k: v for k, v in bindings(frag_bd, Var("et"), "XQB", "XQD",
                                          ket).items() if k in frag_bd.binds}
    for arr in KET_TABLES:
        binds_bd[arr] = Var(arr)

    def i_read(t: tuple[Expr, Expr, Expr]) -> Expr:
        return Index(I_ARRAY, (ef.hermite_index(t),))

    tags = iter(f"t{n}" for n in range(10 ** 6))

    def shifted(x, s):
        out = x
        for d in range(3):
            out = _moved(out, d, s[d])
        return out

    def ac_loop(per_row) -> For:
        def row_a(a):
            def elem(c):
                return (
                    Decl("rc", Scalar.I32, init=ef.cartesian_index("c")),
                    Decl("wt", Scalar.F64, init=_mul(
                        Index("M", (_iadd(Var("oa"), Var("ra")),
                                    _iadd(Var("ob"), Var("rb")))),
                        Index("M", (_iadd(Var("oc"), Var("rc")),
                                    _iadd(Var("od"), Var("rd")))))),
                    *per_row(a, c),
                )
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lc, ("cx", "cy", "cz"), elem, tag="c"))
        return ef.cartesian_loops(la, ("ax", "ay", "az"), row_a, tag="a")

    def ket_fold(b, d, L: int, sb, sd) -> For:
        """`I[p] = s_bd SUM_q (-1)^|q| E^{b'd'}_q R[p + q]`, `|p| <= L`, the
        ket row shifted by `(sb, sd)`."""
        def per_p(p):
            def r_at(q):
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(p[x], q[x]) for x in range(3))),))
            tag = next(tags)
            return (*_table_sum(KET_TABLES, u2, shifted(b, sb),
                                shifted(d, sd), r_at, f"{tag}_v", tag),
                    Assign(Index(I_ARRAY, (ef.hermite_index(p),)),
                           _mul(Var("sq"), Var(f"{tag}_v"))))
        return ef.hermite_loops(L, per_p, tag="h")

    def unit(arr: str, axis: Expr) -> tuple[Stmt, ...]:
        """`arr = 1_axis`, an I32 local of three, `axis` an I32 at run time."""
        return (*(Assign(Index(arr, (Lit(d),)), Lit(0)) for d in range(3)),
                Assign(Index(arr, (axis,)), Lit(1)))

    def along(row, arr: str, by: int):
        """`row + by arr`, componentwise, `arr` a unit local."""
        fn = Fn.ADD if by > 0 else Fn.SUB
        return tuple(_c(fn, row[d], Index(arr, (Lit(d),))) for d in range(3))

    def neg(x: Expr) -> Expr:
        return _c(Fn.SUB, Lit(0), x)

    def rolled_sum(a, c, coef: Expr, guards: list[Expr], acc: str
                   ) -> tuple[Stmt, ...]:
        """`acc += coef SUM_t E^{ac}_t I(t)` over the bra tables, under
        `guards` (each an I32 that must be positive)."""
        tag = next(tags)
        body: tuple[Stmt, ...] = (
            *_table_sum(ef.AXIS_ARRAYS, t2, a, c, i_read, f"{tag}_v", tag),
            Assign(Var(acc), _add(Var(acc), _mul(coef, Var(f"{tag}_v")))),
        )
        for g in reversed(guards):
            body = (If(_c(Fn.LT, Lit(0), g), body),)
        return body

    rows = {"A": "va", "C": "vc"}  # the (a, c) row's components, I32 locals
    two = {"A": "ea2", "C": "ec2"}

    def bb_row(a, c):
        """Every bra-bra component of one (a, c) row, the two derivatives
        `dX_k`, `dX'_l` composed at run time: four shifted rows, each a
        rolled sum, the lowered ones guarded."""
        base = {"A": a, "C": c}
        out: list[Stmt] = [
            *(Assign(Index("va", (Lit(d),)), a[d]) for d in range(3)),
            *(Assign(Index("vc", (Lit(d),)), c[d]) for d in range(3)),
        ]
        for n, kind in enumerate(BB_KINDS):
            s1, s2 = kind[0], kind[1]
            kk, ll = Var("kk"), Var("ll")
            inner: list[Stmt] = [*unit("ul", ll),
                                 Decl("vbb", Scalar.F64, init=Lit(0.0))]
            for by1 in (1, -1):
                row1 = dict(base)
                row1[s1] = along(base[s1], "uk", by1)
                x1 = Index(rows[s1], (kk,))
                c1: Expr = Var(two[s1]) if by1 > 0 else neg(x1)
                g1 = [] if by1 > 0 else [x1]
                for by2 in (1, -1):
                    row2 = dict(row1)
                    row2[s2] = along(row1[s2], "ul", by2)
                    y: Expr = Index(rows[s2], (ll,))
                    if s1 == s2:
                        y = _c(Fn.ADD if by1 > 0 else Fn.SUB, y,
                               Index("uk", (ll,)))
                    c2: Expr = Var(two[s2]) if by2 > 0 else neg(y)
                    g2 = [] if by2 > 0 else [y]
                    inner += rolled_sum(row2["A"], row2["C"], _mul(c1, c2),
                                        g1 + g2, "vbb")
            slot = Index("BB", (_iadd(Lit(9 * n), _iadd(_mul(Lit(3), kk),
                                                         ll)),))
            inner.append(Assign(slot, _add(slot, _mul(Var("wt"),
                                                      Var("vbb")))))
            out.append(For("kk", Lit(0), Lit(3), Lit(1), (
                *unit("uk", kk),
                For("ll", kk if s1 == s2 else Lit(0), Lit(3), Lit(1),
                    tuple(inner)))))
        return tuple(out)

    def bk_row(Y: str):  # noqa: N803
        """Per (a, c) row, the bra's first derivatives `dX_k` (k at run
        time) against `J{Y}{l}` (in I), into `BK`."""
        yi = "BD".index(Y)

        def per_row(a, c):
            base = {"A": a, "C": c}
            kk = Var("kk")
            inner: list[Stmt] = [
                *unit("uk", kk),
            ]
            for si, side in enumerate("AC"):
                x = Index(rows[side], (kk,))
                inner.append(Decl(f"v{side}", Scalar.F64, init=Lit(0.0)))
                for by in (1, -1):
                    row = dict(base)
                    row[side] = along(base[side], "uk", by)
                    coef = Var(two[side]) if by > 0 else neg(x)
                    inner += rolled_sum(row["A"], row["C"], coef,
                                        [] if by > 0 else [x], f"v{side}")
                slot = Index("BK", (_iadd(Lit(18 * si + 3 * yi), _iadd(
                    _mul(Lit(6), kk), Var("ll"))),))
                inner.append(Assign(slot, _add(slot, _mul(
                    Var("wt"), Var(f"v{side}")))))
            return (
                *(Assign(Index("va", (Lit(d),)), a[d]) for d in range(3)),
                *(Assign(Index("vc", (Lit(d),)), c[d]) for d in range(3)),
                For("kk", Lit(0), Lit(3), Lit(1), tuple(inner)),
            )
        return per_row

    def j_fold(b, d, Y: str) -> For:  # noqa: N803
        """`J{Y}{l}[p] = s_bd SUM_q (-1)^|q| (dY_l E^{bd})_q R[p + q]` into
        I, `|p| <= lp + 1`, `l` at run time (`ul`): the rule's two rows,
        the lower one guarded."""
        e2 = "eb2" if Y == "B" else "ed2"
        n = Index("vb" if Y == "B" else "vd", (Var("ll"),))

        def per_p(p):
            def r_at(q):
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(p[x], q[x]) for x in range(3))),))
            tu, tl = next(tags), next(tags)
            ub, ud = ((along(b, "ul", 1), d) if Y == "B"
                      else (b, along(d, "ul", 1)))
            lb_, ld_ = ((along(b, "ul", -1), d) if Y == "B"
                        else (b, along(d, "ul", -1)))
            return (
                *_table_sum(KET_TABLES, u2, ub, ud, r_at, f"{tu}_v", tu),
                Decl(f"{tl}_w", Scalar.F64, init=Lit(0.0)),
                If(_c(Fn.LT, Lit(0), n), (
                    *_table_sum(KET_TABLES, u2, lb_, ld_, r_at, f"{tl}_v",
                                tl),
                    Assign(Var(f"{tl}_w"), _mul(n, Var(f"{tl}_v"))))),
                Assign(Index(I_ARRAY, (ef.hermite_index(p),)), _mul(
                    Var("sq"), _c(Fn.SUB, _mul(Var(e2), Var(f"{tu}_v")),
                                  Var(f"{tl}_w")))),
            )
        return ef.hermite_loops(lp + 1, per_p, tag="h")

    def row_b(b):
        def row_d(d):
            out: list[Stmt] = [
                Decl("rd", Scalar.I32, init=ef.cartesian_index("d")),
                *(Assign(Index("vb", (Lit(x),)), b[x]) for x in range(3)),
                *(Assign(Index("vd", (Lit(x),)), d[x]) for x in range(3)),
                Note(f"I[p] = s_bd SUM_q (-1)^|q| E^bd_q [p+q], |p| <= "
                     f"{lp + 2}, the sign in the ket tables"),
                ket_fold(b, d, lp + 2, ZERO, ZERO),
                Note("the bra-bra term: per (a,c) row, W = D_ab D_cd from "
                     "M; per kind (AA, AC, CC) and axes (k, l) at run time, "
                     "the rule twice as four shifted rows, rolled sums over "
                     "the bra tables against I, into BB"),
                ac_loop(bb_row),
            ]
            for Y in "BD":
                out.append(For("ll", Lit(0), Lit(3), Lit(1), (
                    *unit("ul", Var("ll")),
                    Note(f"J{Y}l[p] = s_bd SUM_q (-1)^|q| (d{Y}_l E^bd)_q "
                         f"[p+q], |p| <= {lp + 1}, into I"),
                    j_fold(b, d, Y),
                    Note(f"the bra-ket term against d{Y}_l: per (a,c) row "
                         "and axis k at run time, the bra's first "
                         "derivatives against J, into BK"),
                    ac_loop(bk_row(Y)),
                )))
            return tuple(out)
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
        Note(f"[r]^(0) by the plan's DAG for the d0 K class ({la + 2},{lc}|"
             f"{lb},{ld}) at L + 2 = {L2}, each stored into R as it forms"),
        *rec,
        Note(f"the undressed per-axis E^bd of this ket record, i <= {u1}, "
             f"j <= {u2}, COMPACT, inlined into Fx Fy Fz"),
        Inline(frag_bd, binds_bd),
        Note("the ket sign (-1)^|q|: every odd-t entry negated, per axis"),
        _ket_signs(u1, u2),
        Decl("sq", Scalar.F64, init=Index("s_bd", (ket,))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (ket,)))),
        Decl("ed2", Scalar.F64, init=_mul(Lit(2.0), Index("ed_ket", (ket,)))),
        Decl("ob", Scalar.I32, init=Index("off_b", (ket,))),
        Decl("od", Scalar.I32, init=Index("off_d", (ket,))),
        ef.cartesian_loops(lb, ("bx", "by", "bz"), row_b, tag="b"),
    ]

    def bk_at(side: str, k: int, Y: str, l: int) -> Index:  # noqa: E741, N803
        return Index("BK", (Lit(18 * "AC".index(side) + 6 * k
                                + 3 * "BD".index(Y) + l),))

    def bb_at(kind: str, k: int, l: int) -> Index:  # noqa: E741
        return Index("BB", (Lit(9 * BB_KINDS.index(kind) + 3 * k + l),))

    bk_tail: list[Stmt] = [Note("the bra-ket values, reduced across the "
                                "warp")]
    bk_adds: list[Stmt] = []
    for side, at in (("A", "aa"), ("C", "ac")):
        for k in range(3):
            for Y, yat in (("B", "yb"), ("D", "yd")):
                for l in range(3):  # noqa: E741
                    r = f"r{side}{k}{Y}{l}"
                    bk_tail += [Decl(r, Scalar.F64), WarpReduce(
                        Var(r), bk_at(side, k, Y, l))]
                    bk_adds.append(AtomicAdd(
                        Index("H", (Var(at), Lit(k), Var(yat), Lit(l))),
                        _mul(Var("sb"), Var(r))))
    seg_body: tuple[Stmt, ...] = (
        Decl("yb", Scalar.I32, init=Index("seg_b", (seg,))),
        Decl("yd", Scalar.I32, init=Index("seg_d", (seg,))),
        Note("the bra-ket accumulators of this segment, BK[X k Y l], the "
             "powers of 2 alpha (2 gamma) taken in per term"),
        For("zk", Lit(0), Lit(36), Lit(1), (
            Assign(Index("BK", (Var("zk"),)), Lit(0.0)),)),
        Note("this segment's kets strip-mined over threadIdx.y, stride "
             "blockDim.y"),
        For(KET, _add(Index("seg_start", (seg,)), Call(Fn.THREAD_Y)),
            Index("seg_start", (_add(seg, Lit(1)),)), Call(Fn.BLOCK_DIM_Y),
            tuple(pair)),
        *bk_tail,
        Note("AtomicAdd by lane 0 into the bra's and the segment's atoms"),
        If(_c(Fn.LT, Call(Fn.THREAD_Y), Lit(1)), tuple(bk_adds)),
    )

    bb_tail: list[Stmt] = [Note("the bra-bra values, reduced across the "
                                "warp")]
    for kind, k, l in _bb_components():  # noqa: E741
        r = f"r{kind}{k}{l}"
        bb_tail += [Decl(r, Scalar.F64),
                    WarpReduce(Var(r), bb_at(kind, k, l))]

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
        Note(f"the undressed per-axis E^{{ac}}_t, i <= {t1}, j <= {t2}: two "
             "indices above the class on each side, COMPACT"),
        *(Decl(arr, Scalar.F64, (Lit(compact_extent(t1, t2)),))
          for arr in ef.AXIS_ARRAYS),
        Inline(frag_ac, binds_ac),
        Note("the bra-bra accumulators BB[kind k l] (AA, AC, CC; the AA and "
             "CC ones at k <= l only), and the bra-ket ones BK[X k Y l]"),
        Decl("BB", Scalar.F64, (Lit(27),)),
        For("zb", Lit(0), Lit(27), Lit(1), (
            Assign(Index("BB", (Var("zb"),)), Lit(0.0)),)),
        Decl("BK", Scalar.F64, (Lit(36),)),
        Note("the rows' components and the run-time unit vectors 1_k, 1_l"),
        *(Decl(v, Scalar.I32, (Lit(3),))
          for v in ("va", "vc", "vb", "vd", "uk", "ul")),
        Decl(R_ARRAY, Scalar.F64, (Lit(len(targets)),)),
        Decl(I_ARRAY, Scalar.F64, (Lit(len(hermite_targets(lp + 2))),)),
        *(Decl(arr, Scalar.F64, (Lit(compact_extent(u1, u2)),))
          for arr in KET_TABLES),
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
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}), ROLLED (#691): H[X, k, "
             "Y, l] += s_ac SUM D_ab D_cd [d_bra d_bra + d_bra d_ket] "
             f"(ac|bd), [r] at L = {L2} (felizjueves.hessian)"),
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
    return Kernel(name or hkx_name(am), _params(), body)
