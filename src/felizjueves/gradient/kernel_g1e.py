"""The one-electron gradient kernels: `Target.S`, `T` and `V` at
`derivative_order = 1`, built as `Kernel`s.

    build_g1e_kernel(spec | (target, la, lb)) -> Kernel   one per class
    g1e_name(spec | (target, la, lb))         -> str      gs_ / gt_ / gv_{la}{lb}
    g1e_spec(target, la, lb)                  -> Spec     the class's d1 key
    launch_shape(n_rec, threads)              -> (grid, block)

Per primitive-pair record of one pair class `(la, lb)` (exponents `alpha`
on `A`, `beta` on `B`, the host's one scalar `s = c_a c_b (pi/zeta)^{3/2}`),
the contribution of the pair's block to

    G[A, k] += SUM_mn M_mn dX_mn / dR_{A,k}

(`felizjueves.gradient`'s package docstring; `gradient.host` is the
reference, one function per target, the same arithmetic in the same rule):

    w_r      = M[oa+i, ob+j] + M[ob+j, oa+i]    off-diagonal shell pair
             = M[oa+i, oa+j]                    diagonal (off_a == off_b)
    E        = e_fragment.axis_tables_fragment(la + 1, lb + 1)    S, V
               e_fragment.axis_tables_fragment(la + 1, lb + 3)    T
    dA E^{ij}_t = 2 alpha E^{i+1,j}_t - i E^{i-1,j}_t   on the axis k
    dB E^{ij}_t = 2 beta  E^{i,j+1}_t - j E^{i,j-1}_t   moved; E elsewhere

- `gs`: `gA_k = SUM_r w_r (f_x f_y) f_z`, `f` the `t = 0` factors with axis
  `k` differentiated; `gB_k` likewise.
- `gt`: the kinetic bracket `K_x S_y S_z + S_x K_y S_z + S_x S_y K_z` of
  `e_fragment.kinetic_fragment`'s `K_ij` (the same expression, its terms in
  its association), the factor on axis `k` replaced by `dA S` or `dA K`,
  `dA K_ij = 2 alpha K_{i+1,j} - i K_{i-1,j}` and `dB K_ij = 2 beta
  K_{i,j+1} - j K_{i,j-1}`.
- `gv`: the record's Hermite densities, ONCE, outside the nucleus loop,

      H0_t   = SUM_r w_r E^{ab}_t                       |t| <= L
      HA_k,t = SUM_r w_r (dA_k E)^{ab}_t, HB likewise   |t| <= L + 1

  then per nucleus `C` (serially, as `kernel_v`): `PQ = P - C`, the plan's
  `[t]^(0)` for every `|t| <= L + 1` (`kernel_v.v_plan((la, lb + 1))`: the
  plan layer at Hermite order `L + 1`, `recurrence_body` unchanged),

      gA_k -= Z_C SUM_t HA_k,t [t]^(0)      gB_k likewise
      G[C, k] += (s Z_C) SUM_t H0_t [t + 1_k]^(0)    Hellmann-Feynman

  (`d/dC_k [t]^(0)(P - C) = -[t + 1_k]^(0)`, times `V`'s `-Z_C`).

Then `G[atom_a, k] += s gA_k` and `G[atom_b, k] += s gB_k`, six
`AtomicAdd`s per record (and three per nucleus for `V`).

THE PARAMETER BLOCK, `kernel_s`'s per-axis spelling plus what a derivative
needs:

    params  n_rec n_ao n_atom (I32)
            zeta_ab  pa_x pa_y pa_z  pb_x pb_y pb_z  s_ab  ea_bra  eb_ket
                [p_x p_y p_z: V]                          [n_rec]  per record
            off_a  off_b  atom_a  atom_b  [n_rec]  (I32)
            M[n_ao, n_ao]                                 the contracted matrix
            [Z[n_atom]  Cnuc[n_atom, 3]: V]               the nuclei = atoms
            G[n_atom, 3]  (OUT, read and AtomicAdd-ed)

- `ea_bra` / `eb_ket` are the two primitive exponents: the derivative rule's
  `2 alpha`, `2 beta` (`kernel_t`'s `eb_ket`, and its bra twin).
- `atom_a` / `atom_b` are the atoms the record's two shells sit on
  (`gradient.host.shell_atoms`); for `V` the atoms ARE the nuclei, row `c`
  of `G` is nucleus `c`'s.
- `M` is read, never written, and need not be symmetric (the weight rule
  is exact for any `M`). `G` is ZEROED BY THE CALLER before a build's first
  gradient launch; every kernel only adds.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec`. The `E` tables are local arrays read at constant indices
(the low-stack path's tables, #586, so the derivative entries share one
recurrence); no `E^{ab}` block is formed.

THE LOW-STACK gv (#614), the default from `la + lb >= LOWSTACK_MIN_LSUM =
5` ((df), (fd), (ff) through f; `g1e_lowstack`; `lowstack=` forces either
form at any `gv` class, `gs` / `gt` refuse `True`; the name, parameters,
launch and `G` contract the same). The unrolled `gv` holds `H0` and the six
derivative densities, `n_t(L) + 6 n_t(L + 1)` locals, across its nucleus
loop, and the pooled derivative factors with them: on `sm_86` 21,104 B of
stack at (df) and 50,016 B at (ff) (140 s of nvcc), over #613's 14,000 B
gradient-family budget. The low-stack form holds none of them:

    per record:  the same head and per-axis tables one index up
    per nucleus: [t]^(0), |t| <= L + 1, into a local R as it forms
                 (recurrence.store_targets); f_k = 0
                 for each block row (a, b), its weight wt from M:
                     gA_k -= Z_C wt (2 alpha SUM E^{a+1_k,b} R
                                     - a_k SUM E^{a-1_k,b} R),  gB likewise
                     f_k  += wt SUM E^{ab}_t R[t + 1_k]
                 G[C, k] += s Z_C f_k

each SUM a rolled loop over the tables (`e_fragment.rolled_e_sum`, its
`up=k` the Hermite shift), so the frame is the tables and `R`: at (ff)
7,712 B of `sm_86` stack (2.4 s) and 6,448 B of `gfx1200` scratch. The
gate and every number are measured rows (data/measured/g1e.json,
`devtools/measure.sh g1e-compile`, its rule declared before any row);
`gs` and `gt` hold no density and fit unrolled at every class through
(ff). Below the gate the text is unchanged byte for byte.

ROUND-OFF, NOT BITWISE, against the host. The host sums each record's
gradient row by row into `gA` from `0.0`; the kernel forms the same sums
as one expression per component, and `V` regroups by Hermite index
(`H0`, `HA`, `HB`) before the nucleus sum and runs the plan's MILD
recurrence. The interpreted test states its bound (test/test_kernel_g1e.py).

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.ir import (
    Assign,
    AtomicAdd,
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
    fundamental_prelude,
    node_name,
    recurrence_body,
    store_targets,
)
from felizjueves.one_electron.kernel_v import v_plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import OPERATOR_OF, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "REC",
    "ZEROED_G",
    "GRADIENT_1E",
    "LOWSTACK_MIN_LSUM",
    "Locals",
    "axis_reader",
    "build_g1e_kernel",
    "derivative_reader",
    "g1e_lowstack",
    "g1e_name",
    "g1e_spec",
    "hermite_densities",
    "launch_shape",
    "record_head",
    "scatter_gradient",
    "weights",
]

REC = "rec"  # the record index: one thread per primitive-pair record
NUC = "c"  # the nucleus index, serial inside the record (V)

GRADIENT_1E = (Target.S, Target.T, Target.V)
_PREFIX = {Target.S: "gs", Target.T: "gt", Target.V: "gv"}

ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "row of G")

ZEROED_G = (
    "PRECONDITION: G is zeroed by the caller before the first gradient "
    "launch of a build; this kernel only adds")

_WEIGHTS = (
    "w_r = M[a_i, b_j] + M[b_j, a_i] off the diagonal, M[a_i, a_j] on a "
    "diagonal shell pair: SUM_mn M_mn X_mn exactly, for any M")

Axis = Callable[[int, int, int, int], Expr]  # (d, i, j, t) -> E^{ij}_t


def g1e_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `target` at derivative order 1."""
    return Spec(OPERATOR_OF[target], 1, (la, lb), target)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else g1e_spec(*cls)
    if spec.target not in GRADIENT_1E or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 S, T or V key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    return spec.target, la, lb


def g1e_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_g1e_kernel(cls)` gives its kernel, unbuilt."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


def launch_shape(n_rec: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per record, `ceil(n_rec/threads)`
    blocks. No `WarpReduce`, so a partial last block is legal."""
    if type(n_rec) is not int or n_rec < 1:
        raise ValueError(f"n_rec = {n_rec!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_rec // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _fold(terms: list[Expr]) -> Expr:
    out = terms[0]
    for t in terms[1:]:
        out = _c(Fn.ADD, out, t)
    return out


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


class Locals:
    """Straight-line F64 locals declared on first request, in request
    order, so nothing is declared that nothing reads (#92's ratchet)."""

    def __init__(self) -> None:
        self.decls: list[Stmt] = []
        self._seen: set[str] = set()

    def get(self, name: str, make: Callable[[], Expr]) -> Var:
        if name not in self._seen:
            self._seen.add(name)
            value = make()  # may itself declare (its dependencies first)
            self.decls.append(Decl(name, Scalar.F64, init=value))
        return Var(name)


def axis_reader(la: int, lb: int) -> Axis:
    """`E^{ij}_t` on axis `d` read from `axis_tables_fragment(la, lb)`'s
    local tables, at a constant index; `t <= i + j` is the caller's."""
    L = la + lb

    def read(d: int, i: int, j: int, t: int) -> Expr:
        if not (0 <= i <= la and 0 <= j <= lb and 0 <= t <= i + j):
            raise ValueError(f"E^{{{i}{j}}}_{t} is not in the ({la}, {lb}) "
                             "tables")
        return Index(ef.AXIS_ARRAYS[d], (ef.axis_table_index(i, j, t, lb, L),))
    return read


def derivative_reader(E: Axis, pool: Locals, side: str) -> Axis:
    """`dA E^{ij}_t` (`side` "A") or `dB E^{ij}_t` ("B") on axis `d`, for
    `t <= i + j + 1`, as pooled locals `dA{x}_{i}_{j}_{t}`; the `- n E`
    term is ABSENT where `n = 0` or its source has no entry `t`."""
    two = Var("ea2" if side == "A" else "eb2")

    def read(d: int, i: int, j: int, t: int) -> Expr:
        def make() -> Expr:
            up = (i + 1, j) if side == "A" else (i, j + 1)
            out: Expr = _mul(two, E(d, *up, t))
            n = i if side == "A" else j
            lo = (i - 1, j) if side == "A" else (i, j - 1)
            if n and t <= lo[0] + lo[1]:
                out = _c(Fn.SUB, out, _mul(Lit(float(n)), E(d, *lo, t)))
            return out
        return pool.get(f"d{side}{'xyz'[d]}_{i}_{j}_{t}", make)
    return read


def record_head(la_t: int, lb_t: int, *, interleaved: bool = False
                ) -> list[Stmt]:
    """Record `rec`'s scalars (`zeta`, `half_inv_zeta`, `sk`, `ea2`,
    `eb2`, `oa`, `ob`, `aa`, `ab`) and the axis tables at `(la_t, lb_t)`,
    inlined. `interleaved` reads `XPA[rec, d]` / `XPB[rec, d]` and `s`
    (the J chain's spelling) instead of `pa_x ...` and `s_ab`."""
    rec = Var(REC)
    frag = ef.axis_tables_fragment(la_t, lb_t)
    n = ef.axis_table_extent(la_t, lb_t)
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              ef.HALF_INV_ZETA: Var(ef.HALF_INV_ZETA)}
    for d, a in enumerate("xyz"):
        if interleaved:
            binds[Factor.pa(d).name_in_kernel] = Index("XPA", (rec, Lit(d)))
            binds[Factor.pb(d).name_in_kernel] = Index("XPB", (rec, Lit(d)))
        else:
            binds[Factor.pa(d).name_in_kernel] = Index(f"pa_{a}", (rec,))
            binds[Factor.pb(d).name_in_kernel] = Index(f"pb_{a}", (rec,))
    for arr in ef.AXIS_ARRAYS:
        binds[arr] = Var(arr)
    return [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
        Decl(ef.HALF_INV_ZETA, Scalar.F64,
             init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))),
        Decl("sk", Scalar.F64,
             init=Index("s" if interleaved else "s_ab", (rec,))),
        Decl("ea2", Scalar.F64,
             init=_mul(Lit(2.0), Index("ea_bra", (rec,)))),
        Decl("eb2", Scalar.F64,
             init=_mul(Lit(2.0), Index("eb_ket", (rec,)))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Decl("aa", Scalar.I32, init=Index("atom_a", (rec,))),
        Decl("ab", Scalar.I32, init=Index("atom_b", (rec,))),
        Note(f"the undressed per-axis E^{{ij}}_t, i <= {la_t}, j <= {lb_t}: "
             "one index above the class on each side, for the derivative "
             "rule"),
        *(Decl(arr, Scalar.F64, (Lit(n),)) for arr in ef.AXIS_ARRAYS),
        Inline(frag, binds),
    ]


def weights(la: int, lb: int) -> list[Stmt]:
    """`w{r}` per Cartesian pair `r` of `block_rows(la, lb)`, from `M`
    (`_WEIGHTS`)."""
    nb = len(ef.cartesians(lb))

    def m(row: Expr, col: Expr) -> Index:
        return Index("M", (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    rows = range(len(ef.block_rows(la, lb)))

    def off(r: int) -> Expr:
        i, j = divmod(r, nb)
        return _c(Fn.ADD, m(plus("oa", i), plus("ob", j)),
                  m(plus("ob", j), plus("oa", i)))

    if la != lb:  # a diagonal shell pair has la == lb
        return [Note(_WEIGHTS),
                *(Decl(f"w{r}", Scalar.F64, init=off(r)) for r in rows)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    return [
        Note(_WEIGHTS),
        *(Decl(f"w{r}", Scalar.F64, init=Lit(0.0)) for r in rows),
        If(differ,
           tuple(Assign(Var(f"w{r}"), off(r)) for r in rows),
           tuple(Assign(Var(f"w{r}"), m(plus("oa", r // nb),
                                        plus("oa", r % nb))) for r in rows)),
    ]


def scatter_gradient() -> list[Stmt]:
    """`G[atom_a, k] += sk gA_k`, `G[atom_b, k] += sk gB_k`."""
    out: list[Stmt] = [Note(ATOMIC_REASON)]
    for side, atom in (("A", "aa"), ("B", "ab")):
        for k in range(3):
            out.append(AtomicAdd(Index("G", (Var(atom), Lit(k))),
                                 _mul(Var("sk"), Var(f"g{side}{k}"))))
    return out


def _reach(a, b) -> tuple[int, int, int]:
    return tuple(a[d] + b[d] for d in range(3))  # type: ignore[return-value]


def hermite_densities(la: int, lb: int, E: Axis, dA: Axis, dB: Axis,
                      *, with_h0: bool) -> tuple[list[Stmt], dict]:
    """The record's Hermite densities as locals: `H0_{n}` over
    `hermite_targets(L)` (when `with_h0`) and `HA{k}_{n}`, `HB{k}_{n}`
    over `hermite_targets(L + 1)`, each `SUM_r w_r (...)` over the rows
    that reach `t`. Returns the statements and `{(kind, n): name}`, where
    kind is "0", "A0".."A2", "B0".."B2"; an entry no row reaches is
    absent."""
    L = la + lb
    rows = ef.block_rows(la, lb)
    out: list[Stmt] = []
    names: dict[tuple[str, int], str] = {}

    def prod(f: list[Expr]) -> Expr:
        return _mul(_mul(f[0], f[1]), f[2])

    if with_h0:
        for n, t in enumerate(hermite_targets(L)):
            terms = [_mul(Var(f"w{r}"), prod([E(d, a[d], b[d], t[d])
                                              for d in range(3)]))
                     for r, (a, b) in enumerate(rows)
                     if all(t[d] <= a[d] + b[d] for d in range(3))]
            if terms:
                names["0", n] = f"H0_{n}"
                out.append(Decl(f"H0_{n}", Scalar.F64, init=_fold(terms)))
    for side, D in (("A", dA), ("B", dB)):
        for k in range(3):
            for n, t in enumerate(hermite_targets(L + 1)):
                terms = []
                for r, (a, b) in enumerate(rows):
                    top = list(_reach(a, b))
                    top[k] += 1
                    if any(t[d] > top[d] for d in range(3)):
                        continue
                    f = [D(d, a[d], b[d], t[d]) if d == k
                         else E(d, a[d], b[d], t[d]) for d in range(3)]
                    terms.append(_mul(Var(f"w{r}"), prod(f)))
                if terms:
                    nm = f"H{side}{k}_{n}"
                    names[f"{side}{k}", n] = nm
                    out.append(Decl(nm, Scalar.F64, init=_fold(terms)))
    return out, names


def _s_values(la: int, lb: int, E: Axis, dA: Axis, dB: Axis) -> list[Stmt]:
    out: list[Stmt] = []
    rows = ef.block_rows(la, lb)
    for side, D in (("A", dA), ("B", dB)):
        for k in range(3):
            terms = []
            for r, (a, b) in enumerate(rows):
                f = [D(d, a[d], b[d], 0) if d == k else E(d, a[d], b[d], 0)
                     for d in range(3)]
                terms.append(_mul(Var(f"w{r}"), _mul(_mul(f[0], f[1]), f[2])))
            out.append(Decl(f"g{side}{k}", Scalar.F64, init=_fold(terms)))
    return out


def _t_values(la: int, lb: int, E: Axis, pool: Locals) -> list[Stmt]:
    """`gt`'s six components (module docstring)."""
    eb = Var("ebk")

    def S(d: int, i: int, j: int) -> Expr:
        return E(d, i, j, 0)

    def K(d: int, i: int, j: int) -> Var:
        def make() -> Expr:  # e_fragment._shift's expression, restated
            k = _c(Fn.ADD, _mul(_mul(_mul(Lit(-2.0), eb), eb), S(d, i, j + 2)),
                   _mul(_mul(eb, Lit(float(2 * j + 1))), S(d, i, j)))
            if j >= 2:
                k = _c(Fn.SUB, k, _mul(Lit(0.5 * j * (j - 1)), S(d, i, j - 2)))
            return k
        return pool.get(f"K{'xyz'[d]}_{i}_{j}", make)

    def deriv(f: Callable[[int, int, int], Expr], side: str, d: int, i: int,
              j: int, tag: str) -> Var:
        def make() -> Expr:
            if side == "A":
                out: Expr = _mul(Var("ea2"), f(d, i + 1, j))
                if i:
                    out = _c(Fn.SUB, out, _mul(Lit(float(i)), f(d, i - 1, j)))
            else:
                out = _mul(Var("eb2"), f(d, i, j + 1))
                if j:
                    out = _c(Fn.SUB, out, _mul(Lit(float(j)), f(d, i, j - 1)))
            return out
        return pool.get(f"d{side}{tag}{'xyz'[d]}_{i}_{j}", make)

    out: list[Stmt] = []
    rows = ef.block_rows(la, lb)
    for side in "AB":
        for k in range(3):
            terms = []
            for r, (a, b) in enumerate(rows):
                parts = []
                for m in range(3):  # the bracket's term with K on axis m
                    f: list[Expr] = []
                    for d in range(3):
                        if d == k:
                            f.append(deriv(K if d == m else S, side, d, a[d],
                                           b[d], "K" if d == m else "S"))
                        else:
                            f.append(K(d, a[d], b[d]) if d == m
                                     else S(d, a[d], b[d]))
                    parts.append(_mul(_mul(f[0], f[1]), f[2]))
                bracket = _c(Fn.ADD, _c(Fn.ADD, parts[0], parts[1]), parts[2])
                terms.append(_mul(Var(f"w{r}"), bracket))
            out.append(Decl(f"g{side}{k}", Scalar.F64, init=_fold(terms)))
    return out


def _v_values(la: int, lb: int, E: Axis, dA: Axis, dB: Axis
              ) -> list[Stmt]:
    """`gv`'s Hermite densities, nucleus loop and Hellmann-Feynman atomics
    (module docstring)."""
    L = la + lb
    dens, names = hermite_densities(la, lb, E, dA, dB, with_h0=True)
    dag, order = v_plan((la, lb + 1))
    t1 = hermite_targets(L + 1)
    index = {t: n for n, t in enumerate(t1)}

    def node(t) -> Var:
        return Var(node_name((t, 0)))

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    nuc = Var(NUC)
    per_nuc: list[Stmt] = [
        Decl("Zc", Scalar.F64, init=Index("Z", (nuc,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("Cnuc", (nuc, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        *prelude,
        *recurrence_body(dag, order, fundamental),
        Note("gA_k -= Z_C SUM_t HA_k,t [t]^(0) (and B): the centres; "
             "G[C, k] += s Z_C SUM_t H0_t [t + 1_k]^(0): the nucleus "
             "(Hellmann-Feynman)"),
        Decl("skz", Scalar.F64, init=_mul(Var("sk"), Var("Zc"))),
    ]
    for side in "AB":
        for k in range(3):
            terms = [_mul(Var(names[f"{side}{k}", n]), node(t))
                     for n, t in enumerate(t1) if (f"{side}{k}", n) in names]
            g = f"g{side}{k}"
            per_nuc.append(Assign(Var(g), _c(Fn.SUB, Var(g), _mul(
                Var("Zc"), _fold(terms)))))
    for k in range(3):
        terms = []
        for n, t in enumerate(hermite_targets(L)):
            if ("0", n) in names:
                up = tuple(t[d] + (d == k) for d in range(3))
                terms.append(_mul(Var(names["0", n]), node(up)))
                assert up in index
        per_nuc.append(AtomicAdd(Index("G", (nuc, Lit(k))),
                                 _mul(Var("skz"), _fold(terms))))
    return [
        *dens,
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        Note("theta = sqrt(zeta): a unit point-charge ket (kernel_v)"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var(ef.ZETA))),
        *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
          for s in "AB" for k in range(3)),
        Note("the nuclei, serially inside the record (kernel_v's mapping)"),
        For(NUC, Lit(0), Var("n_atom"), Lit(1), tuple(per_nuc)),
    ]


# ---------------------------------------------------------------------------
# The low-stack gv (#614)
# ---------------------------------------------------------------------------

# gv from la + lb = 5, (df), (fd) and (ff) through f (#614): the smallest
# la + lb with an inadmissible unrolled gv class (data/measured/g1e.json)
LOWSTACK_MIN_LSUM = 5
R_ARRAY = "R"  # [t]^(0) of one nucleus, hermite_targets(L + 1) order

LOWSTACK_NOTE = (
    "LOW-STACK (#614): no Hermite density is formed; per nucleus [t]^(0) "
    "into a local R as it forms, and per block row, its weight read from M, "
    "the nine derivative terms as rolled sums over the per-axis tables one "
    "index up (2 alpha SUM E^{a+1_k,b} R - a_k SUM E^{a-1_k,b} R, and "
    "SUM E^{ab}_t R[t + 1_k] for the nucleus)")


def g1e_lowstack(cls: Spec | tuple[Target, int, int]) -> bool:
    """THE SELECTION RULE (#614): True for a `gv` class with `la + lb >=
    LOWSTACK_MIN_LSUM`; there `build_g1e_kernel` emits the low-stack form
    by default. `gs` and `gt` have no other form (every unrolled class
    through (ff) is admissible, data/measured/g1e.json)."""
    target, la, lb = _class(cls)
    return target is Target.V and la + lb >= LOWSTACK_MIN_LSUM


def _v_lowstack_values(la: int, lb: int) -> list[Stmt]:
    """`gv`'s per-record body after the head, LOW-STACK (module docstring,
    THE LOW-STACK gv): the same sums as `_v_values`, regrouped per row."""
    L = la + lb
    dag, order = v_plan((la, lb + 1))
    t1 = hermite_targets(L + 1)
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tw = store_targets(recurrence_body(dag, order, fundamental),
                       frozenset(t1), L + 1, R_ARRAY)
    nuc = Var(NUC)

    def r_read(h: Expr) -> Expr:
        return Index(R_ARRAY, (h,))

    def m_at(i: Expr, j: Expr) -> Index:
        return Index("M", (i, j))

    def row(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def elem(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            m = _c(Fn.ADD, Var("oa"), Var("ra"))
            n = _c(Fn.ADD, Var("ob"), Var("rb"))
            off = _c(Fn.ADD, m_at(m, n), m_at(n, m))
            if la != lb:
                weight: tuple[Stmt, ...] = (Decl("wt", Scalar.F64, init=off),)
            else:
                weight = (
                    Decl("wt", Scalar.F64, init=Lit(0.0)),
                    If(Var("differ"), (Assign(Var("wt"), off),),
                       (Assign(Var("wt"), m_at(m, _c(
                           Fn.ADD, Var("oa"), Var("rb")))),)),
                )
            out: list[Stmt] = [
                Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
                *weight,
                Decl("wz", Scalar.F64, init=_mul(Var("Zc"), Var("wt"))),
            ]
            for side, ab, two in (("A", a, "ea2"), ("B", b, "eb2")):
                for k in range(3):
                    def moved(by: int, ab=ab, k=k) -> tuple[Expr, ...]:
                        step = (_c(Fn.ADD, ab[k], Lit(1)) if by > 0
                                else _c(Fn.SUB, ab[k], Lit(1)))
                        return tuple(step if d == k else ab[d]
                                     for d in range(3))
                    up, lo = moved(1), moved(-1)
                    ra_, rb_ = (up, b) if side == "A" else (a, up)
                    la_, lb_ = (lo, b) if side == "A" else (a, lo)
                    hi, dn = f"h{side}{k}", f"l{side}{k}"
                    out += [
                        *ef.rolled_e_sum(la + 1, lb + 1, ra_, rb_, r_read,
                                         hi, tag=f"p{side}{k}"),
                        Decl(dn, Scalar.F64, init=Lit(0.0)),
                        If(_c(Fn.LT, Lit(0), ab[k]), (
                            *ef.rolled_e_sum(la + 1, lb + 1, la_, lb_,
                                             r_read, f"m{side}{k}",
                                             tag=f"q{side}{k}"),
                            Assign(Var(dn), _mul(ab[k], Var(f"m{side}{k}"))),
                        )),
                        Assign(Var(f"g{side}{k}"), _c(
                            Fn.SUB, Var(f"g{side}{k}"), _mul(Var("wz"), _c(
                                Fn.SUB, _mul(Var(two), Var(hi)),
                                Var(dn))))),
                    ]
            for k in range(3):
                out += [
                    *ef.rolled_e_sum(la + 1, lb + 1, a, b, r_read,
                                     f"hC{k}", tag=f"pC{k}", up=k),
                    Assign(Var(f"f{k}"), _c(Fn.ADD, Var(f"f{k}"),
                                            _mul(Var("wt"), Var(f"hC{k}")))),
                ]
            return tuple(out)
        return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                ef.cartesian_loops(lb, ("bx", "by", "bz"), elem, tag="b"))

    per_nuc: list[Stmt] = [
        Decl("Zc", Scalar.F64, init=Index("Z", (nuc,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("Cnuc", (nuc, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        *prelude,
        Note(f"[t]^(0) for every |t| <= L + 1 = {L + 1}, each stored into "
             "R as it forms (recurrence.store_targets)"),
        *tw,
        *(Decl(f"f{k}", Scalar.F64, init=Lit(0.0)) for k in range(3)),
        Note("per block row (a, b), weight wt from M: gA_k -= Z_C wt SUM_t "
             "(dA_k E^{ab})_t R[t] (the Cartesian rule, 2 alpha "
             "E^{a+1_k,b} - a_k E^{a-1_k,b}), gB likewise, and f_k += wt "
             "SUM_t E^{ab}_t R[t + 1_k] (Hellmann-Feynman), each rolled "
             "over the per-axis tables (e_fragment.rolled_e_sum)"),
        ef.cartesian_loops(la, ("ax", "ay", "az"), row, tag="a"),
        Decl("skz", Scalar.F64, init=_mul(Var("sk"), Var("Zc"))),
        *(AtomicAdd(Index("G", (nuc, Lit(k))), _mul(Var("skz"),
                                                    Var(f"f{k}")))
          for k in range(3)),
    ]
    differ: list[Stmt] = [Note(_WEIGHTS)]
    if la == lb:
        differ.append(Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))))
    return [
        Note(LOWSTACK_NOTE),
        *differ,
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        Note("theta = sqrt(zeta): a unit point-charge ket (kernel_v)"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var(ef.ZETA))),
        *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
          for s in "AB" for k in range(3)),
        Decl(R_ARRAY, Scalar.F64, (Lit(len(t1)),)),
        Note("the nuclei, serially inside the record (kernel_v's mapping)"),
        For(NUC, Lit(0), Var("n_atom"), Lit(1), tuple(per_nuc)),
    ]


def build_g1e_kernel(cls: Spec | tuple[Target, int, int], *,
                     name: str | None = None,
                     lowstack: bool | None = None) -> Kernel:
    """The gradient kernel of one one-electron pair class at d1.

    `cls` is a d1 `Target.S` / `T` / `V` `Spec` or `(target, la, lb)`.
    See the module docstring. `lowstack` picks `gv`'s form: `None` (the
    default) is `g1e_lowstack(cls)` (THE LOW-STACK gv), `True` / `False`
    force either at any `gv` class; `gs` / `gt` refuse `True`. Both forms
    have the same name, parameters, launch and `G` contract.
    """
    target, la, lb = _class(cls)
    if lowstack is None:
        lowstack = g1e_lowstack(cls)
    if lowstack and target is not Target.V:
        raise ValueError(f"{g1e_name(cls)}: only gv has a low-stack form")
    lb_t = lb + 3 if target is Target.T else lb + 1
    head = record_head(la + 1, lb_t)
    E = axis_reader(la + 1, lb_t)
    pool = Locals()
    if target is Target.T:
        head.append(Decl("ebk", Scalar.F64, init=Index("eb_ket", (Var(REC),))))
    dA = derivative_reader(E, pool, "A")
    dB = derivative_reader(E, pool, "B")
    if target is Target.S:
        values = _s_values(la, lb, E, dA, dB)
    elif target is Target.T:
        values = _t_values(la, lb, E, pool)
    elif lowstack:
        values = _v_lowstack_values(la, lb)
    else:
        values = _v_values(la, lb, E, dA, dB)
    what = {Target.S: "overlap", Target.T: "kinetic energy",
            Target.V: "nuclear attraction (the Hellmann-Feynman term on "
                      "every nucleus included)"}[target]
    if lowstack:  # no pooled factor and no w{r}: M is read per row
        per_rec = (*head, *values, *scatter_gradient())
    else:
        per_rec = (*head, *weights(la, lb),
                   Note("the derivative factors, each declared where first "
                        "read"),
                   *pool.decls, *values, *scatter_gradient())
    body: tuple[Stmt, ...] = (
        Note(f"gradient of the {what}, class (la, lb) = ({la}, {lb}): "
             "G[A, k] += SUM_mn M_mn dX_mn/dR_{A,k} (felizjueves.gradient)"),
        Note(ZEROED_G),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), per_rec),
    )
    nr, nao, nat = Var("n_rec"), Var("n_ao"), Var("n_atom")
    params = [
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        *(_f64(f"pa_{a}", (nr,)) for a in "xyz"),
        *(_f64(f"pb_{a}", (nr,)) for a in "xyz"),
        *((_f64(f"p_{a}", (nr,)) for a in "xyz") if target is Target.V
          else ()),
        _f64("s_ab", (nr,)),
        _f64("ea_bra", (nr,)),
        _f64("eb_ket", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        _i32("atom_a", (nr,)),
        _i32("atom_b", (nr,)),
        _f64("M", (nao, nao)),
    ]
    if target is Target.V:
        params += [_f64("Z", (nat,)), _f64("Cnuc", (nat, Lit(3)))]
    params.append(Param("G", Scalar.F64, array=True, intent=Intent.OUT,
                        shape=(nat, Lit(3))))
    return Kernel(name or g1e_name((target, la, lb)), tuple(params), body)
