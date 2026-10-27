"""The ket pre-contraction kernel, `Target.DQ`: `D_q` and `Dmax` (#153).

    build_dq_kernel((lc, ld) | spec) -> Kernel     one per ket pair class
    build_dq_kernel(cls, density="ao") -> Kernel    the same, reading D_mu_nu
    build_dq_kernel(cls, lowstack=None) -> Kernel   the form: see LOW-STACK
    dq_lowstack((lc, ld) | spec)     -> bool       the form's selection rule
    dq_name((lc, ld) | spec)         -> str        its name, unbuilt
    launch_shape(n_rec, lanes)       -> (grid, (lanes, 1))

Step 2 of the J chain (docs/implementation/j_engine.md, "The J chain as
device code", "Step 2: the pre-contraction kernel"), built as IR. One kernel
per ket PAIR class `(lc, ld)`, keyed `Spec(COULOMB, 0, (lc, ld), DQ)`. With
`n_q = |hermite_targets(lc + ld)|` and `n_cd = ncart(lc) ncart(ld)`:

    params  n_rec n_pair (I32)
            eta[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s[n_rec]  w[n_rec]
            pair[n_rec] (I32)                    the record's row of Dcd
            Dcd[n_pair, n_cd]                    D_cd per ket shell pair
            D[n_rec, n_q]  (OUT)   Dmax[n_rec]  (OUT)

`n_pair` is the one parameter the specification's block does not spell: it
is the extent `Dcd[n_pair, n_cd]` is declared with, and an extent may only
use scalar `Param`s, so it is a parameter.

Per record `k`, with `E` the UNDRESSED block of record `k`
(`e_fragment.block_fragment(lc, ld)`, inlined):

    acc_q   = SUM_{r in row order} Dcd[pair[k], r] * (s[k] * E[r, q])
    D[k, q] = w[k] * acc_q
    Dmax[k] = max_q |D[k, q]|

THE OPERATION ORDER IS THE HOST'S (`eri.j_coulomb.j.dq_from_host`): `r` outer, `q`
inner, each term `x * (s * e)` added to an accumulator that starts at `+0.0`,
then `w * acc`. So interpreted, the kernel is `==` to the host, bit for bit.
Two departures change no bit, and are why:

- a term whose `E` entry is structurally zero (`t_d > a_d + b_d` on some
  axis) is ABSENT -- the fragment never writes it, and a read of it would be
  a read before a write. The host adds `x * (s * 0.0) = +-0.0` there, and
  adding `+-0.0` to an accumulator that started at `+0.0` is the identity
  (round-to-nearest never yields `-0.0` from a sum that is not `-0 + -0`);
- the host skips `D_cd = 0` terms, and the kernel does not (no data-dependent
  branch): the same `+-0.0` argument.

The factor `s * w` is NOT hoisted out of the sum: that is admitted by the
specification at round-off, but it would cost the bitwise equality.

THE WEIGHT `w` IS A HOST-SUPPLIED SCALAR PER RECORD -- 2.0 for an
off-diagonal ket shell pair, 1.0 for a diagonal one -- and the kernel never
decides it. It is folded into `D_q` and so into `Dmax`. A weight of 1 off
the diagonal is the order-unity error this design is most likely to ship.

THE DRESSING `s = c_c c_d (pi/eta)^{3/2}` is host-supplied too, from
`ao.pairs.dress` through `ao.density.epair`; no kernel forms a contraction
coefficient or `(pi/eta)^{3/2}`, so the inlined `E` stays the undressed `E`
its pointwise Hermite identity checks.

`Dmax` is over the weighted, dressed `D_q`: a reduction over the kernel's own
output, never the AO density's largest element. `|x|` and `max` are spelled
with `LT` and `SUB` (`0.0 - x` is `-x` exactly), because `Fn` has neither and
the closed vocabulary is not widened for them.

THE LAYOUT. Rows of `Dcd` are `e_fragment.block_rows(lc, ld)` (the
Cartesian pairs, `c` outer -- `ao.shell.cartesians`' order, which a test pins
equal) and columns of `D` are `hermite_targets(lc + ld)` (`mmd.hermite`'s
`hermite_order`, pinned equal): the same two helpers the fragment's `E`
block is indexed by, so the row of `Dcd` and the row of `E` are one index.

THE AO DENSITY (`density="ao"`, opt-in). The default reads the host's
gather `Dcd[pair[k], r]`; this variant reads the same values where they
already are, in the symmetric AO density, `Dao[off_c[k] + i, off_d[k] + j]`
for row `r = i * ncart(ld) + j` -- `kernel_jab`'s addressing of `J`, with
the record's own shell offsets. `pair`, `Dcd` and `n_pair` give way to
`off_c`, `off_d`, `Dao[n_ao, n_ao]` and `n_ao` (`AO_PARAMS`; `Dao`, because
`D` is the output). Each term is the same product in the same order, so the
two variants write the same bits (test_kernel_dq.py). What it buys a
consumer whose density lives on the device: no gather kernel and no `Dcd`
buffer, and a shell pair stored swapped (`c` the shell of momentum `lc`,
whichever index it has) reads its own orientation of the symmetric `D` with
no transposed copy. `w` is still the host's. Its name is `dq_{lc}_{ld}_ao`
because its parameters are not the default's. Nothing about either variant
is measured: the AO read strides `n_ao` per row of the block where `Dcd` is
contiguous.

THE LANES. One thread per record, `k = THREAD_X + BLOCK_X * BLOCK_DIM_X`,
guarded by `k < n_rec`; `launch_shape` is that mapping. It is not part of the
numerical contract (the specification says so): no thread reads another's
record, there is no reduction across threads and no atomic.

The output is in RECORD order, not the sorted order the `[p|q]` kernel
requires: the sort on `Iq * Dmax` is the host's, after `Dmax` is read back
(j_engine.md, "The round-trip").

THE LOW-STACK FORM (#586, stage 2), the default where it is needed: `l_1 +
l_2 >= LOWSTACK_MIN_PAIR_L = 6` -- (ff), and past f (gd), (gf), ... The
unrolled kernel above holds the whole `E` block, `E[100][84]` per thread at
(ff), 67 KB: on sm_86 a 72 KB stack frame (`dq_33`), which CUDA reserves for
every resident thread (~7 GB of a 10 GB RTX 3080). `build_dq_kernel(cls)`
therefore emits, whenever `dq_lowstack(cls)`, a kernel with the SAME name,
parameters (either density), launch and output contract and no `E` block:

    tables = e_fragment.axis_tables_fragment(lc, ld)  Ex Ey Ez, local,
                                      3 x (lc+1)(ld+1)(lc+ld+1) doubles
    acc[n_q] = 0                      a local, hermite_targets order
    for each row r = (c, d)           e_fragment.cartesian_loops, as loops,
                                      c outer: block_rows order
        x = Dcd[pair[k], r]           (or Dao[oc + i, od + j])
        for t in the reachable box    t_d <= c_d + d_d, three loops
            e = (Ex[c_x d_x t_x] Ey[c_y d_y t_y]) Ez[c_z d_z t_z]
            acc[hermite_index(t)] += x * (s[k] * e)
    D[k, q] = w[k] * acc[q];  Dmax    one loop over q

THE SAME BITS. `e` is `block_fragment`'s entry, `(E_x E_y) E_z` from the
same recurrence (`_axes_body` unchanged), and each `acc[q]` takes the same
terms in the same order (`r` outer; the order over `t` inside a row does not
touch any one accumulator twice), so interpreted the low-stack form is `==`
to the unrolled one and to the host, as above. `e_fragment.rolled_e_sum` is
NOT used: its factored product `E_x (E_y (E_z X))` is round-off, and this
kernel's contract with the host is bitwise. Below L = 6 -- through d, and
the mixed f classes (sf), (pf), (df) -- the default is the unrolled kernel,
its text unchanged byte for byte (those units already fit: 0 to 936 B on
sm_86, no spill, and the low-stack form is slower there, up to 3.6x for JAB
at (pf)); `lowstack=True` / `False` forces either form at any class. At (ff)
the frame is the tables and `acc`, 420 doubles.

A builder: vendor-neutral IR, no `reference/`, no interpreter, no AO layer.
"""

from __future__ import annotations

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
from felizjueves.mmd.e_fragment import (
    E_ARRAY,
    HALF_INV_ZETA,
    ZETA,
    block_fragment,
    block_rows,
    block_shape,
)
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "AO_PARAMS",
    "DENSITIES",
    "DENSITY_AO",
    "LOWSTACK_MIN_PAIR_L",
    "LOWSTACK_NOTE",
    "PARAMS",
    "WEIGHT_IS_THE_HOSTS",
    "build_dq_kernel",
    "dq_lowstack",
    "dq_name",
    "dq_spec",
    "dq_shape",
    "launch_shape",
]

# The parameter block, in order: the specification's, plus `n_pair`.
PARAMS = ("n_rec", "n_pair", "eta", "XPA", "XPB", "s", "w", "pair", "Dcd",
          "D", "Dmax")

# density="ao": the AO density itself, read through each record's offsets.
AO_PARAMS = ("n_rec", "n_ao", "eta", "XPA", "XPB", "s", "w", "off_c", "off_d",
             "Dao", "D", "Dmax")

# Where the kernel reads D_cd from: "pairs", the host's gathered block per
# ket shell pair (`Dcd[pair[k], r]`, the default and the specification's), or
# "ao", the symmetric AO density (`Dao[off_c[k] + i, off_d[k] + j]`).
DENSITIES = ("pairs", "ao")

DENSITY_AO = (
    "density='ao': D_cd is read from the AO density itself, "
    "Dao[off_c[k] + i, off_d[k] + j] for the record's row r = (i, j), the "
    "values the host gathers into Dcd[pair[k], r] (ao.j._kets), so the sum "
    "is the same product in the same order. PRECONDITION: Dao is symmetric "
    "and off_c / off_d are the record's own shells' first AOs, c the shell "
    "of momentum lc -- a pair stored swapped reads its own orientation"
)

REC = "k"  # the record index: one thread per record

WEIGHT_IS_THE_HOSTS = (
    "PRECONDITION: w[k] is 2.0 for an off-diagonal ket shell pair (c != d) "
    "and 1.0 for a diagonal one -- decided by the host, folded into D_q and "
    "so into Dmax; this kernel never decides it. Only unique ket shell "
    "pairs c >= d are records; D is symmetric (checked by the host, never "
    "symmetrised here); eta > 0 (j_engine.md, 'Step 2')"
)


def dq_spec(lc: int, ld: int) -> Spec:
    """The ket pair class `(lc, ld)` as a generation key."""
    return Spec(Operator.COULOMB, 0, (lc, ld), Target.DQ)


def _spec_of(cls: tuple[int, int] | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 2:
        spec = dq_spec(*cls)
    else:
        raise TypeError(f"build_dq_kernel takes (lc, ld) or a Spec, not {cls!r}")
    if spec.target is not Target.DQ:
        raise ValueError(f"{spec.key()} is not a Target.DQ class")
    if spec.levers:
        raise ValueError(f"{spec.key()}: no lever is defined for Target.DQ")
    return spec


def dq_shape(lc: int, ld: int) -> tuple[int, int]:
    """`(n_cd, n_q)`: a row of `Dcd` and a row of `D`, by the fragment's
    own layout helpers."""
    n_cd, n_q = block_shape(lc, ld)
    assert n_q == len(hermite_targets(lc + ld))
    return n_cd, n_q


def launch_shape(n_rec: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """`(grid, (lanes, 1))`, `grid = ceil(n_rec / lanes)`: one thread per
    record. The mapping is not part of the numerical contract."""
    if type(n_rec) is not int or n_rec < 1:
        raise ValueError(f"n_rec = {n_rec!r} is not a positive int")
    if type(lanes) is not int or lanes < 1:
        raise ValueError(f"lanes = {lanes!r} is not a positive int")
    return -(-n_rec // lanes), (lanes, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def dq_name(cls: tuple[int, int] | Spec, *, density: str = "pairs") -> str:
    """The name `build_dq_kernel(cls, density=)` gives its kernel, without
    building it (#419): `dq_{lc}_{ld}`, and `dq_{lc}_{ld}_ao` for the
    AO-density variant, whose parameters differ."""
    lc, ld = _spec_of(cls).am
    return f"dq_{lc}_{ld}" + ("_ao" if _density(density) == "ao" else "")


def _density(density: object) -> str:
    if density not in DENSITIES:
        raise ValueError(f"density is one of {DENSITIES}, not {density!r}")
    return density  # type: ignore[return-value]


LOWSTACK_MIN_PAIR_L = 6  # the low-stack form at l1 + l2 >= 6 (#586), = kernel_jab's


def dq_lowstack(cls: tuple[int, int] | Spec) -> bool:
    """THE SELECTION RULE (#586): True when the class's pair angular
    momentum `l_1 + l_2 >= LOWSTACK_MIN_PAIR_L` -- (ff), and past f (gd),
    (gf), ... -- where the unrolled form's per-thread `E` block,
    `n_rows x n_t` doubles, outgrows the stack: on sm_86 the unrolled
    (df) frame (L = 5, 60 x 56) is 728 B for DQ and 936 B for JAB, no
    spill, and (ff)'s (L = 6, 100 x 84) is about 72 KB. A class's block
    size is a function of L more than of its split (gd's 90 x 84 sits
    beside (ff), gp's 45 x 56 below (df)), so the cut is on L; there
    `build_dq_kernel` emits the low-stack form by default. Below it the
    unrolled form stays the default and its text is unchanged: it fits,
    and it is faster (#586's timing)."""
    return sum(_spec_of(cls).am) >= LOWSTACK_MIN_PAIR_L


def build_dq_kernel(cls: tuple[int, int] | Spec, *,
                    name: str | None = None,
                    density: str = "pairs",
                    lowstack: bool | None = None) -> Kernel:
    """The pre-contraction kernel of the ket pair class `cls` (module doc).

    `density="ao"` reads `D_cd` from the AO density `Dao[n_ao, n_ao]`
    through per-record shell offsets `off_c`, `off_d` -- `kernel_jab`'s
    addressing of `J` -- in place of the host's gathered `Dcd[pair[k], r]`
    (module docstring, "THE AO DENSITY"). The default is unchanged.

    `lowstack` picks the form: `None` (the default) is `dq_lowstack(cls)`,
    so below L = 6 the unrolled kernel and from L = 6 the low-stack one (module
    docstring, THE LOW-STACK FORM); `True` / `False` force either, at any
    class. Both have the same name, parameters, launch and output."""
    spec = _spec_of(cls)
    ao = _density(density) == "ao"
    if lowstack is None:
        lowstack = dq_lowstack(spec)
    if lowstack:
        return _build_lowstack(spec, ao,
                               name or dq_name(spec, density=density))
    lc, ld = spec.am
    n_cd, n_q = dq_shape(lc, ld)
    n_d = (ld + 1) * (ld + 2) // 2  # ncart(ld): row r = i * n_d + j
    assert n_cd % n_d == 0
    frag = block_fragment(lc, ld)
    rows = block_rows(lc, ld)
    ts = hermite_targets(lc + ld)
    if name is None:
        name = dq_name(spec, density=density)

    k = Var(REC)

    def rec(arr: str, *more: Expr) -> Index:
        return Index(arr, (k, *more))

    bindings: dict[str, Expr] = {ZETA: rec("eta"), E_ARRAY: Var(E_ARRAY)}
    if HALF_INV_ZETA in frag.binds:
        bindings[HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), rec("eta"))
    for d in range(3):
        bindings[Factor.pa(d).name_in_kernel] = rec("XPA", Lit(d))
        bindings[Factor.pb(d).name_in_kernel] = rec("XPB", Lit(d))

    def acc(j: int) -> Var:
        return Var(f"acc{j}")

    if ao:
        read = [Decl("oc", Scalar.I32, init=rec("off_c")),
                Decl("od", Scalar.I32, init=rec("off_d"))]
    else:
        read = [Decl("pk", Scalar.I32, init=rec("pair"))]
    per_rec: list[Stmt] = [
        *read,
        Decl("sk", Scalar.F64, init=rec("s")),
        Decl("wk", Scalar.F64, init=rec("w")),
        Note("the undressed E block of this record: e_fragment.block_fragment, "
             "inlined; the dressing s arrives as a host scalar"),
        Decl(E_ARRAY, Scalar.F64, (Lit(n_cd), Lit(n_q))),
        Inline(frag, bindings),
        *((Note(DENSITY_AO),) if ao else ()),
        Note(("acc_q = SUM_r Dao[oc + i, od + j] * (s[k] * E[r, q])" if ao
              else "acc_q = SUM_r Dcd[pair[k], r] * (s[k] * E[r, q])")
             + ": r outer, q inner, the host's order (ao.j.dq_from_host); an "
             "absent E entry (t_d > a_d + b_d) is a structural zero and is "
             "skipped"),
        *(Decl(acc(j).name, Scalar.F64, init=Lit(0.0)) for j in range(n_q)),
    ]
    for r, (a, b) in enumerate(rows):
        if ao:
            i, j = divmod(r, n_d)
            x = Index("Dao", (_c(Fn.ADD, Var("oc"), Lit(i)),
                              _c(Fn.ADD, Var("od"), Lit(j))))
        else:
            x = Index("Dcd", (Var("pk"), Lit(r)))
        for j, t in enumerate(ts):
            if any(t[d] > a[d] + b[d] for d in range(3)):
                continue
            term = _c(Fn.MUL, x, _c(Fn.MUL, Var("sk"),
                                    Index(E_ARRAY, (Lit(r), Lit(j)))))
            per_rec.append(Assign(acc(j), _c(Fn.ADD, acc(j), term)))
    per_rec += [
        Note("D[k, q] = w[k] * acc_q: the host's weight, applied last"),
    ]
    for j in range(n_q):
        per_rec += [
            Decl(f"dq{j}", Scalar.F64, init=_c(Fn.MUL, Var("wk"), acc(j))),
            Assign(rec("D", Lit(j)), Var(f"dq{j}")),
        ]
    per_rec += [
        Note("Dmax[k] = max_q |D[k, q]|, over the weighted, dressed D_q -- "
             "not the AO density's largest element"),
        Decl("dmax", Scalar.F64, init=Lit(0.0)),
    ]
    for j in range(n_q):
        v = Var(f"dq{j}")
        a = Var(f"adq{j}")
        per_rec += [
            Decl(a.name, Scalar.F64, init=v),
            If(_c(Fn.LT, v, Lit(0.0)), (Assign(a, _c(Fn.SUB, Lit(0.0), v)),)),
            If(_c(Fn.LT, Var("dmax"), a), (Assign(Var("dmax"), a),)),
        ]
    per_rec.append(Assign(rec("Dmax"), Var("dmax")))

    body: tuple[Stmt, ...] = (
        Note(f"D_q = w SUM_cd D_cd s E^cd_q and Dmax = max_q |D_q|, ket pair "
             f"class ({lc},{ld}): j_engine.md 'Step 2: the pre-contraction "
             f"kernel'; [JEngine, v2, eq. 20]"),
        Note(WEIGHT_IS_THE_HOSTS),
        Note("launch: one thread per record, k = threadIdx.x + blockIdx.x * "
             "blockDim.x"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, k, Var("n_rec")), tuple(per_rec)),
    )
    return Kernel(name, _params(ao, n_cd, n_q), body)


def _params(ao: bool, n_cd: int, n_q: int) -> tuple[Param, ...]:
    """The parameter block (module docstring), one for both forms."""
    nr, npair = Var("n_rec"), Var("n_pair")

    def f64(n: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(n, Scalar.F64, array=True, intent=intent, shape=shape)

    per_record = (f64("eta", nr), f64("XPA", nr, Lit(3)),
                  f64("XPB", nr, Lit(3)), f64("s", nr), f64("w", nr))
    out = (f64("D", nr, Lit(n_q), intent=Intent.OUT),
           f64("Dmax", nr, intent=Intent.OUT))
    if ao:
        nao = Var("n_ao")
        params = (
            Param("n_rec", Scalar.I32), Param("n_ao", Scalar.I32),
            *per_record,
            Param("off_c", Scalar.I32, array=True, shape=(nr,)),
            Param("off_d", Scalar.I32, array=True, shape=(nr,)),
            f64("Dao", nao, nao), *out)
        assert tuple(p.name for p in params) == AO_PARAMS
        return params
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_pair", Scalar.I32),
        *per_record,
        Param("pair", Scalar.I32, array=True, shape=(nr,)),
        f64("Dcd", npair, Lit(n_cd)),
        *out,
    )
    assert tuple(p.name for p in params) == PARAMS
    return params


# ---------------------------------------------------------------------------
# The low-stack form (#586, stage 2)
# ---------------------------------------------------------------------------

ACC = "acc"  # the D_q accumulators, hermite_targets order

LOWSTACK_NOTE = (
    "LOW-STACK (#586): per-axis E tables (3 x (la+1)(lb+1)(la+lb+1) "
    "doubles) in place of the E^{ab}_t block; the rows and the Hermite "
    "index as loops, each E entry formed (E_x E_y) E_z where it is used, "
    "the unrolled form's product, so the arithmetic is the same")


def _position(tag: str) -> Expr:
    """The position in `cartesians(l)` of the Cartesian that the
    `e_fragment.cartesian_loops` nest tagged `tag` has open:
    `u (u+1) / 2 + w`, an I32 expression (#586)."""
    u, w = Var(f"{tag}_u"), Var(f"{tag}_w")
    return _c(Fn.ADD, _c(Fn.DIV, _c(Fn.MUL, u, _c(Fn.ADD, u, Lit(1))),
                         Lit(2)), w)


def _build_lowstack(spec: Spec, ao: bool, name: str) -> Kernel:
    """`build_dq_kernel`'s kernel with no `E` block (module docstring, THE
    LOW-STACK FORM): the same parameters, launch and output, the same bits."""
    lc, ld = spec.am
    L = lc + ld
    n_cd, n_q = dq_shape(lc, ld)
    n_d = len(ef.cartesians(ld))
    frag = ef.axis_tables_fragment(lc, ld)
    k = Var(REC)

    def rec(arr: str, *more: Expr) -> Index:
        return Index(arr, (k, *more))

    bindings: dict[str, Expr] = {ZETA: rec("eta"),
                                 **{n: Var(n) for n in ef.AXIS_ARRAYS}}
    if HALF_INV_ZETA in frag.binds:
        bindings[HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), rec("eta"))
    for d in range(3):
        bindings[Factor.pa(d).name_in_kernel] = rec("XPA", Lit(d))
        bindings[Factor.pb(d).name_in_kernel] = rec("XPB", Lit(d))

    if ao:
        read = [Decl("oc", Scalar.I32, init=rec("off_c")),
                Decl("od", Scalar.I32, init=rec("off_d"))]
    else:
        read = [Decl("pk", Scalar.I32, init=rec("pair"))]
    extent = Lit(ef.axis_table_extent(lc, ld))
    q = Var("q")

    def row(c: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def cell(d: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            if ao:
                x: Expr = Index("Dao", (
                    _c(Fn.ADD, Var("oc"), _position("c")),
                    _c(Fn.ADD, Var("od"), _position("d"))))
            else:
                x = Index("Dcd", (Var("pk"), _c(
                    Fn.ADD, _c(Fn.MUL, _position("c"), Lit(n_d)),
                    _position("d"))))
            t = tuple(Var(f"t{a}") for a in "xyz")
            base = [f"e{a}0" for a in "xyz"]

            def e(dd: int) -> Index:
                return Index(ef.AXIS_ARRAYS[dd],
                             (_c(Fn.ADD, Var(base[dd]), t[dd]),))

            def top(dd: int) -> Expr:
                return _c(Fn.ADD, _c(Fn.ADD, c[dd], d[dd]), Lit(1))

            at = Index(ACC, (Var("h"),))
            z_loop = For("tz", Lit(0), top(2), Lit(1), (
                Decl("h", Scalar.I32, init=ef.hermite_index(t)),
                Assign(at, _c(Fn.ADD, at, _c(Fn.MUL, Var("x"), _c(
                    Fn.MUL, Var("sk"), _c(Fn.MUL, Var("exy"), e(2)))))),
            ))
            y_loop = For("ty", Lit(0), top(1), Lit(1), (
                Decl("exy", Scalar.F64, init=_c(Fn.MUL, e(0), e(1))),
                z_loop,
            ))
            return (
                Decl("x", Scalar.F64, init=x),
                *(Decl(base[dd], Scalar.I32,
                       init=ef.axis_table_index(c[dd], d[dd], 0, ld, L))
                  for dd in range(3)),
                For("tx", Lit(0), top(0), Lit(1), (y_loop,)),
            )
        return (ef.cartesian_loops(ld, ("dx", "dy", "dz"), cell, tag="d"),)

    dq, adq, dmax = Var("dq"), Var("adq"), Var("dmax")
    per_rec: list[Stmt] = [
        *read,
        Decl("sk", Scalar.F64, init=rec("s")),
        Decl("wk", Scalar.F64, init=rec("w")),
        Note(LOWSTACK_NOTE),
        Note("the record's per-axis E^{ij}_t, one local table per axis "
             "(e_fragment.axis_tables_fragment), undressed; no E^{cd}_q "
             "block is formed, and the dressing s arrives as a host scalar"),
        *(Decl(n, Scalar.F64, (extent,)) for n in ef.AXIS_ARRAYS),
        Inline(frag, bindings),
        *((Note(DENSITY_AO),) if ao else ()),
        Note(("acc_q += Dao[oc + i, od + j] * (s[k] * E[r, q])" if ao
              else "acc_q += Dcd[pair[k], r] * (s[k] * E[r, q])")
             + ": r outer (block_rows order), the host's order "
             "(ao.j.dq_from_host); q over the row's reachable box only "
             "(t_d <= c_d + d_d), each at hermite_index(t), E[r, q] = "
             "(E_x E_y) E_z formed here"),
        Decl(ACC, Scalar.F64, (Lit(n_q),)),
        For("q", Lit(0), Lit(n_q), Lit(1), (
            Assign(Index(ACC, (q,)), Lit(0.0)),)),
        ef.cartesian_loops(lc, ("cx", "cy", "cz"), row, tag="c"),
        Note("D[k, q] = w[k] * acc_q: the host's weight, applied last; "
             "Dmax[k] = max_q |D[k, q]|, over the weighted, dressed D_q -- "
             "not the AO density's largest element"),
        Decl("dmax", Scalar.F64, init=Lit(0.0)),
        For("q", Lit(0), Lit(n_q), Lit(1), (
            Decl("dq", Scalar.F64, init=_c(Fn.MUL, Var("wk"),
                                           Index(ACC, (q,)))),
            Assign(rec("D", q), dq),
            Decl("adq", Scalar.F64, init=dq),
            If(_c(Fn.LT, dq, Lit(0.0)), (Assign(adq, _c(Fn.SUB, Lit(0.0), dq)),)),
            If(_c(Fn.LT, dmax, adq), (Assign(dmax, adq),)),
        )),
        Assign(rec("Dmax"), dmax),
    ]
    body: tuple[Stmt, ...] = (
        Note(f"D_q = w SUM_cd D_cd s E^cd_q and Dmax = max_q |D_q|, ket pair "
             f"class ({lc},{ld}), LOW-STACK (#586): j_engine.md 'Step 2: the "
             f"pre-contraction kernel'; [JEngine, v2, eq. 20]"),
        Note(WEIGHT_IS_THE_HOSTS),
        Note("launch: one thread per record, k = threadIdx.x + blockIdx.x * "
             "blockDim.x"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, k, Var("n_rec")), tuple(per_rec)),
    )
    return Kernel(name, _params(ao, n_cd, n_q), body)
