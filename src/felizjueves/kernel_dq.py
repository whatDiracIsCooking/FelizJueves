"""The ket pre-contraction kernel, `Target.DQ`: `D_q` and `Dmax` (#153).

    build_dq_kernel((lc, ld) | spec) -> Kernel     one per ket pair class
    build_dq_kernel(cls, density="ao") -> Kernel    the same, reading D_mu_nu
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

THE OPERATION ORDER IS THE HOST'S (`ao.j.dq_from_host`): `r` outer, `q`
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
equal) and columns of `D` are `hermite_targets(lc + ld)` (`ao.hermite`'s
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

A builder: vendor-neutral IR, no `reference/`, no interpreter, no AO layer.
"""

from __future__ import annotations

from felizjueves.e_fragment import (
    E_ARRAY,
    HALF_INV_ZETA,
    ZETA,
    block_fragment,
    block_rows,
    block_shape,
)
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
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
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "AO_PARAMS",
    "DENSITIES",
    "DENSITY_AO",
    "PARAMS",
    "WEIGHT_IS_THE_HOSTS",
    "build_dq_kernel",
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


def build_dq_kernel(cls: tuple[int, int] | Spec, *,
                    name: str | None = None,
                    density: str = "pairs") -> Kernel:
    """The pre-contraction kernel of the ket pair class `cls` (module doc).

    `density="ao"` reads `D_cd` from the AO density `Dao[n_ao, n_ao]`
    through per-record shell offsets `off_c`, `off_d` -- `kernel_jab`'s
    addressing of `J` -- in place of the host's gathered `Dcd[pair[k], r]`
    (module docstring, "THE AO DENSITY"). The default is unchanged."""
    spec = _spec_of(cls)
    ao = _density(density) == "ao"
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
        return Kernel(name, params, body)
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_pair", Scalar.I32),
        *per_record,
        Param("pair", Scalar.I32, array=True, shape=(nr,)),
        f64("Dcd", npair, Lit(n_cd)),
        *out,
    )
    assert tuple(p.name for p in params) == PARAMS
    return Kernel(name, params, body)
