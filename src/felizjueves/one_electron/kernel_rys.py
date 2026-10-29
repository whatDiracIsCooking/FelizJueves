"""The one-electron kernels by the Rys route: `S`, `T` and `V`.

    build_rys_kernel(spec | (target, la, lb)) -> Kernel   one per class
    rys_name(spec | (target, la, lb))         -> str      its name, unbuilt
    rys_spec(target, la, lb)                  -> Spec     the class's key

`Target.S`, `T` and `V` on `Engine.RYS` (`spec.py`, "A RYS ONE-ELECTRON
CLASS"), every shell `l <= L_MAX = 3` (through (ff)), Cartesian. The MMD
kernels (`kernel_s`, `kernel_t`, `kernel_v`) and the HGP ones
(`kernel_hgp`) are untouched; this is the third engine's instance of the
same three matrices, and a DROP-IN for `kernel_hgp`'s unit of the same
class: the same parameters, staging (`OneEClass.hgp_arrays`), launch
(`kernel_hgp.launch_shape`) and output.

THE ROUTE: the Rys factorisation (#468, `experimental/rys/`; the fused F's
`eri/fock/kernel_frys.py` is its two-electron instance). A one-electron
integral over a primitive pair is a product of three one-dimensional
integrals, and for `V` a sum over the nodes of the `N = (la + lb) // 2 + 1`
point Rys rule at `T = zeta |P - C|^2` (`ir.Rys`, the device `fj_rys<N>`,
`felizjueves.rys.roots`), whose weights carry the Boys function:
`SUM_n w_n x_n^m = F_m(T)` for `m <= 2N - 1`, enough for the polynomial of
degree `la + lb` in `x` that the boundary is. Per primitive pair, `PA = P - A`,
per axis `d`:

    overlap  I_d(0) = 1,  I_d(i + 1) = PA_d I_d(i) + i/(2 zeta) I_d(i - 1)
             (e|0)  = u I_x(e_x) I_y(e_y) I_z(e_z),     u = s_ab K_AB
    nuclear  per nucleus C, per node (x_n, w_n):
             I_d(0) = 1,  I_d(i + 1) = C00_d I_d(i) + i B10 I_d(i - 1)
             C00_d  = PA_d + x_n (C - P)_d,   B10 = (1 - x_n) / (2 zeta)
             (e|0) -= Z_C w0 w_n I_x(e_x) I_y(e_y) I_z(e_z),
             w0 = u (2/sqrt(pi)) sqrt(zeta)

The nuclear one IS `rys.recurrence.root_loop`'s bra recurrence, unchanged,
at the point-charge limit of the ket (`q -> infinity`: `W -> C`, so `WP =
C - P`, and `q/(p + q) -> 1`, so `B10 = (1/2p)(1 - x)`), with a ket span of
0 -- the same limit `kernel_hgp` takes of eq 25. The kernel binds `WP_*` and
`q_ppq = 1` so and passes the attractive sign in the weight scale
(`S = -Z w0`), so `root_loop`'s `acc += S w_n I_x I_y I_z` is `kernel_v`'s
`acc - Z u`. The overlap integrand is a Gaussian times a polynomial, exact
with no node at all: its 1D recurrence is the nodeless (`x = 0`) form of the
same one, `C00 = PA`, `B10 = 1/(2 zeta)`. `T` is three overlap tables
weighted by `beta^n` (`kernel_hgp`'s docstring), so it is the overlap's 1D
tables too, one product per boundary element shared by the tables that
read it.

THE BOUNDARY AND THE HRR ARE `kernel_hgp`'s, NOT REWRITTEN. The 1D products
form exactly the boundary `(e|0)`, `la <= |e| <= la + lb` (`T`: to
`la + lb + 2`), that `kernel_hgp.contracted_tables`' HRR DAGs read; they
are accumulated over the pair's records into `kernel_hgp.acc_name`'s
accumulators, and `kernel_hgp.assemble` runs the HRR on the contracted
values (`hgp.hrr`'s eq 28), the operator's combination and the plain
mirrored stores. #468's transfer steps in 1D (per node, per primitive pair)
are not here: on contracted values, once per shell pair, is the cheaper
place, as in `kernel_frys`. So the Rys and HGP kernels of one class differ
in the per-record body and in two notes, and nowhere else.

DEAD CODE IS NEVER EMITTED (#92's filter): a 1D entry, `PA_d`, `inv2p` /
`B10` and a node `x_n` are formed only where a boundary element reads them;
`(ss)` reads none of them.

ROUND-OFF, NOT BITWISE, against the MMD host references
(`one_electron.overlap`, `kinetic`, `potential`) and the HGP kernels:
another recurrence, another association, and for `V` a quadrature in place
of the Boys function (`rys.roots`: within 7e-15 relative of a 40-digit
route, `N <= 5`, `T` in `[0, 1e6]`). The tests state their bounds
(test/test_kernel_rys_one_e.py), and hold every matrix to libcint's too.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`; of `one_electron/` only `kernel_hgp`, whose
skeleton it fills.
"""

from __future__ import annotations

import math

from felizjueves.hgp.vrr import ZERO, Cart
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    Index,
    Kernel,
    Lit,
    Note,
    Rys,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.one_electron.kernel_hgp import (
    acc_name,
    assemble,
    boundary_union,
    contracted_tables,
)
from felizjueves.rys import recurrence as rr
from felizjueves.rys.roots import N_MAX, n_roots
from felizjueves.spec import OPERATOR_OF, Engine, Spec, Target

__all__ = ["L_MAX", "TARGETS", "build_rys_kernel", "rys_name", "rys_spec"]

L_MAX = 3  # through (ff): the scope this instance is built and checked for
TARGETS = (Target.S, Target.T, Target.V)
_PREFIX = {Target.S: "rs", Target.T: "rt", Target.V: "rv"}
_AXES = "xyz"
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_REC = "kr"  # kernel_hgp.assemble's record loop variable
_NUC = "c"  # the nucleus loop's variable


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def rys_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `target` on `Engine.RYS`."""
    return Spec(OPERATOR_OF[target], 0, (la, lb), target, engine=Engine.RYS)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else rys_spec(*cls)
    if spec.target not in TARGETS:
        raise ValueError(f"{spec.key()}: the Rys one-electron kernels are "
                         f"{[t.value for t in TARGETS]}")
    if spec.engine is not Engine.RYS:
        raise ValueError(f"{spec.key()} is not a Rys key; the other engines' "
                         "kernels of this target are kernel_s / kernel_t / "
                         "kernel_v (MMD) and kernel_hgp (HGP)")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Rys one-electron kernels take "
                         "no levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the Rys one-electron kernels are "
                         f"built for shells through f only (l <= {L_MAX})")
    if n_roots(la + lb) > N_MAX:  # unreachable at L_MAX = 3; kept honest
        raise ValueError(f"{spec.key()}: needs {n_roots(la + lb)} Rys "
                         f"nodes; rys.roots stops at N = {N_MAX}")
    return spec.target, la, lb


def rys_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_rys_kernel(cls)` gives its kernel, unbuilt:
    `rs_{la}{lb}`, `rt_`, `rv_` (`kernel_hgp`'s `hs_` with the engine's
    letter)."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


# ---------------------------------------------------------------------------
# Small spellings
# ---------------------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(a: Expr | None, b: Expr | None) -> Expr | None:
    """`a * b`, where None is the literal 1 (the x and y seeds)."""
    if a is None:
        return b
    if b is None:
        return a
    return _c(Fn.MUL, a, b)


def _sum(xs: list[Expr]) -> Expr:
    out = xs[0]
    for x in xs[1:]:
        out = _c(Fn.ADD, out, x)
    return out


def _tops(es: tuple[Cart, ...]) -> tuple[int, int, int]:
    """The highest 1D index each axis's boundary reads."""
    return tuple(max(e[d] for e in es) for d in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The per-record bodies
# ---------------------------------------------------------------------------


def _overlap_tables(es: tuple[Cart, ...]
                    ) -> tuple[list[Stmt], list[dict[int, Expr | None]]]:
    """The nodeless 1D recurrence per axis, the z axis seeded with `u`
    (module docstring): its `Decl`s and, per axis, `i -> I_d(i)` (None for
    the literal 1)."""
    stmts: list[Stmt] = []
    tables: list[dict[int, Expr | None]] = []
    for d, ax in enumerate(_AXES):
        g: dict[int, Expr | None] = {0: Var("u") if d == 2 else None}
        for i in range(_tops(es)[d]):
            e = _mul(Var(f"PA_{ax}"), g[i])
            if i:
                e = _c(Fn.ADD, e, _mul(_c(Fn.MUL, Lit(float(i)),
                                          Var("inv2p")), g[i - 1]))
            n = f"O{ax}_{i + 1}"
            stmts.append(Decl(n, Scalar.F64, init=e))
            g[i + 1] = Var(n)
        tables.append(g)
    return stmts, tables


def _product(tables: list[dict[int, Expr | None]], e: Cart) -> Expr:
    out = _mul(_mul(tables[0][e[0]], tables[1][e[1]]), tables[2][e[2]])
    assert out is not None  # the z axis's seed is never the literal 1
    return out


def _record_scalars(es: tuple[Cart, ...]) -> list[Stmt]:
    """`zeta`, `inv2p` and `PA_d` where the 1D recurrences read them, and
    the seed `u`."""
    k = Var(_REC)
    tops = _tops(es)
    out: list[Stmt] = [Decl("zeta", Scalar.F64, init=Index("zeta_ab", (k,)))]
    if max(tops) > 1:
        out.append(Decl("inv2p", Scalar.F64,
                        init=_c(Fn.DIV, Lit(0.5), Var("zeta"))))
    out += [Decl(f"PA_{ax}", Scalar.F64, init=Index(f"pa_{ax}", (k,)))
            for d, ax in enumerate(_AXES) if tops[d]]
    out += [
        Note("u = (0|0) = s_ab K_AB, K_AB = exp(zeta (P-A).(P-B)): s_ab = "
             "c_a c_b (pi/zeta)^{3/2} is the host's (ao.pairs.dress)"),
        Decl("u", Scalar.F64, init=_c(Fn.MUL, Index("s_ab", (k,)), _c(
            Fn.EXP, _c(Fn.MUL, Var("zeta"), _sum([
                _c(Fn.MUL, Index(f"pa_{ax}", (k,)), Index(f"pb_{ax}", (k,)))
                for ax in _AXES]))))),
    ]
    return out


def _overlap_record(target: Target,
                    bounds: dict[int, tuple[Cart, ...]]) -> list[Stmt]:
    """`S` and `T`: the 1D tables once, then every boundary element's
    product into its tables' accumulators (`T`'s weighted by `beta^n`)."""
    es = boundary_union(bounds)
    record = _record_scalars(es)
    table_stmts, tables = _overlap_tables(es)
    if target is Target.T:
        record += [Decl("eb", Scalar.F64, init=Index("eb_ket", (Var(_REC),))),
                   Note("the tables' weights beta^n: beta the ket exponent"),
                   Decl("eb2", Scalar.F64,
                        init=_c(Fn.MUL, Var("eb"), Var("eb")))]
    record += [Note("the 1D overlap recurrence per axis, I(i + 1) = PA I(i) "
                    "+ i/(2 zeta) I(i - 1), the seed u in the z axis"),
               *table_stmts]
    if target is Target.S:
        record.append(Note("the boundary: (e|0) += I_x I_y I_z"))
        for e in bounds[0]:
            a = Var(acc_name(0, e))
            record.append(Assign(a, _c(Fn.ADD, a, _product(tables, e))))
        return record
    record.append(Note("the boundary: (e|0) = I_x I_y I_z, once per element, "
                       "into each table that reads it weighted by beta^n"))
    for e in es:
        record.append(Decl(f"ov_{'_'.join(map(str, e))}", Scalar.F64,
                           init=_product(tables, e)))
    weight = {0: None, 1: Var("eb"), 2: Var("eb2")}
    for n, bnd in bounds.items():
        for e in bnd:
            a = Var(acc_name(n, e))
            v = _mul(weight[n], Var(f"ov_{'_'.join(map(str, e))}"))
            record.append(Assign(a, _c(Fn.ADD, a, v)))
    return record


def _nuclear_record(la: int, lb: int,
                    bounds: dict[int, tuple[Cart, ...]]) -> list[Stmt]:
    """`V`: per nucleus the Rys rule at `T = zeta |P - C|^2`, then
    `rys.recurrence.root_loop`'s bra recurrence at the point-charge limit
    (module docstring)."""
    es = bounds[0]
    k, c = Var(_REC), Var(_NUC)
    cls = (la, lb, 0, 0)
    need = rr.scalar_needs(cls)
    n = n_roots(la + lb)
    record = _record_scalars(es)
    if "q_ppq" in need:
        record += [Note("the point-charge limit of the ket, q -> infinity: "
                        "q/(p+q) -> 1, so B10 = (1 - x)/(2 zeta)"),
                   Decl("q_ppq", Scalar.F64, init=Lit(1.0))]
    pc = [Var(f"WP_{ax}") for ax in _AXES]
    nucleus: list[Stmt] = [
        Note("the point-charge limit: W -> C, so WP = C - P"),
        *(Decl(f"WP_{ax}", Scalar.F64, init=_c(
            Fn.SUB, Index("Cnuc", (c, Lit(d))), Index(f"p_{ax}", (k,))))
          for d, ax in enumerate(_AXES)),
        Decl(rr.NODES, Scalar.F64, (Lit(n),)),
        Decl(rr.WEIGHTS, Scalar.F64, (Lit(n),)),
        Note(f"the {n}-point Rys rule at T = zeta |P - C|^2: SUM w x^m = F_m"),
        Rys(rr.NODES, rr.WEIGHTS, n, _c(Fn.MUL, Var("zeta"), _sum(
            [_c(Fn.MUL, v, v) for v in pc]))),
        Note("the weight scale, S = -Z w0: the attractive sign enters here, "
             "so the root loop's acc + S w I_x I_y I_z is acc - Z u"),
        Decl(rr.WEIGHT_SCALE, Scalar.F64, init=_c(
            Fn.MUL, Lit(-1.0), _c(Fn.MUL, Index("Z", (c,)), Var("w0")))),
        *rr.root_loop(cls, tuple((e, ZERO) for e in es),
                      lambda e, f: acc_name(0, e), n_roots=n),
    ]
    return [
        *record,
        Note("w0 = s_ab K_AB (2/sqrt(pi)) sqrt(zeta) = c_a c_b K_AB "
             "2 pi / zeta"),
        Decl("w0", Scalar.F64, init=_c(Fn.MUL, _c(
            Fn.MUL, Var("u"), Lit(_TWO_OVER_SQRT_PI)),
            _c(Fn.SQRT, Var("zeta")))),
        Note("the nuclei, serially"),
        For(_NUC, Lit(0), Var("n_nuc"), Lit(1), tuple(nucleus)),
    ]


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def build_rys_kernel(cls: Spec | tuple[Target, int, int], *,
                     name: str | None = None) -> Kernel:
    """The Rys one-electron kernel of one class.

    `cls` is a Rys `Target.S` / `T` / `V` `Spec`, or `(target, la, lb)`.
    See the module docstring.
    """
    target, la, lb = _class(cls)
    hrrs, bounds = contracted_tables(target, la, lb)
    if target is Target.V:
        record = _nuclear_record(la, lb, bounds)
        how = (f"the {n_roots(la + lb)}-point Rys rule per nucleus and its "
               "1D recurrence")
    else:
        record = _overlap_record(target, bounds)
        how = "the nodeless 1D recurrence"
    what = {Target.S: "overlap S", Target.T: "kinetic T",
            Target.V: "nuclear attraction V"}[target]
    return assemble(
        target, la, lb, hrrs, bounds, tuple(record),
        name=name or rys_name((target, la, lb)),
        title=(f"Rys {what}, class (la, lb) = ({la}, {lb}): {how} per "
               "primitive pair into the boundary (e|0), the contraction, the "
               "HRR on contracted values (one_electron/kernel_rys.py)"),
        per_record=("the pair's primitive pairs, serially: the Rys "
                    "factorisation per record, I_x I_y I_z"))
