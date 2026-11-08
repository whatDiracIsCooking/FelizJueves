"""The multipole hessian kernels: `Target.DIPOLE` and `Target.QUADRUPOLE`
at `derivative_order = 2`, built as `Kernel`s.

    build_d2mu_kernel(spec | (target, la, lb)) -> Kernel   one per class
    d2mu_name(spec | (target, la, lb))         -> str      d2mu_ / d2quad_{la}{lb}
    d2mu_spec(target, la, lb)                  -> Spec     the class's d2 key
    n_components(target)                       -> 3 or 6
    launch_shape(n_rec, threads)               -> (grid, block)

Per primitive-pair record of one pair class `(la, lb)` (exponents `alpha`
on `A`, `beta` on `B`, the host's `s = c_a c_b (pi/zeta)^{3/2}`), per
component `c` of the moment `<a| q (r - C)^(ex, ey, ez) |b>` (x, y, z for
the dipole, `e = 1`; xx, xy, xz, yy, yz, zz for the quadrupole's raw
second moments, `e = 2`: `one_electron.kernel_mu`'s order), the
contribution of the pair's block to

    H[c, A, k, B, l] += SUM_mn M_mn d2MU^c_mn / dR_{A,k} dR_{B,l}

(`hessian.host_1e.multipole_hessian` is the reference, by another route:
it differentiates each 3-D function as a list of moved Cartesians, the
kernel each axis's factor). It is `kernel_h1e`'s `d2s` with the overlap
factor `E^{ij}_0` replaced by the moment factor of `kernel_gmu`:

    E        = e_fragment.axis_tables_fragment(la + 2, lb + 2)
    M^n_t    the Hermite moments about C at X_PC = P - C (n <= e),
             `kernel_gmu.moment_reader`
    F^n_ij   = SUM_{t <= min(n, i+j)} E^{ij}_t M^n_t      one axis's factor

and `kernel_h1e.operated`'s rule on it, once or twice per axis
(`kernel_h1e.axis_ops`):

    dA F^n_ij = 2 alpha F^n_{i+1,j} - i F^n_{i-1,j}     dB likewise

    hXY_kl^c = SUM_r w_r PROD_d (op_d F^{e_d})_{a_d b_d}

then `H[c, A, k, A, l] += (q s) hAA_kl^c` (`k <= l` formed, mirrored),
`BB` likewise, `H[c, A, k, B, l] += (q s) hAB_kl^c` and `H[c, B, l, A, k]`
the same value: `36 n_comp` `AtomicAdd`s per record. The rule acts on the
basis function, so the moments, at the record's `P`, are not
differentiated. THE ORIGIN IS NOT A NUCLEUS: `C` is fixed, so there are
no operator terms -- the blocks are centre-centre only, as for `S`.

THE PARAMETER BLOCK is `kernel_gmu`'s, with the output
`H[n_comp, n_atom, 3, n_atom, 3]` in place of `G[n_comp, n_atom, 3]`.
`M` is read, never written, and need not be symmetric. `H` is ZEROED BY
THE CALLER before a build's first launch; the kernel only adds. The origin
and the charge are runtime scalars.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec`.

ROUND-OFF, NOT BITWISE, against the host: the association differs (per
axis here, per 3-D function there). The interpreted test states its bound
(test/test_kernel_d2mu.py).

A builder: vendor-neutral IR. It imports neither `ao/` nor
`hessian.host_1e` (its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import (
    REC,
    Locals,
    axis_reader,
    launch_shape,
    record_head,
    weights,
)
from felizjueves.gradient.kernel_gmu import ORDER, moment_reader, n_components
from felizjueves.hessian.kernel_h1e import COMPONENTS, axis_ops, operated
from felizjueves.ir import (
    AtomicAdd,
    Call,
    Decl,
    Expr,
    Fn,
    If,
    Index,
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
from felizjueves.spec import OPERATOR_OF, Engine, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "HESSIAN_MOMENTS",
    "ZEROED_H",
    "build_d2mu_kernel",
    "d2mu_name",
    "d2mu_spec",
    "launch_shape",
    "n_components",
]

HESSIAN_MOMENTS = tuple(ORDER)
_PREFIX = {Target.DIPOLE: "d2mu", Target.QUADRUPOLE: "d2quad"}

ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "blocks of every component of H")

ZEROED_H = (
    "PRECONDITION: H is zeroed by the caller before the first hessian "
    "launch of a build; this kernel only adds")


def d2mu_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of the dipole or quadrupole at d2."""
    return Spec(OPERATOR_OF[target], 2, (la, lb), target)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else d2mu_spec(*cls)
    if spec.target not in ORDER or spec.derivative_order != 2:
        raise ValueError(f"{spec.key()} is not a d2 dipole or quadrupole key")
    if spec.engine is not Engine.MMD:
        raise ValueError(f"{spec.key()} is not an MMD key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    la, lb = spec.am
    return spec.target, la, lb


def d2mu_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_d2mu_kernel(cls)` gives its kernel, unbuilt:
    `d2mu_{la}{lb}` (the dipole) or `d2quad_{la}{lb}` (the quadrupole)."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


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


def _h(c: int, x: str, k: int, y: str, l: int) -> str:  # noqa: E741
    return f"h{c}{x}{y}_{k}{l}"


def build_d2mu_kernel(cls: Spec | tuple[Target, int, int], *,
                      name: str | None = None) -> Kernel:
    """The hessian kernel of one dipole or quadrupole pair class at d2.

    `cls` is a d2 `Target.DIPOLE` / `QUADRUPOLE` `Spec` or `(target, la,
    lb)`. See the module docstring.
    """
    target, la, lb = _class(cls)
    e = ORDER[target]
    comps = ef.multipole_components(e)
    head = record_head(la + 2, lb + 2)
    E = axis_reader(la + 2, lb + 2)
    pool = Locals()
    moment = moment_reader(e, pool)

    def factor(n: int):
        """`F^n_ij` on axis `d` as an `operated` base (its `t` is 0)."""
        def F(d: int, i: int, j: int, t: int) -> Expr:
            if n == 0:  # the overlap factor: the table entry itself
                return E(d, i, j, 0)

            def make() -> Expr:
                terms: list[Expr] = []
                for u in range(min(n, i + j) + 1):
                    m = moment(d, n, u)
                    ev = E(d, i, j, u)
                    terms.append(ev if isinstance(m, float) and m == 1.0
                                 else _mul(Lit(m) if isinstance(m, float)
                                           else m, ev))
                return _fold(terms)
            return pool.get(f"F{'xyz'[d]}{n}_{i}_{j}", make)
        return F

    op = {n: operated(factor(n), pool, f"F{n}") for n in range(e + 1)}
    rows = ef.block_rows(la, lb)
    values: list[Stmt] = []
    for c, ex in enumerate(comps):
        for comp in COMPONENTS:
            ops = axis_ops(*comp)
            terms = []
            for r, (a, b) in enumerate(rows):
                f = [op[ex[d]](ops[d], d, a[d], b[d], 0) for d in range(3)]
                terms.append(_mul(Var(f"w{r}"),
                                  _mul(_mul(f[0], f[1]), f[2])))
            values.append(Decl(_h(c, *comp), Scalar.F64, init=_fold(terms)))

    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    atom = {"A": Var("aa"), "B": Var("ab")}
    for c in range(len(comps)):
        for x, k, y, l in COMPONENTS:  # noqa: E741
            v = _mul(Var("sq"), Var(_h(c, x, k, y, l)))
            scatter.append(AtomicAdd(Index("H", (
                Lit(c), atom[x], Lit(k), atom[y], Lit(l))), v))
            if x != y or k != l:  # the mirror: (B, l, A, k), or (X, l, X, k)
                scatter.append(AtomicAdd(Index("H", (
                    Lit(c), atom[y], Lit(l), atom[x], Lit(k))), v))

    per_rec = (
        *head,
        Note("sq = q s: the charge applied once per record"),
        Decl("sq", Scalar.F64, init=_mul(Var("q"), Var("sk"))),
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        *weights(la, lb),
        Note("the moments about C, the axis factors F^n_ij and their "
             "operated forms, each declared where first read"),
        *pool.decls,
        *values,
        *scatter,
    )
    what = ("dipole, mu = <a| q (r - C) |b>" if target is Target.DIPOLE
            else "quadrupole, <a| q (r - C)_d (r - C)_e |b>, xx xy xz yy yz "
                 "zz")
    body: tuple[Stmt, ...] = (
        Note(f"hessian of the {what}, class (la, lb) = ({la}, {lb}): "
             "H[c, A, k, B, l] += SUM_mn M_mn d2MU^c_mn/dR_{A,k}dR_{B,l} "
             "(felizjueves.hessian); the origin C is fixed, not a nucleus"),
        Note(ZEROED_H),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), per_rec),
    )
    nr, nao, nat = Var("n_rec"), Var("n_ao"), Var("n_atom")
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        *(_f64(f"pa_{a}", (nr,)) for a in "xyz"),
        *(_f64(f"pb_{a}", (nr,)) for a in "xyz"),
        *(_f64(f"p_{a}", (nr,)) for a in "xyz"),
        _f64("s_ab", (nr,)),
        _f64("ea_bra", (nr,)),
        _f64("eb_ket", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        _i32("atom_a", (nr,)),
        _i32("atom_b", (nr,)),
        _f64("M", (nao, nao)),
        *(Param(ef.origin_name(d), Scalar.F64) for d in range(3)),
        Param("q", Scalar.F64),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(len(comps)), nat, Lit(3), nat, Lit(3))),
    )
    return Kernel(name or d2mu_name((target, la, lb)), params, body)
