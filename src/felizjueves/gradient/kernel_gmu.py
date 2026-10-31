"""The multipole gradient kernels: `Target.DIPOLE` and `Target.QUADRUPOLE`
at `derivative_order = 1`, built as `Kernel`s.

    build_gmu_kernel(spec | (target, la, lb)) -> Kernel   one per class
    gmu_name(spec | (target, la, lb))         -> str      gmu_ / gquad_{la}{lb}
    gmu_spec(target, la, lb)                  -> Spec     the class's d1 key
    n_components(target)                      -> 3 or 6
    launch_shape(n_rec, threads)              -> (grid, block)

Per primitive-pair record of one pair class `(la, lb)` (exponents `alpha`
on `A`, `beta` on `B`, the host's `s = c_a c_b (pi/zeta)^{3/2}`), per
component `c` of the moment `<a| q (r - C)^(ex, ey, ez) |b>` (x, y, z for
the dipole, `e = 1`; xx, xy, xz, yy, yz, zz for the quadrupole's raw second
moments, `e = 2`: `one_electron.kernel_mu`'s order), the contribution of
the pair's block to

    G[c, A, k] += SUM_mn M_mn dMU^c_mn / dR_{A,k}

(`felizjueves.gradient`'s package docstring; `gradient.host.
multipole_gradient` is the reference, the same arithmetic in the same rule):

    w_r      = M[oa+i, ob+j] + M[ob+j, oa+i]   off-diagonal shell pair
             = M[oa+i, oa+j]                   diagonal (off_a == off_b)
    E        = e_fragment.axis_tables_fragment(la + 1, lb + 1)
    M^n_t    the Hermite moments about C at X_PC = P - C (n <= e):
             M^0 = (1), M^1 = (X_PC, 1), M^2 = (X_PC^2 + 1/(2 zeta), 2 X_PC, 2)
    F^n_ij   = SUM_{t <= min(n, i+j)} E^{ij}_t M^n_t     one axis's factor
    dA F^n_ij = 2 alpha F^n_{i+1,j} - i F^n_{i-1,j}     dB likewise

    gA_{c,k} = SUM_r w_r (f_x f_y) f_z,  f_d = F^{e_d}_{a_d b_d}, the axis
               `k` factor replaced by dA F;  gB likewise

then `G[c, atom_a, k] += (q s) gA_{c,k}` and `G[c, atom_b, k] += (q s)
gB_{c,k}`, `6 n_comp` `AtomicAdd`s per record. The derivative rule acts on
the basis function, so the moments, at the record's `P`, are not
differentiated. THE ORIGIN IS NOT A NUCLEUS: `C` is fixed, so there is no
operator term (no Hellmann-Feynman analogue), and `SUM_A G[c, A, k]` is
minus the origin derivative -- `q SUM_mn M_mn S_mn` on the matching axis
for the dipole -- a check (test/test_kernel_gmu.py), not an assumption.

THE PARAMETER BLOCK, `kernel_g1e`'s plus the moment's origin and charge:

    params  n_rec n_ao n_atom (I32)
            zeta_ab  pa_x pa_y pa_z  pb_x pb_y pb_z  p_x p_y p_z  s_ab
                ea_bra  eb_ket                               [n_rec]
            off_a  off_b  atom_a  atom_b  [n_rec]  (I32)
            M[n_ao, n_ao]                                    the contracted matrix
            C_x  C_y  C_z  q  (F64 scalars: the origin and the charge, RUNTIME)
            G[n_comp, n_atom, 3]  (OUT, read and AtomicAdd-ed)

`gradient.host.records` carries every per-record array. `M` is read, never
written, and need not be symmetric. `G` is ZEROED BY THE CALLER before a
build's first launch; the kernel only adds. The origin and the charge are
runtime scalars, as at d0 (`spec.py`, "A DIPOLE CLASS"): neither is a key
field.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec`. The `E` tables are `kernel_g1e.record_head`'s local arrays at
constant indices; each axis factor and its derivative is a straight-line
local declared where first read (`kernel_g1e.Locals`).

ROUND-OFF, NOT BITWISE, against the host: the host sums each record's rows
into `gA` from `0.0` with `+=`; the kernel forms each component as one
expression, and its moment sums in the fragment's association. The
interpreted test states its bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`.
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
    "GRADIENT_MOMENTS",
    "ORDER",
    "ZEROED_G",
    "build_gmu_kernel",
    "gmu_name",
    "gmu_spec",
    "launch_shape",
    "n_components",
]

# The multipole order each target's gradient is built at, and its prefix.
ORDER = {Target.DIPOLE: 1, Target.QUADRUPOLE: 2}
GRADIENT_MOMENTS = tuple(ORDER)
_PREFIX = {Target.DIPOLE: "gmu", Target.QUADRUPOLE: "gquad"}

ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "row of every component of G")

ZEROED_G = (
    "PRECONDITION: G is zeroed by the caller before the first gradient "
    "launch of a build; this kernel only adds")


def gmu_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of the dipole or quadrupole at d1."""
    return Spec(OPERATOR_OF[target], 1, (la, lb), target)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else gmu_spec(*cls)
    if spec.target not in ORDER or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 dipole or quadrupole key")
    if spec.engine is not Engine.MMD:
        raise ValueError(f"{spec.key()} is not an MMD key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    return spec.target, la, lb


def gmu_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_gmu_kernel(cls)` gives its kernel, unbuilt:
    `gmu_{la}{lb}` (the dipole) or `gquad_{la}{lb}` (the quadrupole)."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


def n_components(target: Target) -> int:
    """`G`'s leading extent: 3 for the dipole, 6 for the quadrupole."""
    return len(ef.multipole_components(ORDER[target]))


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


def _moments(e: int, pool: Locals):
    """`M^n_t` on axis `d` for `n <= e` (module docstring): a Python
    number where it is a constant, else a pooled local."""
    if e > 2:
        raise ValueError(f"e = {e}: the moments are spelled through e = 2")

    def xpc(d: int) -> Var:
        return pool.get(ef.xpc_name(d), lambda: _c(
            Fn.SUB, Var(f"P{d}"), Var(ef.origin_name(d))))

    def moment(d: int, n: int, t: int) -> Expr | float:
        if n == 0:
            return 1.0
        if n == 1:
            return xpc(d) if t == 0 else 1.0
        if t == 0:
            return pool.get(f"M{'xyz'[d]}2_0", lambda: _c(
                Fn.ADD, _mul(xpc(d), xpc(d)), Var(ef.HALF_INV_ZETA)))
        if t == 1:
            return pool.get(f"M{'xyz'[d]}2_1", lambda: _mul(
                Lit(2.0), xpc(d)))
        return 2.0
    return moment


def build_gmu_kernel(cls: Spec | tuple[Target, int, int], *,
                     name: str | None = None) -> Kernel:
    """The gradient kernel of one dipole or quadrupole pair class at d1.

    `cls` is a d1 `Target.DIPOLE` / `QUADRUPOLE` `Spec` or `(target, la,
    lb)`. See the module docstring.
    """
    target, la, lb = _class(cls)
    e = ORDER[target]
    comps = ef.multipole_components(e)
    head = record_head(la + 1, lb + 1)
    E = axis_reader(la + 1, lb + 1)
    pool = Locals()
    moment = _moments(e, pool)

    def F(d: int, n: int, i: int, j: int) -> Expr:
        def make() -> Expr:
            terms: list[Expr] = []
            for t in range(min(n, i + j) + 1):
                m = moment(d, n, t)
                ev = E(d, i, j, t)
                terms.append(ev if isinstance(m, float) and m == 1.0 else _mul(
                    Lit(m) if isinstance(m, float) else m, ev))
            return _fold(terms)
        if n == 0:  # the overlap factor: the table entry itself
            return E(d, i, j, 0)
        return pool.get(f"F{'xyz'[d]}{n}_{i}_{j}", make)

    def dF(side: str, d: int, n: int, i: int, j: int) -> Var:
        def make() -> Expr:
            if side == "A":
                out: Expr = _mul(Var("ea2"), F(d, n, i + 1, j))
                if i:
                    out = _c(Fn.SUB, out, _mul(Lit(float(i)),
                                               F(d, n, i - 1, j)))
            else:
                out = _mul(Var("eb2"), F(d, n, i, j + 1))
                if j:
                    out = _c(Fn.SUB, out, _mul(Lit(float(j)),
                                               F(d, n, i, j - 1)))
            return out
        return pool.get(f"d{side}{'xyz'[d]}{n}_{i}_{j}", make)

    rows = ef.block_rows(la, lb)
    values: list[Stmt] = []
    for c, ex in enumerate(comps):
        for side in "AB":
            for k in range(3):
                terms = []
                for r, (a, b) in enumerate(rows):
                    f = [dF(side, d, ex[d], a[d], b[d]) if d == k
                         else F(d, ex[d], a[d], b[d]) for d in range(3)]
                    terms.append(_mul(Var(f"w{r}"),
                                      _mul(_mul(f[0], f[1]), f[2])))
                values.append(Decl(f"g{c}{side}{k}", Scalar.F64,
                                   init=_fold(terms)))
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for c in range(len(comps)):
        for side, atom in (("A", "aa"), ("B", "ab")):
            for k in range(3):
                scatter.append(AtomicAdd(
                    Index("G", (Lit(c), Var(atom), Lit(k))),
                    _mul(Var("sq"), Var(f"g{c}{side}{k}"))))
    per_rec = (
        *head,
        Note("sq = q s: the charge applied once per record"),
        Decl("sq", Scalar.F64, init=_mul(Var("q"), Var("sk"))),
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        *weights(la, lb),
        Note("the moments about C, the axis factors F^n_ij and their "
             "derivatives, each declared where first read"),
        *pool.decls,
        *values,
        *scatter,
    )
    what = ("dipole, mu = <a| q (r - C) |b>" if target is Target.DIPOLE
            else "quadrupole, <a| q (r - C)_d (r - C)_e |b>, xx xy xz yy yz "
                 "zz")
    body: tuple[Stmt, ...] = (
        Note(f"gradient of the {what}, class (la, lb) = ({la}, {lb}): "
             "G[c, A, k] += SUM_mn M_mn dMU^c_mn/dR_{A,k} "
             "(felizjueves.gradient); the origin C is fixed, not a nucleus"),
        Note(ZEROED_G),
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
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(len(comps)), nat, Lit(3))),
    )
    return Kernel(name or gmu_name((target, la, lb)), params, body)
