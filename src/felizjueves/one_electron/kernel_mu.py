"""The dipole kernel, `Target.DIPOLE`, built as a `Kernel` (#521), and the
quadrupole kernel, `Target.QUADRUPOLE`, the same builder at `e = 2`.

    build_mu_kernel(spec | (la, lb), *, with_overlap=False) -> Kernel
    build_quad_kernel(spec | (la, lb))                      -> Kernel
    mu_name(spec | (la, lb), *, with_overlap=False)         -> str, unbuilt
    quad_name(spec | (la, lb))                              -> str, unbuilt
    mu_spec(la, lb) / quad_spec(la, lb)                     -> Spec
    n_components(with_overlap=False, e=1)                   -> 3, 4 or 6
    launch_shape(n_rec, threads)                            -> (grid, block)

THE QUADRUPOLE. `Target.QUADRUPOLE` (`spec.py`, "A QUADRUPOLE CLASS") is
this module's kernel with the fragment at `e = 2`: `quad_{la}{lb}` writes the
six second moments `<a| q (r - C)_d (r - C)_e |b>`, xx, xy, xz, yy, yz, zz
(`e_fragment.multipole_components(2)`), into `MU[6, n_ao, n_ao]`, with the
dipole's parameter block, record, `sq` and scatter unchanged. At `e = 2` the
moments read `1 / (2 zeta)` (`M^2_0 = X_PC^2 + 1/(2 zeta)`), so the record
also forms `hiz = 0.5 / zeta`, as every fragment that reads it does.
`with_overlap` is the dipole's alone (`from_order = 0` at `e = 2` would also
write the dipole); a quadrupole spec refuses it. Everything below that says
"dipole" holds for the quadrupole with six components for three.

ONE FUSED KERNEL PER PAIR CLASS WRITES EVERY COMPONENT. `mu^d_ab =
<a| q (r_d - C_d) |b>`, d = x, y, z, by the Hermite route
(`one_electron/multipole.py`'s module docstring): per primitive-pair record
`k` of one pair class `(la, lb)`, per component `d` and per Cartesian pair
`r = (i, j)`,

    v = (q * s[k]) * ((Mx * My) * Mz),   M_d' = SUM_{t <= min(e_d', i+j)} E_t M^{e_d'}_t

with the record loaded and the `E` tables built ONCE for all components
(`e_fragment.multipole_fragment(la, lb, 1)`, inlined once), and each value
`AtomicAdd`-ed into `MU[d, :, :]` where it is formed, through the
fragment's `store` hook: there is no `[n_comp, rows]` local array, and no
per-component kernel. The scatter is `kernel_jab`'s rule, UNCHANGED (the
dipole operator is real and symmetric in `a, b`), per component:

    off-diagonal (off_a != off_b):  MU[d, off_a+i, off_b+j] += v;
                                    MU[d, off_b+j, off_a+i] += v
    diagonal     (off_a == off_b):  only i >= j;  MU[d, off_a+i, off_a+j] += v,
                                    and for i != j also
                                    MU[d, off_a+j, off_a+i] += v

Because the hook places ONE statement per value, the rule is spelled per
value: `If(od, (both mirror adds), (the diagonal's adds, or none when
i < j))`, `od = (oa < ob) + (ob < oa)` formed once per record. `v`'s
expression is written in each add of its branch, not bound to a local --
the same product, so the same bits.

`with_overlap=True` writes the `e = 0` component first, as component 0
(the fragment's `from_order=0`): `(q * s) * ((E_x E_y) E_z)`, at `q = 1`
bitwise `kernel_s`'s value (`1.0 * s` is `s`), then x, y, z as 1, 2, 3.

THE PARAMETER BLOCK -- `kernel_s`'s, plus `P`, the origin and the charge:

    params  n_rec n_ao (I32)
            zeta_ab  p_x p_y p_z  pa_x pa_y pa_z  pb_x pb_y pb_z  s_ab  [n_rec]
            off_a  off_b  [n_rec]  (I32)
            C_x  C_y  C_z  q  (F64 scalars: the origin and the charge, RUNTIME)
            MU[n_comp, n_ao, n_ao]  (OUT, read and AtomicAdd-ed; the
                                     component stride is n_ao^2)

- Every per-record array is flat, one pointer per scalar, as in `kernel_s`
  (`kernel_s.pa_name` / `pb_name`, imported), and `P` the same way, under
  `kernel_v`'s names `p_x p_y p_z` (`e_fragment.p_name`), in `kernel_v`'s
  place, before `pa_*`. So `launch.OneEClass.arrays(Target.DIPOLE)` is
  `V`'s record arrays, and a plan carries them already.
- The origin `C` and the charge `q` are runtime scalars, not key fields
  (`spec.py`, "A DIPOLE CLASS"): one kernel serves every origin and every
  charge. `q` is applied once per record, `sq = q * s_ab[rec]`, the host's
  `qs = q * s` (`one_electron.multipole.multipole_block`), so `q = -1` is
  exactly the negation of `q = 1`.
- `MU` is `Intent.OUT`, READ by every `AtomicAdd`: ZEROED BY THE CALLER
  before the first launch of a build, carried as a `Note`.

NO PREFACTOR IS FORMED HERE, as in `kernel_s`: `s_ab` is the host's
`c_a c_b (pi/zeta)^{3/2}` (`ao.pairs.dress`'s scale), the fragment's seed
carries `K_AB`, and the Hermite moments omit their `sqrt(pi/zeta)`, which
is in `s_ab`.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec` (`launch_shape`). Interpreted (threads in order, blocks in
order) it runs the records IN ORDER.

WHY `==` IN THE INTERPRETER. `multipole_block` forms, per record, the axis
sums from `0.0` (`0.0 + E_0 X_PC + E_1 * 1.0` at `e = 1`, which the
fragment folds to `E_0 X_PC + E_1`: `0.0 + x` and `1.0 * x` are `x` up to a
zero's sign), the product `((1.0 f_x) f_y) f_z`, and `acc += (q s) v` from
`0.0`, records in `primitive_pairs` order. The kernel does the same
multiplications in the same association and the same `+=` chain into the
caller's zeroed `MU`. As for `S`, the diagonal rule keeps a diagonal shell
pair's LOWER triangle, `one_electron.one_electron.matrix` reads the UPPER:
the bitwise witness is `multipole_block` placed by `kernel_jab`'s rule, and
`multipole_matrix` is held to a stated bound (test/test_kernel_mu.py).

A builder: vendor-neutral IR. It imports neither `ao/` nor
`one_electron.multipole` (the reference it is checked against), nor the
interpreter, nor `reference/`.
"""

from __future__ import annotations

from felizjueves.ir import (
    AtomicAdd,
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
from felizjueves.mmd import e_fragment as ef
from felizjueves.one_electron.kernel_s import AXES, REC, pa_name, pb_name
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "E_ORDER",
    "MU",
    "ORDER",
    "ZEROED_MU",
    "build_mu_kernel",
    "build_quad_kernel",
    "launch_shape",
    "mu_name",
    "mu_spec",
    "n_components",
    "p_name",
    "quad_name",
    "quad_spec",
]

MU = "MU"  # the output: [n_comp, n_ao, n_ao]
E_ORDER = 1  # the dipole: multipole order 1 (a builder fact, not a key field)
# The multipole order each target's kernel is built at, and its name prefix.
ORDER = {Target.DIPOLE: E_ORDER, Target.QUADRUPOLE: 2}
_PREFIX = {Target.DIPOLE: "mu", Target.QUADRUPOLE: "quad"}

ATOMIC_REASON = (
    "AtomicAdd: several primitive records of one shell pair, and both mirror "
    "positions, add into the same MU elements (kernel_jab's scatter rule, "
    "per component)")

ZEROED_MU = (
    "PRECONDITION: MU is zeroed by the caller before the first MU launch of "
    "a build; this kernel only adds")

_DIAGONAL = (
    "diagonal shell pair (od == 0): the lower triangle i >= j only, each "
    "mirrored once -- kernel_jab's diagonal rule")


def mu_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.DIPOLE`."""
    return Spec(Operator.DIPOLE, 0, (la, lb), Target.DIPOLE)


def quad_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.QUADRUPOLE`."""
    return Spec(Operator.QUADRUPOLE, 0, (la, lb), Target.QUADRUPOLE)


def _class(cls: Spec | tuple[int, int],
           default=mu_spec) -> tuple[int, int, Target]:
    """`(la, lb, target)` of a dipole or quadrupole key; a bare pair is
    read through `default` (`mu_spec`, or `quad_spec`)."""
    spec = cls if isinstance(cls, Spec) else default(*cls)
    if spec.target not in ORDER:
        raise ValueError(f"{spec.key()} is not a Target.DIPOLE key (nor a "
                         "Target.QUADRUPOLE one)")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the multipole kernel takes no "
                         "levers")
    la, lb = spec.am
    return la, lb, spec.target


def n_components(with_overlap: bool = False, e: int = E_ORDER) -> int:
    """`MU`'s leading extent: the components of order `e` (x, y, z at the
    dipole's `e = 1`; xx, xy, xz, yy, yz, zz at `e = 2`), and `S` first when
    `with_overlap` (the dipole only)."""
    return len(ef.multipole_components(e)) + bool(with_overlap)


def launch_shape(n_rec: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per record, `ceil(n_rec/threads)`
    blocks. No `WarpReduce`, so a partial last block is legal."""
    if type(n_rec) is not int or n_rec < 1:
        raise ValueError(f"n_rec = {n_rec!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_rec // threads), (threads, 1)


def mu_name(cls: Spec | tuple[int, int], *, with_overlap: bool = False
            ) -> str:
    """The name `build_mu_kernel(cls, with_overlap=...)` gives its kernel,
    without building it: `mu_{la}{lb}`, `mu_{la}{lb}_s` with the overlap,
    and for a quadrupole key `quad_{la}{lb}`."""
    la, lb, target = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}" + ("_s" if with_overlap else "")


def quad_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_quad_kernel(cls)` gives its kernel: `quad_{la}{lb}`."""
    return mu_name(cls if isinstance(cls, Spec) else quad_spec(*cls))


def p_name(d: int) -> str:
    """The parameter holding axis `d` of the pair centre `P`, one double
    per record: `p_x`, `kernel_v`'s name (the fragment's bind is
    `e_fragment.p_name`, `P_x`, so the two never collide)."""
    return f"p_{AXES[d]}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def _store(la: int, lb: int):
    """The fragment's `store(comp, row, value)`: `kernel_jab`'s rule for one
    value of component `comp`, scaled by the record's `sq`."""
    nb = len(ef.cartesians(lb))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    def store(comp: int, row: int, value: Expr) -> Stmt:
        i, j = divmod(row, nb)
        w = _c(Fn.MUL, Var("sq"), value)

        def at(r: Expr, c: Expr) -> Index:
            return Index(MU, (Lit(comp), r, c))

        off_diag = (AtomicAdd(at(plus("oa", i), plus("ob", j)), w),
                    AtomicAdd(at(plus("ob", j), plus("oa", i)), w))
        diag: tuple[Stmt, ...] = ()
        if la == lb and i >= j:
            diag = (AtomicAdd(at(plus("oa", i), plus("oa", j)), w),)
            if i != j:
                diag += (AtomicAdd(at(plus("oa", j), plus("oa", i)), w),)
        return If(Var("od"), off_diag, diag)

    return store


def build_mu_kernel(cls: Spec | tuple[int, int], *,
                    with_overlap: bool = False,
                    name: str | None = None) -> Kernel:
    """The fused dipole kernel of one pair class, every component.

    `cls` is a `Target.DIPOLE` `Spec` or `(la, lb)`, or a
    `Target.QUADRUPOLE` `Spec` (`build_quad_kernel` takes the bare pair).
    See the module docstring.
    """
    la, lb, target = _class(cls)
    e = ORDER[target]
    if with_overlap and target is not Target.DIPOLE:
        raise ValueError(f"with_overlap is the dipole's alone, not "
                         f"{target.value}'s (module docstring)")
    rec = Var(REC)
    frag = ef.multipole_fragment(
        la, lb, e, store=_store(la, lb),
        store_binds=(MU, "sq", "oa", "ob", "od"),
        from_order=0 if with_overlap else None)

    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA)}
    head: list[Stmt] = [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
    ]
    if ef.HALF_INV_ZETA in frag.binds:
        head.append(Decl(ef.HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))))
        binds[ef.HALF_INV_ZETA] = Var(ef.HALF_INV_ZETA)
    for d in range(3):
        binds[Factor.pa(d).name_in_kernel] = Index(pa_name(d), (rec,))
        binds[Factor.pb(d).name_in_kernel] = Index(pb_name(d), (rec,))
        binds[ef.p_name(d)] = Index(p_name(d), (rec,))
        binds[ef.origin_name(d)] = Var(ef.origin_name(d))
    for b in (MU, "sq", "oa", "ob", "od"):
        binds[b] = Var(b)
    binds = {b: v for b, v in binds.items() if b in frag.binds}
    head += [
        Note("sq = q s: the charge applied once per record; s = c_a c_b "
             "(pi/zeta)^{3/2} is the host's (ao.pairs.dress), no prefactor "
             "and no K_AB is formed here"),
        Decl("sq", Scalar.F64, init=_c(Fn.MUL, Var("q"),
                                       Index("s_ab", (rec,)))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Note("od != 0: an off-diagonal shell pair (kernel_jab's rule)"),
        Decl("od", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))),
        Note(ATOMIC_REASON),
        Note("every component of this record from one set of E tables, each "
             "value scattered into MU where it is formed"
             + (" (component 0 the overlap)" if with_overlap else "")),
        Inline(frag, binds),
    ]
    what = ("dipole, class (la, lb) = ({la}, {lb}): mu_ab = <a| q (r - C) "
            "|b>" if target is Target.DIPOLE else
            "quadrupole, class (la, lb) = ({la}, {lb}): Q_ab = <a| q "
            "(r - C)_d (r - C)_e |b>, xx xy xz yy yz zz")
    body: tuple[Stmt, ...] = (
        Note(what.format(la=la, lb=lb)
             + " by the Hermite route (one_electron/multipole.py)"),
        Note(ZEROED_MU),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), tuple(head)),
    )
    nr, nao = Var("n_rec"), Var("n_ao")
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        *(_f64(p_name(d), (nr,)) for d in range(3)),
        *(_f64(pa_name(d), (nr,)) for d in range(3)),
        *(_f64(pb_name(d), (nr,)) for d in range(3)),
        _f64("s_ab", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        *(Param(ef.origin_name(d), Scalar.F64) for d in range(3)),
        Param("q", Scalar.F64),
        Param(MU, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(n_components(with_overlap, e)), nao, nao)),
    )
    return Kernel(name or mu_name(cls, with_overlap=with_overlap),
                  params, body)


def build_quad_kernel(cls: Spec | tuple[int, int], *,
                      name: str | None = None) -> Kernel:
    """The fused quadrupole kernel of one pair class, its six components:
    `build_mu_kernel` at `e = 2` (module docstring, "THE QUADRUPOLE").

    `cls` is a `Target.QUADRUPOLE` `Spec` or `(la, lb)`.
    """
    la, lb, target = _class(cls, default=quad_spec)
    if target is not Target.QUADRUPOLE:
        raise ValueError(f"{mu_spec(la, lb).key()} is not a "
                         "Target.QUADRUPOLE key")
    return build_mu_kernel(quad_spec(la, lb), name=name)
