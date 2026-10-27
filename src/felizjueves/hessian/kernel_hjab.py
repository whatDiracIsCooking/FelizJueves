"""The Coulomb-energy Hessian kernel: `Target.JAB` at `derivative_order =
2`, the J chain's bra back-transform differentiated twice.

    build_hjab_kernel(spec | (la, lb)) -> Kernel   one per bra pair class
    hjab_name(spec | (la, lb))         -> str      hjab_{la}{lb}
    hjab_spec(la, lb)                  -> Spec     the class's d2 key
    hjab_pq2_spec(la, lb, lc, ld)      -> Spec     the [p|q] class of Jp2
    hjab_pq1_spec(la, lb, lc, ld)      -> Spec     the [p|q] class of Jp1
    n_p2(la, lb), n_p1(la, lb)         -> int      their Hermite extents
    launch_shape(n_rec, threads)       -> (grid, block)

THE CHAIN (`felizjueves.hessian`'s package docstring):

    d2 E_J / dx dy = SUM_ab D_ab SUM_t (dx dy E^{ab})_t J2_t       bra-bra
                   + SUM_ab D_ab SUM_t (dx E^{ab})_t J1^{(C,l)}_t  bra-ket

`x` on the bra's centres `A`, `B`; `y = (C, l)` any atom in the bra-ket
term. Both `J`s are the ordinary d0 `[p|q]` kernel, nothing in it new:

- `Jp2[row, t] = J2_t`, the J chain's `J_p` of the bra record two Hermite
  orders up, `|t| <= la + lb + 2`: the class `hjab_pq2_spec(la, lb, lc,
  ld) = (la + 2, lb | lc, ld)` over the J chain's kets (`DQ`, unchanged);
- `Jp1[C, l, row, t] = J1^{(C,l)}_t`, one order up, `|t| <= la + lb + 1`:
  the class `hjab_pq1_spec(la, lb, lc, ld) = (la + 1, lb | lc + 1, ld)`
  run once per atom `C` and axis `l` over the kets that touch `C`, each
  carrying its derivative density of `C` (`kernel_hdq`'s `D1`, gathered by
  the host). A row with no ket on `C` reads zeros, which the caller
  stores.

Per bra record (one thread), with `kernel_g1e.weights` at `M = D` and the
per-axis tables at `(la + 2, lb + 2)`:

    HXY_{kl,t} = SUM_r w_r (dX_k dY_l E)^{ab}_t     |t| <= L + 2
                 XY = AA, BB (k <= l; symmetric), AB (all k, l)
    HX_{k,t}   = SUM_r w_r (dX_k E)^{ab}_t          |t| <= L + 1
                 (`kernel_g1e.hermite_densities`, shared)

    H[aa, k, aa, l] += s SUM_t HAA_{kl,t} Jp2[row, t]    (and BB on ab)
    H[aa, k, ab, l] += s SUM_t HAB_{kl,t} Jp2[row, t],  H[ab, l, aa, k] too
    for C in atoms (serially, as `gv`'s nuclei):
      H[aa, k, C, l] += s SUM_t HA_{k,t} Jp1[C, l, row, t]   (and B)

On one axis a twice-differentiated factor is the rule applied to the
pooled first derivatives (`kernel_g1e.derivative_reader`):

    dA dA E^{ij}_t = 2 alpha dA E^{i+1,j}_t - i dA E^{i-1,j}_t
    dA dB E^{ij}_t = 2 beta  dA E^{i,j+1}_t - j dA E^{i,j-1}_t
    dB dB E^{ij}_t = 2 beta  dB E^{i,j+1}_t - j dB E^{i,j-1}_t

the `- n` term absent where `n = 0` or its source has no entry `t`; on two
different axes it is a product of two first derivatives. 36 `AtomicAdd`s
per record for the bra-bra term, 18 per atom for the bra-ket term. The
result is the whole skeleton `d2 E_J` -- no half, no factor 2.

THE PARAMETER BLOCK, `kernel_gjab`'s with the two `J`s and the Hessian:

    params  n_rec n_ao n_atom n_rows (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s  ea_bra  eb_ket
            off_a  off_b  row  atom_a  atom_b  [n_rec]  (I32)
            Jp2[n_rows, n_p2]               J_p at order L + 2
            Jp1[n_atom, 3, n_rows, n_p1]    J_p of the derivative kets
            M[n_ao, n_ao]                   the density D
            H[n_atom, 3, n_atom, 3]         (OUT, read and AtomicAdd-ed)

`H` is ZEROED BY THE CALLER; every record only adds.

ROUND-OFF, NOT BITWISE, against `hessian.host.coulomb_hessian`; the bound
is in test/test_kernel_hessian.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor `hessian.host`
nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import (
    REC,
    Axis,
    Locals,
    axis_reader,
    derivative_reader,
    hermite_densities,
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
    For,
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
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOM",
    "KINDS",
    "ZEROED_H",
    "build_hjab_kernel",
    "hjab_name",
    "hjab_pq1_spec",
    "hjab_pq2_spec",
    "hjab_spec",
    "launch_shape",
    "n_p1",
    "n_p2",
    "second_reader",
]

ATOM = "c"  # the atom index of the bra-ket term, serial inside the record

# The second-derivative kinds: both on A, one on each, both on B.
KINDS = ("AA", "AB", "BB")

ZEROED_H = (
    "PRECONDITION: H is zeroed by the caller before the first Hessian "
    "launch of a build; this kernel only adds")

_ATOMIC = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "rows of H")


def hjab_spec(la: int, lb: int) -> Spec:
    """The bra pair class `(la, lb)` at derivative order 2, `Target.JAB`."""
    return Spec(Operator.COULOMB, 2, (la, lb), Target.JAB)


def hjab_pq2_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d0 `[p|q]` class whose `J_p` the bra-bra term of bra class
    `(la, lb)` against ket class `(lc, ld)` reads: the bra raised by two."""
    return Spec(Operator.COULOMB, 0, (la + 2, lb, lc, ld), Target.J)


def hjab_pq1_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d0 `[p|q]` class whose `J_p` the bra-ket term reads: the bra
    raised by one and the ket raised by one, over `kernel_hdq`'s densities."""
    return Spec(Operator.COULOMB, 0, (la + 1, lb, lc + 1, ld), Target.J)


def n_p2(la: int, lb: int) -> int:
    """`Jp2`'s second extent: the Hermite functions of order `L + 2`."""
    return len(hermite_targets(la + lb + 2))


def n_p1(la: int, lb: int) -> int:
    """`Jp1`'s last extent: the Hermite functions of order `L + 1`."""
    return len(hermite_targets(la + lb + 1))


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hjab_spec(*cls)
    if spec.target is not Target.JAB or spec.derivative_order != 2:
        raise ValueError(f"{spec.key()} is not a d2 Target.JAB key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Hessian kernels take no levers")
    la, lb = spec.am
    return la, lb


def hjab_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_hjab_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"hjab_{la}{lb}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _fold(terms: list[Expr]) -> Expr:
    if not terms:
        return Lit(0.0)
    out = terms[0]
    for t in terms[1:]:
        out = _c(Fn.ADD, out, t)
    return out


def second_reader(dA: Axis, dB: Axis, pool: Locals, kind: str) -> Axis:
    """`dX dY E^{ij}_t` on one axis `d` (`kind` "AA", "AB" or "BB"), for
    `t <= i + j + 2`, as pooled locals `d{kind}{x}_{i}_{j}_{t}` built on
    the first derivatives (module docstring); the `- n` term is ABSENT
    where `n = 0` or its source has no entry `t`."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {KINDS}")
    inner = dA if kind[0] == "A" else dB
    two = Var("ea2" if kind[1] == "A" else "eb2")

    def read(d: int, i: int, j: int, t: int) -> Expr:
        def make() -> Expr:
            up = (i + 1, j) if kind[1] == "A" else (i, j + 1)
            out: Expr = _mul(two, inner(d, *up, t))
            n = i if kind[1] == "A" else j
            lo = (i - 1, j) if kind[1] == "A" else (i, j - 1)
            if n and t <= lo[0] + lo[1] + 1:
                out = _c(Fn.SUB, out, _mul(Lit(float(n)), inner(d, *lo, t)))
            return out
        return pool.get(f"d{kind}{'xyz'[d]}_{i}_{j}_{t}", make)
    return read


def _pairs(kind: str) -> list[tuple[int, int]]:
    """The `(k, l)` of a kind that are formed: `k <= l` where the kind is
    symmetric in them."""
    return [(k, l) for k in range(3) for l in range(3)  # noqa: E741
            if kind == "AB" or k <= l]


def _second_densities(la: int, lb: int, E: Axis, first: dict[str, Axis],
                      second: dict[str, Axis]) -> tuple[list[Stmt], dict]:
    """`HXY{k}{l}_{n}` over `hermite_targets(L + 2)`, each `SUM_r w_r
    (...)` over the rows that reach `t`. Returns the statements and
    `{(kind, k, l, n): name}`; an entry no row reaches is absent."""
    rows = ef.block_rows(la, lb)
    out: list[Stmt] = []
    names: dict[tuple[str, int, int, int], str] = {}
    for kind in KINDS:
        for k, l in _pairs(kind):  # noqa: E741
            for n, t in enumerate(hermite_targets(la + lb + 2)):
                terms: list[Expr] = []
                for r, (a, b) in enumerate(rows):
                    top = [a[d] + b[d] for d in range(3)]
                    top[k] += 1
                    top[l] += 1
                    if any(t[d] > top[d] for d in range(3)):
                        continue
                    f: list[Expr] = []
                    for d in range(3):
                        if d == k == l:
                            f.append(second[kind](d, a[d], b[d], t[d]))
                        elif d == k:
                            f.append(first[kind[0]](d, a[d], b[d], t[d]))
                        elif d == l:
                            f.append(first[kind[1]](d, a[d], b[d], t[d]))
                        else:
                            f.append(E(d, a[d], b[d], t[d]))
                    terms.append(_mul(Var(f"w{r}"),
                                      _mul(_mul(f[0], f[1]), f[2])))
                if terms:
                    nm = f"H{kind}{k}{l}_{n}"
                    names[kind, k, l, n] = nm
                    out.append(Decl(nm, Scalar.F64, init=_fold(terms)))
    return out, names


def _bra_bra(la: int, lb: int, names: dict) -> list[Stmt]:
    """The bra-bra values `v{kind}{k}{l} = s SUM_t H.. Jp2[row, t]` and
    their 36 atomics."""
    out: list[Stmt] = [
        Note(f"the bra-bra term: J2_t = Jp2[row, t] at Hermite order "
             f"{la + lb + 2}"),
    ]
    for kind in KINDS:
        for k, l in _pairs(kind):  # noqa: E741
            terms = [_mul(Var(names[kind, k, l, n]),
                          Index("Jp2", (Var("jr"), Lit(n))))
                     for n in range(n_p2(la, lb)) if (kind, k, l, n) in names]
            out.append(Decl(f"v{kind}{k}{l}", Scalar.F64,
                            init=_mul(Var("sk"), _fold(terms))))
    out.append(Note(_ATOMIC))

    def h(x: str, k: int, y: str, l: int) -> Index:  # noqa: E741
        return Index("H", (Var(x), Lit(k), Var(y), Lit(l)))

    for kind, atom in (("AA", "aa"), ("BB", "ab")):
        for k in range(3):
            for l in range(3):  # noqa: E741
                lo, hi = min(k, l), max(k, l)
                out.append(AtomicAdd(h(atom, k, atom, l),
                                     Var(f"v{kind}{lo}{hi}")))
    for k in range(3):
        for l in range(3):  # noqa: E741
            out.append(AtomicAdd(h("aa", k, "ab", l), Var(f"vAB{k}{l}")))
            out.append(AtomicAdd(h("ab", l, "aa", k), Var(f"vAB{k}{l}")))
    return out


def _bra_ket(la: int, lb: int, names: dict) -> list[Stmt]:
    """The atom loop of the bra-ket term (module docstring)."""
    atom = Var(ATOM)
    body: list[Stmt] = []
    for l in range(3):  # noqa: E741
        for side, at in (("A", "aa"), ("B", "ab")):
            for k in range(3):
                terms = [_mul(Var(names[f"{side}{k}", n]),
                              Index("Jp1", (atom, Lit(l), Var("jr"), Lit(n))))
                         for n in range(n_p1(la, lb))
                         if (f"{side}{k}", n) in names]
                body.append(AtomicAdd(
                    Index("H", (Var(at), Lit(k), atom, Lit(l))),
                    _mul(Var("sk"), _fold(terms))))
    return [
        Note(f"the bra-ket term: J1^(C,l)_t = Jp1[C, l, row, t] at Hermite "
             f"order {la + lb + 1}, every atom C serially"),
        For(ATOM, Lit(0), Var("n_atom"), Lit(1), tuple(body)),
    ]


def build_hjab_kernel(cls: Spec | tuple[int, int], *,
                      name: str | None = None) -> Kernel:
    """The Coulomb-energy Hessian kernel of one bra pair class.

    `cls` is a d2 `Target.JAB` `Spec` or `(la, lb)`. See the module
    docstring.
    """
    la, lb = _class(cls)
    head = record_head(la + 2, lb + 2, interleaved=True)
    head.append(Decl("jr", Scalar.I32, init=Index("row", (Var(REC),))))
    E = axis_reader(la + 2, lb + 2)
    pool = Locals()
    first = {"A": derivative_reader(E, pool, "A"),
             "B": derivative_reader(E, pool, "B")}
    second = {kind: second_reader(first["A"], first["B"], pool, kind)
              for kind in KINDS}
    d2, names2 = _second_densities(la, lb, E, first, second)
    d1, names1 = hermite_densities(la, lb, E, first["A"], first["B"],
                                   with_h0=False)
    per_rec = (*head, *weights(la, lb),
               Note("the derivative factors, each declared where first "
                    "read"),
               *pool.decls, *d2, *d1, *_bra_bra(la, lb, names2),
               *_bra_ket(la, lb, names1))
    body: tuple[Stmt, ...] = (
        Note(f"Hessian of E_J, bra class (la, lb) = ({la}, {lb}): "
             "H[A, k, B, l] += d2 E_J / dR_{A,k} dR_{B,l} "
             "(felizjueves.hessian)"),
        Note(ZEROED_H),
        Note("launch: one thread per bra primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), per_rec),
    )
    nr, nao, nat, nrows = (Var("n_rec"), Var("n_ao"), Var("n_atom"),
                           Var("n_rows"))

    def f64(nm: str, shape: tuple[Expr, ...]) -> Param:
        return Param(nm, Scalar.F64, array=True, shape=shape)

    def i32(nm: str) -> Param:
        return Param(nm, Scalar.I32, array=True, shape=(nr,))

    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        Param("n_rows", Scalar.I32),
        f64("zeta_ab", (nr,)),
        f64("XPA", (nr, Lit(3))),
        f64("XPB", (nr, Lit(3))),
        f64("s", (nr,)),
        f64("ea_bra", (nr,)),
        f64("eb_ket", (nr,)),
        i32("off_a"),
        i32("off_b"),
        i32("row"),
        i32("atom_a"),
        i32("atom_b"),
        f64("Jp2", (nrows, Lit(n_p2(la, lb)))),
        f64("Jp1", (nat, Lit(3), nrows, Lit(n_p1(la, lb)))),
        f64("M", (nao, nao)),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3), nat, Lit(3))),
    )
    return Kernel(name or hjab_name((la, lb)), params, body)
