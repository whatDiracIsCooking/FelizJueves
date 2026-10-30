"""The Rys 2D vertical recurrence as IR: per root, per axis, then the HGP
boundary `[e0|f0]^(0)` as a product of three 2D integrals.

    root_loop(cls, boundary, acc, *, n_roots, ket_centred=False) -> (Stmt, ...)
    scalar_needs(cls, *, ket_centred=False) -> frozenset[str]   what it reads
    quartet_rule(cls, scalar) -> (Stmt, ...)  those scalars, T, S, the rule

THE FORM (#468, settled against libcint and a 50-digit route, every sign
flipped alone and caught; `experimental/rys/README.md`). With `x` a node,
`p`, `q` the two exponent sums, `W - P = -(q/(p+q)) PQ`, `W - Q = (p/(p+q))
PQ` (`kernel_f`'s derived signs, its `WP_*` / `WQ_*`), per axis `d`:

    B00   = x / (2 (p + q))                        = x inv2pq
    B10   = (1 - x q / (p + q)) / (2p)             = inv2p (1 - x q_ppq)
    B01   = (1 - x p / (p + q)) / (2q)             = inv2q (1 - x p_ppq)
    C00_d = (P - A)_d + x (W - P)_d                = PA_d + x WP_d
    D00_d = (Q - C)_d + x (W - Q)_d                = QC_d + x WQ_d

    I(0, 0)     = 1    (the z axis: the root's weight, see below)
    I(i + 1, 0) = C00 I(i, 0) + i B10 I(i - 1, 0)
    I(i, k + 1) = D00 I(i, k) + k B01 I(i, k - 1) + i B00 I(i - 1, k)

for `i <= la + lb`, `k <= lc + ld`. #468's transfer steps (its `I(i, j+1)`
from `I(i+1, j)`) are NOT here: the HGP boundary is exactly the VRR's
`(e 0 | f 0)` with `la <= |e| <= la + lb`, `lc <= |f| <= lc + ld`, and the
transfers run on CONTRACTED values as `hgp.hrr`'s eq 28, which the fused
kernel already holds. So one boundary value of one primitive quartet is

    [e0|f0]^(0) = S SUM_n w_n I_x(e_x, f_x) I_y(e_y, f_y) I_z(e_z, f_z),

`S = U_P U_Q (2/sqrt(pi)) theta` the HGP fundamental's own scale (`pair`
and `theta`), `(x_n, w_n)` the `N = L // 2 + 1` point rule at `T = rho
|PQ|^2`. The weight `S w_n` is folded into the z axis's seed, so a boundary
value is two multiplies per root, and the root loop accumulates straight
into the contracted accumulators: the `K_AB K_CD` sum and the root sum are
one sum, as eq 27 is for HGP.

The root loop is a real `For` over the `N` nodes, its body the 2D tables
(Decl'd fresh per root) and one `+=` per boundary value; the 2D entries are
plain products and sums, not `ir.Weighted` (their coefficients depend on the
node, which no `Factor` names). Only the coefficients a class reads are
formed: `(ss|ss)` reads none, and its boundary is `S w_0` alone.

A SINGLE-CENTRE KET (`ket_centred=True`): when the ket "pair" is one
Gaussian and an exponent-0 s partner on its own centre -- a DF aux shell,
`eri/density_fit/kernel_rys.py`'s `(mn|P) = (la lb|lP 0)` -- `Q = C`, so
`QC_d = 0` and `D00_d = x (W - Q)_d`: the zero term is not spelled and
`QC_*` is not read (`kernel_hgp._without`'s rule, here in the 2D VRR).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from felizjueves.hgp.recurrence import _TWO_OVER_SQRT_PI, PAIR, PQ_PARAMS, THETA
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    Index,
    Lit,
    Note,
    Rys,
    Scalar,
    Space,
    Stmt,
    Var,
)
from felizjueves.rys.roots import n_roots as _n_roots

__all__ = ["NODES", "ROOT", "WEIGHTS", "WEIGHT_SCALE", "quartet_rule",
           "root_loop", "scalar_needs"]

NODES, WEIGHTS = "rys_x", "rys_w"  # the rule's two local arrays
ROOT = "r"  # the root loop's variable
WEIGHT_SCALE = "S"  # the HGP fundamental's scale, U_P U_Q (2/sqrt(pi)) theta
_AXES = "xyz"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(a: Expr | None, b: Expr | None) -> Expr | None:
    """`a * b`, where None is the literal 1 (the x and y seeds)."""
    if a is None:
        return b
    if b is None:
        return a
    return _c(Fn.MUL, a, b)


def _add(a: Expr | None, b: Expr | None) -> Expr | None:
    if a is None:
        return b
    if b is None:
        return a
    return _c(Fn.ADD, a, b)


def _spans(cls: AmClass) -> tuple[int, int]:
    la, lb, lc, ld = cls
    return la + lb, lc + ld


def scalar_needs(cls: AmClass, *, ket_centred: bool = False
                 ) -> frozenset[str]:
    """The per-quartet scalars `root_loop` reads (besides `pair`, `theta`
    and `T`, which every class reads): `PA_*`, `WP_*` and `q_ppq` for a bra
    VRR, `QC_*`, `WQ_*` and `p_ppq` for a ket one, `inv2p` / `inv2q` from
    the second step, `inv2pq` when both sides step; no `QC_*` when
    `ket_centred` (module docstring, "A SINGLE-CENTRE KET")."""
    lab, lcd = _spans(cls)
    need: set[str] = set()
    if lab:
        need |= {f"PA_{a}" for a in _AXES} | {f"WP_{a}" for a in _AXES}
    if lcd:
        need |= {f"WQ_{a}" for a in _AXES}
        if not ket_centred:
            need |= {f"QC_{a}" for a in _AXES}
    if lab > 1:
        need |= {"inv2p", "q_ppq"}
    if lcd > 1:
        need |= {"inv2q", "p_ppq"}
    if lab and lcd:
        need.add("inv2pq")
    return frozenset(need)


def _name(axis: str, i: int, k: int) -> str:
    return f"I{axis}_{i}_{k}"


def root_loop(cls: AmClass, boundary: Sequence[tuple[Cart, Cart]],
              acc: Callable[[Cart, Cart], str], *, n_roots: int,
              ket_centred: bool = False) -> tuple[Stmt, ...]:
    """The `Rys` rule's root loop for one primitive quartet (module
    docstring): reads the arrays `NODES` / `WEIGHTS` (extent `n_roots`), the
    scale `WEIGHT_SCALE` and `scalar_needs(cls)`, and adds each boundary
    value `(e, f)` into the accumulator `acc(e, f)`. `ket_centred` drops
    `D00`'s zero `QC` term (module docstring, "A SINGLE-CENTRE KET")."""
    lab, lcd = _spans(cls)
    x = Var("rys_xr")
    body: list[Stmt] = [
        *((Decl("rys_xr", Scalar.F64, init=Index(NODES, (Var(ROOT),))),)
          if lab or lcd else ()),
        Decl("rys_wr", Scalar.F64, init=_c(
            Fn.MUL, Var(WEIGHT_SCALE), Index(WEIGHTS, (Var(ROOT),)))),
    ]
    if lab and lcd:
        body.append(Decl("B00", Scalar.F64,
                         init=_c(Fn.MUL, x, Var("inv2pq"))))
    if lab > 1:
        body.append(Decl("B10", Scalar.F64, init=_c(Fn.MUL, Var("inv2p"), _c(
            Fn.SUB, Lit(1.0), _c(Fn.MUL, x, Var("q_ppq"))))))
    if lcd > 1:
        body.append(Decl("B01", Scalar.F64, init=_c(Fn.MUL, Var("inv2q"), _c(
            Fn.SUB, Lit(1.0), _c(Fn.MUL, x, Var("p_ppq"))))))
    tables: list[dict[tuple[int, int], Expr | None]] = []
    for d, ax in enumerate(_AXES):
        if lab:
            body.append(Decl(f"c00_{ax}", Scalar.F64, init=_c(
                Fn.ADD, Var(f"PA_{ax}"), _c(Fn.MUL, x, Var(f"WP_{ax}")))))
        if lcd:
            xwq = _c(Fn.MUL, x, Var(f"WQ_{ax}"))
            body.append(Decl(f"d00_{ax}", Scalar.F64, init=xwq if ket_centred
                             else _c(Fn.ADD, Var(f"QC_{ax}"), xwq)))
        g: dict[tuple[int, int], Expr | None] = {
            (0, 0): Var("rys_wr") if d == 2 else None}

        def put(i: int, k: int, e: Expr | None, ax: str = ax,
                g: dict = g) -> None:
            n = _name(ax, i, k)
            body.append(Decl(n, Scalar.F64, init=e))
            g[i, k] = Var(n)

        c00, d00 = Var(f"c00_{ax}"), Var(f"d00_{ax}")
        for i in range(lab):
            e = _mul(c00, g[i, 0])
            if i:
                e = _add(e, _mul(_c(Fn.MUL, Lit(float(i)), Var("B10")),
                                 g[i - 1, 0]))
            put(i + 1, 0, e)
        for k in range(lcd):
            for i in range(lab + 1):
                e = _mul(d00, g[i, k])
                if k:
                    e = _add(e, _mul(_c(Fn.MUL, Lit(float(k)), Var("B01")),
                                     g[i, k - 1]))
                if i:
                    e = _add(e, _mul(_c(Fn.MUL, Lit(float(i)), Var("B00")),
                                     g[i - 1, k]))
                put(i, k + 1, e)
        tables.append(g)
    body.append(Note("the boundary: [e0|f0]^(0) += I_x I_y I_z, the weight "
                     "S w_r in I_z's seed"))
    for e, f in boundary:
        prod = _mul(_mul(tables[0][e[0], f[0]], tables[1][e[1], f[1]]),
                    tables[2][e[2], f[2]])
        a = Var(acc(e, f))
        body.append(Assign(a, _c(Fn.ADD, a, prod)))
    return (
        Note(f"the Rys root loop: N = {n_roots} nodes, per axis the 2D VRR "
             f"I(i, k), i <= {lab}, k <= {lcd} (rys/recurrence.py)"),
        For(ROOT, Lit(0), Lit(n_roots), Lit(1), tuple(body)),
    )


def quartet_rule(cls: AmClass, scalar: Callable[[str, Expr], Decl]
                 ) -> tuple[Stmt, ...]:
    """One primitive quartet's scalars and its Rys rule, after the two
    records' reads (`Pb0..2`, `p`, `Ub` and `Qk0..2`, `q`, `Uq`, and the
    `PA_*` / `QC_*` that `scalar_needs` names): `PQ`, `p + q`, `rho`,
    `theta`, `W - P = -(q/(p+q)) PQ` and `W - Q = (p/(p+q)) PQ`
    (`kernel_f`'s derived signs), the `1/(2 x)` the class reads, `pair =
    U_P U_Q`, `T = rho |PQ|^2`, `S = pair (2/sqrt(pi)) theta`, then the
    `N`-point rule into `NODES` / `WEIGHTS` -- everything `root_loop`
    reads. `scalar(name, init)` declares each floating local (a builder's
    own typing hook); `T`, `S` and the two arrays are plain fp64."""
    need = scalar_needs(cls)
    n = _n_roots(sum(cls))
    pq = [Var(v) for v in PQ_PARAMS]
    theta = Var(THETA)
    out: list[Stmt] = [
        Note("quartet geometry: W - P and W - Q with kernel_f's derived "
             "signs, and the 2D recurrence's scalars"),
        *(scalar(name, _c(Fn.SUB, Var(f"Pb{d}"), Var(f"Qk{d}")))
          for d, name in enumerate(PQ_PARAMS)),
        scalar("ppq", _c(Fn.ADD, Var("p"), Var("q"))),
        scalar("rho", _c(Fn.DIV, _c(Fn.MUL, Var("p"), Var("q")), Var("ppq"))),
        scalar(THETA, _c(Fn.SQRT, Var("rho"))),
    ]
    wp = any(f"WP_{a}" in need for a in _AXES)
    wq = any(f"WQ_{a}" in need for a in _AXES)
    if wp or "q_ppq" in need:
        out.append(scalar("q_ppq", _c(Fn.DIV, Var("q"), Var("ppq"))))
    if wq or "p_ppq" in need:
        out.append(scalar("p_ppq", _c(Fn.DIV, Var("p"), Var("ppq"))))
    for d, a in enumerate(_AXES):
        if f"WP_{a}" in need:
            out.append(scalar(f"WP_{a}", _c(Fn.SUB, Lit(0.0), _c(
                Fn.MUL, Var("q_ppq"), pq[d]))))
        if f"WQ_{a}" in need:
            out.append(scalar(f"WQ_{a}", _c(Fn.MUL, Var("p_ppq"), pq[d])))
    for name, x in (("inv2p", "p"), ("inv2q", "q"), ("inv2pq", "ppq")):
        if name in need:
            out.append(scalar(name, _c(Fn.DIV, Lit(0.5), Var(x))))
    out += [
        scalar(PAIR, _c(Fn.MUL, Var("Ub"), Var("Uq"))),
        Note(f"the Rys rule: N = {n} nodes and weights of (1/2) x^(-1/2) "
             "exp(-T x) dx on [0, 1] at T = rho |PQ|^2 (rys/roots.py); "
             "S = U_P U_Q (2/sqrt(pi)) theta, the HGP fundamental's scale"),
        Decl("T", Scalar.F64, init=_c(Fn.MUL, _c(Fn.MUL, theta, theta), _c(
            Fn.ADD, _c(Fn.ADD, _c(Fn.MUL, pq[0], pq[0]),
                       _c(Fn.MUL, pq[1], pq[1])), _c(Fn.MUL, pq[2], pq[2])))),
        Decl(WEIGHT_SCALE, Scalar.F64, init=_c(Fn.MUL, Var(PAIR), _c(
            Fn.MUL, Lit(_TWO_OVER_SQRT_PI), theta))),
        Decl(NODES, Scalar.F64, (Lit(n),), Space.LOCAL),
        Decl(WEIGHTS, Scalar.F64, (Lit(n),), Space.LOCAL),
        Rys(NODES, WEIGHTS, n, Var("T")),
    ]
    return tuple(out)
