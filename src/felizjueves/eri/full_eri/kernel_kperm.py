"""The exchange-ordered packed matrix from the packed ERI: `K` as a GEMV.

    build_kperm_kernel(name="eri_kperm") -> Kernel
    launch_shape(n_pp, threads)          -> (grid, block)

THE MATRIX. For a symmetric density, `K_mn = SUM_ls (ml|ns) D_ls` is, over
the AO-pair packing of `full_eri.packed`,

    K_pp = X_pp Dp,   X_pp[pidx(l, s), pidx(m, n)]
                        = 0.5 ((ml|ns) + (ms|nl)),

with `Dp = pack_density(D)` the very vector the Coulomb GEMV takes: an
off-diagonal `(l, s)` carries `D_ls + D_sl = 2 D_ls`, and the two orders of
it contribute `(ml|ns) D_ls + (ms|nl) D_sl`; on the diagonal the two terms
are one and the half restores it. `X_pp` is symmetric in `m <-> n`, in
`l <-> s` and under `(mn) <-> (ls)`, so it is one `n_pp x n_pp` symmetric
matrix like `ERI_pp`, and `kernel_gemv`'s `eri_jgemv` -- reading by column --
is `K` over it unchanged. That is the point: `X_pp` is built ONCE per
geometry from `ERI_pp`, and every SCF iteration's `J` and `K` are then one
GEMV each. The price is a second `n_pp^2` matrix: about 2 GB of doubles for
the pair at `N_AO_MAX = 150`.

EXACTLY SYMMETRIC, as the GEMV needs. `X[j, i]` and `X[i, j]` read the same
two elements of `ERI_pp` (one of them through its mirror, which `kernel_eri`
stores with one value), add them in the two orders -- commutative, so one
value -- and halve, which is exact. So `X == X.T` bit for bit, and `X` is
bitwise the host's `0.5 * (a + b)` (`full_eri.packed.exchange_packed`).

THE LAUNCH. One thread per column `i = pidx(m, n)`, a loop over rows
`j = pidx(l, s)` in order; a warp's 32 stores of one `j` are 32 consecutive
doubles (coalesced), its loads gathered. `pm[i]`, `pn[i]` are the AO pair
of packed index `i` (`full_eri.packed.pair_of`), `pidx` the packing.

    params  n_ao n_pp (I32)  pm pn [n_pp] (I32)  pidx[n_ao, n_ao] (I32)
            eri[n_pp, n_pp]  X[n_pp, n_pp] (OUT)

A builder: vendor-neutral IR; no class, so no `Spec`. A non-symmetric `D`
is out of scope (RHF): `X` folds `D_ls` and `D_sl` together.
"""

from __future__ import annotations

from felizjueves.ir import (
    Assign,
    Call,
    Decl,
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
    Var,
)

__all__ = ["OUT", "build_kperm_kernel", "launch_shape"]

OUT = "X"


def launch_shape(n_pp: int, threads: int = 128
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per packed column."""
    if type(n_pp) is not int or n_pp < 1:
        raise ValueError(f"n_pp = {n_pp!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pp // threads), (threads, 1)


def build_kperm_kernel(name: str = "eri_kperm") -> Kernel:
    """`X[j, i] = 0.5 (eri[pidx(m,l), pidx(n,s)] + eri[pidx(m,s),
    pidx(n,l)])`, `i = (m, n)`, `j = (l, s)` (module docstring)."""
    i, j = Var("i"), Var("j")
    npp = Var("n_pp")
    m, n, l_, s = Var("m"), Var("n"), Var("l"), Var("s")

    def pidx(a: Var, b: Var) -> Index:
        return Index("pidx", (a, b))

    body = (
        Note("X_pp[(ls), (mn)] = ((ml|ns) + (ms|nl)) / 2 "
             "(eri/full_eri/kernel_kperm.py): K_pp = X_pp Dp by eri_jgemv"),
        Decl("i", Scalar.I32, init=Call(Fn.ADD, (Call(Fn.THREAD_X, ()), Call(
            Fn.MUL, (Call(Fn.BLOCK_X, ()), Call(Fn.BLOCK_DIM_X, ())))))),
        If(Call(Fn.LT, (i, npp)), (
            Decl("m", Scalar.I32, init=Index("pm", (i,))),
            Decl("n", Scalar.I32, init=Index("pn", (i,))),
            For("j", Lit(0), npp, Lit(1), (
                Decl("l", Scalar.I32, init=Index("pm", (j,))),
                Decl("s", Scalar.I32, init=Index("pn", (j,))),
                Decl("a", Scalar.F64, init=Index("eri", (pidx(m, l_),
                                                         pidx(n, s)))),
                Decl("b", Scalar.F64, init=Index("eri", (pidx(m, s),
                                                         pidx(n, l_)))),
                Assign(Index(OUT, (j, i)), Call(Fn.MUL, (Lit(0.5), Call(
                    Fn.ADD, (Var("a"), Var("b")))))),
            )),
        )),
    )
    nao = Var("n_ao")
    params = (
        Param("n_ao", Scalar.I32),
        Param("n_pp", Scalar.I32),
        Param("pm", Scalar.I32, array=True, shape=(npp,)),
        Param("pn", Scalar.I32, array=True, shape=(npp,)),
        Param("pidx", Scalar.I32, array=True, shape=(nao, nao)),
        Param("eri", Scalar.F64, array=True, shape=(npp, npp)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(npp, npp)),
    )
    return Kernel(name, params, body)
