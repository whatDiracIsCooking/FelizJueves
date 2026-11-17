"""The GEMV over a packed symmetric matrix: `J_pp = ERI_pp Dp`, and
`K_pp = X_pp Dp` over `kernel_kperm`'s exchange-ordered `X_pp`.

    build_gemv_kernel(name="eri_jgemv") -> Kernel
    launch_shape(n_pp, threads)         -> (grid, block)

One thread per packed pair `i`, a left fold from `+0.0` over `k` in order:

    Jp[i] = SUM_k eri[k, i] * Dp[k]

READ BY COLUMN, ON PURPOSE. `kernel_eri` leaves `eri` EXACTLY symmetric (each
element and its mirror get one store of one value), so column `i` is row `i`,
and reading the column makes a warp's 32 loads of one `k` 32 consecutive
doubles -- coalesced -- where reading the row would stride them `n_pp` apart.
So the product is `ERI_pp Dp` bit for bit with the row form's fold order.

`Dp` and `Jp` are the packed density and Coulomb vectors of
`full_eri.packed` (`pack_density`, `unpack`): the packing is host-side.

    params  n_pp (I32)  eri[n_pp, n_pp]  Dp[n_pp]  Jp[n_pp] (OUT)

THE SAME KERNEL IS `K`. `kernel_kperm`'s `X_pp` is exactly symmetric too,
and `K_pp = X_pp Dp` with the same `Dp` (`full_eri.packed`), so passing `X`
as `eri` and reading `Jp` as `Kp` is the exchange GEMV, unchanged. The
parameter names stay the Coulomb ones so its emitted text does not move.

A builder: vendor-neutral IR; no class, so no `Spec`.
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

__all__ = ["OUT", "build_gemv_kernel", "launch_shape"]

OUT = "Jp"


def launch_shape(n_pp: int, threads: int = 128
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per packed pair."""
    if type(n_pp) is not int or n_pp < 1:
        raise ValueError(f"n_pp = {n_pp!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pp // threads), (threads, 1)


def build_gemv_kernel(name: str = "eri_jgemv") -> Kernel:
    """`Jp = eri Dp`, one thread per row, read by column (module docstring)."""
    i, k = Var("i"), Var("k")
    npp = Var("n_pp")
    body = (
        Note("J_pp = ERI_pp Dp (eri/full_eri/packed.py): one thread per "
             "packed pair, the symmetric matrix read by column (coalesced)"),
        Decl("i", Scalar.I32, init=Call(Fn.ADD, (Call(Fn.THREAD_X, ()), Call(
            Fn.MUL, (Call(Fn.BLOCK_X, ()), Call(Fn.BLOCK_DIM_X, ())))))),
        If(Call(Fn.LT, (i, npp)), (
            Decl("acc", Scalar.F64, init=Lit(0.0)),
            For("k", Lit(0), npp, Lit(1), (
                Assign(Var("acc"), Call(Fn.ADD, (Var("acc"), Call(Fn.MUL, (
                    Index("eri", (k, i)), Index("Dp", (k,))))))),)),
            Assign(Index(OUT, (i,)), Var("acc")),
        )),
    )
    params = (
        Param("n_pp", Scalar.I32),
        Param("eri", Scalar.F64, array=True, shape=(npp, npp)),
        Param("Dp", Scalar.F64, array=True, shape=(npp,)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT, shape=(npp,)),
    )
    return Kernel(name, params, body)
