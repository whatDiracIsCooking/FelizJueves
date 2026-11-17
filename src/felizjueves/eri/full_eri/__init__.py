"""The full Cartesian AO ERI tensor, packed by AO pair; `J` and `K` as GEMVs.

    full_eri.packed       host: the staging, the packed reference, pack / unpack
    full_eri.kernel_eri   `Target.ERI`: one thread per shell-pair quartet
    full_eri.kernel_kperm `X_pp`, the exchange-ordered matrix, from `ERI_pp`
    full_eri.kernel_gemv  `J_pp = ERI_pp Dp` and `K_pp = X_pp Dp`, one thread
                          per packed pair

A small-system path (`n_ao <= N_AO_MAX = 150`): the packed tensor is
`n_pp^2` doubles, `n_pp = n_ao (n_ao + 1) / 2`, about 1 GB at the cap,
and `X_pp` as much again.
"""
