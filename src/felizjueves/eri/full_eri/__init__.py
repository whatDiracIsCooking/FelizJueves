"""The full Cartesian AO ERI tensor, packed by AO pair; `J` and `K` as GEMVs.

    full_eri.packed       host: the staging, the packed reference, pack / unpack
    full_eri.kernel_eri   `Target.ERI`: one thread per shell-pair quartet,
                          or, from f, the low-stack form (`n_a` threads
                          per quartet, no accumulator block)
    full_eri.kernel_kperm `X_pp`, the exchange-ordered matrix, from `ERI_pp`
    full_eri.kernel_gemv  `J_pp = ERI_pp Dp` and `K_pp = X_pp Dp`, one thread
                          per packed pair
    full_eri.dense        host: the dense layouts' staging, row range and
                          reference
    full_eri.kernel_dense `Target.ERI_DENSE`: a row block of the dense J-layout
                          `M_J[(m,n),(l,s)] = (mn|ls)` or K-layout
                          `M_K[(m,n),(l,s)] = (ml|ns)`, one thread per ORDERED
                          shell quartet (from f, the low-stack form), no
                          symmetry

A small-system path (`n_ao <= N_AO_MAX = 150`): the packed tensor is
`n_pp^2` doubles, `n_pp = n_ao (n_ao + 1) / 2`, about 1 GB at the cap,
and `X_pp` as much again.

The dense layouts are the multi-GPU path: no symmetry and no packing, so
`J = M_J vec(D)` and `K = M_K vec(D)` for any `D`, and each launch writes
only a contiguous row block (the shells of `m`), its int32 bound per launch;
there is no `N_AO_MAX` on it.
"""
