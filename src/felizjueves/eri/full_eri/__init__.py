"""The full Cartesian AO ERI tensor, packed by AO pair, and `J` as one GEMV.

    full_eri.eri      host: the staging, the packed reference, pack / unpack
    full_eri.kernel_eri   `Target.ERI`: one thread per shell-pair quartet
    full_eri.kernel_gemv  `J_pp = ERI_pp D_pp`, one thread per packed pair

A small-system path (`n_ao <= N_AO_MAX = 150`): the packed tensor is
`n_pp^2` doubles, `n_pp = n_ao (n_ao + 1) / 2`, about 1 GB at the cap.
"""
