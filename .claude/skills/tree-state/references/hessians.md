# Nuclear Hessians (`derivative_order = 2`)

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**The first `derivative_order = 2` key: the Coulomb energy's analytic
nuclear Hessian by MMD, run on both cards through (dd)** (`docs/implementation/hessian.md`
is the one place with the detail; after `experimental/mmd/hessian`):
`spec.DERIVATIVE_ORDERS` is `(0, 1, 2)`, order 2 legal on
`spec.HESSIAN_TARGETS` (`JAB`, then `S`, `T`, `V`, the exchange Hessian
`K` and the DF `METRIC` and `ERI3`, MMD only; `_HESSIAN_REFUSAL` elsewhere) and
order 1 gaining `DQ`, every d0 and d1 key byte-identical. The skeleton
`H[A, k, B, l] = d2 E_J/dR dR` for a fixed `D` (no CPHF response), split
as bra-bra plus bra-ket (the half of `E_J` cancelling both twos):
`hessian/kernel_hjab.py` (`hjab_{la}{lb}`, one thread per bra record, the
twice-applied rule on `kernel_g1e`'s per-axis tables two indices up)
reads `Jp2`, the d0 `[p|q]` of `hjab_pq2_spec` = `(la + 2, lb | lc, ld)`,
and `Jp1[n_atom, 3, ...]`, the d0 `[p|q]` of `hjab_pq1_spec` = `(la + 1,
lb | lc + 1, ld)` run per atom and axis over the derivative kets that
`hessian/kernel_hdq.py` (`hdq_{lc}{ld}`, `D1[6, n_ket, n_q1]`) writes and
the host gathers. Host reference `hessian/host.py` (`coulomb_hessian`,
`absolute=True` its round-off scale), held to finite differences of
`gradient.host.coulomb_gradient` (~6e-9 of `max|H|`), `H = H^T` and
translational invariance (`test_hessian_host.py`); interpreted, `hdq` is
bitwise the host and `hjab` within 16 eps of the absolute twin for every
class through (dd), and the whole chain through the interpreted `[p|q]`
within 64 eps (`test_kernel_hessian.py`; L > 4 `slow`). Ladder `python -m
felizjueves.emit --hess-l [LMAX]` (1 bare, 0..2; `hjab` is ~1 MB of
unrolled text at (dd)), never in the bare default nor in
`KERNEL_GEN_CMD`; 12 goldens (`test_emit_hess.py`). **On both cards**
(`test_hess_j_device.py`, generator `test/hess_driver.py` with its OUT
`D1` for `hdq`, on the shared runner since #659, which ran on both cards on 2026-10-30): `hdq` and `hjab` for every ordered class (ss)–(dd) and
the whole H₂O/STO-3G `H` (`slow`), against the host with the host's `J`s,
19 passed per card; `hdq` 4.98e-16 of `max|D1|`, `hjab` 9.28e-17 of the
absolute twin, the same on both cards. The on-card chain through the
`[p|q]` kernel is interpreted only. No response term (the exchange
Hessian: below), no whole RHF Hessian, nothing measured.
**Beside it, the one-electron Hessians of `S`, `T` and `V` at d2,
run on both cards through (dd)** (`hessian.md`, "The one-electron Hessians"; the spike's
one-electron half): `HESSIAN_TARGETS` also holds `S`, `T`, `V` (MMD only).
A d2 kernel writes `H[A, k, B, l] += SUM_mn M_mn d2X_mn/dR_{A,k} dR_{B,l}`,
the rule applied twice on ONE pair (`dA dA f_i = 4a^2 f_{i+2} - 2a(2i+1) f_i
+ i(i-1) f_{i-2}`, `dA dB` the rule on each index) on `axis_tables_fragment`
two indices above the class (`T`'s ket four): `hessian/kernel_h1e.py`
(`d2s_`, `d2t_`, `d2v_{la}{lb}` -- not `hs_`, the HGP one-electron
ladder's names -- `kernel_g1e`'s record head, weights and parameter block
reused, the one OUT `H[n_atom, 3, n_atom, 3]`, 21 components per record as
an operator per axis (`axis_ops`), 36 `AtomicAdd`s; `d2v` with the Hermite
densities once per record and, per nucleus, `[t]^(0)` from
`kernel_v.v_plan((la, lb + 2))`, the centre-nucleus and nucleus-nucleus
terms as 45 more atomics). Host reference `hessian/host_1e.py`
(`overlap_hessian`, `kinetic_hessian`, `potential_hessian`, on
`gradient.host`'s sweep, `E_nuc`'s being `hessian.host`'s), by another
route (each 3-D Cartesian differentiated as a list of moved Cartesians,
the spike's), with an `absolute=True` twin: held to central finite
differences of `gradient.host` (~3e-9 to 8e-9 of `max|H|`, gate 1e-7;
`test_hessian_host_1e.py`, cc-pVDZ `slow`), translational invariance
(1e-10; `T`'s O 1s blocks cancel to 4.5e-12) and symmetry; interpreted,
`kernel_h1e` within 16 eps (measured <= 0.35) of the absolute twin for
every ordered class through (ff) on HF/cc-pVTZ (`test_kernel_h1e.py`,
L + 2 > 4 `slow`). Ladder `python -m felizjueves.emit --hess-one-e-l
[LMAX]` (2 bare, 0..3), never in the bare default nor in
`KERNEL_GEN_CMD`; 18 goldens per backend plus `d2v_33`
(`test_emit_hess_1e.py`). **From `la + lb = 4` `d2v` defaults to a
low-stack form** (`build_h1e_kernel(cls)` picks it when
`kernel_h1e.h1e_lowstack(cls)`: `V`'s Hermite order `la + lb + 2 >=
kernel_v.LOWSTACK_MIN_LSUM`; `lowstack=` forces either form, `S` / `T`
refuse `True`). Name, parameters, launch and `H` contract are the same.
It forms no Hermite density: per nucleus `[t]^(0)` goes into a local `R`,
and per block row each axis's operated factors `F{op}[d, t]` (bitwise
the unrolled pooled locals) and the 28 contractions are loops, so the
text is 115 KB at (dd) and 161 KB at (ff), against 1.08 MB and 4.9 MB.
Every default unit through (ff) is under the 512 KiB cap (asserted). By
clang 18's PTX depot the frame is 7.4 / 12.9 KB, mostly the shared
`(la + 2, lb + 2)` axis tables; the unrolled form holds 14 / 30 KB of
densities live across the nucleus loop. Interpreted, within 0.27 eps of
the absolute twin for every ordered class through (ff), and every class
is held in both forms. Below the rule the text is byte-identical; the
unrolled `d2v_22` is pinned by its old digest (`d2v_22.unrolled`) in
`test/golden/hessian_digests.json`. Neither form is measured.
**On both cards, in the unrolled form** (before the low-stack `d2v`) (`test_hess_device.py`, generator `test/hess_driver.py`, on the shared runner since #659, which ran on both cards on 2026-10-30):
H₂O/STO-3G and cc-pVDZ (`slow`), against the host, 6 passed per card;
worst 1.57e-16 (RTX 3080) and 2.30e-16 (RX 9060 XT) of the absolute twin,
`d2v` on cc-pVDZ. Nothing above (dd) on a card, nothing measured; the low-stack `d2v` has not run on a card (clang 18's CUDA front end passed `d2v_22` / `d2v_33`, a proxy).
**The exchange
energy's Hessian too, s and p only, interpreted only** (`hessian.md`, "The
exchange Hessian"): `Target.K` joins `spec.HESSIAN_TARGETS`, a d2 K key in
K's pairing (`coulomb/d2/(la,lc|lb,ld)/b1/k/literal`).
`hessian/kernel_hkx.py` (`hkx_{la}{lc}_{lb}{ld}`) is ONE kernel per K
class writing both terms (`D_ab D_cd` straddles bra and ket, so no
derivative density exists to precompute): `kernel_gk`'s mapping, one warp
per bra record, no screen, the plan of the d0 K class with the bra raised
by two (`hkx_pq_spec`), the bra derivatives as shifted rows composed by
the rule (`_derivative_terms`) and folded once each, the ket's as
`s_bd (dY_l E^{bd})_q` from per-axis tables; the ket's atom varies across
lanes, so the host groups each class's kets by atom pair
(`hessian.host.exchange_hessian_records`: `seg_start`, `seg_b`, `seg_d`)
and the kernel walks the segments serially, 36 warp reductions and
atomics each. Host reference `hessian.host.exchange_hessian`, held to
finite differences of `gradient.host.exchange_gradient` (5.7e-9 of
`max|H|`), `H = H^T` and invariance (`test_hessian_host.py`), and by hand
once to `experimental/mmd/hessian`'s `d2(ab|cd)` (1.2e-14); interpreted,
`hkx` within ~1 eps of the absolute twin for every s, p class
(`test_kernel_hkx.py`, L + 2 > 4 `slow`). Ladder `--hess-k-l [LMAX]` (1
bare, 0..1; `(pp|pp)` is 1.6 MB of text), never in the bare default nor
in `KERNEL_GEN_CMD`; 22 goldens plus 10 digests
(`test_emit_hess_k.py`, `test/golden/hkx_digests.json`). Not compiled by
nvcc or hipcc (clang 18's front end passed all 16 cuda units, a proxy),
no device test, not run on either card. No response term, no whole RHF
Hessian, nothing measured.
**The exchange derivative matrix `K^X` at d1, interpreted through (dd|dd)
and at f, compiled through (ff|ff), on no card** (#689; `hessian.md`, "The
exchange derivative matrix"): the exchange part of the CPHF right-hand
side's `F^x`, `K^X_ab = SUM_cd D_cd d(ac|bd)/dR_X` as an AO matrix. Two
d1-only targets (`spec.D1_ONLY_TARGETS`), `Target.KX1` (ip1, the
derivative on the matrix index, every atom in one launch) and
`Target.KXC` (the contracted index, one launch per atom with the bra
records whose `c` sits on it), keys `coulomb/d1/(la,lc|lb,ld)/b1/kx1/literal`
and `.../kxc/...`; one builder, `hessian/kernel_kx.py`
(`kx1_` / `kxc_{la}{lc}_{lb}{ld}`), `gk`'s mapping and plan in its rolled
form only, scattering `X[3, n_ao, n_ao]` by `AtomicAdd` per `b` row, no
warp reduction; the caller assembles `K^X = P_X K1 + (P_X K1)^T + KC_X +
KC_X^T`. Host reference `hessian/host_kx.py`, held to PySCF
(`test_hessian_host_kx.py`, `slow`): `K1` to the gradient fixtures'
`vk1` (8.3e-14 at H₂O/cc-pVDZ) and whole `K^X` per atom to
`test/oracle/k_deriv/*_kx.json`, `make_h1`'s exchange part (1.5e-13 at
cc-pVDZ; H₂O/STO-3G and its d case too), to finite differences of
`exchange_matrix`, and `SUM_X K^X = 0`. Interpreted, both forms within 16
eps of the host's absolute twin per class, every class through (dd|dd)
and (ff|ff), (fs|fs), (fd|pd) (`test_kernel_kx.py`, L + 1 > 4 `slow`).
Ladder `--hess-kx-l [LMAX]` (1 bare, 0..3), never in the bare default nor
in `KERNEL_GEN_CMD`; 14 goldens plus 498 digests a backend
(`test_emit_hess_kx.py`, `test/golden/kx_digests.json`), the goldens and
(ff|ff) compiled by nvcc and hipcc. The frame of all 512 units on both
arches is in `data/measured/kx.json` (`devtools/measure.sh kx-compile`,
compiler only): all admissible under the declared 14,000 B / 30 s, worst
`kx1_33_33` at 9,824 B (`sm_86`) and 10,192 B (`gfx1200`), slowest
compile 5.6 s. No device test or driver: nothing has
launched on either card. `J^X` (#688) and `h^x` (#687) are not built.
**The DF metric's Hessian too, through (i|i), interpreted and nvcc-compiled
(#694); a device test written, on no card**
(`hessian.md`, "The DF metric Hessian"): `Target.METRIC` joins
`spec.HESSIAN_TARGETS` (MMD only; `metric.hgp` refuses d2), a
d2 key `coulomb/d2/(lP,lQ)/b1/metric/literal`, every d0 and d1 key
byte-identical. `hessian/kernel_d2metric.py` (`d2metric_{lP}{lQ}`, not
`hmetric_`, the HGP metric's) writes `H[A, k, B, l] += SUM_PQ W_PQ
d2(P|Q)/dR dR` for any `W` (DF-J's metric term passes `-1/2 d d^T`,
skeleton only) by the Hermite shift twice: `h_kl = SUM E^P E^Q (-1)^|tau|
[t + tau + 1_k + 1_l]^(0)` (six, `h` symmetric), `+h` on the two
same-centre blocks and `-h` on the mixed ones, 36 `AtomicAdd`s per pair;
`kernel_gmetric`'s mapping, parameters (`H[n_atom, 3, n_atom, 3]` for
`G`), weighted bra density and tower, the latter the plan of `(lP + 2, 0 |
lQ, 0)` cut to the live targets. Host reference `hessian/host_df.py`
(`metric_hessian`, `absolute=True` its twin) by another route,
`host_1e`'s moved Cartesians on the single-center `E`, held to central
finite differences of `gradient.host_df.metric_gradient` (~2.5e-9 of
`max|H|`), `H = H^T` and invariance (`test_hessian_host_df.py`; s/p fast,
d and f/g `slow`); interpreted, every class `lP + lQ <= 4` within 0.17 eps
of the absolute twin, gate 16 (`test_kernel_d2metric.py`, `lP + lQ + 2 >
4` `slow`; launches `test/df_grad_launch.run_d2metric`). **#694 raised
it to the d1 DF ladder's reach**: ladder `--hess-metric-l [AUXMAX]`, now
the aux `l` per shell (2 bare, 6 units; 0..6), never in the bare default
nor in `KERNEL_GEN_CMD`; the 9 goldens per backend unchanged, the other 19
classes through (i|i) pinned by digest (`test_emit_d2metric.py`,
`test/golden/d2metric_digests.json`). A LOW-STACK form (`kernel_gmetric`'s
one order up) is the default from `lP + lQ >= LOWSTACK_MIN_L = 9`, the
floor measured (`devtools/measure.sh d2-df-lowstack-compile`,
`data/measured/d2_df_lowstack.json`, rule declared before any row; the
unrolled (d|i) is 19,216 B of `sm_86` stack). Interpreted, every class
`lP <= lQ <= 6` in its default form, every class below the floor forced
low-stack and (d|i), (g|h) forced unrolled, within 0.34 eps of the twin
(`test_kernel_d2_df_ladder.py`, 52 cases). **nvcc 13.0 (`sm_86`) compiles
all 28 units** (#92's filter, 2026-10-30, a device-less container); every
low-stack frame is <= 9,200 B and <= 1.9 s of nvcc. hipcc has not compiled
them and there are no `gfx1200` rows. A device test and driver are written
(`test_hess_df_device.py`, `hess_driver.d2metric_launches`: H₂O/cc-pVDZ
against cc-pVDZ-JKFIT with DF-J's `W`, and the d1 ladder's aux-i system per
aux `l`) and have **run on neither card**. No `d` response, no assembled
DF-J Hessian, no clock measured.
**The three-center DF integral's Hessian too, through (ff|i), interpreted and
nvcc-compiled (#694); a device test written, on no card** (`hessian.md`, "The three-center DF Hessian"): `Target.ERI3` joins
`spec.HESSIAN_TARGETS` (MMD only), a d2 key
`coulomb/d2/(la,lb|lP)/b1/eri3/literal`. `hessian/kernel_d2eri3.py`
(`d2eri3_{la}{lb}{lP}`, not `heri3_`: the HGP `(mn|P)`'s) writes
`H[X, k, Y, l] += SUM_mnP Gam_mnP d2(mn|P)/dR dR` over the three centres:
`A` and `B` by the Cartesian rule twice on per-axis `E` two indices up
(`kernel_h1e.axis_ops` / `operated`), the aux `C` by the Hermite shift
twice (`[r + 1_k + 1_l]^(0)`, the tower of the J quartet `(la + 2, lb |
lP, 0)`); `kernel_geri3`'s mapping, parameters (`H` for `G`) and weights
(`cell_weights`, made public, its text byte-identical), 45 accumulators
and 81 atomics per cell. Host reference `hessian/host_eri3.py`
(`eri3_hessian`, `absolute=True` its round-off twin) by another route on
every centre (`host_1e`'s moved 3-D Cartesians, the aux one included),
held to finite differences of `gradient.host_df.eri3_gradient` (4.8e-9
of `max|H|`), invariance and symmetry (`test_hessian_host_eri3.py`, H₂O
against JKFIT s, p `slow`); interpreted, every class `la + lb + lP <= 4`
within 16 eps of the twin (measured <= 0.40) and the whole H₂O/STO-3G
(`test_kernel_d2eri3.py`, L + 2 > 4 `slow`). **#694 raised it to the d1
DF ladder's reach**: ladder `--hess-eri3-l [LMAX] [--hess-eri3-aux-l
AUXMAX]`, split as `--grad-df-l` (orbital 1 and aux 2 bare, 9 units;
0..3 and 0..6), never in the bare default nor in `KERNEL_GEN_CMD`; 7
goldens per backend unchanged, every other class of the old and new
ladders (71 in all) pinned by digest in its default form, the 15 old L = 3
and 4 unrolled digests kept as `<name>.unrolled` (`test_emit_d2eri3.py`).
A LOW-STACK form (`kernel_geri3`'s, with the low-stack `d2v`'s operated
factors and rolled contractions, `kernel_h1e.operated_value` / `rolled`,
made public with the `d2v` text unchanged) is the default from `la + lb +
lP >= LOWSTACK_MIN_L = 3`, the floor measured (same sweep; the unrolled
(sp|d) is 14,768 B, (pp|p) 20,280 B). Its frames are RECORDED, NOT GATED
(declared before any row): the bra tables two indices up are 9.5 KB alone
at (ff|.), and 19 units exceed #606's 14 KB, worst (ff|i) at 24,544 B
with 8 KB of spill. Every unit compiles in <= 29 s. Interpreted, every
class of the ladder in its default form, every class below the floor
forced low-stack and (sp|d), (pp|p) forced unrolled, against the host's
twin (`test_kernel_d2_df_ladder.py`). **nvcc 13.0 (`sm_86`) compiles all
71 pinned units** (#92's filter, 2026-10-30); hipcc has not, no
`gfx1200` rows. A device test and driver are written (`test_hess_df_device.py`,
`hess_driver.d2eri3_launches`) and have **run on neither card**. No
assembled DF-J Hessian (with `d2metric_` and `d`'s response), no clock
measured.
