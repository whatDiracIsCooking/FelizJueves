# Nuclear gradients (`derivative_order = 1`)

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**The first `derivative_order = 1` keys: analytic nuclear gradients by
McMurchie–Davidson, run on both cards at L <= 4** (`docs/implementation/gradient.md`
is the one place with the detail; the port of `experimental/mmd/gradient`):
`spec.DERIVATIVE_ORDERS` is `(0, 1)`, order 1 legal on
`spec.GRADIENT_TARGETS` (`S`, `T`, `V`, `JAB`, `K`, `METRIC`, `ERI3`, MMD only;
`_GRADIENT_REFUSAL` elsewhere), every d0 key byte-identical. A d1 kernel
writes `G[A, k] += SUM_mn M_mn dX_mn/dR_{A,k}`, not the derivative
integrals, by the rule `d/dA x^i e^{-a x^2} = 2a x^{i+1} - i x^{i-1}` on
`e_fragment.axis_tables_fragment` one index above the class:
`gradient/kernel_g1e.py` (`gs_`, `gt_`, `gv_{la}{lb}`; `gv` with the
Hellmann–Feynman term per nucleus, `[t]^(0)` from
`kernel_v.v_plan((la, lb + 1))`) and `gradient/kernel_gjab.py`
(`gjab_{la}{lb}`, `dE_J/dR` as the bra derivative alone, reading `J_p` of
the d0 `[p|q]` class with the bra raised by one,
`kernel_gjab.gjab_pq_spec`), and `gradient/kernel_gk.py`
(`gk_{la}{lc}_{lb}{ld}`, K's Hermite pairing: `dE_K/dR`, `E_K = 1/2 SUM
D_ab D_cd (ac|bd)`, as the bra derivative alone over every ordered
quartet, `D` symmetric; `kernel_k`'s mapping, one warp per bra record, no
screen, no shared memory; the plan of the d0 K class with the bra raised
by one, `kernel_gk.gk_pq_spec`; the derivative through raised rows
`V(a +- 1_k, c)` folded once and digested per axis). Host references
`gradient/host.py`, held to finite differences of the libcint-witnessed
`S`, `T`, `V`, `coulomb_matrix` and `exchange_matrix`
(`test_gradient_host.py`, ~5e-9 of `max|G|`, the `O(h²)` floor) and to
translational invariance; interpreted, `gs`/`gt` are bitwise
the host and `gv`/`gjab` within 16 eps of a split-centre scale through
(ff) (`test_kernel_g1e.py`, `test_kernel_gjab.py`, which also runs the
chain through the interpreted raised `[p|q]`), and `gk` within 16 eps
(measured <= 0.7) of the host's `absolute=True` twin for every K class
through `(dd|dd)` (`test_kernel_gk.py`, L + 1 > 4 `slow`). Ladders
`python -m felizjueves.emit --grad-l [LMAX]` (2 bare, 0..3) and
`--grad-k-l [LMAX]` (1 bare, 0..3 since #618), never in the bare default
nor in `KERNEL_GEN_CMD`; 48 + 40 goldens (`test_emit_grad.py`,
`test_emit_grad_k.py`, every other default `gk` unit through `(ff|ff)`
pinned in `test/golden/gk_digests.json`).
**On both cards** (`test_grad_device.py`, generator `test/grad_driver.py`,
2026-10-27): `gs`, `gt`, `gv`, `gjab` on H₂O/STO-3G and cc-pVDZ (`slow`),
`gk` on H₂O/STO-3G only, against the host, 9 passed per card; worst
3.20e-14 of `max|G|` on the RTX 3080 and 2.36e-14 on the RX 9060 XT (`gt`,
cc-pVDZ), under `1e-13`. **`gs`, `gt`, `gv` through (ff) on both cards**
(#614; the same test's `co_ccpvtz_proton-*` cases, `slow`): CO/cc-pVTZ
(f on both atoms, all 16 ordered classes (ss)–(ff) asserted) with a bare
proton as a third nucleus for `gv`'s Hellmann–Feynman term, each program's
compile under a 300 s timeout, 12 passed per card; worst 1.29e-14 /
7.64e-15 of `max|G|` (`gt`, RTX 3080 / RX 9060 XT), `gv` 1.94e-15 /
2.80e-15. **From `la + lb = 5` `gv` defaults to a low-stack form**
(`kernel_g1e.g1e_lowstack`, `LOWSTACK_MIN_LSUM = 5`; `lowstack=` forces
either, `gs` / `gt` refuse `True`; name, parameters, launch and `G` the
same; below the gate the text is byte-identical, goldens unchanged): no
Hermite density, per nucleus `[t]^(0)` into a local `R`
(`store_targets`), per block row the derivative and Hellmann–Feynman
terms as `rolled_e_sum`s over the tables one index up, its weight read
from `M` (`kernel_geri3`'s low-stack shape). **Measured** in
`data/measured/g1e.json` (`measure/measure_g1e.py`, `devtools/measure.sh
g1e-compile`, a compiler and no card, rule declared before any row, a
separate store as #606 / #613; 48 unrolled + 16 low-stack units per
arch, each alone under a 300 s kill, admissible at <= 14,000 B on both
arches; `measure.sh check` holds the gate to the rows): unrolled `gv_33`
50,016 B of `sm_86` stack, 102,540 B spill stores, 144 s (`gfx1200`
15,244 B, 38 s), `gv_23` / `gv_32` 21,104 / 18,472 B; low-stack `gv_33`
7,712 B, 2.4 s (6,448 B on `gfx1200`). `gs` / `gt` fit unrolled
everywhere (`gt_33` 5,296 B), one form; `gv_22` is 9,752 B, over #586's
d0 8 KB but under #613's 14,000 B, so (dd) stays unrolled. Interpreted,
`gv` in its other form at every class (`test_kernel_g1e.py`). No clock.
**`gk` defaults to a ROLLED form from `L = 5` and at every f class**
(#618; `gradient.md`, "The rolled `gk`"; `kernel_gk.gk_lowstack`,
`LOWSTACK_MIN_L = 5`, `UNROLLED_L_MAX = 2`; `lowstack=` forces either;
name, parameters, launch and `G` the same; below the floor the text is
byte-identical, goldens unchanged). No `E` block, no straight-line fold:
the bra's per-axis tables one index up and the ket's (its sign folded
into the odd-`t` entries), both COMPACT (`tables_fragment`, only `t <= i
+ j` stored), `[r]^(0)` into a local `R` (`store_targets`), stage 1 into a
local `I[p]` per `(b, d)` row, and each `(a, c)` row's derivative terms
as loop nests over the tables, `W` read from `M`. The unrolled f form is
never built (the user's decision), nor the block mapping (`kernel_kblock`'s
one warp per `K_AB` block: only a cheap rolled variant was in scope, and
#571 / #583 measured those slower or neutral). **Measured** in
`data/measured/gk.json` (`measure/measure_gk.py`, `devtools/measure.sh
gk-compile`, a compiler and no card, rule declared before any row, every
unit alone under a **10 s kill**, the user's #618 budget, admissible at
<= 14,000 B on both arches; `measure.sh check` holds the gate): the
unrolled form is admissible through L = 4 on both (worst `gk_22_00`
11,696 B on `sm_86`) and over on `sm_86` at L = 5 (`gk_22_10` 14,880 B);
all 256 rolled units fit both, worst `gk_33_33` 11,616 B / 2.8 s on
`sm_86` and 10,736 B / 2.2 s on `gfx1200`, slowest compile 4.8 s. A
tracked **libcint witness**, `test/oracle/k_gradient/co_ccpvtz_k_grad.json`
(`manifest.k_gradient`, `python -m oracle.generate --regenerate
--k-gradient`): `int2e_ip1_cart` on CO/cc-pVTZ contracted with
`co_ccpvtz_j`'s recorded density, per K class (256) and whole; the host
matches it on four s, p, d classes within 2.7e-16 of `max|G|`
(`test_k_gradient_witness.py`, `slow`). On the cards `gk` runs through
`test/gk_driver.py` (every unit its own translation unit, `-c` under the
10 s cap, linked; one `G` per launch) in `test_gk_device.py`, which took
over `test_grad_device.py`'s `gk` case. **On both cards** (2026-10-29):
H₂O/STO-3G, every class in both forms on one upload, whole against the
host 3.22e-15 / 3.37e-15 of `max|G|` (RTX 3080 / RX 9060 XT), rolled
against default 9.2e-16; and CO/cc-pVTZ (`slow`, `high_l`, hand-run), all
256 classes in their default forms against the libcint witness per class
and whole, worst 2.63e-15 / 2.94e-15 of `max|G|` (gate 1e-12), plus the
on-card differential against the unrolled form over the 62 / 58 classes
whose unrolled unit compiled within half the budget, 9.2e-17 / 1.2e-16;
319 / 315 units, every one under the 10 s cap (slowest 5.7 s / 6.1 s).
No whole RHF gradient assembled; `gk` measured for its compile only (the
DF-J terms: below). **`gjab` has a low-stack form, no class's default** (#619;
`kernel_gjab.build_gjab_kernel(cls, lowstack=True)`, `gv`'s low-stack
shape reading the record's `Jp` row from global memory, no `R`; the
per-row weight and derivative terms shared with `gv` through
`kernel_g1e.lowstack_weight` / `lowstack_derivative`, `gv`'s text
byte-identical): interpreted at every class through (ff) in both forms
(`test_kernel_gjab.py`; unrolled 1.8, low-stack 5.7 eps of the
split-centre scale, gate 16) and the interpreted chain in both forms.
`kernel_gjab.LOWSTACK_MIN_LSUM` is **None**: `data/measured/gjab.json`
holds the declared rule (#614's, restated) and **no row**
(`measure/measure_gjab.py`, `devtools/measure.sh gjab-compile`, never run:
the session had no compiler), and `measure.sh check` holds the floor at
None until every class has both arches' rows. Not compiled by nvcc or
hipcc in either new form; the on-card chain (`DQ`, raised `[p|q]`,
`gjab`, no host `J_p`) is not built.
**The DF-J gradient terms are generated too, run on both cards through
(ff|i) and (i|i)** (#613)
(`gradient.md`, "The density-fitting gradients"): `gmetric_{lP}{lQ}`
(`gradient/kernel_gmetric.py`, `SUM_PQ W_PQ d(P|Q)/dR`) and
`geri3_{la}{lb}{lP}` (`gradient/kernel_geri3.py`, `SUM_mnP Gam_mnP
d(mn|P)/dR`, `Gam` `[m][n][P]`), on the d0 kernels' staging and mapping
plus atoms and exponents (`gradient/host_df.py`'s `metric_records` /
`eri3_records`); DF-J is `Gam = D (x) d`, `W = -1/2 d d^T`. The AO
centres by the Cartesian rule on per-axis `E` one index up; every aux
centre by the Hermite shift `[r]^(0) -> [r + 1_k]^(0)` on the tower of the
J quartet one order up (`(mn|P)`'s aux centre computed, not inferred;
`(P|Q)`'s `d/dB = -d/dA` exactly). Host reference `gradient/host_df.py`
takes the Cartesian rule on every centre (another route), held to finite
differences of `metric_matrix`, `eri3_tensor` and a whole DF-J `E_J`
(H₂O/STO-3G against the JKFIT s, p shells `slow`; ~2-3e-9 of `max|G|`)
(`test_gradient_host_df.py`); interpreted, every class at L <= 4 within
4.9 (`gmetric`) and 2.6 (`geri3`) eps of the split-centre scale, gate 16
(`test_kernel_gmetric.py`, `test_kernel_geri3.py`, launches in
`test/df_grad_launch.py`). Ladder `--grad-df-l [LMAX] [--grad-df-aux-l
AUXMAX]` (#613: split as FreeBreakfast's `df_kernels.py`, LMAX the orbital
`l`, 1 bare, 0..3; AUXMAX the aux `l`, 2 bare or omitted, 0..6; `geri3`
for `la <= lb <= LMAX`, `lP <= AUXMAX`, `gmetric` for `lP <= lQ <=
AUXMAX`), never in the bare default nor in `KERNEL_GEN_CMD`; 62 goldens,
the 31 classes at L <= 4 (`test_emit_grad_df.py`). **On both cards**
(`test_grad_df_device.py`, `slow`, on `grad_driver`'s program): DF-J of
H₂O/STO-3G against the JKFIT s, p, d shells, worst 6.58e-15 of `max|G|`
(RTX 3080) and 4.87e-15 (RX 9060 XT).
**Through (ff|i) and (i|i) (#613), with a low-stack form** (`gradient.md`,
"The low-stack form"): `build_geri3_kernel(cls)` /
`build_gmetric_kernel(cls)` pick it from an `L` floor,
`kernel_geri3.LOWSTACK_MIN_L = 5` (49 of 70 classes) and
`kernel_gmetric.LOWSTACK_MIN_L = 10` (4 of 28); `lowstack=` forces either,
name, parameters, launch and `G` contract the same, no L <= 4 class gated
(goldens unchanged). It holds no density or weight: the tower into a local
`R` (order `L + 1`), `U[t]` per aux row, and each element's terms as rolled
sums over per-axis tables (`geri3`: the bra's one index up, `dA_k` as
`2 alpha SUM E^{a+1_k} U - a_k SUM E^{a-1_k} U`, the aux centre through
`e_fragment.rolled_e_sum(..., up=k)`, new), its weight read from `M`. The
unrolled form's contraction folds are chunked (`kernel_gmetric.
chunked_fold`, bitwise the same left fold, text unchanged below 129
terms), so (ff|i) builds at Python's default recursion limit (#613's
`RecursionError`). **Measured** (`data/measured/df_grad_lowstack.json`,
`devtools/measure.sh df-grad-lowstack-compile`, its rule declared before
any row; `measure.sh check` holds the gate to the rows): every class's
low-stack form and the unrolled form level by level in `L` through the
first level with an inadmissible class, each unit alone under a 300 s
budget, admissible at <= 14,000 B of stack on both arches. Unrolled, every
L <= 4 `geri3` and L <= 9 `gmetric` class fits; at L = 5 the unrolled
(sf|d), (pd|d), (pf|p), (dd|p), (df|s) are 14,184 to 27,144 B on `sm_86`,
and (g|i), (h|h) 15,968 / 18,200 B. Every low-stack unit is at most
13,792 B of `sm_86` stack (`geri3_336`, 1,916 B spill stores, 3.1 s) and
12,816 B of `gfx1200` scratch, compiled in at most 5.2 s. Not timed.
Interpreted (`test_kernel_df_grad_ladder.py`, `slow` above L + 1 = 4):
every class above L = 4 against `host_df` within 16 eps of the
tower-twin split-centre scale (`test_kernel_metric.pq_tower` on `|Gam|`;
the signed scale is blind to the tower's cancellation from L ~ 9), every
L <= 4 class in the forced low-stack form too. Against libcint: a tracked
**DF gradient witness** (`test/oracle/df_gradient/prim_spd_f_auxi_grad.json`,
`oracle.manifest.df_gradient`, written by `python -m oracle.generate
--regenerate --df-gradient`): `int3c2e_ip1_cart` / `int3c2e_ip2_cart` and
`int2c2e_ip1_cart` on the aux-i sets' system (#606's `prim_spd_f` against
l = 0..6 aux shells), contracted per class with closed-form `Gam` / `W`,
its inputs held to the tracked sets exactly; Cartesian, not the spherical
integrals #613 named. The host matches it at every ladder class and the
kernel at the top classes, within `1e-12` of max. **On both cards**
(`test_grad_df_device.py` Part 2, `slow` + `high_l`, one program per aux
`l`, each compile under a 300 s timeout): all 70 + 28 classes in their
default forms on that system, 16 passed per card with Part 1; worst
1.20e-14 (RTX 3080) / 9.34e-15 (RX 9060 XT) of `max|G|` against the host
and 5.5e-14 of max against libcint, runs bit-identical. **The DF-K
weights and the whole DF two-electron gradient are staged on the host**
(#620; `gradient.md`, "The density-fitting gradients"):
`host_df.df_k_stage` gives `E_K`, `Gam^K`, `W^K` from the occupied
coefficients (`df_j_stage` the J ones), `df_rhf_gradient` the terms
`G_J`, `G_K` and `G_J - 1/2 G_K` apart; DF-K held to central differences
of `E_K` (3.6e-9 of `max|G|`; H₂O/STO-3G against JKFIT s, p `slow`,
2.6e-9), the whole two-electron gradient to PySCF's DF RHF gradient over
the same aux set (`oracle.manifest.df_rhf`, H₂O/STO-3G against
cc-pVTZ-JKFIT s, p, closed-form orbitals, `slow`): 7.8e-13 absolute, gate
`1e-10`; `geri3` / `gmetric` interpreted with the DF-K weights within 0.37
eps of the split-centre scale (`test_kernel_df_k_grad.py`). Not run on a
card with the DF-K weights; the solve stays host-side. No clock.
