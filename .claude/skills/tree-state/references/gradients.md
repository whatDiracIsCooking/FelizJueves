# Nuclear gradients (`derivative_order = 1`)

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**The first `derivative_order = 1` keys: analytic nuclear gradients by
McMurchie–Davidson, run on both cards at L <= 4** (`docs/implementation/gradient.md`
is the one place with the detail; the port of `experimental/mmd/gradient`):
`spec.DERIVATIVE_ORDERS` is `(0, 1, 2)` (order 2: `hessians.md`), order 1 legal on
`spec.GRADIENT_TARGETS` (`S`, `T`, `V`, `JAB`, `K`, `METRIC`, `ERI3`, and
`DQ` for the Hessian, MMD only;
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
**On both cards** (`test_grad_device.py`, generator `test/grad_driver.py`
-- on the shared runner since #659, which ran on both cards on 2026-10-30 --, 2026-10-27): `gs`, `gt`, `gv`, `gjab` on H₂O/STO-3G and cc-pVDZ (`slow`),
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
10 s cap, linked; one `G` per launch; on the shared runner since #657,
which has not run on a card, its H₂O program compiled under nvcc and
hipcc) in `test_gk_device.py`, which took
over `test_grad_device.py`'s `gk` case. **On both cards** (2026-10-29):
H₂O/STO-3G, every class in both forms on one upload, whole against the
host 3.22e-15 / 3.37e-15 of `max|G|` (RTX 3080 / RX 9060 XT), rolled
against default 9.2e-16; and CO/cc-pVTZ (`slow`, `high_l`, hand-run), all
256 classes in their default forms against the libcint witness per class
and whole, worst 2.63e-15 / 2.94e-15 of `max|G|` (gate 1e-12), plus the
on-card differential against the unrolled form over the 62 / 58 classes
whose unrolled unit compiled within half the budget, 9.2e-17 / 1.2e-16;
319 / 315 units, every one under the 10 s cap (slowest 5.7 s / 6.1 s).
`gk` measured for its compile only (the DF-J terms: below).
**The whole RHF gradient is assembled and interpreted** (#565,
`gradient.md`, "The RHF gradient, assembled"): `gradient/rhf.py`'s
`rhf_gradient(basis, nuclei, D, W, runners)` signs and sums `-W dS`, `D dT`,
`D dV`, `dE_J`, `-1/2 dE_K` and `V_nn^x` (host) from injected per-term
runners, each term returned with their fixed-order total;
`gradient/interpreted.py`'s `interpreted_runners(boys=)` runs `gs_`, `gt_`,
`gv_`, the J chain (`[p|q]` one bra order up through `KernelJP`, then
`gjab_`) and `gk_` in the interpreter, each in its default form. Held to
#550's gradient fixtures (`manifest.gradient_fixture`; `grad:h` split into
`T` and `V` by its stored `int1e_ipkin`) per term and in total within
`1e-10` absolute (`test_rhf_gradient.py`): H₂O/STO-3G, H₂O/6-31G, C1
NH₂F/6-31G, H₂O/cc-pVDZ, and #565's `h2o_sto3g_d_grad` (H₂O/STO-3G plus one
d primitive on O, `Case.extra_shells`: every J and K class through
`(dd|dd)` at STO-3G's cost). The one-electron terms on s, p bases are fast;
every J/K term, every total and every d case `slow`. Worst against PySCF 7.5e-14 per term and 7.9e-14 in total (H₂O/STO-3G), 6.9e-14 / 6.9e-14 on the d case; H₂O/6-31G passes. Not run to the end: the NH₂F/6-31G and H₂O/cc-pVDZ totals (`gk_`'s rolled form, the default from L = 5, interprets for hours on their contractions; the d case covers cc-pVDZ's classes, and NH₂F's p on two atoms is held class by class to the host by the kernel tests). **On the RX 9060
XT only (#566, 2026-10-30; CUDA not run)**: `test_rhf_gradient_device.py`
(`slow`) injects device runners -- `gs_` / `gt_` / `gv_`
(`grad_driver`), the J chain on the card (DQ -> raised `[p|q]` ->
`gjab_`, no host `J_p`) and every `gk_` class (`gk_driver`) -- into
`rhf_gradient` on H₂O/STO-3G and 6-31G, 2 passed: every term and the
total within `1e-10` absolute of PySCF (worst total 8.2e-14, STO-3G) and
of the interpreted `rhf_gradient` (worst 4.4e-15); run to run within
`SPREAD_TOL = 1e-13` (worst 4.4e-15, 6-31G) -- NOT bit-identical, the
kernels' `atomicAdd`s, so the issue's bit-identity was restated as that
bound by the user. **Nothing scored or timed; the milestone's limits are
one `StructuralLimit` record** (#567, `data/literature/jengine/limits.json`;
`gradient.md`, "What is not done" and "What AIMD still needs"): no
published board, Cartesian only, the whole gradient on the RX 9060 XT
only and on s, p bases only, cc-pVDZ's whole gradient on no card, and for
AIMD an SCF producing `D` and `W` (no eigensolver in `src/`), the per-step
host rebuild of pairs, bounds and plans, a derivative screen, d/f in the
whole gradient on a card, RI gradients end to end, UHF. **`gjab` defaults to a low-stack form from `la + lb =
5`, and its whole chain runs on both cards through (ff)** (#619;
`kernel_gjab.build_gjab_kernel(cls, lowstack=True)`, `gv`'s low-stack
shape reading the record's `Jp` row from global memory, no `R`; the
per-row weight and derivative terms shared with `gv` through
`kernel_g1e.lowstack_weight` / `lowstack_derivative`, `gv`'s text
byte-identical): interpreted at every class through (ff) in both forms
(`test_kernel_gjab.py`; unrolled 1.8, low-stack 5.7 eps of the
split-centre scale, gate 16) and the interpreted chain in both forms.
**Measured** in `data/measured/gjab.json` (`measure/measure_gjab.py`,
`devtools/measure.sh gjab-compile`, a compiler and no card, #614's rule
restated and declared before any row, 16 unrolled + 16 low-stack units
per arch, each alone under a 300 s kill, admissible at <= 14,000 B on
both arches; `measure.sh check` holds the floor to the rows): unrolled
`gjab_23` / `gjab_32` 18,976 / 21,240 B of `sm_86` stack and `gjab_33`
46,680 B (93,788 B spill stores, 115 s; `gfx1200` 11,908 B, 15 s), so
`kernel_gjab.LOWSTACK_MIN_LSUM = 5`; through `la + lb = 4` unrolled
(worst `gjab_22` 8,648 B), text and goldens unchanged; every low-stack
unit fits both (worst `gjab_33` 6,352 / 5,424 B, under 3 s). **The chain
on the card** (`test_gjab_chain_ondevice.py`; `grad_driver.raised_plan`
and `gjab_tail` on `plan_driver.driver`'s new `tail=` slot,
`plan_driver.Tail`; a shared-runner sequence since #659, both cards
2026-10-30): `DQ` on the card, read back, the J chain's
plan keyed on its `Dmax`, every bra class raised by one, the raised
`[p|q]` launched, then `gjab` reading `J_p` where it lies, in one process,
no host `J_p`; against `host.coulomb_gradient` at `1e-13` of `max|G|`,
every compile under 300 s. On both cards (2026-10-29), 5 passed per card:
H₂/STO-3G (the PR gate) and H₂O/STO-3G (`slow`) in both forms, worst
2.1e-15 / 2.0e-15 (RTX 3080 / RX 9060 XT); CO/cc-pVTZ (`slow`, `high_l`,
hand-run: all 16 ordered classes, (df), (fd), (ff) low-stack, raised
`[p|q]` to L = 13) 9.0e-16 / 1.07e-15. The CO cell takes ~17 / 20 min,
almost all of it the pure-Python host reference. The forced-unrolled
(df) / (ff) units are compiled, not run on a card. That cell's CO/cc-pVTZ
reference is now tracked so it need not run the host (#639): a
**libcint witness**, `test/oracle/j_gradient/co_ccpvtz_j_grad.json`
(`manifest.j_gradient`, `python -m oracle.generate --regenerate
--j-gradient`), `int2e_ip1_cart` contracted with `co_ccpvtz_j`'s recorded
density into `dE_J/dR` per `gjab` bra class (10) and whole; the host
matches it on every class within 1.25e-15 of `max|G|` (gate 1e-14,
`test_j_gradient_witness.py`, `slow`: 13 min on 4 workers, ~37 CPU-min
of host), and its `oracle` test rebuilds it from PySCF in ~2 s. The
CO cell now holds the card to the witness's whole `G` at the same
`DEV_TOL = 1e-13` (H₂ and H₂O keep the live host), with a fast no-card
check that the witness is on the cell's own density and nuclei; the
cell has **not** been re-run on either card since, so its after-time
is unmeasured (#639).
**`gmetric` has an HGP twin, interpreted and compiled, on no card**
(`docs/implementation/hgp_df.md`, "The metric's nuclear gradient on HGP"):
`gradient/kernel_hgmetric.py`'s `hgmetric_{lP}{lQ}`, keyed
`coulomb/d1/(lP,lQ)/b1/metric.hgp/literal` (`spec.GRADIENT_OFF_NATIVE`,
beside `eri3.hgp`), `gmetric`'s parameters, staging, weights and `G`
contract; `d/dA = 2 alpha [a+1_k|b] - a_k [a-1_k|b]` per primitive from one
HGP VRR of the bra one up and one down (`hgmetric_dag`), `d/dB = -d/dA`.
Interpreted (`test_kernel_hgmetric.py`): every class `lP + lQ <= 4` against
`host_df` within 16 eps of the split-centre scale (worst 7.87, (d|d)) and
against `gmetric` within 32. Ladder `--hgp-grad-metric-l [LMAX]` (0..1,
bare 1), never in the bare default nor in `KERNEL_GEN_CMD`; six goldens,
compiled by nvcc and hipcc warning-free (`test_emit_hgp_df.py`, device-less
cloud container, 2026-10-30). Not run on a card, not measured; its
`(mn|P)` sibling is `hgeri3`, below.
**The metric's derivative uncontracted, `ipmetric_`, interpreted and
compiled through (i|i), on no card** (#693; `gradient.md`, "The
uncontracted metric derivative"): `gradient/kernel_ipmetric.py`'s
`ipmetric_{lP}{lQ}` writes `M1[k, P, Q] = d(P|Q)/dA_k` (`A` the centre of
`P`, `= -int2c2e_ip1`) as `[3, n_aux, n_aux]`, the mirror `(Q, P)` negated;
keyed `coulomb/d1/(lP,lQ)/b1/metric/uncontracted` (the `uncontracted`
kernel-body lever, `metric` at d1 only). Interpreted
(`test_kernel_ipmetric.py`): all 28 classes `lP <= lQ <= 6` against
`host_df.metric_derivative` (worst 3.62 eps, gate 16; above L = 4 `slow`);
the host against libcint `-int2c2e_ip1_cart` on cc-pVDZ-JKFIT/H₂O (the
numeric golden store, one axis per entry, 7.21e-14) and through (i|i)
against the aux-i witness contracted (1.72e-14); the kernel against both
(at most 7.21e-14). Ladder `--ip-metric-l [LMAX]` (0..6, bare 2), never in
the bare default nor in `KERNEL_GEN_CMD`; six goldens; all 28 classes
compiled by nvcc and hipcc warning-free (`test_emit_ipmetric.py`,
device-less cloud container, 2026-10-30). Not run on a card (no device
harness yet), not measured.
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
(`test_grad_df_device.py`, `slow`, on `grad_driver`'s program, on the shared runner since #659, which ran on both cards on 2026-10-30): DF-J of
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
and 5.5e-14 of max against libcint, runs bit-identical. **The `(mn|P)` gradient on the HGP engine too, interpreted and compiled
only** (`gradient.md`, "The three-center gradient on HGP"):
`gradient/kernel_hgeri3.py`'s `hgeri3_{la}{lb}{lP}`, keyed
`.../d1/.../eri3.hgp/...` (`spec.GRADIENT_OFF_NATIVE`, beside
`metric.hgp`), `geri3`'s parameters, staging, mapping and `G` contract. The
Cartesian rule on all three centres (the aux one too, by its exponent-0
partner: no Hermite shift, nothing inferred), each raised class contracted
with its own primitive weight `2 xi` / `2 eta` / `2 gam`, one VRR per
primitive quartet over every derivative HRR's boundary
(`hgp.vrr.group_dag`), up to six bra HRRs per cell. Interpreted
(`test_kernel_hgeri3.py`): every class at L <= 4 against `host_df` within
9.05 eps of the split-centre scale (gate 16) and translational invariance,
against `geri3` within 15.2 eps (gate 32), and 8 s/p/d classes against
the libcint DF gradient witness within 4.95e-15 of its largest `max|G|`.
Ladder `--hgp-grad-eri3-l [LMAX]` (s and p, 0..1, bare 1: 6 units), never
in the bare default nor in `KERNEL_GEN_CMD`; 12 goldens
(`test_emit_hgp_grad_df.py`), compiled by nvcc and hipcc warning-free on a
device-less container (2026-10-30). **No card, no device harness**, not
measured. **The UNCONTRACTED `(mn|P)` derivative integrals, interpreted
and compiled only** (#692; `gradient.md`, "The uncontracted three-center derivative"):
`gradient/kernel_xeri3.py`'s `xeri3_{la}{lb}{lP}`, keyed
`coulomb/d1/(la,lb|lP)/b1/eri3/uncontracted` (#693's kernel-body lever
`uncontracted`, widened from `metric` to MMD `eri3`, d1 only), writes `X[k, m, n, P - aux0] = (nabla_k
m n|P)` -- libcint's `int3c2e_ip1`, no `Gam`, no atom reduction -- over
the cells of one caller-chosen aux block, `X` `[3, n_ao, n_ao, n_blk]`,
plain stores (each slot one cell's), zeroed by the kernel; the aux centre
is the caller's, `-X - X^T` (`int3c2e_ip2`). `geri3`'s low-stack route at
every class, one form. Host reference `host_df.eri3_ip1`, held to libcint
live (`oracle` + `slow`): `int3c2e_ip1_cart` within 3.54e-14 of max on
H₂O/cc-pVDZ with cc-pVDZ-JKFIT and 2.21e-14 on the aux-i system
(`prim_spd_f`, AUX_I: orbital f, aux i), the identity against
`int3c2e_ip2_cart` within 3.09e-14 / 1.95e-14, and through the tracked DF
gradient witness contracted per class (3.91e-14 of `max|G|`, `slow`).
Interpreted (`test_kernel_xeri3.py`): every class at L <= 4 within 4.66
eps of the class's `max|X|` (gate 16), every ladder class above it through
(ff|i) within 0.53 eps of the tower-twin scale, an aux block that is a
strict subset of the cells (its slots the host's, every other entry
untouched), and `X` contracted onto the atoms equal to `host_df.
eri3_gradient` (another route on the aux centre). Ladder `--grad-dfx-l
[LMAX] [--grad-dfx-aux-l AUXMAX]` (`--grad-df-l`'s three-center classes
and limits, bare 9 units), never in the bare default nor in
`KERNEL_GEN_CMD`; 18 goldens (`test_emit_grad_dfx.py`). **Compiled by
nvcc and hipcc, no card**: the 9 bare-ladder units and (dd|g), (ff|i),
(sf|i), 24 cases, warning-free (CUDA 13.0 `sm_86`, ROCm 7.2.4 `gfx1200`,
device-less cloud container, 2026-10-30), and the card program's runners
built (`test_xeri3_device.py`'s compile twins); (ff|i) at 13,536 B of
`sm_86` stack (1,564 B spill stores) and 12,672 B of `gfx1200` scratch
(ptxas / hipcc's resource remarks, one compile, not a sweep). Its device
tests (H₂O/STO-3G whole and a strict block; the ladder through (ff|i) per
aux `l`, `high_l`) are written and have not run on a card; their staging,
run through the interpreter in place of the card, matches the host to
1.2e-16 of max. Not measured.
**The DF-K
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

**The kinetic gradient on HGP is generated, interpreted and compiled,
not run on a card** (`gradient.md`, "The kinetic gradient on HGP"):
`spec.GRADIENT_OFF_NATIVE` makes `T` at d1 legal on `Engine.HGP`
(`kinetic/d1/(la,lb)/b1/t.hgp/literal`; `S` is `hgs`, `V` is `ghv`, below),
built by `gradient/kernel_hgt.py` (`hgt_{la}{lb}`, through f,
`L_MAX = 3`): `kernel_hgp`'s VRR per primitive pair into up to six
contracted tables weighted `(2 alpha)^p beta^n`, their HRRs on contracted
values, `kernel_hgp`'s kinetic combination at `a +- 1_k`, contracted with
`M` (`kernel_g1e.weights`' rule), `B`'s row `-A`'s by translational
invariance; one thread per contracted shell pair, six `AtomicAdd`s per
pair; staged by `host.pair_records`. Interpreted
(`test_kernel_hgt.py`): every class (ss)-(ff) on HF/cc-pVTZ against
`host.kinetic_gradient`, worst 37.2 eps of the split-centre scale (gate
64 eps); the whole gradient of H₂O/STO-3G, H₂O/6-31G and
`h2o_sto3g_d_grad` against libcint's `int1e_ipkin` within `1e-12`
absolute. Ladder `--hgp-grad-l [LMAX]` (1 bare, 0..3), never in the bare
default nor `KERNEL_GEN_CMD`; 6 goldens (`hgt_00`, `_01`, `_11` per
backend, `test_emit_hgp_grad.py`). **Compiled by nvcc and hipcc, not
run on a card**: every class through (ff) warning-free under both
(2026-10-30, a device-less cloud container, slowest `hgt_33` 6.3 s /
14.6 s). No driver, no device test, nothing measured; registry
`hgp/t/d1` L 0..6, compile tier only, its cost modelled.
**The exchange gradient on HGP, `ghk`, is generated, interpreted
through `(dd|dd)` and compiled by nvcc and hipcc through `(dd|pp)`; it
has run on no card**
(`gradient.md`, "The exchange gradient on HGP"): `K` at d1 on
`Engine.HGP` (`spec.GRADIENT_OFF_NATIVE`, beside `J` on HGP;
`coulomb/d1/(la,lc|lb,ld)/b1/k.hgp/literal`), `gradient/kernel_ghk.py`'s
`ghk_{la}{lc}_{lb}{ld}`, the same `G` as `gk`. One HGP VRR per primitive
quartet (`vrr_class`), its ket-inner sums folded per bra primitive pair
into three contracted sets (`2 alpha`, `2 gamma`, 1), then the raised and
lowered bra classes' HRRs (`hrr_classes`) and the deposit `W = M_ab M_cd`
into six thread sums, six `AtomicAdd`s a thread; `kernel_hj`'s nest,
staging and mapping imported; one class mode, no lever, no screen.
`gradient.host.hgp_exchange_launches` packs a class's launches by
contraction degree. Interpreted (`test_kernel_ghk.py`) against
`host.exchange_gradient` at `gk`'s gate, 16 eps of the `absolute=True`
twin: every K class through `(dd|dd)` (L + 1 > 4 `slow`), worst 4.4 eps
(`(ss|ss)`); the whole s, p `dE_K`, a mixed-degree basis, 1/2/32 threads;
and `(dd|dd)` on CO/cc-pVTZ against the libcint `k_gradient` witness,
4.99e-16 of `max|G|` (gate 1e-12, `slow`, 73 s). Ladder `--grad-hk-l
[LMAX]` (1 bare, 0..2), never in the bare default nor in
`KERNEL_GEN_CMD`; 32 goldens (`test_emit_grad_hk.py`), the 16 s, p
classes. nvcc and hipcc compile the goldens and five d classes through
`(dd|pp)` (worst `ghk_22_11` 11.6 s / 44 s); **`ghk_22_22` overran a 600 s cap** (the
straight-line unit in ptxas; no rolled form), so it is not in the compile
set; nothing measured, not in
`gradient.rhf`'s runners. Registry: `hgp`/`k`/d1, L 0..8 interpreted,
compile at L 0..4 and 6, no card.
**The overlap gradient is also built on HGP, through (ff), interpreted
and compiled, not run on a card** (`gradient.md`, "The overlap gradient on
HGP"): `S` at d1 on `Engine.HGP` (`spec.GRADIENT_OFF_NATIVE`, beside
HGP `J`, `T`, `K` and `V`; every other target still refuses HGP at
d1), `gradient/kernel_hgs.py`'s `hgs_{la}{lb}`, `gs_`'s contraction on
`hs_`'s recurrence: one overlap VRR to `la + lb + 1` per primitive pair
into three contracted tables (unweighted, `2 alpha`, `2 beta`), each its
own HRR on contracted values (`kernel_hgp.table_dags`), `gs_`'s weights,
six `AtomicAdd`s per shell pair, both centres computed; one thread per
contracted shell pair, staged by `host.pair_records`. Interpreted
(`test_kernel_hgs.py`): every ordered class (ss)–(ff) on HF/cc-pVTZ
against the host and against `gs_` within 64 eps of the split-centre scale
(worst 17.3 at (sd), the rest <= 9.8; not bitwise -- the contracted HRR),
whole H₂O/STO-3G and CO/cc-pVTZ (`slow`). `hgt_`'s ladder `python -m
felizjueves.emit --hgp-grad-l [LMAX]` writes it too (both per class); 6 goldens (`test_emit_hgs.py`), every
unit through (ff) compiled by nvcc and hipcc with no warning
(2026-10-30, a device-less cloud container). Its card cases are
`test_grad_device.py`'s `*-hs` (H₂O/STO-3G, H₂O/cc-pVDZ and
CO/cc-pVTZ with the bare proton, `grad_driver.hgs_launches`, the same
`DEV_TOL`): written, **not run on either card**. Nothing measured.

**The nuclear-attraction gradient on HGP is generated, interpreted and
compiled, its device test written and not run on a card** (`gradient.md`,
"The nuclear-attraction gradient on HGP"): `(V, HGP)` in
`spec.GRADIENT_OFF_NATIVE` (`nuclear/d1/(la,lb)/b1/v.hgp/literal`), built
by `gradient/kernel_ghv.py` (`ghv_{la}{lb}`, through f, `L_MAX = 3`) on
`kernel_hgp`'s `hv_` route: per nucleus the point-charge VRR per primitive
pair to `la + lb + 1`, three contracted tables weighted `2 alpha`,
`2 beta`, 1 and their HRRs (`kernel_hgp.table_dags`), `dA` and `dB` by the
Cartesian rule, the nucleus row `-(dA + dB)` of that nucleus's own tables
(translational invariance, exact per nucleus); one thread per contracted
shell pair, `G` atomic (three per nucleus, six per pair); staged by
`host.pair_records`; no MMD in its ancestry (`ghv_builder`). Interpreted
(`test_kernel_ghv.py`): every class (ss)-(ff) on HF/cc-pVTZ and the s, p
classes on a synthetic off-axis basis with a bare fourth nucleus, against
`host.potential_gradient`, worst 6.0 eps of the split-centre scale (gate
16 eps); whole H₂O/STO-3G and 6-31G against PySCF's `potential` term,
7.9e-14 / 4.9e-14 absolute (gate `1e-10`). On the `--hgp-grad-l` ladder
beside `hgt_` and `hgs_` (class-major), 6 goldens (`test_emit_ghv.py`).
**Compiled by nvcc and hipcc**: every class through (ff) warning-free, and
the synthetic program (2026-10-30, a device-less cloud container; `ghv_33`
2.5 s / 7.3 s, 6,104 B `sm_86` stack / 4,708 B `gfx1200` scratch -- one
unrolled form, a reading, not a measured row). **`test_ghv_device.py`
(generator `test/ghv_driver.py`; s and p: the synthetic basis, H₂O/STO-3G
and 6-31G) has run on neither card.** Nothing measured; registry
`hgp/v/d1` L 0..6 compile, L 0..2 `gpu`, no card, its cost modelled.

**The fused Fock gradient on HGP is generated, interpreted and compiled, not run on a card**
(`gradient.md`, "The fused Fock gradient on HGP"): `Target.F` joined
`spec.GRADIENT_TARGETS`, on its native HGP engine
(`coulomb/d1/(la,lb|lc,ld)/b1/f/literal`). `gradient/kernel_gf.py`'s
`gf_{la}{lb}{lc}{ld}` writes `dE_J - 1/2 dE_K` (`E_2e = SUM Gam (ab|cd)`,
`Gam = 1/2 D_ab D_cd - 1/4 D_ac D_bd`) in one kernel per F class on
`kernel_f`'s mapping, canonical half and degeneracy factor, the weights
DERIVED from `kernel_f.orbit_group()` (`w = deg (4 D_ab D_cd - D_ac D_bd
- D_ad D_bc)`), all four centres differentiated (no translational
invariance used): one VRR to `L + 1` over the eight derivative classes'
boundaries, five contracted families (unweighted, and `2 alpha` ..
`2 delta` from two exponent fields added to `kernel_f`'s record), one HRR
per derivative class, 12 `AtomicAdd`s, no screen, no lever. The stream
(`gradient.interpreted.gf_launches`, nothing pruned) and `gf_gradient` /
`gf_run` run it interpreted. Held (`test_kernel_gf.py`) to `eri.eri`'s
derivative integrals per class, all 21 ladder classes through `(dd|dd)`
as launched and transposed (1.4e-15 of the abs-value twin at L <= 6, gate
1e-13); to the host's `coulomb_gradient - 1/2 exchange_gradient` on
synthetic bases where `a = b`, `c = d` and `P = Q` all occur (4.8e-15 of
`max|G|`); and to PySCF's `grad:j + grad:k` on H₂O/STO-3G and
`h2o_sto3g_d_grad` (2.2e-14 / 2.1e-14 absolute, gate 1e-10); zero net
force a check. Fast: L + 1 <= 4 and the s-only basis; 26 cases `slow`.
Ladder `--grad-f-l [LMAX]` (1 bare, 0..2), never in the bare default nor
in `KERNEL_GEN_CMD`; one text golden (`gf_0000`, both vendors) and 20
digests (`test/golden/gf_digests.json`, `test_emit_grad_f.py`); `(dd|dd)`
is 11 MB of CUDA. **Compiled** through `(pp|pp)` (the six bare-ladder
units, nvcc `sm_86` and hipcc `gfx1200`, no warning in generated text, 12
passed, 2026-10-30, `test_emit_grad_f.py`); the d classes are not
compiled. **Never on a card, nothing measured**; no low-stack form; not
in `rhf.py`'s `Runners`.

**RHF nuclear-gradient fixtures (#550, schema 5): tracked, consumed by
nothing yet.** `test/oracle/fixtures/<m>_grad.json` beside the H2O
`_j` fixtures (STO-3G, 6-31G; cc-pVDZ is for `slow` tests only) plus a C1
NH2F/6-31G with no `_j` sibling: the converted basis held to the sibling's
exactly, PySCF's converged RHF (`mo_*` recorded; `density`, `W` derived),
libcint's `int1e_ipovlp`/`ipkin`/`ipnuc`, `int1e_iprinv` per nucleus,
`grad.rhf.get_jk`'s `vj1`/`vk1`, and the per-atom `h`, `S`, `J`, `K`, `V_nn`
terms with `nuc_grad_method().kernel()`'s total. `manifest.gradient_fixture`
recomputes each term from the stored arrays (~3e-15, `GRADIENT_TOL` 1e-12),
sums them to the total, and checks zero net force (structural only: it cannot
see a wrong factor). An `oracle` test holds H2O/STO-3G's total to a central
difference of RHF energies. The sign conventions are `manifest`'s docstring.
No card was involved.

**`dE_J/dR` on the HGP recurrence: generated and interpreted only**
(`gradient.md`, "The Coulomb gradient on the HGP recurrence"):
`Spec(COULOMB, 1, (la, lb, lc, ld), Target.J, engine=Engine.HGP)`, legal
through `spec.GRADIENT_OFF_NATIVE` (one of its five off-native d1 pairs, with `K`, `T`, `S` and `V`; MMD `J`
at d1 stays refused), key `coulomb/d1/(la,lb|lc,ld)/b1/j.hgp/literal`.
`gradient/kernel_ghj.py`'s `ghj_{la}{lb}{lc}{ld}`, one unit per ORDERED
class (no canonical mode, no lever, no screen): `kernel_hj`'s nest and
record, one VRR one Boys order up into three contractions (plain, `2
alpha`, `2 beta` per primitive pair, exact because eq 28 is linear), up
to four HRRs (`a +- 1`, `b +- 1`), the bra derivative on `a`'s and `b`'s
atoms by six `AtomicAdd`s. `gradient/interpreted.py`'s
`ghj_class_gradient` / `ghj_gradient` pack and launch it;
`interpreted_runners(coulomb="hgp")` puts it in the RHF assembly.
Interpreted (`test_kernel_ghj.py`): every ordered class with each
`l <= 2`, summed per unordered group, against `host.coulomb_gradient`
restricted to it within `1e-12` of the per-shell absolute scale; the
whole s, p pair against the host; and the Coulomb term of H₂O/STO-3G,
H₂O/STO-3G plus a d primitive and H₂O/6-31G against PySCF's `grad:j`
within `1e-10` absolute (raised `L + 1 > 4` and the whole cases `slow`).
**Not emitted** (no ladder flag, no golden), **not compiled**, **not on
a card**, not measured.
