# Schwarz, the cross-basis overlap and density fitting

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

**The Schwarz matrix `Q[m][n] = sqrt((mn|mn))` is generated, as standalone
output** (nothing screens on it yet): `Target.SCHWARZ` (`spec.py`; `am` a
pair `(la, lb)`, unbatched by decision) whose kernel (`eri/kernel_schwarz.py`)
runs the `Target.J` quartet `(la lb|la lb)`'s plan at `L = 2 (la + lb)`, its
dead targets dropped, one thread per contracted shell pair over both record
loops, the `E` block inlined twice, a clamp before the root and a plain
mirrored store (no atomics, bit-reproducible). Host side, `eri/schwarz.py` is
the reference (the diagonal of `eri.eri`) and the staging (`schwarz_classes`,
every pair `la <= lb`). The ladder is `python -m felizjueves.emit
--schwarz-l [LMAX]` (`schwarz_{la}{lb}`, 0..4, (gg) at L = 16), never in the
bare default and not in `KERNEL_GEN_CMD`. **Correctness is tested through
L <= 4 only** ((ss), (sp), (pp), (sd)), by the user's decision: the host
against libcint's `int2e_diag` (`test_ao_schwarz.py`, which also checks
`|(ab|cd)| <= Q_ab Q_cd` on every tracked libcint block at L <= 4), the
kernel interpreted (`test_kernel_schwarz.py`), and on both cards
(`test_schwarz_device.py`, generator `test/schwarz_driver.py`;
on the shared runner since #658, which ran on both cards on 2026-10-30). Nothing about
it is measured.
**The cross-basis overlap `S_{B1,B2}[m][n] = <m|n>` is generated too**
(`one_electron.md`, "The cross-basis overlap"; nothing consumes it yet, no
projection is built): `Target.SX` (`spec.py`; `am` an **ordered** pair
`(la, lb)`, `la` the row basis's shell, operator `OVERLAP`, unbatched for
`S`'s structural reason) whose kernel (`one_electron/kernel_sx.py`) reuses
`kernel_s.record_values` unchanged (`S`'s text byte-identical) with a
rectangular scatter, one `AtomicAdd` per element, no mirror, no diagonal
rule. Host side, `one_electron.overlap.cross_overlap_matrix` over
`one_electron.one_electron.rectangular` (every ordered shell pair), the screen
`ao.pairs.cross_pair_list`, and the launch `launch.cross_overlap_launch` →
`XLaunch` of `XClass` over `XRec` with `walk_cross_overlap` (a sibling of
`OneELaunch`). The ladder is `python -m felizjueves.emit --sx-l [LMAX]`
(`sx_{la}{lb}`, every ordered class, 0..8), never in the bare default and
not in `KERNEL_GEN_CMD`. Witnessed against libcint's `gto.intor_cross`
through three tracked fixtures (`h2o_sto3g_x_631g`, `_x_ccpvdz`,
`_x_631g_moved`; each array `int1e_ovlp_cross:<rows>` names its row
fixture, `oracle.manifest.cross_overlap`), both orientations
(`test_ao_cross_overlap.py`); the kernel interpreted bitwise
(`test_kernel_sx.py`, (ss) through (ff)); and on both cards
(`test_sx_device.py`, generator `test/sx_driver.py`;
on the shared runner since #658, which ran on both cards on 2026-10-30). Nothing about it is
measured.
**The DF Coulomb metric `(P|Q)` of a Cartesian auxiliary basis is generated
too, and ran on both cards** (the RTX 3080 first, both since #606; milestone 15, #410–#416;
`docs/implementation/metric.md` is the one place with the detail; nothing
consumes it yet, DF-J and DF-K are out of scope): `Target.METRIC` (`spec.py`;
`am` a pair `(lP, lQ)`, operator `COULOMB`, unbatched by decision, no lever)
whose kernel (`eri/density_fit/kernel_metric.py`) runs the `Target.J` quartet `(lP 0|lQ 0)`'s
plan at `L = lP + lQ` (the Schwarz pattern), its parity-dead targets dropped,
one thread per aux shell pair over both record ranges,
`e_fragment.center_fragment` (the single-center `E`, `X_PA` absent, seed 1)
inlined twice, one ket sign per class, and a plain mirrored store (no
atomics, bit-reproducible). The aux record weight is `s = c·(π/α)^{3/2}`,
with **no** extra `N` (`c` already carries libcint's normalisation); `(2/√π)θ`
is the prelude's `W0`. Host side, `eri/density_fit/metric.py` is the reference and the
staging (`metric_classes`, every aux pair `P <= Q`, a swap flagged), and
`eri/density_fit/metric_drive.py` the interpreted adapter. The ladder is `python -m
felizjueves.emit --metric-l [LMAX]` (`metric_{lP}{lQ}` for `lP <= lQ`,
`lP + lQ <= LMAX`, 4 bare, any `LMAX >= 0`), never in the bare default and not
in `KERNEL_GEN_CMD`. The oracle is libcint's `int2c2e_cart` on
`df.make_auxmol(mol, 'cc-pvtz-jkfit')` (45 shells, 166 functions, O to g, H
to f), tracked as four schema-3 row pieces (#411). **Correctness is tested at
`lP + lQ <= 4` only** (the nine classes (s|s) through (d|d); the builder and
emitter accept any L, and L = 5 is only built and emitted, `slow`), against
the host within 16 eps of the **tower's** absolute-value twin
(`test_kernel_metric.pq_tower`; a per-target scale was tried and found
unsound) and against libcint within `1e-12·max|M|`: the host
(`test_ao_metric.py`), the kernel interpreted (`test_kernel_metric.py`), the
nine goldens per backend (`test_emit_metric.py`), and on the RTX 3080
(`test_metric_device.py`, generator `test/metric_driver.py` -- on the
shared runner since #654, run on both cards 2026-10-30 -- PR #430: worst
2.61e-14 of `max|M|` against libcint, 7.17 eps of the tower-twin scale
against the host). **It has not run on the RX 9060 XT**: the card was not
enumerated in the container, so on `gfx1200` it is compiled only; that run,
and a fix to `kernel-tier.sh`'s HIP device check (it tests `/dev/kfd` only
and printed `device: present` with no card), were follow-up #431: the
check was fixed when #431 closed (2026-10-17), and the card run came with
#606 (PR #615, Parts 1 and 2 below).
Nothing about it is measured or scored (no published board; a
`StructuralLimit`, #417); `(mn|P)` was its named follow-up, built next.
**The three-center DF integral `(mn|P)` is generated too, and ran on
both cards** (the RTX 3080 first, both since #606; milestone 16, #441–#446; `docs/implementation/three_center.md`
is the one place with the detail; nothing consumes it yet, DF-J and DF-K are
out of scope): `Target.ERI3` (`spec.py`; `am` the **triple** `(la, lb, lP)`,
the first three-shell class, operator `COULOMB`, unbatched by decision, no
lever) whose kernel (`eri/density_fit/kernel_eri3.py`) runs the `Target.J` quartet
`(la lb|lP 0)`'s plan at `L = la + lb + lP`, its dead targets dropped by the
parity rule, one thread per cell (an AO shell pair crossed with an aux
shell), `e_fragment.block_fragment` (the bra) and `center_fragment` (the
ket) both inlined, one ket sign per class, and a plain
store mirrored in `(m, n)` into `[m][n][P]`, aux fastest (no atomics,
bit-reproducible). The bra is `ao.pairs`' dressed record (`K_AB` in `E`'s
seed, not in `dress`), the ket the metric's aux record unchanged. Host side,
`eri/density_fit/eri3.py` is the reference and the staging (`eri3_classes`, every AO pair
`i <= j` × every aux shell, `la <= lb`, a swap flagged), and
`eri/density_fit/eri3_drive.py` the interpreted adapter; `eri.eri`'s exponent-0 internal
twin is not built, because `ao.shell.Shell` refuses exponent 0 (a test pins
the refusal). The ladder is `python -m felizjueves.emit --eri3-l [LMAX]`
(`eri3_{la}{lb}{lP}` for `la <= lb`, `la + lb + lP <= LMAX`, 4 bare, any
`LMAX >= 0`; `kernel_eri3.eri3_name` `_`-separates the `l`s once any is
above 9, so names are unique at any L), never in the bare default and not in
`KERNEL_GEN_CMD`. The oracle is libcint's `int3c2e_cart` (`aosym="s1"`) on
`h2o_sto3g_j`'s molecule (7 AOs) against `df.make_auxmol(mol,
'cc-pvtz-jkfit')` (166 functions), one schema-4 piece (#442), whose
`(m, n)` symmetry is asserted to `1e-14·max`, not bitwise (libcint's `s1`
tensor differs from its transpose in 1,998 of 8,134 entries, worst 4.4e-16).
**Correctness is tested at `la + lb + lP <= 4` only** (the 22 classes
(ss|s) through (dd|s)), against the host within 16 eps of the tower twin
(no class is bitwise: the association differs) and against libcint within
`1e-12·max|(mn|P)|`: the host (`test_ao_eri3.py`, worst 3.3e-14), the
kernel interpreted (`test_kernel_eri3.py`, worst 1.80 eps), the 22 goldens
per backend (`test_emit_eri3.py`, no digests file), and on the RTX 3080
(`test_eri3_device.py`, generator `test/eri3_driver.py` -- on the shared
runner since #657, which has not run on a card -- PR #457: worst
3.30e-14 of `max|(mn|P)|` against libcint, 4.77 eps of the tower-twin scale
against the host on the fixture and 4.66 on the synthetic bases; the
fixture's (sp|g), (pp|f) and (pp|g) are not launched). **It has not run on
the RX 9060 XT**: kfd did not enumerate the card (`gfx_target_version` 0),
so on `gfx1200` it was compiled only, and by the user's decision that run
was attached to #431; it came with #606 (PR #615, Parts 1 and 2 below). Nothing about it is measured or scored (no published
board; a `StructuralLimit`, #447); no screen (`(mn|P)` has no Schwarz bound
yet), nothing above L = 4 against libcint, Cartesian only.
**Both DF kernels have a low-stack form, for the aux ladder through i**
(#606; `three_center.md` / `metric.md`, "The low-stack form"):
`build_eri3_kernel(cls)` / `build_metric_kernel(cls)` pick it when
`eri3_lowstack` / `metric_lowstack`: `L >= 10`, or unrolled `E` blocks
over `E_DOUBLES_MAX`. eri3's bound is (pf|g)'s 1,575 doubles, set from
the measurement, and the metric keeps its own 1,785. On `la <= lb <= 3`,
`lP <= 6` that is 25 eri3 classes: every (df|.), (ff|.) and aux-i row,
and (dd|g). On `lP <= lQ <= 6` it is 8 metric classes: every i row and
(h|h). None is at `L <= 4`, so every golden is unchanged. `lowstack=`
forces either form, and the name, parameters, launch and output are the
same. The form has per-axis `E` tables for both sides
(`e_fragment.center_tables_fragment`), the tower's `[r]^(0)` stored into
a local `R` (`recurrence.store_targets`), and the contraction as loops
through `U[t]` of one aux row. The elements accumulate in place in the
thread's own output slots: zeroed first, mirrored last, no atomics.
**Measured** in `data/measured/df_lowstack.json` (`measure/
measure_df_lowstack.py`; `devtools/measure.sh df-lowstack-compile` /
`df-lowstack`, the rule `declared` before any row). The compile is 115
units per arch, each alone under a 300 s budget, beyond which a unit is
killed and recorded `over_budget`.
- Every low-stack unit is at most 8,256 B of `sm_86` stack (8,512 B of
  `gfx1200` scratch) and compiles in at most 5.1 s.
- Before / after, `sm_86`: (df|g) unrolled 33,992 B in 52 s against
  4,608 B in 2.0 s. (ff|g) unrolled is **over the 300 s budget** against
  6,320 B in 2.6 s; on `gfx1200` it compiles in 242 s with 34,660 B of
  scratch.
- **Decided:** the unrolled (dd|g) (15,992 B on `sm_86`) is over the
  rule's 14,000 B, so it moved under the gate. (pf|g), 13,928 B, is the
  largest class kept unrolled.
- **The clock** (behind the gate): the low-stack form is **slower**
  wherever both compile, 1.46x to 4.58x on both cards. (dd|g) costs 3.67x
  on the RTX 3080 and 3.36x on the RX 9060 XT. So the clock moves nothing
  else under the gate; `measure_df_lowstack.test_the_gate_is_what_the_
  measured_rows_rule` holds the gate to the rows.
- **Against libcint through (ff|i) / (i|i)** (Cartesian `int3c2e_cart` /
  `int2c2e_cart`, not spherical): the tracked aux-i sets
  `prim_spd_f_auxi_3c` (5 schema-4 pieces) and `prim_spd_f_auxi_metric`
  (4 schema-3 pieces), on the AO parent `prim_spd_f`. That is He s, p, d
  and a contracted Ne f, against l = 0..6 aux shells on both atoms, 168
  functions, and reaches every ladder class.
  - The host is within 1.16e-14 / 2.76e-14 of max, and the gated classes
    interpreted within 4.6e-15 (`test_kernel_df_lowstack.py`, `slow`).
  - **On both cards** (`test_eri3_device.py` / `test_metric_device.py`
    Part 3, `slow` + `high_l`): all 70 + 28 classes in their default
    forms. Worst 2.75 / 2.46 eps (eri3) and 3.71 eps (metric) of the
    tower twin against the host, and 1.19e-14 / 2.76e-14 of max against
    libcint, on the RTX 3080 / RX 9060 XT.
  - Those sessions also ran Parts 1 and 2, the first RX 9060 XT run of
    either DF kernel: 10 passed per card.
**Both DF targets also have an HGP twin, for s and p shells, interpreted
only** (`docs/implementation/hgp_df.md` is the one place with the detail):
`eri/density_fit/kernel_hgp.py`'s `build_hmetric_kernel` (`hmetric_{lP}{lQ}`)
and `build_heri3_kernel` (`heri3_{la}{lb}{lP}`), keyed `Target.METRIC` /
`Target.ERI3` with `engine=Engine.HGP` (slots `metric.hgp` / `eri3.hgp`;
`spec.py` admits HGP there beside `J` and `K`). Each is a drop-in sibling of
its MMD kernel (same parameters, staging, thread mapping and plain mirrored
stores; the adapters and harnesses take `engine=`): an aux Gaussian is an
HGP pair with an exponent-0 partner, so `(P|Q)` is the VRR of `(lP 0|lQ 0)`
alone and `(mn|P)` the VRR of `(max, min|lP 0)` then the bra HRR, the zero
`PA` / `QC` terms not spelled. Ladders `--hgp-metric-l` / `--hgp-eri3-l`
(LMAX per shell, 0..1 only), never in the bare default nor in
`KERNEL_GEN_CMD`. Interpreted against the hosts (worst 3.20 eps of the tower
twin), the MMD kernels and libcint (`test_kernel_hgp_df.py`), nine goldens
per backend (`test_emit_hgp_df.py`); **compiled and run on the RTX 3080
only** (`test_hgp_df_device.py`, 8 passed: worst 6.06 eps of the tower twin
against the host, 2.1e-14 of max against libcint, runs bit-identical); the
RX 9060 XT is not run (no open issue tracks it; #431 closed on the card's detection fix, not the run). Nothing measured.
**The DF metric also has a Rys twin, through (g|h), interpreted and
compiled, NOT run on a card** (`docs/implementation/rys_fock.md`, "The DF
metric on Rys", is the one place with the detail):
`eri/density_fit/kernel_rmetric.py`'s `build_rmetric_kernel`
(`rmetric_{lP}{lQ}`), keyed `Target.METRIC` with `engine=Engine.RYS` (slot
`metric.rys`, no lever), a drop-in sibling of `metric_` / `hmetric_` that
fills `kernel_hgp.metric_kernel`'s skeleton (extracted from
`build_hmetric_kernel`, every `hmetric_` golden byte-identical) with the
`N = (lP + lQ)//2 + 1` point Rys rule (`ir.Rys`, `fj_rys<N>`) and
`rys.recurrence.root_loop(..., single_centre=True)` per record pair, `P - A
= Q - C = 0` not spelled (every other Rys golden byte-identical). Reach
`lP + lQ <= 9` (`N <= 5`). `metric_drive` takes `engine=Engine.RYS` with the
rule injected (`rys=`). Ladder `--rys-metric-l [LMAX]` (total L, 0..9, bare
4), never in the bare default nor `KERNEL_GEN_CMD`. Interpreted
(`test_kernel_rmetric.py`, `L > 4` `slow`): every class through (g|h)
within 3.69 eps of the tower twin against the host and 4.26 against the
MMD kernel (and the HGP twin through p), the whole `cc-pvtz-jkfit` set
within 2.6e-14 of `max|M|` against libcint. Nine goldens per backend
(`test_emit_rmetric.py`); **all 30 units compiled warning-free by nvcc
(sm_86) and hipcc (gfx1200)** on a device-less cloud box, 2026-10-30
(60 passed). **Not run on either card**; nothing measured.
