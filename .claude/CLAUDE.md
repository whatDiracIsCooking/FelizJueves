# CLAUDE.md

## What this is

`FelizJueves` — **Python code generation of GPU kernels** for
high-angular-momentum electron repulsion integrals (ERIs) and Coulomb-matrix
(`J`) construction: the McMurchie-Davidson recurrences and sub-batching of
Palethorpe & Barca, emitted as CUDA and HIP source rather than written by hand.

**Why a generator.** One kernel per `(l₁l₂|l₃l₄)` class, and the recurrence
tree, the register and shared-memory budget, and the chosen sub-batch split are
all functions of that class. The reviewable artefact is the thing that emits the
kernels, not the hundreds of kernels — so the generator is the product, and a
kernel is its output.

**Both vendors, from one source.** This box has an RTX 3080 (sm_86, CUDA 13.0)
and an RX 9060 XT (gfx1200, RDNA4, ROCm 7.2.4), and a generated kernel has to
satisfy **both** device compilers. `hipcc` is clang and `nvcc` is not, so they
disagree about more than their flags; that disagreement is this project's main
portability risk, and `.devcontainer/combined/` exists to put both in one shell.

`docs/references.md` is the citation of record: a flat numbered list, every
entry resolving by **DOI** and no PDF tracked anywhere in the repo. Cite by
number, or by the short key where one is defined — `[Fock]` is entry 23 and
`[JEngine]` is entry 35. A version inside a key, as in `[JEngine, v2, eq. 14]`,
means the ChemRxiv v2 preprint's equation numbering and not the journal
version's, which may renumber.

## The state of the tree — read this before trusting any tier

**The generator is live for the whole `J` chain** (milestone 7): every
arithmetic step of `J_µν` is a generated kernel. They are the `[p\|q]` kernel
(`Target.J`; one batch, and, through `build_pq_kernels`, a split class
(`n_batches > 1`) as atomic batch kernels), the ket pre-contraction
(`Target.DQ`, `eri/j_coulomb/kernel_dq.py`: `D_q` and `Dmax`) and the bra back-transform
and scatter (`Target.JAB`, `eri/j_coulomb/kernel_jab.py`: `J_ab` into `J_µν`), with the
undressed `E` transform (`mmd/e_fragment.py`, an `ir.Fragment`) inlined into the
last two. `DQ` and `JAB` are this repo's own design: `[JEngine]` defers those
steps to `[Fock]`, which was not read for them (`limits.json`, a record
from before milestone 12 read `[Fock]` §3 for the HGP engine). **Around them,
`J_µν` for a real molecule is also built host-side** (`mmd/hermite.py`, `ao/pairs.py`,
`ao/density.py` and `eri/j_coulomb/`: the `E` transform, pair data, the ket pre-contraction, the bra back-transform
and the scatter, all pure Python). That path is the reference each kernel is
held to, bitwise when interpreted, and it is **witnessed against
PySCF/libcint** (milestone 4, `j_engine.md`, "The Coulomb matrix, witnessed
from outside"). **Since milestone 6 a whole molecule's `J` is built from a
launch plan** (`eri/j_coulomb/launch.py`: one `ClassLaunch` per `(lp, lq)`, bra records in
pair order, kets sorted by `I_q · max|D_q|`). On each card, every bra and ket
class is uploaded once, and one `J_p` buffer per bra class takes every batch
kernel through fp64 atomics. Since milestone 7, `DQ` produces the kets'
`D_q` on the card and `JAB` turns each `J_p` buffer into `J_µν` there (#157:
pair kernels through (ff), `J` through d, on both cards). The plan's order is
witnessed by the launch oracle (pure Python) and by libcint (`j_engine.md`,
"The launch plan"). **Still host-side:** the pair list and each record's
inputs, the Schwarz bound, the sort (it reads `Dmax` back), the staging, and
the launch code, a test-side driver run from Python. The record bound is now
also generated, for a consumer whose records are on the device:
`Target.PAIR_BOUND` (`eri/kernel_pair_bound.py`; `am` a pair `(la, lb)`, text a
function of `la + lb` only, never batched), one thread per record, the
closed form of `ao.pairs.pair_bound` at `PQ = 0` (two `SQRT`s and a Horner
fold in `sqrt(ζ)`), within 16 eps of the host through `reference.pq` for
every class `la ≤ lb ≤ 4`, interpreted (`test_kernel_pair_bound.py`) and on
the RTX 3080 (`test_pair_bound_device.py`; the RX 9060 XT run is #431's).
That is J's undressed bound; `K`'s screen is proved against the dressed one,
`eri.k_exchange.k.record_bounds`, generated as `Target.RECORD_BOUND`
(`eri/k_exchange/kernel_record_bound.py`, #499; `am` an **ordered** pair `(la, lb)`, the max
over the first shell's Cartesians, never batched): one thread per record,
`e_fragment.block_fragment` inlined and each `I_p` in the same closed form
(`kernel_pair_bound.hermite_bound_coefficients`), within 16 eps of the host
for every ordered class `la, lb ≤ 3` and every record of H₂O/cc-pVDZ and
the H₁₀ chain, interpreted (`test_kernel_record_bound.py`) and on the RX
9060 XT (`test_record_bound_device.py`; the RTX 3080 was down in that
session, so on CUDA it is compiled only). **Nothing about `DQ` or
`JAB` is measured.** `j_engine.md`, "The J chain as device code", is the one
place that says which steps are generated and on which cards they ran.
Dispatch (`performance/dispatch.py`, #145) chooses `n_batches` per class from measured
wall-clock for a launch plan, but the generator does not follow it:
`Spec.n_batches` defaults to 1. Not built, by decision: `J` resident across
SCF iterations, a `Launch` IR, a `LaunchCost` model (`limits.json`).
Against the predecessor `jgen`, this is parity on its energy-order `J`
surface, with one difference: Table 4, declined. Which plan ships is a
decision, not a gap: `jgen` ships its searched sieve order, and this repo has
that wire too (the opt-in `tuned` lever, #161) but ships the **literal**
plan, because the searched one was measured on both cards and refused on
wall-clock (#166; `j_engine.md`, "Parity with the predecessor").
**Since milestone 11 the one-electron matrices are generated too**
(`docs/implementation/one_electron.md` is the one place with the detail): the
overlap `S` (`Target.S`, `one_electron/kernel_s.py`), the kinetic energy `T` (`Target.T`,
`one_electron/kernel_t.py`) and the nuclear attraction `V` (`Target.V`, `one_electron/kernel_v.py`), one
thread per primitive-pair record on the same `E` fragment, scattered by
`kernel_jab`'s rule; `V` is the `J` engine with the ket replaced by a unit
point charge (`θ → √ζ`), its recurrence and plan layer reused unchanged. Their
host references are `one_electron/overlap.py`, `one_electron/kinetic.py` and `one_electron/potential.py`,
witnessed against libcint's `int1e_*` and PySCF's `hcore` through the tracked
fixtures; on both cards every pair class ran through (ff) against the host,
`S`, `T` and `V` of H₂O through d against libcint, and `h = T + V` with
`Tr(D h)` against `hcore` (#242–#244). **Nothing one-electron is scored** —
`[JEngine]` is J only, so there is no published board, and no cell or claim
was added (`limits.json`) — **and nothing one-electron is measured.** Energy
order only; `V` is one batch by decision. Every kernel-tier run of milestone
11 was capped at L ≤ 8 by the user, and no `J` text changed, so the L = 16
tier was not re-run.
**Milestone 10, the exchange matrix `K`, is partly built and still open**
(#205, #207, #208 and #231 are open; no doc has a `K` section yet, #208).
Built: `Target.K` (`spec.py`, #196; `am = (la, lc, lb, ld)`, the two Hermite
pairs bra first, not the AO quartet order), the host `K` by two sweeps with
its Schwarz bound and screen (`eri/k_exchange/k.py`, #197, #200), the kernel
(`eri/k_exchange/kernel_k.py`, #198: `[p\|q]`'s recurrence and plan unchanged, both `E`
fragments inlined, the contraction staged so neither `(ac\|bd)` nor its
half-transform is an array, one warp per bra record as in `J`) and its split
class as atomic partial-sum batch kernels (#203), the launch plan
(`eri/k_exchange/launch_k.py`, #204: a sibling of `ClassLaunch`, which is J-shaped, its kets
sorted by `I_bd` alone because J's density key cannot be formed for `K`;
since #480 also an opt-in bra–ket-symmetric sweep, `sweep="symmetric"`: per
ket group `b` the bras cut to the shell prefix `a <= b`, the `a < b` blocks
mirrored, walked interpreted only, the ordered default pinned by a
fingerprint), the
duplicated-work count (`eri/k_exchange/kdup.py`, #202) and the libcint `vk` fixtures (#199).
On both cards, `K_mn` from the launch plan ran **only for classes with
L ≤ 2** (#206: H₂, two H₂, H₂O/STO-3G's s shells and its L ≤ 2 classes,
against the host paths and, for H₂ whole and H₂O's s-s block, libcint, worst
`3.23e-15·scale_K`); one cell above that, `(dd\|dd)` at L = 8 and one batch,
matched the host on the RTX 3080 only (#259). Every other `K` test above
L = 2, host-side or on a card, the whole H₂O/STO-3G `K` among them, is
written `slow` and **not yet run** (#231). **Nothing about `K` is timed or scored:**
`kernels.json`'s 20 `K` rows are compile-only (#205, partial), and the three
`K` claims are pending (#207). `KERNEL_GEN_CMD` writes no `K` unit.
**Beside `kernel_k`, a block exchange kernel on opt-UM's mapping** (#481;
`[Fock, §4.3]`): `eri/k_exchange/kernel_kblock.py` (`build_kblock_kernel`, `kb_{la}{lc}_{lb}{ld}`,
one batch and no lever, both refused) runs `kernel_k`'s arithmetic unchanged
(`kernel_k.ket_body` / `bra_scalars`, extracted with its text byte-identical,
and the K seams' default hooks) in another loop nest: **one block (one warp)
per contracted `K_AB` block**, every lane walking the block's bra range,
the lanes taking its kets, one accumulator zeroed once, one warp reduction,
and lane 0 adding the block straight into `K[n_ao, n_ao]` at AO offsets: no
per-record scratch, no scatter, `n_a n_b` atomics per block that never meet
inside a launch, so a build is bit-reproducible. A sibling, not a lever: the
seams fix the launch shape, the parameters and the output, which this
changes. Its plan is `eri/k_exchange/launch_kblock.py` (`kb_launches` → `KBLaunchPlan` of
`KBClassLaunch` with one `KBlock` per `(a, b)`; `k_launches`' records
regrouped, bras sorted by `I_ac` within their first shell) and `walk_kb`,
whose screen (a bra-loop `break` on `Ib · Iq[ket0] · dmax[blk]` beside the
per-pair one) skips, with every block's `dmax = max|D|` (the default),
**exactly** `walk_k`'s pairs, so its `K` is `walk_k`'s to 0. **Since #484
each block carries its own density factor** `dmax[blk]` (a seventh
block-table field; `maxD` left the parameter list): `[Fock, §4.3.1]`'s
`F_ab`, from `eri.k_exchange.k.density_decay` (`R` per shell and partner `l` cut by the
block kernel's own exit test, `Λ` a clamped least-squares fit of the binned
off-atom `|D|`, `H_off` an exact reduction, `F_ab = max|D|` when
`|AB| − R_a − R_b ≤ 0`), passed as `kb_launches(decay=)`, also taken by
`exchange_matrix_screened(decay=)`. K stays at `τ`, never `τ*`, and every
skipped pair's own bound `I_ac I_bd max|D_cd|` is under `τ`
(`test_ao_k_decay.py`, no tolerance, on a decaying six-H₂ chain with a
synthetic density, where it skips 18,154 of 20,736 record pairs against
17,980; `F_ab = max|D|` exactly on H₂O/STO-3G at every `τ`). Dropping whole
blocks from the launch (a launch filter) is not built. Interpreted, one bra
record per block is `kernel_k`'s row bitwise for every class through
`(dd|dd)`; `test/kblock_driver.py` runs either kernel on one upload, and on
the RTX 3080 both matched `walk_kb` and libcint's `vk` on H₂O/STO-3G (τ 0 and
1e-8), two H₂ and H₂O/cc-pVDZ (`slow`), and since #484 the decaying chain
with its per-block factors (against `walk_kb`, 2.60e-16 of max); the RX
9060 XT is compile-only (#431). **Nothing about it is measured yet:** `devtools/measure.sh kblock`
would time both per class on H₂O/cc-pVDZ into `data/measured/kblock.json`,
and its one attempt was refused by the card gate (the RTX 3080 busy from
outside the container), so that file does not exist.
**Since milestone 12 the repo has a second ERI engine and the first `F`**
(`docs/implementation/hgp_fock.md` is the one place with the detail): the
Head-Gordon–Pople recurrence of `[Fock]` §3.1 and the **fused** Fock kernel
of §3.2 on it, `Target.F` (`spec.py`, #272; `am` in AO order
`(la, lb | lc, ld)`, the opposite of `K`'s Hermite pairing; `n_batches = 1`
by its own refusal; the plan layer refuses it). The recurrence is `hgp/`:
the VRR (eqs 25–26, `vrr.py`, a DAG **not graded by `m`**), the HRR (eq 28,
`hrr.py`, on contracted values), the FLOP count (`cost.py`, calibrated
against nothing) and both as IR (`mmd/recurrence.py`, after #277's signed
`Term.c` and two derived `Factor`s). The kernel (`eri/fock/kernel_f.py`, #282, #283)
uses §3.2's mapping: one block per bra pair, one thread per contracted
quartet, the `K⁴` contraction **inside** the kernel because eq 28 needs
contracted values (961 boundary accumulators at `(dd\|dd)`, the bounded
`hgp.vrr.target_set`, not the issues' 1,225), a six-block screen that
**skips, never breaks**, and the 8-fold digestion into `G = J − K/2` with
its weights **derived** (Coulomb 2, exchange −½, `deg = |orbit|/8`) and
held to `eri/fock/fock.py`'s every-ordered-quartet sweep. Around it:
`eri/hgp_drive.py` (HGP ERIs interpreted, pinned to `eri/eri.py` in 21
classes, #281), `eri/fock/launch_f.py` (the pair stream: pruning, degree-uniform
batches, eq. 29's screen in its sound reading, #285), `eri/fock/walk_f.py` (its
oracle, #286), `ir/lower.py`'s loop nest (#284) and `engine.py` (#292, the
`Engine` protocol extracted last, its docstring listing what was pushed back
down into MMD). On both cards every one of the **21 classes** (every shell
`l ≤ 2`, through `(dd\|dd)`) matched the sweep (worst `2.2e-14·scale_G`
against `1e-13`), and the whole `F` of H₂O/STO-3G, 6-31G and cc-pVDZ matched
libcint's `vj − vk/2` (worst 7.35e-15 against `1e-12`), #288 and #290; all
42 F rows are in `kernels.json` with gated wall-clock (`(dd\|dd)`: 255
registers and 97,464 B of spill stores, 2.33 s per launch on `sm_86`; 192
VGPRs, 3.70 s on `gfx1200`). `[Fock]` §3's claims are scored:
10 [Fock] records: 9 asserted, 1 unverifiable, 0 pending
(`data/literature/fock/claims.json`, #291). **No published cell is scored**
(the paper has no register table for the §3.2 kernel, and its tables are
A100/V100 wall-clock); milestone 12 built only the **diagonal** cells of
`[Fock]`'s 2x2 (the HGP split is milestone 14's, below; no fused MMD `F`,
#326); no gradient, no lever, nothing persists across SCF iterations, `f` and above
are not built (#330), and the screening comparison the fused scheme most
wants is unmeasured on compact fixtures (#329, #331). `KERNEL_GEN_CMD`
writes no `F` unit: `--fock-l` / `--fock-class` is its own emit path.
**Since milestone 14 the 2x2's (HGP, split) cell exists too, and ran on the
RTX 3080 only** (#401–#408, PRs #469–#477; `hgp_fock.md`, "The split cell",
is the one place with the detail): `Target.J` and `Target.K` on
`Engine.HGP`. `Spec.engine` (#401) is the recurrence as a key field,
`None` resolving to the target's native engine (`F` HGP, the rest MMD), the
key spelling only an off-native one as a slot suffix (`.../j.hgp/...`), so
every stored key is byte-identical; HGP is legal on `J` and `K` only and MMD
`F` is refused (#326); `am` is the recurrence's pairing (HGP `K` is
`(la, lc | lb, ld)`, as MMD's). `hgp/orbits.py` (#402) is the 8-fold orbits
and each one's cheapest representative (50 ordered classes in 13 orbits at
L ≤ 4 with every shell `l ≤ 2`, 70 in 17 uncapped; the permutation checked
as an exact index map plus `eri.eri` within `1e-12` of scale, worst 2.6e-16,
not bitwise). Two class modes: **canonical** (the literal, one kernel per
orbit, `hj_{rep}` / `hk_{rep}`, a runtime `perm`) and **ordered** (the
`ordered` lever, legal on `j.hgp` / `k.hgp` only, `hj_o_*` / `hk_o_*`). The
kernels (`eri/hjk/kernel_hj.py`, `eri/hjk/kernel_hk.py`, #404, #405) take §3.2's mapping and
`kernel_f`'s record unchanged, deposit every ordered quartet at weight 1 by
`AtomicAdd` (no weights, no `close`), and are held to `eri/j_coulomb/j.py` / `eri/k_exchange/k.py`
within `1e-12·(s_hgp + s_mmd)` in all 70 L ≤ 4 classes, interpreted, the
ERI tap bitwise against `eri/hgp_drive`. The stream (`eri/hjk/launch_hjk.py`, #403)
prunes nothing and screens on one density block per build, a skip, never a
break; the walker (`eri/hjk/walk_hjk.py`, #406) builds whole `J` and `K`
interpreted (H₂, H₂O/STO-3G, H₂O/6-31G; worst 3.82e-15 against libcint),
and `Engine` gained `j` / `k`. On the RTX 3080, #408: 184 passed (every L ≤ 4
class in both modes, and H₂ and H₂O/STO-3G whole), worst 2.59e-15 against
libcint. **It has not run on the RX 9060 XT**: compile-only there (182
passed), the card run attached to #431 by the user's decision. **Nothing is
timed** (`measured_store` refuses an HGP `J`/`K` row, `HGP_JK_REFUSED`),
nothing above L = 4 is tested, nothing persists, and
`fock/symmetry-vs-screening` stays unverifiable (#331).
**The 2x2's two diagonal cells are timed against each other, per class**
(`devtools/measure.sh scheme-ab`, `measure/measure_scheme_ab.py`,
`data/measured/scheme_ab.json`; `hgp_fock.md`, "The diagonal cells timed per
class"): fused HGP `F` (`kernel_f`, literal) against MMD `DQ` + `[p\|q]` +
`JAB` + `kernel_kblock`, (ss|ss) and (pp|pp) on gly5/10/15 in Cartesian
6-31G with an RHF density (`python -m oracle.polyglycine`, PySCF, an
untracked `.measure-cache/` pinned per row by sha256 and E_RHF), each class
as the sub-basis of its shells, the card's `F` held to its `J − K/2` first
(4.27e-15 / 1.83e-15 of `max|G|` at gly5, `τ = 0`), the verdict rule
declared in the store before any row. **All 12 cells `fused` on both cards**,
`mmd / f` 1.8–18 and growing with n. The scheme is still confounded with the
recurrence (the HGP split is untimed), so `fock/symmetry-vs-screening`
stays unverifiable.
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
(`test_schwarz_device.py`, generator `test/schwarz_driver.py`). Nothing about
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
(`test_sx_device.py`, generator `test/sx_driver.py`). Nothing about it is
measured.
**The DF Coulomb metric `(P|Q)` of a Cartesian auxiliary basis is generated
too, and ran on the RTX 3080 only** (milestone 15, #410–#416;
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
(`test_metric_device.py`, generator `test/metric_driver.py`, PR #430: worst
2.61e-14 of `max|M|` against libcint, 7.17 eps of the tower-twin scale
against the host). **It has not run on the RX 9060 XT**: the card was not
enumerated in the container, so on `gfx1200` it is compiled only; that run,
and a fix to `kernel-tier.sh`'s HIP device check (it tests `/dev/kfd` only
and printed `device: present` with no card), are open follow-up #431.
Nothing about it is measured or scored (no published board; a
`StructuralLimit`, #417); `(mn|P)` was its named follow-up, built next.
**The three-center DF integral `(mn|P)` is generated too, and ran on the
RTX 3080 only** (milestone 16, #441–#446; `docs/implementation/three_center.md`
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
(`test_eri3_device.py`, generator `test/eri3_driver.py`, PR #457: worst
3.30e-14 of `max|(mn|P)|` against libcint, 4.77 eps of the tower-twin scale
against the host on the fixture and 4.66 on the synthetic bases; the
fixture's (sp|g), (pp|f) and (pp|g) are not launched). **It has not run on
the RX 9060 XT**: kfd did not enumerate the card (`gfx_target_version` 0),
so on `gfx1200` it is compiled only, and by the user's decision that run is
attached to #431. Nothing about it is measured or scored (no published
board; a `StructuralLimit`, #447); no screen (`(mn|P)` has no Schwarz bound
yet), nothing above L = 4, Cartesian only.
**The device Boys is one template, `fj_boys<M>`** (#301; `emit/_c.py`'s
`BOYS_DEVICE`, every unit with a fundamental): every order `F_0..F_M` of one
`T` from one `exp(−T)`, the Kummer series at `M` and the downward recurrence
below `T = M + 8`, `erf` and the upward recurrence from it. It is witnessed on
the RTX 3080 over `reference.boys`' whole domain, `T ∈ [0, 1e6]`, `M ≤ 16`
(worst 2.1e-15; `test_kernel_device.py`), and compiled only on `gfx1200`
(#431). Its text moved every J, K and F unit, so `kernels.json`'s
`source_sha256` pins are **stale** until those rows are re-measured:
`devtools/measure.sh check` fails until then.
Every tier
reports what it actually touched rather
than implying more: a `gpu`-marked test skips without a card, so a run can be
**green and empty**; `kernel-tier.sh` exits **2** in that case and whenever it
could not run — read a 2 as "could not run", never as a pass.

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both toolchains build; `devtools/kernel-tier.sh` runs end to end on this box. |
| `test/` | **live, and thin — and it measures nothing.** Every wall-clock, counter, ceiling and lever-clock sweep, every recompile compared against a stored row, and every test that re-derives a tracked `data/measured` report or pins its `source_sha256` to today's emission lives in **`measure/`**, run by `devtools/measure.sh <sweep>` (the row below); wherever this cell still names such a sweep or store check, read it as that file's `measure/measure_*.py` counterpart. `test_toolchain.py` asserts nvcc and hipcc compile for this box's targets and that a kernel launched on each card matches a numpy reference, plus the counter witnesses (`ncu` reads a probe's FLOPs and bytes on CUDA; on HIP, since #140, `rocprofv3` loads and two witnesses pin why there is still no gfx1200 row: its byte counters read 0, refused as `measured.ROCPROF_ZERO_BYTES`, and its `--list-avail` has no fp64 counter). `test_kernel_device.py` (`gpu` + `cuda`/`hip`) runs the **generated** `[p\|q]` kernel and `fj_boys` on each card against `reference/`, plus every split cell of `measured_store.SUBGRID` as its batched launch sequence, plus #113's asymmetric study cells, plus opt-in sweeps that skip unless their store is set and run alone: wall-clock (`FJ_WALLCLOCK_STORE`; `FJ_WALLCLOCK_ASYM=1` for #113's arms) and roofline counters (`FJ_ROOFLINE_STORE`, `ncu` on CUDA; the hip test skips with `measured_store.UNMEASURED`'s reason, final since #140). `test_ceilings_device.py` (`gpu`) bounds each card's probe by the vendor's published peak (on gfx1200's fp64, per counted GPU cycle through `rocprofv3`'s `GRBM_GUI_ACTIVE`, #141) and, opt-in (`FJ_CEILINGS_STORE`), records the ceilings. `test_j_launched_ondevice.py` (`gpu` + `cuda`/`hip`; generator `test/plan_driver.py`) builds a whole molecule's `J` from the launch plan on each card against the launch oracle and libcint (H₂, H₂O/STO-3G at 1 and 3 batches and `τ` 0 and 1e-3, H₂O/cc-pVDZ dispatched), the two-H₂ chain under both sort keys, and, opt-in and alone, the whole-build wall-clock (`FJ_LAUNCH_WALLCLOCK_STORE`, #146) and #147's sort-key A/B; its compile-only companions (`cuda`/`hip`, no `gpu`) fail on a warning in generated text. `test_j_chain_ondevice.py` (`gpu` + `cuda`/`hip`, #157) runs the generated chain: part 1 compiles each `dq_{la}_{lb}` and `jab_{la}{lb}` unit the tier wrote alone into a harness and runs it over every record of its class in HF/cc-pVTZ, (ss) through (ff), against `eri.j_coulomb.j.dq_from_host` / `jab_from_host`; part 2 builds `J` for H₂, H₂O/STO-3G (1 and 3 batches) and H₂O/cc-pVDZ with every arithmetic step on the card (a `DQ` program, `Dmax` read back and sorted on the host, milestone 6's driver extended with one `JAB` launch per bra class) against libcint and the launch oracle, under error bounds stated in its docstring; its compile-only companions fail on a warning in a pair unit. Test-side device compiles (`conftest.compile_device_source`, `plan_driver.compile_checked`) go through sccache via `test/devcache.py` (#186; `KERNEL_NO_CACHE` turns it off; `test_devcache.py`). `test_measured.py` checks the measured store and its driver (its `cuda`/`hip` cases recompile and compare; the rest is fast: the byte-for-byte round trip of `kernels.json`, `roofline.json` and `builds.json`, the counter-row and build-row shapes and their loaders' refusals, the ncu/rocprofv3 parsers on recorded or synthetic CSV, the declared verdicts, #147's sort-key verdict held cell by cell). The rest is fast-tier: the `Spec` key, the transcribed tables' structural invariants, the scoring rules on **synthetic** candidates, the batch kernels interpreted against `reference/` (`test_kernel_batched.py`), the goldens (`test_emit_*`), `test_plan_optimum.py` (the exact optimum against an unpruned brute force, the beam as an upper bound), `test_mild_vs_optimum.py` (#111's gap; beam L = 11..16 `slow`), `test_plan_recovery.py` (the widened seed-order search, `slow`), `test_eq17_witness.py` (eq. 17's `1y`, broken by the misprint), `test_ceilings.py` and `test_device.py` (the ceiling records' provenance, the gate's refusals with injected readers, the refusal to score an uncharacterised card), `test_ridge.py` (#114's rule on synthetic rows and its claim on the tracked ones), `measure/measure_kernelrank.py` and `measure/measure_batchreport.py` recomputing `data/measured/{limits,crossread,batch_recommendation}.json` from the stores, and `test_layering.py`'s import allow-list. **The J chain, interpreted** (milestone 7, fast tier unless `slow`): `test_ir.py` and `test_interp.py` (`ir.Fragment` / `Inline` and their binding contract, #149), `test_e_fragment.py` (the `E` fragment bitwise against `mmd.hermite`, (ss) through (ff), and every recurrence step one `Weighted`), `test_kernel_dq.py` and `test_kernel_jab.py` (each kernel bitwise against the host for every pair class; d and f classes `slow`), `test_ao_j_chain.py` (`J` with every step generated, against libcint and the host paths; 6-31G and cc-pVDZ `slow`), `test_emit_cli.py` (the generator CLI's two ladders and the selector's `order`), and `test_spec.py` / `test_measured_keys.py` (a pair-class key never collides with a quartet key, and all 196 J keys are unchanged, and no `K` key joins one). **The plan's choices and the shipped plan** (milestone 8, fast tier unless `slow`): `test_plan_determinism.py` (the plan's output byte-identical across hash seeds, and the sieve a fixed point), `test_plan_order_search.py` (`plan/order_search.py`'s frontier, pruned product and climb), `test_plan_search.py` and `test_scoreboards.py` (the searched record replayed, never believed, and the search not the best of the 58 rules), `measure/measure_sieve_spill_claim.py` (#164's spill reading of the `no-sieve` rows), `measure/measure_planrank.py` (`plan_vs_kernel.json` recomputed) and `measure/measure_shipped_plan.py` (the literal default held to the declared rule and to the literal rows' text, and #166's limit recomputed). **The launch plan** (milestone 6, fast tier, no compiler): `test_launch.py` (the plan's records, classes, stable sort and refusals; the `jengine/ket-sort-key-is-the-product` claim), `test_launch_oracle.py` (the oracle against the host screened sweep to 0, the ordered reference and libcint; both sort keys' skip counts), `test_launch_staged.py` (`eri/j_coulomb/staged.py` interpreted against the host sweep, libcint and the oracle's ket-loop counts; H₂O against libcint `slow`) and `test_dispatch.py` (the choice from synthetic rows, its fallbacks and refusals, the tracked file recomputed). **The AO layer and its oracle** (milestone 4): `test_ao_{shell,hermite,pairs,eri,j,j_interpreted,screening}.py` test `ao/` — `E` against closed-form Hermite polynomials, the Schwarz bound proved over randomised geometry, and the ERI blocks (all 55 classes of HF/cc-pVTZ since #121, split across five fixtures by L) and `J_µν` (by `reference/` and by the interpreted generated kernel) against PySCF/libcint through **tracked fixtures** in `test/oracle/fixtures/` (`test/oracle/`: `generate.py`, `manifest.py`, `basis.py`; `test_oracle_{basis,manifest}.py`, the overlap as the cheapest witness of libcint's conventions). They run without PySCF — s/p in the fast tier, `d`+ and the larger `J` cases `slow`; only the `oracle`-marked drift tests need the `oracle` group, and they skip without it. `test_ao_j_ondevice.py` (`gpu` + `cuda`/`hip`) builds `J_µν` with `J_p` from the class selector's kernels on each card against the same fixture (it reads the `[p\|q]` slice of each cell's `order`, between its `DQ` and its `JAB`). **The one-electron matrices** (milestone 11): `test_ao_{overlap,kinetic,potential}.py` (closed forms, an independent quadrature for `T`, the `2π/ζ` identity, and since #240 libcint's `int1e_ovlp` / `int1e_kin` / `int1e_nuc` and PySCF's `hcore` on the 11 fixtures that carry them, s/p fast, `slow` from l = 2; worst `V` 5.4e-14 of scale), `test_ao_pairs.py`'s one-electron bounds and screen (#236), `test_kernel_{s,t,v}.py` (interpreted: `S` and `T` bitwise against the host (ss) through (ff), `V` within `16 ε M`, its L > 4 classes `slow`), `test_launch.py` / `test_launch_oracle.py`'s `one_electron_launch` and `walk_one_electron` (#241), `test_oracle_manifest.py`'s schema 2; on the cards (`gpu` + `cuda`/`hip`; generator `test/one_electron_driver.py`) `test_one_electron_device.py` (every ordered class of HF/cc-pVTZ against the host through (dd), and through (ff) `slow`, #244; `S`, `T`, `V` of H₂O/STO-3G and cc-pVDZ against libcint, #242) and `test_hcore_device.py` (`h = T + V` and `Tr(D h)` against `hcore`, #243). The test plan's `kernel-tier` / `high-L` markers are not registered; the deferred cells use `slow`. **Exchange `K`** (milestone 10, still open; every case with L = la + lc + lb + ld > 2, host-side or on a card, is written `slow` and **not yet run**, deferred to #231, except #259's one cell below): `test_spec_k.py` (the Hermite `am` and its AO-order trap), `test_ao_k.py` (the host `K` by two sweeps and through `eri.eri`, the scale, and against libcint's `vk` through the four `_k` fixtures: H₂ whole and H₂O/STO-3G's and 6-31G's s-s block fast, the whole H₂O matrices `slow`), `test_ao_k_screening.py` (#200's bound over randomised L ≤ 2 quartets with no tolerance, break against continue on the `τ` ladder, J's key misapplied moving `K`), `test_kernel_k.py` and `test_kernel_k_batched.py` (the kernel and its split class interpreted, bitwise against `eri.k_exchange.k.k_quartet` per bra record, the batches' sum against the one-batch kernel to rounding, the plan `==` J's), `test_launch_k.py` (`walk_k` against the host sweeps; J's key refused), `test_k_duplication.py` (`kdup`'s rows at L ≤ 2 recomputed, the subgrid rows `slow`), and on the cards `test_k_ondevice.py` (`gpu` + `cuda`/`hip`, generator `test/plan_driver.py`'s `k_driver`: `K_mn` from the launch plan at L ≤ 2 against the host twin, `walk_k`, `exchange_matrix` and libcint, #206; its compile-only companions fail on a warning in generated text) and `test_kernel_device.py`'s `K` cases (#205's `K_SUBGRID` checks, written and **not run to completion**; #259's `K_CHECK_CELLS`, `(dd\|dd)` at L = 8, run once at one batch on CUDA and passed; and the opt-in `K` wall-clock sweep, `FJ_WALLCLOCK_K=1`, beside each cell's `J` twin, which no stored row carries yet). Three `test_measured.py` tests state #205's finished store and are strict `xfail` (`K_205_PAUSED`) until it lands. **The fused Fock build** (milestone 12; fast tier unless `slow`): `test_spec_f.py` (`Target.F`'s AO order against `K`'s Hermite pairing, its own batch refusal) and `test_plan_dag.py`'s refusals, `test_hgp_vrr.py` / `test_hgp_hrr.py` (every count of the 21 DAGs pinned, the boundary set `==` `target_set`, 961 not 1,225), `test_hgp_cost.py`, `test_hgp_recurrence.py` (VRR and HRR bitwise against a direct recursion, the seed against `reference.pq`, and the two printed defects, eq 26's sign and eq 14's seed, shown failing), `test_hgp_eri.py` (#281: HGP against `eri/eri.py` in all 21 classes, contracted, and against libcint), `test_ir.py`'s HGP vocabulary, `test_ao_fock.py` and `test_oracle_fock.py` (the sweep against FJ's `J` and `K` and against libcint's `vj − vk/2`, the `_j`/`_k` pair's shared inputs asserted first; whole H₂O `slow`), `test_kernel_f.py` (the kernel interpreted: the ERI tap bitwise against the host path, every class's orbit deposit against the sweep, every degeneracy case, the Coulomb-weight-1 mutation failing), `test_lower.py`'s loop nest, `test_fock_pairs.py` (the pair stream, the printed eq. 29 readings shown unsound), `test_fock_build.py` (`walk_f`: whole `F` interpreted against the MMD sweep and libcint, every skip checked against exact ERIs; cc-pVDZ `slow`), `test_engine.py` (the protocol over both engines), `test_emit_fock.py` (21 goldens per backend, three by digest; compile cases `cuda`/`hip`), `measure/measure_fock_baseline.py` (the cross-tree join; its recompiles `cuda`), `test_fock_coverage.py` and `test_fock_claims.py` (the `[Fock]` reading and two of its defect claims), and on the cards (`gpu` + `cuda`/`hip`; generator `test/fock_driver.py`) `test_fock_ondevice.py`: every class against the sweep and the interpreted launches (16 of 21 `slow`), H₂O whole against the walk and libcint, and, opt-in and alone, the F wall-clock (`FJ_WALLCLOCK_F=1`, `FJ_WALLCLOCK_F_L`). **The split HGP `J`/`K`** (milestone 14; fast tier unless `slow`, nothing above L = 4): `test_spec_engine.py` (the engine field and its refusals), `test_hgp_orbits.py` (the orbits, 50 / 13 and 70 / 17, the exact index map and `eri.eri` within `1e-12` of scale), `test_hjk_pairs.py` (the stream: every ordered quartet once, both modes, every skip proved against exact ERIs, J's key misapplied to K), `test_kernel_hj.py` and `test_kernel_hk.py` (all 70 L ≤ 4 classes in both modes, interpreted: the tap bitwise against `eri/hgp_drive`, the deposit against `eri/j_coulomb/j.py` / `eri/k_exchange/k.py`, the density mutation and K's AO-order trap failing), `test_emit_hjk.py` (126 goldens per backend; compile cases `cuda`/`hip`), `test_hjk_build.py` (`walk_hjk`, whole `J` and `K` against the host and libcint; most H₂O cases `slow`), `test_engine.py`'s `j` / `k`, and on the card (`gpu` + `cuda`/`hip`; generator `test/hjk_driver.py`) `test_hjk_ondevice.py`, which has run on the RTX 3080 only (PR #477, 184 passed; the RX 9060 XT run attached to #431). **The DF metric** (milestone 15; s/p fast, d and above `slow`, nothing above L = 4 checked): `test_spec.py` / `test_measured_keys.py` (`Target.METRIC`'s pair key, its batch refusal, no collision), `test_e_fragment.py` (`center_fragment` bitwise against `mmd.hermite.e_axis` at `X_PA = 0`, l = 0..4), `test_oracle_metric_fixture.py` and `test_oracle_manifest.py`'s schema 3 (the four row pieces reassembled, their refusals), `test_ao_metric.py` (the host against `int2c2e_cart` and the aux `int1e_ovlp`, the (s|s) closed form, the dressing identity, the staging), `test_kernel_metric.py` (interpreted, against the host within 16 eps of the tower twin and against libcint; the kept targets pinned to the parity rule), `test_emit_metric.py` (nine goldens per backend; compile cases `cuda`/`hip`), and on the card (`gpu` + `cuda`/`hip`; generator `test/metric_driver.py`) `test_metric_device.py`, which has run on the RTX 3080 only (PR #430; the RX 9060 XT run is #431). **The three-center integral** (milestone 16; classes with only s/p shells fast, d and above `slow`, nothing above L = 4 checked): `test_spec.py` / `test_measured_keys.py` (`Target.ERI3`'s triple key, its batch refusal, the 510-key J digest unchanged), `test_oracle_three_center_fixture.py` and `test_oracle_manifest.py`'s schema 4 (the piece reassembled, the `(m, n)` symmetry to 1e-14 with its 1,998 mismatches pinned, the refusals), `test_ao_eri3.py` (the host against `int3c2e_cart`, the (ss\|s) closed form, the staging, the internal twin's refusal), `test_kernel_eri3.py` (interpreted, against the host within 16 eps of the tower twin and against libcint; the kept targets pinned to the parity formula; no class bitwise), `test_emit_eri3.py` (22 goldens per backend; compile cases `cuda`/`hip`) and `test_emit_cli.py`'s eri3 names (unique through L = 40, no collision with another ladder), and on the card (`gpu` + `cuda`/`hip`; generator `test/eri3_driver.py`) `test_eri3_device.py`, which has run on the RTX 3080 only (PR #457; the RX 9060 XT run is attached to #431). `test_claims.py` checks `data/literature/jengine/claims.json` against the `claim`-marked tests, both ways, `coverage.json` against the records, `j_engine.md`'s status table and every claims tally a doc quotes against the file, and every count of `limits.json`'s records a doc spells; since #271 it discovers every `<paper>/claims.json` and runs the both-ways check per paper (`jengine`, `fock`), and since #291 it holds any "N [Fock] records: …" tally a doc quotes to the fock file. |
| `measure/` | **live, never a gate.** pytest as a harness, outside `testpaths` (so no bare `pytest`, fast tier or kernel tier collects it), its files `measure_*.py`, its `conftest.py` loading `test/conftest.py`'s fixtures and card lock and putting `test/` on `sys.path`. `devtools/measure.sh list` names the sweeps: the timed ones (`wallclock`, `wallclock-k`, `wallclock-f`, `levers`, `builds`, `sort-key`, `roofline`, `ceilings`, `kblock`, `scheme-ab`; alone, no xdist, behind the gate, writing their stores, which default to the tracked files in `devtools/config.sh`), `scatter-tail` and `probes` (the ncu/rocprofv3 counter witnesses and the probe's published-peak bound), `recheck` (recompile and compare against `kernels.json` / `fock_baseline.json`; a compiler, no card) and `check` (the tracked stores and reports recomputed, ~20 s, no compiler and no card), plus `remeasure` (`python -m felizjueves.performance.measured --remeasure`). A kernel-text change that moves a stored `source_sha256` fails `check`, not the PR gate: re-measure, then `check`. `test_claims.py` scans both trees for `claim` marks. |
| `src/felizjueves/` | **importable, and small.** **The layout** (milestone 18, #501–#515): the package root holds only `__init__.py`, `spec.py`, `engine.py` and `levers.py`, and every other module sits in a package — `ao/` (shells, primitive pairs, the density and ket records), `mmd/` (McMurchie–Davidson: `hermite`, `e_fragment`, `recurrence`, the `[p\|q]` builder `kernel`), `hgp/` (the HGP recurrence), `plan/`, `ir/` (the IR in `__init__`, `interp`, `passes`, `lower`, `lever_ladder`), `emit/`, `eri/` (the shared ERI layer: `eri`, `hgp_drive`, `schwarz`, `kernel_schwarz`, `kernel_pair_bound`, `launch_common`; and one package per build: `j_coulomb/` (the J chain, its launch plan and `levers/`), `k_exchange/` (MMD `K`, `kblock`, `kdup`, `kernel_record_bound` and `levers/`), `fock/` (the fused `F`, its stream, walker and `levers/`), `hjk/` (the split HGP `J`/`K`) and `density_fit/` (the metric and `(mn\|P)`)), `one_electron/` (`S`, `T`, `V`, `SX` and their host references), `performance/` (the measured store and driver, the gate, ceilings, dispatch and the reports), `literature/` and `reference/`. `test/test_root_layout.py` (fast tier) refuses any other module at the root. Below, a module is named by its path under `src/felizjueves/` where it moved. `spec.py` is `Spec`, the frozen key every later subsystem joins on; since #150 its `am` is a quartet for `Target.J` and a pair `(la, lb)` for `Target.DQ` and `Target.JAB` (`AM_LENGTH`), a pair target never batched; since #232 `Operator` also has `OVERLAP`, `KINETIC` and `NUCLEAR`, tied by `OPERATOR_OF` to the pair targets `Target.S`, `T` and `V`, and the batch ban is stated per target (`_PAIR_BATCH_REFUSAL`: structural for `DQ`, `JAB`, `S`, `T`; by decision for `V`, which has Hermite targets); since #196 `Target.K` is a quartet whose `am` is `(la, lc, lb, ld)`, the two Hermite pairs bra first (`kernel.k_spec(L)` gives each `L`'s class, with `pq_spec`'s split), batched exactly as `J` is. `literature/` loads published tables and scores a candidate against them: `tables`, `rules` (`Agreement`, `AtOrUnder`, `ExactGivenInputs` — which since #108 can carry a `refused` decision instead of inputs, scoring UNSET, never PASS), `scoreboard`, `limits`, `claims`. `plan/` (below) computes the Table 1–3 cells it scores. `mmd/kernel.py` builds the `[p\|q]` kernel as IR (`ir/__init__.py`, checked by `ir/interp.py`); `emit/` spells it for each vendor (`python -m felizjueves.emit` is the generator CLI); `build_pq_kernels` builds a split class's batch kernels, each `ir.passes.atomize`d. `ir.Fragment` / `ir.Inline` (#149) are a device-code block with a checked binding contract, one level only (no nested `Inline`); `mmd/e_fragment.py` (#152) is the undressed `E` as fragments (the block composed at build time from one axis builder; every entry one `Weighted`, the seed `exp(+ζ X_PA X_PB)`); `eri/j_coulomb/kernel_dq.py` (#153) and `eri/j_coulomb/kernel_jab.py` (#154) are the pre-contraction and bra-transform-and-scatter kernels that inline it, each with its own layering rule that keeps it off `ao`, `interp`, `emit`, `device` and `reference/`. `one_electron/kernel_s.py` (#237), `one_electron/kernel_t.py` (#238) and `one_electron/kernel_v.py` (#239) are the one-electron kernels, one thread per record with `kernel_jab`'s scatter rule and no prefactor formed: `S` inlines `e_fragment.overlap_fragment` (the `t = 0` column, dead entries dropped by `_live`), `T` `kinetic_fragment` (the axis bodies at `lb + 2`), and `V` `block_fragment` plus `recurrence.fundamental_prelude` / `recurrence_body` over `plan(v_spec(la, lb))`'s DAG, the plan layer's first non-J consumer, with the nuclei serial; Every kernel's Boys is one IR statement, `ir.Boys(dst, top, t)` (#301; `Fn.BOYS` and #266's `Kernel.wide_boys` are gone), filling a local fp64 `boys[0..top]` that each fundamental reads, spelled `fj_boys<top>(T, boys);`. Every kernel takes its per-record arrays as plain positional parameters. The one-electron `S`, `T`, `V` and `SX` call the record weight `s_ab` (as `V` and Schwarz always did; `S`, `T` and `SX` said `s` beside the output `S`) and take each record vector as three flat arrays, one per axis (`pa_x pa_y pa_z`, `pb_*`, `V`'s `p_*`; `kernel_s.pa_name`); every other kernel takes vectors interleaved `[n_rec, 3]` (`XPA[n_rec, 3]`). #424, #422 and #421 moved `S`, `T`, `V`, `SX`, Schwarz, `DQ`, `JAB` and `K` to per-axis arrays in one by-value `PrimPairSoA` struct, and all three were reverted, so there is no `ir.Bundle`, no `Kernel.bundled` and no `include/felizjueves/prim_pair.h`. A consumer that wants a per-axis layout asks for it after the build instead: `ir.passes.split_axes(kernel, {"XPA": ("pa_x", "pa_y", "pa_z"), ...})` turns each named `[n, 3]` parameter into three flat ones under the consumer's names, every `X[i, d]` read as `x_d[i]`, held bit-for-bit to the interleaved kernel in the interpreter (`test_passes.py`; DQ, JAB, `[p\|q]`); the builders' interleaved default is untouched, no struct, plain positional parameters. Likewise `ir.passes.transpose(kernel, ("D",))` stores a 2-D array parameter with its extents swapped, every `X[i, j]` read as `X[j, i]`: applied to `DQ` (which writes `D`) and `[p\|q]` (which reads it), `D` becomes `[n_q, n_ket]`, so a warp's lanes, consecutive kets, read consecutive doubles (`[Fock, §4.2]`'s "transposed before kernel launches to ensure contiguous reads"); held bit-for-bit in the interpreter (`test_passes.py`: DQ both densities, `[p\|q]`, composed with `split_axes` and `atomize`), compiled under both compilers (`test_emit_gpu.py`), the builders' row-per-ket default untouched, not run on a card here (FreeBreakfast's J build is its consumer). `build_dq_kernel(cls, density="ao")` (`dq_{lc}_{ld}_ao`, `AO_PARAMS`) reads the AO density in place, `Dao[off_c[k] + i, off_d[k] + j]`, `JAB`'s addressing, instead of the host's `Dcd[pair[k], r]` gather: `==` the default interpreted for every class (ss)–(ff), the default untouched, not run on a card (`j_engine.md`, step 2). `eri/k_exchange/kernel_k.py` (#198, #203; its own `k_builder` layering rule, off `ao`, `interp`, `reference/`, `emit` and `device`) is the exchange kernel, `build_k_kernel` one batch and `build_k_kernels` a split class as `ir.passes.atomize`d partial-sum batch kernels: the plan, DAG, schedule and thread mapping are `J`'s, both `E` fragments are inlined, `I` and the `K` accumulator share one shared array (`(ff\|ff)` fits 48 KiB, `(gs\|gs)` does not), the screen is `J`'s `break` on `Ip · Iq · maxD` with the kets sorted by `Iq` alone (`NO_DENSITY_KEY`, citing `[Fock, §4.3.1]`), and the recompute form, which `[JEngine, v2, §6]` says has more merit for `K`, is named, not built. `eri/k_exchange/kdup.py` (#202; its own `count` rule, plan only) counts the work a split duplicates, `J` against `K`, per primitive quartet: K's is the larger at every split cell, by two to four orders of magnitude on the subgrid. **Engine #2, the HGP fused `F`** (milestone 12; `docs/implementation/hgp_fock.md`): `hgp/` (`vrr.py`, `hrr.py`, `cost.py`, `mmd/recurrence.py`; layering rules `hgp` and `hgp_builder` keep it off `plan/` and `reference/`, pin 11), `eri/fock/kernel_f.py` (#282, #283; rule `f_builder`, off `plan`, `ao`, `reference/`, `interp`, `emit`, `device`: the fused kernel, `pack`, `density_blocks`, `close`, `derive_weights`, `degeneracy`, `MAX_PRIM_PAIRS = 128` a refusal; staging per thread because the IR has no barrier, #333), `eri/fock/launch_f.py` (#285, rule `f_launch`: `f_launches()` → `FPlan`, the pair stream, its bound injected; it takes the stable sort and the bound check from `launch_common`, #504 (#328), and takes the density and `tau` checks from `ao.density`, #503), `eri/fock/walk_f.py` (#286, rule `f_walk`: the oracle, the plan read by duck type, the launch injected), `eri/fock/fock.py` (#275: `G = J − K/2` by every ordered quartet at weight 1, the `eri=` slot injected, the symmetry witness) and `eri/hgp_drive.py` (#281, rule `hgp_adapter`: HGP's `eri_block` interpreted, Boys injected), and `engine.py` (#292, rule `engine`: `Engine`, `Mmd(pq=)`, `Hgp(boys=)`; the one module that names both engines, beside the two generators `emit.__main__` and `measured`, which may name each only by its kernel builder (`test_layering.py`'s `GENERATORS`, #287, #288); its docstring's pushed-back-down list is the milestone's primary report, #327). **The split HGP `J`/`K`** (milestone 14; `hgp_fock.md`, "The split cell"): `spec.Engine`, `NATIVE_ENGINE`, `Spec.engine` and `Spec.target_slot` (#401; `_ENGINE_REFUSAL`, `levers.ENGINE_SLOTS`, the `ordered` lever, `measured_store.engine_of` and `ledger_levers()`), `hgp/orbits.py` (#402: `orbit`, `representative`, `to_representative` → `Orientation`, `ordered_classes` / `canonical_classes`), `eri/hjk/launch_hjk.py` (#403, rule `hjk_launch`: the ordered pair stream, both modes, the `j` / `k` one-block screens, `MAX_PRIM_PAIRS` restated), `eri/hjk/kernel_hj.py` (#404, rule `hj_builder`) and `eri/hjk/kernel_hk.py` (#405, rule `hk_builder`; its nest and modes imported from `kernel_hj`), and `eri/hjk/walk_hjk.py` (#406, rule `hjk_walk`: the oracle, the plan read by duck type, `run=` injected); `engine.py`'s `Engine` gained `j(spec)` / `k(spec)` on both engines, and its docstring "THE SPLIT HGP CELL" lists what the cell reuses and what it does not. `ir.lower.KernelCost` gained `nest` and `carried` and `ir.lower.tiered` (#284): the loop model, MMD's numbers byte-identical. `ir.Term.c` is a signed nonzero int since #277, and `ir.Factor` has the HGP members. `measured_store` admits an F row (#278: no lever, one batch, gated wall-clock required) and `measured --target f --L ...` writes it (#288). `performance/device.py` holds `Arch` — with each card's measured `fp64_flops_per_s` and `mem_bandwidth_bytes_per_s`, read from `data/measured/ceilings.json` (`0.0` is *not characterised*, and `Arch.ridge` / `verdict` refuse such a card) — and `MeasuredKey`. `performance/gate.py` is the card gate every timed, probed or profiled phase runs behind (card at most `GPU_BUSY_MAX` = 30% busy and CPU at most 10% before, and a settled reading after; non-idle or unreadable refuses). `performance/ceilings.py` is the fp64-FMA and STREAM-triad probe pair (#109). Since #94 the measured layer is two modules: `performance/measured_store.py` is the store — `Measured` rows, `SUBGRID` and, since #201, `K_SUBGRID` (each `J` subgrid cell's `K` twin, the target stored on a non-J row; #205's `K_UNTIMED` records every L = 13 and 16 `K` cell as unlaunchable on both cards under `kernel_k`'s pinned launch, its `Ksh` over the per-block shared memory), the counter fields and per-batch-kernel counter rows (#110), the roofline point and verdict (computed, never stored, #112), the verdicts declared before their rows (`nine_vs_twelve`, `lever_ab`, #147's `sort_key_ab`, and milestone 8's `sieve_verdict` / `sieve_may_leave_the_default`, `tuned_verdict` / `tuned_may_become_the_default` and `sieve_spills`, over the declared arms `ARMS`, each with its kind on the row), and `BuildRow`, the whole-build wall-clock row (#146) — and `performance/measured.py` is the driver that compiles, parses ptxas / AMD remarks and `ncu` / `rocprofv3` output, and runs `--remeasure` (and `--asym`, `--roofline`, `--lever`, `--lever-ab`). `performance/planrank.py` (#165) joins `PlanCost` for literal, `no-sieve` and `tuned` against the measured spills and same-session clock per card: the counts do not predict the kernel on either card. `performance/kernelrank.py` ranks `ir.lower.KernelCost` against those rows (#54); `performance/batchreport.py` derives the recommended `n_batches` per `(L, arch)` from wall-clock alone (#56); `performance/ridge.py` is #114's crossing rule, transcribed from the issue that declared it before the rows; `performance/crossread.py` reads model, clock and counters against each other per `(L, arch)` (#115: the choice is the clock's); `performance/asym.py` is #113's asymmetric-split study path (its own layering rule; `Spec` and `partition` still refuse those counts). `eri/j_coulomb/launch.py` (#138, #142) is the whole-molecule launch plan as a value — `launches()` → `LaunchPlan` of `ClassLaunch(lp, lq, n_batches, bras, kets, kernels, bra_blocks)` over primitive-pair `PairRec`s, kets sorted descending by `I_q · max|D_q|`, stably — and the launch oracle that walks it in pure Python (`walk`, `coulomb_launched`); its bound and `[p\|q]` arrive injected, and its own layering rule keeps it off `reference/`, `interp`, `emit`, `eri.j_coulomb.drive`, `eri.j_coulomb.staged`, `dispatch` and the measured layer. Beside it (#241), `one_electron_launch()` → `OneELaunch` of `OneEClass(la, lb, recs, blocks)` over `OneERec`s is the one-electron build — a sibling, because `ClassLaunch` is J-shaped (a ket side, a density key, a sort) — and `walk_one_electron` rebuilds `S`, `T` or `V` from it. `eri/k_exchange/launch_k.py` (#204) is the `K` launch plan, the other sibling — `k_launches()` → `KLaunchPlan` of `KClassLaunch` per whole quartet class over the **ordered** sweep, its key and bound injected, `walk_k` refusing a ket list not sorted by the bound — sharing `PairRec`, the stable sort, the bound check and the batch admission with `eri/j_coulomb/launch.py` through `launch_common.py` (#504, the `launch` half of #328; no module imports a `_`-prefixed name from `launch`); `eri/j_coulomb/staged.py` refuses a non-J plan. `performance/dispatch.py` (#145) is a class's batch count from measured wall-clock only (`batchreport`'s best median per `(arch, class, input)`, the input required; 1 for a class with no row; an uncharacterised card refused), injected into `launches(resolve=)`, never imported by it. Levers: `no-sieve` and `tuned` (plan; `tuned`, #161, builds from the searched board's recorded cut and per-batch seed orders through `pipeline.tuned_plan`, and refuses a cell with no recorded choice; both A/B'd on both cards, #162 and #163, both NOT SHOWN, so neither moved the default, #166); `ir.passes.atomize` (applied by the builder to every split class, not a choice); and `recycle` (`ir.lower.RECYCLE`, slot recycling, threaded through `Spec.levers` and applied by `measured`'s driver, not by `KERNEL_GEN_CMD`). #55 A/B'd `recycle` over all 20 subgrid cells on both cards: nvcc 13.0 and hipcc / ROCm 7.2.4 emit identical machine code for both arms, so it is neutral — `sm_86` 20/20 neutral, `gfx1200` 16/20 neutral with 4 cells read as slower on that identical code (the protocol's noise floor). It stays opt-in; `Spec.levers` defaults to `()`. The AO layer (milestone 4) is host-side, and its modules import neither `reference/` nor `plan/` (evaluators arrive by injection); since milestone 18 only `shell`, `pairs` and `density` are left in `ao/`, the rest sitting with the engine or family that owns it: `ao/shell` (contracted Cartesian shells; primitive normalisation in one function), `mmd/hermite` (the undressed `E`), `ao/pairs` (primitive-pair records, `dress` — the only place a pair factor or contraction coefficient is multiplied in — and the pair-factor-free Schwarz bound, its evaluator injected), `eri/eri` (`(ab\|cd)` blocks), `ao/density` (#503: the density and `tau` checks, `EPair` / `epair` and the ket records, out of `j` so `k`, `fock`, `schwarz` and the launch plans need not import it), `one_electron/one_electron` (`matrix`, a symmetric AO matrix from per-shell-pair blocks, #233), `one_electron/overlap`, `kinetic` and `potential` (`S`, `T`, `V` on the dressed block, #233–#235, the `[t]^(0)` evaluator injected; `pairs`' `overlap_bound` / `kinetic_bound` / `potential_bound` and `pair_list`, the proved one-electron screen, #236), `eri/j_coulomb/j` (`J_µν` by two sweeps, the `jp=` seam, the `dq=` and `jab=` seams with their host references `dq_from_host` / `jab_from_host` (#151), and #88's screened sweep), `eri/k_exchange/k` (#197, #200: `K_ab = Σ_cd D_cd (ac\|bd)` by the staged quartet `k_quartet`, a unique sweep with K's own weights against the ordered one and `k_from_eris`, the `k=` seam, and K's Schwarz bound `I_ac · I_bd · max\|D\|` with its sorted `break`), `eri/j_coulomb/drive` (`KernelJP`: `J_p` from the interpreted generated kernel; `KernelDQ` and `KernelJAB`, the same for the two new kernels; and `coulomb_matrix_chain` (#155), `J` with every step a generated kernel, interpreted; its own `adapter` rule in `test_layering.py`, which lets it reach the builder, `interp` and `plan` but not `reference/` — Boys is injected), and `eri/j_coulomb/staged` (#143, the same rule: `J` from a launch plan, one `J_p` per bra class, the kernels interpreted with the real screen; since #155 its `ket_data=` and `transform=` hooks take the kets and the back-transform from the chain). `j`'s `Screened` carries `n_examined`, `max_skipped` and `dropped` since #142. `reference/boys.py` is Gauss-Legendre below `T = 125` and the measured large-`T` asymptote from there (#122) to `T_MAX = 1e6` (#265, measured against the Kummer series so `V`'s point-charge kets reach every fixture), refusing beyond. |
| `src/felizjueves/plan/` | **live, host-side only.** Pipeline steps 1–3 of `[JEngine, v2, §6]`: `targets`, `batching` (the minimax batch split), `mild`, `dag` (one MILD DAG per batch), `sieve` (Alg. 3, off with the `no-sieve` lever), `cost` (`PlanCost`, Tables 1–3's quantities), composed by `pipeline.plan(spec)` with a JSON per-batch report. `PYTHONPATH=src uv run python -m felizjueves.plan replay literal` scores a board (exit 0 pass, 1 FAIL, 2 could not run, 3 UNSET/no candidates — `literal` exits 3 today, Tables 1–2's `exact_given_inputs` being refused); `report L n_batches` prints one plan. It counts; it emits no kernel text and compiles nothing. `recovery` recovers the inputs behind the printed cells as rules — the cut (Table 3, 40/40) and, unrecovered, the seed order (`seed_orders(widened=True)`, 58 rules, #108); `search` is the searched board's per-cell cut and per-batch seed-order permutations, climbed by `order_search` (#170, #172: the Pareto frontier, the pruned per-batch product and a seeded hill climb from the 58 rules) and selected by #171's declared rule; `pipeline.tuned_plan` builds a `tuned` Spec from that record. `optimum` (#106) is the exact `(intermediates, FLOPs)` Pareto frontier of a one-batch class by DP over Boys layers, for `L <= EXACT_MAX_L = 5` (above it `BeyondExactCeiling`, with the measured reason), and `beam` an achievable scheme above it — an upper bound on the optimum only; nothing in the pipeline imports it. `schedule` (step 4: a topological order of one batch's DAG, not text) also lives here, and is what `mmd/recurrence.py` and `mmd/kernel.py` consume. |
| `data/` | **live.** `literature/jengine/table{1,2,3}.json` is `[JEngine]` Tables 1–3, 120 cells with provenance. `scoreboards/jengine/{literal,searched}.json` are the two boards. On `literal` every Agreement floor is set to what the plan layer measured (exact cells: Table 3 38 on the minimax reconstruction, Tables 1–2 post-sieve 7 and 15); Table 3's `exact_given_inputs` records its recovered cut and passes 40/40, and Tables 1–2's are **refused** under the rules-only doctrine (#108: the best of 58 seed-order rules reproduces 33/40 and 38/40, and a per-cell fit is not admitted), which scores UNSET. `searched` replays `plan/search.py`'s recorded cut and per-batch seed-order permutations per cell (`searched_cuts.json`, schema 3). Since #172 it PASSes `at_or_under` on Tables 1, 2 and 3 in all 40 cells: 22 land on the printed triple exactly and 18 come in under it. Each order is climbed from the 58 rules within a bounded neighbourhood, and the plan is chosen from the frontier by #171's selection rule. That is a choice the board makes, not a recovered input. `literature/jengine/limits.json` holds twenty-one `StructuralLimit`s: the evidence for the two refusals, the searched board's record (the space it searched, no residual, and the earlier spaces' residual kept as `previous_residual`), `[JEngine, v2, §4.1]`'s comparison as not reproducible here (#111), what #114's ridge answer cannot see, #147's sort-key verdict (contradicted on both cards, two cells unresolved), and #148's three: the `Launch` IR and `LaunchCost` as deliberately absent, the record-versus-shell-pair screen granularity as a reading of Alg. 1 (not a defect), and what the on-card build does not reach (the device `fj_boys` then witnessed on `T ∈ [0, 500]` only, `[0, 1e6]` since #301 with no four-H₂ build added, fixtures only to d, so the dispatched cell only ever chooses 1; the per-pair path untimed; gfx1200's slow upload unexplained), and #158's three: the `DQ` and `JAB` kernels as this repo's own design (the paper defers them to `[Fock]`, not read), the instruction form's reach (`E` fits `Weighted`; four work-arounds where the IR did not reach), and what the generated chain on the cards does not reach (`J` through d only, nothing measured, the `(23)` `DQ` program's ~7.2 GB of the 3080's VRAM), and #160's three: the seed order's spread over the 58 rules (`seed_order_spread.json`), and Alg. 3's re-expansion direction and deletion reach as reconstruction choices, and #164's: the sieve's spill pattern on our cards (non-monotone on both, but the `[5|6]` stand-in spills less on both and the mean falls on gfx1200 only, so the published pattern fails on both and cannot be the paper's witness), and #166's: the refusal to ship the tuned plan (the literal plan stays, on measured wall-clock: `tuned_may_become_the_default` is False and the search NOT SHOWN on both cards), and #266's: the device `fj_boys`, two spellings until #301 unified them into one template (the record now holds the template's witness and its stale-pin follow-up), and #245's: the one-electron family (`S`, `T`, `V`, `h`) as this repo's own design on McMurchie & Davidson's `E` with **no published board** (`[JEngine]` is J only; nothing scored, no claim added), what is not measured (`V` spills from `la + lb = 3`; `V` (ff) runs on the 3080 with ~170 MiB of VRAM headroom) and the named follow-ups, and #417's: the DF metric `(P|Q)` (milestone 15) as this repo's own design on McMurchie & Davidson's recurrence with **no published board** (nothing measured or scored, no claim added), correctness at `lP + lQ <= 4` only, Cartesian only, run on the RTX 3080 only (#431 is the RX 9060 XT run), `(mn|P)` the named follow-up and DF-J/K out of scope (`docs/implementation/metric.md`), and #447's: the three-center DF integral `(mn|P)` (milestone 16) the same way, with **no published board** (nothing measured or scored, no claim added), correctness at `la + lb + lP <= 4` only, Cartesian only, no screen, run on the RTX 3080 only (the RX 9060 XT run attached to #431), the name `eri3` and the `[m][n][P]` layout recorded as decisions (`docs/implementation/three_center.md`). `literature/jengine/claims.json` is the registry of `[JEngine]`'s non-numeric claims — 35 records: 18 asserted, 14 unverifiable, 3 pending (the 3 pending are #195's `K` claims, `k-contraction-at-quartet-level`, `k-recompute-has-more-merit` and `k-splitting-redundant-hermite`, each with its scoring rule fixed before any `K` kernel existed; #207 scores them); each asserted one names a test that fails when the claim is false, each unverifiable one its mechanism (`j_engine.md`, "The `[JEngine]` claims", rendered from the file and checked by `test_claims.py`) — and `coverage.json` (#120) is the unit-by-unit reading of the paper (34 units) it is the inventory of. `mild_vs_optimum.json` (#111) is MILD's gap against `plan/optimum.py`'s exact optimum: equal through L = 4, behind on both axes at L = 5 (+6.25% intermediates, +0.625% FLOPs) — a disagreement with §4.1, recorded as the result — with a beam's lower bound above. Table 4 (spills) is deliberately not transcribed; `data/literature/jengine/README.md` says why. `measured/kernels.json` holds 258 `Measured` rows, 196 `J`, 20 `K` and 42 `F` — registers and spills as nvcc (`-Xptxas -v`, bytes) and hipcc (`kernel-resource-usage`, register counts) report them on `sm_86` and `gfx1200`, keyed on `MeasuredKey.key()`, rewritten by `python -m felizjueves.performance.measured --remeasure`: 66 literal rows (one per one-batch class L = 0..16, and one per cell of `measured_store.SUBGRID`, L ∈ {7, 10, 13, 16} × `n_batches` ∈ {1, 3, 6, 9, 12}, aggregated over its batch kernels, per arch), 40 `.../recycle@...` rows (#55: the recycle arm of every subgrid cell on both arches, each with its same-session literal control in raw `wallclock:control:*`; `measured_store.lever_ab` is the verdict, the store's header `levers` the conclusion), and 16 `.../asym@...` rows (#113: L ∈ {13, 16} × n ∈ {4, 5, 7, 8} on both arches, each timed against its legal neighbours in one session — every cell WORSE), 42 `.../no-sieve@...` rows (#162: the subgrid plus L = 11, one batch, on both arches, `arm_kind` `plan`, each with its same-session literal control; `sieve_verdict` NOT SHOWN on both) and 32 `.../tuned@...` rows (#163: the 16 subgrid cells with a recorded choice on both arches, the four 12-batch cells not measurable; `tuned_verdict` NOT SHOWN on both, so the generator ships literal, #166), and, since #205's partial PR #258, 20 `.../k/literal@...` rows (L = 7 `(2,2\|2,1)` and L = 10 `(3,2\|3,2)` × every subgrid batch count, on both arches, `target` `k`): registers and spills only, **no wall-clock**, and every one spills; L = 13 and 16, the clock beside each cell's `J` twin and the spill-curve verdict are still #205's. The K rows are the ones measured before #421, on the interleaved text the generator writes again since the SoA revert (#421's re-measurement, its `CLOCK_PENDING` and `SOURCE_PENDING` went with it). And 42 `.../f/literal@...` rows (#288, PR #321; #290, PR #324): every one of the 21 F classes on both arches, `target` `f`, keyed `coulomb/d0/(la,lb\|lc,ld)/b1/f/literal@<arch>/<toolchain>`, written by `python -m felizjueves.performance.measured --remeasure --target f --L ...` and timed by `test_<backend>_f_wallclock` run alone: registers, spill, static shared bytes (8,192 everywhere) and **wall-clock** (one off-diagonal launch of 2048 × 128 contracted quartets, `K = 9`, wide input as `runtime_ns`, near as `wallclock:near:*`), every phase behind `gate.wait_quiet` with all four gate readings and `wallclock:gate_discarded_phases = 0`; on `sm_86` every class with at most 40 boundary targets fits and every one with 76 or more spills, and `(dd\|dd)` is 255 registers, 97,464 B of spill stores and 2.33 s per launch (`gfx1200`: 192 VGPRs, 3.70 s). The 40 literal subgrid rows (20 per arch) also carry **wall-clock**: median, IQR and 31 repeats on a wide-T input (`runtime_ns`) and a near-T one (raw `wallclock:near:*`), re-timed by #93 behind the card lock and utilisation gate, one session per arch, on host `1da02cac6d4e (container)` with the RTX 3080 (`nvcc 13.0`) and the RX 9060 XT (`hipcc / ROCm 7.2.4`); no cell dropped. Each such row records the card's state in raw `wallclock:gpu_util_before` / `_after` (and `wallclock:near:...`); the `recycle` rows were timed earlier (#55, host `ddbbc02dbb08 (container)`) and carry `null` there — unknown, not idle (the store's header `gpu_util`). Outside the subgrid nothing is timed. `measured/ceilings.json` (#109) is each card's measured fp64 rate and triad bandwidth, behind the gate: `sm_86` 4.39e11 flop/s and 6.74e11 B/s (ridge 0.65 FLOP/B), `gfx1200` 4.46e11 and 3.00e11 (ridge 1.49) — achievable rates, never datasheet peaks; the vendors' published figures are a sanity bound only. `gfx1200`'s fp64 is **settled** (#141): single issue, bounded per counted GPU cycle (128 flop/clk, `GRBM_GUI_ACTIVE`; 0.40 TFLOPS at the rated 3130 MHz boost), and the recorded 4.46e11 is 111% of that because the card ran at a counted ~3.49 GHz; the ridge reads the recorded value only. #141's session also re-measured gfx1200's triad 1.7% lower (2.947e11) and did not commit it. `measured/roofline.json` (#112) holds 124 counter rows, one per batch kernel of every subgrid cell on `sm_86` (`ncu` 2025.3.1.0, the card's state in each row); `gfx1200` has **none**, and since #140 that is **final** (`measured_store.UNMEASURED`): `rocprofv3` now loads, but gfx12 defines no fp64 counter and the GL2C byte counters read 0 on this box. It is recorded as absent, never as a zero or a "no". The answer (#114): on `sm_86` batching crosses the ridge only at L = 16, at 3 batches; L = 7, 10 and 13 are compute-bound already at one batch. `measured/limits.json` records where `KernelCost` mis-ranks those rows (#54, #60: four records) and #115's four kinds of cross-read disagreement (`asserted_value.cross_read = true`); `measured/crossread.json` is #115's table per `(L, arch)`; `measured/batch_recommendation.json` is the best-median `n_batches` per `(L, arch, input)` with its spread — wide-T picks 1 everywhere, near-T splits from L = 13 and the cards disagree at L = 16 (`j_engine.md`, "Measured"), which is why the generator ships one batch by decision (its `about`, from `batchreport.ABOUT`). `measured/plan_vs_kernel.json` (#165) is `planrank`'s report: `PlanCost` against spills and same-session clock per card, every inversion named. `literature/jengine/k_duplication.json` (#202) is `kdup`'s count per `K` class and batch count (L = 0, 1, 2 and the subgrid), its rule written before the numbers; on counting it supports `k-recompute-has-more-merit`'s arithmetic half, and the claim is not moved. `literature/jengine/seed_order_spread.json` (#160) is the seed order's spread over the 58 rules per cell, and Alg. 3's three open choices with the tests that pin them. `measured/builds.json` (#146, #147) holds 22 whole-build rows, 11 per card, on host `e614efd48a48 (container)`, each with its gate readings (null refused): 3 `baseline` cells (H₂O/STO-3G at τ 0; H₂O/cc-pVDZ dispatched at τ 0 and 1e-3) and #147's 8 `sort-ab` rows per card (two arms × H₂O/cc-pVDZ and the two-H₂ chain × τ 1e-3 and 1e-10, one session per card). The timed region is in its header (zero, launch, synchronize, read back; uploads, eq. 20 and the scatter outside). Every row predates milestone 7, so none times a `DQ` or `JAB` kernel. #147's verdict: the global-`max|D|` key skips no pair anywhere, yet it is faster at some cell on each card, so the expectation is CONTRADICTED on both. Observations of a toolchain and two cards, **never compared against Table 4** or the paper's A100. `ao/screening.json` is #88's `τ` ladder (error and skip fraction per rung, compact H₂O and an extended chain of four H₂ since #122 — as long as `reference.boys`' `T ≤ 2000` allowed then, kept at that extent after #265 widened it), rewritten by `test/test_ao_screening.py --write` and pinned by its `slow` test. The libcint fixtures themselves live in `test/oracle/fixtures/` (28 files, HF/cc-pVTZ's ERI blocks split over five of them by L), with provenance; three of them (`h2o_sto3g_x_631g`, `h2o_sto3g_x_ccpvdz`, `h2o_sto3g_x_631g_moved`) carry libcint's cross-basis overlap against `h2o_sto3g_j`'s AOs (`int1e_ovlp_cross:h2o_sto3g_j`, `gto.intor_cross`), `S_{B1,B2}`'s oracle; three of them (`h2_sto3g_schwarz`, `h2o_sto3g_schwarz`, `h2o_631g_schwarz`) carry libcint's `(mn|mn)` for every AO pair (`int2e_diag`) on their `_j` siblings' inputs, the Schwarz matrix's oracle; since #240 the 23 AO fixtures are schema 2, and 12 of them (all but `h2_sto3g.json`, the five HF/cc-pVTZ ERI files, the three `_schwarz` files and the two same-geometry `_x_` files) carry libcint's `int1e_ovlp`, `int1e_kin`, `int1e_nuc` and PySCF's `hcore`; four of them (`h2_sto3g_k`, `h2o_sto3g_k`, `h2o_631g_k`, `h2o_ccpvdz_k`, #199) carry PySCF's `get_jk` `K` (`vk`) on their `_j` siblings' inputs; milestone 12 adds no fixture (pin 10): `test/oracle/manifest.fock_pair` asserts a `_j`/`_k` pair's shared inputs before combining `vj − vk/2`; the other four files are **schema 3** (#411): the DF metric's oracle, libcint's `int2c2e_cart` and the aux `int1e_ovlp` on `df.make_auxmol(mol, 'cc-pvtz-jkfit')` with `h2o_ccpvdz_j`'s molecule (166 Cartesian functions), split by rows over `h2o_ccpvtz_jkfit_metric_rows{0_43,43_81,81_121,121_166}.json` to stay under the 512 KiB hook, each with a `metric = {set, rows, parent, converted}` key and **no density**, reassembled (and refused on a gap, an overlap or an input mismatch) by `oracle.manifest.metric_set`; and one file is **schema 4** (#442): the three-center oracle, `h2o_sto3g_jkfit_3c_cols0_166.json`, libcint's `int3c2e_cart` (`aosym="s1"`) of `h2o_sto3g_j`'s 7 AOs against `df.make_auxmol(mol, 'cc-pvtz-jkfit')` on that molecule (45 shells, 166 functions), one piece at 193,822 B, with a `three_center = {set, cols, parent, converted}` key and **no density**, reassembled (and refused on a gap, an overlap, a duplicate, an input mismatch or a parent mismatch, the parent compared by atoms, `provenance.mol` / `cart` and `nao`) by `oracle.manifest.three_center_set`. **Milestone 12's data:** `literature/fock/` (#271) is `[Fock]` §3's reading, read from arXiv:2407.21445v1: `coverage.json` (unit by unit), `claims.json` (10 [Fock] records: 9 asserted, 1 unverifiable, 0 pending; scored by #291, the unverifiable one `fock/symmetry-vs-screening`) and `limits.json` (ten records: the tables are A100/V100 timings, §6 is whole-program wall-clock, no published register table for the §3.2 kernel, no `[JEngine]` cell moves, the screens' efficacy unmeasured on compact fixtures, the pair stream's counts, the baseline never a witness, the symmetry-versus-screening trade undecided (with the HGP split held since #409, only the fused MMD cell missing), what the fused `F` does not reach, and, #409, what the split HGP `J`/`K` does not reach), with **no `tables.json`**, by decision (pin 2). `hgp/cost.json` (#276) is `hgp/cost.py`'s FLOP triples per class, and `hgp/kernel_cost.json` (#284) `KernelCost`'s loop-nest reading of each fused kernel. `measured/fock_baseline.json` (#289, #290's addendum) is the cross-tree ptxas baseline: the generated `F` kernel's `sm_86` figures beside the predecessor's hand-spelled kernel for eleven `l ≤ 2` cells under the predecessor's flags, a **quality baseline, never a numeric witness** (common ancestry; no threshold, no ratchet). Its reading: the register boundary moved up one class (`pp\|ps` crossed to fitting), the trees agree where the HRR boundary counts match and we spill more where ours is larger (H1, untested, #332), and fitting follows the boundary target count, not L. |
| the generator | **live, for the whole `J` chain, the one-electron matrices and the fused `F`.** `KERNEL_GEN_CMD` in `devtools/config.sh` is `env PYTHONPATH=src python -m felizjueves.emit --one-e-l 2`, which writes **one ladder per kernel family** (#156) in the backend's spelling (`emit/cuda.py` / `emit/hip.py`). A third backend, `emit/gpu.py`, writes **one text for both compilers** for a consumer that builds either vendor from one tree: each unit includes `include/felizjueves/runtime.h` (FelizJueves's own two headers, `backend.h` and `runtime.h`, no GPU library behind them), which selects the vendor runtime from the compiler's own macros and types the shuffle mask as `FJ_LANE_MASK_T` (`uint32_t` / `uint64_t`); every integer in its text is `int32_t`, never a bare `int` (`CWriter.INT`), and every cast a `static_cast`, one over a sum of launch geometry (`CWriter.cast`; the vendor backends keep `int` and C casts, so no `cuda`/`hip` `source_sha256` moves); a unit with a shuffle also `static_assert`s `FJ_WAVEFRONT` (32 unless built `-DFJ_WAVEFRONT=64`) against the width it was emitted for, and one without has no width in its text. It is the Python API only (not a CLI `--backend`, not in `KERNEL_GEN_CMD`, nothing on a card); `test_emit_gpu.py` holds its goldens (`test/golden/gpu/`) and compiles the same string under nvcc and hipcc, with `conftest.INCLUDE` as the search path `devcache.localize` carries the headers from. A consumer chooses the kernel's name, never the kernel: by default `extern "C" __global__ void s_01(...)`, and with `emit(..., Specialization("s", (0, 1)))` one explicit specialization `s<0, 1>` of a kernel template; `emit_template(kernel, arch, spec, classes)` writes its header `s.h` once, which every unit includes: the primary, defined nowhere (fixed arity, `uint8_t La, Lb, ...` lettered by position, because nvcc 13.0 miscompiles a variadic one's host stub), one explicit specialization declared per class, and `s_kernels`, the consumer's dispatch table indexed by the class's args (`nullptr` where no class was emitted), so a missing class is a link error and a unit whose parameters drifted from the header is a compile error. Only the declarator moves. Each unit compiles as its own TU, launched through the table from another, which nvcc wants as relocatable device code (`-rdc=true`; without it, warning #20280). The device Boys is not carried in a `gpu` unit: `emit_boys()` returns its headers by file name, `eval_boys.h` (`gpu.BOYS_HEADER`) and the `eval_boys_{M}.h` it includes for `M` in 0..`gpu.BOYS_MAX` = 20 (goldens of each), and a unit with a fundamental `#include`s `eval_boys.h` after the runtime and calls `eval_boys<top>`; the consumer writes them all beside the units. `eval_boys<M>` is `BOYS_DEVICE` in this backend's spelling, `static` so every unit's copy stays its own, with the series loop replaced by `eval_boys_series<M>(2T)`: Horner over `gpu.boys_series(M)`, the Kummer coefficients each rounded once from the exact rational, to the least degree whose tail is under 2^-53 of the sum at the switch (40 at M = 0, 61 at M = 20), one `fma` a step and no division or data-dependent exit; `M > 20` is a `static_assert`. `test_eval_boys_device.py` runs it for every M in 0..20 over `BOYS_T` and every switch (worst 1.42e-15 below the switch on the RTX 3080; the RX 9060 XT compile-only, #431). The vendor backends still inline `fj_boys`, loop and all, so no `cuda`/`hip` text or pin moves. Units may still share a TU (`eval_boys.h` is `#pragma once`; records and dynamic shared arrays refused; `emit(..., block_dim=(x, y))` sizes a shared array whose extents are the block's at emit time, a flat `__shared__` declaration with the body unchanged, so `[p\|q]` specializes too, and `emit_template(..., block_dim=)` writes `{template}_block_dim` for the launcher; goldens `pq_22.specialized.cu`, `pq.h`); and `emit(..., any_wavefront=True)` writes **one text for both widths**, opt-in: every shuffle spelled with the header's `FJ_WAVEFRONT` / `FJ_FULL_LANE_MASK`, no `static_assert`, a block-sized shared array `Jsh[10 * FJ_WAVEFRONT]` under `block_dim=(1, gpu.WAVEFRONT)`, the build's `-DFJ_WAVEFRONT` choosing the width at compile time as WarpWraps's `WWR_WARP_SIZE` does (the default text and every pin unchanged; goldens `pq_22.any_wavefront.cu`, `pq.any_wavefront.h`; `test_any_wavefront_device.py` ran it on the RTX 3080 only, wave64 compiled for gfx942, never run) (goldens `s_11.specialized.cu`, `s.h`). With no ladder flag it writes 37 units: the `[p\|q]` ladder (`--L`, default L = 0..16), `build_pq_kernel(L)` — the `[p\|q]` kernel of `[JEngine, v2, Alg. 1]`, default split, **one batch, no atomics**, `pq_L{L}_p{lp}_q{lq}`, byte-identical to before #156 and pinned by `kernels.json`'s `source_sha256` — and the pair ladder (`--pair-l LMAX`, default 3, accepted 0..8), `dq_{la}_{lb}` and `jab_{la}{lb}` for every `la ≤ lb ≤ LMAX` (20 units through (ff)). `--L` alone writes only `[p\|q]`, `--pair-l` alone only the pairs. `--one-e-l [LMAX]` (#242; 3 when bare, accepted 0..8) **adds** the one-electron ladder, `s_{la}{lb}`, `t_{la}{lb}` and `v_{la}{lb}` for every `la ≤ lb ≤ LMAX`, beside whichever J ladders are selected; it is never in the no-flag output, so the 37 units and their pins are unchanged, and `KERNEL_GEN_CMD` asks for `--one-e-l 2` (18 units, through (dd)). The ladder writes `la ≤ lb` only: a real basis keys a class by shell index, so it also reaches `(1,0)`, `(2,0)`, `(2,1)`…, whose units `test/one_electron_driver.py` emits itself (the same text, flags and filter). The emitter casts a scalar parameter the body never reads to `(void)` (`DQ`'s `n_pair`, `JAB`'s `n_rows`), because hipcc's `-Wunused-parameter` otherwise fails #92's filter. The kernel tier generates, compiles all 55 units (37 J-chain, 18 one-electron) with nvcc or hipcc, and its device phase (`-m "gpu and <backend>"`) runs, among the rest, `test/test_kernel_device.py` (`J_p` and `fj_boys` on the card against `reference/` in fp64) and `test_j_chain_ondevice.py` (the pair units the tier wrote, and `J` through the whole chain), and `test_one_electron_device.py` / `test_hcore_device.py` (the one-electron units the tier wrote, `S`, `T`, `V` and `h`). **Split classes** are built by `build_pq_kernels(spec)` — one kernel per batch, `J_p` by atomic add — but are **not** written by `KERNEL_GEN_CMD`: the tests emit them (`test_emit_batched.py`, goldens for L = 4, `n_batches = 3`), and the device phase runs the subgrid's split cells and #113's asymmetric cells. **The last full kernel tier on both cards was #157's (PR #210):** 96 passed / 5 skipped on CUDA (RTX 3080), 97 passed / 5 skipped on HIP (RX 9060 XT), the skips the five opt-in recordings (ceilings, `[p\|q]` wall-clock, roofline counters, build wall-clock, sort-key A/B). **The class selector** (#89) is a second, separate CLI path: `python -m felizjueves.emit --class LA LB LC LD` (or `--shell-ls l...`, with `--n-batches N...` and `--admitted-only`) writes, per cell, the whole chain of a real class `(la lb\|lc ld)` a basis reaches, with an `order` file in launch order: `dq_{lc}_{ld}`, the `[p\|q]` batch kernels, `jab_{la}{lb}` (#156); `KERNEL_GEN_CMD` does not call it. `test_ao_j_ondevice.py` compiles the `[p\|q]` slice of those cells in the device phase and builds `J_µν` for H₂ and H₂O/STO-3G on each card against libcint — #89 (PR #118): worst `5.09e-15·scale_J` against a `1e-12` tolerance. **The launch-plan build** (#144) is a third path: `test/plan_driver.py` writes a device `main()` from a `launch.LaunchPlan` that `#include`s each class's emitted units, and `test_j_launched_ondevice.py` compiles and runs it on each card (worst `1.24e-14·scale_J` against libcint over the unscreened cells, identical on both cards). With `chain=` (#157) the same driver adds one `JAB` launch per bra class after every `[p\|q]` launch, and a separate `DQ` program (`plan_driver.dq_program`) supplies its kets: `J` through d with every arithmetic step generated, worst `1.24e-14·scale_J`, the same on both cards. What is **not generated**: the pair list and each record's inputs, the Schwarz bound, the sort and its `Dmax` read-back, the staging, and the launch code (the driver is test-side). The new kernels are **not measured** (no registers, spills, wall-clock or counters), and neither are `S`, `T` or `V`: their register and spill figures are compiler readings in PRs #297 and #298, not store rows. Dispatch is host-side (`performance/dispatch.py`) and reaches only a launch plan. The shipped kernels come from the **literal** plan, by decision: the searched order reaches `build_pq_kernel` only through the opt-in `tuned` lever, which was measured on both cards and refused as the default on wall-clock (#166). One batch is a decision too (`batchreport.ABOUT`); another count goes through the selector, `--class ... --n-batches 1 3 6` (legal: 1 or a positive multiple of 3), and `--n-batches` on the ladder path is rejected with exit 2. The plan-lever device sessions were #162's and #163's A/B device phases (PR #222: 83 passed / 2 skipped on each card; PR #224: 99 passed / 2 skipped on each card); #225 re-runs the full tier. **The last device sessions per card are milestone 12's**, `kernel-tier.sh --backend cuda\|hip -j 6 -- -k 'fock or f_cells'` (exit 0, `device: present`, 55 units compiled with no warning in generated text, then the F device tests): #288 (PR #321) and #290 (PR #324), each **25 passed / 1 skipped on the RTX 3080 and 25 passed / 1 skipped on the RX 9060 XT**, the skip the opt-in F wall-clock, run alone afterwards on each card and passed. They are a `-k` subset, not a full tier. **Milestone 11's tiers were all capped at L ≤ 8 by the user**, and no J text changed, so the L = 16 tier was not re-run; the widest was #244's deferred tier (generator `--L 0..8 --pair-l 3 --one-e-l 3`, 59 units): 95 passed / 3 skipped on CUDA, 96 passed / 3 skipped on HIP, the skips three opt-in recordings. **`K` is not on the ladder:** `KERNEL_GEN_CMD` writes no `K` unit; `kernel_k`'s kernels are emitted by the tests (`test/plan_driver.py`'s `k_driver`, `test_kernel_device.py`'s `K` harness) and by the measured driver (`python -m felizjueves.performance.measured --remeasure --subgrid --target k`). **`F` is not on the ladder either:** `python -m felizjueves.emit --fock-l LMAX` (0..2) or `--fock-class LA LB LC LD` (repeatable; on the 21-class ladder only, so `(10\|00)` exits 2) writes the fused kernel `f_{la}{lb}{lc}{ld}` (#287), a separate path that refuses `--L`, `--pair-l`, `--one-e-l`, `--class`, `--shell-ls` and `--n-batches`; the no-flag output holds no `f_` unit (asserted). All 21 classes are goldens per backend (`test/golden/{cuda,hip}/f_*`), the three over the 512 KiB file cap (`(pd\|pd)`, `(pd\|dd)`, `(dd\|dd)`) pinned by size and sha256 in `test/golden/fock_digests.json`; `(pd\|dd)` and `(dd\|dd)` compile in over 5 s and are `slow`. `test/fock_driver.py` emits the units the device tests run. **The split HGP `J`/`K` is not on the ladder either:** `python -m felizjueves.emit --hgp-j-l LMAX` / `--hgp-k-l LMAX` (any `LMAX >= 0`) or `--hgp-class L0 L1 L2 L3` (repeatable; both targets), with `--ordered` for the ordered mode instead of the canonical one, writes `hj_*` / `hk_*` (#407), a separate path that refuses every other ladder, the selector, the lever flags and `--n-batches`, never in the no-flag output and not in `KERNEL_GEN_CMD`. Its goldens are 126 units per backend (13 canonical and 50 ordered per target, every shell `l ≤ 2`, `test/golden/{cuda,hip}/h{j,k}_*`, no digests file), all compiled under both compilers with no warning in generated text; for that the emitter now casts an unread **array** parameter to `(void)` as well (`emit/_c.py`; no existing unit's text moved). `test/hjk_driver.py` emits the units the device test runs, and that ran on the RTX 3080 only (PR #477; the RX 9060 XT run attached to #431). **The DF metric is not on the ladder either:** `python -m felizjueves.emit --metric-l [LMAX]` (4 bare, any `LMAX >= 0`) **adds** `metric_{lP}{lQ}` for every `lP <= lQ` with `lP + lQ <= LMAX` beside the J ladders (#415), never in the no-flag output (its 37 units and pins unchanged) and not in `KERNEL_GEN_CMD`; it refuses `--class`, `--shell-ls`, `--fock-*` and `--n-batches`. The nine L <= 4 units are goldens per backend (`test/golden/{cuda,hip}/metric_*`); `test/metric_driver.py` emits the units the device test runs, and that ran on the RTX 3080 only (PR #430; #431). **The three-center integral is not on the ladder either:** `python -m felizjueves.emit --eri3-l [LMAX]` (4 bare, any `LMAX >= 0`, a negative one exits 2) **adds** `eri3_{la}{lb}{lP}` for every `la <= lb` with `la + lb + lP <= LMAX` beside the J ladders (#445), never in the no-flag output and not in `KERNEL_GEN_CMD`; it refuses `--class`, `--shell-ls`, `--fock-l`, `--fock-class` and `--n-batches`. Its names come from `kernel_eri3.eri3_name` (`_`-separated once any `l > 9`, so `(0,11,0)` and `(0,1,10)` no longer share `eri3_0110`). The 22 L <= 4 units are goldens per backend (`test/golden/{cuda,hip}/eri3_*`; the largest, `eri3_220.hip`, is 58,818 B, so no digests file); `test/eri3_driver.py` emits the units the device test runs, and that ran on the RTX 3080 only (PR #457; the RX 9060 XT run attached to #431). Set `KERNEL_GEN_CMD=` and the tier reports generate/compile DORMANT — do not read that as green. |
| `pyproject.toml` | **a virtual uv project** — no `[build-system]`, nothing installed. `src/` is importable from pytest only, via `pythonpath = ["src"]`. Dev tooling plus numpy, and a lock. |
| `docs/` | `references.md`, plus the derived notes in `implementation/`: `j_engine.md` (recurrences, the `E` transform, kernel contract, batching, the measured sections — batch count, asymmetric split, ceilings, counters, roofline rows, the ridge answer, the cross-read, and "The roofline, in one place" — pipeline, validation targets, "The `[JEngine]` claims" status table, the outside witness of `J`, the `τ` ladder, and "The launch plan": the sort key, the record granularity, the residency shape, the on-card results, the build wall-clock, the sort-key A/B, and the parity statement — and "The J chain as device code": which steps are generated, the `DQ` and `JAB` contracts, what stays host-side, and where the chain ran) `mild.md` (the MILD heuristic, the sieve, and MILD against the computed optimum), and `one_electron.md` (milestone 11: the `2π/p` identity, the `S`, `T`, `V` kernels' contract and shared scatter rule, what is reused from the J chain and what is not, where they ran, and the follow-ups; nothing in it is scored), and `hgp_fock.md` (milestone 12: the 2x2 in one place, `Target.F`, the HGP VRR and HRR with the two printed defects, the fused kernel, the digestion's weights derived, the pair stream, `Engine`, where it ran and what is measured, the cross-tree baseline read, what was not done, and the follow-ups #325–#334, and since milestone 14 "The split cell": `Spec.engine`, the two class modes, the HGP `J` and `K` kernels, the stream and its screens, where it ran (the RTX 3080 only; #431) and what it does not reach; no published cell is scored), and `metric.md` (milestone 15: the DF metric `(P|Q)`, `Target.METRIC`, the aux record `s = c·(π/α)^{3/2}`, the parity-dead targets, the schema-3 oracle, the tower-twin bound and why a per-target scale was unsound, where it ran (the RTX 3080 only; #431) and what is not done; nothing measured or scored), and `three_center.md` (milestone 16: the three-center `(mn|P)`, `Target.ERI3` as a triple that runs a quartet, the two sides (the bra's `K_AB` in `E`'s seed, the metric's aux record), the host reference and its dropped internal twin, the kernel contract, the two recorded decisions (the name `eri3` and the `[m][n][P]` layout, with DF-J's preference for aux-major slabs the known trade), the schema-4 oracle and its `1e-14` symmetry, the tolerances, where it ran (the RTX 3080 only; #431) and what is not done; nothing measured or scored). All six are **specifications** the code is read against, and each opens with a status block saying how much of it is implemented. Batching, MILD, the sieve, the per-batch DAG, the schedule, `PlanCost` and the optimum are implemented host-side in `plan/`; the recurrences (`mmd/recurrence.py`) and the `[p\|q]` kernel and its emission (`mmd/kernel.py`, `emit/`) are implemented for one batch and, with atomic write-back and per-batch launches, for a split class. `j_engine.md`'s batch-count and cross-read tables are kept equal to their data files by `measure/measure_batchreport.py` and `measure/measure_kernelrank.py` (`devtools/measure.sh check`), and its claims table by `test_claims.py`. The `E` transform, the pre-contraction, the bra back-transform and the `J_ab` scatter are implemented host-side in `ao/` and witnessed against libcint, and since milestone 7 also generated (`mmd/e_fragment.py`, `eri/j_coulomb/kernel_dq.py`, `eri/j_coulomb/kernel_jab.py`) and run on both cards. The launch plan, its oracle, the staged driver and dispatch are implemented (milestone 6). Not implemented: a device-resident `J`, launch code in the generator, any measurement of the new kernels, and dispatch inside the generator. The HGP recurrence, the fused `F` kernel, its pair stream and walker are implemented (milestone 12) and the kernel measured on both cards; not implemented there: `f` and above, a fused MMD `F`, a lever, a gradient, an `F` kept across SCF iterations. Tracked. |

When you run something, say which of these it touched.

## Setup

```bash
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint
pre-commit install --hook-type pre-push   # the fast test gate
devtools/doctor.sh                        # what is degraded, and the fix for each
```

`uv` is the only assumed host tool, and **`uv.lock` is the only place Python
dependency versions live** — after editing `pyproject.toml`, run `uv lock` and
commit the result; never `uv pip install`. Both images install with
`uv sync --frozen`, which fails rather than re-resolving, so a stale lock breaks
the image build instead of letting the container drift from the host.

The device toolchains live in the images, not in the repo's assumptions — though
on THIS host the CUDA 13.0 toolkit happens to be installed natively, so
`--backend cuda` works without a container and `--backend hip` does not.

## The tiers, and what each one proves

| Tier | Command | What a green run means |
|---|---|---|
| lint | `ruff check .` | `E,F,I,UP,B`, no formatter. |
| fast | `pytest -m "not cuda and not hip and not gpu and not slow and not oracle"` (`FAST_TEST_ARGS`; a bare `pytest` applies no marker filter; 2m10s on `JOBS=8`) | the Python side agrees with itself — and, through tracked fixtures, with libcint — **for classes with L ≤ 4** (`test/tiers.py`'s `FAST_L_MAX`, #419): anything that builds, emits, plans or integrates a class above it is `slow`, and a parametrized test is split at the cut, so the `[JEngine]` board replays and the plan search (every cell L 7–16) are slow-tier only; reads of stored rows and keys stay fast. Compiles nothing, launches nothing, needs no PySCF. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler (a warning located in generated text fails the compile phase, exit 1 — #92) → run on the card → compare against the host reference, and against libcint for `J_µν` (`test_ao_j_ondevice.py`). The device phase is `gpu and <backend> and not slow` (`KERNEL_TEST_ARGS`): L ≤ `FAST_L_MAX` on small inputs (H₂, two H₂, H₂O/STO-3G), the PR gate, 3m44s cold on the RTX 3080. |
| kernel, full | `devtools/kernel-tier.sh --backend cuda\|hip --full` | the same, every `gpu` test with **L ≤ 8** (`KERNEL_FULL_TEST_ARGS`): the L = 7 J and K subgrid cells, #259's L = 8 K check, the `[p\|q]` ladder through L = 8, the 6-31G / cc-pVDZ / cc-pVTZ builds, the per-class F lever checks. For a milestone's close or a kernel-text change above the cut. **Since #301 both kernel selections exclude `high_l`**, every case above `tiers.DEVICE_L_MAX = 8` (the L = 10, 13 and 16 J/K subgrid and asymmetric cells, the ladder's L = 9..16): one L = 10 K unit's compile alone outlasted an hour. They are kept: `pytest -m "gpu and <backend> and high_l"` runs them by hand. |
| measure | `devtools/measure.sh <sweep>` | **not a correctness tier**: timing, counters, ceilings, the recompile-against-stored-rows check and the stored reports recomputed (`check`). Nothing here blocks a PR. |
| oracle | `uv sync --group oracle`, then `pytest -m oracle` | the tracked fixtures in `test/oracle/fixtures/` still equal what PySCF computes today. Without the group these tests skip; a drift is news about PySCF until proven otherwise. |

**The kernel tier is the only place a numerical claim can be made.** No hosted
runner has a GPU. Two consequences worth internalising:

- **A device compiler needs no card.** nvcc and hipcc target an *architecture*,
  so the whole compile phase is green on a device-less box and proves nothing
  about a number.
- **A `gpu`-marked test SKIPS rather than fails** when no card is present, so a
  device-less run is green *and empty*. `devtools/doctor.sh` exists to make that
  visible, and `kernel-tier.sh`'s report prints `device ABSENT -- numbers
  unproven`.

`kernel-tier.sh` exits **2** for "could not run" (no compiler, no image, no
device group). Never read a 2 as a pass.

**A card is present when the kernel enumerates it, not when its device node
exists** (#431). `/dev/kfd` is there whenever `amdgpu` is loaded, bound card or
not — the module binds by PCI id, and kernel 6.8's in-tree one does not know
the RX 9060 XT's `1002:7590`, so this box takes `amdgpu` from DKMS and a kernel
upgrade it was not rebuilt for drops the card while leaving CUDA untouched. So
AMD presence is **an enumerated GPU node in kfd's topology** (a nonzero
`gfx_target_version`), asked in one rule by `lib.sh`'s `amd_gpu_present`,
`conftest._has_amd_device` and `doctor.sh`'s enumeration — no ROCm needed, and
the same answer inside a container as on the host. Before that, HIP reported
`device: present` with no usable card and every `gpu` test failed instead of
skipping, which from inside a container reads as broken passthrough. When HIP
goes dark while CUDA works, run **doctor** first: it names the unbound device
and the `dkms autoinstall` that fixes it.

## What is gated, and what is not

- **`git push`** runs the fast pytest tier, and only when the push touches a
  `.py` (the `files: \.py$` filter on the pre-push hook).
  `FAST_TEST_ARGS` in `devtools/config.sh` excludes the `cuda`/`hip`/`gpu`
  markers, so the gate cannot start invoking a device compiler — which it
  otherwise would, inside a container where both are present — and `slow`
  and `oracle` too.
- **The card is locked by the tests, not by a wrapper.** Every `gpu` test's
  device fixture (`test/conftest.py`, #91) takes an exclusive `flock` on the
  card at `GPU_LOCK_PATH` in `devtools/config.sh` (`/tmp/fj-gpu-{backend}.lock`)
  — under xdist the controller holds it — and a harness that carries the
  VRAM probe checks free memory before its launch, failing with both figures
  rather than running out of memory. Two
  runs on one card serialise; an outer `flock` on the same path is recognised
  as an ancestor and is redundant, not harmful.
- **The kernel tier is deliberately NOT a hook.** An unrolled high-L kernel is
  slow to compile and the tier costs minutes; a gate that costs minutes is one
  people bypass with `--no-verify`. Run it before a PR and say which backend.
- **What blocks a PR, and its budget: under five minutes each.** The fast
  tier (2m10s) and the kernel tier per card without `--full` (3m44s cold on
  the RTX 3080). Both test correctness only. A test that would push either
  past five minutes is `slow` (by L above `FAST_L_MAX`, or by a large input),
  never deleted: `--full` and a bare `pytest` still run it. Measurement is
  never a test: it goes in `measure/`, behind `devtools/measure.sh`.
- **There is no CI yet.** No `.github/` in this repo. When it lands it can lint,
  run the fast tier, and compile — never run a kernel.
- **`protect-main.py` guards the primary (`main`) checkout** and is on by
  default. Set `CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing it
  (repo setup, say); the guard's docstring has the rationale, and every bypass
  is logged to `.logs/protect-main.log`.

## Containers: four files, one diamond

```
                Dockerfile.base
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses install-rocm.sh, not the hip image)
```

| File | What it is |
|---|---|
| `docker/Dockerfile.base` | g++ and clang, pinned CMake/Ninja/sccache/`gh`, a uv venv at `/opt/venv`. No GPU SDK. |
| `docker/Dockerfile.cuda` | `base` + the CUDA toolkit, `sm_86`, Nsight Systems. **The default.** |
| `docker/Dockerfile.hip` | `base` + ROCm's HIP compiler and profilers, `gfx1200`. No CUDA. |
| `docker/Dockerfile.combined` | `cuda` + ROCm. Both device compilers. |

The files chain by **tag**, not by stage — each child opens `FROM
${PARENT_IMAGE}` — so **build them only with `docker/build.sh
<base|cuda|hip|combined>`**, which walks the chain. Build a child by hand with
no parent tagged and docker tries to *pull* it and fails with `pull access
denied`.

**What is deliberately absent, because this tooling was adapted from a dense
linear algebra project:** no reference LAPACK, no cuTENSOR/NCCL/nvCOMP/cuGraph,
no hipCOMP, no clang-20 + libc++ module manifest — and ROCm is installed as the
**compiler plus profilers**, not as `rocm-hip-sdk`'s ~19GB of math libraries
that no generated kernel calls. **sccache is the one thing kept against that
grain**, because it caches nvcc and hipcc output and the kernel tier recompiles
the same generated translation units across runs. Add a library only with the
code that links it, and say what does.

**Two agents are pinned in the three GPU files**, as the last two installs in
each — `docker/install-claude-code.sh`, then `docker/install-agy.sh`, with only
the `USER ubuntu` switch and a toolchain check after them. So each
version is declared three times — bump Claude Code with
`devtools/claude-version.sh --apply`, which rewrites all three, and `agy` with
`devtools/agy-version.sh --apply`; `doctor.sh` reports when either disagrees
across the files or has fallen behind upstream.

`agy`'s pin is **two lines per file** (`ARG AGY_RELEASE` plus `ARG AGY_SHA512`)
because upstream's manifest serves only the current release: a version alone is
not addressable, so the pin is a release id — version plus an opaque build id —
and the sha512 of that exact tarball, which `install-agy.sh` verifies rather
than records. `agy-version.sh` rewrites the pair as one unit and reports a file
whose hash drifted from its release as `sha-differs`; `--verify` re-hashes the
pinned tarball (~62MB) without touching an image. `agy` is installed in the
images but **not logged in** — its state lives in `$HOME/.gemini`, which
nothing here binds.

`docker/README.md` has the build args, the compose services and the AMD
passthrough; the **devbox** skill drives it.

## Conventions

- **Python ≥3.12, `from __future__ import annotations` everywhere.** 3.12 and
  not 3.13 because that is the system interpreter on both sides (this host and
  `ubuntu:24.04`); raising it makes `uv sync` download a managed CPython in the
  image, which `UV_PYTHON_INSTALL_DIR` exists to keep reachable by uid 1000.
- **Lint is narrow (`E,F,I,UP,B`) with no formatter hook**, on purpose: the
  output that matters is generated kernel text, and a formatter arguing with the
  generator over its own string layout is noise. `ruff check .` is clean.
- **Generated kernel text that a test asserts against is a GOLDEN FILE** — it is
  tracked, reviewed, and a diff in it is the whole signal when the generator
  changes. There is no `*.cu` line in `.gitignore` for that reason; scratch
  emission goes to `.kernel-tier/`.
- **The markers separate the two axes**, and the distinction decides what a run
  proves: `cuda`/`hip` need a *compiler*, `gpu` needs a *card*. A test asks for
  exactly the fixtures it needs (`test/conftest.py`), so it skips for the right
  reason. Two more: `oracle` needs PySCF (to *regenerate* a fixture — a
  comparison against a tracked fixture carries no marker), and `claim(id)`
  ties a test to a record in `data/literature/*/claims.json`.
- **The host oracle is numpy, in the device's precision.** Computing a float32
  reference in float64 and then loosening the tolerance to match measures
  something other than the kernel.
- **No `-use_fast_math`, and nothing that reassociates arithmetic**, in
  `NVCC_FLAGS`/`HIPCC_FLAGS`. The tier's only assertion is agreement with the
  reference to a tolerance; a flag that changes the arithmetic changes the thing
  being measured. Add one per experiment, never to the defaults.
- **`devtools/config.sh` is the one file that knows the project.** The scripts
  are meant to survive being copied into the next repo unchanged — put a new
  setting there, not in a script. Two things it cannot reach and that must be
  kept in step by hand: the `.devcontainer/*/devcontainer.json` files, and the
  `FJ_` environment prefix spelled in `devcontainer.sh` and in those files.
- **Install the pre-commit hook from whichever side you commit on.** It bakes an
  absolute `INSTALL_PYTHON` into one file shared by every worktree; installed
  from the wrong side, commits fail with `No module named 'pre_commit'`. The
  **doctor** skill has the fix.

## Skills

Most of the how-to lives in `.claude/skills/`, and the detailed rationale lives
in the code it describes — script headers (`--help` prints them), `README.md`
files, and `docs/`. This file is the map; reach for the skill when you act.

| Skill | Reach for it to… |
|---|---|
| [test](skills/test/SKILL.md) | run the suites the right way and read the result honestly — the fast pytest tier, the kernel tier on each backend, the marker/fixture split, and what a green run does *not* prove. |
| [devbox](skills/devbox/SKILL.md) | drive the containers and the container-backed **sibling** worktrees — `devcontainer.sh` (up/rebuild/shell/test/down) and `worktree.sh` (add/rm/sync/gc). |
| [worktree](skills/worktree/SKILL.md) | create and clean up the lightweight `.claude/worktrees/<name>` checkouts — no container, cheap and disposable. |
| [doctor](skills/doctor/SKILL.md) | diagnose a degraded environment — `doctor.sh` plus the failures it cannot see. |
| [pr](skills/pr/SKILL.md) | ship the current work: commit → push → PR → merge, with this repo's gotchas. |
| [issue](skills/issue/SKILL.md) | resolve one GitHub issue end to end in a fresh worktree, tested on **both** cards, then `/pr full`. |
| [milestone](skills/milestone/SKILL.md) | turn a plan into a GitHub milestone plus a DAG of PR-sized issues. |
| [milestone-run](skills/milestone-run/SKILL.md) | execute a milestone with one subagent per ready issue. |
| [audit](skills/audit/SKILL.md) | verify that documentation (skills, READMEs, this file) still matches reality — extract each claim, check it, report drift. |
