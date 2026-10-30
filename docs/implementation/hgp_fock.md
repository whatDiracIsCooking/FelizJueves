# The fused Fock build on the HGP recurrence: `[Fock]` §3

**Status: implemented, and run on both cards (milestone 12, #271–#292,
PRs #303–#324).** This is the derived note for engine #2. It covers the
Head-Gordon–Pople (HGP) recurrence of `[Fock]` §3.1 and the **fused** Fock
kernel of §3.2 built on it, which digests every contracted ERI straight
into the two-electron Fock matrix `G = J − K/2` and uses 8-fold
permutational symmetry. No other build in this repo digests through that
symmetry. It is the contract the code was built against, and the one place
that says what is generated, what is host-side, where it ran, and what it
does not reach. **Energy order only**, for one density, rebuilt on every
call.

**Since milestone 14 (#401–#408, PRs #469–#477) it also holds the 2x2's
(HGP, split) cell**: `Target.J` and `Target.K` on `Engine.HGP`, every ordered
quartet at weight 1. It was interpreted at every class with L ≤ 4 and run on
the **RTX 3080 only** ("The split cell", below). Nothing about it is timed.

**Nothing here is scored against a published table.** `[Fock]` is
`docs/references.md` entry 23 (ct4c00994). The text read is the arXiv
preprint, arXiv:2407.21445v1, because the journal version is paywalled, so
its section, equation and table numbers may differ from the journal's
(`data/literature/fock/README.md`). Its tables are A100 and V100 wall-clock,
§6 is whole-program timing on Perlmutter, and **the paper prints no
register table for the §3.2 kernel at all**. So milestone 12 scores no
published cell, adds no `tables.json`, and moves no `[JEngine]` cell, floor,
claim or status (pin 2; `data/literature/fock/limits.json`). What it does
score is `[Fock]` §3's non-numeric claims:
**10 [Fock] records: 9 asserted, 1 unverifiable, 0 pending**
(`data/literature/fock/claims.json`, scored by #291). Correctness is
witnessed only from outside: `eri/fock/fock.py`'s every-ordered-quartet sweep with
MMD ERIs, and libcint's `vj − vk/2` through the tracked fixtures.

## The 2x2, in one place

`[Fock]` confounds two decisions and says so (§1, §4.3). §3's opt-Brc is
**HGP + fused + 8-fold symmetry + six-block screen**. §4's opt-UM is
**MMD + split `J`/`K` + one-block screen**. "Split `J` and `K`" and "give up
8-fold symmetry" are the same decision, so the paper's two schemes sit on
the diagonal of a 2x2 table:

| | split `J` / `K`, every ordered quartet | fused `F`, 8-fold symmetry |
|---|---|---|
| **MMD** | **held**: `Target.J` (milestones 2–7), `Target.K` (milestone 10, still open) | follow-up, #326 |
| **HGP** | **held**: `Target.J` / `Target.K` on `Engine.HGP` (milestone 14) | **held**: `Target.F` (milestone 12) |

**FJ now holds both of the paper's diagonal cells under one IR (`ir/__init__.py`),
one harness (`interp.py`, `emit/`, the tracked fixtures, the measured store
and its gate), and two cards** (RTX 3080 `sm_86`, RX 9060 XT `gfx1200`).
The two cells have not reached the cards equally. The MMD split cell's `J`
is generated end to end and ran through d on both cards. Its `K` has run on
both cards only for classes with L ≤ 2 (#206, and one `(dd|dd)` cell on the
3080, #259), and its deferred tier is #231. The HGP fused cell ran every
class through `(dd|dd)` on both cards (#288, #290). **One off-diagonal
cell now exists:** the split HGP `J`/`K` (milestone 14). Pin 1 had kept it
out of milestone 12, because it is the predecessor's `brc26` row JK, not
§3. It ran every class with L ≤ 4 on the RTX 3080 only; its RX 9060 XT
run has not been made and no open issue tracks it (#431 closed on
2026-10-17 with the card's detection fixed, not with the run). **The fused
MMD `F` (#326) does not exist.** The
HGP row now holds both schemes on one recurrence, so timing `Target.F`
against HGP `Target.J + Target.K` would no longer confound the recurrence
with the scheme. **Only the two diagonal cells have been timed against
each other**, per class, (ss|ss) and (pp|pp) on polyglycine/6-31G ("The
diagonal cells timed per class", below): the fused HGP `F` was faster in
every cell on both cards. The HGP split, the comparison that would separate
the scheme from the recurrence, is untimed, and the cards are not the
paper's, so `fock/symmetry-vs-screening` stays unverifiable (#331).

## The key: `Target.F`

`Spec` (#272) gains `Target.F`, `AM_LENGTH = 4`, `OPERATOR_OF = COULOMB`. Its
quartet is in **AO order** `(la, lb | lc, ld)`, the opposite of
`Target.K`'s Hermite pairing `(la, lc, lb, ld)`. A mix-up between the two is
silent in both directions, and `test_spec_f.py` shows that the same four
shells give different keys under the two. `n_batches` is 1 only, refused
with its own reason (`spec._F_BATCH_REFUSAL`): an HGP class has no Hermite
targets at all. The message names VRR target grouping as the field's future
meaning. The plan layer **refuses** a `Target.F` spec (`plan.dag.refuse_hgp`,
called by `dag.build` and `pipeline.tuned_plan`), because `total_am` indexes
a `[JEngine]` row an HGP class does not have (pin 11). Every pre-existing key
is byte-identical.

The ladder is **21 classes**: every AO-order quartet with each shell
`l ≤ 2`, canonicalised `la ≤ lb`, `lc ≤ ld` and bra pair ≤ ket pair, which
is the six pairs (ss)…(dd) taken as unordered pairs of pairs, topping out at
`(dd|dd)` (`hgp.vrr.CLASSES`, pin 3). The predecessor canonicalises
`a ≥ b`. Nothing at `f` or above is built (#330).

## The recurrence

`src/felizjueves/hgp/` is engine #2's package. A layering rule keeps it off
`plan/` and `reference/` (pin 11; `test_layering.py`, rules `hgp` and
`hgp_builder`).

**The VRR, eqs 25–26** (`hgp/vrr.py`, #273), over nodes
`(e, f, m) = [e,0|f,0]^(m)`:

    [e+1_d,0|f,0]^(m) = PA_d [e|f]^(m) + WP_d [e|f]^(m+1)
                      + e_d/(2p)    ([e−1_d|f]^(m) − ρ/p [e−1_d|f]^(m+1))
                      + f_d/(2(p+q)) [e|f−1_d]^(m+1)

    [e,0|f+1_d,0]^(m) = QC_d [e|f]^(m) + WQ_d [e|f]^(m+1)
                      + f_d/(2q)    ([e|f−1_d]^(m) − ρ/q [e|f−1_d]^(m+1))
                      + e_d/(2(p+q)) [e−1_d|f]^(m+1)

with each lower term absent when its index is zero, and `ρ = pq/(p+q)`.
**The DAG is not graded by `m`**, and that is asserted, not described. MMD's
`(r, m)` DAG raises `m` by one on every edge. HGP's has same-`m` edges
(`PA`, `QC`, `1/(2p)`, `1/(2q)`), so its order has to be constructed. This
is the structural reason the plan layer cannot absorb it. The build order
is the bra first with the ket held at zero, then the ket by `|f|` ascending,
because eq 26's `e_d/(2(p+q))` term reads a lower bra at `m+1`. Dead-node
pruning is on by default. At `(dd|dd)` the pruned DAG has 2,330 nodes, and
1,089 of them are at `m = 0`.

**The HRR, eq 28** (`hgp/hrr.py`, #274), over **contracted** nodes
`(a, b, c, d)` with no `m`:

    (a, b+1_k | c, d) = (a+1_k, b | c, d) + AB_k (a, b | c, d)
    (a, b | c, d+1_k) = (a, b | c+1_k, d) + CD_k (a, b | c, d)

It holds only on contracted values, because `AB` is common to every
primitive pair of the shell pair. So the `K⁴` contraction sum completes
before the HRR runs, and that forces the kernel's shape (pin 5, below). The
ket is reduced first, so in build order the bra HRR runs first. Both side
orders are pinned.

**The boundary set is the bounded target set. This is the decision of
record.** The HRR's leaves are exactly the VRR's targets: every `(e, f)`
with `la ≤ |e| ≤ la+lb` and `lc ≤ |f| ≤ lc+ld` (`hgp.vrr.target_set`,
asserted equal for all 21 classes). At `(dd|dd)` that is **31² = 961**, not
the milestone issues' **1,225 = 35²**, which is the same set with the lower
bound dropped (the `m = 0` layer of the unpruned forward build). #274's
issue text gives 6,254 HRR statements, and that figure confirms the reading:
961 boundary reads plus 5,293 two-term updates is 6,254, while 1,225
boundary values would give 6,518. So 1,225 was a misprint and 961 is pinned
(`test_kernel_f.py::test_dd_dd_is_961_boundary_values_not_1225`). The
predecessor's (a3) table is not on this box, so every count here was
computed and pinned, not copied.

**As IR** (`hgp/recurrence.py`, #279). `vrr_body` and `hrr_body` emit one
`Decl` per DAG node in build order. The direction is read off the DAG, never
re-derived. HGP needed two widenings of `ir/__init__.py`'s instruction form (#277):

- **a signed coefficient**, because the `(m+1)` lowering term's multiplier
  is `−e_d`;
- **two derived scalars**, `INV2P_ALPHA_OVER_P` and `INV2Q_ALPHA_OVER_Q`,
  for the only products of two named scalars that any DAG holds. `Term`
  keeps a single factor.

`PA_d` is reused, not duplicated: it means `P − A` in both engines. The HRR
fitted the existing form unchanged. The interpreter and both emitters needed
no change.

**Two defects found in the printed text, both asserted by #291:**

- **Eq 26's sign** (`fock/eq26-ket-vrr-sign`). Under the `XY = X − Y`
  convention that eqs 25 and 28 fix, the ket's `(m+1)` displacement is
  `W − Q = +p/(p+q) PQ`. Eq 26 prints `−`. The kernel binds `WQ` as the
  derived value, so the sign lives in the bound value and not in the IR.
  `test_hgp_recurrence.py` runs both signs against `eri/eri.py` on a quartet
  with `P ≠ Q`. The derived sign is within `1e-13·scale`, and the printed
  one is off by more than `1e-2·scale`. #281 repeats the check on
  contracted quartets.
- **Eq 14's seed** (`fock/eq14-seed-inconsistent`). Derived against FJ's one
  fundamental (`reference/pq.py`), the seed is
  `[00|00]^(m) = U_P U_Q (2/√π) θ F_m(T)`, which is §3.1's own `R0_m`.
  Eq 14 prints `U_P U_Q θ^m F_m(T)`. That lacks `(2/√π) θ`, and its `θ^m`
  breaks the Obara–Saika coefficients. As printed, eq 14 misses `(ss|ss)`.
  With its normalisation repaired, it still misses `(sp|ss)`, one VRR step
  up.

**The cross-engine pin** (#281): `eri/hgp_drive.py` interprets the VRR per
primitive quartet, completes the `K⁴` boundary sum and runs the HRR once.
Its signature is `eri.eri.eri_block`'s. All 21 classes on HF/cc-pVTZ,
H₂O/cc-pVDZ and He–Ne agree with MMD within
`1e-12·(scale_mmd + scale_hgp)` (worst 5.4e-16), and agree with libcint
(worst 2.9e-14). The HGP scale is needed: the HRR's transfers cancel in a
way MMD's scale cannot see.

**FLOPs** (`hgp/cost.py`, #276; `data/hgp/cost.json`) are read off the two
DAGs, never the IR, and tiered per contracted quartet as
`x K⁴ + y K² + z`. `(dd|dd)` is (16,427, 48, 10,602) pruned, a total of
27,077 against 36,078 unpruned. The model is **calibrated against
nothing**: `[Fock]` publishes no FLOP table, and the model is not compared
with `plan/cost.py`'s instrument.

## The fused kernel (`kernel_f.py`, #282, #283)

**The nest is §3.2's** (pin 7; not `Target.J`'s warp per bra record):

    bra = blockIdx.x                       one block per bra shell pair
    ab_sh[128][8] <- |ab)                  the bra's primitive pairs, shared
    for k = threadIdx.x; k < n_ket; k += blockDim.x:   one thread per
                                           contracted quartet
        skip if k < diag·bra               the canonical half
        skip if G_ab G_cd dmax < tau       the six-block screen
        c_(e,f) = 0                        |boundary| accumulators
        for kcd < K_CD: for kab < K_AB:    the contraction, INSIDE
            geometry, F_m(T), the VRR; c_(e,f) += [e0|f0]^(0)   (eq 27)
        the HRR on c_(e,f)                 after the loops: contracted
        digest the ERIs into six buffers; AtomicAdd five of them
    AtomicAdd f_ab                         once per thread

and the host closes `F ← F + Fᵀ` after the last launch (`kernel_f.close`).

**The contraction loop is inside the kernel, and that is forced** (pin 5).
Eq 28 needs contracted values, so all `|boundary|` accumulators are live
across the `K⁴` loop: 961 at `(dd|dd)`. Staging the boundary instead is
priced out at 9.8 KB per contracted quartet at `(dd|dd)`. `ir.lower.KernelCost`
(#284) now models the loop nest. Its `carried` field equals `|boundary|` in
all 21 classes, and at `(dd|dd)` `peak_live` is 2,140
(`data/hgp/kernel_cost.json`).

**The screen is a skip, never a break** (pin 8; `fock/skip-not-break`). The
six-block bound `G_ab G_cd max(|w| Dsh)` is not monotone along any one sort,
so a screened quartet cannot speak for the kets after it. The kernel holds
no `Break` anywhere. This is the inverse of `Target.J`'s and `Target.K`'s
sorted early break, and the test compares the emitted text with `Target.J`'s
`break;`.

**The staging is per thread**, because the IR has no barrier. Every thread
copies the whole `|ab)` record into one shared `ab_sh` of 8,192 B, and both
compilers report that size for all 21 classes. That is correct without a
`__syncthreads()`, but it is not the bandwidth saving §3.2 stages for. The
cooperative load is #333. `MAX_PRIM_PAIRS = 128` is a refusal past the cap,
never an overflow.

## The digestion: weights derived, never transcribed

The predecessor shipped this digestion with the Coulomb blocks weighted 1
instead of 2. That is wrong in every symmetric Fock matrix, and it went
unnoticed because it was checked against a re-typing of itself (pin 6). So
here the weights are **computed** (`kernel_f.derive_weights`,
`kernel_f.degeneracy`), and the witness is a sweep with no symmetry in it.

**The witness** (`eri/fock/fock.py`, #275) sweeps every ordered shell quartet once
at weight 1: `J[a][b] += D[c][d] x`, `K[a][c] += D[b][d] x`, and
`G = J − K/2`. The `1/2` is derived from closed-shell RHF
(`G^σ = J[D] − K[P^σ]` with `P^σ = D/N_SPIN`), not read off PySCF's
`vj − vk/2`. The ERI is injected (`eri=`), so the same sweep runs on MMD or
HGP.

**The derivation.** The kernel visits one representative per orbit of the
order-8 group generated by `a↔b`, `c↔d` and `(ab)↔(cd)`. Enumerate the eight
images. Each image puts its Coulomb term at its own `(p, q)` and its
exchange term at `(p, r)`, and every output lands on one of six unordered
position pairs with the complementary pair as its density. Count, per pair,
the members that land **in order**: `{a,b}` and `{c,d}` take two Coulomb
terms each, and `{a,c}`, `{a,d}`, `{b,c}`, `{b,d}` take one exchange term
each. The host's `close` supplies the transposed ones exactly. So per unique
quartet:

    F_ab += 2 D_cd x      F_cd += 2 D_ab x
    F_ac −= ½ D_bd x      F_ad −= ½ D_bc x      F_bc −= ½ D_ad x      F_bd −= ½ D_ac x

**Coulomb 2** is the fraction 1 times two in-order members. **Exchange −½**
is `−1/N_SPIN` times one. When shells coincide, the representative is scaled
by `deg = 1/|Stab| = |orbit|/8 = (½ if a=b)(½ if c=d)(½ if P=Q)`, by
orbit-stabiliser. The kernel reads `[P=Q]` only in a diagonal launch. The
full member table and the six-case orbit table are in `kernel_f.py`'s
docstring.

**Held** (`test_kernel_f.py`): for all 21 classes, one unique quartet's
closed deposit equals the sweep restricted to that quartet's orbit within
`1e-13·scale_G` (worst about 1e-15). The orbit is enumerated by brute force,
not taken from the group code. Every degeneracy case is exercised, diagonal
batches included. **The mutations fail**: a Coulomb weight of 1 fails by
more than `1e6×` the tolerance in every class, and so do a flipped exchange
sign, a dropped `deg`, and Fig. 1's printed unit weights
(`fock/fig1-digestion-unweighted`). The `AtomicAdd` count per class equals
the six blocks' size exactly (216 at `(dd|dd)`), and its ratio to the ERI
count falls from 6 to 1/6. Atomics outnumber ERIs up to `(0,1|1,2)`
(`fock/predigest-bounds-atomics`, a statement count, not a timing).

## The batch plan: the pair stream (`launch_f.py`, #285)

`f_launches(basis, D, *, bound, tau, n) → FPlan` is the plan as a value, a
sibling of `launch.py` and `launch_k.py`. The bound is injected
(`ao.pairs.primitive_schwarz_bounds`), and no ERI is evaluated.

- **Contracted pairs**, one per unordered shell pair, `la ≤ lb`.
- **Pruning.** Primitive pairs are sorted descending by `G`, and the longest
  tail whose sum is below `ε = τ/(2 G_max)` is deleted, so no ERI moves by
  `τ` or more. A pair with every primitive pair pruned is deleted outright.
- **Degree-uniform batches** keyed on `(la, lb, K_AB, magnitude)`, with no
  padding. Pruning can therefore add batches by splitting degrees.
- **Eq. 29.** Magnitudes are `m_i = τ^(i/n)`. The pre-launch screen removes
  a batch pair exactly when `i + j ≥ n`, and the plan refuses itself if a
  removed pair's `max G_I · max G_J` is ever `≥ τ` in floating point.

**What is built is the sound reading, and it is the one Fig. 1 shows.** Two
defects are asserted by #291 on the computed two-electron `F`. The printed
"largest magnitude less than `G_ab`" is a *lower* bound, while Fig. 1 labels
the magnitudes "Upper bound" (`fock/batch-magnitude-bound-direction`). The
printed `i + j ≤ n` would remove every batch pair at `n = 2`
(`fock/eq29-screen-inequality-reversed`). The printed forms ship only for
the tests that show the difference.

**The walker** (`walk_f.py`, #286) is the oracle in milestone 6's pattern.
It reads the plan by duck type without importing `launch_f`, refuses a plan
that does not cover the basis, rebuilds each record from the shells, runs
every launch through an injected executor, and splits every unique quartet
three ways: launched, pre-launch screened, or touching a deleted pair. At
`τ = 0` the interpreted whole-molecule `F` matches the MMD sweep within
`1e-13·scale_G` and libcint within `1e-12·scale_G`. At `τ > 0` every skipped
quartet is checked against the exact contracted ERIs.

## `Engine`, extracted last (`engine.py`, #292)

The protocol was written after both engines existed (pin 12), so it is a
residue: `name`, `target`, `eri`, `eri_and_scale` and `kernel(spec)`, with
`Mmd(pq=)` and `Hgp(boys=)`. Milestone 14 (#406) adds `j(spec)` and
`k(spec)`, because both engines now build a split `J` and `K`. `pair_bound` is not a member, because only MMD
has one. **The milestone's primary report is what was pushed back down**
into the MMD engine as engine-private: the `[JEngine]` batch partition,
MILD, the sieve and its levers, the graded `(r, m)` DAG, the
`total_am`-keyed table row, `plan/cost.py`'s instrument, the thread mapping
and launch plan, and the J chain's pre- and post-kernels. The list is
`engine.py`'s docstring, and #327 is the follow-up that asks, item by item,
whether HGP should get an analogue. A layering rule makes `engine` the one
module that names both engines through their ERI and kernel entry points.
The two generators, `emit/__main__` and `measured`, name each engine's
kernel builder only (`kernel`, `kernel_f`), under a narrow exemption added
by #287 and #288 (`test_layering.py`'s `GENERATORS`). `launch_f` no longer
borrows private helpers: it takes the public `launch_common.sorted_desc` /
`nonneg` and `ao.density`'s checks (#328; "Named follow-ups", below).

## The generator

`python -m felizjueves.emit --fock-l LMAX` (0..2) or
`--fock-class LA LB LC LD` (repeatable, on the ladder only, so `(10|00)`
exits 2) writes `f_{la}{lb}{lc}{ld}.{cu,hip}` (#287). It is a separate path
from `--L`, `--pair-l`, `--one-e-l` and `--class`, and **`KERNEL_GEN_CMD`
does not call it**: the no-flag output is still the 37 J-chain units, and a
test asserts it holds no `f_` unit. All 21 classes are goldens on both
backends (`test/golden/{cuda,hip}/f_*`). The three over the repo's 512 KiB
file cap, `(pd|pd)`, `(pd|dd)` and `(dd|dd)`, are pinned by byte count and
sha256 in `test/golden/fock_digests.json`. HIP's text equals CUDA's apart
from the prologue. There is no `Raw` escape hatch. `(pd|dd)` and `(dd|dd)`
compile in over 5 s, about 15 s at `(dd|dd)` on each backend, so they are
`slow`.

## Where it ran

**On both cards**, the RTX 3080 (`sm_86`, nvcc 13.0) and the RX 9060 XT
(`gfx1200`, hipcc / ROCm 7.2.4), through `test/fock_driver.py` and
`test/test_fock_ondevice.py` (#288, PR #321; #290, PR #324). The driver
records the launches `walk_f` makes, and one binary serves both the checks
and the timing. Since #657 that binary is the shared runner (#649,
`test/harness/`) over the kernel set's units, each check or timing a
sequence of the recorded launches; that port has run through the real
runner on the host against a scratch stub runtime only, not yet on a card.

- **Part 1, per class.** H₂O/cc-pVDZ's plan at `τ = 0` reaches all 21
  classes. Each class's launches on the card are held to the same launches
  interpreted (`2·TOL`) and to `eri.fock.fock`'s sweep with MMD ERIs (`TOL =
  1e-13·scale_G`). Worst against the sweep: 2.2e-14 at `(00|02)` over the
  default five, and 1.04e-14 (CUDA) / 9.49e-15 (HIP) at `(02|02)` over the
  16 deferred classes.
- **Part 2, whole molecule.** H₂O/STO-3G at `τ = 0` and at `τ = 0.1, n = 4`
  (the screens fire), H₂O/6-31G, and H₂O/cc-pVDZ. Each is held to the
  interpreted walk and, at `τ = 0`, to libcint's `vj − vk/2` within
  `1e-12·scale_G`: 3.27e-15, 6.81e-15 and 7.35e-15 (CUDA) / 7.34e-15
  (HIP). Runs are not bit-identical because the fp64 atomics' order varies.
  The run-to-run difference is at most 3.0e-16.
- **Both sessions gave 25 passed / 1 skipped on each card.** The skip is the
  opt-in F wall-clock test, which was run alone afterwards on each card and
  passed. The `slow` classes were not deselected. No tolerance was changed.

**Measured**: `data/measured/kernels.json` holds **42 F rows**, all 21
classes × `sm_86` and `gfx1200`, each with registers, spill, static shared
bytes and wall-clock. They were timed behind `gate.wait_quiet` with every
gate reading present and 0 discarded phases. The timed launch is 2048 × 128
contracted quartets, `K = 9`, off-diagonal, on a wide input (`runtime_ns`)
and a near one. A few rows (milliseconds per launch, wide input):

| class | targets | `sm_86` regs / spill stores (B) / ms | `gfx1200` VGPRs / spilled regs / ms |
|---|---|---|---|
| `(ss\|ss)` | 1 | 72 / 0 / 84.7 | 46 / 0 / 119.2 |
| `(sp\|sp)` | 16 | 156 / 0 / 246.6 | 125 / 2 SGPR / 321.6 |
| `(ss\|dd)` | 31 | 168 / 0 / 446.9 | 191 / 68 / 584.4 |
| `(sp\|pd)` | 76 | 255 / 104 / 418.9 | 192 / 410 / 552.4 |
| `(pp\|pp)` | 81 | 255 / 1,264 / 415.0 | 192 / 525 / 544.5 |
| `(dd\|dd)` | 961 | 255 / 97,464 / 2,333.6 | 192 / 16,043 / 3,701.7 |

So `(dd|dd)` costs about 2.3 s per launch on the 3080 and 3.7 s on the 9060
XT. From `(pd|dd)` up, the near and wide inputs cost about the same, so the
spill traffic dominates the Boys evaluation. Compare these within one arch
only: nvcc reports spill in bytes and hipcc in registers. **Nothing is
compared against `[Fock]`'s A100** (pin 2).

## The cross-tree baseline, read

`data/measured/fock_baseline.json` (#289, with #290's addendum) puts the
generated kernel's `sm_86` ptxas figures beside the predecessor's
hand-spelled kernel for its eleven `l ≤ 2` cells, both under nvcc 13.0 and
the **predecessor's** flags (`-dc -O2 --std=c++17 -arch=sm_86`). It is the
only test anywhere of whether an IR-generated kernel is as good as a
hand-written one. It is a **quality baseline, never a numeric witness**: the
trees share ancestry, so no correctness claim may cite it, it has no
threshold and no ratchet, and `test_fock_coverage.py` refuses any claim
record that rests on it. The join maps the predecessor's `a ≥ b` onto this
tree's `la ≤ lb` through an asserted bijection.

- **The register boundary moved.** In the predecessor it falls inside
  L = 3: `ds|ps` fits and `pp|ps` spills 264 B. Here it falls between L = 3
  and L = 4. `pp|ps` (`sp|pp` here) is the one class that crossed, from
  spilling to fitting at 211 registers. None crossed the other way.
- **Where the boundary counts match, the kernels land close.** At
  `(dd|dd)`, spill stores are −5%, loads +2% and stack −4%. At `(pp|pp)`,
  stores are −20%. **Where they do not match, this tree spills more**:
  `ds|ds` has 4.3× the stores and `dp|dp` 1.5×. Every fitting cell uses
  fewer registers here.
- **H1, the orientation sets the accumulator count.** `la ≤ lb` puts the
  angular momentum in the HRR and reads the boundary from 0. Where
  `|target_set|` is equal (81, 961) the trees agree, and where ours is
  larger (100 against 36, 361 against 256) we spill more. #290's addendum
  adds this tree's 21-class ladder as further evidence: under the repo's
  flags on `sm_86`, **fitting follows the boundary target count, not L**.
  Every class with at most 40 targets fits, `(ss|dd)` among them (L = 4,
  168 registers), and every class with 76 or more spills. **H2**: a smaller
  fixed per-thread state outside the recurrence. Both are hypotheses. No
  mirrored-orientation kernel was built, and neither allocation was read
  past `ptxas -v` (#332).

## The split cell: HGP `J` and `K` (milestone 14)

**The key, `Spec.engine`** (#401, PR #470). `Target` names the digestion
(`J`/`K` split, `F` fused), so the field a `Spec` lacked was the recurrence:
`Engine.MMD` or `Engine.HGP`. `NATIVE_ENGINE` maps `F` to HGP and every
other target to MMD. `engine=None` resolves to the native engine, and the
key spells an engine only when it is off-native, as a suffix of the target
slot: `coulomb/d0/(1,1|1,0)/b1/j.hgp/literal`. Every stored key is
byte-identical. HGP was legal on `J` and `K` only; since #598 and #599 it
is also legal on `S`, `T`, `V`, `DIPOLE`, `METRIC` and `ERI3`, and a
third engine, `Engine.RYS`, on `F`, `J`, `K`, `S`, `T` and `V` (#621–#626,
`rys_fock.md`). Every other off-native
pairing is refused with its reason (`_ENGINE_REFUSAL`), MMD `F` among them
until #326. **`am` is the pairing the recurrence uses.** HGP `J` is
`(la, lb | lc, ld)`. HGP `K` is `(la, lc | lb, ld)`, the same as MMD `K` and
not `F`'s AO order. The `ordered` lever is legal on the slots `j.hgp` and
`k.hgp` only, and `measured_store.ledger_levers()` keeps it out of the lever
ledger.

**The two class modes** (`hgp/orbits.py`, #402, PR #469). The 8-fold ERI
group splits the ordered classes into orbits. `representative` picks each
orbit's cheapest member, ranked by HRR statements, then pruned VRR nodes,
then lexicographic order. `to_representative` returns the shell map and the
block transpose back to the ordered class. At L ≤ 4 there are 50 ordered
classes in 13 orbits when every shell has `l ≤ 2`, and 70 in 17 with no
cap. Neither engine is bitwise-symmetric under the permutations, so the
permutation check is two checks. The index map is checked exactly, on a
symbolic label tensor. `eri/eri.py`'s blocks are checked within
`1e-12·(scale_rep + scale_ordered)`, worst 2.6e-16. The issue's HRR example
named the wrong class: its "(ds|ss) 16/34 against (sd|ss) 19/60" is really
(dp|ss) against (pd|ss).

- **canonical** (the literal): one kernel per orbit, built on the
  representative and named `hj_{rep}` / `hk_{rep}`. A launch passes the
  runtime scalar `perm`, the `GROUP` index of its ordered class's
  orientation. The kernel holds one deposit per orientation the orbit
  reaches, and any other `perm` deposits nothing.
- **ordered** (the `ordered` lever): one kernel per ordered class, with the
  identity deposit and no `perm`, named `hj_o_{class}` / `hk_o_{class}`.

**The kernels** (`kernel_hj.py`, #404, PR #473; `kernel_hk.py`, #405,
PR #474). Both use §3.2's mapping, as `kernel_f` does: one block per bra
pair, one thread per contracted quartet, the `K⁴` contraction in registers
before the HRR, the per-thread staging, and `kernel_f`'s record, `pack` and
`MAX_PRIM_PAIRS`. `kernel_hk` imports its nest and modes from `kernel_hj`.
The primitive nest is bra outer and ket inner, the opposite of `kernel_f`'s,
so the ERI tap is `==` `eri/hgp_drive.eri_block` bit for bit. The deposit is
`J_ab += D_cd (ab|cd)` or `K_ac += D_bd (ac|bd)` by `AtomicAdd`, with no
weights, no degeneracy factor, no canonical half and no `close`. Two
engines do not agree bitwise, so each class's deposit is held to `eri/j_coulomb/j.py`
or `eri/k_exchange/k.py` within `1e-12·(s_hgp + s_mmd)`. That bound is derived from
`test_hgp_eri.py`'s per-ERI `TOL`, and no bound was widened. All 70 L ≤ 4
classes are tested interpreted, in both modes.

**The stream and its screens** (`launch_hjk.py`, #403, PR #471). There is
no symmetry pruning. Every ordered shell pair is a pair and every ordered
batch pair is a launch, so each ordered quartet is visited exactly once.
Batches are degree-uniform and sorted by `Q` alone. The canonical mode
routes a launch to its representative's class and carries the
`Orientation`. Each build screens on **one** density block: the `j` key is
`Q_ab Q_cd max|D_cd|` and the `k` key is `Q_ab Q_cd max|D_bd|`. A screen
**skips, never breaks**, and `τ = 0` skips nothing. On a `τ` ladder every
skipped quartet is proved against exact `eri/eri.py` ERIs, and J's key used
for K is shown to skip a quartet K needs. The stream borrows no private
helper (#328).

**The walker** (`walk_hjk.py`, #406, PR #476) assembles the whole `J` or `K`
from the stream through the interpreted kernels, in both modes, and refuses
any plan it cannot account for. It was run interpreted on H₂/STO-3G,
H₂O/STO-3G and H₂O/6-31G, against `eri/j_coulomb/j.py` / `eri/k_exchange/k.py` and libcint's
`vj` / `vk`. The worst error against libcint was 3.82e-15 of the scale
(6-31G `J`). The H₂O cases are `slow` and were run once by hand (32
passed). cc-pVDZ was not run.

**The generator** (#407, PR #475). `python -m felizjueves.emit --hgp-j-l
LMAX`, `--hgp-k-l LMAX` or `--hgp-class L0 L1 L2 L3` (repeatable), with
`--ordered` for the ordered mode, is its own path. It refuses every other
ladder, the selector, the lever flags and `--n-batches`, and it is not in
`KERNEL_GEN_CMD`. The goldens are 126 units per backend: per target, 13
canonical and 50 ordered, every shell `l ≤ 2`. All 252 compiled under nvcc
(`sm_86`) and hipcc (`gfx1200`) with no warning in generated text. hipcc's
`-Wunused-parameter` led to one emitter change: `emit/_c.py` now casts an
unread **array** parameter to `(void)`, and no existing unit's text moved.
`measured_store` refuses any HGP `J`/`K` row (`HGP_JK_REFUSED`), because
nothing is timed.

**Where it ran: the RTX 3080 only** (#408, PR #477; `test/hjk_driver.py`,
`test/test_hjk_ondevice.py`; on the shared runner since #657, not yet run
on a card). Part 1 ran all 70 L ≤ 4 ordered classes and
all 17 canonical kernels, for each screen. Each ran on a synthetic basis
holding only its own shells, against `eri.fock.fock`'s weight-1 sweep. Part 2 ran
H₂/STO-3G and H₂O/STO-3G at `τ = 0`, in both modes for both screens, against
the walk, the host and libcint. It also ran H₂O/STO-3G canonical at
`τ = 1e-2`, where the screen fires. The result was **184 passed, 0
skipped**. The worst error against the host was 5.90e-16 of the scale
(`hj_o_2100`), and against libcint 2.59e-15 (H₂O/STO-3G). **The RX 9060 XT
did not run it**, because the card was not enumerated on this box then.
The HIP side is compile-only (182 passed). The card run has not been made
and no open issue tracks it: #431 closed on 2026-10-17 when the card's
detection was fixed, not when this was run.

**What the split cell does not reach** (`data/literature/fock/limits.json`):

- **Nothing above L = 4 is tested.** The builders and the emitter accept any
  L, and the goldens stop at shell `l ≤ 2`.
- **Nothing is timed or stored.** The store holds no registers, spills or
  wall-clock for it, and it has not been timed against `Target.F` (#331).
- **Nothing persists across SCF iterations.** `J` and `K` are rebuilt from
  the density on every call. The pair list, bounds and staging are
  host-side, and the launch code is test-side.
- **One batch and one lever.** `n_batches` is 1, because an HGP class has no
  Hermite targets, and `ordered` is the only lever.
- **The screens' efficacy is unmeasured**, for the compact-fixture reason
  of #329.
- **No fused MMD `F`** (#326), the 2x2's last cell.

## The diagonal cells timed per class: fused HGP `F` against MMD `J` + `K`

**Measured on both cards, 2026-10-23** (`devtools/measure.sh scheme-ab`,
`measure/measure_scheme_ab.py`, store `data/measured/scheme_ab.json`).
`[Fock]` §6.1 finds opt-Brc faster than opt-UM below about 3,000 basis
functions on double-ζ sets. It times whole builds only, and §6.4.2 shows the
six s/p classes are over 60% of opt-Brc's time. This sweep asks the
per-class question that whole-build timing leaves open: is the fused HGP `F`
faster than MMD's `J` + `K` for (ss|ss) and (pp|pp)?

**The inputs.** `[Fock]` §6's polyglycines, gly_n = H(C₂H₃NO)_n OH for
n = 5, 10, 15 (`data/molecules/gly{n}.xyz`), in Cartesian 6-31G, with a
converged RHF density. The density decays, so the screens fire as they would
in an SCF. It comes from `python -m oracle.polyglycine` (PySCF) into an
untracked cache, because gly15's 643² density is about 9 MB of JSON. Every
row pins the file by sha256 and E_RHF.

**One class, by its shells.** 6-31G has only s and p shells. So the (ss|ss)
part of `G` is exactly `G` of the sub-basis of the s shells, and (pp|pp)'s is
exactly that of the p shells. Each arm plans and screens that sub-basis on
its own. The one screen input the cut changes is `kb`'s block factor, which
is `max|D|` of the sub-density.

**The arms.** One gated process per arm, three rounds of 3 warm-up and 11
timed runs, GPU events around the launches, and `τ = 1e-10` for every arm:

- `f`: `kernel_f`, literal, over `launch_f`'s plan (4 magnitudes), recorded
  by `fock_driver.record_plan`. This is `walk_f`'s launches without the walk,
  whose set of every unique quartet is about 1e9 entries at gly15.
- `f_block_fab`: the same launches with `block-fab`. It is reported only and
  decides nothing.
- `mmd`: the sum, per timed run, of `dq` (`Target.DQ`), `j` (every `[p|q]`
  launch, literal and one batch, plus `JAB`, on #157's chain driver, keyed on
  the DQ kernel's own `Dmax`) and `k` (`kernel_kblock`, #481).

Before anything is timed, the card's closed `F` is held to the card's
`J − K/2`. These are two recurrences, with no host reference. At gly5 and
`τ = 0` they agree to 4.27e-15 (ss) and 1.83e-15 (pp) of `max|G|` on both
cards, which is also MMD `K`'s first check on a card at (pp|pp), L = 4. At
`τ = 1e-10` they agree to 1e-10 of `max|G|` or better in every cell.

**The verdict rule was declared in the store before any row**: `fused` when
`q3(f) < q1(mmd)`. **All 12 cells read `fused`, on both cards.** Median ms,
with `mmd / f` in brackets:

| class | gly_n (AOs in the class) | `sm_86` `f` | `sm_86` `mmd` | `gfx1200` `f` | `gfx1200` `mmd` |
|---|---|---|---|---|---|
| (ss\|ss) | gly5 (97) | 226 | 816 (3.60) | 161 | 871 (5.42) |
| (ss\|ss) | gly10 (187) | 370 | 3,671 (9.92) | 374 | 4,320 (11.56) |
| (ss\|ss) | gly15 (277) | 597 | 10,562 (17.69) | 628 | 11,468 (18.26) |
| (pp\|pp) | gly5 (126) | 40.7 | 72.1 (1.77) | 38.2 | 83.1 (2.17) |
| (pp\|pp) | gly10 (246) | 73.5 | 320 (4.35) | 75.2 | 367 (4.88) |
| (pp\|pp) | gly15 (366) | 91.0 | 782 (8.59) | 108 | 950 (8.79) |

**How to read it:**

- **The ratio grows with n in both classes, on both cards.** From gly5 to
  gly15, `f` grows 2.2–3.9× and `mmd` 11–13×.
- **MMD's time is mostly `j` for (ss|ss)** (82% at gly15). That is the late
  contraction `[Fock]` §6.1 names. The 6-31G core 1s has six primitives, so
  an s pair has up to 36 primitive pairs, and `[p|q]` runs per primitive
  pair while `f` contracts inside the thread. For (pp|pp), `j` and `k` are
  about even.
- **`dq` is noise** (under 0.15 ms).
- **`block-fab` changes nothing here** (within 5%), unlike its 2–4× on the
  synthetic `wallclock-f` launch.
- **On the RTX 3080 the gate saw another tenant**: up to 27% before a phase.
  One phase was discarded and retaken. The `gfx1200` card read 0–1%.

**What it does not settle:**

- **The scheme is still confounded with the recurrence.** HGP split `J`/`K`
  is the arm that separates them, and `measured_store` refuses its row
  (`HGP_JK_REFUSED`), so `fock/symmetry-vs-screening` stays unverifiable
  (#331).
- **Every molecule is far under `[Fock]`'s ~3,000-function crossover**, where
  `fused` was expected.
- **Each class is timed alone**, not as a whole build.
- **The sweep records no quartet counts**, so it cannot say whether `f`'s
  slow growth is fewer surviving quartets or a less-loaded card.
- **MMD runs at its shipped defaults**: one batch, no lever, `max|D|` block
  factors (#484's decay factors not passed).

## What milestone 12 did not do

Said plainly, and recorded in `data/literature/fock/limits.json`:

- **No published cell is scored.** There is no `tables.json`, no floor and
  no ratchet, and no `[JEngine]` cell, floor, claim or status moved.
- **No HGP `J`/`K` split** (#325), and **no fused MMD `F`** (#326): the
  2x2's off-diagonal cells. Milestone 14 built the first ("The split cell",
  above), and #325 was closed into it.
- **No gradient.** `derivative_order` stays 0.
- **No lever.** The store refuses a lever on an F key, and `n_batches` is 1
  by `Target.F`'s own refusal.
- **Nothing persists across SCF iterations.** `F` is rebuilt from the
  density on every call, and the pair list, bounds, sort and staging are
  host-side, the launch code test-side.
- **`f` and above are not built** (#330). The ladder ends at `(dd|dd)`.
- **The screening comparison the fused scheme most wants is unmeasured.**
  Every fixture is compact. At `τ = 1e-10` no batch pair is removed before
  launch on any fixture, though primitive pruning fires on 6-31G (15 of
  280) and cc-pVDZ (23 of 593). Those are counts, not efficacy (#329). So
  `fock/symmetry-vs-screening` stays unverifiable (#331).
- **§3.2's staging bandwidth is not reproduced**, because the staging is
  per thread (#333).

## Named follow-ups

Filed as issues by #293:

- #325: the split HGP `J`/`K`, the (HGP, split) cell. It was closed into
  milestone 14, which built the cell.
- #326: the fused MMD `F`, the (MMD, fused) cell.
- #327: `engine.py`'s pushed-back-down list, an HGP analogue for each item
  or a recorded reason why not.
- #328: push the launch plan's stable sort and bound check up out of their
  MMD-named module: done, they are `launch_common.sorted_desc` / `nonneg`
  since #504 (the density and `tau` checks moved to `ao.density` in #503).
- #329: the fused `F`'s screens measured on an extended molecule.
- #330: `f` and above.
- #331: settle `fock/symmetry-vs-screening`, fused `F` against split
  `J + K`, timed on both cards.
- #332: the baseline's H1 and H2, by building the mirrored orientation.
- #333: a barrier in the IR, and the cooperative `|ab)` staging.
- #334: the 2020 letter's three-cell Table 1 (pin 2), read and declared.
