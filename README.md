# FelizJueves

Python code generation for electronic structure tensors.

## What is here

**A generator of the kernels of a Coulomb-matrix build, and that build.** The
generator emits, in CUDA and HIP, the McMurchie–Davidson `[p|q]` kernel of
`[JEngine]` and, since milestone 7, the kernels around it: the `E` transform
(inlined), the ket pre-contraction and the bra back-transform with its
scatter. A host-side AO layer turns a real molecule into their inputs and
remains the reference they are held to, and the `J` they build is compared
against PySCF/libcint — a program with no shared ancestry. Since milestone
11 it also generates the **one-electron matrices** `S`, `T` and `V` on the
same `E` fragment, and the core Hamiltonian `h = T + V` they sum to.
Milestone 10, the **exchange matrix** `K` on the J engine's recurrence, is
**partly built and still open**: its kernel, launch plan and host reference
exist, it has run on both cards only for classes with L ≤ 2, and nothing
about it is scored or timed yet. Since milestone 12 there is a **second ERI
engine and the first Fock matrix**: the Head-Gordon–Pople recurrence of
`[Fock]` §3 and the fused kernel built on it (`Target.F`), which digests
every contracted ERI straight into `G = J − K/2` using 8-fold permutational
symmetry, the first use of that symmetry anywhere in this repo. With the
MMD split `J`/`K` already here, FJ now holds both of `[Fock]`'s diagonal
cells (MMD split, HGP fused) under one IR and one harness, on two cards.
The two off-diagonal cells, a split HGP `J`/`K` and a fused MMD `F`, are
named follow-ups (#325, #326; `docs/implementation/hgp_fock.md`, "The 2x2,
in one place"). Every tier reports what it touched rather than implying
more.

**In three questions:**

- **What does it generate?** Every arithmetic step of `J_µν`, in CUDA and HIP:
  the `[p|q]` kernels (`J_p = Σ_q [p|q] D_q`, one per `(lp, lq)` class and
  batch count), the pre-contraction `D_q` and `Dmax` (`Target.DQ`, one kernel
  per ket pair class) and the bra back-transform and scatter into `J_µν`
  (`Target.JAB`, one per bra pair class), with the undressed `E` coefficients
  inlined into the last two. What stays on the host is what decides
  launches: the pair list and each record's inputs, the Schwarz bound, the
  sort (which reads `Dmax` back), the staging and the launch code. The
  pre-contraction and scatter kernels are this repo's own design, because
  `[JEngine]` defers those steps to another paper (`limits.json`). Also
generated (milestone 11): the overlap `S`, the kinetic energy `T` and the
nuclear attraction `V` (`Target.S`, `T`, `V`), one kernel per pair class
and one thread per primitive-pair record. `V` is the `J` engine with the
ket replaced by a unit point charge, its recurrence and plan layer
reused unchanged. These are this repo's own design too, and **nothing
about them is scored**: `[JEngine]` is J only, so there is no published
one-electron board (`limits.json`; `docs/implementation/one_electron.md`).
Also generated, but only by the tests (milestone 10, still open): the
exchange kernel `K_ab = Σ_cd D_cd (ac|bd)` (`Target.K`, `kernel_k.py`), one
kernel per quartet class `(la, lc | lb, ld)` in Hermite order, bra first. It
runs `[p|q]`'s recurrence and plan unchanged, inlines both `E` fragments,
and stages the contraction so that neither `(ac|bd)` nor its half-transform
is ever an array; a split class is one atomic partial-sum kernel per batch.
Its launch plan (`launch_k.py`) is a sibling of J's, because `ClassLaunch`
is J-shaped, and its screen sorts the kets by `I_bd` alone: J's density key
cannot be formed for `K`, whose `D_cd` straddles the bra and the ket.
`KERNEL_GEN_CMD` writes no `K` unit.
Also generated (milestone 12), through its own flags (`--fock-l`,
`--fock-class`) and not by `KERNEL_GEN_CMD`: the fused Fock kernel
(`Target.F`, `kernel_f.py`), one per AO-order class `(la lb|lc ld)` with
every shell `l ≤ 2`, 21 classes up to `(dd|dd)`. It follows `[Fock]` §3.2:
one block per bra shell pair, one thread per contracted quartet, the
contraction loop inside the kernel because the HGP horizontal recurrence
needs contracted values, a six-block density screen that skips and never
breaks, and the six-block digestion with its weights **derived** (Coulomb
2, exchange −½) and held to a sweep over every ordered quartet with no
symmetry in it. The weights are derived because the predecessor shipped
this digestion with the Coulomb weight 1. The pair stream that feeds it
(`launch_f.py`) and its walker (`walk_f.py`) are host-side.
- **How is a whole molecule's `J` built?** From a **launch plan**
  (`launch.py`, milestone 6). It has one entry per `(lp, lq)` class, with bra
  records in pair order and ket records sorted by `I_q · max|D_q|`
  (`[JEngine, v2, §3.3]`). On the card, each bra class and each ket class is
  uploaded once, and every batch kernel is launched with fp64 atomics into
  one `J_p` buffer per bra class. Since milestone 7 the `DQ` kernels produce
  the kets' `D_q` and the `JAB` kernels turn each `J_p` buffer into `J_µν` on
  the card, launched from Python through a test-side driver. The plan's order
  is witnessed by an independent Python walker (the launch oracle) and by
  libcint. A class's batch count can be chosen from measured wall-clock
  (`dispatch.py`).
- **On which cards has that run?** The RTX 3080 (`sm_86`) and the RX 9060 XT
  (`gfx1200`). The whole chain built `J` for H₂, H₂O/STO-3G and H₂O/cc-pVDZ
  (s, p and d shells, `[p|q]` classes up to L = 8), within `1e-12·scale_J` of
  libcint (worst 1.24e-14). The pre-contraction and scatter kernels ran alone
  for every pair class through (ff), and the `[p|q]` kernels alone to L = 16,
  each against the host. Milestone 6's build, with `[p|q]` alone on the
  card, is timed on both cards (`data/measured/builds.json`). **Nothing about the new kernels is
  measured.** `docs/implementation/j_engine.md`, "The J chain as device code",
  is the one place with the detail. The one-electron kernels ran on the same
  two cards: every pair class of HF/cc-pVTZ through (ff) against the host,
  `S`, `T` and `V` of H₂O/STO-3G and H₂O/cc-pVDZ (s, p and d) within
  `1e-12·scale` of libcint (worst `V`, 3.1e-14), and `h = T + V` against
  PySCF's `hcore` (worst 1.88e-14), with `Tr(D h)` (H₂O/cc-pVDZ: 7.11e-14
  off, 4.46e-5 of its tolerance), the same digits on both cards. Nothing
  one-electron is measured either, and `V` spills from `la + lb = 3`
  (`docs/implementation/one_electron.md`). `K` has run on the two cards only
  for classes with L ≤ 2 (#206): `K_mn` from its launch plan for H₂, two H₂
  6 bohr apart, H₂O/STO-3G's s shells and its L ≤ 2 classes, against the
  host paths and, for H₂ whole and H₂O's s-s block, within `1e-12·scale_K`
  of libcint's `vk` (worst 3.23e-15), the same figures on both cards. One
  cell above that, `(dd|dd)` at L = 8 and one batch, matched the host on
  the RTX 3080 only (#259). The whole H₂O/STO-3G `K` on a card, and every
  other `K` test above L = 2, are deferred to #231, still open. **Nothing about
  `K` is timed**: its 20 rows in `data/measured/kernels.json` (L = 7 and
  10, every batch count, both cards) are registers and spills only, and
  every one spills; L = 13 and 16, the clock beside J's, and the spill
  verdict are #205's, still open. The three `[JEngine]` claims about `K`
  stay pending until #207 scores them. The fused `F` ran on both cards
  (#288, #290). Every one of its 21 classes matched the every-ordered-quartet
  sweep within `1e-13·scale_G` (worst 2.2e-14). The whole `F` of H₂O/STO-3G,
  H₂O/6-31G and H₂O/cc-pVDZ matched libcint's `vj − vk/2` within
  `1e-12·scale_G` (worst 7.35e-15 on the 3080, 7.34e-15 on the 9060 XT).
  **It is measured:** 42 rows in
  `data/measured/kernels.json` (21 classes × both cards; registers, spills
  and wall-clock behind the card gate). `(dd|dd)` uses 255 registers with
  97,464 B of spill stores and takes 2.33 s per launch on the 3080, and 192
  VGPRs and 3.70 s on the 9060 XT. On `sm_86` a class fits its registers
  when its HRR boundary has at most 40 targets and spills from 76.
  `data/measured/fock_baseline.json` sets it beside the predecessor's
  hand-written kernel as a quality baseline. It is never a numeric witness,
  because the two trees share ancestry.

Not built: a `J` kept on the card across SCF iterations, a `Launch` IR or
`LaunchCost` model (declined, `limits.json`), dispatch inside the generator,
and anything with gradients (one-electron derivatives included), a batched
`V` or a warp-per-record `V` (named follow-ups). Milestone 12 did not score
any published cell: `[Fock]` prints no register table for its §3.2 kernel,
and its tables are A100 and V100 wall-clock, so only its non-numeric claims
are scored (`data/literature/fock/`). It also did not build an HGP `J`/`K`
split, a gradient, a lever, an `F` kept across SCF iterations, or anything
at `f` and above (#330). The screening comparison the fused scheme most
wants is unmeasured, because every fixture here is compact (#329, #331).
Against the predecessor `jgen`, this is
parity on its energy-order `J` surface, with one difference: Table 4,
declined. Which plan ships is a decision, not a gap: `jgen` ships its
searched sieve order, and this repo has that wire too (the opt-in `tuned`
lever) but ships the **literal** plan, because the searched one was measured
on both cards and refused on wall-clock (#166;
`docs/implementation/j_engine.md`, "Parity with the predecessor").

`.claude/CLAUDE.md` ("The state of the tree") has the detail; in short:

| | State |
|---|---|
| `docker/`, `.devcontainer/`, `devtools/` | **live.** Both device toolchains build, and `devtools/kernel-tier.sh` runs end to end. |
| `test/` | **live.** Fast tier: the plan layer against `[JEngine]` Tables 1–3, the kernel IR interpreted against `reference/`, the golden kernel texts, and the AO layer against **tracked libcint fixtures** (`test/oracle/`), so PySCF is not needed to run it. Fast tier also: the launch plan, its oracle and the staged interpreted driver (`test_launch*.py`), and dispatch on synthetic and tracked rows. Kernel tier (`gpu`): the generated kernels on each card against `reference/`, `J_µν` for H₂ and H₂O against libcint, the whole molecule built from the launch plan on each card against the oracle and libcint (`test_j_launched_ondevice.py`), and the generated J chain on each card: every pre-contraction and scatter kernel through (ff) against the host, and `J` through d with every arithmetic step a generated kernel against libcint (`test_j_chain_ondevice.py`). Fast tier also: the `E` fragment and the two new kernels interpreted, bitwise against the host (`test_e_fragment.py`, `test_kernel_dq.py`, `test_kernel_jab.py`), and the whole chain interpreted against libcint (`test_ao_j_chain.py`). Last full kernel tier, PR #210 (#157): CUDA, RTX 3080, 96 passed / 5 skipped; HIP, RX 9060 XT, 97 passed / 5 skipped; the skips are the five opt-in recordings (#225 re-runs the full tier). Since then the plan-lever A/Bs' device phases (`test_kernel_device.py` and `test_measured.py`) ran: #162 (PR #222), 83 passed / 2 skipped on each card, and #163 (PR #224), 99 passed / 2 skipped on each card, the skips the opt-in wall-clock and roofline sweeps. The last device sessions per card are milestone 12's fused-`F` phases (`kernel-tier.sh -k 'fock or f_cells'`, a subset, not a full tier): #288 (PR #321) and #290 (PR #324), each 25 passed / 1 skipped on the RTX 3080 and 25 passed / 1 skipped on the RX 9060 XT, the skip the opt-in F wall-clock, run alone afterwards on each card and passed. One-electron (milestone 11): the host `S`, `T`, `V` against closed forms and, since #240, libcint's `int1e_*` and PySCF's `hcore` through the tracked fixtures; the three kernels interpreted (`test_kernel_{s,t,v}.py`; `S` and `T` bitwise, `V` to round-off); and, on each card, every pair class through (ff) against the host and H₂O against libcint (`test_one_electron_device.py`), and `h` and `Tr(D h)` against `hcore` (`test_hcore_device.py`). Exchange `K` (milestone 10, still open), its host and interpreted cases at L ≤ 2 in the fast tier and every case above L = 2 but one (below) written `slow` and **not yet run** (#231): the host `K` by two sweeps and through `ao.eri`, its Schwarz bound and screen (`test_ao_k.py`, `test_ao_k_screening.py`), H₂ and H₂O's s-s block against libcint's `vk` through the tracked fixtures; the `K` kernel and its split class interpreted, bitwise against the host per bra record (`test_kernel_k.py`, `test_kernel_k_batched.py`); the `K` launch plan walked (`test_launch_k.py`); the duplicated-work count (`test_k_duplication.py`); and, on each card, `K_mn` from the launch plan at L ≤ 2 against the host and libcint (`test_k_ondevice.py`, #206), plus, in `test_kernel_device.py`, one `(dd\|dd)` cell at one batch on the RTX 3080 (#259); its `K_SUBGRID` device checks were written by #205 and not yet run to completion. Three tests of #205's finished store are strict `xfail` until it lands. Milestone 11's kernel tiers were all capped at L ≤ 8 by the user, and no J text changed, so the L = 16 tier was not re-run: its widest, #244's, was 95 passed / 3 skipped on CUDA and 96 passed / 3 skipped on HIP. Fast tier also: the plan's output pinned across hash seeds (`test_plan_determinism.py`), the per-batch order search (`test_plan_order_search.py`), the sieve's spill reading (`test_sieve_spill_claim.py`), the plan-layer counts against the measured kernels (`test_planrank.py`), and the shipped plan held to the literal rows (`test_shipped_plan.py`). The fused Fock build (milestone 12): fast tier, the HGP DAGs, FLOP count and recurrence (`test_hgp_*.py`, the two printed defects shown failing), HGP against MMD in all 21 classes and against libcint (`test_hgp_eri.py`), the no-symmetry sweep against FJ's `J`, `K` and libcint's `vj − vk/2` (`test_ao_fock.py`, `test_oracle_fock.py`), the fused kernel interpreted with its derived weights and their mutations (`test_kernel_f.py`), the pair stream and its walker (`test_fock_pairs.py`, `test_fock_build.py`), `Engine` over both engines (`test_engine.py`), 21 goldens per backend (`test_emit_fock.py`), the cross-tree baseline join (`test_fock_baseline.py`) and the `[Fock]` reading and claims (`test_fock_coverage.py`, `test_fock_claims.py`); on each card, every class against the sweep and H₂O whole against libcint (`test_fock_ondevice.py`, 16 of its 21 classes `slow`). |
| `src/felizjueves/` | **importable.** `plan/` (batching, MILD, the sieve, the per-batch counts, `optimum.py`, the exact tree-search optimum for L ≤ 5, `search.py` and `order_search.py`, the searched board's per-batch seed orders, and `pipeline.tuned_plan`, the opt-in `tuned` lever that builds from them), `planrank.py` (whether the plan-layer counts predict the measured kernel: on neither card), `kernel.py` + `emit/` (the generator), `e_fragment.py`, `kernel_dq.py` and `kernel_jab.py` (the `E` fragment and the pre-contraction and scatter kernels, milestone 7), `kernel_s.py`, `kernel_t.py` and `kernel_v.py` (the one-electron kernels, milestone 11), `kernel_k.py` (the exchange kernel and its split class, milestone 10) and `kdup.py` (the work a split duplicates, J against K, counted from the plan), `measured_store.py` (the measured store: registers, spills, wall-clock on both cards; roofline counters on `sm_86`) and `measured.py` (the driver that compiles, parses and profiles into it), `gate.py` (the card gate every timed or profiled phase runs behind), `ceilings.py` (each card's measured fp64 and bandwidth ceilings), `ridge.py` (whether batching crosses our cards' ridge, under the rule #114 declared before the rows), `crossread.py` (model, clock and counters read against each other), `asym.py` (the asymmetric-split study), `literature/` (published tables and claims, and the rules that score against them), `ao/` (shells, the host `E` transform, pair data and the Schwarz bound, ERI blocks, `J_µν` with its `jp=`, `dq=` and `jab=` seams, `staged.py`, `J` from a launch plan with the kernels interpreted, and `drive.py`'s `coulomb_matrix_chain`, `J` with every step a generated kernel, interpreted; and `one_electron.py`, `overlap.py`, `kinetic.py` and `potential.py`, the host `S`, `T` and `V`, with `pairs.py`'s one-electron screen; and `k.py`, the host `K` with its Schwarz bound and screen), `launch.py` (the whole-molecule launch plan as a value, and the launch oracle that walks it in Python; and the one-electron launch, `OneEClass`, a sibling because `ClassLaunch` is J-shaped), `launch_k.py` (the `K` launch plan and its walk, a sibling for the same reason) and `dispatch.py` (a class's batch count from measured wall-clock, injected into a plan). Milestone 12, engine #2: `hgp/` (the HGP vertical and horizontal recurrences as DAGs, their FLOP count, and both as IR), `kernel_f.py` (the fused Fock kernel and its derived digestion), `launch_f.py` (the pair stream: pruning, degree-uniform batches, eq. 29's pre-launch screen) and `walk_f.py` (its oracle), `ao/fock.py` (`G = J − K/2` over every ordered quartet with no symmetry, the witness) and `ao/hgp_drive.py` (HGP ERIs, interpreted), and `engine.py` (the `Engine` protocol, extracted last from the two engines; its docstring lists what was pushed back down into MMD). |
| `data/` | **live.** `[JEngine]` Tables 1–3 with provenance and the two scoreboards over them; the registry of the paper's non-numeric claims (`claims.json`, 35 records: 18 asserted, 14 unverifiable, 3 pending), the unit-by-unit reading it is the inventory of (`coverage.json`), and what a model cannot reach, this machine cannot reproduce, or this repo declined (`limits.json`, which also records that the one-electron family has no published board); the measured store and the recommended batch count per `(L, arch)`; the A/Bs of the `recycle`, `asym`, `no-sieve` and `tuned` levers against the literal kernel on both cards, none of which moved the default, and `plan_vs_kernel.json` (#165); the seed order's spread over 58 rules (`seed_order_spread.json`); each card's measured roofline ceilings, roofline rows on `sm_86` and the answer they give — **batching crosses the ridge only at L = 16 (at 3 batches); L = 7, 10 and 13 are compute-bound already at one batch; `gfx1200` could not be measured** (`docs/implementation/j_engine.md`, "The roofline, in one place" and "The `[JEngine]` claims"). That refusal is **final** (#140): `rocprofv3` now loads in the ROCm image, but gfx1200 defines no fp64 counter and its byte counters read 0 on this box. **gfx1200's fp64 ceiling is settled** (#141): single issue, 128 flop per counted GPU cycle, so the recorded 4.46e11 flop/s is the card at a counted ~3.49 GHz and the ridge stays 1.49 FLOP/B. Also: the whole-build wall-clock on both cards and #147's sort-key A/B (`measured/builds.json`; the expectation that the global `max\|D\|` key is worse is **contradicted on both cards**), and the `τ`-ladder screening record (`ao/screening.json`). For `K` (milestone 10, still open): the three `[JEngine]` claims about it, recorded before anything was built and still the 3 pending above (#207 scores them); the duplicated-work count, J against K (`k_duplication.json`: on counting, K's is the larger at every split cell); and 20 compile-only `K` rows in the measured store (L = 7 and 10, both cards; registers and spills, no wall-clock; the rest is #205's). For the fused `F` (milestone 12): `[Fock]` §3's reading (`literature/fock/`: `coverage.json`, `claims.json` with 10 [Fock] records: 9 asserted, 1 unverifiable, 0 pending, and `limits.json`, with no `tables.json` by decision), 42 timed `F` rows in the measured store (all 21 classes, both cards), so the store holds 258 rows (196 `J`, 20 `K`, 42 `F`), the cross-tree ptxas baseline against the predecessor (`measured/fock_baseline.json`, a quality baseline, never a witness), and the HGP FLOP and loop-nest counts (`hgp/cost.json`, `hgp/kernel_cost.json`). |
| the generator | **live, for the whole J chain, the one-electron matrices and the fused `F`.** `KERNEL_GEN_CMD` in `devtools/config.sh` runs `python -m felizjueves.emit --one-e-l 2`, which writes, in CUDA or HIP, the single-batch `[p\|q]` kernel for L = 0..16 and the pair kernels `dq_{la}_{lb}` and `jab_{la}{lb}` for `la ≤ lb ≤ 3` (37 units, what the bare command writes), plus the one-electron ladder `s_{la}{lb}`, `t_{la}{lb}` and `v_{la}{lb}` for `la ≤ lb ≤ 2` (18 units; `--one-e-l` adds it, accepted 0..8). The ladder writes `la ≤ lb` only; the `la > lb` classes a real basis reaches are emitted by the test-side driver. Split classes (`build_pq_kernels`, atomic write-back) and, through `--class LA LB LC LD`, each real class's whole chain (`DQ`, the `[p\|q]` batches, `JAB`) are written by the tests. **Generated: every arithmetic step of `J`, and `S`, `T` and `V`.** The exchange kernel `K` is generated too, but only by the tests (`test/plan_driver.py`'s `K` driver, `test_kernel_device.py`'s `K` harness) and by the measured driver; `KERNEL_GEN_CMD` writes no `K` unit. The pair list, the bound, the sort with its `Dmax` read-back, the staging and the launch code stay host-side, and nothing about the new kernels is measured. Dispatch (`dispatch.py`) chooses a batch count per class for a launch plan; `KERNEL_GEN_CMD` does not follow it, and `Spec.n_batches` defaults to 1 by decision (another count: `python -m felizjueves.emit --class LA LB LC LD --n-batches 1 3 6`). **What it ships is the literal plan, by decision:** the searched plan reaches the emitter through the opt-in `tuned` lever, and on both cards it was measured against literal and refused on wall-clock (#166). The levers measured on both cards are `recycle`, `asym`, `no-sieve` and `tuned`, and none moved the default (`docs/implementation/j_engine.md`, "What the generator ships, and why"). The fused `F` has its own path, `python -m felizjueves.emit --fock-l LMAX` (0..2) or `--fock-class LA LB LC LD`, which writes `f_{la}{lb}{lc}{ld}` for the 21-class ladder and nothing else; `KERNEL_GEN_CMD` writes no `F` unit, and the tests emit the ones they run (`test/fock_driver.py`). |
| `pyproject.toml` | **a virtual uv project** — dev tooling, numpy, and a lock; PySCF is a separate `oracle` group a bare `uv sync` does not install. Nothing is installed; pytest imports `src/` through `pythonpath`. |

When you run something here, say which of these it touched.

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

The device toolchains do not have to be on your host — they are in the images:

```bash
docker/build.sh cuda                      # base, then cuda; tags fj:cuda and fj:latest
devtools/devcontainer.sh shell            # interactive, in the CUDA container
devtools/devcontainer.sh --hip shell      # the ROCm one
```

`docker/README.md` has the four Dockerfiles, the build args and the two front
ends.

## The tiers, and what each one proves

| Tier | Command | What a green run means |
|---|---|---|
| lint | `ruff check .` | narrow by design: `E,F,I,UP,B`, no formatter. |
| fast | `pytest -m "not cuda and not hip and not gpu and not slow and not oracle"` | the Python side agrees with itself, and with libcint through tracked fixtures. Compiles nothing, launches nothing, needs no PySCF. This is the push gate, and `FAST_TEST_ARGS` is that marker expression: it excludes every marker that would reach a compiler, a card, PySCF (`oracle`) or minutes (`slow`). A bare `pytest` applies no marker filter. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip` | generate → compile with the real device compiler (a warning in generated text fails it) → run on the card → compare against the host reference, and `J_µν` against libcint. Without `--full` the device phase skips `slow` (the PR gate, under five minutes per card); `--full` runs everything. |
| measure | `devtools/measure.sh <sweep>` | not a test tier: wall-clock, counters, ceilings and the stored `data/measured` reports, in `measure/`. Never a gate. |
| oracle | `uv sync --group oracle`, then `pytest -m oracle` | the tracked fixtures still equal what PySCF computes today. |

**The kernel tier is the only place a claim about a card can be made.** No
hosted CI runner has a GPU. A `gpu`-marked test *skips* on a device-less box
rather than failing, so a green run there is green and empty —
`devtools/doctor.sh` exists to make that visible. The `gpu` fixtures lock the
card themselves (`GPU_LOCK_PATH` in `devtools/config.sh`), so two runs on one
card serialise.

This box is the reference machine for both halves: an **RTX 3080** (sm_86, CUDA
13.0) and an **RX 9060 XT** (gfx1200, RDNA4, ROCm 7.2.4). A generated kernel has
to satisfy both compilers, and `hipcc` is clang while `nvcc` is not — that
disagreement is this project's main portability risk, which is what
`.devcontainer/combined/` is for.

## Layout

```
devtools/        the scripts: doctor, devcontainer, worktree, kernel tier, claude pin
docker/          four Dockerfiles (base -> cuda/hip -> combined), build.sh, compose
.devcontainer/   one variant per GPU image, driven by devtools/devcontainer.sh
docs/            references.md, and the derived notes on what gets generated (j_engine.md, mild.md, one_electron.md, hgp_fock.md)
data/            published tables and claims (literature/), scoreboards, the measured store, ao/
src/felizjueves/ plan/, hgp/, the kernel builders and emit/, launch*.py, walk_f.py, engine.py and dispatch.py, the measured store and driver, literature/, and ao/
test/            pytest; markers separate "needs a compiler", "needs a card", "needs PySCF";
                 test/oracle/ holds the libcint fixtures and the PySCF converter
```

Depth lives where it is used: each script's header explains itself (`--help`
prints it), `docker/README.md` covers the images, and `.claude/CLAUDE.md` is the
map for working in here.
