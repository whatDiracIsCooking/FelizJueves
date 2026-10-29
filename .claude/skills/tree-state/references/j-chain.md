# The `J` chain and the one-electron `S`, `T`, `V`

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

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
the RTX 3080 (`test_pair_bound_device.py`; the RX 9060 XT is not run, and
no open issue tracks it; #431 closed on the card's detection fix, not the run).
That is J's undressed bound; `K`'s screen is proved against the dressed one,
`eri.k_exchange.k.record_bounds`, generated as `Target.RECORD_BOUND`
(`eri/k_exchange/kernel_record_bound.py`, #499; `am` an **ordered** pair `(la, lb)`, the max
over the first shell's Cartesians, never batched): one thread per record,
`e_fragment.block_fragment` inlined and each `I_p` in the same closed form
(`kernel_pair_bound.hermite_bound_coefficients`), within 16 eps of the host
for every ordered class `la, lb ≤ 3` and every record of H₂O/cc-pVDZ and
the H₁₀ chain, interpreted (`test_kernel_record_bound.py`) and on the RX
9060 XT (`test_record_bound_device.py`; the RTX 3080 was down in that
session, so on CUDA it is compiled only). **From f up it defaults to a
low-stack form** (#586, stage 3): `build_record_bound_kernel(cls)` picks
it when `kernel_record_bound.record_bound_lowstack(cls)` (a shell with
`l ≥ 3`), `lowstack=` forces either, and the name, parameters, launch and
`bound` are the same. It holds stage 1's per-axis tables, taken `|·|` in
place, and a local `I[n_p]` instead of `E[100][84]`, so `record_bound_33`'s
`sm_86` frame is 3,360 B with no spill, against 76,608 B. The d text is
pinned by digest, and the per-unit numbers are in the module docstring.
**Nothing about `DQ` or
`JAB` is measured**, except #586's f-unit compiles and one hand timing
(below). `j_engine.md`, "The J chain as device code", is the one
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
**At (ff), `V` defaults to a low-stack form** (#586, stage 1, narrowed):
`build_v_kernel(cls)` picks it when `kernel_v.v_lowstack(cls)`
(`la + lb ≥ 6`, so (ff) alone through f: the only unrolled frame over the
8 KB budget; `sm_86` stack unrolled / low-stack is 0 / 0 B at `v_03`,
1,904 / 1,488 at `v_13`, 6,192 / 2,656 at `v_23`, 85,312 / 4,160 at
`v_33`, and by hand the low-stack form is up to ~1.25× slower at (df),
slightly faster at (ff)). `lowstack=` forces either form. The name, parameters, launch
and `V` contract are the same in both. Instead of the `E[100][84]` block
(an ~85 KB `sm_86` frame at (ff), reserved for every resident thread, so
~8.5 GB of the RTX 3080), it holds per-axis `E` tables
(`mmd/e_fragment.py`'s `axis_tables_fragment`), `[t]^(0)` in a local `R`,
and the row contraction as loops (`cartesian_loops`, `rolled_e_sum`,
`hermite_index`, the shared piece that stages 2 and 3 reuse for `DQ`,
`JAB` and `RECORD_BOUND`). At (ff) the frame is 4,160 B with no spill.
Every other class's text, the mixed f ones included, is byte-identical to
cc5d16f's, pinned by digest in `test_kernel_v.py`. The numbers are in `one_electron.md`.
**At (ff), `DQ` and `JAB` default to a low-stack form too** (#586,
stage 2): `build_dq_kernel(cls)` / `build_jab_kernel(cls)` pick it when
`kernel_dq.dq_lowstack(cls)` / `kernel_jab.jab_lowstack(cls)` (one rule,
`la + lb >= LOWSTACK_MIN_PAIR_L = 6`: (ff), and past f (gd) up). The
mixed f classes stay unrolled: their frames are 0 to 936 B with no
spill, and the low-stack form is slower there. `lowstack=` forces either
form, and `density="ao"` takes both. Name, parameters, launch and output
contract are the same, so free-breakfast's emitter calls the same
builders. Per-axis `E` tables (`axis_tables_fragment`), the rows as
`cartesian_loops`, the Hermite index in closed form -- but NOT
`rolled_e_sum`: each `E` entry is `block_fragment`'s own `(E_x E_y) E_z`
and the sums keep the host's order, so both forms stay `==` to the host,
interpreted, at every class through (ff). At (ff) the `sm_86` frames are
3,360 B (`dq_33`) and 2,688 B (`jab_33`), no spill, against 72,464 and
71,456 B. Below L = 6 the text is byte-identical, pinned by digest in
`test_kernel_dq.py` / `test_kernel_jab.py`. The numbers and a hand timing
are in `j_engine.md`, "The low-stack `DQ` and `JAB` at f".
