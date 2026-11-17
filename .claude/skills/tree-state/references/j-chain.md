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
The opt-in `fused` lever (#344) is FreeBreakfast's low-stack `e_hermite` at
(7,6), (6,6), (7,5) and (5,6) (FreeBreakfast#1337): (7,0|6,0) 18,160 ->
2,920 B of `sm_86` stack, 0.30-0.75x the literal time on the RX 9060 XT,
all four interpreted, and (7,6) / (6,6) on that card, against the reference
at the unchanged pin (`levers/fused.py` has the numbers); not bitwise the
literal kernel.
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
**The packed full ERI and `J` by GEMV** (`Target.ERI`, `eri/full_eri/`) are
a small-system path (`n_ao <= packed.N_AO_MAX = 150`, about 1 GB of doubles
there): `kernel_eri.build_eri_kernel((la, lb, lc, ld))` writes every
`(mn|ls)` of a class into one symmetric `n_pp x n_pp` matrix packed by AO
pair (`n_pp = n_ao (n_ao + 1) / 2`), one thread per (bra shell pair, ket
shell pair) task, the Schwarz kernel's skeleton widened to a whole block on
the `Target.J` quartet's plan, plain stores, no atomics, so bit-reproducible
and exactly symmetric; `kernel_gemv.build_gemv_kernel()` (`eri_jgemv`) is
`J_pp = ERI_pp Dp`, one thread per packed pair reading the column
(coalesced). The staging (`packed.eri_classes`) reuses
`eri.schwarz.schwarz_classes`' pair classes and owns the eight-fold symmetry:
every unordered pair of shell pairs is one task. The packing of `D` and the
unpacking of `J` are host-side. Emitted by the opt-in `--eri-l [LMAX]`
ladder (bare: s and p, 6 classes plus the GEMV; at most f, (ff|ff), 55
classes; from f the low-stack form, below).
**Interpreted** every class at L <= 4 (`test_kernel_eri_full.py`): within
16 eps of the host's `eri.eri` scale, every element written once, and `J`
from the interpreted GEMV over H2 and H2O/STO-3G within 1e-12 of PySCF's
`vj` (6e-15 absolute on H2O). **Compiled by nvcc and hipcc** (the programs of
`test_eri_full_device.py`). **On a card: CUDA only, an RTX A6000 (sm_86,
CUDA 13.0), 2026-11-17** -- the first run of its `gpu` tests: every class at
L <= 4 of the synthetic basis within its bound (worst 4.7e-2 of it; max
|ERI - host| 7.1e-15),
runs bit-identical; H2 and H2O/STO-3G's `J` within 3.8e-15 / 7.5e-16 of
`vj` (relative to max |vj|). HIP is still owed (no gfx1200 run). Nothing is
measured.

**`K` by GEMV over the same tensor** (`kernel_kperm.build_kperm_kernel()`,
`eri_kperm`): for a symmetric `D`, `K_pp = X_pp Dp` with the Coulomb GEMV's
own `Dp`, `X_pp[pidx(l,s), pidx(m,n)] = 0.5 ((ml|ns) + (ms|nl))` -- one more
symmetric `n_pp x n_pp` matrix (so ~2 GB for the pair at the cap), gathered
from `ERI_pp` once per geometry, one thread per packed column, plain stores,
bitwise the host's `packed.exchange_packed` and exactly symmetric; `K` is
then `eri_jgemv` over `X_pp`, unchanged. Emitted beside the GEMV by
`--eri-l`. **Interpreted** (`test_kernel_eri_full.py`): `X_pp` `==` the host
and its transpose; the host's packed `K` within 2e-16 of a dense `einsum`
over the unpacked tensor (s/p synthetic basis, random symmetric `D`); `K`
from the interpreted chain over H2 and H2O/STO-3G within 3e-15 / 6e-16
(relative to max |vk|) of PySCF's `vk`. **Compiled by nvcc and hipcc**
(`test_X_eri_programs_compile`, cloud session, 2026-11-17). **On CUDA, an
RTX A6000, 2026-11-17**: `X_pp` bitwise the host's `exchange_packed` of the
card's ERI and exactly symmetric, `K` within 2.9e-15 (H2) / 6.2e-16 (H2O) of
`vk`, runs bit-identical; HIP owed. Nothing is measured.

**The dense J- and K-layout ERI** (`Target.ERI_DENSE`,
`kernel_dense.build_dense_kernel(cls, layout)`, `eri_dj_{la}{lb}{lc}{ld}` /
`eri_dk_...`; host staging and reference in `full_eri/dense.py`) is the
multi-GPU path, and the packed one is left as it was. With AO pairs unpacked,
`(m, n) -> m n_ao + n`, it writes `M_J[(m,n), (l,s)] = (mn|ls)` or
`M_K[(m,n), (l,s)] = (ml|ns)`, each dense `n_ao^2 x n_ao^2`, so `J = M_J
vec(D)` and `K = M_K vec(D)` for ANY `D` -- no packing, no `Dp`, no
symmetric-`D` assumption. **One family, a layout argument**: both layouts are
the same block `(ab|cd)` of one ORDERED shell quartet, placed at
`M_J[(a,b), (c,d)]` or `M_K[(a,c), (b,d)]`; everything up to the scatter is
`kernel_eri.contraction`, factored out of `build_eri_kernel` for it (the
packed kernels' emitted text is byte-identical, checked for all 21 classes on
both backends). The key is the class's, not the layout's. **No symmetry, by
decision**: the class is the AO-order quartet of every ordering (81 at d), a
task is one ordered shell quartet, nothing is mirrored or skipped, every
element has one owner and one plain store (no atomics, bit-reproducible run
to run). Recomputation against `kernel_eri`'s unique-quartet build: up to 8x
(`n_shell^4` tasks against ~`n_shell^4 / 8`), per layout; a split over GPUs
adds none. **The row range** is first-class: in both layouts the row's `m`
is in the bra's first shell, so a launch over the tasks whose first shell is
in `[sh0, sh1)` (`dense.dense_classes(basis, (sh0, sh1))`) writes exactly the
contiguous rows `[m0 n_ao, (m0 + n_m) n_ao)`, every column, into a LOCAL
block (`eri[n_row, n_col]`, params `m0 n_row n_col`); disjoint shell ranges
are disjoint row blocks, collected by a gather. The int32 bound is
`n_row n_ao^2 < 2^31` per launch (`dense.row_range` refuses past it); there
is no `N_AO_MAX` on this path, and sizing the blocks is the caller's job.
Emitted by the opt-in `--dense-eri-l [LMAX]` (bare: s and p, 16 classes x 2
layouts; at most f, 256 x 2, low-stack from f), never by default. **Interpreted**
(`test_kernel_eri_dense.py`): every ordered class at L <= 4 of the synthetic
s/p/d basis, both layouts, against `eri.eri` placed by the kernel's rule,
within `(2 gamma_N + delta) scale` (test_eri_full_device's analysis of the
same contraction) -- worst 1.0e-2 of the bound; H2 and H2O/STO-3G
whole, `J` and `K` by one product within 4e-15 of PySCF's `vj` / `vk`; a
non-symmetric `D + A`: `J` against `vj` (`J(A) = 0`), `(K(D) + K(D^T)) / 2`
against `vk`, `K(A)` against a dense `einsum` of the host tensor, all within
4e-15, and with the oracle group PySCF's `get_jk(hermi=0)` itself (`J` and
`K` within 3.6e-15); H2O's shells cut three ways, each range launched alone,
stacked `==` the whole. **On a card: CUDA only, an RTX A6000 (sm_86), not
this box's RTX 3080** -- 2026-11-17, CUDA 13.0: every ordered class at L <= 4, both
layouts, within the bound (worst 4.7e-2 of it), runs bit-identical; H2 and
H2O `J` and `K` within 3.8e-15 of PySCF, for the fixture's `D` and for
`D + A` as interpreted; H2O's three row ranges on the card stacked `==` the
whole matrix, both layouts. **hipcc never compiled it and no gfx1200
ran it**: HIP is owed. Nothing is measured.

**The low-stack full ERI, f and (ff|ff)** (`kernel_eri.eri_lowstack`: any
class with an f shell; `lowstack=` forces either form at any class; the
module docstrings have the design). The one-thread-per-task form holds the
whole block in accumulators: measured `sm_86` ptxas frames 43.9 KB at
(dd|dd) (`eri_2222`, 145 KB of spill stores, a 150 s cold compile), so
(ff|ff) is out of reach. The low-stack form holds no block: per record pair
`[t]^(0)` once into a local `R`, per-axis `E` tables (the ket's from `-X`,
folding `(-1)^{|q|}` into one class sign), and per bra row the expanded
row `E`, `G[q] = SUM_p E_p R[p+q]` and each column's rolled ket sum,
accumulated IN PLACE in the output (the first record pair stores). The
frame at (ff|ff) is 1,251 doubles (R 455, G 84, row E 27, six tables of
112, Boys 13): `eri_3333` / `eri_dj_3333` ~9.8 KB, < 100 B spill, ~3 s
cold compile. Packed: one bra row per thread (`task_threads` = `n_a n_b`);
dense: the `n_b` rows of one bra Cartesian (`kernel_dense.task_threads` =
`n_a`), each by measurement on an RTX A6000 (few tasks favour one row a
thread, up to 5x; a full card favours `n_b`, `[t]^(0)` formed once per
thread); the task is the fast thread index so a warp's lanes share trip
counts. Plain stores, one owner per element (a self-paired task stores
`(pr, pc)` from the row with `pc <= pr`), bit-reproducible, exactly
symmetric; round-off, not bitwise, against the other form. Through (dd|dd)
the default text is byte-identical to #737's (both kernels, both
backends), so FreeBreakfast's emitted d units do not move. The forced
low-stack (dd|dd) was faster than today's in 5 of 6 measured shapes and
1.16x slower in one (packed, 666 tasks, 3-primitive d shells), so at d it
stays an option, off. **Interpreted**: every packed class and every
ordered dense class (both layouts) of an s-to-f basis through (ff|ff),
two-f-shell (ff|ff), and the forced form at L <= 4, within the analytic
`2 gamma_N + delta` bound. **On a card: CUDA only, an RTX A6000, partly** -- the new
device cases (every packed class through (ff|ff), every dense ordering by
L, the forced form at L <= 4, HF/cc-pVTZ's 55 classes against libcint,
CO/cc-pVTZ `J` and `K` against PySCF) were written and a run was stopped
at 77 of 98 cases, none failing; which cases those were is not recorded,
so the f device checks are owed (the follow-up issue). nvcc compiles
every f unit (512 dense + 34 packed, ptxas above); hipcc never, HIP owed.
