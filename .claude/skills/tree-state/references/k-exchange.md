# Exchange `K` (milestone 10) and the block kernels

Moved verbatim from `.claude/CLAUDE.md`, "The state of the tree" (indexed by `../SKILL.md`).

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
`(dd|dd)`; `test/kblock_driver.py` runs either kernel on one upload (on the
shared runner since #657, which has not run on a card), and on
the RTX 3080 both matched `walk_kb` and libcint's `vk` on H₂O/STO-3G (τ 0 and
1e-8), two H₂ and H₂O/cc-pVDZ (`slow`), and since #484 the decaying chain
with its per-block factors (against `walk_kb`, 2.60e-16 of max); the RX
9060 XT is compile-only (
no open issue tracks it; #431 closed on the card's detection fix, not the run). **Nothing about it is measured yet:** `devtools/measure.sh kblock`
would time both per class on H₂O/cc-pVDZ into `data/measured/kblock.json`,
and its one attempt was refused by the card gate (the RTX 3080 busy from
outside the container), so that file does not exist.
**Beside it, the block kernel split per column or element, with a density
stack** (#548; `[Fock, §4.4]`): `eri/k_exchange/kernel_kblock_split.py`
(`build_kblock_split_kernels(cls, granularity, R=, lanes=)`, `kbs_*`), a
sibling of `kernel_kblock`, not a lever: its body and parameters unchanged
but the accumulator extent, moved through `KHooks.keep` — `block` (`n_a n_b`
slots, the control), `column` (one kernel per `b_j`, `n_a` slots; stage 1 not
recomputed) or `element` (one per `K[a_i, b_j]`, one slot) — and `R`
densities sharing one sweep (`Dcd[R, ...]`, `K[R, ...]`, `V` formed once
and digested `R` times; `dmax[blk]` is the stack's bound, the host's).
`shared_doubles(am, accumulators, R) = n_p + accumulators·R`, over 48 KiB
refused (`SHARED_REFUSAL`; `(dd|dd)` admits R = 1 / 10 / 61 at wave64).
Interpreted, every set is `kernel_kblock`'s `K` **bit for bit** (the per-slot
fold is kept), a stack is `R` one-density builds bit for bit, and the plan
through the split sets is the block kernel's bitwise on two H₂ and
H₂O/STO-3G (`test_kernel_kblock_split.py`, `(dd|dd)` `slow`).
`test/kblock_driver.py`'s opt-in MODE 2 (`split=`) launches a set on the
block kernel's upload; `test_kblock_split_ondevice.py` (`slow`, `(dd|dd)`
only by the user's decision, on two contracted d shells) ran on the **RX
9060 XT only**: column and element both within 9.28e-16 of `max|K|` of
`walk_kb`, 7.74e-17 of the block kernel (not bitwise on the card); the RTX
3080 was not enumerable in that session ("no CUDA-capable device"), so on
CUDA it is compiled only. `devtools/measure.sh
kblock-split` (a compiler, no card) writes registers and spill per kernel
into `data/measured/kblock_split.json`, its verdict rule declared before any
row: `(dd|dd)` only; on `sm_86` `kb` spills 14,732 B, the worst column kernel
6,724 B and the worst element kernel 1,340 B (all 255 registers); on
`gfx1200` 2,204, 1,659 and 715 spilled VGPRs (all 192 VGPRs): a spill win
on both, never a clock claim. Nothing is timed; the launch plan over the sets
and the stack's host API are follow-ups.
**Beside it, the block kernel with `E` from global memory and the
contraction rolled, which takes K to f** (#571; #569's factored `E` did not:
the binding constraint was the compile, the unrolled `E^ac [p|q] E^bd D`
products, `kb_33_33` 22 MB of source and still in `cicc` at 2.7 h):
`eri/k_exchange/kernel_kblock_rolled.py` (`build_kblock_rolled_kernel`,
`kbr_{la}{lc}_{lb}{ld}`, opt-in, one batch, no lever), `kernel_kblock`'s
mapping, plan, screen, `dmax[blk]` and `K[n_ao, n_ao]` output unchanged and
`kernel_k.bra_scalars` / `ket_body` (the generated recurrence) shared, with
three K-seam hooks replaced: the recurrence's `[r]^(0)` stored into a local
`R[n_r]`, no `E` fragment inlined (`Eac[n_bra, n_eac]` / `Ebd[n_ket, n_ebd]`
arrive per record, nonzero entries only, from the host's
`launch_kblock.e_table` on `mmd.hermite.e_block`, the ket's carrying
`(-1)^|q|`), and `kernel_k`'s stage 1 and fused stages 2 and 3 emitted as
**real loops** over five class tables (`tables(am)`: the two `E` patterns'
CSR and `hpq`, the `R` index of `p + q`), so per lane only `R`, `I[n_p]` and
`K_AB` are live and the text is the recurrence plus a fixed set of loops.
Interpreted it is `kernel_kblock`'s `K` **bitwise** for every class with
every `l <= 2`, and within `ORACLE_TOL` of `walk_kb` for every class with
an f shell through `(ff|ff)` (`test_kernel_kblock_rolled.py`, above
L = 4 `slow`). `test/kblock_driver.py`'s `rolled=True` stages the `E`
tables and class tables and launches it as MODE 0 (`with_k=False` leaves
`kernel_k` out). On the **RTX 3080** (`test_kblock_ondevice.py`'s `-rolled`
cells): H₂O/STO-3G at τ 0 and 1e-8, two H₂, the decaying chain and
H₂O/cc-pVDZ (`slow`) against the walk, the host and libcint's `vk` (worst
3.75e-15 of `scale_K`, `kb`'s), and the f cells (`slow`, `high_l`, run by
hand) `sf` (16 classes) and `spdf` (one s, p, d, f shell, two primitives:
every K class with every `l <= 3`, 256 of them, `(ff|ff)` included) against
the walk, worst 2.28e-16 of max, every run bit-identical. The RX 9060 XT
is out of scope by the user's decision (#431, since closed on the card's
detection fix; no open issue tracks the run). `devtools/measure.sh
kblock-rolled-compile` (a compiler, no card) writes
`data/measured/kblock_rolled.json`'s `compile` rows: on `sm_86` `kb_22_22`
1.67 MB, 64 s, 816 MB peak RSS, 255 registers, 14,732 B spill stores;
`kbr_22_22` 37 KB, 1.0 s, 191 MB, 128 registers, no spill, 1,320 B stack;
`kbr_33_33` 97 KB, 1.1 s, 192 MB, 254 registers, no spill, 3,640 B stack
(`R`). `devtools/measure.sh kblock-rolled` timed `(dd|dd)` on #569's input
(2,304 records a side, 256 blocks) behind the gate: `kb` 1,110 ms, the
rolled kernel 2,620 ms, **2.36x slower** (its declared rule's `slower`;
re-timed with #575's third arm in one session, 1,090 and 2,547 ms, 2.34x), a
record and not a gate: `kb` stays the default through d, the rolled kernel
is for the classes nothing else compiles. **Why, read off `ncu` on the RTX
3080** (#574; `devtools/measure.sh kblock-rolled-profile`, one launch each,
behind the gate, pinned to 16 cores; `kblock_rolled.json`'s `profile` rows,
read under `profile_declared`, written first, into `profile_reading`):
`kbr` executes **5.21x** `kb`'s warp instructions (5.74e10 against 1.10e10;
1.43x the fp64 work, the `sq * Ebd` product per term) and issues 2.25x as
often (20.2 against 45.3 cycles between a warp's issues), so 2.37x the time.
Both run one warp per scheduler (achieved occupancy 7.8% of a theoretical
10.4%, shared-memory-limited, grid 256), and both are latency-bound: H1 (`kbr`
latency-bound on dependent loads) **supported**: long + short scoreboard
81% of its stall, 83% of its PC samples on the two rolled loops, the
`bd_q -> hpq -> R` chain (line `Ebd[(k * 336 + ie)]`, 55%); H2 (`kb`'s spills
hit L1 or are cold) **contradicted**: its local loads hit L1 1.2% of the
time (1.93e10 sectors) and long scoreboard is 53% of its stall; H3 (the
unrolled text buys issue parallelism) **contradicted**: `kb` issues 2.21% of
cycles against `kbr`'s 4.97%. `kbr`'s global loads average 6.38 sectors per
request (81% L1 hits); `Ebd`'s own is **not measured** (ncu's SASS-patched
source counters end in LaunchFailed on `kbr`, recorded), only read off the
text: a 2,688 B lane stride, 32 sectors per full-warp request
(`profile_reading.summary.ebd`, for #575 and #576). `ncu` replays here
also LaunchFail intermittently and are retaken (`gate_discarded_runs`).
**Between them, a dense-Hermite form, measured and not chosen** (#575):
`kernel_kblock_rolled.build_kblock_dense_kernel` (`kbd_{la}{lc}_{lb}{ld}`,
opt-in, one batch, no lever; `kbr`'s text byte-identical) keeps the `(b, d)`
and `(a, c)` row loops rolled and unrolls the Hermite `p` and `q` over `E`
rows stored DENSE, zeros kept (`launch_kblock.e_table(..., dense=True)`,
`Eac[n_bra, n_ac n_p]` / `Ebd[n_ket, n_bd n_q]`, no class tables), so every
`p + q` is a constant: no `R` array and none of #574's `bd_q -> hpq -> R`
chain. Interpreted it equals `kb` **by value** for every class with every
`l <= 2` and `walk_kb` within `ORACLE_TOL` with an f shell through `(ff|ff)`
(`test_kernel_kblock_rolled.py`, above L = 4 `slow`); on the **RTX 3080**
`test_kblock_ondevice.py`'s `-dense` cells (the rolled cells' four, H₂O/
cc-pVDZ `slow`) and its `sf` / `spdf` f cells (worst 5.57e-16 of max
against the walk, runs bit-identical) passed; the RX 9060 XT is out of
scope (#431's, since closed on the detection fix; no open issue tracks it). Compile (`kblock-rolled-compile`, `sm_86`): `kbd_22_22` 111 KB,
0.9 s, 191 MB, 255 registers, 1,012 B spill stores, 1,016 B stack;
`kbd_33_33` 509 KB, 2.3 s, 197 MB, 255 registers, 11,372 B spill stores,
10,512 B stack -- f compiles. Clock (`kblock-rolled`, behind the gate, rule
`dense_declared` written before any dense row): `(dd|dd)` `kb` 1,090 ms,
`kbr` 2,547 ms, `kbd` **4,140 ms** (3.80x `kb`, 1.63x `kbr`); `(ff|ff)` on
16 f shells, one primitive each (256 records a side, 256 blocks), `kbr`
1,266 ms, `kbd` **3,827 ms** (3.02x) -- `slower` everywhere: the structural
zeros' multiplies (3.75x `kbr`'s contraction terms at d, 4.4x at f) cost
more than the indirection they remove. **Which form a class uses: through
d `kb`, at f `kbr`; `kbd` at neither.**
**`kbr` with its ket `E` transposed, measured and not made the default**
(#576): `build_kblock_rolled_kernel(..., transposed=True)` (`kbrt_*`, arm
`rolled-T`, opt-in) is `ir.passes.transpose(kernel, ("Ebd",))`, `Ebd`
stored `[n_ebd, n_ket]` so a warp's consecutive kets read adjacent doubles
(`Eac`, a broadcast, unchanged); the default text is byte-identical, pinned
by digest (`test_kernel_kblock_rolled.py`) and against #574's profiled
`kbr_22_22` (`measure.sh check`). `test/kblock_driver.py`'s
`transposed=True` stages `Ebd` so. Interpreted it is `kbr`'s `K` **bit for
bit** on every #571 case (every class through `(ff|ff)`, the s, s, p plan,
two H₂ screened); on the **RTX 3080** its `-rolled-T` cells (the four
rolled cells, H₂O/cc-pVDZ `slow`, `sf` / `spdf` `high_l`) passed with
`kbr`'s numbers, runs bit-identical. Clock (`devtools/measure.sh
kblock-transpose`, behind the gate, rule `transpose_declared` written
before any row, rows in `transpose_clock`): `(dd|dd)` `kb` 1,075 ms, `kbr`
2,552 ms, `kbrt` **2,391 ms** (`faster` than `kbr`, 0.937; 2.22x `kb`
against `kbr`'s same-session 2.37x); `(ff|ff)` on #575's f input `kbr`
1,262 ms, `kbrt` 1,266 ms (1.003, `neutral`). A first session, lost
before it was stored, read the same (0.924 and 0.982). The rule moves the
default only if `rolled-T` wins at both classes, so **the default stays
record-major and the recommendation is unchanged: through d `kb`, at f
`kbr`**; the layout narrowed the d gap by about 6% and moved f not at all.
Sectors (`kblock-transpose-profile`, `ncu`, one launch each): kernel-level
global loads 6.376 sectors per request on `kbr_22_22` against **2.404** on
`kbrt_22_22`, the same 1.17e10 requests, 0.377x the sectors, 0.929x the
time under `ncu` -- so the transposed reads coalesce and the kernel stays
latency-bound on what is left. The `Ebd` load's own per-instruction
sectors are again not measured (SASS-patched counters LaunchFailed on
`kbr`), and on `kbrt` #574's full `details` set LaunchFailed on all 12
takes of three sessions while the narrow sector set ran, recorded in its
row (`transpose_profile`, `transpose_profile_reading`).
**`kbr`'s column form, which takes K to g, measured and not made the
default** (#583): `build_kblock_rolled_kernel(..., column=True)` (`kbrc_*`,
arm `rolled-C`, opt-in; `kbr`'s parameters and staging, the default text
byte-identical, `kernel_k.ket_body` split into its screen and
`ket_quartet` with its text unchanged). No lane carries a whole `K_AB`
partial to the end of the block. Per (bra, ket strip, `b`), the lane's
column `col[a]` in `Ksh[n_p + a]` is reduced by one `WarpReduce` per `a`.
Lane 0 adds each sum into one static `Kblk[n_a n_b]` a block (lane 0's
alone, so no barrier) and then `Kblk` into `K`. The control flow is
warp-uniform: strips from `ket0` in steps of `blockDim.y`, a `break` on
the strip's first (largest) ket, and a lane past `ket1` or under its own
bound (the walk's per-pair test) running a real ket with `s_bd` zeroed,
which adds exact zeros. Footprint per 32-lane block (`column_shared_bytes`):
10,784 B at `(dd|dd)` against 18,176, 24,864 at `(ff|ff)` against 47,104,
and **47,880 at `(gg|gg)` against 99,840, under 48 KiB with no opt-in**. It
reassociates the sum, so it is never bitwise against `kb` / `kbr`.
Interpreted it is within `ORACLE_TOL` of `walk_kb` for every class through
`(ff|ff)` at four lanes, the s, s, p plan and two H₂ screened
(`test_kernel_kblock_rolled.py`, above L = 4 `slow`). **On the RTX 3080**
(`test_kblock_ondevice.py`'s `-rolled-C` cells), runs bit-identical:
- the four rolled cells, against the walk, the host and libcint;
- H₂O/cc-pVDZ (`slow`): 81 classes, 3.78e-16 of the walk, 3.75e-15 of
  `scale_K` against libcint;
- `sf` and `spdf` (`high_l`): 2.23e-16 and 2.73e-16 of the walk;
- **`gg`** (`high_l`): one g shell, two primitives, the one class
  `(gg|gg)`, held to `walk_kb` only, 4.60e-16. It is the first g launch
  of any K form. A 625-class s, p, d, f, g cell was tried and dropped:
  about 45 minutes of it was pure-Python host references.

On the RX 9060 XT only the fast cells ran, inside the kernel-tier gate
(the rest is out of scope; #431, which held it, closed on the detection
fix and no open issue tracks it). Compile (`devtools/measure.sh
kblock-column-compile`, each unit alone, against a 120 s budget):
- `sm_86`: `kbrc_22_22` 1.0 s, 96 registers (`kbr`: 128), no spill;
  `kbrc_33_33` 1.1 s, 254 registers, no spill; `kbrc_44_44` 2.4 s, 255
  registers, 8,160 B of spill stores, 14,736 B of stack.
- `gfx1200`: 192 VGPRs each, 1.6 / 1.7 / 3.1 s.

Clock (`devtools/measure.sh kblock-column`, behind the gate, rule
`column_declared` written before any row, rows in `column_clock`):
- `(dd|dd)`: `kb` 1,078 ms, `kbr` 2,530 ms, `kbrc` 2,619 ms, so 1.035x
  `kbr` (`neutral`) and 2.43x `kb` (`slower`);
- `(ff|ff)`: `kbr` 1,264 ms, `kbrc` 1,227 ms, 0.971 (`neutral`).

The rule moves nothing, so **the recommendation is unchanged: through d
`kb`, at f `kbr`, and at g `kbrc`, the only form that launches there**.
Why, read off `ncu` (`kblock-column-profile`, one `details` take each,
`column_profile_reading`):
- theoretical occupancy went from 10.4% to 16.7%, the shared-memory limit
  from 5 to 8 blocks an SM;
- achieved occupancy stayed 7.78%, one warp per scheduler. #569's
  `(dd|dd)` input is 256 blocks on 68 SMs, under 4 warps an SM, so it is
  grid-limited, not shared-memory-limited;
- the stall mix is unchanged (long scoreboard 56%, long + short 81%),
  with 1.06x the warp instructions (the reductions and the dead lanes).

The occupancy gain needs a grid wider than this input to show.
