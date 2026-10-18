# The one-electron matrices: `S`, `T`, `V` and `h` on the `E` fragment

**Status: implemented, and run on both cards (milestone 11, #232–#244,
#265, #266).** This is the derived note for the overlap `S_ab = <a|b>`, the
kinetic energy `T_ab = <a|-½∇²|b>`, the nuclear attraction
`V_ab = -Σ_C Z_C <a|1/r_C|b>`, and the core Hamiltonian `h = T + V` they sum
to. It is the contract the three kernels were built against, and the one
place that says what is generated, what is reused from the `J` engine, and
where it ran. **Energy order only:** `derivative_order` stays 0, and
one-electron gradients and Hessians are out of scope.

**Nothing here is scored against a published table.** `[JEngine]`
(`docs/references.md` entry 35) is about `J` only. It prints no
one-electron table, count or claim, and no published one-electron board
exists anywhere in `data/literature/`. The `E` coefficients come from the
McMurchie & Davidson report (entry 46). `S`, `T` and `V` are **this
repo's own design** built on it, which is the same position as `DQ` and
`JAB` relative to `[Fock]`. `data/literature/jengine/limits.json` records
this as a `StructuralLimit` (#245) and not as a second literature file.
Milestone 11 added **no** `tables.json` cell, scoreboard, floor or
`claims.json` record, and it changed or scored none.
`test/test_claims.py` and `test/test_scoreboards.py` pass unchanged. A pair
class's `Spec.total_am` is `la + lb`, the Hermite order of one pair. It
does not index a row of `[JEngine]` Tables 1–3 and must never be scored
against one; this is the `DQ`/`JAB` rule again. There is an outside
witness, but it is a different kind of warrant: libcint, through the
tracked fixtures, plus the `h` and `Tr(D h)` capstone.

## The family, as keys

`Spec` (#232) widened `Operator` for the first time. It now has
`OVERLAP`, `KINETIC` and `NUCLEAR` beside `COULOMB`. There are three new
pair targets, `Target.S`, `Target.T` and `Target.V`, each with
`AM_LENGTH = 2`, and each is tied to its operator by `OPERATOR_OF`
(`Spec(COULOMB, 0, (1, 1), Target.S)` is refused). Keys keep their shape:
`overlap/d0/(2,1)/b1/s/literal`, `kinetic/…/t/…` and `nuclear/…/v/…`. Every
pre-existing key is byte-identical, including all 196 measured ones.

The pair-batching ban is now stated **per target**
(`spec._PAIR_BATCH_REFUSAL`). For `DQ`, `JAB`, `S` and `T` the reason is
structural: batching partitions a quartet's Hermite targets, and these have
none. `V` does have Hermite targets, every `t` with `|t| ≤ la + lb`, so it
gets a different refusal: it is unbatched **by milestone 11's decision**,
and a batched `V` kernel is a named follow-up.

## The identity: `2π/p`, and where each half lives

Expand the primitive product in Hermite Gaussians,
`G_a G_b = Σ_t E^{ab}_t Λ_t(ζ, P)` (entry 46). Now replace the `J` ket by a
unit point charge at `C`. This is the `η → ∞` limit of a Gaussian ket. Under
the `J` engine's convention `θ² = ζη/(ζ+η)` it gives **`θ → √ζ`**, and with
`PQ = P − C` the engine's own base, `recurrence.fundamental_prelude`,
becomes

    [0]^(m)(P − C, √ζ) = (2/√π) √ζ (−2ζ)^m F_m(ζ |P − C|²)

That is the physical Hermite Coulomb base `(−2p)^m F_m` times one constant.
The constant is completed **exactly** by the `(π/ζ)^{3/2}` that
`ao.pairs.dress` already multiplies into every bra record:

    (2/√π) √ζ · (π/ζ)^{3/2} = 2π/ζ

which is the whole one-electron prefactor. The two halves live in different
places:

- `(2/√π) θ` is the prelude's `W0`, the same statement the `[p|q]`
  kernel runs.
- `(π/ζ)^{3/2}` is inside the host's per-record scalar `s`
  (`ao.pairs.dress`, `ao.j.epair`'s `s`). In `V` it is passed as `s_ab`.

So `V_ab = −Σ_C Z_C Σ_pp Σ_t s E^{ab}_t [t]^(0)(P − C, √ζ)`, summed over
the dressed block and nothing else. **No kernel and no host module forms a
constant.** A `V` that formed its own `2π/p`, or carried a second `(π/p)`,
would be wrong by construction. `test_ao_potential.py` asserts the identity
(and `θ(ζ, η → ∞) → √ζ`) as a test.

**`K_AB` is already in `E`, per axis.** The seed is
`E^{00}_0 = exp(ζ X_PA X_PB) = exp(−μ X_AB²)`, so each axis table carries its
own share of the Gaussian-product constant, and the product over three
axes is `K_AB`. The predecessor's `sgen`/`tgen`/`vgen` multiply `K_AB` back
in by hand. That step **does not port**: done here, it would multiply
`K_AB` in twice, a smooth per-pair factor of exactly the kind the libcint
comparison exists to catch. The kinetic product rule across the three axes
is clean for the same reason.

The three operators on the dressed block:

| | per Cartesian pair `(a, b)`, per record | reads |
|---|---|---|
| `S` | `s · E_x^{a_x b_x}_0 E_y^{…}_0 E_z^{…}_0`, the `t = 0` column | the axis tables to `(la, lb)` |
| `T` | `s · ((K_x S_y) S_z + (S_x K_y) S_z) + (S_x S_y) K_z`, with `K_ij = −2β² S_{i,j+2} + β(2j+1) S_{ij} − ½ j(j−1) S_{i,j−2}` (the last term **absent**, not zero-weighted, for `j < 2`) | the axis tables to `(la, lb + 2)`, ket widened only; `β` the ket primitive's exponent |
| `V` | `s · Σ_C (−Z_C) Σ_t E^{ab}_t [t]^(0)(P − C, √ζ)` | the whole block, every reachable `t` |

## The host layer (`src/felizjueves/ao/`)

The reference each kernel is held to, all in pure Python:

- **`one_electron.matrix(basis, block, *, pairs=)`** (#233) assembles a
  symmetric AO matrix from a per-shell-pair block. It calls `block` once per
  shell pair `i ≤ j`, copies the lower triangle, and reads a diagonal pair
  from its **upper** triangle, so the result is exactly symmetric. It needs
  no permutation to compare against libcint.
- **`overlap.py`** (#233) builds `S` as element 0 of each dressed block.
  **`kinetic.py`** (#234) builds `T` by the fixed ±2 ket shift on the
  overlap axis tables. **`potential.py`** (#235) builds `V` as above, with
  the `[t]^(0)` evaluator **injected** (`ao.pairs.OneCentre`). Tests pass
  `reference.pq.one_centre`, so no new reference code was written and `ao`
  still imports no `reference/`.
- **`pairs.py`** (#236) adds the one-electron pair bounds, each read off the
  same dressed block: `overlap_bound`, `kinetic_bound` and `potential_bound`
  (`Σ|Z|` times a `T = la + lb` bound weighted by `point_charge_bounds`,
  proved by a Fourier argument). It also adds `pair_list(basis, *, tau,
  bound)`, which returns the kept and dropped shell pairs. The `S` and `T`
  bounds hold in floating point; the `V` bound is proved in exact arithmetic
  and checked on computed blocks with no margin. On the compact molecules
  the screen drops only same-centre pairs of opposite parity, whose block is
  zero by symmetry. Only the H₂ chain exercises the distance decay.
- `reference/boys.py` reaches `T_MAX = 1e6` since #265, measured against the
  Kummer series. `V`'s point-charge kets have no `η` to damp `T`, and they
  reach `T ≈ 1.25e5` on HF/cc-pVTZ (F 1s against H).

## The three kernels

`kernel_s.py` (#237), `kernel_t.py` (#238) and `kernel_v.py` (#239). Each
module has `build_{s,t,v}_kernel((la, lb) | spec)`, `{s,t,v}_spec` and
`launch_shape`. Each builder is vendor-neutral IR with its own layering
rule, which keeps it off `ao`, `interp`, `emit`, `device` and `reference/`.

**One thread per primitive-pair record**, `rec = THREAD_X + BLOCK_X ·
BLOCK_DIM_X`, guarded by `rec < n_rec`. All three use **`kernel_jab`'s
scatter rule, unchanged**. These are symmetric matrices, and the rule is
already proved by `test_kernel_jab.py`:

    off-diagonal (off_a ≠ off_b):  M[off_a+i, off_b+j] += v;  M[off_b+j, off_a+i] += v
    diagonal     (off_a = off_b):  only i ≥ j;  M[off_a+i, off_a+j] += v,
                                   and for i ≠ j also M[off_a+j, off_a+i] += v

Every write is an `AtomicAdd` into a matrix **the caller zeroes**. Because
the interpreter runs threads in order and blocks in order, `S` and `T`
interpret **bitwise `==`** to the host block placed by this rule. The rule
keeps a diagonal pair's **lower** triangle while `ao.one_electron.matrix`
reads the upper one. For a symmetric operator the two agree to rounding but
not necessarily bitwise, so the full host matrix is held to a stated
`8 ε` bound on diagonal-pair elements.

**The parameter blocks:**

| kernel | scalars | per record | per nucleus | out |
|---|---|---|---|---|
| `s_{la}{lb}` | `n_rec n_ao` | `zeta_ab s XPA YPA ZPA XPB YPB ZPB off_a off_b` | — | `S[n_ao,n_ao]` |
| `t_{la}{lb}` | `n_rec n_ao` | `zeta_ab s eb_ket XPA YPA ZPA XPB YPB ZPB off_a off_b` | — | `T[n_ao,n_ao]` |
| `v_{la}{lb}` | `n_rec n_ao n_nuc` | `zeta_ab s_ab XP YP ZP XPA YPA ZPA XPB YPB ZPB off_a off_b` | `Z Cnuc[,3]` | `V[n_ao,n_ao]` |

**The record arrays are structure of arrays, and travel as one struct.**
Every per-record array is flat and `n_rec` long. A Cartesian quantity is
split per axis (`XPA YPA ZPA`, not an interleaved `XPA[n_rec, 3]`), so with
one thread per record a warp's lanes read consecutive doubles on every load.
The arrays cross the call as one `PrimPairSoA` (`prim_pair.py`, an
`ir.Bundle`), a struct of 14 pointers passed by value, so the signature is
`s_{la}{lb}(int n_rec, int n_ao, PrimPairSoA pp, double* S)`. The struct
is the superset of what the four families read (`zeta s eb XP YP ZP XPA YPA
ZPA XPB YPB ZPB off_a off_b`). A kernel binds the fields it reads, at the
top of its body, under its own parameter names (`const double* const s_ab =
pp.s;`), and leaves the rest unread, so a host fills one value per class
and hands it to every family. Inside the kernel, and to the interpreter,
each field is still the array `Param` of its own name. The `gpu` backend
includes `include/felizjueves/prim_pair.h`; the vendor backends spell the
same text in the unit (`emit._c.bundle_struct`, which `test_emit_gpu.py`
holds the header to). Since #422 the Schwarz kernel (`schwarz_{la}{lb}`)
takes the same struct, binding `zeta s XP..ZPB`; its `off_a` / `off_b` are
per shell pair, not per record, so they stay plain arguments beside `rec0` /
`rec1` (`kernel_schwarz`'s docstring). Since #421 the J chain's pair kernels
(`DQ`, `JAB`) take it too, and `K` two values of it (its bra and ket
records); `[p|q]` and the DF metric still take interleaved `[n_rec, 3]`
arrays.

The names are chosen for hygiene. `zeta_ab` is not `zeta` because the
fragment binds a scalar `zeta`. `eb_ket` is not `eb` because the kinetic
fragment binds `eb`, and it is passed rather than rebuilt from `ζ`, which
would need `α` too and would round. `V`'s `s_ab` is not `s` because the
reused prelude declares a local `s = −2θ²`. `V` needs `P` because
`PQ = P − C` cannot be formed from `XPA`/`XPB` alone.

**What each kernel inlines:**

- **`S`**: `e_fragment.overlap_fragment(la, lb)`. This is the three axis
  bodies from the unchanged `_axis_body`, forming only the `t = 0` products
  into a local `E0`. A liveness pass (`_live`) drops axis entries that no
  `t = 0` product reads, so no kernel declares a local it never uses. Before
  the pass, `s_21` gave hipcc 18 `-Wunused-variable` warnings, which #92's
  filter fails.
- **`T`**: `e_fragment.kinetic_fragment(la, lb)`. This is the axis bodies of
  `_axis_body(la, lb + 2, d)`, one `K{d}_i_j` per axis, and the bracket into
  a local `KE`, in `ao.kinetic`'s association operation for operation.
- **`V`**: `e_fragment.block_fragment(la, lb)`, the whole block and the
  first consumer to read every column. `θ = √ζ` is formed once. Then the
  kernel loops over the nuclei **serially** (`ir.For` over `n_nuc`), and
  for each one runs `PQ = P − C`, `fundamental_prelude`, `recurrence_body`,
  then `u_r = Σ_t E[r,t] [t]^(0)` over the reachable columns and
  `acc_r −= Z_C u_r`. Finally it forms `v_r = s_ab · acc_r`.

`V` is **round-off, not bitwise**, against `ao.potential`. The kernel
follows MILD's directions and the sieve's re-parenting in the schedule's
order, while the host recurses along `reference.pq.one_centre`'s fixed
rule. It also applies `s` once per row after the nucleus sum. The bound is
`16 ε M`, where `M` is the absolute-value sum the element is built from;
the worst observed is `1.7 ε M`.

**`V` carries the wide `fj_boys`** (#266). `build_v_kernel` sets
`Kernel.wide_boys`, so its Boys function is `emit/_c.py`'s
`BOYS_DEVICE_WIDE`: the Kummer series below `T = 125` and the asymptote
from there. Every `J`-chain unit keeps the narrow series, because its text
is pinned by `kernels.json`. `limits.json` records why there are two
spellings, and the follow-up to unify them when `J` is next re-measured.

## Reused verbatim, and deliberately not

**Reused unchanged** (no edit to `plan/`, `recurrence.py` or the axis
builder):

- `e_fragment._axis_body`, the one axis recurrence. `S` and `T` take a
  subsequence of its statements, and `T` reads it at `lb + 2`.
- `e_fragment.block_fragment`, inlined whole into `V`. It was refactored
  onto `_axes_body`/`_product`, and its emitted text did not change; the
  `e_block_probe_11` goldens are byte-identical.
- `recurrence.fundamental_prelude` and `recurrence.recurrence_body`, at
  `θ = √ζ`. `test_kernel_v.py` builds them from `plan(v_spec(la, lb))`
  itself and asserts that they appear contiguously, exactly once, in the
  nucleus loop.
- `plan/dag.build` and the rest of the plan layer. `dag.build` reads only
  `spec.total_am` and `spec.n_batches`, so at Hermite order `la + lb` `V`
  inherits MILD, the sieve, the partition, the schedule and the cost model.
  `V` is the plan layer's first non-`J` consumer. `S` and `T` use none of
  it: no Boys, no `[r]` tower, no DAG and no schedule.
- `kernel_jab`'s scatter rule, and `emit/` itself, which already emits
  `fj_boys` whenever a kernel calls `Fn.BOYS`.

**Deliberately not reused:**

- **`ClassLaunch`.** It is `J`-shaped: a bra side and a ket side, ket records
  keyed by `I_q · max|D_q|` and sorted so Alg. 1's `break` can end a bra
  row, and a batch count per `(lp, lq)`. A one-electron build has one side
  and no density. `launch.py` has a sibling type instead (#241):
  `one_electron_launch(basis, nuclei, *, tau, bound) → OneELaunch` of
  `OneEClass(la, lb, recs, blocks)` over `OneERec`s. One record set serves
  all three kernels (`OneEClass.arrays(target)`, in each kernel's own
  parameter order), and the screen drops whole shell pairs **before** the
  launch through `pair_list`. `walk_one_electron` rebuilds `S`, `T` or `V`
  from the records alone, and the result is `==` to the host on every
  tracked basis. `PairRec`, `ClassLaunch`, `LaunchPlan` and `launches` are
  untouched, and the `J`-plan fingerprint is unchanged.
- **The `D_q` pre-contraction** (`Target.DQ`). There is no density to
  contract.
- **The sort key** `I_q · max|D_q|` (#147). It is a property of the density,
  and here it does not exist.

## The generator

`python -m felizjueves.emit --one-e-l [LMAX]` (#242; `3` when the flag is
bare, accepted 0..8, anything else exits 2) **adds** the one-electron
ladder, `s_{la}{lb}`, `t_{la}{lb}` and `v_{la}{lb}` for every `la ≤ lb ≤
LMAX`, beside whichever `J` ladders `--L` and `--pair-l` select. The no-flag
command is unchanged: it writes 37 units, and every `kernels.json`
`source_sha256` pin still holds. `KERNEL_GEN_CMD` in `devtools/config.sh`
is now `env PYTHONPATH=src python -m felizjueves.emit --one-e-l 2`, so the
default kernel tier writes **55 units**: the 37 `J`-chain units plus 18
one-electron units through (dd).

**The ladder writes `la ≤ lb` only.** A real basis's launch keys a class by
shell index `i ≤ j`, not by angular momentum. So H₂O (O's p shells before
H's s) and HF/cc-pVTZ also reach `(1,0)`, `(2,0)`, `(2,1)` and so on, and
`test/one_electron_driver.py` emits those `la > lb` units itself: the same
text, under the same flags and warning filter. When the generate phase
wrote a unit, the driver includes that file and asserts it byte-identical
to `emit(...)`. Ordered ladder classes are a named follow-up.

## Witnesses, and what ran where

**Interpreted, fast tier:**

- `test_e_fragment.py`: each fragment bitwise against `ao.hermite.e_axis`
  and `ao.kinetic.kinetic_axis`.
- `test_kernel_s.py` and `test_kernel_t.py`: every class (ss) through (ff),
  plus HF/cc-pVTZ's whole matrix, bitwise `==` to the host.
- `test_kernel_v.py`: within `16 ε M`. Classes past L = 4 are `slow`.

**Host against libcint** (#240):

The fixtures are at schema 2. Each of 12 fixtures (11 at #240; the
cross-overlap's `h2o_sto3g_x_631g_moved` is the twelfth) carries libcint's
`int1e_ovlp`, `int1e_kin` and `int1e_nuc` (point nuclei) and PySCF's
`hcore`, or none of the four. Every element is compared within
`1e-12 · scale`, where `scale` is the absolute-value sum the build adds,
completed by Cauchy–Schwarz:

| matrix | worst err / scale | case |
|---|---|---|
| `S` vs `int1e_ovlp` | 6.2e-16 | prim_spdf |
| `T` vs `int1e_kin` | 7.6e-16 | hf_ccpvtz |
| `V` vs `int1e_nuc` | 5.4e-14 | prim_spdf |
| `T + V` vs `hcore` | 3.5e-14 | prim_spdf |

s/p classes run in the fast tier, and classes from l = 2 are `slow`.

**On both cards**, the RTX 3080 (`sm_86`, nvcc 13.0) and the RX 9060 XT
(`gfx1200`, hipcc / ROCm 7.2.4):

- `test/test_one_electron_device.py` (`gpu` + `cuda`/`hip`; #242, and
  #244's deferred tier through (ff)):
  - **Part 1.** Every ordered class of HF/cc-pVTZ, through (dd) by default
    and through (ff) under `slow`, against the host. The per-element
    running-error bound `(2γ_{N+K} + δ) M_abs` is stated in the module
    docstring before any run. Worst fractions of that bound: `S` 8.2e-2,
    `T` 2.6e-2, `V` 1.1e-2, the same on both cards; `V`'s runs were
    bit-identical on both.
  - **Part 2.** `S`, `T` and `V` of H₂O/STO-3G and H₂O/cc-pVDZ against
    libcint at `1e-12 · scale`. Worst: `S` 5.6e-16, `T` 2.1e-16, `V`
    3.1e-14, identical digits on both cards.
- `test/test_hcore_device.py` (#243), the capstone: `h_dev = T_dev + V_dev`
  from one launch, compared with `hcore` element by element (worst
  1.88e-14 of scale). It also compares `Tr(D h_dev)` with `Tr(D hcore)`,
  where `D` is the fixtures' Hilbert density, so this is the `Tr(D h)`
  functional at that density and not an SCF energy. H₂O/STO-3G gives
  `|ΔTr|` 7.11e-15, and H₂O/cc-pVDZ 7.11e-14, which is 4.46e-5 of its
  tolerance. A flipped `−Z_C` sign, a dropped `T` or `V`, or a 1% slip in d
  normalisation each moves the trace by at least `1e6` tolerances, and that
  floor is asserted.

**Every kernel-tier run of milestone 11 was capped at L ≤ 8 by the user.**
No `J` kernel text changed, so the L = 16 `J` tier was not re-run.
#244's deferred tier ran with the generator at
`--L 0..8 --pair-l 3 --one-e-l 3` (59 units). It gave **CUDA 95 passed /
3 skipped** and **HIP 96 passed / 3 skipped**; the skips are three opt-in
recordings. The default `KERNEL_GEN_CMD` (the `J` ladder to L = 16 plus
`--one-e-l 2`) has not been run as a whole.

## What is not measured, and what does not reach

**Nothing about the one-electron kernels is measured.** No
`measured_store` row exists, and there is no wall-clock and no counters.
The register and spill figures below are compiler readings reported in
PRs #297 and #298 (nvcc `-Xptxas -v` in bytes; hipcc
`kernel-resource-usage`). They are not written to `data/measured/`:

- **`S` and `T` do not spill** on either card through (dd). At (ff),
  `gfx1200` spills 36 VGPRs for `S` and 182 for `T`, and `sm_86` spills
  80 B for `T`.
- **`V` spills from `la + lb = 3` on both cards.** At (ff), `sm_86` reports
  23,588 B of spill stores and 32,444 B of spill loads in an 84,944 B stack
  frame. `gfx1200` reports 4,154 VGPRs spilled and 14,820 B of scratch.
- **`V` (ff) on the RTX 3080 has about 170 MiB of headroom.** The
  per-thread stack is reserved for every resident thread (104,448), so the
  program's VRAM probe asked for 8,461 MiB with 8,633 MiB free. It ran and
  passed. About 170 MiB more of that card in use by anything else would
  make the probe refuse it. The RX 9060 XT asked for 463 MiB.

The test plan's `kernel-tier` / `high-L` markers are not registered in
`pyproject.toml`. The deferred cells use the registered `slow` marker,
which keeps them off the pre-push gate.

## The cross-basis overlap `S_{B1,B2}`

**Status: implemented, and run on both cards.** This section covers the
rectangular overlap `S_{B1,B2}[m, n] = <m|n>`, where `m` is an AO of the
**row** basis `B1` and `n` an AO of the **column** basis `B2`. The two can
be one molecule in two basis sets (a minimal-to-larger projection's
overlap) or two geometries. The integral is `S`'s, the `t = 0` column of
the dressed block. Only the **matrix** differs: it is `n_a × n_b`, it has no
mirror, and it has no diagonal rule. That is why it is its own target and
not a lever on `S`. Like the rest of this note, nothing here is scored or
measured.

**The key.** `Target.SX` (`sx`, operator `OVERLAP`) is an **ordered** pair
class `(la, lb)`, where `la` is always the row basis's shell. No symmetry
relates `(la, lb)` to `(lb, la)` across two bases, and both are reached. Its
batching refusal is `S`'s structural one. It has no kernel-body lever (its
`levers.DEFAULT_REFUSALS` row). An example key is
`overlap/d0/(2,1)/b1/sx/literal`, and every pre-existing key is unchanged.

**The host.** `ao.overlap.cross_overlap_matrix(basis_a, basis_b, *,
pairs=)` reuses `overlap_block` unchanged. Its blocks are placed by
`ao.one_electron.rectangular`, which calls `block` once per **ordered**
shell pair `(i ∈ A, j ∈ B)` in row-major order and writes every element
once, nothing mirrored. `ao.pairs.cross_pair_list` is `pair_list`'s screen
over those pairs, with the bound injected (`overlap_bound`). A basis
against itself is `overlap_matrix` exactly wherever `matrix` reads the same
block entry: the upper triangle by shell, and a diagonal pair's upper
triangle. Elsewhere it agrees within `8 ε` of scale, because `matrix`
mirrors while `rectangular` computes `<b|a>` along the other `E` path.

**The launch.** `launch.cross_overlap_launch(basis_a, basis_b, *, tau,
bound) → XLaunch` builds `XClass(la, lb, recs, blocks)` over `XRec`s. This
is a sibling of `OneELaunch`, not a widening of it: that record set is one
basis's `i ≤ j` pairs, its diagonal is read from `off_a == off_b`, and it
carries nuclei and the ket exponent. Here `off_a` and `off_b` index
different matrices, so a record has no diagonal. `walk_cross_overlap`
rebuilds the matrix from the records and equals the host to 0.

**The kernel.** `kernel_sx.build_sx_kernel((la, lb))` emits `sx_{la}{lb}`.
It reuses `kernel_s.record_values` (the per-record statements, factored out
of `kernel_s` with `S`'s emitted text byte-identical through (gg) on both
backends). Its scatter is one `AtomicAdd` per element,
`S[off_a+i, off_b+j] += v`, with no offset comparison. The parameter block
is `S`'s with `n_ao` split: `n_rec n_a n_b | zeta_ab s XPA YPA ZPA XPB
YPB ZPB off_a off_b | S[n_a, n_b]`, the record arrays in `S`'s
`PrimPairSoA`. Interpreted, the whole rectangular matrix is
**bitwise `==`** to `cross_overlap_matrix`, which is a stronger statement
than `S`'s, because no triangle is read. Its layering rule `sx_builder` is
`kernel_s`'s plus `kernel_s` itself.

**The generator.** `python -m felizjueves.emit --sx-l [LMAX]` (2 when bare,
accepted 0..8) **adds** every ordered class `(la, lb)` with
`la, lb ≤ LMAX` beside the J ladders. It is never in the no-flag output,
and `KERNEL_GEN_CMD` does not ask for it.

**The witnesses.**

- *libcint.* Three tracked fixtures each carry
  `int1e_ovlp_cross:<rows>`, libcint's `gto.intor_cross('int1e_ovlp',
  rows, cols)`. It is cross-checked at generation against
  `gto.conc_mol`'s off-diagonal block. The fixtures are
  `h2o_sto3g_x_631g` and `h2o_sto3g_x_ccpvdz` (rows `h2o_sto3g_j`'s AOs,
  columns the `_j` twin's inputs exactly) and `h2o_sto3g_x_631g_moved`
  (the columns at another geometry). `oracle.manifest.cross_overlap`
  loads the row fixture and checks it. Each fixture is read in both
  orientations, so every ordered class through d is reached. The
  tolerance is `ONE_E_TOL` of the element's absolute-value scale,
  Cauchy-Schwarz completed from each basis's own diagonal; the host's
  worst is about `3e-16`.
- *Fast tier.* `test_ao_cross_overlap.py` covers the host, the screen,
  the launch and the walk. `test_kernel_sx.py` checks the kernel
  interpreted, every ordered class (ss) through (ff) on HF/cc-pVTZ
  against itself and the three cases whole, all `np.array_equal`.
- *Both cards.* `test_sx_device.py` (generator `test/sx_driver.py`) runs
  the same cases under `S`'s device bound, `2 γ_{N+K} Mabs` with
  `N = 4L + 9`, and checks run-to-run spread within `2 γ_K Mabs`. On the
  RTX 3080 and the RX 9060 XT all 7 cases passed. The worst result was
  `0.103` of the bound (HF/cc-pVTZ through (ff)), and every case agreed
  with libcint to at most `3.3e-16` of scale.

**Not built:** the projection that would consume it (for example
`C_B = S_BB⁻¹ S_BA C_A`), a cross-basis `T` or `V`, and any measurement.

## Named follow-ups

- **A batched `V` kernel.** `V` has Hermite targets, so the ban that #232
  restated per target is a decision for `V` and not structural.
- **A warp-per-record `V`** with the nuclei strip-mined over `THREAD_Y`, the
  `Target.J` shape. This is where `V`'s spills and its 3080 headroom would
  be answered.
- **One-electron derivatives** (gradients and Hessians), excluded by this
  milestone's scope.
- **One `fj_boys` spelling** (the #266 record): unify on the wide routine
  when `J` is next re-measured.
- **Ordered ladder classes** (`la > lb`), so the generator, rather than the
  test-side driver, writes every class a real basis reaches.
