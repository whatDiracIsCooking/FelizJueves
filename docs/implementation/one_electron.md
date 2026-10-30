# The one-electron matrices: `S`, `T`, `V` and `h` on the `E` fragment

**Status: implemented, and run on both cards through (ff) (milestone 11,
#232–#244, #265, #266; `V`'s low-stack form at (ff), #586).** This is the derived note for the overlap `S_ab = <a|b>`, the
kinetic energy `T_ab = <a|-½∇²|b>`, the nuclear attraction
`V_ab = -Σ_C Z_C <a|1/r_C|b>`, and the core Hamiltonian `h = T + V` they sum
to. It is the contract the three kernels were built against, and the one
place that says what is generated, what is reused from the `J` engine, and
where it ran. **Energy order only here:** the d1 keys of `S`, `T` and
`V`, their contracted nuclear gradients, are `gradient.md`'s; their contracted hessians
(d2) are `hessian.md`'s.

**Nothing here is scored against a published table.** `[JEngine]`
(`docs/references.md` entry 35) is about `J` only. It prints no
one-electron table, count or claim, and no published one-electron board
exists anywhere in `data/literature/`. The `E` coefficients come from the
McMurchie & Davidson report (entry 46). `S`, `T` and `V` are **this
repo's own design** built on it, which is the same position as `DQ` and
`JAB` relative to `[Fock]`. `data/literature/jengine/limits.json` records
this as a `StructuralLimit` (#245) and not as a second literature file.
Milestone 11 added **no** `table{1,2,3}.json` cell, scoreboard, floor or
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
  (`ao.pairs.dress`, `ao.density.epair`'s `s`). In `V` it is passed as `s_ab`.

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

## The host layer (`src/felizjueves/one_electron/` and `ao/`)

The reference each kernel is held to, all in pure Python:

- **`one_electron.one_electron.matrix(basis, block, *, pairs=)`** (#233)
  assembles a
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
keeps a diagonal pair's **lower** triangle while `one_electron.one_electron.matrix`
reads the upper one. For a symmetric operator the two agree to rounding but
not necessarily bitwise, so the full host matrix is held to a stated
`8 ε` bound on diagonal-pair elements.

**The parameter blocks:**

| kernel | scalars | per record | per nucleus | out |
|---|---|---|---|---|
| `s_{la}{lb}` | `n_rec n_ao` | `zeta_ab pa_{x,y,z} pb_{x,y,z} s_ab off_a off_b` | — | `S[n_ao,n_ao]` |
| `t_{la}{lb}` | `n_rec n_ao` | `zeta_ab pa_{x,y,z} pb_{x,y,z} s_ab eb_ket off_a off_b` | — | `T[n_ao,n_ao]` |
| `v_{la}{lb}` | `n_rec n_ao n_nuc` | `zeta_ab p_{x,y,z} pa_{x,y,z} pb_{x,y,z} s_ab off_a off_b` | `Z Cnuc[,3]` | `V[n_ao,n_ao]` |

Every per-record array is flat, one pointer per scalar: a record vector is
three parameters, one per axis (`pa_x pa_y pa_z` is `P − A`), so with one
thread per record a warp's lanes read consecutive doubles on every load.
They are plain positional parameters, not a struct (#424's `PrimPairSoA`
was reverted). The per-nucleus `Cnuc` stays `[n_nuc, 3]`: every lane reads
the same nucleus.

The names are chosen for hygiene. `zeta_ab` is not `zeta` because the
fragment binds a scalar `zeta`. `eb_ket` is not `eb` because the kinetic
fragment binds `eb`, and it is passed rather than rebuilt from `ζ`, which
would need `α` too and would round. The record weight is `s_ab`, not `s`,
in every family: `V`'s reused prelude declares a local `s = −2θ²`, and in
`S` and `T` a bare `s` sat beside the output `S` and read as it. `V` needs `P` because
`PQ = P − C` cannot be formed from `P − A` and `P − B` alone.

**What each kernel inlines:**

- **`S`**: `e_fragment.overlap_fragment(la, lb)`. This is the three axis
  bodies from the unchanged `_axis_body`, forming only the `t = 0` products
  into a local `E0`. A liveness pass (`_live`) drops axis entries that no
  `t = 0` product reads, so no kernel declares a local it never uses. Before
  the pass, `s_21` gave hipcc 18 `-Wunused-variable` warnings, which #92's
  filter fails.
- **`T`**: `e_fragment.kinetic_fragment(la, lb)`. This is the axis bodies of
  `_axis_body(la, lb + 2, d)`, one `K{d}_i_j` per axis, and the bracket into
  a local `KE`, in `one_electron.kinetic`'s association operation for operation.
- **`V`**: `e_fragment.block_fragment(la, lb)`, the whole block and the
  first consumer to read every column. `θ = √ζ` is formed once. Then the
  kernel loops over the nuclei **serially** (`ir.For` over `n_nuc`), and
  for each one runs `PQ = P − C`, `fundamental_prelude`, `recurrence_body`,
  then `u_r = Σ_t E[r,t] [t]^(0)` over the reachable columns and
  `acc_r −= Z_C u_r`. Finally it forms `v_r = s_ab · acc_r`. That is the
  default form at every class but **(ff)** (`la + lb ≥ 6`), where the
  default is the low-stack form below.

**`V`'s low-stack form, the default at (ff)** (#586, stage 1; the rule
narrowed after it). `build_v_kernel(cls)` picks the form by
`kernel_v.v_lowstack(cls)`: true when `la + lb ≥ LOWSTACK_MIN_LSUM = 6`,
which through f is (ff) alone. `lowstack=True` / `False` forces either form
at any class. At every other class the text is the unrolled kernel's, byte
for byte. `test_kernel_v.py` pins the cuda and hip text of all fifteen
classes below `la + lb = 6` (through (dd), and `v_03`, `v_13`, `v_23` and
their mirrors) by sha256 from cc5d16f.

**Why `la + lb ≥ 6`.** A slower form ships only where the faster one does
not fit. Stage 1 first made every class with an f shell low-stack, but its
own timing (below) has the low-stack form slower at the mixed f classes,
up to about 1.25× at (df), and only the (ff) unrolled frame blows the
8 KB budget (85 KB, against 6 KB at (df)). The rule is the class's Hermite
order `L = la + lb`, which fixes the recurrence and the block's width
(`hermite_targets(L)` columns); its rows, `n_cart(la) · n_cart(lb)`, are
largest at the most even split. So at every `L ≤ 5` no class has a larger
block than (df)'s 60 × 56, the measured 6 KB frame, and past f (gs) and
(gp) stay unrolled. At `L ≥ 6` the rule picks the low-stack form, measured
at (ff) only. A lopsided class such as (is) (28 × 84) might fit unrolled,
but nothing past f is generated, and the rule errs to the smaller frame. Both forms have the same name
`v_{la}{lb}`, the same parameters in the same order, the same
one-thread-per-record launch and the same `V` contract (the caller zeroes
`V`, the kernel only `AtomicAdd`s). So a launcher needs no change, and
free-breakfast's `tools/emit_one_electron_kernels.py` keeps calling
`build_v_kernel(cls)`. The unrolled kernel materialises `E[100][84]` per
thread at (ff), 67 KB. The low-stack form holds no `E^{ab}` block:

- **Per-axis `E` tables.** `e_fragment.axis_tables_fragment(la, lb)` runs
  `_axes_body` unchanged and stores each `E^{ij}_t` into local `Ex`, `Ey`
  and `Ez` of `(la+1)(lb+1)(la+lb+1)` doubles at
  `axis_table_index(i, j, t)`. That is 336 doubles at (ff), against the
  block's 8,400.
- **`[t]^(0)` into a local `R`**, per nucleus, in `hermite_targets` order,
  from the plan's recurrence as before (84 doubles at (ff)).
- **The contraction as real loops.** `e_fragment.cartesian_loops` walks the
  block rows in `cartesians` order. `e_fragment.rolled_e_sum` forms
  `u = Σ_tx Ex (Σ_ty Ey (Σ_tz Ez R[h(t)]))` over exactly the reachable box
  `t_d ≤ a_d + b_d`, with `h = hermite_index(t)` in closed form. The
  accumulators are a local `acc[n_rows]`, and the scatter is the same
  rule, rolled.

So the frame is the three tables, `acc` and `R` (4,160 B at (ff)). The
text is the recurrence plus a fixed set of loops. The rolled pieces live
in `mmd/e_fragment.py` so that `DQ`, `JAB` and `RECORD_BOUND` (#586 stages 2
and 3) can reuse them. `rolled_e_sum` takes the consumer's `X[t]` read as a
callable.

**Why this design, and not kbr's.** `kbr` (#571) passes its `E` values and
class tables as kernel parameters. `V` must keep its ABI, and the IR has no
constant-initialised array. So the row tables are loops, and the Hermite
index is arithmetic. The per-axis product costs two extra multiplies per
reachable term, and no structural zero is multiplied. #575's dense form
lost to exactly those zeros.

Interpreted, the low-stack form agrees with the host within the unchanged
`16 ε M` at every class through (ff) (worst 1.85 ε M, at (dp); 1.32 at an f class). The unrolled form
does too, at every class, and the two agree with each other within the
same bound. All of it is in `test_kernel_v.py`, with L > 4 `slow`.
`test_e_fragment.py` holds the tables to `e_axis` bitwise,
`hermite_index` to `hermite_targets` through L = 8, and the rolled row sum
to `e_block`'s row `· X` within `8 ε Σ|E||X|`. The golden
`test/golden/{cuda,hip}/v_33.*` is the (ff) default text.

`V` is **round-off, not bitwise**, against `one_electron.potential`. The kernel
follows MILD's directions and the sieve's re-parenting in the schedule's
order, while the host recurses along `reference.pq.one_centre`'s fixed
rule. It also applies `s` once per row after the nucleus sum. The bound is
`16 ε M`, where `M` is the absolute-value sum the element is built from;
the worst observed is `1.7 ε M`.

**`V`'s Boys function.** `T = ζ |P − C|²` is undamped (~1.25e5 on
HF/cc-pVTZ), so `V` was the first kernel to need more than the Kummer
series, and #266 gave it a second, wide `fj_boys`. Since #301 there is one:
`emit/_c.py`'s `fj_boys<M>` template, every order `F_0..F_M` of one `T` from
one `exp(−T)`, witnessed on the card over `reference.boys`' whole domain
(`T ∈ [0, 1e6]`). `limits.json` (the #266 record) has the history.

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
  `fj_boys` whenever a kernel holds a Boys fill.

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
is now `... --L 0 1 2 3 4 5 6 7 8 --pair-l 3 --one-e-l 2` (#625), so the
default kernel tier writes **47 units**: the `J` chain to `[p|q]` L = 8 plus
18 one-electron units through (dd). `--full` generates
`KERNEL_FULL_GEN_CMD`, `--one-e-l 2` alone: the 37 `J`-chain units plus
the same 18, 55 units.

**The ladder writes `la ≤ lb` only.** A real basis's launch keys a class by
shell index `i ≤ j`, not by angular momentum. So H₂O (O's p shells before
H's s) and HF/cc-pVTZ also reach `(1,0)`, `(2,0)`, `(2,1)` and so on, and
`test/one_electron_driver.py` emits those `la > lb` units itself: the same
text, under the same flags and warning filter. When the generate phase
wrote a unit, the driver includes that file and asserts it byte-identical
to `emit(...)`. Since #658 the driver runs on the shared runner
(`test/harness/`): one runner per class's unit, the matrix carried from
class to class, and the whole chain run `RUNS` times from zero, since the
fp64 atomics' order may differ between runs; that port ran on both
cards on 2026-10-30. Ordered ladder classes are a named follow-up.

## Witnesses, and what ran where

**Interpreted, fast tier:**

- `test_e_fragment.py`: each fragment bitwise against `mmd.hermite.e_axis`
  and `one_electron.kinetic.kinetic_axis`.
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
recordings. The `KERNEL_GEN_CMD` of that time (the `J` ladder to L = 16
plus `--one-e-l 2`, now `KERNEL_FULL_GEN_CMD`) was not run as a whole in
milestone 11.

## What is not measured, and what does not reach

**Nothing about the one-electron kernels is measured.** No
`measured_store` row exists, and there is no wall-clock and no counters.
The register and spill figures below are compiler readings reported in
PRs #297 and #298 (nvcc `-Xptxas -v` in bytes; hipcc
`kernel-resource-usage`). They are not written to `data/measured/`:

- **`S` and `T` do not spill** on either card through (dd). At (ff),
  `gfx1200` spills 36 VGPRs for `S` and 182 for `T`, and `sm_86` spills
  80 B for `T`.
- **`V` spills from `la + lb = 3` on both cards** in the unrolled form,
  which is the default everywhere but (ff). At (ff) it was a frame of about
  85 KB on `sm_86`, `E[100][84]` and spills. Because CUDA reserves the
  per-thread stack for every resident thread (104,448 on the RTX 3080),
  the VRAM probe asked for about 8.5 GB of the 10 GB card. It passed alone
  and failed whenever anything else held about 170 MiB (#586).
- **The f units, each form** (#586). The compiles used
  `nvcc -O3 -arch=sm_86 -Xptxas -v -cubin` (CUDA 13.0) and
  `hipcc --offload-arch=gfx1200 -O3 -Rpass-analysis=kernel-resource-usage`
  (ROCm 7.2), on free-breakfast's spelling (`gpu.emit` with its
  `potential_en<la, lb>` specialization). Columns: generated source,
  `sm_86` stack / spill stores / registers / compile time, then `gfx1200`
  scratch per lane / VGPRs.

  | unit | form | source | sm_86 stack | spill st | regs | nvcc | gfx1200 scratch | VGPRs |
  |---|---|---|---|---|---|---|---|---|
  | `v_03` | unrolled (cc5d16f), **default** | 14,756 B | 0 B | 0 B | 218 | 0.3 s | 36 B | 192 |
  | `v_03` | low-stack | 12,060 B | 0 B | 0 B | 164 | 0.3 s | 0 B | 154 |
  | `v_13` | unrolled, **default** | 42,960 B | 1,872 B | 2,068 B | 255 | 0.5 s | 1,860 B | 192 |
  | `v_13` | low-stack | 18,455 B | 1,488 B | 0 B | 128 | 0.4 s | 1,488 B | 96 |
  | `v_23` | unrolled, **default** | 103,924 B | 6,192 B | 7,820 B | 255 | 1.5 s | 6,380 B | 192 |
  | `v_23` | low-stack | 27,484 B | 2,656 B | 0 B | 168 | 0.5 s | 2,672 B | 174 |
  | `v_33` | unrolled | 224,889 B | 85,472 B | 24,008 B | 255 | 9.0 s | 14,756 B | 192 |
  | `v_33` | low-stack, **default** | 40,213 B | **4,160 B** | 0 B | 254 | 0.5 s | 4,208 B | 192 |

  The issue's 85,312 B for `potential_en<3,3>` was read through
  free-breakfast's build; FJ's spelling gives 85,376–85,472 B. FJ's own
  `emit.cuda` text compiled with `nvcc -O3 -arch=sm_86 -rdc=true -Xptxas
  -v -c` (CUDA 13.0) reads the unrolled `v_03` / `v_13` / `v_23` /
  `v_33` at 0 B / 1,904 B (2,064 B of spill stores) / 6,192 B (7,920 B) /
  85,312 B of stack, and the low-stack ones at 0 / 1,488 / 2,656 /
  4,160 B. Only `v_33` is over the 8 KB budget, so it alone is low-stack
  by default. At (ff)
  the low-stack frame is exactly the five local arrays (3 × 112 + 100 +
  84 doubles) and spills nothing. On the RTX 3080, the CO/cc-pVTZ and
  HF/cc-pVTZ (ff) probes now ask for 415 MiB (4,160 B × 104,448). The
  ordered `v_30`, `v_31`, `v_32` match their mirrors (0, 1,488 and
  2,656 B).
- **A wall-clock reading, not a store row** (#586; hand-written harness,
  not behind the gate; the card's lock held). The input was 65,536 random
  records, 8 nuclei and the best of 6 launches. The two forms took
  `v_03` 1.97 / 1.80–2.10 ms, `v_13` 2.88 / 3.16–3.61 ms,
  `v_23` 5.82 / 7.27–7.31 ms and `v_33` 17.97 / 16.54–16.63 ms (unrolled
  / low-stack). So `(ff)` is about 0.92× the unrolled time, and the
  mixed f classes cost up to about 1.25× (`(df)`). They agree to
  `≤ 1e-15` of `max|V|`. Stage 1 kept "every class with an f shell" for
  one simple selection; the rule was then narrowed to `la + lb ≥ 6`,
  (ff) alone, because the repo ships a slower path only where it is
  needed, and `v_03`, `v_13`, `v_23` fit unrolled.
The test plan's `kernel-tier` / `high-L` markers are not registered in
`pyproject.toml` (its `high_l`, since #301, is L > 8 only). The deferred
cells use the registered `slow` marker, which keeps them off the pre-push
gate.

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

**The host.** `one_electron.overlap.cross_overlap_matrix(basis_a, basis_b, *,
pairs=)` reuses `overlap_block` unchanged. Its blocks are placed by
`one_electron.one_electron.rectangular`, which calls `block` once per **ordered**
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
is `S`'s with `n_ao` split: `n_rec n_a n_b | zeta_ab pa_{x,y,z}
pb_{x,y,z} s_ab off_a off_b | S[n_a, n_b]`. Interpreted, the whole rectangular matrix is
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
- *Both cards.* `test_sx_device.py` (generator `test/sx_driver.py`, on
  the shared runner since #658, which ran on both cards on 2026-10-30) runs
  the same cases under `S`'s device bound, `2 γ_{N+K} Mabs` with
  `N = 4L + 9`, and checks run-to-run spread within `2 γ_K Mabs`. On the
  RTX 3080 and the RX 9060 XT all 7 cases passed. The worst result was
  `0.103` of the bound (HF/cc-pVTZ through (ff)), and every case agreed
  with libcint to at most `3.3e-16` of scale.

**Not built:** the projection that would consume it (for example
`C_B = S_BB⁻¹ S_BA C_A`), a cross-basis `T` or `V`, and any measurement.

## The dipole `MU`

**Status: implemented, and run on both cards (milestone 19, #517–#524),
through (dd); through (ff) since #627, on both cards since the #658
runner port (2026-10-30).** This section covers the Cartesian dipole moment
integrals `mu^d_ab = <a| q (r_d - C_d) |b>`, `d ∈ {x, y, z}`: the
multipole moments of order `e = 1` about an origin `C`, scaled by a
charge `q`. Like the rest of this note, nothing here is scored or
measured, and the same `StructuralLimit` reasoning applies
(`data/literature/jengine/limits.json`, #524's record).

**The key.** `Target.DIPOLE` (`dipole`, operator `Operator.DIPOLE`, #518)
is a pair class `(la, lb)`, `AM_LENGTH = 2`; an example key is
`dipole/d0/(2,1)/b1/dipole/literal`, and every pre-existing key is
unchanged (the 510-key J digest included). Its batching refusal is `S`'s
structural one, HGP is refused as for `S` (`_NO_HGP_ONE_E`), and it has no
kernel-body lever (its `levers.DEFAULT_REFUSALS` row, the one-thread-per-
record wording of `s`, `t` and `sx`). **The order `e` is a builder
argument, not a key field** (`spec.py`, "A DIPOLE CLASS"): the quadrupole
is its own `Target.QUADRUPOLE` sharing the builder (below), so `Spec`
gained no field and no stored key moved. **The origin `C` and the charge `q` are
runtime inputs, not key fields either**: one kernel serves every origin and
every charge.

**The Hermite route.** Per axis, the moment of one Cartesian pair `(i, j)`
is `SUM_{t <= min(e, i+j)} E^{ij}_t M^e_t`, with the Hermite moments
`M^0_0 = 1` and `M^{e+1}_t = t M^e_{t-1} + X_PC M^e_t + M^e_{t+1}/(2ζ)`,
`X_PC = P - C` (Helgaker, Jørgensen & Olsen, §9.5, `docs/references.md`
entry 48; **the equation number inside §9.5 was not checked against the
text**, and entries 39 and 46 were not found to state the moments). `E`
entries with `t > i + j` are structurally absent and dropped from the sum,
never read as zero. At `e = 1` an axis is `E^{ij}_1 + X_PC E^{ij}_0`, and
on the axes other than `d` it is the overlap's `E^{ij}_0`. The value is
the axis product in the overlap's association, `(q s) ((Mx My) Mz)`, with
`s = c_a c_b (π/ζ)^{3/2}` (`dress`'s scale) and the moments' own
`sqrt(π/ζ)` left in `s`. `q` is applied once per record, `q s`, so `q = -1`
is exactly the negation of `q = 1`. Both the host and the fragment are
built general in `e`; **only `e = 1` ships**.

**The host.** `one_electron.multipole` (#520; at `one_electron/` and not
`ao/`, because milestone 18 left only `shell`, `pairs` and `density` in
`ao/`): `multipole_block(sa, sb, C, e=1, q=1.0)` →
`[n_comp][n_cart(sa)][n_cart(sb)]` and `multipole_matrix(basis, C, e=1,
q=1.0)` → `[n_comp][n_ao][n_ao]`, placed by `one_electron.one_electron.matrix`
so every component is exactly symmetric. Components come in `cartesians(e)`
order, x, y, z at `e = 1`, libcint's. At `e = 0`, `q = 1` it is
`overlap_block` bitwise. `walk_one_electron(plan, basis, Target.DIPOLE,
origin=C, q=q)` rebuilds `MU` from a `one_electron_launch` plan and equals
`multipole_matrix` exactly; `OneEClass.arrays(Target.DIPOLE)` is `V`'s
record arrays (#521).

**The fragment.** `e_fragment.multipole_fragment(la, lb, e)` (#519) builds
the three axes from `_axes_body` unchanged, forms `XPC_d` from the scalar
binds `P_d` and `C_d`, folds the integer moments (`M^0_0`, `M^e_e = e!`)
at build time, and writes one value per component and block row. It has a
**`store=` hook**: the consumer supplies the statement placed where each
value is formed, so a kernel can `AtomicAdd` each value as it forms and no
`[n_comp, rows]` local array exists. `from_order=0` writes the `e = 0`
component first from the same axis tables.

**One fused kernel writes every component.**
`kernel_mu.build_mu_kernel((la, lb))` (#521) emits `mu_{la}{lb}`: one
thread per primitive-pair record, the record loaded once and
`multipole_fragment(la, lb, 1)` inlined once, so the `E` tables are built
once for all three components, and each value is `AtomicAdd`-ed into
`MU[n_comp, n_ao, n_ao]` (component stride `n_ao²`) through the `store=`
hook. There is no per-component kernel. The scatter is `kernel_jab`'s rule
per component, unchanged, spelled per value as `If(od, (both mirror adds),
(the diagonal's adds, or none))` with `od` formed once per record. The
parameter block is `kernel_s`'s plus `P` (under `kernel_v`'s names, in its
place) and the four runtime scalars: `n_rec n_ao | zeta_ab p_{x,y,z}
pa_{x,y,z} pb_{x,y,z} s_ab off_a off_b | C_x C_y C_z q | MU[n_comp, n_ao,
n_ao]`. `MU` must be zeroed by the caller (a `Note`). `with_overlap=True`
adds `S` as component 0, bitwise `kernel_s`'s at `q = 1`. Interpreted, the
kernel is **bitwise `==`** to `multipole_block` placed by `kernel_jab`'s
rule; against `multipole_matrix` off-diagonal shell pairs are exactly
equal and diagonal pairs are held to `16 ε max|MU_k|` (measured 0),
because `matrix` reads a diagonal pair's upper triangle and the kernel
keeps its lower. Its layering rule `mu_builder` keeps it off `ao`,
`one_electron.multipole`, `ir.interp`, `emit`, `performance.device`,
`reference/` and `launch`.

**The generator.** `python -m felizjueves.emit --mu-l [LMAX]` (#522; 2 when
bare) **adds** `mu_{la}{lb}` for every `la ≤ lb ≤ LMAX` beside whichever J
ladders are selected. **It accepts 0..3 only** (0..2 until #627): above
that, or negative, it exits 2 (`the dipole is validated through (ff) only
(#627)`), so the CLI cannot write a g unit. It is refused with `--class`,
`--shell-ls`, `--fock-l`, `--fock-class`, `--n-batches` and on the HGP
path, never in the no-flag output, and `KERNEL_GEN_CMD` does not ask for
it. Ten goldens per backend, `test/golden/{cuda,hip}/mu_{00,01,02,03,11,
12,13,22,23,33}` (the largest, `mu_33`, 135,887 B, no digests file); the
six through (dd) were unchanged byte for byte by #627.

**The witnesses.** Through (dd) every shell is `l ≤ 2`, so every class is
L ≤ 4 and fast-tier on the host. At f (#627), (sf) and (pf) are fast-tier
and (df) and (ff), L = 5 and 6, are `slow` (`tiers.FAST_L_MAX`).

- *libcint and PySCF* (#517, #520). Three fixtures, `h2o_sto3g_dipole`,
  `h2o_631g_dipole` and `h2o_ccpvdz_dipole`, on their `_j` siblings'
  inputs exactly (`oracle.manifest.dipole_pair` asserts it), carry
  libcint's `int1e_r_cart` under `mol.with_common_orig(C)` at
  `C = (0.37, -0.81, 1.23)` bohr (every component non-zero, the three
  magnitudes distinct) and at `C = 0`, and PySCF's RHF `dip_moment` at
  both origins with the charges and coordinates it read. A fourth,
  `hf_ccpvtz_spd_dipole` (HF in Cartesian cc-pVTZ with every f shell
  removed, 40 AOs, no density), carries `int1e_r_cart` at both origins
  and `int1e_ovlp`; its d shells sit on both atoms, so all nine ordered
  classes (ss) through (dd) occur between two centres. The generator
  checks `r_C = r_0 - C S` before writing. A fifth (#627),
  `hf_ccpvtz_dipole`, is the same molecule with its f shells kept (F
  4s3p2d1f, H 3s2p1d; 50 AOs, no density): all sixteen ordered classes
  (ss) through (ff) occur, each with one f between two centres, (ff) on F
  alone. It carries `int1e_r_cart` at both origins and `ovlp` (libcint's
  `int1e_ovlp`) but none of the four one-electron arrays, which would put
  it over the 512 KB hook (541 KB); `hf_ccpvtz` carries them for its
  inputs. `manifest.dipole_integrals(F_DIPOLE)` loads it under
  `DIPOLE_F_L_MAX = 3`.
- *Host.* `test_ao_multipole.py`: whole H₂O matrices against
  `int1e_r_cart` at both origins within `1e-12` of scale (worst 7.6e-16),
  every block of `hf_ccpvtz_spd_dipole` (worst 8.8e-16) and, per class
  through (ff), of `hf_ccpvtz_dipole` (#627), `dip_moment` as
  `-Tr(D mu) + SUM_A Z_A (R_A - C)`, the (s|s) and (s|p) closed forms,
  the **origin-shift identity** `mu(C') = mu(C) + q (C - C') S`, and
  `q = -1` against `q = 1`.
- *Fast tier, interpreted.* `test_e_fragment.py` (the fragment at
  `e = 0, 1, 2` against `mmd.hermite.e_axis` and a direct `M^e_t`
  recursion, and against Gauss-Hermite quadrature), `test_kernel_mu.py`
  (the bitwise statements above, every class in both orders, the plan
  launched and walked against libcint; at f, #627, every ordered class with
  an f bitwise the host block, `q` an exact factor and the overlap
  component `kernel_s`'s, the plan within 2.25e-15 of `max|r|` of
  `int1e_r_cart`, the origin shift), `test_launch.py` /
  `test_launch_oracle.py` (the dipole walk), `test_emit_mu.py` and
  `test_emit_cli.py` (the ladder, its cap and refusals; the ten compile
  cases per compiler, `cuda`/`hip`, all green on a device-less box for
  #627).
- *Both cards* (#523). `test_mu_device.py` (generator `test/mu_driver.py`,
  which takes `C` and `q` as runtime values, so one build serves every
  origin and charge -- on the shared runner since #658, which ran on
  both cards on 2026-10-30 -- and refuses a class above (ff) since #627): per class on
  `hf_ccpvtz_spd_dipole` against the host under `2 γ_{N+K} MUabs` with
  `N = 4L + 16` and against `int1e_r_cart`; whole H₂O/STO-3G and 6-31G
  (cc-pVDZ `slow`) against `int1e_r_cart` and, at `q = -1`, against
  `dip_moment`; and the origin shift with `S` built on the card by
  `kernel_s`, within `1e-14` of `max|MU|`. On the **RTX 3080 and the
  RX 9060 XT** the tier (`-k mu`) gave 4 passed / 0 skipped each, and the
  slow case by hand 5 passed each. The two cards agree on every quoted
  maximum: `8.62e-16` of `max|r|` against libcint, `3.16e-16` against
  `dip_moment`, `7.23e-2` of the per-element host bound, and `2.36e-16`
  of `max|MU|` for the origin shift against the card's `S`. Runs are not
  always bit-identical between repeats (the fp64 atomics' order), within
  the run-to-run bound `2 γ_K MUabs`. **At f (#627) the card cases**, Part 1 and Part 3 on `hf_ccpvtz_dipole`
  (`slow`), first ran with the #658 runner port on 2026-10-30 and passed
  on both cards; their compile-only companions pass under nvcc and hipcc.

**Not built, and what does not reach.** `e = 1` only; Cartesian only;
nothing above (ff) is built, emitted, launched or checked, and `--mu-l`
refuses above 3. Nothing is measured: #544's `nvcc` compile reported 0
spill bytes for `mu_22` (128 registers) and `mu_22` with the overlap (200),
and #627's, at the tier's flags, 0 stack and 0 spill bytes on sm_86 for
every class through (ff) (`mu_23` 255 registers, `mu_33` 234) and 0
scratch on gfx1200 except `mu_33`, 92 bytes a lane (192 VGPRs) -- so the
dipole needs no low-stack form at (ff), unlike `V` (#586, 85 KB a thread):
sanity readings, not store rows. No consumer (an SCF dipole driver, say) is
built.

## The quadrupole

**Status: implemented and interpreted through (dd); emitted for both
vendors; compiled by nvcc and hipcc and run on both cards since the #658
runner port (2026-10-30)** (the session that built it had neither compiler
nor card; all six units passed clang 18's
CUDA front end, device side, `-Wall -Wextra`, a declaration stub for the
runtime: a proxy). This section covers the Cartesian second moments
`Q^{de}_ab = <a| q (r_d - C_d)(r_e - C_e) |b>`, the six components xx, xy,
xz, yy, yz, zz (`cartesians(2)` order, the upper triangle of libcint's
nine-component `int1e_rr`). These are the RAW second moments, not the
traceless (Buckingham) quadrupole tensor `Θ = (3 Q - tr(Q) 1) / 2`, which
is a linear combination of them the consumer forms. Nothing here is scored
or measured.

**The key.** `Target.QUADRUPOLE` (`quadrupole`, operator
`Operator.QUADRUPOLE`) is a pair class `(la, lb)`; an example key is
`quadrupole/d0/(2,1)/b1/quadrupole/literal`. Every pre-existing key is
unchanged. Its batching refusal is the dipole's structural one; HGP is
legal (`quadrupole.hgp`, "The HGP route" below), and Rys is refused by the
general `_NO_RYS`.

**The kernel is the dipole's builder at `e = 2`.** `kernel_mu.ORDER` maps
each target to its order, and `build_quad_kernel(cls)` (`quad_{la}{lb}`) is
`build_mu_kernel` on a quadrupole key: the same parameter block (`P`, the
runtime origin `C` and charge `q`), the same `sq = q s` per record, the same
`multipole_fragment` inlined once (now at `e = 2`) and the same scatter per
component, into `MU[6, n_ao, n_ao]`. At `e = 2` the moments read `1/(2ζ)`
(`M^2_0 = X_PC X_PC + 1/(2ζ)`, `M^2_1 = X_PC + X_PC`, `M^2_2 = 2`), so the
record also forms `half_inv_zeta`. `with_overlap` is the dipole's alone and
a quadrupole key refuses it. The plan carries it unchanged:
`OneEClass.spec` / `arrays(Target.QUADRUPOLE)` (the dipole's record arrays)
and `walk_one_electron(..., Target.QUADRUPOLE, origin=, q=)`, which returns
the six components.

**The host** is `one_electron.multipole.multipole_matrix(basis, C, e=2)`,
already general in `e`.

**The oracle.** A separate witness, not a fixture:
`test/oracle/quadrupole/hf_ccpvtz_spd_quadrupole.json`, libcint's
`int1e_rr_cart` under `mol.with_common_orig(C)` on `hf_ccpvtz_spd_dipole`'s
inputs (HF/cc-pVTZ without f, every class (ss) through (dd) between two
centres) at that fixture's own origin and at 0. `python -m oracle.generate
--regenerate --quadrupole` writes it and refuses unless libcint's mirrored
components (yx, zx, zy) equal their twins bitwise and the origin shift
`Q_de(C) = Q_de(0) - C_d r_e(0) - C_e r_d(0) + C_d C_e S` holds to `1e-14`.
`manifest.quadrupole_integrals()` loads it with its parent and refuses any
input (`mol`, `basis`, `cart`, `nao`, the origin, the component order) that
is not the parent's exactly.

**Witnesses, all interpreted** (fast tier):

- the host against `int1e_rr_cart` at both origins, worst 1.1e-15 of
  `max|rr|`, and libcint's `rr` at 0 from the host's at `C` through
  libcint's own `r` and `S` (`test_ao_multipole.py`);
- the kernel, per ordered class (ss)–(dd), one launch over every record:
  **bitwise** the host block placed by `kernel_jab`'s rule at both origins,
  `q = 1`, `-1` and `0.75` (`q = -1` the exact negation); the whole matrix
  bitwise the placed host, and `multipole_matrix(e=2)` off the diagonal
  shell pairs bitwise and on them within `16 eps` of each component's scale
  (measured 0); the plan launched and walked against `int1e_rr_cart`,
  whole and per shell-pair block, within `1e-12` (measured 1.1e-15); the
  walk `==` `multipole_matrix`; and the origin shift
  `Q_de(C') = Q_de(C) + δ_d μ_e + δ_e μ_d + δ_d δ_e q S`, every term an
  interpreted kernel (`μ` and `q S` from the dipole's `with_overlap`
  launch), within `1e-14` (`test_kernel_quad.py`);
- six goldens per backend, `test/golden/{cuda,hip}/quad_*`, the largest
  `quad_22` at 99 KB (`test_emit_quad.py`).

**The generator.** `python -m felizjueves.emit --quad-l [LMAX]` (2 bare,
**0..2 only**, exit 2 above, the dipole's cap and its reason) adds
`quad_{la}{lb}` for every `la <= lb <= LMAX` beside the J ladders, never in
the bare default nor in `KERNEL_GEN_CMD`, refusing the selector, the Fock
and HGP paths and `--n-batches` as `--mu-l` does.

**On a card: compiled, and run on both cards.** `test/mu_driver.py` takes `target=`
(`Target.DIPOLE` by default, every dipole call unchanged), and
`test_quad_device.py` (`gpu` + `cuda`/`hip`) builds the spd basis's
quadrupole from its plan on each card against libcint (`1e-12` of
`max|rr|`) and the host under a per-element bound stated in its docstring
(`2 γ_{N+K} Qabs`, `N = 4L + 8 + 16`), with `q = -1` from the same binary.
Its fast-tier cases (the steps' parameters, the staging through the
interpreter) pass, and since #658 (on the shared runner) its compile cases
pass under nvcc and hipcc with no warning; its device case first ran on
2026-10-30 and passed on the RTX 3080 and the RX 9060 XT. Nothing
above (dd), nothing spherical, no traceless tensor, no consumer.

## The HGP route: `S`, `T`, `V`, `MU` and the quadrupole on engine #2

**Status: `S`, `T`, `V` and `MU` built and interpreted, s and p shells;
compiled and run on both cards (the RX 9060 XT since the #658 runner port).
The dipole and the quadrupole through (ff): interpreted and compiled by nvcc
and hipcc, on no card above (pp), the quadrupole on none at all** (below,
"The moments through f"). `one_electron/kernel_hgp.py` builds the same five matrices
on Head-Gordon & Pople's split of Obara & Saika's one-electron recurrences,
promoted from the spike `experimental/hgp/one_electron/` (whose route and
oracle it keeps). The MMD kernels above are untouched, byte for byte.

**Keys.** `Target.S`, `T`, `V`, `DIPOLE` and `QUADRUPOLE` with
`engine=Engine.HGP` (`kernel_hgp.hgp_spec`), legal since this change
(`spec.py`, "AN HGP ONE-ELECTRON CLASS"; `SX` stays refused). The key spells
the engine as a slot suffix, `overlap/d0/(1,0)/b1/s.hgp/literal`, so no MMD
key moves; `s.hgp`, `t.hgp`, `v.hgp`, `dipole.hgp` and `quadrupole.hgp` are
engine slots (`levers.ENGINE_SLOTS`) that refuse every lever. Units are
`hs_{la}{lb}`, `ht_`, `hv_`, `hmu_`, `hquad_`.

**The route.** Per primitive pair a VRR on the bra centre builds `(e|0)`;
the pairs are summed; then the HRR `(a|b+1_d) = (a+1_d|b) + AB_d (a|b)` runs
once per shell pair on contracted values. Each matrix is a combination of
contracted overlap-type integrals:

| Matrix | From contracted `(a\|b)` | VRR order |
|---|---|---|
| `S` | `(a\|b)` | `la + lb` |
| `MU_d` | `q [(a\|b+1_d) + (B_d − C_d)(a\|b)]` | `la + lb + 1` |
| `Q_de` | `q [(a\|b+1_d+1_e) + BC_e (a\|b+1_d) + BC_d (a\|b+1_e) + BC_d BC_e (a\|b)]` | `la + lb + 2` |
| `T` | `(2lb+3)(a\|b)₁ − 2 Σ_d (a\|b+2_d)₂ − ½ Σ_d b_d(b_d−1)(a\|b−2_d)₀` | `la + lb + 2` |
| `V` | `−Σ_C Z_C (a\|b)⁽⁰⁾_C` | `la + lb` (Boys `m ≤ la + lb`) |

`T`'s subscript `n` is a table weighted by `β^n` (`β` the ket exponent),
each with its own HRR; `V` sums the nuclei into one table first. The moments
put the operator on the ket, `(r − C)_d = (r − B)_d + BC_d` with
`BC = B − C` and `(r − B)_d |b⟩ = |b+1_d⟩`; `Q`'s six components are
`kernel_mu.build_quad_kernel`'s, xx xy xz yy yz zz, the raw second moments
into `MU[6, n_ao, n_ao]`.

**The recurrences are `hgp/`'s, not rewritten.** The VRR is `hgp.vrr`'s
eq 25 DAG at `f = 0`, spelled by `hgp.recurrence.vrr_body` unchanged, in
two limits read off its coefficients: for `V`, the ket a unit point charge
(`W → C`, so `WP = C − P`, and `ρ/p → 1`, so `inv2p_alpha_over_p = inv2p`);
for the overlap, the `m + 1` terms dropped and the DAG cut to `m = 0`. The
HRR is `hgp.hrr.reduce` (eq 28's bra half) spelled by
`hgp.recurrence.hrr_body`, which gained an additive `name=` hook (default
`hrr_name`, so no existing unit's text moved) for `T`'s three tables. The
VRR is cut to exactly the boundary the HRRs read and every recurrence
scalar is declared only where a term reads it, so no unit declares an
unread local (#92's filter; `test_kernel_hgp.py` checks it on the IR).

**Mapping.** One thread per contracted shell pair (the Schwarz kernel's),
looping its records `[rec0, rec1)` serially, because the HRR needs
contracted values. Plain stores, no atomics: each element belongs to one
shell pair, so a build is bit-reproducible and exactly symmetric. The
seed `(0|0) = s_ab K_AB` forms `K_AB = exp(ζ (P−A)·(P−B))` per record from
the MMD kernels' own record arrays; `V`'s leaf is
`s_ab K_AB (2/√π) √ζ F_m`. Staging is `OneEClass.hgp_arrays(target, basis)`:
the MMD record arrays plus `rec0 rec1 off_a off_b ab_*` (and `b_*` for the
dipole) per shell pair.

**Witnesses** (`test/test_kernel_hgp.py`, fast tier, interpreted): the MMD
host matrices on a synthetic contracted s/p basis on three off-axis
centres under four nuclei, within `1e-13` of `max|M|`, every ordered class
in one launch each; libcint's `int1e_ovlp`, `int1e_kin`, `int1e_nuc` on
H₂O/STO-3G and 6-31G and `int1e_r_cart` at both fixture origins, within
`1e-12` of `max|M|` (observed ≤ 1.0e-15); `q = −1` the exact negation; the
origin shift. Goldens: `test/golden/{cuda,hip}/{hs,ht,hv,hmu}_{00,01,11}`
(`test_emit_hgp_one_e.py`).

**On the card** (`test/test_hgp_one_e_device.py`, generator
`test/hgp_one_e_driver.py`: one thread per contracted shell pair over
`one_electron_launch`'s plan, `hgp_arrays` staged, plain stores; on the
shared runner since #658): before the port, on the
RTX 3080, 16 passed -- each of `S`, `T`, `V`, `MU` on the synthetic basis
against the MMD host (worst 5.03e-16 of `max|M|`, `T`) and on H₂O/STO-3G
and 6-31G against libcint (worst 1.01e-15, `V`), the dipole at both
origins; runs bit-identical, every matrix exactly symmetric. On the runner (2026-10-30) its 12 `gpu` cases passed on the RTX 3080
and, the first run there, on the RX 9060 XT.

**The generator.** `python -m felizjueves.emit --hgp-one-e-l [LMAX]` (1
bare, **0..1 only**, exit 2 above) adds the 12 units beside the J ladders;
never in the bare default, not in `KERNEL_GEN_CMD`.

**The moments through f.** The builder takes the dipole and the
quadrupole through (ff) (`kernel_hgp.MOMENT_L_MAX = 3`, `l_max(target)`;
`S`, `T`, `V` stay at p) -- every `hmu_` text through (pp) byte-identical.
Interpreted (`test_kernel_hgp.py`), against the MMD host's
`multipole_matrix(e = 1, 2)` on the synthetic basis with d shells added
(`SYNTHETIC_D`, fast) and d and f (`SYNTHETIC_F`, `slow`): worst 2.9e-16
of `max|M|`; against libcint, `int1e_r_cart` on `hf_ccpvtz_spd_dipole` and
`hf_ccpvtz_dipole` (worst 2.1e-15, the f basis at 0) and `int1e_rr_cart`
on `hf_ccpvtz_spd_quadrupole` (worst 1.5e-15), at each fixture's origin and
at 0; the quadrupole's origin shift, `q = −1` the exact negation. **No
libcint quadrupole witness is tracked at f**: there the quadrupole is held
to the MMD host only. The ladder is `python -m felizjueves.emit --hgp-mu-l
[LMAX]` (`hmu_`, `hquad_`; 2 bare, **0..3 only**, exit 2 above), never in
the bare default nor `KERNEL_GEN_CMD`; goldens
`test/golden/{cuda,hip}/{hmu,hquad}_{00,01,02,11,12,22}`
(`test_emit_hgp_mu.py`). Every unit through (ff) compiles warning-free
under nvcc (sm_86) and hipcc (gfx1200) (`test_emit_hgp_mu.py`, 2026-10-30,
a device-less cloud container), and the card harness
(`hgp_one_e_driver`, `test_hgp_one_e_device.py`'s
`hgp_moment_program_compiles`) builds both programs over `SYNTHETIC_F`
under both. Their `gpu` cases (`hgp_moments_through_ff_match_host_and_libcint`,
the same bounds) are written and **have run on no card**.

**Not done.** `S`, `T`, `V` at d and above (the recurrences are general in
`l`; the builder refuses `l > 1` for them, the requested scope); the moments
above f; no measurement;
no `(1, 0)` ladder unit (a basis's driver builds it, as for the MMD
ladder); no screening inside the kernel; no comparison of the two engines'
cost.

## The Rys route: `S`, `T` and `V` on engine #3

**Status: built, interpreted and emitted through (ff); NOT compiled by
nvcc or hipcc and NOT run on a card** (the session that built it had
neither compiler nor card; all 30 units passed clang 18's CUDA front end,
`-fsyntax-only -Wall -Wextra`, a proxy). `one_electron/kernel_rys.py` builds
`S`, `T` and `V` on the Rys factorisation of #468's spike
(`experimental/rys/`), whose two-electron instance is the fused F's
`eri/fock/kernel_frys.py` (`docs/implementation/rys_fock.md`). The MMD and
HGP kernels are untouched; every HGP golden is byte-identical.

**Keys.** `Target.S`, `T` and `V` with `engine=Engine.RYS`
(`kernel_rys.rys_spec`; `spec.py`, "A RYS ONE-ELECTRON CLASS"), keyed
`overlap/d0/(1,0)/b1/s.rys/literal`; `s.rys`, `t.rys`, `v.rys` are engine
slots that refuse every lever. `DIPOLE` and `SX` stay refused on Rys. Units
are `rs_{la}{lb}`, `rt_`, `rv_`, every shell `l <= 3` (`kernel_rys.L_MAX`).

**The route.** Per primitive pair, each boundary value is a product of
three 1D integrals:

| Matrix | Per primitive pair | 1D recurrence |
|---|---|---|
| `S`, `T` | `(e\|0) = u I_x(e_x) I_y(e_y) I_z(e_z)` | `I(i+1) = PA I(i) + i/(2ζ) I(i−1)` |
| `V` | `(e\|0) −= Z w0 Σ_n w_n I_x I_y I_z`, per nucleus | `I(i+1) = C00 I(i) + i B10 I(i−1)`, `C00 = PA + x_n (C − P)`, `B10 = (1 − x_n)/(2ζ)` |

`(x_n, w_n)` is the `N = (la + lb)//2 + 1` point Rys rule at `T = ζ|P − C|²`
(`ir.Rys`, `fj_rys<N>`, `rys.roots`; `Σ w x^m = F_m(T)`). `V`'s recurrence
is `rys.recurrence.root_loop`'s bra half unchanged, at the point-charge
limit of the ket (`q → ∞`: `WP = C − P`, `q/(p+q) = 1`), the attractive sign
in the weight scale `S = −Z w0`. The overlap needs no node: its 1D
recurrence is the nodeless form of the same one. `T` is the overlap's 1D
tables with the HGP route's three `β^n` tables.

**The skeleton is `kernel_hgp`'s.** The products form exactly the boundary
`(e|0)` that `kernel_hgp`'s HRR reads, so `kernel_hgp` now exposes its pair
skeleton -- `contracted_tables`, `boundary_union`, `acc_name`, `assemble`
(one thread per contracted shell pair, the record loop, the HRR on
contracted values, the combination, plain mirrored stores) -- and both
builders fill it, differing in the per-record body and two notes. An
`rs_` / `rt_` / `rv_` unit has the parameter block of the `hs_` / `ht_` /
`hv_` unit of its class (tested equal), so it takes the same staging
(`OneEClass.hgp_arrays`), launch (`kernel_hgp.launch_shape`) and output: a
drop-in.

**Witnesses** (`test/test_kernel_rys_one_e.py`, interpreted, the rule
injected as `rys.roots.roots_weights`; within `1e-13` of `max|M|` of a
reference engine, `1e-12` of libcint), observed:

| Against | `S` | `T` | `V` |
|---|---|---|---|
| MMD host, synthetic s–d (fast) | 1.8e-16 | 1.0e-15 | 4.8e-16 |
| MMD host, synthetic s–f (`slow`) | 2.0e-16 | 2.8e-15 | 5.5e-16 |
| HGP kernels, synthetic s, p | 1.1e-16 | 8.0e-16 | 1.5e-16 |
| libcint, H₂O/STO-3G | 3.3e-16 | 1.2e-16 | 1.0e-15 |
| libcint, H₂O/6-31G | 2.8e-16 | 2.4e-16 | 9.7e-16 |
| libcint, H₂O/cc-pVDZ (`slow`) | 1.8e-16 | 6.1e-16 | 3.1e-15 |
| libcint, `prim_spdf` (`slow`) | 6.2e-16 | 3.3e-15 | 7.5e-14 |

`prim_spdf`'s `V` is the MMD host's distance from libcint too (7.48e-14);
the Rys kernel is 8.5e-16 from the host there. A nucleus far enough that
every `T` takes the rule's Laguerre branch is held to the host too. Goldens:
`test/golden/{cuda,hip}/{rs,rt,rv}_{la}{lb}` for every class with
`la + lb <= 4`, (df) and (ff) pinned by sha256 in
`test/golden/rys_one_e_digests.json` (`test_emit_rys_one_e.py`, whose
nvcc / hipcc compile cases have not run).

**The generator.** `python -m felizjueves.emit --rys-one-e-l [LMAX]` (3
bare, **0..3 only**, exit 2 above) adds the units beside the J ladders;
never in the bare default, not in `KERNEL_GEN_CMD`. The `cuda` and `hip`
backends carry `fj_rys`; `emit/gpu.py` refuses a `Rys` statement, so `rv_`
has no `gpu` spelling yet.

**Not done.** No nvcc / hipcc compile and no card run (a device driver
would be `test/hgp_one_e_driver.py` with the builder swapped); the dipole
and `SX` on Rys; g and above (the rule reaches `N = 5`, (gg), but the
builder refuses `l > 3`); nothing measured, and no cost comparison with the
MMD and HGP kernels.

## Named follow-ups

- **A batched `V` kernel.** `V` has Hermite targets, so the ban that #232
  restated per target is a decision for `V` and not structural.
- **A warp-per-record `V`** with the nuclei strip-mined over `THREAD_Y`, the
  `Target.J` shape. `V`'s 3080 headroom at f was answered instead by the
  low-stack form (#586).
- **One-electron derivatives** (gradients and Hessians), excluded by this
  milestone's scope: done since, `gs_`/`gt_`/`gv_` (#600, #614;
  `gradient.md`) and `d2s_`/`d2t_`/`d2v_` (#604, #610; `hessian.md`).
- **One `fj_boys` spelling** (the #266 record): done in #301, one
  template for every unit.
- **Ordered ladder classes** (`la > lb`), so the generator, rather than the
  test-side driver, writes every class a real basis reaches.
- **The quadrupole's emit compile cases**: `test_emit_quad.py`'s (built,
  "The quadrupole"); `test_quad_device.py` ran on both cards with #658.
- **The dipole at f and above**, past `--mu-l`'s cap of 2.
- **The HGP moments on a card**: `test_hgp_one_e_device.py`'s
  `hgp_moments_through_ff_match_host_and_libcint` cases, written, never run.
- **A three-kernel dipole A/B arm** (one kernel per component against the
  fused one), opened only if the fused kernel is found to spill.
