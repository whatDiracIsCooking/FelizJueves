# The DF integrals `(P|Q)` and `(mn|P)` on the Head-Gordon–Pople recurrence

**Status: implemented for s and p shells, interpreted and emitted;
compiled and run on the RTX 3080 only.** This is the derived note for the two
density-fitting targets of milestones 15 and 16, the metric `(P|Q)`
(`metric.md`) and the three-center `(mn|P)` (`three_center.md`), built a
second way: on the HGP VRR (`[Fock]` eqs 25–26) and HRR (eq 28) of
`hgp_fock.md` instead of the McMurchie–Davidson Hermite tower. It is the one
place that says how they are built, what they reuse, how they were checked
and what is not done.

- **Generated:** `eri/density_fit/kernel_hgp.py`:
  `build_hmetric_kernel((lP, lQ))` → `hmetric_{lP}{lQ}` and
  `build_heri3_kernel((la, lb, lP))` → `heri3_{la}{lb}{lP}`, keyed
  `Spec(..., Target.METRIC | Target.ERI3, engine=Engine.HGP)`, so the key's
  slot is `metric.hgp` / `eri3.hgp` (`Spec.target_slot`; every stored key
  unchanged). `spec.py` now admits HGP on these two targets beside `J` and
  `K`; `levers.ENGINE_SLOTS` gained the two slots, each refusing every
  kernel-body lever.
- **Emitted by** `python -m felizjueves.emit --hgp-metric-l [LMAX]` and
  `--hgp-eri3-l [LMAX]`. Here LMAX bounds each **shell**, and only
  `0..HGP_DF_L_MAX = 1` is accepted (bare: 1); above is exit 2. Both add
  units beside the J ladders. Neither is in the bare default or in
  `KERNEL_GEN_CMD`.
- **The metric's skeleton is a seam:** `kernel_hgp.metric_kernel` (thread
  mapping, accumulators, record loops, scatter, parameters) is what
  `build_hmetric_kernel` fills with the HGP VRR and the Rys metric
  (`kernel_rmetric`, `rys_fock.md`, "The DF metric on Rys") fills with the
  Rys rule; every `hmetric_` golden is byte-identical to before the seam.
- **Correctness is tested for s and p shells only:** (s|s), (s|p), (p|p) and
  the six `(mn|P)` classes `la ≤ lb ≤ 1`, `lP ≤ 1`. The builders accept any
  class the DAGs accept.

**Nothing here is measured or scored.** There is no published board (as for
the MMD kernels: `limits.json`'s #417 and #447 records) and no
`data/measured/` row.

## A sibling, not a lever

Each HGP kernel is a drop-in for its MMD twin (`kernel_metric`,
`kernel_eri3`). It has the same parameters in the same order (restated, not
imported, and held equal by `test_kernel_hgp_df.py`) and reads the same
staging (`MetricClass.arrays()` / `Eri3Class.arrays()`, `s` as `s_aux`). It
uses the same thread mapping: one thread per aux shell pair or per cell, its
two record ranges walked serially, bra outer. It writes the same plain
mirrored stores into the same output, with no atomics, so runs are
bit-reproducible. The interpreted adapters take `engine=`
(`metric_drive.metric_matrix_kernel`, `eri3_drive.eri3_tensor_kernel`), and
so do the on-card harnesses (`test/metric_driver.py`,
`test/eri3_driver.py`, `build_on_card(..., engine=)`, both on the shared
runner since #654 / #657). Nothing else changes.

## A DF class is an HGP quartet with exponent-0 partners

An aux Gaussian is the primitive pair `(P, 1)` whose partner is an s function
of exponent 0 on the same centre. That gives `p = α`, `P = A` and `K = 1`,
and the pair factor `c (π/α)^{3/2}` is exactly the aux record's `s`
(`metric.aux_dress`). So:

| target | HGP class built | recurrence |
|---|---|---|
| `(P\|Q)` | `(lP 0 \| lQ 0)` | the VRR alone: `b = d = 0`, no HRR |
| `(mn\|P)` | `(max(la,lb) min(la,lb) \| lP 0)` | the VRR, then the bra HRR on contracted values |

On a single-centre side `QC = Q − C = 0`. For the metric's bra,
`PA = P − A = 0` too. Every VRR term with one of those factors is exactly
zero, so it is **not spelled**. One backward liveness pass then drops every
node, Boys order and quartet scalar that only such terms read. That pass is
needed anyway: a declared but unread local fails #92's filter. The DAG is
`hgp.vrr.build`'s, unchanged.

**The bra orientation.** `(mn|P) = (nm|P)`, so the class is built with the
higher `l` first. The HRR then moves units onto the lower shell, which is
nothing when that shell is s, so through p only `(pp|P)` runs an HRR
transfer. The staging keeps `la ≤ lb`. A swapped build therefore reads the
record's `XPB` as its `PA`, takes `AB = B − A`, and stores `(a, b)` from the
built `(b a|q)`.

## The quartet scalars

The scalars are `eri/hgp_drive.py`'s binding with `q = α_k` and
`Q = C = A_k`:

- `PQ = P − A_k`, `ρ = pq/(p+q)`, `θ = √ρ`
- `WP = −q/(p+q) PQ`, `WQ = +p/(p+q) PQ` (the derived sign, #291)
- `1/2p`, `1/2q`, `1/2(p+q)`
- `ρ/(2p²)`, `ρ/(2q²)`
- `pair = U_P U_Q`

`U_Q` is the aux record's `s_aux`. For the metric, `U_P` is the bra aux
record's `s_aux`. For `(mn|P)`, `U_P` is `s_ab K_AB`. The MMD staging keeps
`K_AB` out of `s_ab`, because there it rides in `E`'s seed, so the HGP kernel
forms it per record from the record's own `X_PA`, `X_PB`:
`K_AB = exp(p X_PA·X_PB)`. The fundamental is
`hgp.recurrence.fundamental_prelude`'s, `pair (2/√π) θ F_m(T)`, through the
one device `fj_boys`.

The contraction completes in registers (`c{e}__{f}`, one accumulator per
VRR boundary value) **before** the HRR reads it, because eq 28 holds only on
contracted values (milestone 12, pin 5). `AB` is read off the cell's first
bra record.

## How it was checked

All checks below are interpreted, in the fast tier, in
`test/test_kernel_hgp_df.py`:

- **Against the hosts** (`eri.density_fit.metric` / `eri3`, MMD on
  `reference.pq`). The bound is 16 eps of the MMD tests' tower-twin
  absolute-value scale, on the MMD tests' synthetic bases cut to s and p
  (diagonal, off-diagonal and swapped pairs; one-centre blocks). Measured
  worst at writing: 2.89 eps (`(p|p)`) and 3.20 eps (`(pp|p)`).
- **Against the MMD kernels**, interpreted on the same staging: within
  32 eps of the same scale.
- **Tight exponents** past `T = 709`, from the MMD tests' `TIGHT` sets.
- **Against libcint**: cc-pVTZ-jkfit's s and p aux shells against
  `int2c2e_cart`, 6.3e-15 of `max|M|`; H₂O/STO-3G against those shells
  against `int3c2e_cart`, 2.1e-14 of `max|(mn|P)|`. The bound is 1e-12, as
  in the MMD tests.
- **Structure**: the metric reads no `PA`, `QC` or `AB`. `(mn|P)` reads no
  `QC`, reads `PA` exactly when the built bra has a p shell, and reads `AB`
  exactly for `(pp|P)`. Every store is a plain store.

Nine goldens per backend are in `test/golden/{cuda,hip}/{hmetric,heri3}_*`
(`test/test_emit_hgp_df.py`, which also holds the CLI's refusals).

## Where it ran

**On the RTX 3080 only.** The `cuda` compile cases (`test_emit_hgp_df.py`,
`test_hgp_df_device.py`) and the `gpu` device tests (`test_hgp_df_device.py`,
8 passed) ran: against the host, worst 3.08 / 6.06 eps of the tower twin for
`(P|Q)` (synthetic / fixture) and 2.98 / 4.90 for `(mn|P)`; against libcint
6.24e-15 of `max|M|` (`int2c2e_cart`) and 2.05e-14 of `max|T|`
(`int3c2e_cart`); runs bit-identical. The RX 9060 XT has not run it, and
no open issue tracks that run: #431 closed on 2026-10-17 when the card's
detection was fixed, not when this was run.

## What is not done

- **The RX 9060 XT** compile and card run (above; no open issue).
- **d and above.** The builders accept them, but nothing is tested there and
  the CLI refuses them.
- **Measurement**, and any comparison against the MMD kernels' clock or
  spill.
- **A screen, a lever, batching, spherical functions, and DF-J/K
  themselves**, all out of scope as for the MMD kernels.
