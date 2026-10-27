# experimental/hgp/df — the DF tensors `(P|Q)` and `(mn|P)` by HGP

A standalone spike: the two density-fitting tensors, the Coulomb metric
`(P|Q)` and the three-center integral `(mn|P)`, computed by this repo's
Head-Gordon–Pople engine (`src/felizjueves/hgp/`, through
`eri/hgp_drive.py`), s and p shells only on both bases, Cartesian. Nothing
in `src/felizjueves/` imports it, nothing in `test/` collects it
(`testpaths = ["test"]`), and no tier runs it. It touches no `src/` file.
Host-side and interpreted: no kernel is emitted, compiled or run on a card.
DF-J, DF-K and the fit itself are out of scope; it stops at the two tensors.

```bash
uv sync --group oracle
uv run --group oracle python experimental/hgp/df/check.py   # ~35 s, exit 0 = PASS
```

Without the `oracle` group, table 1 is skipped (and says so); tables 2–4
need only numpy.

| File | What it is |
|---|---|
| `df.py` | `metric(aux, *, boys)` → `DF(value, scale)`, `(P\|Q)` as `[naux, naux]`; `three_center(basis, aux, *, boys)` → `DF`, `(mn\|P)` as `[nao, nao, naux]`; `metric_block`, `three_center_block` per shell; `Unit`, the exponent-0 s partner; `LMAX = 1` (a d shell on either side is a `ValueError`). |
| `check.py` | The four tables below; exits nonzero on a miss. |

## How the HGP engine computes a two- or three-center integral

A DF integral is a four-center one with a unit s partner on the aux side —
`(P|Q) = (P u|Q u)`, `(mn|P) = (mn|P u)`, `u` an s Gaussian of exponent 0
and coefficient 1 on the aux shell's centre (the device
`experimental/mmd/gradient/df.py` uses for MMD). Each block is then exactly
`eri.hgp_drive.eri_block_and_scale`: `hgp.vrr`'s VRR (`[Fock]` eqs 25–26)
per primitive quartet, summed, then `hgp.hrr`'s HRR (eq 28) once on the
contracted values, both run as IR in `ir.interp`. No recurrence is restated.

The partner is harmless in `hgp_drive.primitive_scalars`: `p = α`, `P = A`
exactly, `K_AB = 1`, and the pair factor `c (π/α)^{3/2}` is
`eri.density_fit.metric.aux_dress`'s `s` (the normalisation is already in
`c`); with `lb = 0` or `ld = 0` the HRR on that side is the identity, so the
zero exponent is never divided by. `ao.shell.Shell` refuses exponent 0, so
`Unit` is a duck-typed shell with the five attributes `hgp_drive` reads.

The sweeps are `eri.density_fit`'s: `(P|Q)` over aux shell pairs `i <= j`,
mirrored; `(mn|P)` over AO shell pairs `i <= j` × every aux shell, mirrored in
`(m, n)`, layout `[m][n][P]`. The scale is `hgp_drive`'s absolute-value twin,
placed alike.

## The numbers (`check.py`, this box, 33 s)

Molecules: `experimental/mmd/gradient`'s distorted water
(`O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5`) in STO-3G (7 AOs) and
6-31G (13 AOs), and HCN in STO-3G (11 AOs); aux `cc-pvdz-jkfit` cut to its s
and p shells (57 functions on water, 75 on HCN). Gates relative to `max|X|`,
stated before the numbers.

| Table | What | Worst | Gate |
|---|---|---|---|
| 1. libcint, live | `int2c2e_cart`, `int3c2e_cart` (`aosym="s1"`) through PySCF | `(P\|Q)` 1.52e-14, `(mn\|P)` 2.50e-14 (HCN) | 1e-12 |
| 2. libcint, tracked | `h2o_ccpvtz_jkfit_metric` and `h2o_sto3g_jkfit_3c`, aux cut to s and p (57 of 166), no PySCF | `(P\|Q)` 1.38e-14, `(mn\|P)` 2.54e-14 | 1e-12 |
| 3. MMD, same shells | `eri.density_fit.metric.metric_matrix`, `eri3.eri3_tensor` on `reference.pq` | `(P\|Q)` 1.73e-16, `(mn\|P)` 2.38e-16 | 1e-13 |
| 4. structure | every element under its twin; both tensors exactly symmetric; `(ba\|P)` against `(ab\|P)ᵀ` (the sweep evaluates `a <= b` only, so the other order is exercised here); a d shell refused on each side | twin, symmetry: all exact; swap 3.96e-16; 3 of 3 refused | 1e-14 |

The two engines agree to a few ulps of `max|X|` on the same shells, and both
sit ~1e-14 from libcint, the distance `test_ao_metric.py` and
`test_ao_eri3.py` already record for MMD — so the residual against libcint is
libcint's own summation, not the recurrence.

## What is not done

- Nothing above p, on either basis; `LMAX` refuses it rather than leaving it
  untested. The HGP DAGs are built for any class, so lifting it is a matter
  of the checks, not the code.
- No kernel: the VRR and HRR run in the interpreter, one VRR launch per
  primitive quartet. Nothing is emitted, compiled, run on a card or timed.
- No screen, no DF-J / DF-K, no fit, no gradient.
