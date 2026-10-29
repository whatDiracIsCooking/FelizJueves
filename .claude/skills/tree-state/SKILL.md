---
name: tree-state
description: >-
  The detailed state of every kernel family in this repo: what is generated,
  what each one is held to (host reference, libcint, interpreter), which
  card it ran on, what is measured, and what was decided or left open. It is
  the long form of `.claude/CLAUDE.md`'s "The state of the tree". Read the
  matching reference file before you claim, change, test or measure anything
  in a family (J chain, S/T/V, K and its block kernels, HGP F and split J/K,
  Schwarz, cross overlap, DF metric and (mn|P), dipole, gradients, Hessians,
  Rys, the device Boys), or before you write a doc sentence about one.
---

# The state of the tree, per family

`.claude/CLAUDE.md` carries the one-table summary. These files carry the
detail, moved verbatim from it, so the issue and PR numbers, the figures
and the "ran on" / "not run" wording are the record. The design of each
family is in `docs/implementation/*.md`, and each of those opens with a
status block.

**Keep them true.** When a PR changes what a family reaches (a new class, a
card run, a measurement, a default), update its file here in the same PR.
Update the summary row in `.claude/CLAUDE.md` too if the change moves it.
`test/test_claims.py` holds every tally, `StructuralLimit` count and plan
wording in these files to the data, as it does for `CLAUDE.md`.

**`test/registry/` is these files as data** (#648): one module per kernel
file below, declaring each family's cases as `(engine, quantity, deriv, L,
cards, tiers)` and how each device test's cases map onto them.
`test/test_registry.py` holds it to the collected `gpu` and compile-only
cases in both directions. When a PR adds a device test, a class or a card
run, change the family's module in the same PR; the test names what is
missing.

## Kernel families

| File | Covers |
|---|---|
| [j-chain.md](references/j-chain.md) | The `J` chain (`[p\|q]`, `DQ`, `JAB`, the `E` fragment), the host path and libcint witness, the launch plan, `PAIR_BOUND` / `RECORD_BOUND`, dispatch, the one-electron `S`, `T`, `V` (milestone 11), and the low-stack `V`, `DQ`, `JAB` at (ff) (#586). |
| [k-exchange.md](references/k-exchange.md) | Exchange `K` (milestone 10, open): `kernel_k`, its launch plan and the symmetric sweep, then the block kernels `kb` (#481, #484), the split `kbs` (#548), the rolled `kbr` (#571, #574), the dense `kbd` (#575), the transposed `kbrt` (#576) and the column `kbrc` (#583), with their measurements and the per-class recommendation. |
| [fock-hgp.md](references/fock-hgp.md) | The HGP engine and the fused `F` (milestone 12), the split HGP `J`/`K` (milestone 14), and the scheme A/B of the 2x2's diagonal cells. |
| [schwarz-overlap-df.md](references/schwarz-overlap-df.md) | The Schwarz matrix, the cross-basis overlap `SX`, the DF metric `(P\|Q)` (milestone 15), the three-center `(mn\|P)` (milestone 16), their low-stack forms (#606) and their HGP twins. |
| [one-electron-extra.md](references/one-electron-extra.md) | The dipole `MU` (milestone 19), the quadrupole, and `S`, `T`, `V`, dipole on the HGP and Rys engines. |
| [gradients.md](references/gradients.md) | `derivative_order = 1`: `gs`, `gt`, `gv`, `gjab`, `gk`, the DF-J gradient terms, the host-staged DF-K weights and DF two-electron gradient, and their low-stack forms. |
| [hessians.md](references/hessians.md) | `derivative_order = 2`: `hjab` / `hdq`, the one-electron `d2s` / `d2t` / `d2v`, `hkx`, `d2metric`, `d2eri3`. |
| [rys-and-boys.md](references/rys-and-boys.md) | The fused `F` and split `J` and `K` on Rys quadrature, and the device Boys `fj_boys<M>` (#301) with its stale-pin consequence. |

## The tree, directory by directory

These are the long cells of `CLAUDE.md`'s "State" table.

| File | Covers |
|---|---|
| [test-dir.md](references/test-dir.md) | `test/`: every test file per family, its tier and markers, and what has run on which card. |
| [measure-dir.md](references/measure-dir.md) | `measure/`: the sweeps `devtools/measure.sh` runs. |
| [src-dir.md](references/src-dir.md) | `src/felizjueves/`: the package layout, `Spec`, the IR, the passes, the layering rules, the measured layer and the levers. |
| [plan-dir.md](references/plan-dir.md) | `src/felizjueves/plan/`: MILD, the sieve, cost, recovery, search, the optimum. |
| [data-dir.md](references/data-dir.md) | `data/`: the transcribed tables, the scoreboards, claims and limits, every `data/measured/` store, and the libcint fixtures. |
| [generator.md](references/generator.md) | The generator: `KERNEL_GEN_CMD`, the ladders, the `gpu` backend, the class selector, the launch-plan driver, and the last device sessions. |
| [docs-dir.md](references/docs-dir.md) | `docs/`: what each implementation note specifies. |
