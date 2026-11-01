# FelizJueves

Python code generation of GPU kernels for electronic-structure tensors:
electron repulsion integrals and the matrices built from them (`J`, `K`,
`F`, density fitting), the one-electron matrices, and their first and
second nuclear derivatives. The kernels are emitted as **CUDA and HIP from
one source**, and every one has to satisfy both device compilers.

**Why a generator.** There is one kernel per angular-momentum class, and
its recurrence tree, its register and shared-memory budget and its
sub-batch split are all functions of that class. So the reviewable
artefact is the program that emits the kernels, not the hundreds of
kernels: the generator is the product, and a kernel is its output.

**Three recurrences, one IR.** Every kernel is built on one of three
integral engines (`spec.Engine`) and lowered through one IR
(`src/felizjueves/ir/`) to text that `nvcc` and `hipcc` both accept:

- **MMD**, McMurchie–Davidson in the Hermite basis, with the sub-batching
  of Palethorpe & Barca (`[JEngine]`). The workhorse: every family, and
  the only engine with Hessians.
- **HGP**, Head-Gordon–Pople (vertical then horizontal recurrence), the
  engine of the fused Fock kernel of `[Fock]` §3.
- **Rys**, Rys quadrature with a 2D vertical recurrence feeding the same
  HGP horizontal step, so a Rys unit is a drop-in for its HGP twin.

`docs/references.md` is the citation of record (cite by number; `[Fock]` is
entry 23, `[JEngine]` entry 35). Each family's design is in
`docs/implementation/*.md`, each opening with a status block.

## What it generates, and how far each one has got

"Generated" does not mean "run on a card", and "run on a card" does not
mean "measured". Each cell names the furthest tier its kernels have
reached, and through which class:

- **both cards**: compiled, launched on the RTX 3080 (`sm_86`) **and** the
  RX 9060 XT (`gfx1200`), and matched to the host reference (and to libcint
  or PySCF where a witness exists);
- **RTX 3080** / **RX 9060 XT**: the same, on that card only;
- **compiled**: the emitted text compiled warning-free by `nvcc` and
  `hipcc`, on no card (proves no number);
- **interpreted**: the IR run on the host against the reference, and
  emitted, but not put through a vendor compiler;
- **—**: not built.

Classes are named by their shells: `(ff)` is a pair, `(dd|dd)` a quartet,
`(ff|i)` a three-center `(mn|P)`, `(i|i)` a metric pair. A bare `L` is the
class's total angular momentum.

### Energy (`derivative_order = 0`)

| Tensor | MMD | HGP | Rys |
|---|---|---|---|
| Coulomb `J` (`[p\|q]`, pre-contraction `DQ`, scatter `JAB`) | **both cards**: `[p\|q]` to L = 16, `DQ` / `JAB` through (ff); whole-molecule `J` against libcint | split `J` (`hj_`): **RTX 3080**, L ≤ 4 | split `J` (`rj_`): interpreted, through (dd\|dd) |
| Exchange `K` (milestone 10, **open**) | `kernel_k`: **both cards** at L ≤ 2, (dd\|dd) on the RTX 3080; block forms `kb` / `kbr` / `kbrc`: **RTX 3080**, through (gg\|gg) | split `K` (`hk_`): **RTX 3080**, L ≤ 4 | split `K` (`rk_`): interpreted, through (dd\|dd) |
| Fused Fock `G = J − K/2` | — | `f_`: **both cards**, through (dd\|dd) | `fr_`: interpreted, through (dd\|dd) |
| Schwarz `Q = sqrt((mn\|mn))` | **both cards**, through (pp) / (sd) | — | — |
| DF metric `(P\|Q)` | **both cards**, through (i\|i) | s and p: **RTX 3080** | compiled, through (g\|h) |
| Three-center `(mn\|P)` | **both cards**, through (ff\|i) | s and p: **RTX 3080** | compiled, `la + lb + lP ≤ 9` |
| Overlap `S`, kinetic `T`, nuclear `V` | **both cards**, through (ff) | s and p: **both cards** | interpreted, through (ff) |
| Cross-basis overlap `S_{B1,B2}` | **both cards** | — | — |
| Dipole `MU` | **both cards**, through (ff) | **both cards** at s and p; compiled through (ff) | — |
| Quadrupole (raw second moments) | **both cards**, through (dd) | compiled, through (ff) | — |

### First derivatives (`derivative_order = 1`)

| Tensor | MMD | HGP | Rys |
|---|---|---|---|
| `S`, `T`, `V` (`gs`, `gt`, `gv`) | **both cards**, through (ff) | `hgs`, `hgt_`, `ghv_`: compiled, through (ff) | — |
| Coulomb (`DQ` → raised `[p\|q]` → `gjab`) | **both cards**, through (ff) | `ghj_`: interpreted, not emitted | — |
| Exchange (`gk`) | **both cards** | `ghk`: interpreted through (dd\|dd), compiled through (dd\|pp) | — |
| Fused Fock (`gf_`) | — | interpreted through (dd\|dd), held to PySCF; compiled through (pp\|pp) | — |
| DF metric | **both cards**; uncontracted `ipmetric_`: **both cards**, through (i\|i) | `hgmetric`: compiled | — |
| `(mn\|P)` | **both cards**; uncontracted `xeri3_`: compiled, through (ff\|i) | `hgeri3`: compiled | — |
| Dipole, quadrupole (`gmu_`, `gquad_`) | compiled, through (ff) | — | — |
| Derivative matrices `S^x`, `T^x`, `V^x` (`ips_`, `ipt_`, `ipv_`) | compiled, through (ff) | — | — |
| CPHF Coulomb derivative `J^X` (`jx`) | compiled, through (ff) | — | — |
| `F^x`'s exchange part (`kx1`, `kxc`) | **RX 9060 XT**, through (ff\|ff) | — | — |
| The whole RHF gradient | assembled and held to PySCF; **RX 9060 XT**, s and p bases | — | — |

### Second derivatives (`derivative_order = 2`)

| Tensor | MMD | HGP | Rys |
|---|---|---|---|
| Coulomb (`hdq` → `[p\|q]` to L = 14 → `hjab`) | **both cards**, through (ff); H₂O/cc-pVDZ against libcint | — | — |
| Exchange (`hkx`) | **both cards**, through (ff\|ff); H₂O/cc-pVDZ against libcint | — | — |
| `S`, `T`, `V` (`d2s`, `d2t`, `d2v`) | **both cards**, through (dd) | — | — |
| DF metric, `(mn\|P)` (`d2metric`, `d2eri3`) | **both cards**, through (i\|i) / (ff\|i) | — | — |

Underneath them all is the device Boys function, one template
`fj_boys<M>` shared by every unit with a fundamental: witnessed on the RTX
3080 over `T ∈ [0, 1e6]`, `M ≤ 16`, and compiled only on `gfx1200`.

**What is measured, and what is scored.** Little of the above is timed:
the `[p|q]` kernels and the whole `J` build (both cards), the fused HGP `F`
(all 21 classes, both cards) and the `K` block forms (RTX 3080), all in
`data/measured/`. Only `J` has a published board, the plan layer's replay
of `[JEngine]` Tables 1–3. Against the predecessor `jgen`, this is parity
on its energy-order `J` surface, with one difference: Table 4, declined.
Which plan ships is a decision, not a gap: the opt-in `tuned` lever builds
the searched plan, but the **literal** plan ships, because the searched one
was measured on both cards and refused on wall-clock (#166).

The detail behind every cell (issue numbers, worst errors, which test ran
on which card, what was measured and decided) is in the
[tree-state](.claude/skills/tree-state/SKILL.md) skill, one file per
family, and as data in the case registry, `test/registry/`.

## Generating kernels

```bash
uv run python -m felizjueves.emit --backend cuda --out .kernel-tier/   # the bare default: the J chain
uv run python -m felizjueves.emit --backend hip --fock-l 2             # the HGP fused F, 21 classes
uv run python -m felizjueves.emit --fock-l 2 --rys                     # the same on Rys
```

The bare command writes the `J` chain. Every other family has its own
opt-in ladder, never in the bare default (`--one-e-l`, `--mu-l`,
`--quad-l`, `--schwarz-l`, `--sx-l`, `--metric-l`, `--eri3-l`, the
`--hgp-*-l` and `--rys-*-l` twins, `--grad-*-l`, `--hess-*-l`, … ;
`--help` lists them), `--rys` moves `--fock-l`, `--hgp-j-l`, `--hgp-k-l`
and `--hgp-class` onto Rys, and `--class LA LB LC LD` writes one class's
whole `J` chain. `KERNEL_GEN_CMD` in `devtools/config.sh` is what the gate
generates. Scratch output goes to `.kernel-tier/`; generated text a test
asserts against is a tracked golden file.

## Setup

```bash
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint
pre-commit install --hook-type pre-push   # the fast test gate
devtools/doctor.sh                        # what is degraded, and the fix for each
```

`uv` is the only assumed host tool, and **`uv.lock` is the only place Python
dependency versions live**: after editing `pyproject.toml`, run `uv lock` and
commit the result; never `uv pip install`. PySCF is a separate `oracle`
group that a bare `uv sync` does not install.

The device toolchains do not have to be on your host; they are in the images:

```bash
docker/build.sh combined                  # base -> cuda -> combined: nvcc and hipcc in one image
devtools/devcontainer.sh shell            # interactive, in the CUDA container
devtools/devcontainer.sh --hip shell      # the ROCm one
```

`docker/README.md` has the four Dockerfiles (base → cuda / hip → combined),
the build args and the AMD passthrough.

## The tiers, and what each one proves

| Tier | Command | What a green run means |
|---|---|---|
| **gate** | `devtools/gate.sh` | **the one check that blocks a merge**: lint, fast, a sampled compile phase under nvcc **and** hipcc, and a sampled gpu phase on each usable card. Needs both compilers in one shell (the combined container). With no card it passes and names the backends left unverified on card. |
| lint | `ruff check .` | narrow by design: `E,F,I,UP,B`, no formatter. |
| fast | `pytest -m "not cuda and not hip and not gpu and not slow and not oracle"` | the Python side agrees with itself, and with libcint through tracked fixtures, for classes with L ≤ 4. Compiles nothing, launches nothing, needs no PySCF. This is the push hook. |
| compile | `pytest -m "(cuda or hip) and not gpu and not slow"` | emitted text survives nvcc and hipcc. Needs no card; proves no number. |
| kernel | `devtools/kernel-tier.sh --backend cuda\|hip [--full]` | generate → compile (a warning in generated text fails it) → run on the card → compare against the host reference and libcint. `--full` goes to L ≤ 8. |
| measure | `devtools/measure.sh <sweep>` | not a test tier: wall-clock, counters and the stored `data/measured/` reports. Never a gate. |
| oracle | `uv sync --group oracle`, then `pytest -m oracle` | the tracked fixtures still equal what PySCF computes today. |

**The kernel tier is the only place a numerical claim can be made.** A
`gpu`-marked test *skips* on a box without a usable card, so a green run
there is green and empty; `kernel-tier.sh` exits **2** when it could not
run, and a 2 is never a pass. `devtools/doctor.sh` makes this visible. The
`gpu` fixtures lock the card themselves, so two runs on one card serialise.

The reference machine has both halves: an **RTX 3080** (sm_86, CUDA 13.0)
and an **RX 9060 XT** (gfx1200, RDNA4, ROCm 7.2.4). `hipcc` is clang and
`nvcc` is not, and that disagreement is this project's main portability
risk.

## Layout

```
src/felizjueves/  spec.py (the key every subsystem joins on), engine.py, levers.py, and
                  mmd/ hgp/ rys/      the three recurrences
                  ir/ emit/           the IR, its passes and interpreter; the CUDA and HIP emitters
                  eri/                j_coulomb/ k_exchange/ fock/ hjk/ density_fit/, Schwarz
                  one_electron/       S, T, V, SX, the multipoles
                  gradient/ hessian/  derivative kernels and their host references
                  plan/ performance/ literature/ reference/ ao/
test/             pytest; markers split "needs a compiler" / "needs a card" / "needs PySCF";
                  registry/ (the case registry), golden/, oracle/ (libcint fixtures)
measure/          the sweeps behind devtools/measure.sh
data/             transcribed [JEngine] / [Fock] tables, claims and limits, data/measured/, basis sets
docs/             references.md, and implementation/ (one design doc per family)
devtools/         gate, kernel tier, doctor, devcontainer, worktree, measure, version pins
docker/           four Dockerfiles, build.sh, compose
.devcontainer/    one variant per GPU image, driven by devtools/devcontainer.sh
```

Depth lives where it is used: each script's header explains itself
(`--help` prints it), `docker/README.md` covers the images, and
`.claude/CLAUDE.md` is the map for working in here.
