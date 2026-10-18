---
name: test
description: >-
  Run this repo's suites the right way and interpret the result honestly — the
  fast pytest tier (compiles nothing, launches nothing), the kernel tier
  (devtools/kernel-tier.sh: generate → nvcc/hipcc → run on a card → compare
  against the numpy reference), and the marker/fixture split that decides which
  of them a given test needs. ALWAYS audit what skipped, or was never there,
  before calling a run green.
  Use when the user asks to run tests, check a change, or verify an edit.
---

# Running tests, and trusting the result

Three rules, and the third is the one this project turns on:

1. **A green run is meaningless until you read the skip list.** A test that
   needs a tool or a card the environment lacks skips *silently*, so "all
   passed" can mean "the tests that would have caught this never ran." Always
   pass `-rs`.
2. **Say which tier you ran, and on which backend.** "Tests pass" is not a
   result here; "the fast tier passes, and the kernel tier passes on CUDA"
   is.
3. **No numerical claim is proved without a card.** A device compiler targets
   an *architecture*, so every compile is happy on a device-less box. Only the
   kernel tier, on this machine, can say a kernel computes the right integrals.

## The two tiers

### Fast tier — `pytest`

```bash
FAST='not cuda and not hip and not gpu and not slow and not oracle'
pytest -rs -m "$FAST"           # the fast tier, as the push gate runs it
pytest -rs -m "$FAST" -n 8      # host: 8 workers (JOBS in devtools/config.sh)
pytest -rs -m "$FAST" -k toolchain   # one file's worth
```

Touches no device compiler and launches no kernel, **because that marker
expression (`FAST_TEST_ARGS` in `devtools/config.sh`) excludes the `cuda`,
`hip` and `gpu` markers**. A **bare `pytest` applies no marker filter**
(`addopts` is only `-ra`): on this host, where nvcc is native, it compiles
and, with a card present, launches. That exclusion is what makes it honest as a push gate: inside a
container both compilers are present, and without it the "seconds-fast" tier
would start invoking nvcc.

`devtools/prepush-tests.sh` is this tier as the push gate sees it — run it to
see exactly what a `git push` will run. The hook fires only when the push
touches a `.py`.

### Kernel tier — `devtools/kernel-tier.sh`

```bash
devtools/kernel-tier.sh --backend cuda          # nvcc + the RTX 3080 (PR gate)
devtools/kernel-tier.sh --backend hip           # hipcc + the RX 9060 XT (PR gate)
devtools/kernel-tier.sh --backend cuda --full   # every gpu test, slow included
devtools/kernel-tier.sh --backend cuda --fresh  # wipe .kernel-tier/ first
devtools/kernel-tier.sh --backend cuda -- -k saxpy    # extra pytest args
```

Four phases, and **the report at the end says which of them did anything**:

| Phase | What it proves |
|---|---|
| preflight | the compiler is here; whether a card is |
| generate | `KERNEL_GEN_CMD` (`python -m felizjueves.emit --one-e-l 2`) writes 55 units into `.kernel-tier/src-<backend>/`: the J chain's 37 (the `[p\|q]` kernels, L = 0..16, and the `dq_`/`jab_` pair ladder through (ff) — all the bare command writes) and the one-electron ladder's 18 (`s_`, `t_`, `v_` for `la ≤ lb ≤ 2`, #242) — **dormant** if that variable is set empty. To cap it, set it yourself, e.g. `KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit --L 0 1 2 3 4 --pair-l 3 --one-e-l 2"` |
| compile | every emitted translation unit survives the real device compiler. The CUDA side takes `*.cu`; the **HIP side takes `*.hip` AND `*.cu`** (with `-x hip`), so one generated source can be checked against both compilers — which is the point |
| device | the `gpu`-marked suite **for that backend**, `slow` excluded (`-m "gpu and <backend> and not slow"`, the PR gate: L ≤ 4 on H₂ / H₂O/STO-3G, 3m44s cold on CUDA; `--full` drops the `not slow`, 72 min) agrees with the host reference — among others `test_kernel_device.py` (the generated kernels' `J_p` and both `fj_boys` spellings), the J chain and launch-plan builds, `test_one_electron_device.py` and `test_hcore_device.py` (`S`, `T`, `V` and `h`, milestone 11), plus `test_toolchain.py`'s saxpy |

**The cache covers the tests' compiles too (#186).** `test/devcache.py` puts
the same sccache in front of every compile a test issues
(`compile_device_source`, `plan_driver.compile_checked`), from a
content-addressed directory under `.kernel-tier/test-cache/`, so a warm re-run
recompiles only what changed. The compiler always runs (a hit replays its
diagnostics, so #92's warning filter still bites). `--no-cache` exports
`KERNEL_NO_CACHE=1` and the tests compile cold; the same variable works with
bare pytest. `--fresh` clears the test build directory, never the sccache
store. The result block prints the cache's hits and misses for the compile and
device phases, and the device phase's wall-clock.

**Read `generated nothing (KERNEL_GEN_CMD empty)` in the report as "this was not
a kernel tier run".** What such a run still proves is that this box compiles
and runs device code. A generator that runs and writes nothing fails the tier
(`GENERATE PRODUCED NOTHING`) rather than passing it empty.

A failed compile stops the tier before the device suite and says
`COMPILE FAILED` with the source and object counts — a pass from a stale object
would be a lie. **Exit 2 is "could not run", never "passed"** — no compiler, no image, or an
unresolvable device group. On a host with no `hipcc`, `--backend hip` re-execs
itself inside `fj:combined` with the device nodes and host gids passed in; with
no such image it exits 2 and names `docker/build.sh combined`.

### Measurement is not a tier — `devtools/measure.sh`

No test times, profiles or reads back a stored measurement. The timed sweeps,
the profiler witnesses, the recompile-against-`kernels.json` check and the
tracked reports recomputed live in `measure/` (`measure_*.py`, outside
`testpaths`) and run only through `devtools/measure.sh <sweep>`
(`measure.sh list`). `measure.sh check` (~20 s, no compiler, no card) is the
one to run after a re-measure, or when a kernel-text change has moved a
stored `source_sha256`. None of it blocks a PR.

## The markers, and why a test skips

The two axes are independent, and conflating them is how a green-and-empty run
gets reported as a pass:

| Marker | Needs | Fixture that enforces it |
|---|---|---|
| `cuda` | nvcc — **no card required** | `nvcc` |
| `hip` | hipcc — **no card required** | `hipcc` |
| `gpu` | a live device | `cuda_device` / `hip_device` |
| `slow` | nothing; costs minutes (an exhaustive search re-run, a class above `FAST_L_MAX`, or a large input: 6-31G, cc-pVDZ, cc-pVTZ); off the pre-push gate via `FAST_TEST_ARGS` and off the kernel tier unless `--full` | — |

Those are the registered markers (with `oracle` and `claim(id)`). An issue that
names a `kernel-tier` or `high-L` marker means `slow`: neither is registered.

A test takes exactly the fixtures it needs, so the skip message names the real
reason (`hipcc not found -- it lives in the fj:hip / fj:combined image`, versus
`no AMD device on this machine`). When you add a test, pick the fixtures the
same way: asking for `cuda_device` in a compile-only test makes it skip on a
machine that could have run it.

**Read the AMD skip to the end — it says which of two things is wrong** (#431).
`/dev/kfd absent` is a container passthrough question. `/dev/kfd is here but
amdkfd lists no GPU node` is a **host** question: `amdgpu` bound no card, so
nothing in the container can fix it and there is no point reading
`devcontainer.json`. Usual cause is a kernel upgrade that `amdgpu`'s DKMS
module was not rebuilt for, which leaves CUDA working and only HIP dark —
`dkms status` and `devtools/doctor.sh` name it and print the fix. The tier is
blunter still: `device: ABSENT` and exit **2**, never a pass.

`test/conftest.py` holds `compile_device_source` and `run_device_binary` — the
compile-and-run machinery every device test shares. Put kernel text in the
test; put plumbing there.

## What a run on each machine means

| Where | Fast tier | Kernel tier |
|---|---|---|
| this host | passes; `cuda`/`hip`/`gpu` excluded by marker | **CUDA works natively** (the 13.0 toolkit is installed); `--backend hip` needs the container |
| `fj:cuda` container | same | CUDA half, fully |
| `fj:hip` container | same | HIP half, fully |
| `fj:combined` container | same | **both** — the one place a generated kernel is checked against both compilers |
| a hosted CI runner | the only tier that can run | never: no card |

## Before calling anything green

- `-rs` was passed, and you read the skips.
- You named the tier and the backend.
- If the kernel tier ran: the report said `device present`, not `device ABSENT`.
- If you touched generation: both backends, or say explicitly which one you did
  not run. `hipcc` is clang and `nvcc` is not — a kernel that only one accepts
  is this project's most likely defect.
- `ruff check .` is clean (the commit hook fails on a finding in a file you
  touched).

Run `devtools/doctor.sh` when a result is surprising — it reports which half of
the workflow you are on, and the **doctor** skill turns each finding into its
fix.
