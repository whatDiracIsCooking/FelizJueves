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
pytest -rs                      # everything collectable here
pytest -rs -n 8                 # host: 8 workers (JOBS in devtools/config.sh)
pytest -rs -k toolchain         # one file's worth
```

Touches no device compiler and launches no kernel, **because
`FAST_TEST_ARGS` in `devtools/config.sh` excludes the `cuda`, `hip` and `gpu`
markers**. That exclusion is what makes it honest as a push gate: inside a
container both compilers are present, and without it the "seconds-fast" tier
would start invoking nvcc.

`devtools/prepush-tests.sh` is this tier as the push gate sees it — run it to
see exactly what a `git push` will run. The hook fires only when the push
touches a `.py`.

### Kernel tier — `devtools/kernel-tier.sh`

```bash
devtools/kernel-tier.sh --backend cuda          # nvcc + the RTX 3080
devtools/kernel-tier.sh --backend hip           # hipcc + the RX 9060 XT
devtools/kernel-tier.sh --backend cuda --fresh  # wipe .kernel-tier/ first
devtools/kernel-tier.sh --backend cuda -- -k saxpy    # extra pytest args
```

Four phases, and **the report at the end says which of them did anything**:

| Phase | What it proves |
|---|---|
| preflight | the compiler is here; whether a card is |
| generate | `KERNEL_GEN_CMD` (`python -m felizjueves.emit`) writes the `[p\|q]` kernels, L = 0..16, into `.kernel-tier/src-<backend>/` — **dormant** if that variable is set empty |
| compile | every emitted translation unit survives the real device compiler. The CUDA side takes `*.cu`; the **HIP side takes `*.hip` AND `*.cu`** (with `-x hip`), so one generated source can be checked against both compilers — which is the point |
| device | the `gpu`-marked suite **for that backend** (`-m "gpu and <backend>"`) agrees with the host reference — today `test_kernel_device.py` (the generated kernels' `J_p` and `fj_boys`) plus `test_toolchain.py`'s saxpy |

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

## The markers, and why a test skips

The two axes are independent, and conflating them is how a green-and-empty run
gets reported as a pass:

| Marker | Needs | Fixture that enforces it |
|---|---|---|
| `cuda` | nvcc — **no card required** | `nvcc` |
| `hip` | hipcc — **no card required** | `hipcc` |
| `gpu` | a live device | `cuda_device` / `hip_device` |
| `slow` | nothing; costs minutes (an exhaustive search re-run); off the pre-push gate via `FAST_TEST_ARGS` | — |

A test takes exactly the fixtures it needs, so the skip message names the real
reason (`hipcc not found -- it lives in the fj:hip / fj:combined image`, versus
`no AMD device on this machine`). When you add a test, pick the fixtures the
same way: asking for `cuda_device` in a compile-only test makes it skip on a
machine that could have run it.

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
