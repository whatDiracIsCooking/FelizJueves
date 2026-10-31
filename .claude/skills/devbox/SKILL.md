---
name: devbox
description: >-
  Drive this repo's containers and its container-backed sibling worktrees —
  devtools/devcontainer.sh (up/rebuild/shell/test/down) and
  devtools/worktree.sh (add/rm/sync/gc/list). Use when the user wants to bring
  a container up, get a shell or run the suite inside it, rebuild after editing
  a Dockerfile or devcontainer.json, create or tear down a full container-backed
  worktree, or clean up leaked docker state. NOT for the lightweight
  .claude/worktrees/ checkouts — those are the `worktree` skill.
---

# Containers and container-backed worktrees

Two scripts, one model: **a container is keyed on its workspace folder path**,
so every checkout gets its own container, its own named volumes and its own
build image. That is what makes several worktrees usable at once, and it is
also why removal needs `worktree.sh rm` rather than `git worktree remove`.

Both scripts resolve their workspace from their own location, so they act on
*the checkout they live in*, from any cwd. `devtools/devcontainer.sh up` run
from worktree `foo` brings up `foo`'s container, never main's.

## The container

```bash
devtools/devcontainer.sh up        # reuse if it exists, else create
devtools/devcontainer.sh rebuild   # recreate from scratch
devtools/devcontainer.sh shell     # bash inside
devtools/devcontainer.sh test      # the Python tier inside (see the `test` skill)
devtools/devcontainer.sh gpu       # can it still open its cards?
devtools/devcontainer.sh down      # stop and remove the container
```

- **Use `rebuild`, never `up`, after editing `devcontainer.json` or a
  `Dockerfile`.** `up` reuses the running container, applies none of the change,
  and reports success with the same container id — the single most expensive
  mistake here, because everything downstream looks like a code bug. The
  workspace is a bind mount and the caches are named volumes, so both survive a
  rebuild; anything written to the container's own filesystem does not.
- `down` removes the container but **keeps the volumes** (the pytest tmp tree,
  the agent's plugin state). That is deliberate: `up` again is cheap and
  nothing is lost. Reclaiming the volumes is `worktree.sh rm`'s job.
- `shell -c "<cmd>"` runs one command inside — how you inspect a container
  without an interactive session, and how the kernel tier is run:
  `devtools/devcontainer.sh shell -c devtools/kernel-tier.sh`.
- **`shell` and `test` need the container already running**, and say so:
  `devcontainer exec` neither creates one nor starts a stopped one, so after a
  reboot or a `docker stop` they refuse with *"the `<variant>` container …
  exists, but is stopped"* and print the `up` that fixes it. `up` on a stopped
  container is a `docker start` — no rebuild, seconds. Reach for `rebuild`
  only if the config or a `Dockerfile` has changed since.

### Three variants, one toolchain

`DEVCONTAINER_CONFIG` in `devtools/config.sh` picks which `devcontainer.json`
every subcommand drives. A relative path resolves against the repo root, so it
works from any cwd and any worktree. Each drives its own Dockerfile under
`docker/`, and all three share `docker/Dockerfile.base` as the parent:

| Config | Dockerfile | Toolchain |
|---|---|---|
| `.devcontainer/cuda/` | `Dockerfile.cuda` | g++/clang, pinned CMake+Ninja, sccache, the CUDA toolkit (`sm_86`), Nsight Systems. Needs the NVIDIA card + the container toolkit. **The default.** |
| `.devcontainer/hip/` | `Dockerfile.hip` | the same host toolchain with ROCm's HIP compiler and profilers (`gfx1200`) and no CUDA. Needs the AMD card. |
| `.devcontainer/combined/` | `Dockerfile.combined` | both device compilers. The one to open to check a generated kernel against nvcc *and* hipcc. |

All three are deliberately **thin**: no vendor math libraries, because no
generated kernel calls one (`docker/README.md`, "What is deliberately *not*
installed"). The host-side oracle is numpy, from `/opt/venv`. If you find
yourself wanting a library in a container, `apt-get install` inside it is the
wrong fix — the next `rebuild` loses it and nothing says so. Add it to the
Dockerfile, with the code that links it.

Nothing is prebuilt, because there is nothing to prebuild: what costs time in
here is **compiling generated kernels**, and an unrolled high-angular-momentum
kernel is slow on its own. That is what the shared `sccache` store is for — the
same emitted translation unit across runs is a cache hit. The store is a bind
mount, not a named volume, and it is the PRIMARY checkout's `.sccache` for
every worktree —
`devtools/devcontainer.sh up`/`rebuild` resolves it and creates it. It
survives a `rebuild`, since nothing in the container's own filesystem holds it.

What it shares is the **store**, not the **hits**: the devcontainer mounts each
worktree at its own host path, and a restored BMI keeps the absolute path it was
built under, so main and `.claude/worktrees/<name>` do not hit each other's
entries. `docker/compose.yaml` mounts every service at `/workspace` instead, so
hits DO cross worktrees there. Neither path sets `SCCACHE_BASEDIRS`: a cache
key covers the compile command line, absolute paths included, and normalising
them across differing roots buys wrong output rather than a wider cache.

Each config carries an **`initializeCommand` that builds its parent image**
(`docker/build.sh base`, or `cuda` for the combined variant) before the
devcontainer's own `docker build` runs — the files chain by tag, and a
devcontainer build cannot produce a parent on its own. It runs on every `up` and
`rebuild`, so a `Dockerfile.base` edit propagates with no extra step.

**The CUDA one is the default**, because it is where the project actually
builds. Override for one call with a variant flag, or for a whole session with
the variable:

```bash
devtools/devcontainer.sh --hip shell
export DEVCONTAINER_CONFIG=.devcontainer/hip/devcontainer.json
```

`doctor.sh` prints which config is active; a path that does not exist is a FAIL,
and every container command refuses until it is fixed.

### Bumping the Claude Code in the image

The agent is **pinned** (`ARG CLAUDE_CODE_VERSION` in each GPU `Dockerfile`,
installed by `docker/install-claude-code.sh` as the second-to-last layer --
`install-agy.sh` is the leaf), not installed by a devcontainer feature. So it
moves only when you move it:

```bash
devtools/claude-version.sh                 # pinned / registry / installed here
devtools/claude-version.sh --apply         # rewrite all three to the registry's latest
devtools/claude-version.sh --apply 2.1.300 # or to one you name
devtools/claude-version.sh --rebuild       # ...and rebuild the images already built here
```

Then `devtools/devcontainer.sh rebuild` for whichever variant you work in — the
pin is in the image, so a `rebuild` without a re-`docker build` would recreate
the container from the same image and change nothing. `up` and `rebuild` print
a one-line warning per agent when a pin is behind upstream or the three files
disagree, and `doctor.sh` says the same thing on demand.

**Do not `npm install -g @anthropic-ai/claude-code` inside a container.** It is
lost on the next rebuild, and `doctor.sh` reports it as *"claude on PATH is A,
this image pinned B"* — the same class of mistake as `apt-get install`ing a
library in there instead of in the Dockerfile.

### Bumping the Antigravity CLI (`agy`) in the image

Pinned the same way, in the same three files, and installed by
`docker/install-agy.sh` as the leaf layer — but by **release**, not version, and
as **two lines** (`ARG AGY_RELEASE` plus `ARG AGY_SHA512`). Upstream serves only
the current release, so a bare version is not addressable the way an npm one is.

```bash
devtools/agy-version.sh                    # pinned / upstream / installed here
devtools/agy-version.sh --apply            # rewrite all three to upstream's current release
devtools/agy-version.sh --apply 1.3.3-5524738307653632   # or to a release you name
devtools/agy-version.sh --verify           # re-hash the pinned tarball against the pin
```

Two things to know before reaching for it:

- **Never rewrite one of the two lines by hand.** The sha512 belongs to the
  tarball the release names, and `install-agy.sh` *verifies* it, so a release
  bumped without its hash is a build that fails its own checksum and reports a
  corrupt-or-compromised payload. `--apply` always writes both, from one
  source: upstream's manifest when no release is named, or the tarball's own
  hash (a ~62MB download, and it says so) when one is.
- **A named release is for rolling back**, to a value that was in the pin
  before — upstream publishes no index of older releases, so git is the index.

`doctor.sh` reports a hash that drifted from its release as `sha-differs` in the
file list, which is the one failure the Claude Code pin cannot have.

### Bounding a container's CPU

`CPUSET` (host cores) and `CPUS` (a core-count quota) come from
`devtools/config.sh`, or from the environment for one call, and are applied by
`up` and `rebuild` with `docker update`:

```bash
CPUSET=0-11 devtools/devcontainer.sh up      # pin to host cores 0-11
CPUS=8      devtools/devcontainer.sh up      # cap at 8 cores' worth
CPUSET=0-11 devtools/devcontainer.sh up      # re-pin one already running
```

This is **not** `JOBS`. That is a worker count — pytest's, and the kernel
tier's parallel compiles (`kernel-tier.sh -j N`) — and it reaches neither a
compile started from a `shell` by hand nor anything else in the container.
`CPUSET`/`CPUS` bound everything in there.

A device compile is memory-hungry per job, so for the kernel tier the right
`-j` is usually *lower* than the core count: an OOM-killed nvcc reads as a
mysterious failure rather than as a memory problem.

Reach for it when two worktree containers are up at once: pinned to disjoint
cores (`CPUSET=0-11` in one, `CPUSET=12-23` in the other) they physically
cannot contend, which is the only way a timing measurement in one of them means
anything. Limits belong to the **container**, and a container is per workspace
folder — so pin per worktree, not per shell; two `shell`s from the same
checkout share one container and one set of limits.

Three things to know:

- **On the CUDA container this costs a restart, deliberately.** `docker update`
  regenerates the device cgroup from `HostConfig.Devices`, which for `--gpus
  all` is **empty** — the GPU ask lives in `DeviceRequests`, honoured by the
  NVIDIA runtime's OCI hook only at container *start*. So a bare `docker update`
  silently revokes GPU access (`nvidia-smi: Failed to initialize NVML`, and in
  this repo a kernel tier whose compile phase still passes while every `gpu`
  test fails on a card that is right there). `devcontainer.sh` restarts to
  re-run the hook; the limits persist across it because they live in the
  container config. The HIP variant is untouched — `--device=/dev/kfd`
  populates `HostConfig.Devices`, which `docker update` preserves — so the
  restart is gated on `DeviceRequests`. Since the cuda and combined configs
  now name their NVIDIA nodes with `--device` too, the restart is belt and
  braces rather than the only thing holding it up — see "A card that drops
  mid-session" below.
- `docker update` changes only what it is passed, so a second call with just
  `CPUSET` leaves an earlier `CPUS` quota in place, and running `up` with both
  unset **clears nothing** — it only stops applying. `rebuild` is how you get
  an unbounded container back; a fresh one starts with no limits.
- If nothing is running, the limits are skipped with a `cpu-limit: no running
  container …` line on stderr rather than an error — `up` had already done its
  work.

### Killing a container test run on the host does not kill it inside

`npx` dies; the pytest master and its xdist workers keep running at 100% CPU,
and nothing says so — the next run just comes out slower and you measure *that*.
After interrupting:

```bash
devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
```

`pkill -f pytest` reaches only the master (a worker's command line is a bare
`python3 -u -c import sys;...`), so kill the workers **by PID**.

### A card that drops mid-session

A container's device grant can be **revoked while it runs**, and the symptom
names neither the cause nor the fix. `--gpus all` lands in
`HostConfig.DeviceRequests` and is injected by the NVIDIA runtime's OCI hook at
container *start*, so anything that regenerates the device cgroup afterwards —
a host driver reload, a `systemctl daemon-reload`, a `docker update` — drops it
while leaving **every `/dev/nvidia*` node visible inside**. Presence probes
still say the card is there; `open()` returns `EPERM`; `nvidia-smi` says
`Failed to initialize NVML: Unknown Error`.

```bash
devtools/devcontainer.sh gpu          # one row per backend the variant reaches
```

Exit **0** = every card opens (or the variant asks for no device), **1** =
present and not openable, **2** = **could not run** (no container, no docker,
no probe in there) — the same contract as `kernel-tier.sh`, so a 2 is never a
pass. `up` and `rebuild` run the same check for you.

- **The fix is `rebuild`, not `up`.** A plain `up` reuses the container, and a
  reuse cannot re-run the hook that grants the devices — so the one command
  people retry is the one that cannot work.
- **If a fresh container still warns, stop rebuilding.** The fault is
  host-side: run `devtools/doctor.sh`, whose `card usable` row FAILs with the
  denied nodes named.
- The cuda and combined configs also pass their NVIDIA nodes as `--device`, so
  the grant lives in the list docker *rebuilds* the cgroup from and survives a
  regeneration. Only the **first** card is pinned; a second still rides on
  `--gpus all` alone.
- A revoked card counts as **absent** for every tier decision (#712), so `gpu`
  tests skip and `gate.sh` labels the PR `unverified-on-card` and merges rather
  than failing every case. It is never *silently* absent: this command exits 1,
  doctor's `card usable` row FAILs, and the skip message names the denied nodes
  and the recreate. Fix the container and the next run verifies for real.

## Container-backed worktrees

Sibling directories under the repo root, named after their branch —
`<root>/main`, `<root>/<name>` — each with a container, three named volumes and
a build image.

```bash
devtools/worktree.sh list
devtools/worktree.sh add <name> [start-point] [--up]
devtools/worktree.sh rm  <name> [--force]
devtools/worktree.sh sync
devtools/worktree.sh gc [--dry-run] [--yes]
```

**`add`** creates or attaches branch `<name>` at `<root>/<name>`; `--up` also
brings the container up and drops you into a shell. It forks from the calling
checkout's **local HEAD**, and warns (offline, from the last fetch) when that
HEAD is behind its upstream — heed it, or the new branch starts on stale code.
It refuses the name `main`.

If it fails with *"$ROOT is not writable"*: the repo root, not the checkout,
needs to be writable. Fix once with the **numeric** uid —

```bash
sudo chown $(id -u) <root>
```

— because the host user and the container's `ubuntu` are typically both uid
1000 under different names, so `chown ubuntu` fails on the host.

**`rm`** is the one that matters. `git worktree remove` knows nothing about the
container, the three `<PROJECT_NAME>-{pytest-tmp,claude-plugins,claude-backups}-<name>`
volumes or the `vsc-<name>-<64hex>` image; `rm` tears down all of it, then
removes the worktree and deletes the branch — with `-d`, so an unmerged branch
is left in place and reported rather than lost. It refuses to remove the
checkout you are calling it from: run it from another one (typically main).

**`sync`** fast-forwards `main` to `origin/main`. Nothing else does this: after
a PR merges `origin/main` moves and local `main` does not, and `add` forks from
local HEAD. It finds whichever checkout has `main` on it (so it works from any
worktree, including a `.claude/worktrees/` one) and is `--ff-only` — a diverged
`main` is refused rather than turned into a merge commit.

**`gc`** reconciles docker against the worktrees that actually exist, for
everything an earlier plain `git worktree remove` leaked. Always show
`--dry-run` first; it prints a `orphans: N (…)` line and deletes nothing.
It is scoped to this repo — volumes by `PROJECT_NAME` prefix, images by the
`devcontainer.project` label, containers by folders under this repo's root or
its `.claude/worktrees/` — so it is safe beside other docker workloads. An
image with no label is never touched.

`doctor.sh` runs `gc --dry-run` for its "docker hygiene" line, which is how
leaked state surfaces without anyone going looking.

## The other front end: docker compose

`docker/compose.yaml` runs **the same `docker/Dockerfile.cuda` image** as one-shot
batch jobs — configure, build, test, exit — teeing everything to `.log/`. It is
not layered on the devcontainer and neither is deprecated:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)     # required; compose.yaml uses :? not a default
alias dc='docker compose -f docker/compose.yaml run --rm'
dc build ; dc test ; dc asan ; dc compute-sanitizer
```

Use compose when you want to read a result and throw the container away — CI,
nightlies, a sanitizer sweep, a clean reproducible run. Use the devcontainer
when you want to *stay inside*: iterating, agents, debugging. Compose services
are global (not per worktree) and honour no `CPUSET`/`CPUS`, so two concurrent
compose runs will contend. `docker/README.md` has the full variable list.

The image is built by `docker/build.sh`, which walks the tag chain
(`Dockerfile.base`, then `Dockerfile.cuda`) and tags the result both
`<project>:cuda` and `<project>:latest` — the second being what `FJ_IMAGE`
defaults to:

```bash
docker/build.sh cuda
```

Run it from anywhere; the build context is always the repo root, because the
Dockerfiles read `pyproject.toml`, `uv.lock` and the install scripts relative to
it. Do **not** hand-roll `docker build -f docker/Dockerfile.cuda` — with no
parent tagged it tries to *pull* `<project>:base` and fails with `pull access
denied`, which reads like a registry problem.

## When NOT to use this skill

For a quick isolated checkout, a spike, or agent scratch space, use the
lightweight `.claude/worktrees/<name>` layout and the **`worktree`** skill — no
container, no volumes, no image, and cheap to throw away. Reach for the
container-backed ones only when the work needs the image: the full suite, a
toolchain that exists only inside, or a GPU.
