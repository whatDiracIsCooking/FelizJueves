#!/usr/bin/env bash
# Shared plumbing for the devtools scripts. Sourced, never run:
#
#   ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
#   . "$ROOT/devtools/lib.sh"
#
# Sourcing this also sources devtools/config.sh, so a script gets the project's
# settings and these helpers in one line. Nothing project-specific belongs in
# here -- that is config.sh's job.

# REPO_ROOT is the checkout this file lives in, resolved from its own location
# rather than from $PWD, so every script works from any cwd and in any worktree.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# shellcheck source=./config.sh
. "$REPO_ROOT/devtools/config.sh"

# Split a config value that may contain quoted arguments -- e.g. a marker
# expression like `-m "not gpu"`, three characters of which are quoting -- into
# an array. Plain word splitting would hand pytest `"not` and `gpu"`.
#
#   config_args "$FAST_TEST_ARGS"; cmd "${CONFIG_ARGS[@]}"
config_args() {
  # shellcheck disable=SC2034  # read by the caller as "${CONFIG_ARGS[@]}"
  CONFIG_ARGS=()
  [ -n "${1:-}" ] || return 0
  eval "CONFIG_ARGS=($1)"
}

# Split a newline-separated config value (the `name:description` lists) into an
# array, dropping blank lines. Values are taken literally -- no quote handling,
# because these are descriptions rather than command lines.
#
#   config_lines "$DOCTOR_OPTIONAL_TOOLS"; for e in "${CONFIG_LINES[@]}"; ...
config_lines() {
  CONFIG_LINES=()
  local line
  while IFS= read -r line; do
    [ -n "${line// /}" ] || continue
    CONFIG_LINES+=("$line")
  done <<<"${1:-}"
}

# Print a script's leading comment block as its help text: from line 2 down to
# the first `#:` sentinel, or to the first non-comment line if there is none,
# with the leading `# ` stripped.
#
#   usage() { usage_from_header "${BASH_SOURCE[0]}"; }
#
# Hardcoded line ranges (`sed -n '2,16p'`) were what this replaced, and all
# three scripts had already lost their last line to one: a sentence added to a
# header silently truncates the help nobody re-reads. Put a `#:` line where the
# help should stop when the header continues into notes for a reader of the
# file, as devtools/worktree.sh does.
usage_from_header() {
  sed -n '2,${
    /^#:/q
    /^[^#]/q
    s/^# \{0,1\}//p
  }' "$1"
}

# The xdist worker flag, plus XDIST_DIST's scheduler when set, or nothing when
# JOBS is empty (a suite without pytest-xdist installed). Used as
# `${XDIST[@]}`.
xdist_args() {
  # shellcheck disable=SC2034  # both assignments are read by the CALLER, as
  # "${XDIST[@]}"; shellcheck cannot see across the function boundary
  XDIST=()
  # shellcheck disable=SC2034
  [ -n "${JOBS:-}" ] && XDIST=(-n "$JOBS" ${XDIST_DIST:+--dist "$XDIST_DIST"})
  return 0
}

# Is this venv's interpreter actually RUNNABLE? Memoised, because find_tool is
# called a dozen times in one doctor run and this spawns a process.
#
# WHY IT IS NOT ENOUGH THAT THE FILE IS EXECUTABLE. `.venv` is a bind-mounted
# part of the workspace, so it is visible INSIDE the containers too -- but it
# was created on the host, and its `bin/python` symlinks to a host interpreter
# (uv keeps managed CPythons under $HOME) that the container does not have.
# Every entry point in it is still a perfectly executable file with a shebang
# pointing at nothing, so a plain `[ -x ]` test picks it in preference to
# /opt/venv and the run dies with
#   /workspace/.venv/bin/pytest: cannot execute: required file not found
# which names neither the venv nor the mount. Checking the interpreter is what
# makes VENV_PATHS mean "the first one that WORKS" rather than "the first one
# that exists", on either side.
_venv_runnable() {
  local p=$1
  case " ${_VENV_CHECKED:-} " in
    *" $p=1 "*) return 0 ;;
    *" $p=0 "*) return 1 ;;
  esac
  local ok=0
  if [ -x "$p/bin/python" ] && "$p/bin/python" -c '' 2>/dev/null; then ok=1; fi
  _VENV_CHECKED="${_VENV_CHECKED:-} $p=$ok"
  [ "$ok" = 1 ]
}

# Locate a tool without assuming it is on PATH: each VENV_PATHS entry's bin/ in
# turn (relative ones resolved against the repo root), then PATH. Echoes the
# path; returns 1 with nothing echoed when there is none, so callers can report
# the project's own setup command rather than a generic error.
#
# A bare `command -v` is wrong for anything `uv sync` installs. On the host
# those live in .venv/bin, which is NOT on PATH unless you activated the venv,
# so doctor.sh reported a venv tool missing on a machine where it worked
# perfectly; inside the image /opt/venv/bin IS on PATH, so the same check
# passed there. Opposite verdicts for the same working tool, neither of them
# about whether the tool works.
find_tool() {
  local v p
  for v in $VENV_PATHS; do
    case "$v" in /*) p="$v" ;; *) p="$REPO_ROOT/$v" ;; esac
    _venv_runnable "$p" || continue
    [ -x "$p/bin/$1" ] && { echo "$p/bin/$1"; return 0; }
  done
  command -v "$1" 2>/dev/null && return 0
  return 1
}

# Resolve a group NAME to its numeric gid on THIS host, echoing the gid or
# nothing. The `|| true` is load-bearing: getent exits 2 when the key is not
# found, and under `set -euo pipefail` that 2 would kill the CALLER here --
# before its own diagnostic, and with an exit code indistinguishable from a
# deliberate "cannot run". Both ROCm-on-a-real-card paths need the numeric HOST
# gid of the device groups (render, video), so the trap lives here once rather
# than copied into each; see kernel-tier.sh --backend hip and devcontainer.sh.
host_gid() {
  getent group "$1" 2>/dev/null | cut -d: -f3 || true
}

# The pytest the project's scripts should drive. Thin wrapper kept because the
# callers read better for it, and because "which pytest" is a question worth
# naming.
find_pytest() { find_tool pytest; }

# Same search, for the venv itself -- doctor.sh reports which one is in use.
# Echoes the directory, or nothing.
find_venv() {
  local v p
  for v in $VENV_PATHS; do
    case "$v" in /*) p="$v" ;; *) p="$REPO_ROOT/$v" ;; esac
    _venv_runnable "$p" && { echo "$p"; return 0; }
  done
  return 1
}

# --- which card: one lock, and one set of probes, per card ------------------
#
# A run launches on device 0 of whatever its runtime can see, and the variable
# that narrows that view says which PHYSICAL card that is. So the card is read
# from the environment rather than chosen here, and every probe below -- the
# lock, the enumeration, the device node -- asks about that one card. With no
# variable set it is card 0, the default device, which on a one-card-per-vendor
# box is the only card and is what every probe asked before cards were told
# apart.
#
#   gpu_card_index cuda|hip   -> the card's index, one line
#
#   cuda  the first entry of CUDA_VISIBLE_DEVICES.
#   hip   the first entry of HIP_VISIBLE_DEVICES (0 when unset), read THROUGH
#         ROCR_VISIBLE_DEVICES when that is set: ROCr filters the agents first
#         and HIP indexes what is left, so HIP_VISIBLE_DEVICES=1 under
#         ROCR_VISIBLE_DEVICES=2,3 is physical card 3.
#
# Only the FIRST visible card counts: the tests launch on device 0 alone, so
# `CUDA_VISIBLE_DEVICES=2,3` locks card 2 and leaves 3 to another run. An
# entry that is not an index (a GPU UUID, a MIG id) is kept as it is, spelled
# safe for a file name: it still names one card consistently, it just does not
# match the index spelling of the same card, and the node probes below fall
# back to every node for it. Whitespace is dropped; nothing else is parsed.
#
# test/conftest.py's card_index is the Python twin, and test/test_card_lock.py
# runs both over the same environments.
gpu_card_index() {
  local sel outer first list
  case "${1:-}" in
    cuda) sel=${CUDA_VISIBLE_DEVICES:-}; outer= ;;
    hip)  sel=${HIP_VISIBLE_DEVICES:-}; outer=${ROCR_VISIBLE_DEVICES:-} ;;
    *) echo "gpu_card_index: unknown backend '${1:-}' (cuda|hip)" >&2; return 2 ;;
  esac
  sel=${sel//[[:space:]]/}; outer=${outer//[[:space:]]/}
  first=${sel%%,*}
  [ -n "$first" ] || first=0
  if [ -n "$outer" ] && [[ $first =~ ^[0-9]+$ ]]; then
    IFS=, read -ra list <<<"$outer"
    # 10#: a leading zero is decimal here, as it is to the runtime.
    if (( 10#$first < ${#list[@]} )) && [ -n "${list[10#$first]}" ]; then
      first=${list[10#$first]}
    fi
  fi
  printf '%s\n' "${first//[^A-Za-z0-9-]/_}"
}

# The card's lock file: GPU_LOCK_PATH (config.sh) with `{backend}` and
# `{index}` filled in. The one spelling doctor.sh, kernel-tier.sh and the
# auto-pick read; conftest's lock_path fills the same template the same way.
gpu_lock_path() {
  local index path
  index=$(gpu_card_index "${1:-}") || return 2
  path=${GPU_LOCK_PATH//\{backend\}/$1}
  printf '%s\n' "${path//\{index\}/$index}"
}

# How many cards the backend's driver enumerates (0 when none), physical,
# whatever the *_VISIBLE_DEVICES variables say.
gpu_card_count() {
  local d n=0 props gfxver
  case "${1:-}" in
    cuda)
      for d in /proc/driver/nvidia/gpus/*/; do
        [ -d "$d" ] && n=$((n + 1))
      done
      if [ "$n" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
        n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU [0-9]') || n=0
      fi
      ;;
    hip)
      for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$props" ] || continue
        # 2>/dev/null: amd_enumerated below has why.
        gfxver=$(sed -n 's/^gfx_target_version //p' "$props" 2>/dev/null | head -1)
        [ -n "$gfxver" ] && [ "$gfxver" != 0 ] && n=$((n + 1))
      done
      ;;
    *) echo "gpu_card_count: unknown backend '${1:-}' (cuda|hip)" >&2; return 2 ;;
  esac
  echo "$n"
}

# Is the selected card among them? An index past the last card is no card --
# the runtime would see none either. A non-index entry (a UUID) is taken on
# trust when any card is here.
gpu_card_enumerated() {
  local index n
  index=$(gpu_card_index "$1") || return 2
  n=$(gpu_card_count "$1") || return 2
  [ "$n" -gt 0 ] || return 1
  [[ $index =~ ^[0-9]+$ ]] || return 0
  (( 10#$index < n ))
}

# The first card whose lock is free, for `--device auto`; card 0 when every
# one is held (the run then waits on it, saying for whom, as any run does).
# A hint, not a reservation: the lock is taken by the run itself, so two
# auto-picks racing for one free card both get it and the second waits. Only
# this project's locks are asked -- a card busy under another project's lock
# looks free here.
gpu_free_card() {
  local backend=$1 n i var path
  case $backend in
    cuda) var=CUDA_VISIBLE_DEVICES ;;
    hip)  var=HIP_VISIBLE_DEVICES ;;
    *) echo "gpu_free_card: unknown backend '$backend' (cuda|hip)" >&2; return 2 ;;
  esac
  n=$(gpu_card_count "$backend")
  for ((i = 0; i < n; i++)); do
    path=$(export "$var=$i"; gpu_lock_path "$backend")
    if [ ! -e "$path" ] || flock -n "$path" true 2>/dev/null; then
      echo "$i"; return 0
    fi
  done
  echo 0
}

# `--device N|auto` for the tiers: export the backend's *_VISIBLE_DEVICES so
# the run -- its probes, its lock and every launch under it -- is on card N.
#
#   select_card <backend> <N|auto>    (sets SELECTED_CARD; 64 on a bad value)
select_card() {
  local backend=$1 device=$2
  [ "$device" != auto ] || device=$(gpu_free_card "$backend")
  if ! [[ $device =~ ^[0-9]+$ ]]; then
    echo "--device takes a card index or 'auto', not '$device'" >&2
    return 64
  fi
  case $backend in
    cuda) export CUDA_VISIBLE_DEVICES=$device ;;
    hip)  export HIP_VISIBLE_DEVICES=$device ;;
    *) echo "select_card: unknown backend '$backend' (cuda|hip)" >&2; return 2 ;;
  esac
  # shellcheck disable=SC2034  # read by the caller
  SELECTED_CARD=$device
}

# Does amdkfd have the selected GPU (gpu_card_index)? Echoes nothing; returns 0
# when kfd's topology lists that many GPU nodes, 1 otherwise.
#
# WHY NOT `[ -e /dev/kfd ]`, which is what kernel-tier.sh asked before #431:
# that node exists whenever the amdkfd/amdgpu module is LOADED, whether or not
# the driver bound a card. The module binds by PCI id, so an amdgpu that
# predates the card -- kernel 6.8's in-tree one does not know the RX 9060 XT's
# 1002:7590 (gfx1200, RDNA4), which is why this box gets amdgpu from DKMS --
# loads happily and claims nothing, and /dev/kfd is still there. The tier then
# printed `device: present` on a box with no usable AMD card, every `gpu` test
# failed instead of skipping, and from inside a container that reads as broken
# passthrough rather than as an unbound driver on the host. It is the same
# reading either way: the topology below is identical on the host and inside a
# container given `--device=/dev/kfd`, which is the property that makes it the
# right probe.
#
# WHY NOT `rocminfo | grep gfx`, which #431 suggested: rocminfo is ROCm's, and
# ROCm lives in the hip/combined image -- so on a host that has the card but no
# ROCm, a vendor-tool probe answers "absent" about a perfectly good card.
# Reading the kernel's own topology needs nothing installed, which is the same
# reason _nvidia_enumerated (test/conftest.py) reads the driver's own procfs
# rather than calling nvidia-smi.
#
# Node 0 is the CPU and reports gfx_target_version 0; a GPU reports its ISA
# version there (gfx1200 -> 120000). doctor.sh has always skipped the CPU node
# on exactly this field -- this is that rule, lifted to where every caller can
# reach it.
amd_enumerated() {
  # The node walk is gpu_card_count's, which skips the CPU node on
  # gfx_target_version and swallows a refused read (below).
  #
  # 2>/dev/null on its sed because `-r` is not the whole answer: it asks
  # access(2), which reads the permission bits, while the READ itself can
  # still be refused -- measured in a container with kfd's sysfs visible and
  # no --device=/dev/kfd, where every node answered `Operation not permitted`
  # and sed sprayed that onto the stderr of every tier that asked.
  gpu_card_enumerated hip
}

# Does the NVIDIA driver have the selected GPU enumerated? Returns 0 when it does.
#
# The same question amd_enumerated asks, for the other vendor, and with the
# same trap avoided: /dev/nvidiactl exists whenever the nvidia module is
# loaded, card or not, so it is not asked here. /proc/driver/nvidia/gpus/ holds
# one directory per GPU the driver bound (the container runtime mounts it under
# --gpus), which needs nothing installed; `nvidia-smi -L` is the fallback for a
# container that got the tool but not that procfs view. gpu_card_count does the
# counting; the selected card (gpu_card_index) must be among the cards counted.
nvidia_enumerated() {
  gpu_card_enumerated cuda
}

# ENUMERATED IS NOT USABLE, and `present` means usable (#712).
#
# Both functions above answer what the kernel and the driver say is attached.
# That is the right question for "did amdgpu bind a card" and the wrong one for
# "can this process launch on it", because a container can hold an enumerated
# card whose device-cgroup grant has been revoked: the nodes stay visible and
# every open() returns EPERM. On 2026-10-31 that state had doctor.sh printing
# `card verdict: cuda hip -- a PR from here is verified on both cards` two
# lines under an `[ok] nvidia:` row that was really an NVML failure, and it put
# gate.sh into a cuda gpu phase that failed 311 cases on
# `no CUDA-capable device is detected` instead of skipping and labelling.
#
# So presence folds usability in, exactly as #431 folded enumeration into it
# for AMD: one rule, asked by card_present / card_absent, gate.sh's gpu phase,
# verify-unverified.sh, doctor.sh's verdict and conftest's device fixtures.
# A revoked card is then ABSENT -- the gpu sample is skipped, the PR carries
# `unverified-on-card` and merges, and verify-unverified.sh repays it later --
# while doctor.sh's `card usable` row and devcontainer.sh still FAIL loudly on
# it, so a revoked card is never silently absent. card_revoked below is what
# keeps those two apart.
#
# STUBBING: gpu_usable is the single point a test redefines after sourcing this
# file (test/test_device.py), which is what makes the rule testable on a box
# whose real answer cannot be changed.
# Both normalise to 0/1. gpu_usable distinguishes "denied" (1) from "nothing to
# probe" (2) because doctor.sh's row needs that, but neither is a card you can
# launch on, and card_present's contract keeps 2 for an unknown BACKEND -- so
# the two are collapsed here rather than passed through.
nvidia_gpu_present() {
  nvidia_enumerated || return 1
  gpu_usable cuda || return 1
}

amd_gpu_present() {
  amd_enumerated || return 1
  gpu_usable hip || return 1
}

# Is a card for this backend here? `card_present cuda|hip` returns 0 or 1.
# With no argument it echoes the backends that have one, space-separated (empty
# when none), and returns 0 only when BOTH do -- the condition under which a PR
# needs no `unverified-on-card` label (#651; the `pr` skill has the rule).
#
# This is the one place the verdict is computed: the pr skill and doctor.sh's
# `card verdict` line both call it, so they cannot disagree.
card_present() {
  case "${1:-}" in
    cuda) nvidia_gpu_present ;;
    hip)  amd_gpu_present ;;
    "")
      local b present=()
      for b in cuda hip; do card_present "$b" && present+=("$b"); done
      echo "${present[*]}"
      [ "${#present[@]}" -eq 2 ]
      ;;
    *) echo "card_present: unknown backend '$1' (cuda|hip)" >&2; return 2 ;;
  esac
}

# The backends with no card here, space-separated (empty when both have one):
# what a PR from this session states as unverified, under the label.
card_absent() {
  local b absent=()
  for b in cuda hip; do card_present "$b" || absent+=("$b"); done
  echo "${absent[*]}"
}

# --- present is not usable: the mid-session revocation ----------------------
#
# #431 settled that a card is PRESENT when the kernel enumerates it. That rule
# is still right, and it is still not enough: a container can hold an
# enumerated card it is no longer allowed to OPEN. The device cgroup and the
# driver's enumeration are two different things, and only one of them moves.
#
# HOW IT HAPPENS, and why it reads as a hardware fault. `--gpus all` is not a
# device list: the ask lives in HostConfig.DeviceRequests and the NVIDIA
# container runtime's OCI hook injects the nodes at container START. Anything
# that regenerates the container's device cgroup afterwards -- `docker update`
# (apply_cpu_limits below has the long version), a host driver reload, a
# `systemctl daemon-reload` -- drops the hook's grant while leaving every
# /dev/nvidia* node VISIBLE inside. So mid-session the card stops working and
# every presence probe still says it is there:
#
#   /proc/driver/nvidia/gpus/   still populated  -> nvidia_gpu_present: yes
#   /dev/nvidia0                still a node     -> ls: yes
#   open("/dev/nvidia0", O_RDWR)                 -> EPERM
#
# and the vendor tool reports it as `Failed to initialize NVML: Unknown Error`,
# which names neither the cause nor the fix. A plain `up` then REUSES that
# container and cannot restore the grant -- only a recreate re-runs the hook.
#
# WHY open(), AND NOT A VENDOR CALL. The probe has to answer for both vendors,
# on the host and inside a container, with nothing installed -- the same
# constraint that made amd_gpu_present read kfd's topology instead of running
# rocminfo. cuInit/hipInit would answer honestly but need libcuda.so.1 /
# libamdhip64.so, which a device-less or pre-ROCm box does not have, and then
# "cannot probe" is indistinguishable from "revoked". An open() needs nothing,
# and it is the EXACT operation the cgroup filter adjudicates: measured in both
# directions on this box -- the nodes bind-mounted into a container with no
# grant deny it, the same nodes under `--gpus all` allow it.
#
# It also catches the AMD-side analogue, which is a permissions fault too: the
# container user holding a `render` gid the bind-mounted /dev/kfd does not
# grant (export_host_gids in devcontainer.sh has that story) surfaces as
# hipErrorNoDevice, and shows up here as a denied open on the node itself.
#
# `[ -c ]` is load-bearing, not a tidiness check: bash's `<>` CREATES a missing
# file, so an unguarded probe of a node that is not there reports success and
# leaves a regular file in /dev.

# The device nodes a backend's runtime must be able to open, one per line.
# Only nodes that EXIST are listed, so the caller probes what is actually here
# and an absent card yields an empty list rather than a false denial.
gpu_nodes() {
  local n
  case "${1:-}" in
    cuda)
      # nvidiactl is the control node every CUDA process opens first;
      # nvidia-uvm is what the runtime needs for managed memory; nvidia[0-9]*
      # are the cards. nvidia-modeset is display-only and never opened here.
      #
      # NOT nvidia-uvm-tools, which devcontainer.sh's export_nvidia_devices
      # does pin. The two lists answer different questions and are meant to
      # differ: pin everything whose grant should survive a cgroup
      # regeneration (the profilers' node included), probe only what a LAUNCH
      # needs -- so a box with no profiler node is not reported as a revoked
      # card.
      #
      # ONE CARD'S NODE, the selected one (gpu_card_index): a launch on card
      # 2 opens /dev/nvidia2, and a revoked grant on card 3 is not this run's
      # problem. The CUDA index is taken as the device minor, which holds
      # where the cards are alike and CUDA orders them by PCI bus, as nvidia
      # does; when that node is not here -- a container handed one card,
      # which is /dev/nvidia2 inside and device 0 to CUDA, or a UUID entry --
      # every card node is probed instead, which is what this asked before.
      local card one=
      card=$(gpu_card_index cuda)
      [[ $card =~ ^[0-9]+$ ]] && [ -c "/dev/nvidia$((10#$card))" ] &&
        one=/dev/nvidia$((10#$card))
      for n in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia[0-9]*; do
        [ -c "$n" ] || continue
        case $n in
          /dev/nvidiactl|/dev/nvidia-uvm) ;;
          *) [ -z "$one" ] || [ "$n" = "$one" ] || continue ;;
        esac
        echo "$n"
      done
      ;;
    hip)
      # kfd is the compute driver; the DRM render nodes are what it allocates
      # through. /dev/dri/card* is display-side and not probed.
      [ -c /dev/kfd ] && echo /dev/kfd
      # EVERY AMD card's nodes, not the selected one's: kfd is one node for
      # all of them, and mapping a HIP index to a render node needs ROCm's
      # own ordering, which this probe must not depend on. One AMD card is
      # all this repo has run on.
      #
      # AMD's render nodes ONLY, chosen by the driver that owns each one.
      # A bare /dev/dri/renderD* glob is wrong on a two-vendor box: this one
      # has renderD128 on amdgpu and renderD129 on nvidia, and ROCm never
      # touches the second. Probing it would make an NVIDIA-side permissions
      # quirk read as a revoked AMD card. Same sysfs vendor field doctor.sh
      # already picks the amdgpu card by for its VRAM reading.
      for n in /sys/class/drm/renderD*; do
        [ -r "$n/device/vendor" ] || continue
        [ "$(cat "$n/device/vendor")" = 0x1002 ] || continue
        n=/dev/dri/$(basename "$n")
        [ -c "$n" ] && echo "$n"
      done
      # A container with no readable sysfs here contributes no render node and
      # probes /dev/kfd alone. That is the right way to degrade: kfd is where
      # the AMD-side permissions fault actually bites (a container user holding
      # a `render` gid the bind-mounted node does not grant), so the check
      # still catches it -- it just stops being able to name a second node.
      ;;
    *) echo "gpu_nodes: unknown backend '${1:-}' (cuda|hip)" >&2; return 2 ;;
  esac
  return 0
}

# The nodes this process may NOT open, one per line; empty when every node a
# backend needs opens. Silent, so a caller decides how loud to be.
#
# The subshell around the redirect is required: `:` is a POSIX special
# built-in, and a redirection error on one is fatal to a non-interactive shell
# -- under /bin/sh the bare form exits the whole script, printing nothing.
# Contained in a subshell, the status is just a status.
gpu_denied_nodes() {
  local n
  while read -r n; do
    [ -n "$n" ] || continue
    ( : <> "$n" ) 2>/dev/null || echo "$n"
  done < <(gpu_nodes "$1")
}

# Can this process actually USE the backend's card? Returns 0 when every node
# it needs opens, 1 when any is denied, and 2 when there is no card here at all
# -- a distinction the callers need, because "absent" is a skip and "denied" is
# a broken container with a known fix.
gpu_usable() {
  local nodes denied
  nodes=$(gpu_nodes "$1") || return 2
  [ -n "$nodes" ] || return 2
  denied=$(gpu_denied_nodes "$1")
  [ -z "$denied" ]
}

# Is the backend's card present-but-revoked -- ENUMERATED by the kernel, and
# not openable? Returns 0 only in that one state, so it is safe to use as the
# trigger for a warning: a device-less box and a working card both return 1.
#
# Asked of the *_enumerated functions, never of card_present: presence now
# folds usability in, so a revoked card is absent by that rule and asking it
# here would make this always false. The two questions are the whole point --
# card_present decides whether a tier runs and whether a PR is labelled, and
# this decides whether a human is told their container is broken.
#
# This is the one place the distinction is computed; devcontainer.sh's warning,
# its `gpu` command, doctor.sh's `card usable` row and test/conftest.py's
# device fixtures all ask it, so they cannot disagree about what is wrong.
card_revoked() {
  local rc=0
  case "${1:-}" in
    cuda) nvidia_enumerated || return 1 ;;
    hip)  amd_enumerated || return 1 ;;
    *) echo "card_revoked: unknown backend '${1:-}' (cuda|hip)" >&2; return 2 ;;
  esac
  gpu_usable "$1" || rc=$?
  # Only rc 1 is a revocation. rc 0 is a working card, and rc 2 is "no nodes to
  # probe" -- enumerated with nothing to open, which is how a device-less
  # container looks, not how a revoked one does.
  [ "$rc" = 1 ]
}
