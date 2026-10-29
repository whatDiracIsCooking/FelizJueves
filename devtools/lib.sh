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

# Does amdkfd have an actual GPU node? Echoes nothing; returns 0 when at least
# one node in kfd's topology is a GPU, 1 otherwise.
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
# reason _has_nvidia_device (test/conftest.py) reads /dev/nvidiactl rather than
# calling nvidia-smi.
#
# Node 0 is the CPU and reports gfx_target_version 0; a GPU reports its ISA
# version there (gfx1200 -> 120000). doctor.sh has always skipped the CPU node
# on exactly this field -- this is that rule, lifted to where every caller can
# reach it.
amd_gpu_present() {
  local props gfxver
  for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -r "$props" ] || continue
    gfxver=$(sed -n 's/^gfx_target_version //p' "$props" | head -1)
    [ -n "$gfxver" ] && [ "$gfxver" != 0 ] && return 0
  done
  return 1
}
