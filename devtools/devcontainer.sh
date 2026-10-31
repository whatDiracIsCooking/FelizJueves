#!/usr/bin/env bash
# Drive this worktree's devcontainer. There is usually no `devcontainer` binary
# on PATH, so every call goes through npx.
#
#   devtools/devcontainer.sh up          bring it up, reusing an existing one
#   devtools/devcontainer.sh rebuild     recreate from scratch
#   devtools/devcontainer.sh shell       open a bash shell inside
#   devtools/devcontainer.sh test [...]  run the test suite inside
#   devtools/devcontainer.sh gpu         can it still open its cards?
#   devtools/devcontainer.sh down        stop and remove the container
#   devtools/devcontainer.sh down --all  ... every GPU variant of this worktree
#
# `gpu` answers the question a long session eventually needs: a container's
# device grant can be REVOKED mid-life -- a host driver reload, a systemd
# daemon-reload -- leaving every device node visible inside and none of them
# openable. `up` and `rebuild` run the same check for you; this is the one to
# run when a card that worked an hour ago does not. Exit 1 means revoked, and
# the fix is `rebuild` (a plain `up` reuses the container and cannot restore
# it). lib.sh's card_revoked carries the rule.
#
# WHICH VARIANT: a flag BEFORE the command, naming a directory under
# .devcontainer/ -- so this repo's three configs are reachable as:
#
#   devtools/devcontainer.sh --hip shell        .devcontainer/hip/
#   devtools/devcontainer.sh --combined up      .devcontainer/combined/
#   devtools/devcontainer.sh --cuda rebuild     .devcontainer/cuda/ (the default)
#
# The flag must come before the command, because everything after `shell` and
# `test` is passed through to what runs inside. Use it on EVERY call in a
# session, teardown included -- `down` matches on the config file, so
# `--hip up` followed by a bare `down` takes down the CUDA container and leaves
# the HIP one running. `down --all` is the escape hatch.
#
# CPUSET / CPUS (from config.sh, or set for one call) bound the container's
# share of the host on `up` and `rebuild`:
#
#   CPUSET=0-11 devtools/devcontainer.sh up      pin to host cores 0-11
#   CPUS=8      devtools/devcontainer.sh up      cap at 8 cores' worth
#
# Unlike JOBS, which only bounds the test runner, these bound everything the
# container runs -- compiles started from a `shell` included.
#
# The three configs, and what each is:
#
#   .devcontainer/cuda/devcontainer.json      CUDA 13 (the default)
#   .devcontainer/hip/devcontainer.json       ROCm, no CUDA at all
#   .devcontainer/combined/devcontainer.json  both SDKs, ~40GB
#
# Three ways to choose, highest precedence first: the flag above, the
# DEVCONTAINER_CONFIG environment variable, and its default in config.sh (the
# CUDA variant -- the one that can actually build the C++ tree). The variable
# is still the way to point at a config that is not one of these, and the way
# to make a whole shell session use one:
#
#   export DEVCONTAINER_CONFIG=.devcontainer/hip/devcontainer.json
#
# Containers key on the workspace folder path AND the config file, so each
# worktree gets its own container per variant, and the variants of one worktree
# can be up at the same time. They SHARE that worktree's named volumes (see the
# comment in the json), so run the suite in one variant at a time. The script
# resolves the workspace from its own location, so it does the right thing
# whichever checkout you call it from, and from any cwd.
#
# IMAGES do not key on the config file, and that asymmetry surprises people:
# the CLI names a built image `vsc-<workspace basename>-<sha256 of the
# workspace PATH>`, so all three variants of one checkout build to that ONE
# name and each `up` retags it away from the last. The containers are
# unharmed -- each holds its image by id -- but the variant that lost the tag
# goes untagged and unexplainable in `docker images`, so `up` and `rebuild`
# add a `:<variant>` tag beside the CLI's `:latest`. See tag_variant_image.
#
# Project-specific values (the test command, the worker count) come from
# devtools/config.sh -- edit that, not this.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

WORKSPACE=$REPO_ROOT
CLI=(npx -y @devcontainers/cli)

# A leading variant flag selects a config by DIRECTORY NAME under
# .devcontainer/. Resolved by convention rather than from a list, so adding a
# .devcontainer/<name>/ needs no edit here and this script stays copyable into
# the next repo -- the same convention doctor.sh globs for.
#
# A leading `--foo` that does not name an existing config is NOT silently
# treated as a variant: it falls through to the command dispatch and is
# reported as unknown, so a typo says so instead of failing later with a
# confusing path. Two variant flags in one call is an error rather than
# last-one-wins, which would make a copy-pasted command line quietly do
# something other than what it reads as.
variant_flag=
while [ $# -gt 0 ]; do
  case "$1" in
    # The name is restricted to a single plain directory name, which is what
    # makes interpolating it into a path safe: a name with slashes or dots
    # (`--../../elsewhere`) would otherwise reach outside .devcontainer/, and a
    # bare `--` would yield `.devcontainer//devcontainer.json`.
    --[A-Za-z0-9]*)
      name=${1#--}
      case "$name" in
        *[!A-Za-z0-9._-]*) break ;;
        *..*) break ;;
      esac
      candidate=$REPO_ROOT/.devcontainer/$name/devcontainer.json ;;
    *) break ;;
  esac
  [ -f "$candidate" ] || break
  if [ -n "$variant_flag" ]; then
    echo "error: two variant flags: $variant_flag and $1" >&2
    exit 2
  fi
  variant_flag=$1
  DEVCONTAINER_CONFIG=$candidate
  shift
done

# Which devcontainer.json to drive. Defaults in config.sh to the CUDA variant
# -- the one with the C++ toolchain -- and the flag above or an environment
# variable still wins for one call. A RELATIVE path is resolved against the
# repo root here, not the cwd: the CLI resolves --config relative to where you
# invoked it, so a bare `.devcontainer/cuda/devcontainer.json` would otherwise
# only work from the root. Empty would leave it to the CLI, which finds no
# top-level .devcontainer/devcontainer.json to fall back to -- but config.sh
# always sets it.
DC_CONFIG=()
if [ -n "${DEVCONTAINER_CONFIG:-}" ]; then
  case "$DEVCONTAINER_CONFIG" in
    /*) dc_config_path=$DEVCONTAINER_CONFIG ;;
    *) dc_config_path=$REPO_ROOT/$DEVCONTAINER_CONFIG ;;
  esac
  if [ ! -f "$dc_config_path" ]; then
    echo "error: DEVCONTAINER_CONFIG does not exist: $dc_config_path" >&2
    exit 1
  fi
  DC_CONFIG=(--config "$dc_config_path")
fi

# The variant this call drives: the directory name under .devcontainer/, which
# is what names a config here -- cuda, hip, combined. Empty only if
# DEVCONTAINER_CONFIG was cleared by hand, since config.sh always sets it.
# Derived from the RESOLVED path rather than from the flag, so it is still
# right when the config came from the environment or from config.sh's default.
VARIANT=
if [ -n "${dc_config_path:-}" ]; then
  VARIANT=$(basename "$(dirname "$dc_config_path")")
fi

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

# This workspace's container ids, one per line. Empty when nothing matches.
#
# TWO labels, not one. Every GPU variant of a worktree -- cuda, hip, combined
# -- is built for the SAME workspace folder and so carries the same
# `devcontainer.local_folder`; only `devcontainer.config_file` tells them
# apart, and the CLI sets both on every container it creates. Matching on the
# folder alone would make `down` remove a sibling variant's container and
# `apply_cpu_limits` pin whichever one `head -1` happened to return.
#
#   --stopped      include exited containers -- what `down` wants, since a
#                  crashed run keeps the labels. The bare form is the running
#                  container, the only thing `docker update` can act on.
#   --any-config   every variant for this workspace, not just the active one.
container_ids() {
  local ps=(docker ps -q) any_config=
  while [ $# -gt 0 ]; do
    case $1 in
      --stopped) ps=(docker ps -aq) ;;
      --any-config) any_config=1 ;;
    esac
    shift
  done
  local filters=(--filter "label=devcontainer.local_folder=$WORKSPACE")
  # dc_config_path is set whenever DEVCONTAINER_CONFIG is, and config.sh always
  # defaults it. With neither, the CLI picks the config itself and there is no
  # path to match on, so fall back to the folder alone -- which is the old
  # behaviour, and only reachable by clearing DEVCONTAINER_CONFIG by hand.
  if [ -z "$any_config" ] && [ -n "${dc_config_path:-}" ]; then
    filters+=(--filter "label=devcontainer.config_file=$dc_config_path")
  fi
  "${ps[@]}" "${filters[@]}"
}

# Apply CPUSET/CPUS to the running container with `docker update`, then restart
# it if it holds an NVIDIA GPU. A no-op when neither is set, and a warning
# rather than an error when nothing is running: the limits are a refinement of
# `up`, not a precondition for it.
#
# `docker update` changes only the limits it is given, so a later call with
# just CPUSET leaves an earlier CPUS quota in place, and unsetting both here
# clears nothing -- it just stops applying. To get back to an unbounded
# container, `rebuild` it (a fresh container starts with no limits).
#
# THE RESTART IS LOAD-BEARING, and only for `--gpus`. `docker update`
# regenerates the container's device cgroup from HostConfig.Devices -- and for
# `--gpus all` that list is EMPTY. The ask lives in HostConfig.DeviceRequests
# and is honoured by the NVIDIA container runtime's OCI hook at container
# START, so regenerating the cgroup silently revokes GPU access while leaving
# every /dev/nvidia* node visible inside. The symptom is
# `nvidia-smi: Failed to initialize NVML: Unknown Error`, and because the
# device nodes are still there it reads like a driver problem rather than a
# permissions one. Worse here: CMAKE_CUDA_ARCHITECTURES=native then fails at
# CONFIGURE time, so a perfectly good tree looks like a broken change.
#
# Restarting re-runs the hook and re-injects the devices; the limits persist
# across it (they are in the container config, not the cgroup). The HIP
# variant needs none of this -- `--device=/dev/kfd` populates
# HostConfig.Devices, which `docker update` preserves -- so the restart is
# gated on DeviceRequests rather than applied blindly.
#
# THE RESTART IS NOW BELT AND BRACES, AND KEPT ANYWAY. The cuda and combined
# configs name their NVIDIA nodes with `--device` as well, precisely so the
# grant lives in the list `docker update` rebuilds from, so this call no longer
# revokes anything it then has to repair. It stays because the pin covers only
# the FIRST card (export_nvidia_devices says why) and because a config edited
# to drop the pin should degrade to a working container, not a silently dead
# one. warn_if_gpu_revoked runs after this either way, so if the pair of them
# ever stops being enough, the next `up` says so.
apply_cpu_limits() {
  [ -z "${CPUSET:-}" ] && [ -z "${CPUS:-}" ] && return 0
  local id flags=()
  # `|| true`: docker may be absent or its daemon down, and a missing CPU
  # pin is not a reason to fail a command that has already done its work.
  id=$(container_ids | head -1 || true)
  if [ -z "$id" ]; then
    echo "cpu-limit: no running container for $WORKSPACE (skipped)" >&2
    return 0
  fi
  [ -n "${CPUSET:-}" ] && flags+=(--cpuset-cpus "$CPUSET")
  [ -n "${CPUS:-}" ] && flags+=(--cpus "$CPUS")
  docker update "${flags[@]}" "$id" >/dev/null
  echo "cpu-limit: ${CPUSET:+cpuset-cpus=$CPUSET }${CPUS:+cpus=$CPUS }-> $id" >&2

  # See the comment above: only a hook-injected GPU needs this. `tr -cd` rather
  # than a bare test, so a container that died between the two calls yields an
  # empty string and not a stray newline that would read as non-zero. The
  # restart is non-fatal for the same reason the pin is: `up` has already done
  # its work by this point.
  local requests
  requests=$(docker inspect "$id" \
    --format '{{len .HostConfig.DeviceRequests}}' 2>/dev/null | tr -cd '0-9')
  if [ -n "$requests" ] && [ "$requests" != "0" ]; then
    if docker restart "$id" >/dev/null 2>&1; then
      echo "cpu-limit: restarted $id to restore its GPU device cgroup" >&2
    else
      echo "cpu-limit: could not restart $id -- its GPU may be unusable until you do" >&2
    fi
  fi
}

# --- the GPU grant: pin it, then check it ----------------------------------

# Export one FJ_NVIDIA_* per CUDA device node, which the `--device` entries in
# the cuda and combined runArgs read.
#
# WHY PIN THE NODES WHEN `--gpus all` ALREADY ASKS FOR THEM: because the two
# asks are kept in different places and only one of them survives. `--gpus all`
# lands in HostConfig.DeviceRequests, honoured by the NVIDIA container
# runtime's OCI hook at container START; an explicit `--device` lands in
# HostConfig.Devices, which is the list docker REGENERATES the device cgroup
# from. Regenerate it -- `docker update`, a host driver reload, a
# `systemctl daemon-reload` -- and a DeviceRequests-only grant is gone while
# the nodes stay visible inside (lib.sh's card_revoked has the full story).
# Naming the nodes as well puts the same grant in the list that is rebuilt, so
# the card survives a regeneration instead of needing a recreate.
#
# BOTH, not one: the hook does more than grant the cgroup -- it injects the
# driver libraries, nvidia-smi, and the /proc/driver/nvidia view -- so
# `--gpus all` stays. This is belt and braces, and the braces are the new part.
#
# WHY EACH ENTRY FALLS BACK TO A REAL NODE. runArgs is static JSON: it cannot
# ask whether a node exists, and `--device` on a path that is not there fails
# container CREATION. So every reference carries a `${localEnv:FJ_…:/dev/null}`
# default, and a host with no NVIDIA driver gets a harmless `--device /dev/null`
# instead of a container that will not start -- the same shape, and the same
# reason, as the baked-in gid defaults in export_host_gids below. Duplicate
# `--device /dev/null` entries are accepted by docker (measured), so all three
# collapsing onto the fallback is fine.
#
# Multi-GPU: only the FIRST card is pinned. The hook still injects the rest
# under `--gpus all`, so nothing is lost today -- but on a box where a second
# card has to survive a cgroup regeneration too, add it here and in the json.
export_nvidia_devices() {
  local var node
  # var:path pairs. The names are spelled in the cuda and combined runArgs;
  # CLAUDE.md's note about the FJ_ prefix living in two places covers these.
  for var in FJ_NVIDIA_CTL:/dev/nvidiactl \
             FJ_NVIDIA_UVM:/dev/nvidia-uvm \
             FJ_NVIDIA_UVM_TOOLS:/dev/nvidia-uvm-tools; do
    node=${var#*:}
    [ -c "$node" ] && export "${var%%:*}=$node"
  done
  # The card itself, whose node number is not fixed.
  for node in /dev/nvidia[0-9]*; do
    [ -c "$node" ] || continue
    export "FJ_NVIDIA_GPU=$node"
    break
  done
}

# The backends the ACTIVE variant can actually reach: cuda for the cuda
# variant, hip for hip, both for combined. Read from the runArgs the config
# passes rather than from the variant NAME, so a new .devcontainer/<name>/
# needs no edit here -- the same convention the variant flag resolves by.
variant_backends() {
  local cfg=${dc_config_path:-}
  [ -f "$cfg" ] || return 0
  grep -q -- '"--gpus"' "$cfg" && echo cuda
  grep -q -- '/dev/kfd' "$cfg" && echo hip
  return 0
}

# Ask lib.sh's card_revoked INSIDE the running container, for one backend.
# Returns 0 when that backend's card is present there and cannot be opened.
#
# THE RULE IS NOT RESTATED HERE. The workspace is bind-mounted at
# DEVCONTAINER_WORKSPACE_FOLDER (config.sh), so the container has this repo's
# own devtools/lib.sh at a known path and the probe is the same function the
# host, doctor.sh and the test fixtures ask -- which is the only way they
# cannot drift apart. `bash -c`, not the devcontainer CLI: this is a
# diagnostic, and run_cli's sed filter would buffer it.
container_card_revoked() {
  local id backend=$1 lib=${DEVCONTAINER_WORKSPACE_FOLDER:-/workspace}/devtools/lib.sh
  id=$(container_ids | head -1 || true)
  [ -n "$id" ] || return 1
  docker exec "$id" bash -c ". '$lib' && card_revoked '$backend'" 2>/dev/null
}

# Warn -- loudly, to stderr -- when the running container holds a card it is no
# longer allowed to open. Called after `up` and `rebuild` so the state is named
# where it happens, instead of surfacing later as a tier that fails for no
# visible reason.
#
# Why it matters most on `up`: a plain `up` REUSES an existing container, and a
# reuse cannot re-run the OCI hook that grants the devices. So the one command
# people retry when the GPU looks broken is the one command that cannot fix it.
#
# Never fatal: `up` has done its work by the time this runs, and a degraded
# card is a thing to report, not a reason to tear down a working container.
#
# IT FAILS QUIET, and that is the one deliberate hole. If the probe cannot be
# asked at all -- no bash in there, or the workspace mounted somewhere other
# than DEVCONTAINER_WORKSPACE_FOLDER -- container_card_revoked returns
# non-zero and this says nothing, rather than warning about a card it did not
# actually look at. The loud form is `devcontainer.sh gpu`, which separates
# "could not ask" (exit 2) from "revoked" (exit 1); doctor.sh run INSIDE the
# container is the other way round to the same answer.
warn_if_gpu_revoked() {
  command -v docker >/dev/null 2>&1 || return 0
  local backend revoked=()
  while read -r backend; do
    [ -n "$backend" ] || continue
    container_card_revoked "$backend" && revoked+=("$backend")
  done < <(variant_backends)
  [ "${#revoked[@]}" -gt 0 ] || return 0
  echo "devcontainer: WARNING -- ${revoked[*]} card(s) are PRESENT in the" >&2
  echo "devcontainer:   container and cannot be OPENED: the device cgroup no" >&2
  echo "devcontainer:   longer grants them (a host driver reload, a systemd" >&2
  echo "devcontainer:   daemon-reload, or a \`docker update\` after start)." >&2
  echo "devcontainer:   Every node is still visible in there, so the vendor" >&2
  echo "devcontainer:   tools will blame the driver. Recover by RECREATING it:" >&2
  echo "devcontainer:" >&2
  echo "devcontainer:       devtools/devcontainer.sh ${variant_flag:+$variant_flag }rebuild" >&2
  echo "devcontainer:" >&2
  echo "devcontainer:   A plain \`up\` reuses this container and will NOT" >&2
  echo "devcontainer:   restore the grant. If a fresh container still warns," >&2
  echo "devcontainer:   the fault is host-side -- run devtools/doctor.sh." >&2
}

# The .git bind mount in devcontainer.json reads its host path from
# ${localEnv:FJ_GIT_DIR}, which this resolves and exports before the CLI
# runs -- so no machine-specific path is ever written into the tracked json.
# (The literal after the colon there is only a fallback for a direct
# `devcontainer up` / VS Code "Reopen in Container", which does not run this.)
#
# WHY A GUARD, NOT JUST AN EXPORT: an unresolved mount source makes docker's
# --mount refuse a path that does not exist, and the devcontainer CLI reports
# that by printing the ENTIRE failing `docker run` line -- which by then has
# -e GH_TOKEN=<your token> in it. So a bad mount puts a credential on the
# terminal. Resolving it here, before the CLI is invoked, is what stops that.
#
# The source is the repo's COMMON git dir: a worktree's own `.git` is a file
# pointing into <main>/.git/worktrees/<name>, so the whole common dir has to be
# visible inside at its host path (source == target) for git to work there.
require_git_dir() {
  local d
  d=$(git -C "$REPO_ROOT" rev-parse --path-format=absolute \
        --git-common-dir 2>/dev/null || true)
  if [ -z "$d" ] || [ ! -d "$d" ]; then
    echo "error: cannot resolve this repo's git dir for the container mount." >&2
    echo "       run devtools/devcontainer.sh from inside the FelizJueves checkout." >&2
    exit 1
  fi
  export FJ_GIT_DIR="$d"
}

# The sccache bind mount reads its host path from
# ${localEnv:FJ_SCCACHE_DIR} the same way, and for the same reason: no
# machine-specific path in the tracked json.
#
# ONE STORE FOR EVERY CHECKOUT, and it is the PRIMARY one's. The common git dir
# resolved above is <primary>/.git whether this runs from the main checkout or
# from .claude/worktrees/<name>, so its dirname is the primary root in both
# cases -- which is what makes every worktree name the same directory.
#
# WHY THE mkdir: docker does not refuse a bind whose source is missing, it
# CREATES it, owned by root. The container runs as uid 1000, and the first
# compile then dies with `failed to create directory /home/ubuntu/.sccache/
# preprocessor: Permission denied` -- a message that never mentions a mount --
# while clearing the leftover needs sudo. Making it here is what stops that.
# (compose says the same thing declaratively, with create_host_path: false.)
#
# Note the store is shared but HITS are not, between trees mounted at different
# absolute paths: a restored BMI keeps the path it was built under. That is
# correct behaviour, not a misconfiguration: a cache key covers the compile
# command line, and that line carries the absolute path. Sharing the store
# still dedups what each tree recompiles.
require_sccache_dir() {
  local d="${FJ_SCCACHE_DIR:-$(dirname "$FJ_GIT_DIR")/.sccache}"
  if ! mkdir -p "$d" 2>/dev/null; then
    echo "error: cannot create the sccache store at $d" >&2
    echo "       set FJ_SCCACHE_DIR to a writable directory." >&2
    exit 1
  fi
  export FJ_SCCACHE_DIR="$d"
}

# Resolve ROCM_GROUPS to numeric HOST gids and export one FJ_<NAME>_GID
# per group, which is what the `${localEnv:...}` references in the hip and
# combined runArgs read.
#
# WHY THE JSON CANNOT SIMPLY NAME THE GROUP: `--group-add render` resolves the
# name INSIDE the container, where docker/install-rocm.sh made `render` gid
# 110, while the bind-mounted /dev/kfd and /dev/dri/renderD* keep their HOST
# ownership. The container user then joins a group the devices do not grant,
# the container starts clean, and the first device call fails with
# hipErrorNoDevice -- indistinguishable from a machine with no AMD card.
# kernel-tier.sh --backend hip resolves these with getent; this is the same
# resolution for the interactive path, off the same config.sh list.
#
# NON-FATAL, unlike kernel-tier.sh's, because each reference carries a baked-in
# default after the colon (`${localEnv:FJ_RENDER_GID:109}`). An
# unresolvable group leaves that value standing instead of passing an empty
# --group-add, and that default is also what a VS Code "Reopen in Container"
# gets -- nothing exports these for it. So the default is a last-known-good
# host GID, never a group NAME: a name there would be exactly the silent wrong
# answer this function exists to remove.
#
# The warning is gated on the ACTIVE config actually reading the variable, so
# the CUDA variant -- which passes no --group-add at all -- stays quiet on a
# host with no render group.
export_host_gids() {
  local grp gid var cfg=${dc_config_path:-}
  # Name the VARIANT in the warning: every config file is called
  # devcontainer.json, so the basename alone would not say which one.
  [ -z "$cfg" ] || cfg=$VARIANT/$(basename "$cfg")
  while IFS= read -r grp; do
    [ -n "$grp" ] || continue
    var=FJ_${grp//-/_}_GID
    var=${var^^}
    # A group name that does not survive into an identifier is skipped rather
    # than exported: `export 'FJ_A B_GID=1'` fails, and under `set -e` that
    # would take down an `up` over a config typo.
    case "$var" in
      *[!A-Za-z0-9_]*) continue ;;
    esac
    # Numeric HOST gid, or empty; host_gid (lib.sh) owns the getent exit-2 trap.
    gid=$(host_gid "$grp")
    if [ -n "$gid" ]; then
      export "$var=$gid"
    elif grep -q "$var" "${dc_config_path:-/dev/null}" 2>/dev/null; then
      echo "warning: no '$grp' group on this host (getent found none), so" >&2
      echo "         $var is unset and $cfg falls back" >&2
      echo "         to the gid baked into it. If the GPU is then invisible" >&2
      echo "         in there, fix ROCM_GROUPS in config.sh or that fallback." >&2
    fi
  done <<<"${ROCM_GROUPS:-}"
}

# Give every gid the container user holds a NAME inside the container.
#
# The --group-add values the hip and combined runArgs pass are numeric HOST
# gids and have to be (export_host_gids above owns that story), so the
# container user lands in a group its own /etc/group has no entry for: `render`
# is gid 109 on this host, while docker/install-rocm.sh made the container's
# own `render` 110. Nothing functional is wrong -- the gid is what the device
# nodes grant on, and a name is only a label -- but every getgrgid() miss
# surfaces as an error, and the one that greets you is the `groups` call in
# Ubuntu's /etc/bash.bashrc sudo hint, on every interactive shell:
#
#   groups: cannot find name for group ID 109
#
# which reads like a broken container rather than a working one. So add the
# missing entry as `host-<group>`, and NEVER by renumbering the container's own
# `render` to match: that is the mismatch-chasing the json comments warn
# against, and it is the bind-mounted nodes' HOST ownership that decides which
# gid actually works.
#
# Cosmetic, so never fatal. `docker exec` rather than a devcontainer lifecycle
# command because groupadd needs root and the remoteUser's sudo asks for a
# password; a VS Code "Reopen in Container", which never calls this script,
# keeps the warning.
name_host_gids() {
  command -v docker >/dev/null 2>&1 || return 0
  local id gid name gids=()
  # `|| true` for the same reason as apply_cpu_limits: a down daemon is not a
  # reason to fail an `up` that has already done its work.
  id=$(container_ids | head -1 || true)
  [ -n "$id" ] || return 0
  # `id -G` in there, rather than ROCM_GROUPS out here, so this covers whatever
  # the active config actually passed -- and stays a no-op for the cuda
  # variant, which passes no --group-add at all.
  read -ra gids <<<"$(docker exec "$id" id -G 2>/dev/null || true)"
  for gid in "${gids[@]}"; do
    docker exec "$id" getent group "$gid" >/dev/null 2>&1 && continue
    # The HOST's name for the gid, since that is what it means here; bare
    # `host-<gid>` when the host has no entry for it either.
    name=$(getent group "$gid" 2>/dev/null | cut -d: -f1 || true)
    docker exec -u root "$id" groupadd -g "$gid" "host-${name:-$gid}" \
      >/dev/null 2>&1 ||
      echo "warning: gid $gid has no name inside the container, and groupadd" \
        "could not add one -- harmless, but \`groups\` will complain" >&2
  done
}

# Fail before `exec` does, naming the variant and the command that fixes it.
#
# `devcontainer exec` neither creates a container nor STARTS a stopped one. Run
# against a variant that is merely stopped -- a reboot, a `docker stop`, a
# machine that slept -- it fails with the daemon's raw
# `container <id> is not running`, which names an id you have never seen, says
# nothing about WHICH variant it belongs to, and reads like a broken container
# rather than one that is simply down. A variant that was never built reads
# much the same way. Both are one `up` away.
#
# A diagnostic, not a gate: with no docker on PATH there is nothing to inspect
# here, so let the CLI produce its own error rather than inventing one.
require_running() {
  command -v docker >/dev/null 2>&1 || return 0
  [ -n "$(container_ids || true)" ] && return 0
  if [ -n "$(container_ids --stopped || true)" ]; then
    echo "error: the ${VARIANT:-active} container for $WORKSPACE exists, but is stopped." >&2
  else
    echo "error: no ${VARIANT:-active} container for $WORKSPACE." >&2
  fi
  echo "       start it: devtools/devcontainer.sh ${variant_flag:+$variant_flag }up" >&2
  echo "       (rebuild instead if its devcontainer.json or Dockerfile has moved on)" >&2
  exit 1
}

# Tag the image this variant just built with the variant's name.
#
# WHY: the CLI derives a built image's name from the workspace folder ALONE --
# `vsc-<basename>-<sha256 of the path>`, with the config file nowhere in it --
# while it keys the CONTAINER on both. So the three variants of one checkout
# share a single image name, and every `up` retags it away from whichever
# variant built it last, leaving that one's image untagged: still on disk,
# still held by its container, and no longer attributable in `docker images`.
# There is no `--image-name` on `up` to head that off -- the CLI offers that
# flag on `build` only -- so the tag goes on afterwards.
#
# It is a SECOND tag on the SAME repository, deliberately: `vsc-<...>:cuda`
# beside the CLI's `vsc-<...>:latest`, not a name of our own. worktree.sh's
# `rm` and `gc` recognise this project's build images by that
# `vsc-<basename>-<64hex>` repository and delete by image id, so they keep
# sweeping these with no edit there; a parallel naming scheme would leak past
# both.
#
# Non-fatal throughout, like apply_cpu_limits: `up` has already done its work
# by this point, and a missing tag costs legibility rather than a container.
tag_variant_image() {
  [ -n "$VARIANT" ] || return 0
  local cid repo img
  cid=$(container_ids | head -1 || true)
  [ -n "$cid" ] || return 0
  repo=$(docker inspect "$cid" --format '{{.Config.Image}}' 2>/dev/null || true)
  img=$(docker inspect "$cid" --format '{{.Image}}' 2>/dev/null || true)
  # Only an image the CLI built for this workspace. A config that names a
  # prebuilt `image` runs someone else's tag, and stamping one of our variant
  # names onto that would claim something about an image we did not build.
  case "$repo" in vsc-*) ;; *) return 0 ;; esac
  [ -n "$img" ] || return 0
  if docker tag "$img" "${repo%%:*}:$VARIANT" 2>/dev/null; then
    echo "image: tagged ${repo%%:*}:$VARIANT" >&2
  fi
}

# Say so when the Claude Code the image installs has fallen behind, or when
# the three Dockerfiles have drifted apart.
#
# WHY HERE: `up` and `rebuild` are the only moments the version in the
# container can change, and the failure this guards against is a SILENT one --
# the agent in the image used to come from a devcontainer feature that
# npm-installed an unpinned version into a permanently cached layer, so every
# rebuild reported success and moved nothing (docker/install-claude-code.sh
# has the measurements). Now that the pin is a line in git, the only thing
# left to notice is that the line is old, and nothing else would.
#
# Quiet when there is nothing to say, non-fatal always -- the container is
# already up by this point, and a registry that cannot be reached is not a
# reason to colour the run red. The timeout is short for the same reason: this
# is on the path of every `up`.
report_claude_pin() {
  local cv out pinned registry files
  cv="$REPO_ROOT/devtools/claude-version.sh"
  [ -x "$cv" ] || return 0
  [ -n "${CLAUDE_PIN_FILES:-}" ] || return 0
  out=$(CLAUDE_REGISTRY_TIMEOUT=${CLAUDE_REGISTRY_TIMEOUT:-3} "$cv" --porcelain 2>/dev/null)
  pinned=$(printf '%s\n' "$out" | sed -n 's/^pinned //p')
  registry=$(printf '%s\n' "$out" | sed -n 's/^registry //p')
  files=$(printf '%s\n' "$out" | sed -n 's/^files //p')
  if [ -z "$pinned" ] || [ "$pinned" = "-" ]; then
    echo "claude: the version pin disagrees across the Dockerfiles -- $files" >&2
    echo "        fix: devtools/claude-version.sh --apply <version>" >&2
  elif [ -n "$registry" ] && [ "$pinned" != "$registry" ]; then
    echo "claude: image pins $pinned, npm has $registry" >&2
    echo "        bump: devtools/claude-version.sh --apply, then $0 rebuild" >&2
  fi
}

# The same, for the Antigravity CLI. A separate function rather than a loop
# over both pins: the two differ in what "behind" even means -- claude's pin is
# a version the npm registry serves on demand, agy's is a release id plus the
# sha512 of one tarball, and only the second can be internally inconsistent
# (a release rewritten without its hash). That case gets its own line, because
# a build would refuse the download and report a corrupt-or-compromised
# payload, which reads like an attack rather than a half-finished bump.
report_agy_pin() {
  local av out pinned upstream files
  av="$REPO_ROOT/devtools/agy-version.sh"
  [ -x "$av" ] || return 0
  [ -n "${AGY_PIN_FILES:-}" ] || return 0
  out=$(AGY_REGISTRY_TIMEOUT=${AGY_REGISTRY_TIMEOUT:-3} "$av" --porcelain 2>/dev/null)
  pinned=$(printf '%s\n' "$out" | sed -n 's/^pinned //p')
  upstream=$(printf '%s\n' "$out" | sed -n 's/^upstream //p')
  files=$(printf '%s\n' "$out" | sed -n 's/^files //p')
  if [ -z "$pinned" ] || [ "$pinned" = "-" ]; then
    echo "agy: the release pin disagrees across the Dockerfiles -- $files" >&2
    echo "     fix: devtools/agy-version.sh --apply <release>" >&2
  elif [ -n "$upstream" ] && [ "$upstream" != "-" ] && [ "$pinned" != "$upstream" ]; then
    echo "agy: image pins $pinned, upstream has $upstream" >&2
    echo "     bump: devtools/agy-version.sh --apply, then $0 rebuild" >&2
  fi
}

# The CLI prints the whole `docker run` invocation on failure, GH_TOKEN and
# all. Everything below routes through this so a secret cannot reach the
# terminal; the exit status is the CLI's, not sed's.
#
# `sed -u` is load-bearing, not a micro-optimisation. sed's own stdout is the
# terminal, so without it sed is LINE-buffered: anything not terminated by a
# newline is held indefinitely. That silently swallows every partial-line
# write the CLI passes through -- pytest's progress dots, a compiler's
# carriage-returned status, and (before `shell` stopped routing through here)
# the shell prompt itself. GNU sed only; the images and the host are both
# Ubuntu.
run_cli() {
  local rc=0
  "${CLI[@]}" "$@" > >(sed -u -E 's/(GH_TOKEN|GITHUB_TOKEN)=[A-Za-z0-9_-]+/\1=***REDACTED***/g') \
    2> >(sed -u -E 's/(GH_TOKEN|GITHUB_TOKEN)=[A-Za-z0-9_-]+/\1=***REDACTED***/g' >&2) || rc=$?
  # Let the redirections drain before the caller reads the next line.
  wait
  return "$rc"
}

case "${1:-}" in
  up)
    shift
    require_git_dir
    require_sccache_dir
    export_host_gids
    export_nvidia_devices
    run_cli up --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" "$@"
    name_host_gids
    tag_variant_image
    apply_cpu_limits
    # AFTER apply_cpu_limits, which is itself a `docker update` and so the one
    # cgroup regeneration this script causes -- checking before it would clear
    # the container and then break it. `up` may also have reused a container
    # whose grant was already gone, which a reuse cannot restore; either way
    # this is where it gets named.
    warn_if_gpu_revoked
    report_claude_pin
    report_agy_pin
    ;;
  rebuild)
    # Needed after editing devcontainer.json or the Dockerfile: a plain `up`
    # reuses the running container and silently applies none of it, while
    # reporting success and the same container id. The workspace is a bind
    # mount and the caches are named volumes, so both survive; anything in the
    # container's own filesystem does not.
    shift
    require_git_dir
    require_sccache_dir
    export_host_gids
    export_nvidia_devices
    run_cli up --workspace-folder "$WORKSPACE" \
      --remove-existing-container "${DC_CONFIG[@]}" "$@"
    # Again on `rebuild`, not only `up`: a fresh container means a fresh
    # /etc/group, so the entry has to be re-added rather than inherited.
    name_host_gids
    tag_variant_image
    apply_cpu_limits
    # A fresh container should hold its cards. If this still warns, the fault
    # is host-side -- an unbound driver, a stale module -- and doctor.sh names
    # it; a second `rebuild` will not help.
    warn_if_gpu_revoked
    report_claude_pin
    report_agy_pin
    ;;
  shell)
    shift
    require_running
    # An INTERACTIVE shell must not route through run_cli, and the reason is
    # not the redaction but the pipe it needs to do it. Two things break once
    # the CLI's stdout is not a terminal:
    #
    #   1. sed withholds the prompt. A prompt ends in "$ ", not a newline, so
    #      a line-buffered filter holds it forever -- you get a blank screen
    #      and no echo of what you type, on a container that is working
    #      perfectly. (`sed -u` above fixes this much.)
    #   2. The terminal SIZE is lost regardless. The CLI decides whether to
    #      ask docker for a pty from process.stdin.isTTY, so a pty is still
    #      allocated, but it is sized from process.stdout.columns -- undefined
    #      through a pipe. Every full-screen program in there then draws into
    #      a default 80x24 that does not match your window.
    #
    # There is nothing to redact on this path anyway: GH_TOKEN reaches the
    # terminal from the `docker run` line that `up`/`rebuild` print on
    # failure, and `exec` prints no such line -- the token was baked into the
    # container's environment at create time.
    #
    # WITH arguments (`shell -c '...'`) it is a one-shot command rather than a
    # terminal session, so keep the filter: that is the path scripts and
    # agents use, and its output is what gets pasted into an issue.
    if [ $# -eq 0 ]; then
      exec "${CLI[@]}" exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" bash
    fi
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" bash "$@"
    ;;
  test)
    shift
    require_running
    # TEST_CMD may carry its own arguments, so split it as a command line
    # rather than on whitespace alone.
    config_args "$TEST_CMD"
    cmd=("${CONFIG_ARGS[@]}")
    # -rs because tests that degrade-to-skip on a missing tool make a green run
    # meaningless until you have read the skip reasons.
    xdist_args
    cmd+=("${XDIST[@]}" -rs)
    # Default to the fast tier, unless the caller passed their own marker
    # expression -- then which tier runs is their call.
    case " $* " in
      *" -m "*) ;;
      *) config_args "$FAST_TEST_ARGS"; cmd+=("${CONFIG_ARGS[@]}") ;;
    esac
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
      "${cmd[@]}" "$@"
    ;;
  gpu)
    # The on-demand form of the check `up` and `rebuild` run for you: can the
    # running container open its cards RIGHT NOW? Worth its own command because
    # the grant is lost mid-session, long after any `up`, and the question
    # otherwise only gets asked by a tier that then fails.
    #
    # One row per backend the active variant can reach, and an exit status that
    # separates the three answers a caller cares about:
    #
    #   0  every reachable backend's card opens -- or the container has none,
    #      which is a device-less setup working as intended
    #   1  a card is present and cannot be opened: REVOKED, and `rebuild` is
    #      the fix (warn_if_gpu_revoked prints it)
    #   2  COULD NOT RUN -- no container, no docker, no probe in there
    #
    # 2 is the repo's "could not run" and never a pass, the same contract
    # kernel-tier.sh exits on. Which is why this arm does NOT call
    # require_running: that helper exits 1, which is right where the ask was
    # "run this for me" and wrong here, where 1 already means something else.
    # A revoked card outranks a missing probe if both turn up.
    shift
    if ! command -v docker >/dev/null 2>&1; then
      echo "gpu: no docker on PATH -- cannot ask" >&2
      exit 2
    fi
    id=$(container_ids | head -1 || true)
    if [ -z "$id" ]; then
      echo "gpu: no running ${VARIANT:-active} container for $WORKSPACE -- cannot ask" >&2
      echo "     start it: devtools/devcontainer.sh ${variant_flag:+$variant_flag }up" >&2
      exit 2
    fi
    backends=$(variant_backends)
    if [ -z "$backends" ]; then
      echo "gpu: ${VARIANT:-the active variant} asks for no device -- nothing to check"
      exit 0
    fi
    # any_revoked, not `revoked`: warn_if_gpu_revoked keeps a local ARRAY of
    # that name, and one spelling used both ways is the kind of thing that
    # reads fine and then surprises someone (shellcheck SC2178/SC2128).
    any_revoked=0 unasked=0
    lib=${DEVCONTAINER_WORKSPACE_FOLDER:-/workspace}/devtools/lib.sh
    while read -r backend; do
      [ -n "$backend" ] || continue
      # Three states, asked through the repo's own rule rather than restated:
      # present and usable, present and denied, or simply not here.
      # card_revoked FIRST: presence folds usability in (#712), so a revoked
      # card answers "absent" to card_present and asking that first would
      # report the one state this command exists for as a device-less box.
      state=$(docker exec "$id" bash -c ". '$lib' &&
        if card_revoked '$backend'; then echo revoked
        elif card_present '$backend'; then echo usable
        else echo absent; fi" 2>/dev/null || echo unknown)
      case "$state" in
        usable)  echo "gpu: $backend -- card present and openable in $id" ;;
        revoked) echo "gpu: $backend -- card PRESENT but not openable in $id"
                 any_revoked=1 ;;
        absent)  echo "gpu: $backend -- no card in $id (device-less, or none passed through)" ;;
        *)       echo "gpu: $backend -- could not ask: no bash, or no $lib, in $id" >&2
                 unasked=1 ;;
      esac
    done <<<"$backends"
    if [ "$any_revoked" = 1 ]; then
      warn_if_gpu_revoked
      exit 1
    fi
    [ "$unasked" = 1 ] && exit 2
    exit 0
    ;;
  down)
    shift
    # By default this takes down the ACTIVE variant only -- the one
    # DEVCONTAINER_CONFIG names -- because a cuda and a hip container for the
    # same worktree are two different containers that merely share a workspace
    # folder. `down --all` takes down every variant of this workspace, which
    # is what you want before `worktree.sh rm` or when you have lost track.
    scope=()
    [ "${1:-}" = --all ] && scope=(--any-config)
    ids=$(container_ids --stopped "${scope[@]}")
    if [ -z "$ids" ]; then
      echo "no container for $WORKSPACE${scope:+ (any config)}"
    else
      # One id per line, and there can be more than one, so loop: a single
      # quoted "$ids" would reach docker as one newline-joined name and fail
      # with "No such container" while leaving both behind.
      while read -r id; do
        [ -n "$id" ] || continue
        docker rm -f "$id"
      done <<<"$ids"
    fi
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $1" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
