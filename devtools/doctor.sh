#!/usr/bin/env bash
# Report what is degraded in this environment, so "why did half the suite skip"
# is one command rather than a green run you cannot trust. Runs on the host and
# inside the container. Exits non-zero only on a FAIL (the core suite cannot
# run); a WARN means a slice of the suite will skip, which is often fine.
#
#   devtools/doctor.sh
#
# What counts as optional here is project-specific and comes from
# devtools/config.sh: DOCTOR_OPTIONAL_TOOLS, DOCTOR_GPU_VENDORS and
# DOCTOR_REQUIRED_PATHS. Add the project's own device compiler, GPU vendor,
# submodule or fixture tree there rather than editing this script.
set -uo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

fails=0
warns=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n'  "$1"; warns=$((warns+1)); }
fail() { printf '  \033[31m[FAIL]\033[0m %s\n'  "$1"; fails=$((fails+1)); }
note() { printf '         %s\n' "$1"; }

# --- repo -----------------------------------------------------------------
echo "repo"
TOP=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$TOP" ]; then
  fail "not inside a git repository"
  echo; echo "cannot check anything else outside the repo"; exit 1
fi
ok "git repo at $TOP"

# The <root>/<branch> layout is anchored at the PRIMARY checkout, not at
# whichever worktree doctor happens to be run from: the parent of the shared
# git common dir is the primary worktree, and its parent is the root. Deriving
# these from $TOP was wrong in exactly the place it matters -- run from
# .claude/worktrees/<name> it called .claude/worktrees the "worktree root" and
# looked for main at .claude/worktrees/main.
COMMON=$(git -C "$TOP" rev-parse --git-common-dir 2>/dev/null)
case "$COMMON" in /*) : ;; *) COMMON="$TOP/$COMMON" ;; esac
PRIMARY=$(cd "$(dirname "$COMMON")" 2>/dev/null && pwd) || PRIMARY=$TOP

# The worktree root must be writable or `devtools/worktree.sh add` cannot make
# sibling checkouts.
ROOT=$(dirname "$PRIMARY")
if [ -w "$ROOT" ]; then
  ok "worktree root writable ($ROOT)"
else
  warn "worktree root not writable: $ROOT"
  note "worktree.sh add will fail; fix once: sudo chown $(id -u) $ROOT"
fi

# Every devcontainer.json in the tree, relative to $TOP: one per GPU variant
# (cuda, hip, combined). Discovered rather than listed, so adding a variant does
# not need an edit here -- the checks below are exactly the ones a new variant is
# most likely to get wrong.
dc_configs() {
  local p
  for p in "$TOP"/.devcontainer/*/devcontainer.json; do
    [ -f "$p" ] || continue          # no match: the glob comes back literal
    echo "${p#"$TOP"/}"
  done
}

# PROJECT_NAME is duplicated in every devcontainer.json (JSON cannot source
# shell), and if they drift, worktree.sh rm/gc stop recognising this project's
# volumes and silently leak them. Every variant is checked, not just the active
# one: a rename that missed any of them leaks from whichever side you forgot.
while read -r rel; do
  DCJSON="$TOP/$rel"
  [ -f "$DCJSON" ] || continue
  if grep -q "source=$PROJECT_NAME-" "$DCJSON"; then
    ok "PROJECT_NAME '$PROJECT_NAME' matches $rel volume names"
  elif grep -q 'type=volume' "$DCJSON"; then
    warn "$rel names volumes that do not start with '$PROJECT_NAME-'"
    note "worktree.sh rm/gc will not recognise them; align the two"
    note "  config: devtools/config.sh PROJECT_NAME   json: $DCJSON"
  fi
done < <(dc_configs)

# Which of them devcontainer.sh will actually drive. A path that does not exist
# is a hard stop for every container command, so it is a FAIL rather than a
# warning.
case "${DEVCONTAINER_CONFIG:-}" in
  "") note "DEVCONTAINER_CONFIG unset: there is no top-level .devcontainer/devcontainer.json; pass a variant flag or set it" ;;
  /*) DCACTIVE=$DEVCONTAINER_CONFIG ;;
  *) DCACTIVE="$TOP/$DEVCONTAINER_CONFIG" ;;
esac
if [ -n "${DCACTIVE:-}" ]; then
  if [ -f "$DCACTIVE" ]; then
    ok "devcontainer config: $DEVCONTAINER_CONFIG"
  else
    fail "DEVCONTAINER_CONFIG does not exist: $DCACTIVE"
    note "every devtools/devcontainer.sh command will refuse; fix devtools/config.sh"
  fi
fi

# The .git bind mount reads its host path from ${localEnv:FJ_GIT_DIR}, which
# devtools/devcontainer.sh injects on up/rebuild (the literal in the json is only
# a fallback for a direct VS Code "Reopen in Container"). So the thing worth
# checking is that the path it will inject actually resolves -- an unresolved
# mount makes the CLI print the failing `docker run` line, GH_TOKEN and all.
gitdir=$(git -C "$TOP" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [ -n "$gitdir" ] && [ -d "$gitdir" ]; then
  ok "container .git mount resolves via FJ_GIT_DIR ($gitdir)"
else
  warn "cannot resolve this repo's git dir for the container .git mount"
  note "run doctor from inside the FelizJueves checkout"
fi

# --- worktrees ------------------------------------------------------------
# Two failure modes that are local usability damage rather than lost work
# (committed work is safe on the remote), so these are warn/note, never fail:
# a primary checkout owned by another uid, and linked worktrees whose admin
# state has broken.
echo "worktrees"

owner=$(stat -c %u "$PRIMARY" 2>/dev/null)
if [ -n "$owner" ] && [ "$owner" != "$(id -u)" ]; then
  warn "primary checkout owned by uid $owner, not $(id -u): $PRIMARY"
  note "git and worktree ops there fail; fix: sudo chown -R $(id -u) $ROOT"
else
  ok "primary checkout owned by current uid ($PRIMARY)"
fi
if [ ! -d "$ROOT/main" ]; then
  note "no sibling main checkout at $ROOT/main (not the <root>/<branch> layout here)"
fi

# `prunable` means git found a linked worktree whose gitdir points nowhere --
# the working tree was removed out from under it. A worktrees/<name> admin dir
# with no readable `gitdir` file is the other half. `git worktree prune` clears
# both.
prunable=$(git worktree list --porcelain 2>/dev/null | grep -c '^prunable')
corrupt=0
if [ -d "$COMMON/worktrees" ]; then
  for wtd in "$COMMON"/worktrees/*/; do
    [ -d "$wtd" ] || continue
    [ -r "${wtd}gitdir" ] || corrupt=$((corrupt+1))
  done
fi
broken=$((prunable + corrupt))
if [ "$broken" -gt 0 ]; then
  warn "$broken broken/prunable linked worktree(s) ($prunable prunable, $corrupt corrupt admin dir)"
  note "clean up: git worktree prune; then recreate via devtools/worktree.sh"
else
  ok "no broken/prunable linked worktrees"
fi

# --- python / test tooling (the core suite) -------------------------------
echo "python / test tooling"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  warn "uv not found -- setup builds the venv from uv.lock with it"
fi
if venv=$(find_venv); then
  ok "venv: $venv"
else
  warn "no venv found in '$VENV_PATHS' -- run 'uv sync'"
fi

# find_venv's `-x bin/python` FOLLOWS the symlink, so a venv whose interpreter
# is unreachable reads as no venv at all -- and `uv sync`, the advice above,
# needs that same interpreter and fails with it. An interpreter that IS
# reachable can still be unusable: one whose stdlib sits behind a directory this
# uid cannot traverse prints --version happily, then dies on `import encodings`.
# Either way the message above points at the wrong thing, so name the real
# fault. Inside a container it means a stale image, nothing a command here fixes.
for vpath in $VENV_PATHS; do
  case "$vpath" in /*) vdir=$vpath ;; *) vdir=$TOP/$vpath ;; esac
  [ -L "$vdir/bin/python" ] || [ -f "$vdir/bin/python" ] || continue
  if "$vdir/bin/python" -c "import json" >/dev/null 2>&1; then
    ok "interpreter usable: $vdir/bin/python"
  else
    warn "'$vdir/bin/python' exists but cannot run as uid $(id -u)"
    note "target: $(readlink "$vdir/bin/python" 2>/dev/null || echo '<not a symlink>')"
    note "in a container: stale image -- rebuild with docker/build.sh <variant>"
    note "on the host: rm -rf '$vdir' && uv sync"
  fi
done
if pytest_bin=$(find_pytest); then
  ok "pytest: $pytest_bin"
else
  warn "pytest not found -- run 'uv sync' (dev group)"
fi

# uv.lock records the project name, and `uv sync --frozen` -- which is what both
# Dockerfiles and CI run -- refuses a lock that disagrees with pyproject.toml.
# The failure lands at IMAGE BUILD time with a message about the lock being out
# of date, a long way from the pyproject edit or the half-finished rename that
# caused it, so it is worth one grep here. Only the name is checked: anything
# deeper is uv's job, and `uv lock` is the fix for all of it.
if [ -f "$TOP/pyproject.toml" ] && [ -f "$TOP/uv.lock" ]; then
  py_name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$TOP/pyproject.toml" | head -1)
  lock_name=$(sed -n 's/^name = "\(.*\)"/\1/p' "$TOP/uv.lock" |
    grep -Fx "$py_name" | head -1)
  if [ -n "$py_name" ] && [ "$lock_name" = "$py_name" ]; then
    ok "uv.lock names the project '$py_name'"
  else
    warn "uv.lock does not name the project '$py_name' -- it is stale"
    note "'uv sync --frozen' will fail in both Dockerfiles and in CI"
    note "fix: uv lock, then commit the result (never hand-edit uv.lock)"
  fi
fi

# --- optional tools (project-declared) ------------------------------------
echo "optional tools"
config_lines "$DOCTOR_OPTIONAL_TOOLS"
if [ "${#CONFIG_LINES[@]}" -eq 0 ]; then
  note "none declared (DOCTOR_OPTIONAL_TOOLS in devtools/config.sh)"
fi
for entry in "${CONFIG_LINES[@]}"; do
  tool=${entry%%:*}
  what=${entry#*:}
  # find_tool, not `command -v`: some tools come from the project venv
  # and are on PATH inside the image but not on the host. See devtools/lib.sh.
  if toolpath=$(find_tool "$tool"); then
    # The first line that names a version (ncu, compute-sanitizer and the
    # shell linter all open with a banner), else the first line.
    vout=$("$toolpath" --version 2>/dev/null)
    ver=$(grep -m1 -E '[Vv]ersion:? +[0-9]' <<<"$vout" | sed 's/^ *//')
    [ -n "$ver" ] || ver=$(head -1 <<<"$vout")
    ok "$tool${ver:+: $ver}"
    command -v "$tool" >/dev/null 2>&1 || note "not on PATH; using $toolpath"
  else
    warn "$tool not found -- lost:${what:+ $what}"
  fi
done

# gh is the one optional tool with a second half: installed but unauthenticated
# is a distinct state, and the PR flow fails at push time rather than at start.
if command -v gh >/dev/null 2>&1 && [ -z "${GH_TOKEN:-}" ]; then
  warn "gh present but GH_TOKEN empty -- gh is unauthenticated"
  note "export a fine-grained PAT (Contents + Pull requests: R/W) on the host"
# A token can authenticate as the right user and still be unable to see THIS
# repo: a fine-grained PAT names its repositories one by one, and one that has
# not been given this one 404s. `gh auth status` reports a clean login either
# way, so the first sign of trouble is `gh pr create` failing with "Could not
# resolve to a Repository" after the branch is already pushed. One API call
# here turns that into a warning you get before starting.
elif command -v gh >/dev/null 2>&1; then
  # Match on github.com FIRST, then extract. Stripping known URL prefixes and
  # accepting anything left containing a slash was wrong: a clone from a local
  # path has an origin like /home/you/projects/thing, which survives every
  # substitution and still looks like `owner/repo` to a `*/*` glob. The probe
  # then asked GitHub for `repos//home/you/...`, got a 404, and reported "the
  # token lacks access to this repository" about a repository that is not on
  # GitHub at all. Same for a GitLab remote, and for a fresh `git init` with no
  # origin yet.
  ORIGIN=$(git -C "$TOP" remote get-url origin 2>/dev/null || true)
  case "$ORIGIN" in
    *github.com[:/]*)
      SLUG=$(printf '%s' "$ORIGIN" | sed -E 's#^.*github\.com[:/]+##; s#\.git$##; s#/+$##')
      # owner/repo and nothing else: two non-empty segments, no extra slashes.
      if printf '%s' "$SLUG" | grep -qE '^[^/]+/[^/]+$'; then
        if ${GH_PROBE_TIMEOUT:-timeout 10} gh api "repos/$SLUG" >/dev/null 2>&1; then
          ok "gh can reach $SLUG"
        else
          warn "gh cannot see $SLUG -- the PR and milestone flows will fail"
          note "the token authenticates but lacks access to this repository;"
          note "add it under Repository access on the fine-grained PAT"
        fi
      else
        note "origin looks like github but is not owner/repo: $ORIGIN"
      fi
      ;;
    "") note "no origin remote yet -- nothing to check gh against" ;;
    *) note "origin is not a github remote; skipping the gh reachability check" ;;
  esac
fi

# --- gpu devices (is a card actually here?) --------------------------------
#
# Deliberately separate from the toolchain checks above, because the two fail
# independently and only this half is invisible without a probe. A missing
# nvcc announces itself; a missing CARD does not, and its symptoms are easy to
# read as a broken change:
#
#   - A device compiler does not need a card: nvcc and hipcc target an
#     ARCHITECTURE, so the whole kernel tier compiles clean on a device-less
#     box and proves nothing about a number.
#   - The suites that do need one are marked `gpu` (pyproject.toml), and a
#     marked test SKIPS rather than failing -- so a device-less run is green
#     and empty, which is the failure this section exists to make visible.
#
# Probed through the kernel (sysfs / procfs) before any vendor tool, so the
# answer is the same on the host and inside a container, with or without
# nvidia-smi or rocminfo installed.
if [ -n "${DOCTOR_GPU_VENDORS// /}" ]; then
  echo "gpu devices"
  gpus_found=0

  # gfx_target_version is major*10000 + minor*100 + step, and the step prints
  # in HEX in the ISA name -- 90010 is gfx90a, not gfx9010. Getting that wrong
  # produces an --offload-arch the compiler silently has no code for.
  amd_gfx_name() {
    printf 'gfx%d%d%x' "$(( $1 / 10000 ))" "$(( ($1 / 100) % 100 ))" "$(( $1 % 100 ))"
  }

  # THE CARD-STATE ROW (#91): utilisation, free VRAM, and whether the card's
  # lock (GPU_LOCK_PATH, taken by test/conftest.py's device fixtures) is held
  # -- so a process outside this container holding the card is diagnosed
  # BEFORE a run, not inferred from an out-of-memory error after one. A reading
  # the tool cannot give is UNKNOWN, never idle: NVML says "Unknown Error"
  # intermittently, and amdgpu's gpu_busy_percent can refuse a read (EBUSY).
  # A held lock is a WARN: the next `gpu` run will wait behind it.
  #
  # THE COUNTER TOOL, PER SIDE (#110): the profiler that reads this card's
  # performance counters -- ncu for cuda, rocprofv3's counter mode for hip --
  # with its version, or a WARN naming what is lost. Present is not usable, and
  # the two ways it is not are checked without running a kernel: NVIDIA's
  # driver can restrict the counters to admins (ncu then fails with
  # ERR_NVGPUCTRPERM), and rocprofv3 can be installed without the libraries
  # its SDK loads (libdw, the AQL profiler). Either reads as "could not
  # measure" in felizjueves.performance.measured.profile, never as a zero.
  #
  #   counter_tool_state <cuda|hip>
  counter_tool_state() {
    local backend=$1 tool path ver lib missing
    if [ "$backend" = cuda ]; then tool=ncu; else tool=rocprofv3; fi
    path=$(find_tool "$tool" 2>/dev/null) ||
      for c in /usr/local/cuda/bin/$tool /opt/rocm/bin/$tool; do
        [ -x "$c" ] && { path=$c; break; }
      done
    if [ -z "$path" ]; then
      if [ "$backend" = cuda ]; then
        warn "card present but ncu is not -- lost: fp64 FLOP and DRAM-byte counters, so every sm_86 counter row and its arithmetic intensity"
      else
        warn "card present but rocprofv3 is not -- lost: DRAM-side byte counters, so every gfx1200 counter row"
      fi
      return
    fi
    if [ "$backend" = cuda ]; then
      ver=$("$path" --version 2>/dev/null | sed -n 's/^Version \([^ ]*\).*/\1/p' | head -1)
      ok "counters: ncu ${ver:-(version unread)} ($path)"
      if grep -qs '^RmProfilingAdminOnly: 1' /proc/driver/nvidia/params; then
        warn "the driver restricts GPU counters to admins -- ncu will fail with ERR_NVGPUCTRPERM: could not measure"
        note "fix (host): NVreg_RestrictProfilingToAdminUsers=0 in a modprobe.d file, then reload the driver"
      fi
    else
      ver=$("$path" --version 2>/dev/null | sed -n 's/^ *version: *\([^ ]*\).*/\1/p' | head -1)
      lib="$(dirname "$(readlink -f "$path")")/../lib/librocprofiler-sdk.so"
      missing=$(ldd "$lib" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | paste -sd, -)
      if [ -n "$missing" ]; then
        warn "rocprofv3 ${ver:-} present but its SDK cannot load ${missing//,/, } -- counter mode cannot run: could not measure"
        note "the image lacks them: libdw1 and hsa-amd-aqlprofile (docker/install-rocm.sh)"
      else
        ok "counters: rocprofv3 ${ver:-(version unread)} ($path)"
        # #140: loading is not measuring. gfx1200 defines no fp64 FLOP
        # counter, and its GL2C byte counters read 0 on this box, so it has
        # no roofline rows, by a final refusal -- said here so a working
        # profiler is not read as "the AMD rows are owed".
        note "loads, but gfx1200 has no fp64 FLOP counter and its byte counters read 0: no gfx1200 roofline rows (final: measured_store.UNMEASURED, #140)"
      fi
    fi
  }

  #   card_state <vendor> <backend> <util%|""> <free MiB|""> <total MiB|"">
  card_state() {
    local vendor=$1 backend=$2 util=$3 free=$4 total=$5 lock state holder
    lock=${GPU_LOCK_PATH//\{backend\}/$backend}
    [[ $util =~ ^[0-9]+$ ]] && util="$util%" || util="unknown"
    if [[ $free =~ ^[0-9]+$ && $total =~ ^[0-9]+$ ]]; then
      free="$free / $total MiB free"
    else
      free="free VRAM unknown"
    fi
    if [ ! -e "$lock" ]; then
      state="free (no lock file yet)"
    elif flock -n "$lock" true 2>/dev/null; then
      state="free"
    else
      state="HELD"
      # /proc/locks names the holder by device and inode: MAJ:MIN in hex.
      local dev ino key pid
      dev=$((0x$(stat -c %D "$lock"))); ino=$(stat -c %i "$lock")
      key=$(printf '%02x:%02x:%s' $(( (dev >> 8) & 0xfff )) \
            $(( (dev & 0xff) | ((dev >> 12) & 0xfff00) )) "$ino")
      while read -r pid; do
        if [ "$pid" -le 0 ]; then
          holder+="${holder:+; }a process outside this pid namespace"
        else
          holder+="${holder:+; }pid $pid ($(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100))"
        fi
      done < <(awk -v k="$key" '$2 == "FLOCK" && $6 == k {print $5}' /proc/locks)
    fi
    local row="$vendor card state: utilisation $util, $free, lock $lock $state"
    if [ "$state" = HELD ]; then
      warn "$row"
      note "held by ${holder:-a holder /proc/locks does not show} -- the next \`gpu\` run waits behind it"
    elif [ "$util" = unknown ] || [ "$free" = "free VRAM unknown" ]; then
      warn "$row"
      note "an unreadable reading is unknown, not idle -- re-run, or read the card's own SMI"
    else
      ok "$row"
    fi
  }

  for vendor in $DOCTOR_GPU_VENDORS; do
    case $vendor in
      nvidia)
        # nvidia-smi first when present: it is the only one of these that also
        # reports the compute capability, which is CUDA_ARCH in config.sh.
        # The container runtime injects it under --gpus, so this is the usual
        # path on both sides.
        nv_listed=0
        if command -v nvidia-smi >/dev/null 2>&1; then
          while IFS=, read -r nv_name nv_cap; do
            nv_name=${nv_name# }; nv_cap=${nv_cap# }
            [ -n "$nv_name" ] || continue
            ok "nvidia: $nv_name (compute ${nv_cap} -> CUDA_ARCH=${nv_cap//./})"
            nv_listed=$((nv_listed+1))
          done < <(nvidia-smi --query-gpu=name,compute_cap \
                     --format=csv,noheader 2>/dev/null)
        fi
        if [ "$nv_listed" -eq 0 ]; then
          # No nvidia-smi (or it failed): fall back to the driver's own procfs,
          # which names the card but not its compute capability.
          for info in /proc/driver/nvidia/gpus/*/information; do
            [ -r "$info" ] || continue
            nv_name=$(sed -n 's/^Model: *//p' "$info" | head -1)
            [ -n "$nv_name" ] || continue
            ok "nvidia: $nv_name"
            note "nvidia-smi absent -- compute capability not read; set CUDA_ARCH"
            nv_listed=$((nv_listed+1))
          done
        fi
        if [ "$nv_listed" -gt 0 ]; then
          gpus_found=$((gpus_found+nv_listed))
          IFS=', ' read -r nv_util nv_free nv_total < <(nvidia-smi \
            --query-gpu=utilization.gpu,memory.free,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | head -1)
          card_state nvidia cuda "${nv_util:-}" "${nv_free:-}" "${nv_total:-}"
          find_tool nvcc >/dev/null 2>&1 ||
            note "card present but nvcc is not -- the CUDA toolchain is in the \`cuda\` image"
          counter_tool_state cuda
        else
          note "no nvidia device -- nvcc still compiles, but no kernel can run"
        fi
        ;;
      amd)
        # /sys/class/kfd is amdkfd's own topology, the same source rocminfo
        # reads, and it is visible inside a container given --device=/dev/kfd.
        # Node 0 is the CPU node and reports gfx_target_version 0; skip it.
        #
        # This is the rule lib.sh's amd_gpu_present now carries for callers
        # that only need the yes/no (kernel-tier.sh, test/conftest.py -- #431).
        # Kept as a loop here rather than delegated to it because this report
        # wants each node's ISA, PCI id and name, not just whether one exists.
        amd_listed=0
        for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
          [ -r "$props" ] || continue
          gfxver=$(sed -n 's/^gfx_target_version //p' "$props" | head -1)
          # shellcheck disable=SC2015  # not if-then-else, and that is correct:
          # skip when gfx_target_version is missing OR zero
          [ -n "$gfxver" ] && [ "$gfxver" != 0 ] || continue
          nodedir=${props%/properties}
          isa=$(amd_gfx_name "$gfxver")
          # The PCI ids, not the node's `name`: amdkfd fills that with the
          # generic "ip discovery" on recent cards, and a marketing name is
          # exposed nowhere in sysfs (lspci's database may not know a new id
          # either). vendor/device are decimal in properties; print them hex,
          # which is how every lspci line and driver table spells them.
          amd_vid=$(sed -n 's/^vendor_id //p' "$props" | head -1)
          amd_did=$(sed -n 's/^device_id //p' "$props" | head -1)
          amd_pci=$(printf '%04x:%04x' "${amd_vid:-0}" "${amd_did:-0}")
          amd_name=$(tr -d '\0' < "$nodedir/name" 2>/dev/null)
          case $amd_name in ip\ discovery|'') amd_name= ;; *) amd_name=" $amd_name" ;; esac
          ok "amd:$amd_name $isa [$amd_pci] (-> GPU_TARGETS=$isa)"
          amd_listed=$((amd_listed+1))
        done
        if [ "$amd_listed" -gt 0 ]; then
          gpus_found=$((gpus_found+amd_listed))
          # Utilisation from rocm-smi, the source the timing sweep waits on
          # (sysfs gpu_busy_percent answers EBUSY on this box's RX 9060 XT);
          # VRAM from amdgpu's own sysfs, the first DRM card of vendor 0x1002.
          amd_util='' amd_free='' amd_total=''
          if command -v rocm-smi >/dev/null 2>&1 || [ -x /opt/rocm/bin/rocm-smi ]; then
            amd_util=$("$(command -v rocm-smi || echo /opt/rocm/bin/rocm-smi)" \
              --showuse 2>/dev/null | sed -n 's/.*GPU use (%):[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
          fi
          for amd_dev in /sys/class/drm/card[0-9]*/device; do
            [ "$(cat "$amd_dev/vendor" 2>/dev/null)" = 0x1002 ] || continue
            [ -r "$amd_dev/mem_info_vram_total" ] || continue
            [ -n "$amd_util" ] || amd_util=$(cat "$amd_dev/gpu_busy_percent" 2>/dev/null)
            amd_total=$(cat "$amd_dev/mem_info_vram_total" 2>/dev/null)
            amd_used=$(cat "$amd_dev/mem_info_vram_used" 2>/dev/null)
            if [[ $amd_total =~ ^[0-9]+$ && $amd_used =~ ^[0-9]+$ ]]; then
              amd_free=$(( (amd_total - amd_used) >> 20 )); amd_total=$(( amd_total >> 20 ))
            fi
            break
          done
          card_state amd hip "$amd_util" "$amd_free" "$amd_total"
          [ -e /dev/kfd ] ||
            note "/dev/kfd absent -- pass --device=/dev/kfd --device=/dev/dri to reach it"
          find_tool hipconfig >/dev/null 2>&1 ||
            note "card present but hipconfig is not -- ROCm is in the \`hip\`/\`combined\` image"
          counter_tool_state hip
        else
          note "no amd device -- hipcc still compiles, but no kernel can run"
          # WHICH no: the two causes have completely different fixes, and
          # before #431 neither was said out loud. `/dev/kfd` absent is a
          # passthrough (or module) question; `/dev/kfd` present with an empty
          # topology is a HOST driver question -- amdgpu loaded and bound
          # nothing -- and that is the one that got read as a broken
          # devcontainer mount for days.
          if [ ! -e /dev/kfd ]; then
            note "/dev/kfd absent -- in a container, pass --device=/dev/kfd --device=/dev/dri"
            note "(outside one, the amdgpu/amdkfd module is not loaded at all)"
          else
            warn "/dev/kfd is here but amdkfd lists no GPU node -- amdgpu bound no card"
            note "this is NOT container passthrough: the same emptiness is on the host"
            # The card the driver should have claimed. An AMD VGA/display
            # function with no `Kernel driver in use:` line is the signature.
            if command -v lspci >/dev/null 2>&1; then
              while IFS= read -r slot; do
                [ -n "$slot" ] || continue
                [ -e "/sys/bus/pci/devices/$slot/driver" ] && continue
                note "pci $slot is an unbound AMD display device -- nothing claimed it"
              done < <(lspci -Dn 2>/dev/null |
                       sed -n 's/^\([0-9a-f:.]*\) 03[0-9a-f][0-9a-f]: 1002:.*/\1/p')
            fi
            # The usual cause on this box: a kernel upgrade that NVIDIA's DKMS
            # module was rebuilt for and AMD's was not, so amdgpu silently
            # falls back to an in-tree one that does not know the card's PCI
            # id (6.8's does not know gfx1200's 1002:7590).
            if command -v dkms >/dev/null 2>&1; then
              amd_dkms=$(dkms status 2>/dev/null | sed -n 's/^\(amdgpu\/[^,]*\),.*/\1/p' | head -1)
              if [ -n "$amd_dkms" ] &&
                 ! dkms status 2>/dev/null | grep -q "^amdgpu/.*, $(uname -r),"; then
                warn "$amd_dkms is built for another kernel, not the running $(uname -r)"
                note "fix (host, needs root): sudo dkms autoinstall -k $(uname -r)"
                note "then reload: sudo modprobe -r amdgpu && sudo modprobe amdgpu (or reboot)"
              fi
            fi
          fi
        fi
        ;;
      *) note "unknown vendor '$vendor' in DOCTOR_GPU_VENDORS; skipping" ;;
    esac
  done

  if [ "$gpus_found" -eq 0 ]; then
    warn "no GPU of any declared vendor found -- every \`gpu\` test will SKIP"
    note "a green run here says the generator agrees with its golden files,"
    note "and says NOTHING about what a kernel computes; run the kernel tier"
    note "on a box with a card before trusting a numerical claim"
  fi
fi

# --- required paths (fixture trees, reference data) -----------------------
config_lines "$DOCTOR_REQUIRED_PATHS"
if [ "${#CONFIG_LINES[@]}" -gt 0 ]; then
  echo "required paths"
  for entry in "${CONFIG_LINES[@]}"; do
    rpath=${entry%%:*}
    what=${entry#*:}
    # Absolute entries are taken as-is, for a path that lives outside the
    # checkout -- something baked into the image, say. This project declares
    # none at all today (see config.sh), so both branches exist for the next
    # repo to copy this into. Relative ones stay repo-relative, so doctor works
    # from any worktree.
    case "$rpath" in /*) abs=$rpath ;; *) abs="$TOP/$rpath" ;; esac
    if [ -e "$abs" ]; then
      ok "$rpath present"
    else
      # The entry's own description carries the fix (config.sh writes it
       # there), so there is no generic suggestion to add here -- an earlier
       # copy of this script guessed "git submodule update", which this repo
       # has no submodule to run it on.
      warn "$rpath absent --${what:+ $what}"
    fi
  done
fi

# --- oracle (PySCF group, tracked fixtures) -------------------------------
# Self-contained. The `oracle` dependency group (PySCF) is deliberately NOT in
# `dev`, so its ABSENCE is the normal state: the fast tier compares against
# the fixtures in test/oracle/fixtures/ with numpy alone, and only the
# `oracle`-marked drift test (which regenerates them) skips. So absent is ok,
# not a warn; a fixture that will not LOAD is a warn, because the fast tier
# then fails rather than skips.
if [ -d "$TOP/test/oracle" ]; then
  echo "oracle"
  opy=""
  if ovenv=$(find_venv); then opy="$ovenv/bin/python"; fi
  if [ -z "$opy" ]; then
    warn "no venv -- cannot check the oracle group or load its fixtures"
  else
    if pv=$("$opy" -c "import pyscf; print(pyscf.__version__)" 2>/dev/null); then
      ok "oracle group installed (pyscf $pv) -- 'pytest -m oracle' can regenerate"
    else
      ok "oracle group not installed (by design) -- 'oracle' tests skip"
      note "to regenerate or drift-check fixtures: uv sync --group oracle"
    fi
    if out=$(PYTHONPATH="$TOP/test" "$opy" -m oracle.manifest 2>&1); then
      while IFS= read -r line; do ok "fixture $line"; done <<<"$out"
    else
      warn "oracle fixtures do not load -- the fast tier will fail"
      while IFS= read -r line; do note "$line"; done <<<"$out"
    fi
  fi
fi

# --- pre-commit hook (the shared-hook footgun) ----------------------------
echo "pre-commit hook"
# The hook lives in the *common* git dir, shared by the main repo and every
# worktree, and bakes in an absolute INSTALL_PYTHON. Installed from the wrong
# side (container vs host) it points at a python that does not exist here and
# every commit fails with "No module named 'pre_commit'". Test that exact path
# -- the same check the hook itself does -- rather than guessing which side we
# are on, so a hybrid environment does not produce a false alarm.
HOOK="$COMMON/hooks/pre-commit"
if [ ! -e "$HOOK" ]; then
  warn "no pre-commit hook installed -- run 'pre-commit install' on your commit side"
else
  ip=$(sed -n 's/^INSTALL_PYTHON=//p' "$HOOK" | head -1)
  if [ -n "$ip" ] && [ -x "$ip" ]; then
    ok "pre-commit hook usable (baked python)"
  elif command -v pre-commit >/dev/null 2>&1; then
    ok "pre-commit hook usable (pre-commit on PATH; baked python absent here)"
  else
    warn "pre-commit hook unusable: baked python not executable and no pre-commit on PATH"
    note "commits fail with 'No module named pre_commit'; fix: pre-commit install -f (from this side)"
  fi
fi

# --- claude code pin (the agent the image ships) ---------------------------
# The GPU Dockerfiles install Claude Code at an explicit ARG
# CLAUDE_CODE_VERSION -- see docker/install-claude-code.sh for the drift that
# replaced: a devcontainer FEATURE npm-installs an unpinned version, its layer
# is a permanent cache hit, and `rebuild` recreates the container from the same
# image, so an image can sit ~90 releases behind with nothing reporting it.
# Three ways that can still go wrong, one check each. CLAUDE_PIN_FILES in
# config.sh is the list of files carrying the pin; empty drops the section.
CV="$TOP/devtools/claude-version.sh"
if [ -n "${CLAUDE_PIN_FILES:-}" ] && [ -x "$CV" ]; then
  echo "claude code pin"
  cv=$("$CV" --porcelain 2>/dev/null)
  cv_pinned=$(printf '%s\n' "$cv" | sed -n 's/^pinned //p')
  cv_files=$(printf '%s\n' "$cv" | sed -n 's/^files //p')
  cv_registry=$(printf '%s\n' "$cv" | sed -n 's/^registry //p')
  cv_installed=$(printf '%s\n' "$cv" | sed -n 's/^installed //p')

  # 1. the copies agree with each other.
  if [ -z "$cv_pinned" ] || [ "$cv_pinned" = "-" ]; then
    warn "the Claude Code pin disagrees across the Dockerfiles"
    note "$cv_files"
    note "fix: devtools/claude-version.sh --apply <version>"
  # 2. and with the registry. Behind is a WARN, never a FAIL: a pin is a
  #    deliberate choice, and the point is only that the gap is visible.
  elif [ -z "$cv_registry" ]; then
    ok "Claude Code pinned at $cv_pinned"
    note "npm registry unreachable from here -- cannot say whether it is current"
  elif [ "$cv_pinned" != "$cv_registry" ]; then
    warn "Claude Code pin $cv_pinned is behind the registry ($cv_registry)"
    note "bump: devtools/claude-version.sh --apply && docker/build.sh cuda"
  else
    ok "Claude Code pinned at $cv_pinned (current)"
  fi

  # 3. and, inside an image built from that pin, with what is on PATH.
  #    CLAUDE_CODE_VERSION in the environment is the image's own ENV, so its
  #    presence is what tells container from host.
  if [ -n "${CLAUDE_CODE_VERSION:-}" ]; then
    if [ "$cv_installed" != "$CLAUDE_CODE_VERSION" ]; then
      warn "claude on PATH is ${cv_installed:-none}, this image pinned $CLAUDE_CODE_VERSION"
      note "something npm-installed over the image's copy; rebuild the container"
    elif [ -n "$cv_pinned" ] && [ "$CLAUDE_CODE_VERSION" != "$cv_pinned" ]; then
      warn "this image was built at $CLAUDE_CODE_VERSION, the tree now pins $cv_pinned"
      note "rebuild: devtools/devcontainer.sh rebuild"
    else
      ok "claude in here is the pinned $CLAUDE_CODE_VERSION"
    fi
  else
    note "claude here is ${cv_installed:-not installed} (a host install, unrelated to the pin)"
  fi
fi

# --- antigravity cli pin (the other agent the image ships) -----------------
# The sibling of the section above, and the same three failure modes -- but the
# pin is a RELEASE plus a sha512, which adds one: the two can be rewritten out
# of step, leaving two images that install different bytes under one release
# id. agy-version.sh reports that as `sha-differs` in its file list, so the
# first check below covers it without a fourth branch.
#
# What the installed binary CANNOT tell us: `agy --version` prints 1.3.3 and
# never the build id, so it can only ever confirm the version half. The
# image's own ENV AGY_RELEASE carries the whole release, which is what lets the
# third check tell a stale image from a current one.
AV="$TOP/devtools/agy-version.sh"
if [ -n "${AGY_PIN_FILES:-}" ] && [ -x "$AV" ]; then
  echo "antigravity cli pin"
  av=$("$AV" --porcelain 2>/dev/null)
  av_pinned=$(printf '%s\n' "$av" | sed -n 's/^pinned //p')
  av_files=$(printf '%s\n' "$av" | sed -n 's/^files //p')
  av_upstream=$(printf '%s\n' "$av" | sed -n 's/^upstream //p')
  av_installed=$(printf '%s\n' "$av" | sed -n 's/^installed //p')

  # 1. the copies agree with each other -- release AND hash.
  if [ -z "$av_pinned" ] || [ "$av_pinned" = "-" ]; then
    warn "the Antigravity CLI pin disagrees across the Dockerfiles"
    note "$av_files"
    note "fix: devtools/agy-version.sh --apply <release>"
  # 2. and with upstream. Behind is a WARN, never a FAIL: a pin is a
  #    deliberate choice, and the point is only that the gap is visible.
  elif [ -z "$av_upstream" ] || [ "$av_upstream" = "-" ]; then
    ok "Antigravity CLI pinned at $av_pinned"
    note "the release manifest is unreachable from here -- cannot say whether it is current"
  elif [ "$av_pinned" != "$av_upstream" ]; then
    warn "Antigravity CLI pin $av_pinned is behind upstream ($av_upstream)"
    note "bump: devtools/agy-version.sh --apply && docker/build.sh cuda"
  else
    ok "Antigravity CLI pinned at $av_pinned (current)"
  fi

  # 3. and, inside an image built from that pin, with what that image recorded.
  #    AGY_RELEASE in the environment is the image's own ENV, so its presence
  #    is what tells container from host.
  if [ -n "${AGY_RELEASE:-}" ]; then
    if [ -n "$av_pinned" ] && [ "$av_pinned" != "-" ] && [ "$AGY_RELEASE" != "$av_pinned" ]; then
      warn "this image was built at $AGY_RELEASE, the tree now pins $av_pinned"
      note "rebuild: devtools/devcontainer.sh rebuild"
    elif [ -z "$av_installed" ]; then
      warn "this image pinned $AGY_RELEASE but no agy is on PATH"
      note "install-agy.sh checks that at build time; a missing binary means the image predates the pin"
    elif [ "$av_installed" != "${AGY_RELEASE%%-*}" ]; then
      warn "agy on PATH reports $av_installed, this image pinned ${AGY_RELEASE%%-*} ($AGY_RELEASE)"
      note "something installed over the image's copy; rebuild the container"
    else
      ok "agy in here is the pinned $AGY_RELEASE"
    fi
  else
    note "agy here is ${av_installed:-not installed} (a host install, unrelated to the pin)"
  fi
fi

# --- docker hygiene (report-only) -----------------------------------------
# Worktree containers/volumes/images leak when a worktree is removed with plain
# `git worktree remove` (bypassing worktree.sh rm), or via an agent's worktrees
# under .claude/worktrees/. Surface the count instead of letting it accrete.
# The enumeration lives in `worktree.sh gc` -- called in --dry-run, which
# neither prunes nor deletes, so there is one source of truth.
echo "docker hygiene"
if command -v docker >/dev/null 2>&1; then
  wt="$TOP/devtools/worktree.sh"
  if [ -x "$wt" ]; then
    n=$("$wt" gc --dry-run 2>/dev/null \
        | sed -n 's/^orphans: \([0-9]*\).*/\1/p')
    if [ "${n:-0}" -gt 0 ]; then
      warn "$n orphaned docker resource(s) from removed worktrees"
      note "review and remove: devtools/worktree.sh gc"
    else
      ok "no orphaned containers/volumes/images"
    fi
  else
    note "devtools/worktree.sh not executable -- skipping orphan check"
  fi
else
  note "docker not found here (expected inside the container; run on the host)"
fi

# --- summary --------------------------------------------------------------
echo
if [ "$fails" -gt 0 ]; then
  printf '\033[31m%d FAIL\033[0m, %d warn\n' "$fails" "$warns"
  exit 1
elif [ "$warns" -gt 0 ]; then
  printf 'no failures, \033[33m%d warn\033[0m (some tests will skip)\n' "$warns"
else
  printf '\033[32mall good\033[0m\n'
fi
