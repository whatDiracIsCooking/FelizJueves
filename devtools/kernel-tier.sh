#!/usr/bin/env bash
# Generate the kernels, compile them with the real device compiler, run them on
# a real card, and report what each phase actually proved.
#
#   devtools/kernel-tier.sh [--backend cuda|hip] [--full] [--fresh]
#                           [--no-cache] [--no-test] [-j N] [--report DIR]
#                           [-- <pytest args>]
#
# THIS IS THE ONLY PLACE A NUMERICAL CLAIM CAN BE MADE. No hosted CI runner has
# a GPU, so CI can lint the generator and diff it against its golden files and
# nothing more; whether a generated kernel computes the right integrals is
# answered here, on a box with a card, or not at all. Run it before opening a
# PR that touches generation or kernel text, and say which backend you ran.
#
# FOUR PHASES, and the report says which of them did anything:
#
#   preflight  is the device compiler here, and is a card here? They fail
#              independently: nvcc and hipcc target an ARCHITECTURE, so the
#              compile phase is perfectly happy on a device-less box -- and a
#              `gpu`-marked test SKIPS there rather than failing, which is how
#              a green, empty run happens.
#   generate   KERNEL_GEN_CMD (devtools/config.sh) emits sources into
#              $KERNEL_TIER_OUT, a per-backend directory under
#              KERNEL_BUILD_DIR that is emptied first: by default the J
#              chain's 37 units plus the one-electron ladder through (dd)
#              (`--one-e-l 2`: s_, t_, v_, 18 units, #242). DORMANT while the
#              command is empty, and this script says so instead of implying
#              a tier ran.
#   compile    every translation unit the generate phase wrote, through the
#              backend's compiler, in parallel, through sccache when it is
#              available. A WARNING located in a generated unit fails the
#              phase (exit 1, reported as COMPILE WARNINGS and distinct from
#              COMPILE FAILED), naming the kernel and the line; one in a
#              vendor header is reported and does not (config.sh,
#              NVCC_FLAGS / HIPCC_FLAGS).
#   device    the `gpu`-marked pytest suite (KERNEL_TEST_ARGS: not `slow`, the
#              PR gate, under five minutes per card; --full takes
#              KERNEL_FULL_TEST_ARGS, every `gpu` test with L <= 8, both
#              excluding `high_l`, test/tiers.py), which is what
#              compares a kernel's output against the host reference (the
#              one-electron units: test/test_one_electron_device.py, which
#              includes the very files the generate phase wrote). Its
#              own compiles go through the same sccache (test/devcache.py,
#              #186), from a content-addressed directory under
#              KERNEL_BUILD_DIR, so a warm re-run recompiles only what
#              changed.
#
# THE CACHE. --no-cache drops sccache from the compile phase AND exports
# KERNEL_NO_CACHE (config.sh) so the device suite compiles cold too; setting
# that variable does the same for this script or a bare pytest. --fresh
# removes KERNEL_BUILD_DIR, the test-side build directory with it, never the
# sccache store. The result block reports the cache's hits and misses for the
# compile phase and for the device phase, and the device phase's wall-clock.
#
# WHAT THE GENERATOR AND THE DEVICE SUITE ARE TOLD, as environment variables
# (the generator reads them as its defaults, the suite to find what was
# generated):
#
#   KERNEL_TIER_BACKEND  cuda | hip
#   KERNEL_TIER_ARCH     the arch the compiler is driven at (sm_86, gfx1200)
#   KERNEL_TIER_OUT      where the generator must write, and the only place
#                        the compile phase looks
#
# KERNEL_GEN_CMD runs from the repo root with the project venv's bin/ first on
# PATH (the first of VENV_PATHS that runs), so `python` there is the project's.
# A `{backend}` in KERNEL_TEST_ARGS is replaced by the backend, so the device
# phase can select only that backend's tests.
#
# It is NOT a git hook. An unrolled high-angular-momentum kernel is slow to
# compile and this tier costs minutes; a gate that costs minutes is one people
# learn to bypass with --no-verify. .pre-commit-config.yaml gates the seconds-
# fast pytest tier instead.
#:
# --- for a reader of the file ----------------------------------------------
#
# ON THE HOST WITH NO hipcc, `--backend hip` re-execs this same script inside
# ROCM_IMAGE with the device nodes and the resolved host group ids passed in
# (config.sh: ROCM_DEVICES, ROCM_GROUPS, ROCM_IMAGE). That passthrough is
# fiddly and its failure mode -- hipErrorNoDevice -- is indistinguishable from
# having no card, which is exactly why it lives in a script rather than in
# anyone's shell history. FJ_IN_CONTAINER=1 is set across that boundary so the
# inner run cannot recurse.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

backend=$KERNEL_BACKEND
fresh=
no_cache=$KERNEL_NO_CACHE
run_tests=1
jobs=${JOBS:-}
report=$KERNEL_TIER_REPORTS
pytest_extra=()
test_args=$KERNEL_TEST_ARGS

while [ $# -gt 0 ]; do
  case $1 in
    --backend) backend=${2:?--backend needs cuda or hip}; shift 2 ;;
    --cuda)    backend=cuda; shift ;;
    --hip)     backend=hip;  shift ;;
    --full)    test_args=$KERNEL_FULL_TEST_ARGS; shift ;;
    --fresh)   fresh=1; shift ;;
    --no-cache) no_cache=1; shift ;;
    --no-test) run_tests=0; shift ;;
    -j)        jobs=${2:?-j needs a number}; shift 2 ;;
    --report)  report=${2:?--report needs a directory}; shift 2 ;;
    --)        shift; pytest_extra=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "kernel-tier: unknown argument '$1'" >&2; usage >&2; exit 64 ;;
  esac
done

case $backend in
  cuda)
    compiler=$NVCC
    config_args "$NVCC_FLAGS"; compiler_flags=("${CONFIG_ARGS[@]}" "-arch=sm_${CUDA_ARCH}")
    source_exts=(cu)
    target_desc="sm_${CUDA_ARCH}"
    ;;
  hip)
    compiler=$HIPCC
    config_args "$HIPCC_FLAGS"
    # `-x hip` AND BOTH EXTENSIONS, deliberately. The point of this project is
    # that ONE generated source satisfies both device compilers, so the HIP
    # side has to be able to compile a `.cu` -- globbing `*.hip` alone would
    # silently find nothing and report a green, empty compile phase for a
    # generator that emits the shared spelling. hipcc needs to be told the
    # language explicitly for a `.cu`, and `-x hip` is harmless for a `.hip`.
    compiler_flags=("${CONFIG_ARGS[@]}" "--offload-arch=${HIP_ARCH}" -x hip)
    source_exts=(hip cu)
    target_desc="$HIP_ARCH"
    ;;
  *) echo "kernel-tier: --backend must be cuda or hip, not '$backend'" >&2; exit 64 ;;
esac

ts=$(date +%Y%m%d-%H%M%S)
case "$report" in /*) report_dir="$report/$backend-$ts" ;; *) report_dir="$REPO_ROOT/$report/$backend-$ts" ;; esac
mkdir -p "$report_dir"

phase() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note()  { printf '   %s\n' "$1"; }
# Exit 2 is "could not run", never "passed". Nothing may read a 2 as a green
# tier -- that distinction is the whole reason this script has a preflight.
cannot() { echo "kernel-tier: cannot run: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
phase "preflight ($backend, target $target_desc)"

if ! compiler_path=$(find_tool "$compiler"); then
  # The one case worth rescuing automatically: this host has the AMD card but
  # not the ROCm toolchain (it lives in the image), and the container knows how
  # to do the whole job.
  if [ "$backend" = hip ] && [ -z "${FJ_IN_CONTAINER:-}" ] && command -v docker >/dev/null 2>&1; then
    if ! docker image inspect "$ROCM_IMAGE" >/dev/null 2>&1; then
      cannot "no $compiler here and no $ROCM_IMAGE image. Build it:
                 docker/build.sh hip
               (or set ROCM_IMAGE in devtools/config.sh to one you have)"
    fi
    note "no $compiler on this host -- re-running inside $ROCM_IMAGE"
    docker_args=(--rm -v "$REPO_ROOT:/workspace" -w /workspace
                 -u "$(id -u):$(id -g)" -e FJ_IN_CONTAINER=1)
    while IFS= read -r dev; do
      [ -n "$dev" ] || continue
      [ -e "$dev" ] || cannot "device node $dev does not exist on this host"
      docker_args+=(--device "$dev")
    done <<<"$ROCM_DEVICES"
    while IFS= read -r grp; do
      [ -n "$grp" ] || continue
      gid=$(host_gid "$grp")
      # A NAME here would be the silent wrong answer: it resolves inside the
      # container, where the gid differs from the one owning the bind-mounted
      # nodes. See config.sh (ROCM_GROUPS).
      [ -n "$gid" ] || cannot "no '$grp' group on this host; fix ROCM_GROUPS in devtools/config.sh"
      docker_args+=(--group-add "$gid")
    done <<<"$ROCM_GROUPS"
    # Built as an ARRAY, not interpolated: `$(... && echo --no-test)` inside a
    # command line is an unquoted expansion, and an empty one contributes a
    # word on some shells -- which docker then reads as the image's command.
    inner=(devtools/kernel-tier.sh --backend hip)
    [ -z "$fresh" ]    || inner+=(--fresh)
    [ -z "$no_cache" ] || inner+=(--no-cache)
    [ -z "$jobs" ]     || inner+=(-j "$jobs")
    [ "$run_tests" = 1 ] || inner+=(--no-test)
    [ "${#pytest_extra[@]}" -eq 0 ] || inner+=(-- "${pytest_extra[@]}")
    exec docker run "${docker_args[@]}" "$ROCM_IMAGE" "${inner[@]}"
  fi
  cannot "$compiler not found (searched $VENV_PATHS and PATH).
                 The device compilers live in the images:
                   docker/build.sh cuda   &&  devtools/devcontainer.sh shell
                   docker/build.sh hip    &&  devtools/devcontainer.sh --hip shell"
fi
note "compiler: $compiler_path"
# NOT `--version | head -1`: head exits after one line, the compiler keeps
# writing, and SIGPIPE kills it -- which under `set -o pipefail` + `set -e`
# aborts this script with 141 before the preflight finishes. nvcc hides the bug
# (its banner fits the pipe buffer) and hipcc does not. Read it all, then take
# the first line in the shell.
compiler_version=$("$compiler_path" --version 2>&1 || true)
note "${compiler_version%%$'\n'*}"

# The card, probed through the kernel rather than a vendor tool so the answer
# is the same on the host and inside a container.
#
# The AMD side asks amd_gpu_present (lib.sh) for an enumerated GPU node and
# NOT `[ -e /dev/kfd ]`, which is what it asked before #431: that node is
# there whenever the module is loaded, bound card or not, so this printed
# `device: present` on a box whose amdgpu had claimed nothing. lib.sh carries
# the why; the extra line below is for the case that actually happens, because
# "ABSENT" alone sends you looking at the container's passthrough when the
# answer is on the host.
device_present=0
case $backend in
  cuda) [ -e /dev/nvidiactl ] || [ -d /proc/driver/nvidia/gpus ] && device_present=1 ;;
  hip)  amd_gpu_present && device_present=1 ;;
esac
if [ "$device_present" = 1 ]; then
  note "device: present"
else
  note "device: ABSENT -- the compile phase still runs and proves nothing about"
  note "        a number; every \`gpu\` test will SKIP, not fail."
  if [ "$backend" = hip ] && [ -e /dev/kfd ]; then
    note "        /dev/kfd IS here but kfd lists no GPU node, so this is NOT a"
    note "        container passthrough problem: amdgpu bound no card on the"
    note "        HOST. Usual cause is a kernel upgrade the DKMS module was"
    note "        not rebuilt for -- 'dkms status' and devtools/doctor.sh say."
  fi
fi

PYTEST=$(find_pytest) || cannot "no pytest in $VENV_PATHS or on PATH -- run 'uv sync'"

# sccache as a compiler launcher. Wired in HERE rather than baked into the
# compiler flags so an image or host without it still works, just colder --
# which is the same contract docker/Dockerfile.base's comment describes.
#
# The device suite's own compiles (test/devcache.py, #186) resolve the same
# launcher and read the same off switch: --no-cache exports KERNEL_NO_CACHE
# (config.sh), so the tests run cold too.
launcher=()
if [ -z "$no_cache" ] && sccache_path=$(find_tool sccache); then
  launcher=("$sccache_path")
  note "cache: $sccache_path (SCCACHE_DIR=${SCCACHE_DIR:-unset})"
else
  note "cache: none -- every translation unit is a cold compile"
fi
if [ -n "$no_cache" ]; then
  export KERNEL_NO_CACHE=1
fi

# The server's cumulative hit and miss counts, as `hits misses`, every
# language summed (nvcc counts each of its stages). Read before and after a
# phase, and the difference is that phase's -- not a --zero-stats, which would
# erase the counts of a concurrent run on the other backend sharing the server.
# Prints nothing without a launcher or when the server cannot be asked.
cache_counts() {
  [ "${#launcher[@]}" -gt 0 ] || return 0
  "${launcher[0]}" --show-stats --stats-format=json 2>/dev/null \
    | python3 -c 'import json, sys
s = json.load(sys.stdin)["stats"]
print(sum(s["cache_hits"]["counts"].values()), sum(s["cache_misses"]["counts"].values()))' \
    2>/dev/null || true
}
# `<label> <before> <after>` -> one report line with the difference.
cache_delta() {
  local label=$1 b=$2 a=$3
  if [ -z "$b" ] || [ -z "$a" ]; then
    note "$label  -- (no cache stats: $([ "${#launcher[@]}" -gt 0 ] && echo 'server not readable' || echo 'no launcher'))"
    return
  fi
  local bh bm ah am
  read -r bh bm <<<"$b"; read -r ah am <<<"$a"
  note "$label  hits $((ah - bh)) misses $((am - bm))"
}

# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
phase "generate"
build_dir=$KERNEL_BUILD_DIR
case "$build_dir" in /*) : ;; *) build_dir="$REPO_ROOT/$build_dir" ;; esac
[ -z "$fresh" ] || { note "--fresh: removing $build_dir"; rm -rf "$build_dir"; }
mkdir -p "$build_dir"
# Per backend, and emptied every run: the compile phase compiles exactly what
# THIS run generated -- never a source left by the other backend's spelling or
# by an earlier generator.
src_dir="$build_dir/src-$backend"
rm -rf "$src_dir"
mkdir -p "$src_dir"
export KERNEL_TIER_BACKEND="$backend" KERNEL_TIER_ARCH="$target_desc" KERNEL_TIER_OUT="$src_dir"

if [ -z "$KERNEL_GEN_CMD" ]; then
  note "DORMANT: KERNEL_GEN_CMD is empty in devtools/config.sh, so nothing was"
  note "generated and the compile phase below has nothing to compile. Do not"
  note "read the result of this run as a green kernel tier."
else
  gen_path=$PATH
  venv=$(find_venv) && gen_path="$venv/bin:$PATH"
  note "+ $KERNEL_GEN_CMD"
  note "  -> $src_dir (backend $backend, arch $target_desc)"
  set +e
  # shellcheck disable=SC2086  # deliberate word splitting: it is a command line
  PATH=$gen_path $KERNEL_GEN_CMD 2>&1 | tee "$report_dir/generate.log"
  gen_rc=${PIPESTATUS[0]}
  set -e
  # A generator's own 2 keeps its meaning: it could not run (bad arguments,
  # an arch it does not know), which is not a failed kernel either.
  [ "$gen_rc" != 2 ] || cannot "the generator could not run (rc=2); see $report_dir/generate.log"
  if [ "$gen_rc" != 0 ]; then
    note "GENERATE FAILED (rc=$gen_rc): nothing is compiled or run from a"
    note "generator that did not finish."
    note "report    ${report_dir#"$REPO_ROOT"/}"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
phase "compile"
find_args=()
for ext in "${source_exts[@]}"; do
  [ "${#find_args[@]}" -eq 0 ] || find_args+=(-o)
  find_args+=(-name "*.$ext")
done
mapfile -t sources < <(find "$src_dir" -type f \( "${find_args[@]}" \) | sort)
exts_desc=$(printf '*.%s ' "${source_exts[@]}")

if [ "${#sources[@]}" -eq 0 ]; then
  note "no ${exts_desc% } under $src_dir -- skipped"
  # A generator that RAN and wrote nothing is the green-and-empty failure this
  # tier exists to refuse, not a dormant phase.
  if [ -n "$KERNEL_GEN_CMD" ]; then
    note "GENERATE PRODUCED NOTHING: KERNEL_GEN_CMD ran but wrote no ${exts_desc% }"
    note "under KERNEL_TIER_OUT -- nothing to compile, so nothing proven."
    note "report    ${report_dir#"$REPO_ROOT"/}"
    exit 1
  fi
else
  obj_dir="$build_dir/obj-$backend"
  # Emptied, so `objects` counts this run's compile and not a stale one.
  rm -rf "$obj_dir"
  mkdir -p "$obj_dir"
  note "${#sources[@]} translation unit(s) -> $obj_dir, ${jobs:-1} at a time"
  # Objects rather than executables: this phase proves each generated unit
  # survives the real compiler with the configured flags; the device suite
  # builds its own harness around the same text. One file per ERI class is the
  # unit sccache can hit on. Compiling in parallel with xargs rather than a build
  # system because there is nothing to express here that a dependency graph
  # would capture -- each unit is independent by construction.
  #
  # Each unit's diagnostics go to its OWN log first ($OBJ_DIR/<unit>.log) and
  # are printed as one block after it finishes: parallel compilers writing to
  # one pipe interleave mid-line, and the warning scan below reads per unit.
  cc_cmp0=$(cache_counts)
  set +e
  # shellcheck disable=SC2016  # the single quotes are the point: $1/$OBJ_DIR
  # are expanded by the xargs CHILD, not here
  printf '%s\n' "${sources[@]}" \
    | xargs -P "${jobs:-1}" -I{} -- \
        env OBJ_DIR="$obj_dir" bash -c \
        'src="$1"; shift; stem="$OBJ_DIR/$(basename "${src%.*}")";
         "$@" -c "$src" -o "$stem.o" >"$stem.log" 2>&1; rc=$?;
         { echo "   cc $(basename "$src")"; cat "$stem.log"; } ; exit "$rc"' \
        _ {} "${launcher[@]}" "$compiler_path" "${compiler_flags[@]}" \
    2>&1 | tee "$report_dir/compile.log"
  compile_rc=${PIPESTATUS[1]}
  set -e
  cc_cmp1=$(cache_counts)
  objects=$(find "$obj_dir" -name '*.o' | wc -l)
  note "objects: $objects"

  # SAY IT, rather than letting `set -e` kill the run mid-phase. A failed
  # compile used to abort here with no summary line, which in a tier whose
  # whole job is to report what it proved is the wrong failure: the reader sees
  # a compiler diagnostic and no verdict. xargs exits 123 when any invocation
  # failed.
  if [ "$compile_rc" != 0 ]; then
    note "COMPILE FAILED (xargs rc=$compile_rc): ${#sources[@]} source(s), $objects object(s)"
    note "the device suite is NOT run -- a kernel that does not compile cannot"
    note "be tested, and a pass from a stale object would be a lie."
    note "report    ${report_dir#"$REPO_ROOT"/}"
    exit 1
  fi

  # A WARNING IN GENERATED TEXT FAILS THE TIER (#92). The flags that make the
  # compilers say it are NVCC_FLAGS / HIPCC_FLAGS in config.sh; this is the
  # verdict. It is a FILTER BY FILE, not -Werror: a diagnostic counts against
  # the generator only when its location is the generated unit itself, so a
  # warning inside a vendor header, or one with no source location at all (a
  # ptxas/linker remark), is reported here and never fails the run -- the
  # generator cannot fix it. Both spellings are matched: nvcc's front end
  # writes `file(LINE): warning #N-D: ...`, clang and gcc write
  # `file:LINE:COL: warning: ...`. hipcc compiles each unit for the device and
  # for the host and says the same thing twice; duplicates are dropped.
  #
  # Exit 1, like a failed compile: the generator produced bad text. Exit 2
  # keeps meaning "could not run" and nothing here can produce one.
  gen_warnings=()
  other_warnings=0
  for src in "${sources[@]}"; do
    kernel=$(basename "${src%.*}")
    log="$obj_dir/$kernel.log"
    [ -s "$log" ] || continue
    # One awk pass per unit: `G <text>` for a warning located in the generated
    # unit, `O` for any other warning diagnostic (a header, a toolchain stage).
    while IFS= read -r hit; do
      case $hit in
        G\ *) gen_warnings+=("${hit#G }") ;;
        O)    other_warnings=$((other_warnings + 1)) ;;
      esac
    done < <(awk -v f="$src" -v k="$kernel" '
      index($0, f) == 1 {
        rest = substr($0, length(f) + 1)
        if (rest ~ /^\([0-9]+\): warning/ || rest ~ /^:[0-9]+:([0-9]+:)? warning:/) {
          line = rest; sub(/^[(:]/, "", line); sub(/[^0-9].*/, "", line)
          msg = rest; sub(/^[^ ]* warning:? */, "", msg)
          out = "kernel " k ", line " line ": " msg
          if (!(out in seen)) { seen[out] = 1; print "G " out }
        }
        next
      }
      /(\([0-9]+\)|:[0-9]+(:[0-9]+)?): warning/ || /^(ptxas|nvlink|ld\.lld|lld)[^:]*warning/ {
        if (!($0 in seen)) { seen[$0] = 1; print "O" }
      }' "$log")
  done

  if [ "${#gen_warnings[@]}" -gt 0 ]; then
    printf '%s\n' "${gen_warnings[@]}" > "$report_dir/warnings.log"
    note "COMPILE WARNINGS: ${#gen_warnings[@]} warning(s) in generated text --"
    note "every unit compiled (no error), but a warning in emitted text is a"
    note "generator bug and fails the tier. Fix the emitter; do not relax a flag."
    for w in "${gen_warnings[@]}"; do note "  $w"; done
    note "the device suite is NOT run."
    note "report    ${report_dir#"$REPO_ROOT"/}"
    exit 1
  fi
  note "warnings: none in generated text"
  if [ "$other_warnings" -gt 0 ]; then
    note "          ($other_warnings warning(s) OUTSIDE it -- a vendor header or a"
    note "          toolchain stage; reported, not failing: see compile.log)"
  fi
fi

# ---------------------------------------------------------------------------
# device suite
# ---------------------------------------------------------------------------
phase "device suite"
if [ "$run_tests" = 0 ]; then
  note "--no-test: skipped"
else
  xdist_args
  config_args "${test_args//\{backend\}/$backend}"
  cc_dev0=$(cache_counts)
  t_dev0=$(date +%s.%N)
  set +e
  "$PYTEST" -rs "${XDIST[@]}" "${CONFIG_ARGS[@]}" "${pytest_extra[@]}" \
    --junitxml="$report_dir/pytest.xml" 2>&1 | tee "$report_dir/pytest.log"
  rc=${PIPESTATUS[0]}
  set -e
  t_dev1=$(date +%s.%N)
  cc_dev1=$(cache_counts)
fi

phase "result"
note "backend   $backend ($target_desc)"
note "device    $([ "$device_present" = 1 ] && echo present || echo ABSENT -- numbers unproven)"
note "generated $([ -n "$KERNEL_GEN_CMD" ] && echo "$KERNEL_GEN_CMD" || echo "nothing (KERNEL_GEN_CMD empty)")"
note "compiled  ${#sources[@]} translation unit(s) from ${src_dir#"$REPO_ROOT"/}"
note "tests     ${test_args//\{backend\}/$backend}$([ "$run_tests" = 1 ] || echo ' (NOT RUN: --no-test)')"
# What the compiler cache did, per phase (#186): the tier's own units, and
# every compile the device suite issued (test/devcache.py). Counts from the
# shared sccache server, so a concurrent run's compiles would land here too.
note "cache     $([ "${#launcher[@]}" -gt 0 ] && echo "${launcher[0]}" || echo "none (cold)")"
[ -z "${cc_cmp0+x}" ] || cache_delta "  compile" "$cc_cmp0" "$cc_cmp1"
[ -z "${cc_dev0+x}" ] || cache_delta "  device " "$cc_dev0" "$cc_dev1"
[ -z "${t_dev0:-}" ] || note "device wall-clock $(awk -v a="$t_dev0" -v b="$t_dev1" 'BEGIN { printf "%.1f s", b - a }')"
note "report    ${report_dir#"$REPO_ROOT"/}"
# NO CARD IS "COULD NOT RUN", not a pass: the device suite above ran so its
# skips are on the record, but every `gpu` test in it skipped, and a green,
# empty device phase is exactly what this tier exists to refuse. (--no-test
# asked for the compile check only, and keeps its own exit status.)
if [ "$device_present" = 0 ] && [ "$run_tests" = 1 ] && [ "${rc:-0}" = 0 ]; then
  note "NOT A PASS: device ABSENT, so every gpu test skipped -- exit 2"
  exit 2
fi
exit "${rc:-0}"
