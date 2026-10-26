#!/usr/bin/env bash
# Measure: time, count, recompile and re-derive -- everything the test
# suite does NOT do. test/ checks correctness only; what measures, or reads
# back what was measured, lives in measure/ and runs from here.
#
#   devtools/measure.sh <sweep> [--backend cuda|hip] [--full] [-- <pytest args>]
#   devtools/measure.sh list
#   devtools/measure.sh remeasure [<python -m felizjueves.performance.measured args>]
#
# THE SWEEPS (`list` prints them), by what each needs:
#
#   card + compiler, TIMED or PROFILED, run alone (MEASURE_TIMED_ARGS: no
#   xdist), each writing its store -- the variable defaults to the tracked
#   file (config.sh, MEASURE_*_STORE) unless already set:
#     wallclock     J subgrid wall-clock (#53; FJ_WALLCLOCK_BATCHES, _LEVER,
#                   _ASYM, _RESUME pass through)        FJ_WALLCLOCK_STORE
#     wallclock-k   K subgrid beside its J twins (#205) FJ_WALLCLOCK_STORE
#     wallclock-f   the fused F classes (#288, #290;
#                   FJ_WALLCLOCK_F_L selects by L)      FJ_WALLCLOCK_STORE
#     levers        every ledger arm against literal (#363;
#                   FJ_WALLCLOCK_ARMS restricts)         FJ_WALLCLOCK_STORE
#     builds        whole-molecule J build (#146)       FJ_LAUNCH_WALLCLOCK_STORE
#     sort-key      #147's sort-key A/B                 FJ_LAUNCH_WALLCLOCK_STORE
#     roofline      ncu counter rows (#112; sm_86 only) FJ_ROOFLINE_STORE
#     ceilings      the fp64 / triad ceilings (#109)    FJ_CEILINGS_STORE
#     kblock        the block K kernel against kernel_k,
#                   per class on H2O/cc-pVDZ (#481)     FJ_KBLOCK_STORE
#     scheme-ab     fused HGP F against MMD J + K, (ss|ss)
#                   and (pp|pp) on gly5/10/15 / 6-31G   FJ_SCHEME_AB_STORE
#                   (inputs: python -m oracle.polyglycine, PySCF)
#     kblock-split  the split block K kernel's registers and
#                   spill beside kb, (dd|dd) (#548; a
#                   compiler, no card)                  FJ_KBLOCK_SPLIT_STORE
#     kblock-rolled-compile  the rolled (kbr) and dense-Hermite
#                   (kbd) block K kernels' compile (source,
#                   wall, peak RSS, registers, spill,
#                   stack) at (dd|dd) and (ff|ff) beside
#                   kb (#571, #575; a compiler, no card) FJ_KBLOCK_ROLLED_STORE
#     kblock-rolled kb, kbr and kbd, (dd|dd) wall-clock,
#                   and kbr against kbd at (ff|ff)
#                   (#571, #575)                        FJ_KBLOCK_ROLLED_STORE
#     kblock-rolled-profile  kb_22_22 against kbr_22_22 under
#                   ncu, one launch each, and the three
#                   hypotheses read (#574; CUDA only)   FJ_KBLOCK_ROLLED_STORE
#     scatter-tail  F scatter write-back tail (#358), printed, no store
#     probes        the profilers' counter witnesses and the ceiling probe's
#                   published-peak bound; nothing written
#   compiler, no card:
#     recheck       recompile and compare against the stored rows
#   neither (fast; xdist):
#     check         the tracked stores and reports are what their rows
#                   recompute to, and the recorded hashes today's text
#
# --backend picks the card / compiler (default KERNEL_BACKEND); `check`
# ignores it. --full adds the `slow` cases (the rows and cells above
# tiers.FAST_L_MAX) to `check` and `recheck`. `-- <args>` reach pytest, e.g.
# `-- --collect-only -q` to see what a sweep would run without running it.
#
# EXIT: pytest's, except 2 = COULD NOT RUN -- no pytest, no compiler, no
# card, or nothing collected. A device sweep whose card is absent is refused
# up front rather than run as a green, empty session. A SKIP in a timed
# sweep's summary is not a measurement: read the -rs lines.
#:
# --- for a reader of the file ----------------------------------------------
#
# WHY NOT kernel-tier.sh. The kernel tier is the PR gate on a card, run with
# xdist beside its own compiles; a timing taken there is contended, and a
# gate that measures is a gate that costs an hour. This script runs one sweep
# per invocation with nothing beside it, behind the card lock and the card
# gate that measure/conftest.py takes from test/conftest.py. It does not
# re-exec into the ROCm image the way kernel-tier.sh does: on a host without
# hipcc, run it inside the hip devcontainer (devtools/devcontainer.sh --hip
# shell).
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }
cannot() { echo "measure: cannot run: $*" >&2; exit 2; }

# name|kind|description -- kind: timed (card, store), device (card, no
# store), compile (compiler), check (neither)
SWEEPS=(
  "wallclock|timed|J subgrid wall-clock and its lever/asym arms -> kernels.json"
  "wallclock-k|timed|K subgrid wall-clock beside each J twin -> kernels.json"
  "wallclock-f|timed|fused F wall-clock per class -> kernels.json"
  "levers|timed|every ledger arm against its literal class -> kernels.json"
  "builds|timed|whole-molecule J build wall-clock -> builds.json"
  "sort-key|timed|#147's sort-key A/B -> builds.json"
  "roofline|timed|ncu counter rows of every subgrid batch kernel -> roofline.json"
  "ceilings|timed|fp64 and triad ceilings behind the gate -> ceilings.json"
  "kblock|timed|block K kernel vs kernel_k per class (H2O/cc-pVDZ) -> kblock.json"
  "scheme-ab|timed|fused HGP F vs MMD J+K, (ss|ss) (pp|pp), polyglycine -> scheme_ab.json"
  "kblock-split|compile|split block K kernel's registers/spill vs kb, (dd|dd) -> kblock_split.json"
  "kblock-rolled-compile|compile|kbr and kbd block K kernels' compile at (dd|dd), (ff|ff) vs kb (#571, #575) -> kblock_rolled.json"
  "kblock-rolled|timed|kb, kbr, kbd at (dd|dd), kbr vs kbd at (ff|ff), wall-clock (#571, #575) -> kblock_rolled.json"
  "kblock-rolled-profile|timed|kb_22_22 vs kbr_22_22 under ncu, CUDA only (#574) -> kblock_rolled.json"
  "scatter-tail|device|F scatter write-back tail, printed only"
  "probes|device|profiler counter witnesses and the published-peak bound"
  "recheck|compile|recompile and compare against the stored rows (no card)"
  "check|check|tracked stores and reports recomputed (no compiler, no card)"
  "remeasure|driver|python -m felizjueves.performance.measured --remeasure <args>"
)

[ $# -gt 0 ] || { usage >&2; exit 64; }
sweep=$1; shift
case $sweep in
  -h|--help) usage; exit 0 ;;
  list)
    for s in "${SWEEPS[@]}"; do
      IFS='|' read -r name kind what <<<"$s"
      printf '  %-13s %-8s %s\n' "$name" "$kind" "$what"
    done
    exit 0 ;;
  remeasure)
    py=$(find_tool python) || cannot "no python in $VENV_PATHS or on PATH -- run 'uv sync'"
    exec env PYTHONPATH=src "$py" -m felizjueves.performance.measured --remeasure "$@" ;;
esac

kind=
for s in "${SWEEPS[@]}"; do
  IFS='|' read -r name k _ <<<"$s"
  [ "$name" = "$sweep" ] && kind=$k
done
[ -n "$kind" ] || { echo "measure: unknown sweep '$sweep' (try: list)" >&2; exit 64; }

backend=$KERNEL_BACKEND
full=
pytest_extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --backend) backend=${2:?--backend needs cuda or hip}; shift 2 ;;
    --cuda)    backend=cuda; shift ;;
    --hip)     backend=hip;  shift ;;
    --full)    full=1; shift ;;
    --)        shift; pytest_extra=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "measure: unknown argument '$1'" >&2; usage >&2; exit 64 ;;
  esac
done
case $backend in
  cuda) compiler=$NVCC ;;
  hip)  compiler=$HIPCC ;;
  *) echo "measure: --backend must be cuda or hip, not '$backend'" >&2; exit 64 ;;
esac

M=$MEASURE_DIR
files=() select=() envs=()
case $sweep in
  wallclock)    files=("$M/measure_kernel_device.py"); select=(-k wallclock_over_the_subgrid)
                envs=("FJ_WALLCLOCK_STORE=${FJ_WALLCLOCK_STORE:-$MEASURE_KERNELS_STORE}") ;;
  wallclock-k)  files=("$M/measure_kernel_device.py"); select=(-k k_wallclock)
                envs=("FJ_WALLCLOCK_STORE=${FJ_WALLCLOCK_STORE:-$MEASURE_KERNELS_STORE}"
                      "FJ_WALLCLOCK_K=1") ;;
  wallclock-f)  files=("$M/measure_fock_ondevice.py"); select=(-k f_wallclock)
                envs=("FJ_WALLCLOCK_STORE=${FJ_WALLCLOCK_STORE:-$MEASURE_KERNELS_STORE}"
                      "FJ_WALLCLOCK_F=1") ;;
  levers)       files=("$M/measure_lever_clock_device.py"); select=(-k lever_clock)
                envs=("FJ_WALLCLOCK_STORE=${FJ_WALLCLOCK_STORE:-$MEASURE_KERNELS_STORE}"
                      "FJ_WALLCLOCK_LEVERS=1") ;;
  builds)       files=("$M/measure_j_launched_ondevice.py"); select=(-k build_wallclock)
                envs=("FJ_LAUNCH_WALLCLOCK_STORE=${FJ_LAUNCH_WALLCLOCK_STORE:-$MEASURE_BUILDS_STORE}") ;;
  sort-key)     files=("$M/measure_j_launched_ondevice.py"); select=(-k sort_key_ab)
                envs=("FJ_LAUNCH_WALLCLOCK_STORE=${FJ_LAUNCH_WALLCLOCK_STORE:-$MEASURE_BUILDS_STORE}") ;;
  roofline)     files=("$M/measure_kernel_device.py"); select=(-k roofline_over_the_subgrid)
                envs=("FJ_ROOFLINE_STORE=${FJ_ROOFLINE_STORE:-$MEASURE_ROOFLINE_STORE}") ;;
  ceilings)     files=("$M/measure_ceilings_device.py"); select=(-k record_ceilings)
                envs=("FJ_CEILINGS_STORE=${FJ_CEILINGS_STORE:-$MEASURE_CEILINGS_STORE}") ;;
  kblock)       files=("$M/measure_kblock_device.py"); select=(-k kblock_clock)
                envs=("FJ_KBLOCK_STORE=${FJ_KBLOCK_STORE:-$MEASURE_KBLOCK_STORE}") ;;
  scheme-ab)    files=("$M/measure_scheme_ab.py"); select=(-k scheme_ab_clock)
                envs=("FJ_SCHEME_AB_STORE=${FJ_SCHEME_AB_STORE:-$MEASURE_SCHEME_AB_STORE}") ;;
  kblock-split) files=("$M/measure_kblock_split.py"); select=(-k kblock_split_compile)
                envs=("FJ_KBLOCK_SPLIT_STORE=${FJ_KBLOCK_SPLIT_STORE:-$MEASURE_KBLOCK_SPLIT_STORE}") ;;
  kblock-rolled-compile) files=("$M/measure_kblock_rolled.py"); select=(-k kblock_rolled_compile)
                envs=("FJ_KBLOCK_ROLLED_STORE=${FJ_KBLOCK_ROLLED_STORE:-$MEASURE_KBLOCK_ROLLED_STORE}") ;;
  kblock-rolled) files=("$M/measure_kblock_rolled.py"); select=(-k kblock_rolled_clock)
                envs=("FJ_KBLOCK_ROLLED_STORE=${FJ_KBLOCK_ROLLED_STORE:-$MEASURE_KBLOCK_ROLLED_STORE}") ;;
  kblock-rolled-profile) files=("$M/measure_kblock_rolled.py"); select=(-k kblock_rolled_profile)
                envs=("FJ_KBLOCK_ROLLED_STORE=${FJ_KBLOCK_ROLLED_STORE:-$MEASURE_KBLOCK_ROLLED_STORE}") ;;
  scatter-tail) files=("$M/measure_f_scatter_tail.py") ;;
  probes)       files=("$M/measure_toolchain.py" "$M/measure_ceilings_device.py")
                select=(-k "not record_ceilings") ;;
  recheck|check) files=("$M") ;;
esac

PYTEST=$(find_pytest) || cannot "no pytest in $VENV_PATHS or on PATH -- run 'uv sync'"
config_args "$MEASURE_PYTEST_ARGS"; base=("${CONFIG_ARGS[@]}")

# A compiler, unless this is `check`. Looked for where conftest looks (PATH,
# then the SDKs' own bin directories), so the two never disagree.
if [ "$kind" != check ]; then
  found=$(find_tool "$compiler" || true)
  for d in /usr/local/cuda/bin /opt/rocm/bin; do
    [ -n "$found" ] || { [ -x "$d/$compiler" ] && found=$d/$compiler; }
  done
  [ -n "$found" ] || cannot "$compiler not found -- the device compilers live in the images
               (devtools/devcontainer.sh shell, or --hip shell)"
  echo "measure: $sweep on $backend, compiler $found"
fi

# A card, for the sweeps that launch: probed as kernel-tier.sh probes it.
if [ "$kind" = timed ] || [ "$kind" = device ]; then
  present=0
  case $backend in
    cuda) { [ -e /dev/nvidiactl ] || [ -d /proc/driver/nvidia/gpus ]; } && present=1 ;;
    hip)  amd_gpu_present && present=1 ;;
  esac
  collect_only=0
  for a in "${pytest_extra[@]}"; do
    case $a in --collect-only|--co) collect_only=1 ;; esac
  done
  if [ "$present" = 0 ] && [ "$collect_only" = 0 ]; then
    cannot "no $backend card enumerated (devtools/doctor.sh says why) -- a $sweep
               run here would skip every case and measure nothing"
  fi
fi

case $kind in
  timed|device)
    config_args "$MEASURE_TIMED_ARGS"; run=("${CONFIG_ARGS[@]}")
    marks=(-m "gpu and $backend") ;;
  compile)
    xdist_args; run=("${XDIST[@]}")
    marks=(-m "$backend and not gpu$([ -n "$full" ] || echo ' and not slow')") ;;
  check)
    xdist_args; run=("${XDIST[@]}")
    marks=(-m "not cuda and not hip and not gpu$([ -n "$full" ] || echo ' and not slow')") ;;
esac

for e in "${envs[@]}"; do echo "measure: $e"; done
set +e
env "${envs[@]}" "$PYTEST" "${base[@]}" "${run[@]}" "${marks[@]}" \
  "${files[@]}" "${select[@]}" "${pytest_extra[@]}"
rc=$?
set -e
if [ "$rc" = 5 ]; then
  echo "measure: $sweep collected nothing -- exit 2 (could not run)" >&2
  exit 2
fi
exit "$rc"
