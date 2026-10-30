#!/usr/bin/env bash
# The PR gate: the one check that blocks a merge (#656), chosen by what this
# box has. The `pr` skill runs it before merging and pastes its summary into
# the PR body.
#
#   devtools/gate.sh [--seed S] [--no-gpu] [-j N] [--report DIR]
#
# PHASES, in order, stopping at the first that fails:
#
#   preflight  which cards are enumerated (lib.sh's card_present, the rule
#              doctor.sh and the `pr` skill read), and whether ruff, pytest,
#              nvcc and hipcc are here. A missing tool is "could not run"
#              (exit 2): the compile sample needs BOTH device compilers, so
#              on this box's host, which has no hipcc, run it from the
#              combined container (`devtools/devcontainer.sh --combined
#              shell`).
#   lint       `ruff check .`
#   fast       the fast tier: FAST_TEST_ARGS, as the push gate runs it.
#   compile    the compile sample: GATE_COMPILE_TEST_ARGS (config.sh) with
#              `--fj-sample`, nvcc and hipcc in one run, no card needed;
#              its `min` band is L <= GATE_COMPILE_MIN_L_MAX (0).
#   gpu        per backend whose card is enumerated: kernel-tier.sh
#              --backend <b> with KERNEL_TEST_ARGS=GATE_GPU_TEST_ARGS and
#              `--fj-sample` -- generate, compile, then the gpu sample on
#              that card. Skipped, and SAID, for a backend with no card.
#
# BY ENVIRONMENT. Both cards enumerated: every phase, the gpu sample on both
# cards. No card (a Claude cloud container): lint + fast + compile, exit 0,
# and the summary names cuda and hip as unverified on card -- the PR carries
# `unverified-on-card` (#651; the `pr` skill has the rule). One card: the gpu
# sample on that one, the other named unverified. --no-gpu skips the gpu
# phase whatever is enumerated, and its backends count as unverified.
#
# THE SAMPLE (test/registry/sample.py, #653). Both sampled phases select
# `not slow or l_cut` and let the registry sampler choose the classes above
# its `min` band (L <= 2 on a card, L = 0 for the compile sample): every case
# in that band and every case that covers no class runs, the
# `med` and `max` bands are drawn per pool to FJ_BUDGET_MED / FJ_BUDGET_MAX
# seconds, and the families the diff touches are drawn first. The seed is a
# hash of the merge-base with FJ_SAMPLE_BASE and HEAD, so a rerun on the
# same commit draws the same cases; --seed S (FJ_SAMPLE_SEED) fixes it.
#
# THE REPORT goes to DIR (default KERNEL_TIER_REPORTS/gate-<time>): each
# phase's log, each sampled phase's JSON draw, and summary.md -- the mode,
# every phase's verdict and wall-clock, the seed, the budgets and every case
# drawn, written to be pasted into a PR body as its record of what ran.
#
# EXIT 0 every phase that ran passed (a skipped gpu phase is named, not a
# failure); 1 a phase failed; 2 the gate could not run -- never a pass, as
# with kernel-tier.sh.
#:
# --- for a reader of the file ----------------------------------------------
#
# The gpu phase is kernel-tier.sh, not a second copy of it: its compile phase
# still fails on a warning in generated text (#92), and its device phase
# still exits 2 when a card it was promised is absent. The gate only swaps
# its selection (KERNEL_TEST_ARGS, read from the environment by config.sh)
# and adds the flag. kernel-tier.sh's docker re-exec for a host with no hipcc
# would lose that environment, which is one more reason the preflight below
# refuses a box without both compilers rather than reaching it.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

no_gpu=
report=
while [ $# -gt 0 ]; do
  case $1 in
    --seed)   export FJ_SAMPLE_SEED=${2:?--seed needs a value}; shift 2 ;;
    --no-gpu) no_gpu=1; shift ;;
    -j)       JOBS=${2:?-j needs a number}; shift 2 ;;
    --report) report=${2:?--report needs a directory}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "gate: unknown argument '$1'" >&2; usage >&2; exit 64 ;;
  esac
done

ts=$(date +%Y%m%d-%H%M%S)
[ -n "$report" ] || report="$KERNEL_TIER_REPORTS/gate-$ts"
case "$report" in /*) : ;; *) report="$REPO_ROOT/$report" ;; esac
mkdir -p "$report"
rel() { echo "${1#"$REPO_ROOT"/}"; }

phase() { printf '\n\033[1m== gate: %s\033[0m\n' "$1"; }
note()  { printf '   %s\n' "$1"; }
cannot() { echo "gate: cannot run: $*" >&2; exit 2; }

# One line per phase for the summary: `name|verdict|seconds|detail`.
results=()
record() { results+=("$1|$2|$3|${4:-}"); }
now() { date +%s; }

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
phase preflight
present=$(card_present || true)
absent=$(card_absent)
note "cards enumerated: ${present:-none}; absent: ${absent:-none}"

RUFF=$(find_tool ruff) || cannot "no ruff in $VENV_PATHS or on PATH -- run 'uv sync'"
PYTEST=$(find_pytest) || cannot "no pytest in $VENV_PATHS or on PATH -- run 'uv sync'"
VENV_PY=$(dirname "$PYTEST")/python
missing=()
for c in "$NVCC" "$HIPCC"; do
  if p=$(find_tool "$c"); then note "compiler: $p"; else missing+=("$c"); fi
done
[ "${#missing[@]}" -eq 0 ] || cannot "no ${missing[*]} here (searched $VENV_PATHS and PATH).
               The compile sample needs nvcc AND hipcc in one shell: run the
               gate in the combined container,
                 docker/build.sh combined && devtools/devcontainer.sh --combined shell
               or put an installed toolkit's bin/ on PATH (/usr/local/cuda/bin,
               /opt/rocm/bin)."

gpu_backends=()
unverified=()
for b in cuda hip; do
  if [ -z "$no_gpu" ] && card_present "$b"; then
    gpu_backends+=("$b")
  else
    unverified+=("$b")
  fi
done
if [ "${#gpu_backends[@]}" -eq 2 ]; then
  mode="both cards: lint + fast + compile sample + gpu sample on cuda and hip"
elif [ "${#gpu_backends[@]}" -eq 0 ]; then
  mode="no card$([ -z "$no_gpu" ] || echo ' (--no-gpu)'): lint + fast + compile sample only"
else
  mode="one card: lint + fast + compile sample + gpu sample on ${gpu_backends[*]}"
fi
note "mode: $mode"
[ "${#unverified[@]}" -eq 0 ] \
  || note "unverified on card: ${unverified[*]} -- the PR carries \`unverified-on-card\`"

xdist_args

# Run one pytest phase: `run_pytest <name> <log> <args...>`; records it and
# returns pytest's status. A collection with nothing to run (rc 5) fails the
# phase: an empty gate is the green-and-empty run this repo refuses.
run_pytest() {
  local name=$1 log=$2; shift 2
  local t0 rc
  t0=$(now)
  set +e
  "$PYTEST" -rs -p no:cacheprovider "${XDIST[@]}" "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  record "$name" "$([ "$rc" = 0 ] && echo pass || echo "FAIL (rc=$rc)")" \
    "$(($(now) - t0))" "$(tail -1 "$log" | sed 's/=//g; s/^ *//; s/ *$//')"
  return "$rc"
}

summary() {
  local md="$report/summary.md" r name verdict secs detail
  {
    echo "### PR gate (\`devtools/gate.sh\`)"
    echo
    echo "- mode: $mode"
    echo "- unverified on card: ${unverified[*]:-none}"
    echo "- HEAD \`$(git rev-parse --short=12 HEAD 2>/dev/null || echo none)\`, $(date -u +%Y-%m-%dT%H:%MZ), JOBS=${JOBS:-none}"
    echo
    echo "| phase | verdict | wall-clock | result |"
    echo "|---|---|---|---|"
    for r in "${results[@]}"; do
      IFS='|' read -r name verdict secs detail <<<"$r"
      printf '| %s | %s | %dm%02ds | %s |\n' "$name" "$verdict" \
        $((secs / 60)) $((secs % 60)) "$detail"
    done
    for json in "$report"/sample-*.json; do
      [ -e "$json" ] || continue
      echo
      title=$(basename "$json" .json); title=${title#sample-}
      PYTHONPATH=src:test "$VENV_PY" -m registry.sample \
        --markdown "sample: $title" "$json" \
        || echo "(could not render $(rel "$json"))"
    done
  } >"$md"
  phase result
  for r in "${results[@]}"; do
    IFS='|' read -r name verdict secs detail <<<"$r"
    note "$(printf '%-12s %-14s %4ds  %s' "$name" "$verdict" "$secs" "$detail")"
  done
  note "unverified on card: ${unverified[*]:-none}"
  note "report     $(rel "$report") (summary.md is the PR body's section)"
}

finish() {  # finish <exit status>
  summary
  exit "$1"
}

# ---------------------------------------------------------------------------
# lint, fast
# ---------------------------------------------------------------------------
phase lint
t0=$(now)
if "$RUFF" check . 2>&1 | tee "$report/lint.log"; then
  record lint pass "$(($(now) - t0))" "ruff check ."
else
  record lint FAIL "$(($(now) - t0))" "ruff check . (see lint.log)"
  finish 1
fi

phase fast
config_args "$FAST_TEST_ARGS"
run_pytest fast "$report/fast.log" "${CONFIG_ARGS[@]}" || finish 1

# ---------------------------------------------------------------------------
# compile sample: nvcc and hipcc, no card
# ---------------------------------------------------------------------------
phase "compile sample"
config_args "$GATE_COMPILE_TEST_ARGS"
FJ_SAMPLE_REPORT=$(rel "$report/sample-compile.json") \
  FJ_MIN_L_MAX=$GATE_COMPILE_MIN_L_MAX \
  run_pytest compile "$report/compile.log" "${CONFIG_ARGS[@]}" --fj-sample \
  --junitxml="$report/compile.xml" || finish 1

# ---------------------------------------------------------------------------
# gpu sample: each enumerated card, through kernel-tier.sh
# ---------------------------------------------------------------------------
for b in cuda hip; do
  case " ${gpu_backends[*]} " in
    *" $b "*) ;;
    *) record "gpu $b" skipped 0 "no card$([ -z "$no_gpu" ] || echo ' (--no-gpu)'): unverified on card"
       continue ;;
  esac
  phase "gpu sample ($b)"
  t0=$(now)
  set +e
  KERNEL_TEST_ARGS=$GATE_GPU_TEST_ARGS \
    FJ_SAMPLE_REPORT=$(rel "$report/sample-gpu-$b.json") \
    devtools/kernel-tier.sh --backend "$b" --report "$report" -- --fj-sample \
    2>&1 | tee "$report/gpu-$b.log"
  rc=${PIPESTATUS[0]}
  set -e
  last=$(grep -E '^=+ .* in [0-9.]+s' "$report/gpu-$b.log" | tail -1 | sed 's/=//g; s/^ *//; s/ *$//' || true)
  case $rc in
    0) record "gpu $b" pass "$(($(now) - t0))" "$last" ;;
    2) record "gpu $b" "COULD NOT RUN" "$(($(now) - t0))" "kernel-tier.sh exit 2"
       finish 2 ;;
    *) record "gpu $b" "FAIL (rc=$rc)" "$(($(now) - t0))" "${last:-see gpu-$b.log}"
       finish 1 ;;
  esac
done

finish 0
