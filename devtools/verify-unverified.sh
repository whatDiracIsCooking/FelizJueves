#!/usr/bin/env bash
# The catch-up card run for merges that went in `unverified-on-card` (#660):
# replay the PR gate's gpu sample on the merged commit, on the backends its
# gate could not reach, and settle the label on the result. Run by hand on a
# box with the cards -- this one, from the combined container.
#
#   devtools/verify-unverified.sh [--dry-run] [--backend cuda|hip] [--seed S]
#                                 [--keep] [-j N] [PR ...]
#
# WHICH PRs. With none named, every merged PR labelled `unverified-on-card`,
# oldest merge first; one after another, one backend at a time, so one card
# is in use at once. A named PR that is not merged, or carries no label, is
# refused (exit 64) rather than verified and commented on.
#
# WHICH BACKENDS. The ones the PR body names as unverified: each line that
# says "unverified" (gate.sh's summary writes `- unverified on card: cuda
# hip` and `| gpu cuda | skipped | ... unverified on card |`) and names
# cuda or hip, case-insensitive. A body naming neither gets both -- the
# label alone says a card run is owed, and running more never verifies less.
# --backend B (repeatable) overrides the parse. --dry-run prints each PR's
# merge commit, backends and mode, and touches nothing.
#
# PER PR, in a throwaway worktree of its merge commit (detached, under
# VERIFY_UNVERIFIED_DIR, default ${TMPDIR:-/tmp}/fj-verify-unverified):
#
#   venv     `uv sync --frozen` into the worktree's own .venv, ALWAYS with
#            UV_PROJECT_ENVIRONMENT=<worktree>/.venv: an inherited one (the
#            FreeBreakfast devcontainer exports /opt/venv) would re-sync a
#            shared environment to an older commit's lock. VENV_PATHS is
#            pinned to that .venv for the run, so its tools are that
#            commit's.
#   gpu      per backend, exactly gate.sh's gpu phase, with that commit's own
#            devtools: KERNEL_TEST_ARGS=GATE_GPU_TEST_ARGS and
#            `kernel-tier.sh --backend <b> -- --fj-sample`. The seed is the
#            gate's rule -- a hash of the merge-base and HEAD -- with
#            FJ_SAMPLE_BASE set to the merge commit's first parent, so HEAD
#            is the merge commit, the merge-base its parent, the change
#            weighting sees exactly what the PR changed, and a rerun draws
#            the same cases. --seed S (FJ_SAMPLE_SEED) fixes it instead.
#            A commit from before gate.sh (no GATE_GPU_TEST_ARGS in its
#            config.sh) has no sample to replay: it runs that commit's
#            kernel-tier.sh PR gate, unsampled, and the comment says so.
#
# THE OUTCOME, per PR:
#
#   pass         every backend passed: a comment with the seed, the budgets
#                and every case drawn, and the label is removed.
#   fail         a backend failed: a comment naming the failing cases (from
#                kernel-tier.sh's pytest.xml) and every case drawn; the
#                label stays. A real failure on a merged commit is a
#                finding -- file it; never loosen a tolerance to clear it.
#   could not run  no card or compiler for a backend, a failed sync, or
#                kernel-tier.sh exit 2: nothing is posted, the label stays.
#
# Each run's logs, draws and the posted comment (comment.md) go to
# KERNEL_TIER_REPORTS/verify-pr<N>-<time> in THIS checkout, so they outlive
# the throwaway worktree, which is removed afterwards unless --keep.
#
# EXIT 0 every PR passed; 1 a PR failed; 2 a PR could not run (worst wins,
# 2 over 1); 64 a usage error.
#:
# --- for a reader of the file ----------------------------------------------
#
# The gpu phase is the four lines gate.sh runs, not gate.sh: the gate has no
# gpu-only mode, and its lint, fast and compile phases already ran on the PR
# before it merged. Keep the KERNEL_TEST_ARGS / FJ_SAMPLE_REPORT /
# kernel-tier.sh invocation below in step with gate.sh's gpu phase.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

LABEL=unverified-on-card
dry_run=
keep=
seed=
backends_override=()
prs=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry_run=1; shift ;;
    --backend)
      case ${2:-} in cuda|hip) backends_override+=("$2") ;;
        *) echo "verify-unverified: --backend takes cuda or hip" >&2; exit 64 ;; esac
      shift 2 ;;
    --seed)    seed=${2:?--seed needs a value}; shift 2 ;;
    --keep)    keep=1; shift ;;
    -j)        JOBS=${2:?-j needs a number}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "verify-unverified: unknown argument '$1'" >&2; usage >&2; exit 64 ;;
    *)  prs+=("${1#\#}"); shift ;;
  esac
done
export JOBS

phase() { printf '\n\033[1m== verify-unverified: %s\033[0m\n' "$1"; }
note()  { printf '   %s\n' "$1"; }
rel()   { echo "${1#"$REPO_ROOT"/}"; }

command -v gh >/dev/null || { echo "verify-unverified: cannot run: no gh" >&2; exit 2; }
command -v uv >/dev/null || [ -n "$dry_run" ] \
  || { echo "verify-unverified: cannot run: no uv (the throwaway worktree's venv)" >&2; exit 2; }

if [ "${#prs[@]}" -eq 0 ]; then
  mapfile -t prs < <(gh pr list --state merged --label "$LABEL" --limit 200 \
    --json number,mergedAt --jq 'sort_by(.mergedAt) | .[].number')
  [ "${#prs[@]}" -gt 0 ] || { echo "verify-unverified: no merged PR carries '$LABEL'"; exit 0; }
fi

# The backends a PR body names as unverified; both when it names neither.
backends_of() {  # backends_of <body>
  local found
  found=$(grep -i 'unverified' <<<"$1" | grep -ioE '\b(cuda|hip)\b' \
    | tr '[:upper:]' '[:lower:]' | sort -u | tr '\n' ' ' || true)
  echo "${found:-cuda hip}"
}

workroot=${VERIFY_UNVERIFIED_DIR:-${TMPDIR:-/tmp}/fj-verify-unverified}
worst=0
bump() { [ "$1" -le "$worst" ] || worst=$1; }

verify_one() {  # verify_one <pr>; returns 0 pass, 1 fail, 2 could not run
  local pr=$1 info state labels merge parent body backends b
  info=$(gh pr view "$pr" --json state,labels,mergeCommit,body,title) || return 2
  state=$(jq -r .state <<<"$info")
  labels=$(jq -r '[.labels[].name] | join(" ")' <<<"$info")
  merge=$(jq -r '.mergeCommit.oid // empty' <<<"$info")
  body=$(jq -r .body <<<"$info")
  phase "PR #$pr: $(jq -r .title <<<"$info")"
  if [ "$state" != MERGED ] || [ -z "$merge" ]; then
    echo "verify-unverified: PR #$pr is $state, not merged" >&2; exit 64
  fi
  case " $labels " in *" $LABEL "*) ;;
    *) echo "verify-unverified: PR #$pr does not carry '$LABEL'" >&2; exit 64 ;; esac
  if [ "${#backends_override[@]}" -gt 0 ]; then
    backends=$(printf '%s\n' "${backends_override[@]}" | sort -u | tr '\n' ' ')
  else
    backends=$(backends_of "$body")
  fi
  backends=${backends% }
  git cat-file -e "$merge^{commit}" 2>/dev/null || git fetch --quiet origin "$merge" \
    || { note "COULD NOT RUN: cannot fetch $merge"; return 2; }
  parent=$(git rev-parse "$merge^1") || return 2
  local mode
  if git show "$merge:devtools/config.sh" 2>/dev/null | grep -q '^GATE_GPU_TEST_ARGS='; then
    mode=sample
  else
    mode=legacy
  fi
  note "merge commit ${merge:0:12} (parent ${parent:0:12})"
  note "backends     $backends"
  note "mode         $([ "$mode" = sample ] \
    && echo "gate.sh's gpu sample, seed ${seed:-from merge-base ${parent:0:12} and HEAD ${merge:0:12}}" \
    || echo "predates gate.sh: that commit's kernel-tier.sh PR gate, unsampled")"
  [ -z "$dry_run" ] || return 0

  local b_missing=()
  for b in $backends; do card_present "$b" || b_missing+=("$b"); done
  if [ "${#b_missing[@]}" -gt 0 ]; then
    note "COULD NOT RUN: no ${b_missing[*]} card enumerated here; nothing posted, label kept"
    return 2
  fi

  local ts report wt
  ts=$(date +%Y%m%d-%H%M%S)
  report="$REPO_ROOT/$KERNEL_TIER_REPORTS/verify-pr$pr-$ts"
  case "$KERNEL_TIER_REPORTS" in /*) report="$KERNEL_TIER_REPORTS/verify-pr$pr-$ts" ;; esac
  mkdir -p "$report" "$workroot"
  wt="$workroot/pr$pr-$ts"
  git worktree add --quiet --detach "$wt" "$merge" \
    || { note "COULD NOT RUN: git worktree add failed"; return 2; }
  CURRENT_WT=$wt

  note "worktree     $wt"
  note "report       $(rel "$report")"
  if ! (cd "$wt" && UV_PROJECT_ENVIRONMENT="$wt/.venv" uv sync --frozen --quiet) \
       >"$report/uv-sync.log" 2>&1; then
    note "COULD NOT RUN: uv sync failed (see $(rel "$report")/uv-sync.log); nothing posted, label kept"
    return 2
  fi

  local gate_args results=() rc verdict t0 secs last any_fail='' cnr=''
  gate_args=$(cd "$wt" && bash -c '. devtools/lib.sh >/dev/null 2>&1; printf %s "${GATE_GPU_TEST_ARGS-}"')
  for b in $backends; do
    phase "PR #$pr: gpu $b"
    t0=$(date +%s)
    mkdir -p "$report/$b"
    if [ "$mode" = sample ]; then
      (cd "$wt" && VENV_PATHS=.venv KERNEL_TEST_ARGS=$gate_args \
        FJ_SAMPLE_BASE=$parent FJ_SAMPLE_SEED=$seed \
        FJ_SAMPLE_REPORT="$report/sample-gpu-$b.json" \
        devtools/kernel-tier.sh --backend "$b" -j "$JOBS" --report "$report/$b" -- --fj-sample) \
        2>&1 | tee "$report/gpu-$b.log"
    else
      (cd "$wt" && VENV_PATHS=.venv \
        devtools/kernel-tier.sh --backend "$b" -j "$JOBS" --report "$report/$b") \
        2>&1 | tee "$report/gpu-$b.log"
    fi
    rc=${PIPESTATUS[0]}
    secs=$(($(date +%s) - t0))
    last=$(grep -E '^=+ .* in [0-9.]+s' "$report/gpu-$b.log" | tail -1 \
      | sed 's/=//g; s/^ *//; s/ *$//' || true)
    case $rc in
      0) verdict=pass ;;
      2) verdict="COULD NOT RUN"; cnr=1 ;;
      *) verdict="FAIL (rc=$rc)"; any_fail=1 ;;
    esac
    results+=("$b|$verdict|$secs|${last:-see gpu-$b.log}")
  done

  if [ -n "$cnr" ]; then
    note "COULD NOT RUN on a backend (kernel-tier.sh exit 2); nothing posted, label kept"
    return 2
  fi

  # The comment: a verdict table, the failing cases, every draw.
  local md="$report/comment.md" r name secs_ detail json xml
  {
    echo "### Catch-up card run (\`devtools/verify-unverified.sh\`, #660)"
    echo
    echo "- merge commit \`${merge:0:12}\`, $(date -u +%Y-%m-%dT%H:%MZ), JOBS=$JOBS"
    if [ "$mode" = sample ]; then
      echo "- gate.sh's gpu sample on that commit (\`KERNEL_TEST_ARGS=$gate_args\`, \`--fj-sample\`), seed from merge-base \`${parent:0:12}\` and HEAD \`${merge:0:12}\`${seed:+ (overridden: \`$seed\`)}"
    else
      echo "- this commit predates gate.sh, so there is no sample to replay: its own \`kernel-tier.sh\` PR gate ran, unsampled"
    fi
    echo "- backends owed (from the PR body): $backends; cards: $(card_present | tr '\n' ' ')"
    echo
    echo "| backend | verdict | wall-clock | result |"
    echo "|---|---|---|---|"
    for r in "${results[@]}"; do
      IFS='|' read -r name verdict secs_ detail <<<"$r"
      printf '| %s | %s | %dm%02ds | %s |\n' "$name" "$verdict" \
        $((secs_ / 60)) $((secs_ % 60)) "$detail"
    done
    if [ -n "$any_fail" ]; then
      echo
      echo "#### Failing cases"
      echo
      for b in $backends; do
        for xml in "$report/$b"/*/pytest.xml; do
          [ -e "$xml" ] || continue
          "$wt/.venv/bin/python" - "$xml" "$b" <<'PY'
import sys
import xml.etree.ElementTree as ET

xml, backend = sys.argv[1], sys.argv[2]
for case in ET.parse(xml).iter("testcase"):
    for kind in ("failure", "error"):
        el = case.find(kind)
        if el is not None:
            msg = (el.get("message") or "").splitlines()[0][:160] if el.get("message") else ""
            name = f"{case.get('classname')}::{case.get('name')}"
            print(f"- `{backend}` {kind}: `{name}`" + (f" -- {msg}" if msg else ""))
PY
        done
      done
      echo
      echo "The label stays: a card run is still owed on the failing backend(s)."
    else
      echo
      echo "Every owed backend passed on card: \`$LABEL\` removed."
    fi
    for json in "$report"/sample-gpu-*.json; do
      [ -e "$json" ] || continue
      echo
      (cd "$wt" && PYTHONPATH=src:test .venv/bin/python -m registry.sample \
        --markdown "sample: gpu $(basename "$json" .json | sed 's/^sample-gpu-//')" "$json") \
        || echo "(could not render $(basename "$json"))"
    done
  } >"$md"

  phase "PR #$pr: result"
  for r in "${results[@]}"; do
    IFS='|' read -r name verdict secs_ detail <<<"$r"
    note "$(printf '%-6s %-14s %4ds  %s' "$name" "$verdict" "$secs_" "$detail")"
  done
  local url
  url=$(gh pr comment "$pr" --body-file "$md") \
    || { note "COULD NOT POST the comment ($(rel "$md")); label kept"; return 2; }
  note "comment      $url"
  if [ -n "$any_fail" ]; then
    note "label        kept"
    return 1
  fi
  gh pr edit "$pr" --remove-label "$LABEL" >/dev/null \
    || { note "COULD NOT remove the label (the comment is posted)"; return 2; }
  note "label        removed"
  return 0
}

# The throwaway worktree goes when its PR is done, or the script is stopped.
CURRENT_WT=
drop_worktree() {
  [ -n "$CURRENT_WT" ] && [ -z "$keep" ] || return 0
  git -C "$REPO_ROOT" worktree remove --force "$CURRENT_WT" 2>/dev/null || true
  CURRENT_WT=
}
trap drop_worktree EXIT

# verify_one runs in an `if`, so errexit is off inside it: every step that
# matters checks its own status and returns 2.
for pr in "${prs[@]}"; do
  if verify_one "$pr"; then s=0; else s=$?; fi
  [ -z "$CURRENT_WT" ] || [ -z "$keep" ] || note "kept         $CURRENT_WT"
  drop_worktree
  bump "$s"
done
exit "$worst"
