---
name: pr
description: >-
  Ship the current work: commit the relevant changes, push the branch, open a
  GitHub PR with gh, and merge it. Use when the user asks to "PR this", "ship
  it", "commit/push/merge", or otherwise wants the working-tree changes landed
  on main end to end. Requires gh + GH_TOKEN.
---

# Commit → push → PR → merge

Drive the current changes all the way to `main` in one pass. Each step has a
gotcha; follow them in order.

## 1. Commit the relevant changes

- **Never commit on `main`.** If the session is on `main`, create a branch
  first (`git switch -c <topic>`). If the session is in a `.claude/worktrees/`
  worktree, the branch already exists — use it.
- Stage **relevant** changes, not everything: read `git status` and `git diff`,
  stage what belongs to this task, and leave unrelated untracked files behind.
  No blanket `git add -A` without looking.
- The pre-commit hook lints every touched file — a finding blocks the commit;
  fix it, don't `--no-verify`.
- If `pyproject.toml` changed, run `uv lock` and commit the lock with it. The
  lock is the only place exact versions live, and the image installs it with
  `--frozen`, so a stale lock breaks the container rather than the host.

## 2. Check the repo's own invariants

Before pushing, look at what the diff touches and whether this repo guards it:

- a **ratchet** (a test pinning a measured number or a floor) may only move in
  the improving direction, in its own commit, saying what moved. A loosened
  threshold produces a green run — that is exactly why it needs a human.
- the **reference chain** (whatever independently checks correctness here) is
  not something to adjust to make a test pass.

A repo with neither has nothing to do in this step.

## 2½. Run the gate

```bash
devtools/gate.sh            # lint, fast, compile sample, gpu sample per card
```

`devtools/gate.sh` (#656) is **the one check that blocks a merge**, and it
picks its phases from what this box has:

- **both cards enumerated**: lint + fast + the compile sample (nvcc and
  hipcc) + the gpu sample on **both** cards (each through
  `kernel-tier.sh --backend <b>`);
- **no card** (a Claude cloud container): lint + fast + the compile sample,
  exit 0, and it says cuda and hip are **unverified on card** -- the PR
  takes the `unverified-on-card` label (below);
- **one card**: the gpu sample on that one, the other unverified.

The compile sample needs **both** device compilers in one shell, so on this
box's host (no hipcc) it exits 2; run it from the combined container
(`devtools/devcontainer.sh --combined shell`). In a cloud container with the
toolkits installed but off `PATH`, put `/usr/local/cuda/bin` and
`/opt/rocm/bin` on it first. **Exit 2 is "could not run", never a pass** --
do not merge on it.

The sampled phases draw their classes above L = 2 (L = 0 for the compile
sample) from the case registry (#653), seeded by the merge-base and HEAD, so
a rerun on one commit draws the same cases. The gate writes
`<report>/summary.md`: the mode, every phase's verdict and wall-clock, **the
seed, the budgets and every case drawn**. Paste it into the PR body's *What
was run* -- it is the record of what this merge was checked against. Run
the gate on the commit you push; a later commit draws afresh.

## 3. Push

```bash
git push -u origin <branch>
```

- A push touching any `.py` auto-runs the fast test gate
  (`devtools/prepush-tests.sh`). A successful push therefore *implies* that
  gate passed. Do not bypass with `--no-verify`.
- If the worktree's git dir has gone missing (`fatal: not a git repository:
  …/worktrees/<name>`), push from the main checkout instead:
  `git -C <repo-root>/main push origin <branch>`.

## 4. Create the PR

```bash
gh pr create --title "<concise title>" --body-file <path>
```

**There is no `.github/pull_request_template.md` in this repo yet**, so write
the body yourself and give it the two sections that matter: *What and why*, and
**What was run** — the tier, the backend, the card, and the pass counts. That
record is not a formality here: no runner can verify a kernel, so the PR body
is the only evidence the work was tested at all. Write it to a scratch file and
pass `--body-file`; `gh pr create --body "…"` is fine too, but a file keeps
heredoc quoting out of it. When a template does land, fill it in rather than
bypassing it — `gh pr create` applies it only when no body is supplied.

The harness's PR-body footer rules apply.

### No card, or one: the `unverified-on-card` label

Ask which cards this session can reach before writing the body:

```bash
bash -c '. devtools/lib.sh; echo "present: [$(card_present)] absent: [$(card_absent)]"'
```

`card_present` (in `devtools/lib.sh`) uses the enumeration rule the tiers use
— an AMD GPU node in kfd's topology (`gfx_target_version` nonzero), an NVIDIA
GPU under `/proc/driver/nvidia/gpus/` or in `nvidia-smi -L`, **and in both
cases a card this process can actually `open`** (#712) — and
`devtools/doctor.sh` prints the same verdict as its `card verdict` line.

A card that is enumerated but whose device-cgroup grant has been revoked
mid-session therefore counts as **absent**, and this PR takes the label rather
than failing: that is the intended path, not a workaround. It is still a broken
container, so fix it when you can — `devtools/devcontainer.sh gpu` names it,
`rebuild` repairs it (a plain `up` cannot), and doctor's `card usable` row
FAILs meanwhile. A gate run after the repair verifies for real and needs no
label.

- **Both cards present**: `devtools/gate.sh` runs the gpu sample on both
  cards. No label. A change to kernel text above the sample's reach still
  wants `kernel-tier.sh --full` on each backend it touches.
- **One card, or none** (a Claude cloud container has none): the PR may still
  merge after `devtools/gate.sh` is green, which there means lint, the
  **fast** tier and the **compile sample** (both compilers, no card) — but it carries
  the `unverified-on-card` label (`gh pr create … --label unverified-on-card`;
  the label already exists), and the body's *What was run* names each backend
  in `card_absent` as **unverified on card** -- the gate's summary does. A session with one card still
  runs the kernel tier on that one and says so.

The label is the record that a card run is owed, and it never excuses a red
fast or compile tier. Whoever next has both cards pays it back with
`devtools/verify-unverified.sh` (#660), by hand:

```bash
devtools/verify-unverified.sh --dry-run     # every labelled merged PR: commit, backends, mode
devtools/verify-unverified.sh 675           # one PR; no number = all of them, oldest first
```

For each PR it checks the merge commit out in a throwaway worktree (its own
`.venv`, synced with `UV_PROJECT_ENVIRONMENT` pointed into it, so the
devcontainer's `/opt/venv` is never re-synced) and runs gate.sh's gpu sample
there on the backends the body names as unverified (both, if it names
neither), seeded from the merge commit and its parent so a rerun draws the
same cases. **Pass**: it comments with the cases drawn and removes the label.
**Fail**: it comments with the failing cases and keeps the label -- a
failure on a merged commit is a finding to file, not a tolerance to loosen.
**Could not run** (exit 2): nothing is posted. A commit from before gate.sh
runs its own `kernel-tier.sh` PR gate instead, and the comment says so.

## 5. Merge

```bash
gh pr merge <number> --squash
# the remote branch is auto-deleted on merge (see below) -- nothing to run here
```

- **There is no CI yet** — no `.github/workflows/` in this repo. So there is
  nothing to wait for, and nothing server-side will catch what you did not run
  locally: `devtools/gate.sh` (step 2½) is the whole check. Its summary in
  the PR says which backends ran. When CI lands, wait for it
  with `gh pr checks <number> --watch` and do not merge red.
- **Never pass `--admin`.** It merges past required status checks, and it is the
  one move in this whole flow that can defeat the branch ruleset. Red checks
  mean fix them or hand the PR back — never override. A ruleset with an empty
  bypass list makes the server refuse it anyway, but do not lean on that: the
  bypass list is a setting someone can widen, and this rule is the intent.
- **Never pass `--delete-branch`**: it attempts a local `git checkout main`,
  which fails when `main` is held by another worktree. The remote branch is
  reclaimed automatically instead (next bullet).
- **Remote branch cleanup is automatic.** This repo has GitHub's "Automatically
  delete head branches" setting on, so the head branch is deleted when the PR
  merges. Do **not** run `git push origin --delete` — the ref is already gone, so
  it only errors, and the auto-mode classifier blocks it as destructive anyway.
  Confirm with `git ls-remote --heads origin <branch>` (empty output = gone). The
  *local* branch is still yours to delete in step 6.
- If GitHub reports the PR as not yet mergeable, wait a moment and re-check
  with `gh pr view <number> --json mergeable,mergeStateStatus` before retrying
  — do not force.

## 6. After the merge

`origin/main` has moved but local `main` has not — and `worktree.sh add` forks
from local HEAD, so a stale `main` silently seeds stale branches:

```bash
devtools/worktree.sh sync
```

It finds whichever checkout has `main` on it and fast-forwards it there, so it
runs from any worktree; it refuses (`--ff-only`) rather than merging if that
`main` has diverged.

If the work happened in a `.claude/worktrees/` worktree, whether to tear it
down depends on how this skill was invoked:

- **Bare `/pr`** — leave the worktree standing. The work is merged, but the
  checkout stays for follow-up commits; do not remove it unless asked.
- **`/pr full`** — reclaim it now, per the `worktree` skill's "`/pr full` —
  ship, then tear the worktree down" section: exit with
  `ExitWorktree {"action": "keep"}` if the session is inside it, then
  `git worktree remove` + `git branch -d <name>` (`-D` after a squash merge).

Otherwise — a plain topic branch on an ordinary checkout — the local branch can
go with `git branch -d <branch>` (or `-D` after a squash merge, since the SHAs
differ; the merge already confirmed the patches landed).

## When NOT to merge

Stop after step 4 and hand the PR to the user instead of merging when:

- CI is red, `devtools/gate.sh` failed or could not run (exit 1 or 2), or
  the push gate was bypassed with `--no-verify`;
- the diff moves a ratchet floor or touches the correctness references in ways
  only a human should sign off on;
- the user asked for a PR but not a merge.
