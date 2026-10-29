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
GPU under `/proc/driver/nvidia/gpus/` or in `nvidia-smi -L` — and
`devtools/doctor.sh` prints the same verdict as its `card verdict` line.

- **Both cards present**: the kernel tier is expected on each backend the diff
  touches, as above. No label.
- **One card, or none** (a Claude cloud container has none): the PR may still
  merge after the **fast** tier and the **compile** tier (`(cuda or hip) and
  not gpu and not slow`, both compilers, no card) are green — but it carries
  the `unverified-on-card` label (`gh pr create … --label unverified-on-card`;
  the label already exists), and the body's *What was run* names each backend
  in `card_absent` as **unverified on card**. A session with one card still
  runs the kernel tier on that one and says so.

The label is the record that a card run is owed: whoever next has both cards
runs the kernel tier for the labelled PRs' families and removes it. It never
excuses a red fast or compile tier.

## 5. Merge

```bash
gh pr merge <number> --squash
# the remote branch is auto-deleted on merge (see below) -- nothing to run here
```

- **There is no CI yet** — no `.github/workflows/` in this repo. So there is
  nothing to wait for, and nothing server-side will catch what you did not run
  locally: the fast tier, `ruff check .`, and the kernel tier on each backend
  you touched. Say in the PR which backends you ran. When CI lands, wait for it
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

- CI is red, or the push gate was bypassed with `--no-verify`;
- the diff moves a ratchet floor or touches the correctness references in ways
  only a human should sign off on;
- the user asked for a PR but not a merge.
