---
name: milestone-run
description: >-
  Complete a GitHub milestone end to end by orchestrating subagents: read the
  milestone's issue DAG, fan out one subagent per ready issue (each in its own
  .claude/worktrees/ worktree, building and testing on a GPU card), merge
  their PRs one at a time, and launch newly unblocked issues until the
  milestone is done. Use when the user invokes `/milestone-run <milestone
  URL|number> [max=N] [both] [dry-run]` or asks to "complete / execute / ship
  this milestone with subagents". Invoking it is the user's merge grant for
  this milestone's PRs. Requires gh + GH_TOKEN. Not for creating a milestone —
  that is `milestone`.
---

# `/milestone-run <milestone> [max=N] [both] [dry-run]` — milestone DAG → merged PRs

`milestone` turns a plan into a DAG of PR-sized issues; this skill **executes**
one. You are the **orchestrator**: you schedule, create worktrees, review, merge
and tear down. Subagents implement one issue each, ship a green PR, and **hand
it back without merging**.

**Invoking `/milestone-run` is the merge grant** — for PRs that close an issue
in *this* milestone, opened by *this* run's subagents, once the gates in step 5
are green. It does not cover any other PR, `--admin`, or merging red. If the
auto-mode classifier still blocks a merge, stop merging and report the green PR
links; do not work around the block.

## Arguments

- **Milestone** — a URL (`…/milestone/<N>`), `#N`, or a bare number. The
  number is all you need.
- **`max=N`** — the most subagents in flight at once. Default **3**. Every
  agent builds the tree and runs suites on a shared card, so the box, not
  GitHub, is the limit: past ~4, warn the user that builds will thrash and GPU
  runs will queue, then honour the number they gave.
- **`both`** — require every PR to pass its tests on **both** cards (CUDA and
  HIP), as the `issue` skill does. Without it, **one backend suffices**: each
  agent tests on the card it is assigned, and **nothing else checks the other**
  — there is no CI in this repo yet, so a one-backend run is genuinely
  unverified on the second compiler. Prefer `both` for anything that changes
  emitted kernel text; one backend is for work that cannot reach a kernel
  (tooling, docs, the host side of the generator).
- **`dry-run`** — do steps 1–2, print the schedule, and stop. No worktree, no
  agent, no `gh` write.

## 1. Read the milestone and rebuild the DAG

```bash
gh api repos/{owner}/{repo}/milestones/<N> --jq '{title,state,open_issues,closed_issues}'
gh issue list --milestone "<title>" --state all --limit 200 \
  --json number,title,state,body
```

`--milestone` wants the **title**, not the number. For each issue, take its
blockers from the native dependency API, falling back to the `## Depends on`
lines that `milestone` writes into every body:

```bash
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by --jq '.[].number'
```

Then classify every issue:

- **done** — closed.
- **ready** — open, every blocker closed, and no open PR already references it
  (`gh pr list --search "<n> in:body" --state open --json number,headRefName`).
- **resumable** — open, with an open PR whose head branch is `issue-<n>-*`
  *and* a local `.claude/worktrees/issue-<n>-*` checkout. An earlier run of
  this skill died or was compacted mid-flight: adopt it. A green PR goes
  straight to step 5. A red one gets a fresh agent pointed at the existing
  worktree and PR ("finish PR #M") instead of a new branch. An open PR that
  fails either test is someone else's: leave it alone and report it.
- **blocked** — some blocker still open. A blocker *outside* the milestone that
  is open is a hard stop for that branch: report it, don't work around it.

A cycle, or an issue with no `Acceptance` section, means the milestone is
malformed: stop and say so rather than guess an order.

## 2. Show the schedule, then start

Print the wave table (issue, title, blockers, state), the ready set, `max`, and
whether `both` is on. On `dry-run`, stop here. Otherwise start without waiting
for a reply — the invocation was the go-ahead. Stop and ask instead only when
step 1 found something odd (cycle, external open blocker, an issue that looks
bigger than one PR).

## 3. Pre-flight, once, on main

```bash
git fetch origin main --quiet && devtools/worktree.sh sync
nvidia-smi -L                                     # CUDA card alive?
rocminfo | grep -m1 gfx                           # HIP card alive?
devtools/kernel-tier.sh --backend cuda --no-test   # warm sccache: CUDA
devtools/kernel-tier.sh --backend hip  --no-test   # warm sccache: HIP
```

Several cold kernel compiles at once will contend for this box, so a warm cache
is what makes `max` agents affordable. Warm only the backends you will use. The
live cards decide the assignment pool: both alive → alternate; one dead → every
agent gets the live one (and say so); both dead, or one dead under `both` → no
agent can pass its gate, so don't launch. Report it and stop.

## 4. Dispatch loop

While any issue is ready or in flight:

1. **Launch** `min(max − in_flight, |ready|)` ready issues. For each, **you**
   create the worktree, one at a time, so parallel agents never race on
   `git fetch` or the ref lock:

   ```bash
   git fetch origin main --quiet
   git worktree add .claude/worktrees/<wt> -b <wt> origin/main
   git branch --unset-upstream <wt>
   ```

   `<wt>` is `issue-<n>-<slug>` as in the `issue` skill. Don't use the Agent
   tool's `isolation: "worktree"`: it lives elsewhere and tears itself down,
   which removes the checkout you need for a rebase. Also hand each agent:

   - **a backend** (unless `both`): alternate CUDA / HIP across launches so
     each card carries about half the test load. The agent falls back to the
     other card if its own is down.
   - **the files it owns.** Two agents regenerating the same golden kernel
     files conflict on every merge, and a golden file is a whole-file diff with
     no clean 3-way resolution. Partition the work so each in-flight agent owns
     disjoint output — different ERI classes, different modules — and say which
     explicitly in its prompt.
   - **a job count** `J = $(nproc) / max` (24 cores, `max=3` → `-j 8`).
     `kernel-tier.sh` otherwise defaults every agent to `JOBS`.

   Then spawn a background `general-purpose` subagent with the prompt in
   "Subagent prompt" below. Record `issue → {wt, agent id, backend, anchor}`.

2. **Wait for notifications**; never poll on a short timer. When an agent
   reports:
   - **"PR #M green"** → step 5. Merges are serial: if one is in progress,
     queue this one.
   - **stuck** (ambiguous issue, a suite failed, would need a tolerance
     loosened, scope bigger than one PR) → don't merge. Mark the issue
     **stuck**. Its dependents stay blocked. Keep its worktree. Carry on with
     every other branch of the DAG.

3. **Watch for stragglers.** If an agent runs long, check `gh pr checks <M>`
   yourself. A green PR whose agent is still "watching CI" is waiting on the
   `images` workflow's `build (cuda)`/`(hip)`, which never runs on a code PR:
   nudge the agent with `SendMessage`. Use `TaskStop` on one that has already
   handed back.

**Status line.** After every launch, merge, or stuck report, print one line —
`merged #301 → PR #310 (3/9 done) · launched #305 [hip] · in flight: #302 #304
· stuck: none` — so the user can follow a long run without reading the
transcript.

## 5. Review and merge — one PR at a time

Merge **sequentially**, never in parallel and never with `--auto`. Before each
merge, check:

```bash
gh pr view <M> --json mergeable,statusCheckRollup,body,files
ruff check .                                  # agents' --no-verify skips it
```

- `mergeable` is `MERGEABLE` (there is no CI check to wait for yet — which is
  exactly why the agent's own reported run is the only evidence);
- the body has `Closes #<n>` and, under *What was run*, the backend, card, and
  test pass counts — at least one card, or **both** under `both`. A
  compile-only run is not a test, and a kernel-tier exit 2 is not a pass;
- no tolerance was loosened to get green.

Then `gh pr merge <M> --squash`.

**Conflicts.** Each merge moves `main`, so a later sibling can still conflict
despite the anchors. Two cases:

- **A pure list or registry conflict** (both sides appended an entry) →
  resolve it yourself; another GPU round trip buys nothing. In the PR's
  worktree: `git -C <wt> fetch origin main`, `git -C <wt> rebase origin/main`,
  keep both sides, `ruff check .`, `git -C <wt> rebase --continue`,
  `git -C <wt> push --force-with-lease`.
- **Anything else** — generator code, a test, or any **golden kernel file** →
  send it back to the agent that wrote the PR (`SendMessage` resumes it with
  its context): rebase, resolve, **regenerate** rather than hand-merging the
  emitted text, re-run the kernel tier, push, re-report green.

**After each merge:**

- tear the worktree down: `git worktree remove .claude/worktrees/<wt> &&
  git branch -D <wt>`;
- confirm the issue closed (`gh issue view <n> --json state`);
- print the status line, recompute **ready**, and go back to step 4.1.

## 6. Finish

When nothing is ready or in flight:

- if every issue is closed, close the milestone
  (`gh api -X PATCH repos/{owner}/{repo}/milestones/<N> -f state=closed`);
- report one line per issue: `#n → PR #M merged [cuda|hip|both]`, or
  **stuck** with its reason, or **blocked** naming its blocker. Spend words
  only on what deviated: blocked merges, rebases, stuck issues, card
  fallbacks, decisions made for the user.

Leave the worktrees of stuck issues standing, and say where they are.

## Subagent prompt

Fill in `<…>`. Keep it self-contained — the agent has none of your context.
`<card>` is `cuda` or `hip`. Under `both`, give both command pairs and drop the
fallback line.

```
Resolve GitHub issue #<n> (<title>) in this repo. Your worktree already exists
at <abs path to .claude/worktrees/<wt>>, on branch <wt>, forked from
origin/main — EnterWorktree {"path": …} into it and do not create another. Run
`uv sync` in it first.

Follow .claude/skills/issue/SKILL.md steps 1, 3, 4 and the PR half of 5, with
these overrides:
- Do NOT merge and do NOT tear down the worktree. Your job ends at an open PR
  with your own run reported in it. Then reply exactly: "PR #<M> green" + one
  line naming the card, tests, and pass count.
- There is NO CI on this repo, so there is no check to wait for: your reported
  run is the only evidence the PR carries. Do not claim a check passed.
- Test on ONE card, your assigned <card> — this overrides the issue skill's
  both-cards rule. Nothing else verifies the other backend, so say in the PR
  which one you did not run. Siblings share the card, so compile unlocked and
  take the card's lock only to run:
    devtools/kernel-tier.sh --backend <card> -j <J> --no-test
    flock /tmp/fj-gpu-<card>.lock devtools/kernel-tier.sh --backend <card> -j <J> -- -k '<pattern>'
  If your card is down (nvidia-smi -L / rocminfo), use the other one and say
  so in the PR. An out-of-memory error on the card means a sibling's run
  overlapped yours: re-run under the lock before calling it a failure.
- If a suite fails, or passing would mean loosening a tolerance, stop and
  reply "STUCK: <reason>" — no PR needed.
- Commit and push with `git -C <abs worktree path> …` (protect-main
  false-positive). Never --no-verify, never CLAUDE_ALLOW_MAIN_EDITS.
- PR body via `--body-file` in your scratchpad, with `Closes #<n>`, and under
  "What was run" the backend, card, tests and pass counts.
- Run `ruff check .` before committing.
- You own these files and ONLY these: <the agent's partition>. Siblings are
  generating in parallel, and a golden kernel file has no clean 3-way merge --
  if you need to change a file outside your partition, reply "STUCK: <what>"
  instead. If asked to rebase, REGENERATE emitted files rather than
  hand-merging their text.
- If the issue is ambiguous in a way the code can't answer, reply
  "STUCK: <question>" rather than guessing.
```
