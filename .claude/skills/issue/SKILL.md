---
name: issue
description: >-
  Resolve one GitHub issue end to end: read it, fork a fresh
  .claude/worktrees/ worktree, implement the fix, build AND run the relevant
  suites on both the NVIDIA (CUDA) and AMD (HIP) cards, then `/pr full` and
  merge. Use when the user invokes `/issue <number|url>` or says "resolve issue
  #N in a new worktree, build+test amd+nvidia, /pr full, you may merge".
  Invoking it is the user's merge grant for that one PR. Requires gh + GH_TOKEN.
---

# `/issue <N | issue URL>` — issue → worktree → both cards → merged PR

This skill is the user's standing kickoff, written down once:

> https://github.com/whatDiracIsCooking/FelizJueves/issues/N — resolve in a new
> worktree; build+test AMD + NVIDIA; after the relevant tests pass, `/pr full`;
> you have my permission to merge.

**Invoking `/issue` is that permission** — to merge *this issue's* PR, once its
gates below are green. It does not extend to any other PR, to `--admin`, or to
merging red. If the auto-mode classifier still blocks the merge, stop and hand
back the green PR link; do not work around the block.

One issue per invocation. Several numbers → do them one after another, or ask
whether to fan out (the parallel-wave shape has its own gotchas: two agents
regenerating the same golden kernel files, rebase races).

## 1. Read the issue — and decide whether it is ready

```bash
gh issue view <N> --comments
```

Accept a bare number, `#N`, or the full URL; the number is all you need.

Stop and ask, rather than start, when:

- the issue is **closed**, or an open PR already references it
  (`gh pr list --search "<N> in:body" --state all`);
- it says **blocked by / depends on** an issue that is still open;
- the acceptance criteria are genuinely ambiguous — a question the code cannot
  answer. A missing detail with a conventional answer is not ambiguous: pick it
  and say so in the PR.

## 2. Fork a fresh worktree from `origin/main`

Per the **worktree** skill, from the main checkout root. Name it
`issue-<N>-<slug>` (slug: 2–4 words from the title, e.g. `issue-222-trttf`) —
never a bare word like `cleanup…` that reads as a verb.

```bash
git fetch origin main --quiet
devtools/worktree.sh sync                     # keep local main honest too
git worktree add .claude/worktrees/<name> -b <name> origin/main
git branch --unset-upstream <name>            # else it tracks origin/main
```

Then `EnterWorktree {"path": ".claude/worktrees/<name>"}`. There is no
submodule to initialise. A worktree does not inherit `.venv` or
`.kernel-tier/`: run `uv sync` in it, and its first kernel-tier run recompiles
from scratch (the shared sccache store makes the second cheap).

## 3. Implement

Read the code the issue names before writing. The house rules that bite here:

- **Look at a sibling.** A new piece of the generator follows the shape of an
  existing one, with its test beside it under `test/`, and its emitted text
  checked against a golden file.
- The conventions that bite are in `.claude/CLAUDE.md` ("Conventions"):
  `from __future__ import annotations`, the marker/fixture split, golden files
  are tracked, and no arithmetic-relaxing compiler flags.
- **One source, both compilers.** A generated kernel has to satisfy nvcc AND
  hipcc; hipcc is clang and nvcc is not. Emitting anything only one accepts is
  the defect class to look for.
- The oracle is numpy, in the device's precision. Never loosen a tolerance to
  get green — that is a "when NOT to merge" case.

## 4. Build and test on BOTH cards

Check both cards are alive first — a dead card is a skipped tier, not a pass:

```bash
nvidia-smi -L        # RTX 3080; NVML "Unknown Error" is intermittent
rocminfo | grep -m1 gfx   # gfx1200
```

Then, per the **test** skill, run the kernel tier on each backend — this runs
the kernels against the numpy reference, which no hosted runner can do:

```bash
devtools/kernel-tier.sh --backend cuda -- -k '<pattern>'   # NVIDIA
devtools/kernel-tier.sh --backend hip  -- -k '<pattern>'   # AMD
```

"Relevant" means the tests for everything the diff touches **and** everything
that imports it (`grep -rl '<module>' . --include='*.py'`). A change to a
shared part of the generator, to `test/conftest.py`, or to the compiler flags
in `devtools/config.sh` makes the whole tier relevant: drop the `-k`.

Both legs must pass with the skip list read, and the report must say `device
present` on each. If one card is down, say so plainly and do **not** merge on
one backend's word — hand back the PR. `--no-test` is not "tested", and exit 2
is "could not run", not a pass.

Before committing:

```bash
ruff check .          # the commit hook fails on a finding in a file you touched
```

## 5. Ship: `/pr full`

Invoke the **pr** skill with `full`, with these specifics:

- PR body via `--body-file` (written to the scratchpad — a heredoc trips the
  redirect false-positive); there is no template in this repo yet, so see the
  **pr** skill for the two sections it needs. Put **`Closes #<N>`** under *What
  and why*, and under *What was run* name both backends, both devices, and the
  tests with their pass counts.
- Commit and push with `git -C <abs-worktree-path> …` if protect-main
  misfires; never `--no-verify`, never `CLAUDE_ALLOW_MAIN_EDITS`.
- **There is no CI in this repo yet** (no `.github/`), so there is no check to
  wait for and nothing server-side will catch what you did not run locally.
  Merge with `gh pr merge <n> --squash`, and when CI does land, wait for it.
- If `main` moved and the PR conflicts, rebase, re-run the step-4 suites that
  the conflict touched, push, and wait again.
- Teardown: `ExitWorktree {"action": "keep"}`, then `git worktree remove
  .claude/worktrees/<name> && git branch -D <name>` (no submodule now, so the
  plain removal works). Confirm the issue closed (`gh issue view <N> --json
  state`).

## 6. Report

On a clean run, one or two lines: "Merged PR #M (closes #N); CUDA + HIP
`<Suite>` green." Narrate only what deviated — a blocked merge, a dead card, a
rebase, a decision you made on an under-specified issue.
