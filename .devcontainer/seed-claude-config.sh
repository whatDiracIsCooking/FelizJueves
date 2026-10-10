#!/usr/bin/env bash
# postCreateCommand for every variant under .devcontainer/: make the agent's
# per-container config usable, and report what it got.
#
# Runs as `ubuntu` INSIDE the container, once per container creation (not per
# shell). It is deliberately tiny and idempotent -- two of the three paths it
# touches are named volumes that survive a `rebuild`, so anything written here
# must tolerate already existing and must never clobber a user's own setting.
#
# WHY ANYTHING IS NEEDED AT ALL: ~/.claude/plugins and ~/.claude/backups are
# volumes, and a volume mounted over a directory inherits that directory's
# ownership from the IMAGE on first use -- which is why Dockerfile.base
# pre-creates both as uid 1000. If that ever regresses, the symptom is a
# permission error from the agent rather than from docker, so this script
# checks rather than assumes.
set -euo pipefail

echo "seed-claude-config: user $(id -un) ($(id -u):$(id -g))"

for d in "$HOME/.claude" "$HOME/.claude/plugins" "$HOME/.claude/backups"; do
  mkdir -p "$d"
  [ -w "$d" ] || { echo "  NOT WRITABLE: $d -- check the volume's owner in docker/Dockerfile.base" >&2; exit 1; }
done
echo "  config dirs ok"

# The pin, reported rather than enforced: docker/install-claude-code.sh already
# failed the BUILD if what landed was not what was asked for, so this is only
# so the container says out loud what it carries.
if command -v claude >/dev/null 2>&1; then
  echo "  claude $(claude --version 2>/dev/null | cut -d' ' -f1) (pinned ${CLAUDE_CODE_VERSION:-unset})"
else
  echo "  claude not on PATH -- expected in a GPU image; see docker/install-claude-code.sh" >&2
fi

# Auto-update is off through DISABLE_AUTOUPDATER in each devcontainer.json's
# containerEnv, not from here: npm's global prefix in these images is /usr and
# root-owned, so an in-container update cannot succeed. The version is a
# reviewable line in git instead -- devtools/claude-version.sh --apply.
if [ -n "${DISABLE_AUTOUPDATER:-}" ]; then
  echo "  autoupdater: disabled"
else
  echo "  autoupdater: ENABLED -- unexpected in a container; see containerEnv" >&2
fi

echo "seed-claude-config: done"
