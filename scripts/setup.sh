#!/usr/bin/env bash
# Step 1 of the README: fetch the pinned upstream PrivEscalate code, apply the harness patch, and
# create the harness environment (.venv).
#
#   bash scripts/setup.sh
#
# Upstream is NOT redistributed in this repository (it grants no project-wide license); it is cloned
# from GitHub at the pinned commit. Set PRIVREP_UPSTREAM (env or secrets.env) to use another location.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
URL=https://github.com/yxsec/PrivEscalate.git
COMMIT=cfb362d
PATCH="$PWD/patches/upstream-harness-fixes.patch"
UP=$(setting PRIVREP_UPSTREAM upstream)

command -v uv >/dev/null || { echo "ERROR: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 1; }

# 1. upstream at the pinned commit
if [ ! -d "$UP/.git" ]; then
  echo "== cloning $URL -> $UP"
  git clone "$URL" "$UP"
fi
if [ "$(git -C "$UP" rev-parse HEAD | cut -c1-7)" != "$COMMIT" ]; then
  if [ -n "$(git -C "$UP" status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: $UP has local changes and is not at $COMMIT; refusing to touch it" >&2; exit 1
  fi
  git -C "$UP" checkout --quiet --detach "$COMMIT"
fi
echo "== upstream at $(git -C "$UP" rev-parse --short HEAD)"

# 2. the harness patch (19 added / 9 removed lines, env-gated; see docs/DEVIATIONS.md section A)
if git -C "$UP" apply --check "$PATCH" 2>/dev/null; then
  git -C "$UP" apply "$PATCH"; echo "== patch applied"
elif git -C "$UP" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "== patch already applied"
else
  echo "ERROR: $PATCH does not apply to $UP; is it a clean checkout of $COMMIT?" >&2; exit 1
fi

# 3. harness environment
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/harness.txt
uv pip install --python .venv/bin/python --no-deps -e "$UP/baselines/hackingBuddyGPT"

[ -f secrets.env ] || { cp secrets.env.example secrets.env
  sed -i "s/^VLLM_API_KEY=$/VLLM_API_KEY=$(.venv/bin/python -c 'import secrets;print(secrets.token_hex(16))')/" secrets.env
  echo "== created secrets.env with a random VLLM_API_KEY"; }
echo "== done. Next: bash scripts/setup_vllm.sh"
