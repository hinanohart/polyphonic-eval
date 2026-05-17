#!/usr/bin/env bash
# setup-pypi-publisher.sh
#
# One-shot helper to finish the only step that polyphonic-eval cannot
# automate from CI: registering the PyPI Trusted Publisher binding so the
# OIDC-based publish workflow can actually upload to PyPI. PyPI's publisher
# registration page requires interactive browser auth, so this is the one
# manual touchpoint per project / per account.
#
# What it does:
#   1. Verifies prerequisites (gh CLI, gh auth, the repo).
#   2. Prints the exact form fields PyPI asks for.
#   3. Opens https://pypi.org/manage/account/publishing/ in your browser.
#   4. Waits for you to confirm "submitted" and then re-runs the most
#      recent FAILED publish workflow on this repo via `gh run rerun`.
#   5. Tails the rerun until it finishes and reports success / failure.
#
# Idempotent: if PyPI already has the publisher, the re-run just publishes
# the existing tag (or no-ops if already uploaded). Safe to re-run.

set -euo pipefail

PROJECT_NAME="polyphonic-eval"
OWNER="hinanohart"
REPO="polyphonic-eval"
WORKFLOW_FILE="publish.yml"
ENV_NAME="pypi"
PUBLISHING_URL="https://pypi.org/manage/account/publishing/"

cd "$(dirname "$0")/.."

c_red()   { printf '\033[31m%s\033[0m\n' "$*"; }
c_grn()   { printf '\033[32m%s\033[0m\n' "$*"; }
c_yel()   { printf '\033[33m%s\033[0m\n' "$*"; }
c_cyn()   { printf '\033[36m%s\033[0m\n' "$*"; }
c_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

step() { c_bold ""; c_bold "==> $*"; }

# ── 1. prereqs ───────────────────────────────────────────────────────────
step "Step 1/5: Checking prerequisites"

if ! command -v gh >/dev/null 2>&1; then
  c_red "gh CLI not found. Install: https://cli.github.com/"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  c_red "gh CLI not authenticated. Run: gh auth login"
  exit 1
fi

ACTIVE_REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)
if [ "$ACTIVE_REPO" != "$OWNER/$REPO" ]; then
  c_yel "Warning: current gh default repo is '$ACTIVE_REPO', expected '$OWNER/$REPO'."
  c_yel "Continuing using explicit --repo flags."
fi

c_grn "  gh CLI: OK"
c_grn "  Auth:   OK"

# ── 2. show form ─────────────────────────────────────────────────────────
step "Step 2/5: PyPI Trusted Publisher form fields"

cat <<EOF
Open: $PUBLISHING_URL
Log in if prompted, then under "Add a new pending publisher" enter:

  ┌─────────────────────────────────────────────┐
  │ PyPI Project Name:      $PROJECT_NAME       │
  │ Owner (GitHub):         $OWNER              │
  │ Repository name:        $REPO               │
  │ Workflow filename:      $WORKFLOW_FILE      │
  │ Environment (optional): $ENV_NAME           │
  └─────────────────────────────────────────────┘

Click "Add". You should see the publisher listed under
"Pending publishers". That's all — no token, no copy/paste.

This script does NOT touch tokens or passwords; it only re-runs the
failed CI workflow once you confirm the form was submitted.
EOF

# ── 3. open browser ──────────────────────────────────────────────────────
step "Step 3/5: Opening PyPI publisher page in your browser"

opened=0
for opener in xdg-open open wslview powershell.exe; do
  if command -v "$opener" >/dev/null 2>&1; then
    if [ "$opener" = "powershell.exe" ]; then
      "$opener" -NoProfile -Command "Start-Process '$PUBLISHING_URL'" >/dev/null 2>&1 && opened=1 && break
    else
      "$opener" "$PUBLISHING_URL" >/dev/null 2>&1 && opened=1 && break
    fi
  fi
done

if [ "$opened" -eq 0 ]; then
  c_yel "Could not auto-open a browser. Visit manually: $PUBLISHING_URL"
fi

# ── 4. wait for confirmation ─────────────────────────────────────────────
step "Step 4/5: Confirm registration"

c_cyn "After you click 'Add' on the PyPI page, press Enter to continue."
c_cyn "(Or Ctrl-C to abort if you need to come back later.)"
read -r _

# Try latest FAILED publish run; fall back to latest publish run.
RUN_ID=$(gh run list \
  --repo "$OWNER/$REPO" \
  --workflow "$WORKFLOW_FILE" \
  --status failure \
  --limit 1 \
  --json databaseId -q '.[0].databaseId' 2>/dev/null || true)

if [ -z "${RUN_ID:-}" ] || [ "$RUN_ID" = "null" ]; then
  c_yel "No failed publish run found. Latest run anyway:"
  RUN_ID=$(gh run list \
    --repo "$OWNER/$REPO" \
    --workflow "$WORKFLOW_FILE" \
    --limit 1 \
    --json databaseId -q '.[0].databaseId' 2>/dev/null || true)
fi

if [ -z "${RUN_ID:-}" ] || [ "$RUN_ID" = "null" ]; then
  c_red "No publish workflow run found at all. Push a tag first:"
  c_red "    git tag vX.Y.Z && git push origin vX.Y.Z"
  exit 1
fi

c_grn "Target run: $RUN_ID"
c_grn "Re-running failed jobs..."

if ! gh run rerun --repo "$OWNER/$REPO" "$RUN_ID" --failed >/dev/null 2>&1; then
  # `--failed` only works on completed runs with failure; try plain rerun.
  c_yel "  --failed rerun rejected (run may already be in progress)."
  c_yel "  Trying full rerun..."
  gh run rerun --repo "$OWNER/$REPO" "$RUN_ID" >/dev/null
fi

# ── 5. watch & report ────────────────────────────────────────────────────
step "Step 5/5: Watching the rerun (this may take 1–3 minutes)"

# `gh run watch` follows until exit; exits non-zero on failure.
if gh run watch --repo "$OWNER/$REPO" --exit-status "$RUN_ID"; then
  c_grn ""
  c_grn "================================================================"
  c_grn "  PyPI publish SUCCESS for $PROJECT_NAME"
  c_grn "  https://pypi.org/project/$PROJECT_NAME/"
  c_grn "================================================================"
  exit 0
else
  status=$?
  c_red ""
  c_red "================================================================"
  c_red "  Rerun did not succeed (exit $status)."
  c_red "  Inspect logs:"
  c_red "    gh run view --repo $OWNER/$REPO $RUN_ID --log-failed"
  c_red "  If it shows 'invalid-publisher' again, the form on PyPI was"
  c_red "  not actually saved — re-open the page, verify the entry is"
  c_red "  listed, then re-run this script."
  c_red "================================================================"
  exit "$status"
fi
