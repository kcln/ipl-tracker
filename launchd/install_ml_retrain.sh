#!/usr/bin/env bash
# Install (or reinstall) the nightly ML retrain launchd job.
#
# Templates launchd/com.kcln.ipl-ml-retrain.plist by replacing
#   __REPO_PATH__ -> absolute repo path
#   __HOME__      -> $HOME
# Writes the rendered plist to ~/Library/LaunchAgents/ and bootstraps it
# into the user's gui domain. Idempotent: safely re-runs.
#
# Usage:
#   bash launchd/install_ml_retrain.sh           # render + load
#   bash launchd/install_ml_retrain.sh --dry-run # render only, do not load
#
# After install:
#   tail -f ~/Library/Logs/ipl-ml-retrain.log
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
LABEL="com.kcln.ipl-ml-retrain"
TEMPLATE="$SCRIPT_DIR/$LABEL.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET="$TARGET_DIR/$LABEL.plist"
GUI_TARGET="gui/$(id -u)/$LABEL"

if [ ! -f "$TEMPLATE" ]; then
    echo "template not found: $TEMPLATE" >&2
    exit 1
fi

VENV_PY="$REPO_PATH/ml/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "warning: $VENV_PY not found — the launchd job will fail until the ml venv is built." >&2
fi

mkdir -p "$TARGET_DIR"

# Render template -> target plist (atomic write).
TMP="$(mktemp "$TARGET.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
sed \
    -e "s|__REPO_PATH__|$REPO_PATH|g" \
    -e "s|__HOME__|$HOME|g" \
    "$TEMPLATE" > "$TMP"

if ! plutil -lint "$TMP" >/dev/null; then
    echo "rendered plist failed plutil lint" >&2
    exit 1
fi

mv "$TMP" "$TARGET"
trap - EXIT
echo "rendered -> $TARGET"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "dry-run: skipping launchctl bootstrap"
    exit 0
fi

launchctl bootout "$GUI_TARGET" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
echo "loaded $GUI_TARGET"
launchctl print "$GUI_TARGET" | head -20
