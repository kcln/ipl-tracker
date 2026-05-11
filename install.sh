#!/usr/bin/env bash
# IPL 2026 tracker — one-shot installer.
#
# Sets up venv, installs deps, configures launchd, prompts for iMessage
# recipient, and prints next steps. Idempotent: safe to re-run.

set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_SRC="$REPO_PATH/launchd/com.kcln.ipltracker.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.kcln.ipltracker.plist"

# Pick Python: prefer Homebrew 3.13/3.12/3.11, fall back to python3
PYTHON_BIN=""
for cand in /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11; do
    if [[ -x "$cand" ]]; then PYTHON_BIN="$cand"; break; fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

echo "==> Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# 1. venv + deps
if [[ ! -d "$REPO_PATH/venv" ]]; then
    echo "==> Creating venv..."
    "$PYTHON_BIN" -m venv "$REPO_PATH/venv"
fi
echo "==> Installing dependencies..."
"$REPO_PATH/venv/bin/pip" install --quiet --upgrade pip
"$REPO_PATH/venv/bin/pip" install --quiet -r "$REPO_PATH/requirements.txt"

# 2. iMessage recipient
RECIPIENT="${IMESSAGE_RECIPIENT:-}"
if [[ -z "$RECIPIENT" ]]; then
    if grep -q "IMESSAGE_RECIPIENT" "$HOME/.zshrc" 2>/dev/null; then
        echo "==> IMESSAGE_RECIPIENT already in ~/.zshrc"
        RECIPIENT="$(grep "IMESSAGE_RECIPIENT" "$HOME/.zshrc" | head -1 | sed -E 's/.*IMESSAGE_RECIPIENT="?([^"]+)"?.*/\1/')"
    else
        printf "Enter the iMessage recipient (your phone in +14155551234 or Apple-ID email): "
        read -r RECIPIENT
        printf '\nexport IMESSAGE_RECIPIENT="%s"\n' "$RECIPIENT" >> "$HOME/.zshrc"
        echo "==> Wrote IMESSAGE_RECIPIENT to ~/.zshrc (open a new shell to pick it up)"
    fi
fi

# 3. Render plist with absolute paths
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
# CricAPI key is optional — empty string disables CricAPI tier
CRICAPI_KEY_VAL="${CRICAPI_KEY:-}"
sed \
    -e "s|__REPO_PATH__|$REPO_PATH|g" \
    -e "s|__HOME__|$HOME|g" \
    -e "s|__IMESSAGE_RECIPIENT__|$RECIPIENT|g" \
    -e "s|__CRICAPI_KEY__|$CRICAPI_KEY_VAL|g" \
    "$PLIST_SRC" > "$PLIST_DST"

# 4. (Re)load launchd job
if launchctl list 2>/dev/null | grep -q com.kcln.ipltracker; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi
launchctl load "$PLIST_DST"
echo "==> launchd job loaded: $PLIST_DST"

# 5. Print next steps
cat <<EOF

──────────────────────────────────────────────────────────────────
Install complete.

Next steps:

  1. Grant Automation permission so the script can drive Messages.app:
     System Settings → Privacy & Security → Automation
       → enable "osascript → Messages" (and "Terminal → Messages"
         if you'll run it manually).

  2. Create the GitHub repo and push:
       gh repo create kcln/ipl-tracker --public --source="$REPO_PATH" --remote=origin --push
     Then enable GitHub Pages: Settings → Pages → source = main /docs

  3. Tail the logs:
       tail -f ~/Library/Logs/ipl-tracker.log

  4. Trigger a run now:
       "$REPO_PATH/venv/bin/python3" "$REPO_PATH/src/tracker.py"

  5. Manage the launchd job:
       launchctl unload  $PLIST_DST   # stop
       launchctl load    $PLIST_DST   # start
──────────────────────────────────────────────────────────────────
EOF
