#!/usr/bin/env bash
# Manage iMessage recipients for the IPL tracker.
#
# Edits recipients.txt — no launchd reload needed, the next 15-min run
# picks it up automatically.
#
# Usage:
#   ./recipients.sh list
#   ./recipients.sh add    +14155551234
#   ./recipients.sh add    friend@icloud.com
#   ./recipients.sh remove +14155551234

set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE="$REPO_PATH/recipients.txt"

cmd="${1:-list}"
arg="${2:-}"

ensure_file() {
    if [[ ! -f "$FILE" ]]; then
        cat > "$FILE" <<'EOF'
# iMessage recipients for IPL tracker. One per line.
# Phones: +14155551234 (E.164, with country code).
# Emails: any Apple-ID email tied to iMessage.
# Lines starting with # are comments. Edit anytime — picked up next run.

EOF
    fi
}

case "$cmd" in
    list)
        ensure_file
        echo "Recipients in $FILE:"
        grep -vE '^\s*(#|$)' "$FILE" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
        ;;

    add)
        if [[ -z "$arg" ]]; then echo "usage: $0 add <handle>"; exit 2; fi
        ensure_file
        if grep -qxF "$arg" "$FILE"; then
            echo "already present: $arg"
            exit 0
        fi
        printf '%s\n' "$arg" >> "$FILE"
        echo "added: $arg"
        ;;

    remove|rm|delete)
        if [[ -z "$arg" ]]; then echo "usage: $0 remove <handle>"; exit 2; fi
        if [[ ! -f "$FILE" ]]; then echo "no recipients.txt yet"; exit 0; fi
        # POSIX-safe in-place edit
        tmp=$(mktemp)
        grep -vxF "$arg" "$FILE" > "$tmp" || true
        mv "$tmp" "$FILE"
        echo "removed: $arg"
        ;;

    test)
        # Send a one-line self-test to every recipient currently in the file
        if [[ ! -x "$REPO_PATH/venv/bin/python3" ]]; then
            echo "venv missing — run ./install.sh first"; exit 1
        fi
        "$REPO_PATH/venv/bin/python3" -c "
from src import imessage_sender
from datetime import datetime
ok = imessage_sender.send('ipl-tracker recipients test ' + datetime.now().isoformat(timespec='seconds'))
import sys; sys.exit(0 if ok else 1)
"
        ;;

    *)
        cat <<EOF
usage:
  $0 list                    show current recipients
  $0 add <handle>            add a recipient (phone +E.164 or Apple-ID email)
  $0 remove <handle>         remove a recipient
  $0 test                    send a self-test line to every recipient

Examples:
  $0 add +14155551234
  $0 add friend@icloud.com
  $0 remove +14155551234
EOF
        exit 2
        ;;
esac
