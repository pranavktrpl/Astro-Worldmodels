#!/bin/bash
# Launch run_all_evals.sh inside a detached tmux session so it survives
# logout. Attach with: tmux attach -t <TMUX_SESSION>
#
#   bash Evals/cluster/tmux_launch.sh [path/to/evals.env]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/evals.env}"
[ -f "$ENV_FILE" ] || { echo "No env file at $ENV_FILE — cp evals.env.example evals.env and edit it."; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Session '$TMUX_SESSION' already exists — attach with: tmux attach -t $TMUX_SESSION"
    exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" \
    "bash '$SCRIPT_DIR/run_all_evals.sh' '$ENV_FILE'; status=\$?; echo; echo \"run_all_evals.sh exited with \$status — press enter to close\"; read"

echo "Started tmux session '$TMUX_SESSION'."
echo "  watch:   tmux attach -t $TMUX_SESSION   (detach: Ctrl-b d)"
echo "  logs:    $LOG_DIR"
