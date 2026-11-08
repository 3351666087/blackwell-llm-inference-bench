#!/bin/bash
# Developer helper.
#   tools/sync.sh push   -> the GPU instance fetches the pushed commit from GitHub and checks it out
#                           (results/ and data/requests/*.gz are untracked and stay untouched)
#   tools/sync.sh pull   -> copy results back from the instance
#   tools/sync.sh rsync  -> legacy: rsync the working tree (slow uplink: avoid when .git is large)
# Uses the local SSH alias "bench" defined in ~/.ssh/autodl_bench.conf (not in repo).
set -euo pipefail
cd "$(dirname "$0")/.."
SSH="ssh -F $HOME/.ssh/autodl_bench.conf -o BatchMode=yes"
REMOTE_DIR=/root/autodl-tmp/bench-repo
case "${1:-push}" in
  push)
    head=$(git rev-parse HEAD)
    $SSH bench "cd $REMOTE_DIR && (source /etc/network_turbo >/dev/null 2>&1; git fetch -q origin main) && git reset -q --hard $head && git log --oneline -1" ;;
  pull) rsync -az -e "$SSH" --exclude '*.log.tmp' "bench:$REMOTE_DIR/results/" ./results/ ;;
  rsync) rsync -az -e "$SSH" --delete --exclude results/ --exclude .venv/ --exclude __pycache__/ ./ "bench:$REMOTE_DIR/" ;;
  *) echo "usage: $0 push|pull|rsync"; exit 2 ;;
esac
