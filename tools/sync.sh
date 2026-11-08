#!/bin/bash
# Developer helper: push the working tree to the GPU instance (code only; never
# deletes remote results) or pull results back.
#   tools/sync.sh push | pull
# Uses the local SSH alias "bench" defined in ~/.ssh/autodl_bench.conf (not in repo).
set -euo pipefail
cd "$(dirname "$0")/.."
SSH="ssh -F $HOME/.ssh/autodl_bench.conf -o BatchMode=yes"
REMOTE=bench:/root/autodl-tmp/bench-repo
case "${1:-push}" in
  push) rsync -az -e "$SSH" --delete --exclude results/ --exclude .venv/ --exclude __pycache__/ ./ "$REMOTE/" ;;
  pull) rsync -az -e "$SSH" --exclude '*.log.tmp' "$REMOTE/results/" ./results/ ;;
  *) echo "usage: $0 push|pull"; exit 2 ;;
esac
