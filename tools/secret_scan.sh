#!/bin/bash
# Repository secret / private-info scan (task 7.5). Exit 1 on any finding.
# Scans tracked + untracked-but-not-ignored files.
set -uo pipefail
cd "$(dirname "$0")/.."
files=$(git ls-files -co --exclude-standard 2>/dev/null | grep -v -E '\.(gz|parquet|png|jpg|qdrep|nsys-rep|sqlite)$' | grep -v '^tools/secret_scan.sh$')
pat='(-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|seetacloud\.com|autodl\.(com|pro):[0-9]+|connect\.[a-z0-9-]+\.(seetacloud|autodl)|ssh -p [0-9]{4,5}|root@[a-z0-9.-]+\.(com|cn|pro)|Port [0-9]{4,5}|jupyter_token|root_password|password *[:=] *[^ ]{4,}|3351666087abc)'
hits=$(echo "$files" | tr '\n' '\0' | xargs -0 grep -n -I -E "$pat" 2>/dev/null)
# gz files: scan decompressed content too (results / request files)
gzhits=$(git ls-files -co --exclude-standard | grep '\.gz$' | while read f; do gzip -dc "$f" 2>/dev/null | grep -l -I -E "$pat" >/dev/null && echo "$f"; done)
if [ -n "$hits$gzhits" ]; then
  echo "SECRET SCAN: FINDINGS"; echo "$hits" | head -50; echo "$gzhits"
  exit 1
fi
echo "SECRET SCAN: clean ($(echo "$files" | wc -l | tr -d ' ') text files + gz files scanned)"
