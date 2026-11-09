#!/usr/bin/env python3
"""Repository secret / private-info scan (task 7.5). Exit 1 on any finding.

Portable (pure Python; the earlier shell version silently skipped .gz contents on
macOS because BSD grep treated the decompressed stream as binary).

Scans every tracked + untracked-not-ignored file:
  * text files: all patterns, including generic `password = ...`;
  * .gz files (raw results): decompressed and scanned for high-confidence credential
    formats and our own infrastructure identifiers only — model outputs routinely
    contain code with placeholder `password` variables.
Usage: python3 tools/secret_scan.py [--all-history]
"""
import gzip
import re
import subprocess
import sys

STRICT = {
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "github_token": r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}",
    "hf_token": r"\bhf_[A-Za-z0-9]{30,}",
    "openai_style_key": r"\bsk-[A-Za-z0-9]{20,}",
    "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
    "autodl_host": r"seetacloud\.com|autodl\.(com|pro):[0-9]+|connect\.[a-z0-9-]+\.(seetacloud|autodl)",
    "ssh_with_port": r"ssh -p [0-9]{4,5}",
    "root_at_host": r"root@[a-z0-9.-]+\.(com|cn|pro)\b",
    "autodl_secrets": r"jupyter_token|root_password",
    "user_email": r"3351666087abc",
}
TEXT_ONLY = {
    "ssh_config_port": r"^\s*Port [0-9]{4,5}\s*$",
    # the literal placeholder value "password" (quoted model output in LOG.md) is not a secret
    "password_assignment": r"(?i)\bpassword\s*[:=]\s*['\"]?(?!password\b)[^\s'\"$<{]{6,}",
}
SKIP = {"tools/secret_scan.py", "tools/secret_scan.sh"}


def files():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f and f not in SKIP]


def scan_text(name, text, pats):
    hits = []
    for k, p in pats.items():
        for m in re.finditer(p, text, re.M):
            ln = text.count("\n", 0, m.start()) + 1
            hits.append(f"{name}:{ln}: [{k}] {text[max(0, m.start() - 20):m.end() + 10]!r}"[:200])
    return hits


def scan_history():
    """Every blob reachable from any ref (all branches, tags, stashes), once per blob."""
    objs = subprocess.run(["git", "rev-list", "--objects", "--all"], capture_output=True, text=True).stdout
    paths = {}
    for line in objs.splitlines():
        sha, _, path = line.partition(" ")
        if path and path not in SKIP and not path.endswith((".png", ".jpg", ".parquet", ".nsys-rep", ".sqlite", ".pdf")):
            paths.setdefault(sha, path)
    import gzip as _gz
    proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    hits, n = [], 0
    for sha, path in paths.items():
        proc.stdin.write((sha + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) < 3 or header[1] != b"blob":
            if len(header) >= 3:
                proc.stdout.read(int(header[2]) + 1)
            continue
        data = proc.stdout.read(int(header[2]))
        proc.stdout.read(1)
        try:
            if path.endswith(".gz"):
                text, pats = _gz.decompress(data).decode("utf-8", "replace"), STRICT
            else:
                text, pats = data.decode("utf-8", "replace"), {**STRICT, **TEXT_ONLY}
        except (OSError, EOFError):
            continue
        n += 1
        hits += scan_text(f"{path}@{sha[:10]}", text, pats)
    proc.stdin.close()
    proc.wait()
    return hits, n


def main():
    if "--all-history" in sys.argv:
        hits, n = scan_history()
        if hits:
            print(f"SECRET SCAN (history): FINDINGS in {len({h.split(':')[0] for h in hits})} blobs")
            print("\n".join(hits[:200]))
            sys.exit(1)
        print(f"SECRET SCAN (history): clean ({n} blobs in all refs, gz contents included)")
        return
    hits, n = [], 0
    for f in files():
        if f.endswith((".png", ".jpg", ".parquet", ".nsys-rep", ".sqlite", ".pdf")):
            continue
        try:
            if f.endswith(".gz"):
                text = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
                pats = STRICT
            else:
                text = open(f, encoding="utf-8", errors="replace").read()
                pats = {**STRICT, **TEXT_ONLY}
        except (OSError, EOFError):
            continue
        n += 1
        hits += scan_text(f, text, pats)
    if hits:
        print("SECRET SCAN: FINDINGS")
        print("\n".join(hits[:200]))
        sys.exit(1)
    print(f"SECRET SCAN: clean ({n} files scanned, gz contents included)")


if __name__ == "__main__":
    main()
