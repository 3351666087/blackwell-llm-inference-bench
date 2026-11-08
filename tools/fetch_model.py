#!/usr/bin/env python3
"""Download a model snapshot from ModelScope at a pinned commit, with aria2c
multi-connection downloads and SHA-256 verification against the hub's metadata.

Writes <dest>/MODEL_MANIFEST.json (source, repo, revision, per-file sha256/size)
so every benchmark result can be traced to exact weights.

Usage:
  python tools/fetch_model.py Qwen/Qwen3-8B /root/autodl-tmp/models/Qwen3-8B [--revision <commit>]
Only the Python standard library and the `aria2c` binary are required.
"""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.request

API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev}&Recursive=true"
# NOTE: /models/<repo>/resolve/<commit>/<file> returns the git-LFS *pointer* for large
# files, so we use the repo API endpoint and verify SHA-256 against the pinned listing.
URL = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision={rev}&FilePath={path}"


def list_files(repo, rev):
    with urllib.request.urlopen(API.format(repo=repo, rev=rev), timeout=60) as r:
        data = json.load(r)
    files = [f for f in data["Data"]["Files"] if f["Type"] == "blob"]
    if not files:
        sys.exit(f"no files for {repo}@{rev}: {str(data)[:300]}")
    return files


def sha256(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--connections", type=int, default=16)
    a = ap.parse_args()

    files = list_files(a.repo, a.revision)
    # pin to the commit the listing resolved to, so the manifest is exact
    commit = files[0]["Revision"]
    os.makedirs(a.dest, exist_ok=True)
    todo = []
    for f in files:
        p = os.path.join(a.dest, f["Path"])
        if os.path.exists(p) and os.path.getsize(p) == f["Size"] and not os.path.exists(p + ".aria2"):
            continue
        if os.path.exists(p) and not os.path.exists(p + ".aria2"):
            os.remove(p)  # wrong size (e.g. an LFS pointer from an earlier attempt)
        todo.append(f)
    if todo:
        lst = os.path.join(a.dest, ".aria2_input.txt")
        with open(lst, "w") as fh:
            for f in todo:
                fh.write(URL.format(repo=a.repo, rev=a.revision, path=f["Path"]) + "\n")
                fh.write(f"  out={f['Path']}\n  dir={a.dest}\n")
        t0 = time.time()
        cmd = ["aria2c", "-i", lst, "-x", str(a.connections), "-s", str(a.connections),
               "-k", "8M", "-j", "4", "--continue=true", "--max-tries=20", "--retry-wait=5",
               "--file-allocation=none", "--console-log-level=warn", "--summary-interval=60"]
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit(f"aria2c failed rc={rc}")
        os.remove(lst)
        print(f"downloaded {len(todo)} files in {time.time()-t0:.0f}s", flush=True)

    manifest = {"source": "modelscope.cn", "repo": a.repo, "revision": commit,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": []}
    bad = []
    for f in files:
        p = os.path.join(a.dest, f["Path"])
        got = sha256(p)
        ok = (got == f.get("Sha256")) if f.get("Sha256") else None
        if ok is False:
            bad.append(f["Path"])
        manifest["files"].append({"path": f["Path"], "size": f["Size"], "sha256": got,
                                  "hub_sha256": f.get("Sha256"), "verified": ok})
    with open(os.path.join(a.dest, "MODEL_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    if bad:
        sys.exit(f"SHA256 MISMATCH: {bad}")
    print(f"OK {a.repo}@{commit}: {len(files)} files verified", flush=True)


if __name__ == "__main__":
    main()
