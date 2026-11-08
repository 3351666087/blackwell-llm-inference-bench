"""Closed-loop streaming load generator for OpenAI-compatible `/v1/completions`.

Design notes (these are part of the measurement method, see PREREGISTRATION.md):

* Closed loop: `concurrency` workers each keep exactly one request in flight.
  For agent workloads a worker owns a *session* and sends its turns strictly in
  order (turn k+1 is sent the moment turn k finishes; no think time).
* All three frameworks receive byte-identical prompt strings from the same request
  file (prompts are already chat-templated), with `max_tokens` = the target output
  length and `ignore_eos=True`, so every framework generates the same number of
  tokens for every request.
* Timestamps use `time.monotonic()` (CLOCK_MONOTONIC on Linux, shared by all
  processes on the host). Per request we store the send time, the arrival time of
  every SSE chunk that carried text, and the completion time.
* To keep the client from becoming the bottleneck at high concurrency, workers can
  be spread over several processes (`procs`). Each process records its own CPU
  time so the sanity checker can flag client saturation.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field

try:  # uvloop is optional; it lowers per-chunk client overhead
    import uvloop  # type: ignore
except Exception:  # pragma: no cover
    uvloop = None

import aiohttp


@dataclass
class ClientConfig:
    base_url: str               # e.g. http://127.0.0.1:8000
    model: str                  # served model name
    concurrency: int
    timeout_s: float = 1800.0
    procs: int = 1
    temperature: float = 0.0
    ignore_eos: bool = True
    extra_body: dict = field(default_factory=dict)
    keep_text: int = 0          # store full output text for the first N requests
    text_prefix_chars: int = 160


def load_requests(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_sessions(reqs: list[dict]) -> list[list[dict]]:
    """Requests with a `session` key are grouped and ordered by `turn`.
    Requests without one become single-request sessions."""
    sessions: dict = {}
    order = []
    for r in reqs:
        key = r.get("session")
        if key is None:
            key = ("__single__", r["id"])
        if key not in sessions:
            sessions[key] = []
            order.append(key)
        sessions[key].append(r)
    out = []
    for k in order:
        s = sessions[k]
        s.sort(key=lambda r: r.get("turn", 0))
        out.append(s)
    return out


async def _one_request(http: aiohttp.ClientSession, cfg: ClientConfig, req: dict, store_text: bool) -> dict:
    payload = {
        "model": cfg.model,
        "prompt": req["prompt"],
        "max_tokens": int(req["max_tokens"]),
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": req.get("ignore_eos", cfg.ignore_eos),
    }
    payload.update(cfg.extra_body)
    rec = {
        "id": req["id"],
        "session": req.get("session"),
        "turn": req.get("turn"),
        "exp_in": req.get("input_tokens"),
        "exp_out": int(req["max_tokens"]),
    }
    chunk_t: list[float] = []      # seconds since send, for chunks that carried text
    chunk_chars: list[int] = []
    text_parts: list[str] = []
    usage = None
    finish = None
    err = None
    t_send = time.monotonic()
    try:
        async with http.post(
            cfg.base_url + "/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                err = f"HTTP {resp.status}: {body[:400]}"
            else:
                async for raw in resp.content:
                    line = raw.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    now = time.monotonic()
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        err = f"bad SSE json: {data[:200]!r}"
                        break
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if "error" in obj and not obj.get("choices"):
                        err = f"stream error: {str(obj['error'])[:400]}"
                        break
                    for ch in obj.get("choices") or []:
                        txt = ch.get("text") or ""
                        if txt:
                            chunk_t.append(now - t_send)
                            chunk_chars.append(len(txt))
                            text_parts.append(txt)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except asyncio.TimeoutError:
        err = f"timeout after {cfg.timeout_s}s"
    except aiohttp.ClientError as e:
        err = f"client error: {type(e).__name__}: {e}"[:400]
    t_end = time.monotonic()

    text = "".join(text_parts)
    rec.update({
        "t_send": t_send,
        "t_end": t_end,
        "ttft": chunk_t[0] if chunk_t else None,
        "e2e": t_end - t_send,
        "chunk_t": [round(x, 6) for x in chunk_t],
        "n_chunks": len(chunk_t),
        "n_chars": len(text),
        "usage_in": (usage or {}).get("prompt_tokens"),
        "usage_out": (usage or {}).get("completion_tokens"),
        "finish": finish,
        "error": err,
        "text_head": text[: cfg.text_prefix_chars],
    })
    if store_text:
        rec["text"] = text
    if err is None and not chunk_t:
        rec["error"] = "no tokens received"
    return rec


async def _worker(http, cfg, queue: asyncio.Queue, out: list, text_budget: list):
    while True:
        try:
            session = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        for req in session:
            store = False
            if text_budget[0] > 0:
                text_budget[0] -= 1
                store = True
            rec = await _one_request(http, cfg, req, store)
            out.append(rec)
            if rec["error"]:
                break  # a failed turn ends its session; later turns would be meaningless


async def _run_async(cfg: ClientConfig, sessions: list[list[dict]], conc: int, keep_text: int) -> list[dict]:
    queue: asyncio.Queue = asyncio.Queue()
    for s in sessions:
        queue.put_nowait(s)
    out: list[dict] = []
    connector = aiohttp.TCPConnector(limit=0, force_close=False, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector, read_bufsize=1 << 20) as http:
        workers = [asyncio.create_task(_worker(http, cfg, queue, out, [keep_text if i == 0 else 0]))
                   for i in range(conc)]
        await asyncio.gather(*workers)
    return out


def _proc_main(cfg: ClientConfig, sessions, conc, keep_text, start_at, conn):
    # all processes start together so the offered load is a clean step
    delay = start_at - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    c0 = time.process_time()
    w0 = time.monotonic()
    coro = _run_async(cfg, sessions, conc, keep_text)
    if uvloop is not None:
        recs = asyncio.run(coro, loop_factory=uvloop.new_event_loop)
    else:
        recs = asyncio.run(coro)
    cpu = time.process_time() - c0
    wall = time.monotonic() - w0
    conn.send({"records": recs, "cpu_s": cpu, "wall_s": wall, "pid": os.getpid()})
    conn.close()


def run_load(cfg: ClientConfig, requests: list[dict]) -> dict:
    """Run one closed-loop load test. Returns records and client stats."""
    sessions = group_sessions(requests)
    procs = max(1, min(cfg.procs, cfg.concurrency))
    # round-robin shard sessions and concurrency over processes
    shards = [sessions[i::procs] for i in range(procs)]
    concs = [cfg.concurrency // procs + (1 if i < cfg.concurrency % procs else 0) for i in range(procs)]
    start_at = time.monotonic() + 1.0
    t_wall0 = time.time()
    t_mono0 = time.monotonic()
    if procs == 1:
        r, w = mp.Pipe(duplex=False)
        _proc_main(cfg, shards[0], concs[0], cfg.keep_text, start_at, w)
        results = [r.recv()]
    else:
        ctx = mp.get_context("fork")
        pipes, ps = [], []
        for i in range(procs):
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_proc_main,
                            args=(cfg, shards[i], concs[i], cfg.keep_text if i == 0 else 0, start_at, w))
            p.start()
            pipes.append(r)
            ps.append(p)
        results = [r.recv() for r in pipes]
        for p in ps:
            p.join()
    recs = [x for res in results for x in res["records"]]
    recs.sort(key=lambda x: x["t_send"])
    return {
        "records": recs,
        "client": {
            "procs": procs,
            "per_proc": [{"cpu_s": r["cpu_s"], "wall_s": r["wall_s"],
                          "cpu_frac": r["cpu_s"] / r["wall_s"] if r["wall_s"] > 0 else None}
                         for r in results],
            "clock": {"wall_at_mono0": t_wall0, "mono0": t_mono0},
        },
    }


def write_records(path: str, recs: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_records(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
