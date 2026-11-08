"""Minimal OpenAI-compatible streaming server with a *known* timing model.

Used by tests/ to validate the load generator and metric code without a GPU:
  TTFT = ttft_ms (+ queueing if more than `slots` requests are in flight),
  then one token every `tpot_ms` until max_tokens.
The client must recover these numbers; tests/test_client.py checks that it does.
"""
import argparse
import asyncio
import json

from aiohttp import web


def make_app(ttft_ms: float, tpot_ms: float, slots: int):
    sem = asyncio.Semaphore(slots)

    async def health(_):
        return web.Response(text="ok")

    async def models(_):
        return web.json_response({"data": [{"id": "bench"}]})

    async def reset(_):
        return web.json_response({"ok": True})

    async def completions(req):
        body = await req.json()
        n = int(body["max_tokens"])
        n_in = len(body["prompt"].split())  # mock "tokenizer": whitespace words
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        async with sem:
            await asyncio.sleep(ttft_ms / 1000)
            for i in range(n):
                if i:
                    await asyncio.sleep(tpot_ms / 1000)
                fin = "length" if i == n - 1 else None
                ev = {"choices": [{"index": 0, "text": f" t{i}", "finish_reason": fin}]}
                await resp.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            ev = {"choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": n}}
            await resp.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.add_routes([web.get("/health", health), web.get("/v1/models", models),
                    web.post("/v1/completions", completions), web.post("/reset_prefix_cache", reset)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--ttft-ms", type=float, default=50)
    ap.add_argument("--tpot-ms", type=float, default=10)
    ap.add_argument("--slots", type=int, default=1000)
    a = ap.parse_args()
    web.run_app(make_app(a.ttft_ms, a.tpot_ms, a.slots), host="127.0.0.1", port=a.port, print=None)
