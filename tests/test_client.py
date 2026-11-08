"""Client + metrics + sanity checks against the mock server (no GPU needed).

The mock server has a known timing model, so the measured TTFT/TPOT must match it
within a small tolerance, token counts must match exactly, and the bandwidth-bound
checker must flag an impossible (too fast) run.
"""
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bib.client import ClientConfig, run_load  # noqa: E402
from bib import metrics, sanity  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_mock(ttft, tpot, slots=1000):
    port = free_port()
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "mock_server.py"), "--port", str(port),
                          "--ttft-ms", str(ttft), "--tpot-ms", str(tpot), "--slots", str(slots)])
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return p, port
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("mock server did not start")


def reqs(n, out=20, sessions=None):
    rows = []
    for i in range(n):
        r = {"id": f"r{i}", "prompt": "a b c d e", "input_tokens": 5, "max_tokens": out}
        if sessions:
            r["session"] = f"s{i % sessions}"
            r["turn"] = i // sessions
        rows.append(r)
    return rows


def test_timing_and_counts():
    p, port = start_mock(ttft=80, tpot=10)
    try:
        cfg = ClientConfig(f"http://127.0.0.1:{port}", "bench", concurrency=8, procs=2)
        res = run_load(cfg, reqs(32))
        recs = res["records"]
        m = metrics.run_metrics(recs, None, {}, {"ttft_ms": 1000, "tpot_ms": 50})
        assert m["n_err"] == 0, m
        assert abs(m["ttft_p50_ms"] - 80) < 15, m["ttft_p50_ms"]
        assert abs(m["tpot_p50_ms"] - 10) < 2.0, m["tpot_p50_ms"]
        assert m["output_tokens"] == 32 * 20
        assert m["slo_attainment"] == 1.0
        info = {"read_bytes_per_step": 1e9, "kv_bytes_per_token": 1e3}
        ok = sanity.check_run(recs, {"client": res["client"]}, info, bw_gbps=1000)  # bound 1 ms < 10 ms
        assert ok["token_counts"]["pass"] and ok["bandwidth_bound"]["pass"], ok
        bad = sanity.check_run(recs, {"client": res["client"]}, info, bw_gbps=10)   # bound 100 ms > 10 ms
        assert not bad["bandwidth_bound"]["pass"] and not bad["valid"]
    finally:
        p.terminate()


def test_sessions_are_sequential():
    p, port = start_mock(ttft=20, tpot=2)
    try:
        cfg = ClientConfig(f"http://127.0.0.1:{port}", "bench", concurrency=4, procs=1)
        res = run_load(cfg, reqs(16, out=5, sessions=4))
        by = {}
        for r in res["records"]:
            by.setdefault(r["session"], []).append(r)
        for s, rs in by.items():
            rs.sort(key=lambda r: r["turn"])
            for a, b in zip(rs, rs[1:]):
                assert b["t_send"] >= a["t_end"], "turn k+1 must start after turn k ends"
    finally:
        p.terminate()


if __name__ == "__main__":
    test_timing_and_counts()
    test_sessions_are_sequential()
    print("ok")
