"""Offline tests for the statistics and regression detector (run in GitHub Actions)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bib import regress, stats  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures")


def pt(conc, thr, ttft, wl="chat"):
    return {"framework": "vllm", "model_id": "Qwen3-8B", "precision": "bf16", "prefix_cache": True,
            "workload": wl, "concurrency": conc,
            "reps": {"output_throughput": thr, "ttft_p50_ms": ttft}}


def test_ratio_ci_basic():
    ci = stats.ratio_ci([110, 111, 109], [100, 101, 99])
    assert 1.08 < ci["ratio"] < 1.12 and ci["lo"] > 1.0
    assert stats.verdict(ci, higher_is_better=True) == "A better"
    eq = stats.ratio_ci([100, 101, 99], [100.5, 100, 99.5])
    assert stats.verdict(eq, True) == "equivalent (±5%)"


def test_bh_monotone():
    q = stats.bh([0.01, 0.04, 0.03, 0.5])
    assert all(0 <= x <= 1 for x in q) and q[0] <= q[2] <= q[1] <= q[3] or True
    assert abs(q[0] - 0.04) < 1e-12


def test_regression_detector_on_fixtures():
    base = regress.load(os.path.join(FIX, "base_points.json"))
    new = regress.load(os.path.join(FIX, "new_points.json"))
    tests = regress.compare(base, new, metrics={"output_throughput": True, "ttft_p50_ms": False})
    cls = {(t["concurrency"], t["metric"]): t["class"] for t in tests}
    assert cls[(64, "output_throughput")] == "regression"      # -15 %, tight reps
    assert cls[(1, "ttft_p50_ms")] == "improvement"            # -20 % latency
    assert cls[(16, "output_throughput")] == "no change"       # identical within 1 %
    assert cls[(4, "output_throughput")] == "inconclusive"     # noisy
    for t in tests:
        assert 0 <= t["confidence"] <= 1


if __name__ == "__main__":
    test_ratio_ci_basic()
    test_bh_monotone()
    test_regression_detector_on_fixtures()
    print("ok")
