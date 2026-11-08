"""Smoke test for one framework (task 2.6). Not a benchmark: one server, a few
requests, and protocol checks the benchmark relies on.

  python tools/smoke.py --framework vllm --env /root/envs/vllm --model /root/autodl-tmp/models/Qwen3-8B \
      --out results/smoke/vllm-q8b-bf16

Checks:
  1. server starts and answers /health and /v1/models
  2. three greedy chat prompts -> texts saved for human review (must be sensible)
  3. streaming + stream_options.include_usage + ignore_eos: completion_tokens == max_tokens,
     prompt_tokens == our tokenizer count
  4. cache reset endpoint (vLLM/SGLang)
  5. our pre-rendered ChatML string == tokenizer.apply_chat_template(..., enable_thinking=False)
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bib.client import ClientConfig, run_load  # noqa: E402
from bib.servers import Server  # noqa: E402
from bib.workloads import Tok, chatml_user  # noqa: E402

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Show the calculation briefly.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
]

TEMPLATE_CHECK = r'''
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
msgs = json.loads(sys.argv[2])
out = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False) for m in msgs]
print(json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="extra server args, space separated")
    ap.add_argument("--env-var", action="append", default=[], help="KEY=VALUE for the server process")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {"framework": a.framework, "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    tok = Tok(os.path.join(a.model, "tokenizer.json"))

    # 5. chat template equivalence (uses the framework env's transformers)
    try:
        r = subprocess.run([os.path.join(a.env, "bin", "python"), "-c", TEMPLATE_CHECK, a.model, json.dumps(PROMPTS)],
                           capture_output=True, text=True, timeout=300)
        official = json.loads(r.stdout.strip().splitlines()[-1])
        ours = [chatml_user(p) for p in PROMPTS]
        report["checks"]["chat_template_identical"] = official == ours
        if official != ours:
            report["template_diff"] = {"official": official[0], "ours": ours[0]}
    except Exception as e:
        report["checks"]["chat_template_identical"] = f"error: {e} {r.stderr[-500:] if 'r' in dir() else ''}"

    cfg = {"framework": a.framework, "env": a.env, "model": a.model, "max_model_len": 40960,
           "extra_args": a.extra.split() if a.extra else [],
           "env_vars": dict(kv.split("=", 1) for kv in a.env_var)}
    report["server_env_vars"] = cfg["env_vars"]
    srv = Server(cfg, a.out, port=8000)
    try:
        t = srv.start(3600)
        report["load_s"] = t
        report["checks"]["server_up"] = True
        outs = []
        for p in PROMPTS:
            resp = srv.complete(chatml_user(p), 200)
            outs.append({"prompt": p, "text": resp["choices"][0]["text"], "usage": resp.get("usage"),
                         "finish": resp["choices"][0].get("finish_reason")})
        report["greedy_outputs"] = outs
        report["checks"]["paris_in_answer"] = "Paris" in outs[0]["text"]
        report["checks"]["speed_80_in_answer"] = "80" in outs[1]["text"]
        report["checks"]["def_in_code_answer"] = "def " in outs[2]["text"]

        # 3. streaming usage + ignore_eos
        p = chatml_user("Say hi.")
        req = [{"id": "s0", "prompt": p, "input_tokens": tok.n(p), "max_tokens": 64}]
        res = run_load(ClientConfig(srv.base, srv.model_name, 1), req)["records"][0]
        report["stream_record"] = {k: res.get(k) for k in ("usage_in", "usage_out", "exp_in", "exp_out", "n_chunks",
                                                           "finish", "error", "ttft", "e2e", "text_head")}
        report["checks"]["stream_usage_present"] = res.get("usage_out") is not None
        report["checks"]["ignore_eos_exact_len"] = res.get("usage_out") == 64
        report["checks"]["prompt_tokens_match"] = res.get("usage_in") == tok.n(p)
        report["checks"]["one_chunk_per_token"] = res.get("n_chunks") == res.get("usage_out")

        # 4. cache reset
        try:
            report["checks"]["cache_reset"] = srv.reset_cache()
        except Exception as e:
            report["checks"]["cache_reset"] = f"FAILED: {e}"
    except Exception as e:
        report["checks"]["server_up"] = False
        report["error"] = str(e)
        report["log_tail"] = srv.log_tail(80)
    finally:
        srv.stop()
    json.dump(report, open(os.path.join(a.out, "smoke_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report["checks"], indent=1))
    for o in report.get("greedy_outputs", []):
        print("---", o["prompt"], "\n", o["text"][:400])


if __name__ == "__main__":
    main()
