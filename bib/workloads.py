"""Deterministic request-file generator.

Workloads (all prompts are pre-rendered Qwen3 ChatML strings, non-thinking mode,
so every framework receives byte-identical prompts):

  chat       single-turn ShareGPT V3 prompts; output length = token length of the
             real first GPT reply (vLLM benchmark_serving filtering: prompt 4..1024
             tokens, prompt+output <= 2048).
  prefill8k / prefill16k / prefill32k
             long-context prefill: a unique document header + ShareGPT text taken
             from conversations disjoint from the chat set, cut to an exact token
             budget (8000/16000/32000 input tokens), output 128 tokens.
  agent      synthetic multi-turn tool-calling trajectories (method below).
  quality    GSM8K test questions for the output-quality check (natural EOS).

Agent trajectory synthesis (seeded, documented in README):
  * one shared system prompt: agent instructions + 16 JSON tool schemas in the
    Qwen3 `<tools>` format (~2.6k tokens), identical for all sessions;
  * user task = a ShareGPT human prompt (<=300 tokens);
  * T ~ U{4..12} turns; assistant turn k has a_k ~ clip(LogNormal(ln 120, 0.5), 32, 400)
    tokens (a short thought + a `<tool_call>` JSON); tool result r_k ~
    clip(LogNormal(ln 900, 0.8), 80, 4000) tokens of ShareGPT text inside
    `<tool_response>`; a session stops early if its context would exceed 24k tokens;
  * request k of a session = full history up to turn k, `max_tokens = a_k`.
    The history uses the *recorded* assistant text, not the live model output, so
    all frameworks see identical prefixes. Prompt k is a strict string prefix of
    prompt k+1 (stored as deltas to keep files small).

Every file is written with gzip mtime=0 and sorted JSON keys, so re-running the
generator with the same sources/seed/tokenizer reproduces identical SHA-256s.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import re

GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chatml_user(text: str, system: str | None = None) -> str:
    s = ""
    if system:
        s += f"<|im_start|>system\n{system}<|im_end|>\n"
    return s + f"<|im_start|>user\n{text}<|im_end|>\n" + GEN


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl_gz(path: str, rows: list[dict]) -> str:
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    with open(path, "wb") as f:
        f.write(raw.getvalue())
    return sha256_file(path)


class Tok:
    def __init__(self, tokenizer_json: str):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(tokenizer_json)
        self.sha256 = sha256_file(tokenizer_json)

    def n(self, text: str) -> int:
        return len(self.t.encode(text, add_special_tokens=False).ids)

    def ids(self, text: str) -> list[int]:
        return self.t.encode(text, add_special_tokens=False).ids

    def dec(self, ids: list[int]) -> str:
        return self.t.decode(ids, skip_special_tokens=False)


def load_sharegpt(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    seen = set()
    for c in data:
        conv = c.get("conversations") or []
        if len(conv) < 2 or conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        p, r = conv[0]["value"], conv[1]["value"]
        if not p.strip() or not r.strip() or p in seen:
            continue
        seen.add(p)
        out.append({"id": c.get("id"), "prompt": p, "reply": r,
                    "all_text": "\n\n".join(m["value"] for m in conv if m.get("value"))})
    return out


SPECIAL = re.compile(r"<\|(im_start|im_end|endoftext)\|>|</?(tool_call|tool_response|think|tools)>")


def clean(text: str) -> str:
    """Remove strings that would be parsed as Qwen special/control tokens."""
    return SPECIAL.sub(" ", text)


# ---------------------------------------------------------------- chat
def build_chat(convs, tok: Tok, n: int, rng: random.Random, prefix: str):
    rows = []
    for c in convs:
        prompt = chatml_user(clean(c["prompt"]))
        n_in = tok.n(prompt)
        n_out = tok.n(clean(c["reply"]))
        if n_in < 4 or n_out < 4 or n_in > 1024 or n_in + n_out > 2048:
            continue
        rows.append({"id": f"{prefix}{len(rows):05d}", "prompt": prompt, "input_tokens": n_in,
                     "max_tokens": n_out, "src": c["id"]})
        if len(rows) == n:
            break
    return rows


# ---------------------------------------------------------------- long prefill
class Corpus:
    """Token stream made of ShareGPT conversation text (cleaned)."""

    def __init__(self, convs, tok: Tok, max_tokens: int):
        ids = []
        for c in convs:
            ids.extend(tok.ids(clean(c["all_text"]) + "\n\n"))
            if len(ids) >= max_tokens:
                break
        self.ids = ids
        self.tok = tok

    def text(self, rng: random.Random, n_tokens: int) -> str:
        start = rng.randrange(0, max(1, len(self.ids) - n_tokens - 1))
        return self.tok.dec(self.ids[start:start + n_tokens])


def exact_prompt(tok: Tok, corpus: Corpus, rng, target: int, header: str, question: str):
    """Build a ChatML prompt whose token count is exactly `target` (or within 8)."""
    overhead = tok.n(chatml_user(header + question))
    body_len = target - overhead
    start = rng.randrange(0, len(corpus.ids) - body_len - 64)
    best = None
    for _ in range(8):
        body = tok.dec(corpus.ids[start:start + body_len])
        p = chatml_user(header + clean(body) + question)
        n = tok.n(p)
        best = (p, n)
        if n == target:
            break
        body_len += target - n
    return best


def build_prefill(corpus: Corpus, tok: Tok, n: int, target: int, out_len: int, rng, prefix: str):
    rows = []
    for i in range(n):
        uid = "%032x" % rng.getrandbits(128)
        header = f"Document ID {uid}. Read the document carefully.\n\n"
        question = "\n\nQuestion: Summarize the document above in three sentences."
        p, n_in = exact_prompt(tok, corpus, rng, target, header, question)
        rows.append({"id": f"{prefix}{i:04d}", "prompt": p, "input_tokens": n_in, "max_tokens": out_len})
    return rows


# ---------------------------------------------------------------- agent
TOOLS = [
    ("read_file", "Read a UTF-8 text file from the repository and return its contents with line numbers.",
     {"path": "string", "start_line": "integer", "end_line": "integer"}),
    ("write_file", "Create or overwrite a file with the given content.", {"path": "string", "content": "string"}),
    ("edit_file", "Replace an exact snippet in a file with new text; fails if the snippet is not unique.",
     {"path": "string", "old": "string", "new": "string"}),
    ("list_dir", "List files and directories under a path, optionally recursively.",
     {"path": "string", "recursive": "boolean", "max_entries": "integer"}),
    ("grep", "Search files for a regular expression and return matching lines with file names.",
     {"pattern": "string", "path": "string", "case_sensitive": "boolean"}),
    ("find_symbol", "Locate the definition of a function, class or variable in the code base.",
     {"name": "string", "language": "string"}),
    ("run_shell", "Run a shell command in the project sandbox and return stdout, stderr and exit code.",
     {"command": "string", "timeout_s": "integer", "cwd": "string"}),
    ("run_tests", "Run the project's test suite or a subset selected by a pattern.",
     {"selector": "string", "verbose": "boolean"}),
    ("python_exec", "Execute a Python snippet in a persistent interpreter and return the result.",
     {"code": "string"}),
    ("git_diff", "Show the working-tree diff, optionally limited to a path.", {"path": "string", "staged": "boolean"}),
    ("git_commit", "Commit staged changes with a message.", {"message": "string", "add_all": "boolean"}),
    ("web_search", "Search the web and return the top results with titles, URLs and snippets.",
     {"query": "string", "num_results": "integer"}),
    ("fetch_url", "Download a web page and return its readable text.", {"url": "string", "max_chars": "integer"}),
    ("open_issue", "Open an issue in the project tracker.", {"title": "string", "body": "string", "labels": "array"}),
    ("ask_user", "Ask the user a clarifying question and wait for the answer.", {"question": "string"}),
    ("submit", "Finish the task and report the final answer or summary.", {"summary": "string", "status": "string"}),
]

AGENT_RULES = """You are a careful autonomous software engineering agent working inside a code repository.
Follow these rules strictly:
1. Understand the task before acting. Read the relevant files and search the code base before you edit anything.
2. Prefer small, verifiable steps. After each change, run the relevant tests or commands and read their output.
3. Never invent file contents, command output or test results; only rely on tool results you have actually received.
4. When a command fails, read the error message, form a hypothesis, and test it with the smallest possible experiment.
5. Keep edits minimal and consistent with the surrounding style. Do not reformat unrelated code.
6. Do not access the network unless the task requires it, and never exfiltrate secrets or credentials.
7. When requirements are ambiguous, state your assumption explicitly or ask the user a clarifying question.
8. Before finishing, review the full diff, make sure all tests pass, and summarize what changed and why.
9. Report negative results honestly. If something could not be done, say exactly what blocked you.
10. Use exactly one tool call per turn and wait for its result before deciding the next step.
"""


def tools_block() -> str:
    lines = []
    for name, desc, params in TOOLS:
        props = {k: {"type": v, "description": f"The {k.replace('_', ' ')} argument for {name}."} for k, v in params.items()}
        lines.append(json.dumps({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(params)[:1]}}},
            ensure_ascii=False))
    return ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            + "\n".join(lines) +
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
            "\"arguments\": <args-json-object>}\n</tool_call>")


def agent_system(tok: Tok, corpus: Corpus, rng) -> str:
    # pad the instructions with a fixed "project context" section to ~1.2k tokens,
    # mimicking repo-specific guidance files that agents load into the system prompt
    ctx = clean(corpus.text(rng, 900))
    return ("<|im_start|>system\n" + AGENT_RULES + "\n# Project context\n" + ctx + tools_block()
            + "<|im_end|>\n")


def clip_lognormal(rng, median, sigma, lo, hi):
    return int(min(hi, max(lo, round(math.exp(rng.gauss(math.log(median), sigma))))))


def build_agent(convs, corpus: Corpus, tok: Tok, n_sessions: int, rng, system: str, prefix: str,
                max_ctx: int = 24000):
    rows = []
    names = [t[0] for t in TOOLS]
    ci = 0
    for s in range(n_sessions):
        while True:
            task = clean(convs[ci]["prompt"])
            ci += 1
            ids = tok.ids(task)
            if len(ids) >= 8:
                break
        task = tok.dec(ids[:300])
        prompt = system + f"<|im_start|>user\nTask: {task}<|im_end|>\n" + GEN
        prev = ""
        T = rng.randint(4, 12)
        sid = f"{prefix}{s:04d}"
        for k in range(T):
            a_k = clip_lognormal(rng, 120, 0.5, 32, 400)
            n_in = tok.n(prompt)
            if n_in + a_k > max_ctx:
                break
            rows.append({"id": f"{sid}-t{k:02d}", "session": sid, "turn": k,
                         "prompt_delta": prompt[len(prev):], "input_tokens": n_in, "max_tokens": a_k})
            prev = prompt
            # recorded assistant turn (thought + tool call) of ~a_k tokens
            tool = rng.choice(names)
            thought_len = max(8, a_k - 40)
            thought = clean(corpus.text(rng, thought_len)).strip()
            arg = clean(corpus.text(rng, 12)).strip().replace("\n", " ")
            assistant = (thought + "\n<tool_call>\n" +
                         json.dumps({"name": tool, "arguments": {"query": arg}}, ensure_ascii=False) +
                         "\n</tool_call>")
            r_k = clip_lognormal(rng, 900, 0.8, 80, 4000)
            result = clean(corpus.text(rng, r_k))
            prompt = (prompt + assistant + "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result
                      + "\n</tool_response><|im_end|>\n" + GEN)
    return rows


def group_sessions_for_subset(rows: list[dict], n_sessions: int, by_session: bool = True) -> list[dict]:
    """Requests of the first `n_sessions` sessions (in file order). Rows without a
    `session` key count as one-request sessions."""
    keep, out = set(), []
    for r in rows:
        key = r.get("session", r["id"])
        if key not in keep:
            if len(keep) >= n_sessions:
                continue
            keep.add(key)
        out.append(r)
    return out


def n_sessions(rows: list[dict]) -> int:
    return len({r.get("session", r["id"]) for r in rows})


def expand_prompts(rows: list[dict]) -> list[dict]:
    """Rebuild full prompts for rows stored as per-session deltas."""
    acc: dict = {}
    out = []
    for r in rows:
        if "prompt_delta" in r:
            sid = r["session"]
            acc[sid] = acc.get(sid, "") + r["prompt_delta"]
            r = dict(r)
            r["prompt"] = acc[sid]
            del r["prompt_delta"]
        out.append(r)
    return out


# ---------------------------------------------------------------- quality (GSM8K)
def build_quality(parquet_path: str, tok: Tok, n: int, rng, prefix: str):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path).to_pylist()
    idx = list(range(len(t)))
    rng.shuffle(idx)
    rows = []
    for i in idx[:n]:
        q = t[i]["question"]
        ans = t[i]["answer"].split("####")[-1].strip().replace(",", "")
        p = chatml_user(q + "\nPlease reason step by step, and put your final numeric answer after \"####\".")
        rows.append({"id": f"{prefix}{len(rows):04d}", "prompt": p, "input_tokens": tok.n(p),
                     "max_tokens": 1024, "ignore_eos": False, "answer": ans, "gsm8k_index": i})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--out", default="data/requests")
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = Tok(a.tokenizer)
    rng = random.Random(a.seed)
    convs = load_sharegpt(a.sharegpt)
    rng.shuffle(convs)
    # disjoint slices of the shuffled conversation list
    print("sharegpt conversations after filtering/dedup:", len(convs), flush=True)
    # disjoint slices: chat | chat warmup | long-context corpus | agent tasks | agent warmup tasks
    chat_src, warm_src = convs[:20000], convs[20000:24000]
    corpus_src = convs[24000:40000]
    agent_src, agent_warm_src = convs[40000:42000], convs[42000:43000]
    assert len(agent_warm_src) == 1000, len(convs)

    files = {}

    def emit(name, rows):
        path = os.path.join(a.out, f"{name}.jsonl.gz")
        files[name] = {"sha256": write_jsonl_gz(path, rows), "n": len(rows),
                       "input_tokens_mean": round(sum(r["input_tokens"] for r in rows) / len(rows), 1),
                       "max_tokens_mean": round(sum(r["max_tokens"] for r in rows) / len(rows), 1)}
        print(name, files[name], flush=True)

    emit("chat", build_chat(chat_src, tok, 4096, random.Random(rng.random()), "chat-"))
    emit("chat_warmup", build_chat(warm_src, tok, 512, random.Random(rng.random()), "chatw-"))

    corpus = Corpus(corpus_src, tok, 6_000_000)
    for L, name in [(8000, "prefill8k"), (16000, "prefill16k"), (32000, "prefill32k")]:
        r1 = random.Random(rng.random())
        emit(name, build_prefill(corpus, tok, 96, L, 128, r1, f"{name}-"))
        emit(name + "_warmup", build_prefill(corpus, tok, 16, L, 128, r1, f"{name}w-"))

    system = agent_system(tok, corpus, random.Random(rng.random()))
    emit("agent", build_agent(agent_src, corpus, tok, 512, random.Random(rng.random()), system, "ag-"))
    emit("agent_warmup", build_agent(agent_warm_src, corpus, tok, 64, random.Random(rng.random()), system, "agw-"))

    emit("quality_gsm8k", build_quality(a.gsm8k, tok, 250, random.Random(rng.random()), "gsm-"))

    import tokenizers
    manifest = {
        "generator": "bib/workloads.py", "seed": a.seed,
        "sources": {"sharegpt": {"file": os.path.basename(a.sharegpt), "sha256": sha256_file(a.sharegpt),
                                 "origin": "huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered@192ab218"},
                    "gsm8k": {"file": os.path.basename(a.gsm8k), "sha256": sha256_file(a.gsm8k),
                              "origin": "modelscope.cn/datasets/AI-ModelScope/gsm8k main/test"}},
        "tokenizer_json_sha256": tok.sha256, "tokenizers_version": tokenizers.__version__,
        "agent_system_prompt_tokens": tok.n(system),
        "files": files,
    }
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        for name, v in sorted(files.items()):
            f.write(f"{v['sha256']}  {name}.jsonl.gz\n")


if __name__ == "__main__":
    main()
