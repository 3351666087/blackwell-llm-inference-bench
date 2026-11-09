"""Make a "BF16 KV cache" view of a ModelOpt checkpoint without touching its files (AMENDMENTS A9).

  python tools/kv_overlay.py /root/autodl-tmp/models/Qwen3-32B-NVFP4 /root/autodl-tmp/models/Qwen3-32B-NVFP4-kvbf16

TensorRT-LLM 1.2.1 only accepts kv_cache_config.dtype in {fp8, nvfp4, auto}; with "auto" it follows the
checkpoint's `kv_cache_quant_algo` (FP8 for nv-community/Qwen3-32B-NVFP4). The overlay directory
symlinks every file of the SHA-verified checkpoint except the two quantization configs, which are
copied with the KV-cache quantization removed (`hf_quant_config.json: kv_cache_quant_algo = null`,
`config.json: quantization_config.kv_cache_scheme` deleted). Weights and activation quantization are
unchanged; the KV cache falls back to the model dtype (BF16), as for vLLM/SGLang `--kv-cache-dtype bfloat16`.
"""
import json
import os
import sys


def main(src, dst):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name in ("hf_quant_config.json", "config.json", "MODEL_MANIFEST.json"):
            continue
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(s, d)
    q = json.load(open(os.path.join(src, "hf_quant_config.json")))
    q["quantization"]["kv_cache_quant_algo"] = None
    json.dump(q, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=4)
    c = json.load(open(os.path.join(src, "config.json")))
    removed = c.get("quantization_config", {}).pop("kv_cache_scheme", None)
    json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=2)
    json.dump({"overlay_of": src, "changes": {"hf_quant_config.json": "quantization.kv_cache_quant_algo -> null",
                                              "config.json": f"quantization_config.kv_cache_scheme removed (was {removed})"},
               "source_manifest": os.path.join(src, "MODEL_MANIFEST.json")},
              open(os.path.join(dst, "OVERLAY.json"), "w"), indent=1)
    print(f"overlay {dst} -> {src}: KV-cache quantization removed")


if __name__ == "__main__":
    main(*sys.argv[1:3])
