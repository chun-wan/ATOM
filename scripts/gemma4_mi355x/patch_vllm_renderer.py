#!/usr/bin/env python3
"""Patch vLLM renderer to skip tokenizer deepcopy (Gemma 4 workaround)."""
import glob, os

for f in glob.glob("/usr/local/lib/python3.*/dist-packages/vllm/renderers/base.py"):
    code = open(f).read()
    old = "mm_tokenizer = copy.deepcopy(tokenizer)"
    new = "mm_tokenizer = tokenizer  # PATCHED: skip deepcopy for Gemma 4"
    if old in code:
        code = code.replace(old, new)
        open(f, "w").write(code)
        print(f"[OK] Patched {f}")
    elif "PATCHED" in code:
        print(f"[SKIP] Already patched")
    else:
        print(f"[FAIL] Pattern not found in {f}")

# Clear pycache
os.system("find /usr/local/lib -name __pycache__ -path '*/vllm/*' -exec rm -rf {} + 2>/dev/null")
print("Done")
