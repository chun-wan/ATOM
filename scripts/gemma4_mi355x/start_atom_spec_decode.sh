#!/bin/bash
# Gemma 4 Speculative Decoding with MTP
# Requires: Gemma 4 MTP draft model (not yet released by Google)
# Usage: Start this after the MTP model is available
set -e

# Apply all patches
for f in /tmp/fix_*.py; do python3 "$f" 2>/dev/null || true; done
find /app -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

DRAFT_MODEL="/workspace/models/gemma-4-mtp"  # Update path when available

if [ ! -d "$DRAFT_MODEL" ]; then
    echo "ERROR: MTP draft model not found at $DRAFT_MODEL"
    echo "Gemma 4 MTP model has not been released yet."
    echo "Falling back to non-speculative mode..."
    exec python3 -m atom.entrypoints.openai_server \
        --model /workspace/models/gemma-4-31b-it \
        --host 0.0.0.0 --server-port 8001 \
        --trust-remote-code --level 2 \
        --kv_cache_dtype bf16 --tensor-parallel-size 1
fi

exec python3 -m atom.entrypoints.openai_server \
    --model /workspace/models/gemma-4-31b-it \
    --host 0.0.0.0 --server-port 8001 \
    --trust-remote-code --level 2 \
    --kv_cache_dtype bf16 --tensor-parallel-size 1 \
    --method mtp --num-speculative-tokens 3
