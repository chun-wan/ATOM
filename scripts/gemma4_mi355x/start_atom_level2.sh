#!/bin/bash
set -e
echo "=== Gemma4 ATOM Level 2 (torch.compile + CUDAGraph) ==="

# Apply patches (same as level1)
python3 /workspace/patch_all_crashes.py 2>/dev/null || true
python3 /workspace/patch_kv_cache_v2.py 2>/dev/null || true
for f in /tmp/fix_rope_shape.py /tmp/fix_is_ssm.py /tmp/fix_gemma_rmsnorm.py /tmp/fix_weight_cache_invalidate.py /tmp/fix_fused_qk_rope.py; do
    python3 "$f" 2>/dev/null || true
done
find /app -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

echo "Starting ATOM server with --level 2 (CUDAGraph)..."
exec python3 -m atom.entrypoints.openai_server \
    --model /workspace/models/gemma-4-31b-it \
    --host 0.0.0.0 --server-port 8001 \
    --trust-remote-code \
    --level 2 \
    --kv_cache_dtype bf16 \
    --tensor-parallel-size 1
