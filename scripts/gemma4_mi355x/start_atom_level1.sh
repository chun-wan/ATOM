#!/bin/bash
set -e

echo "=== Gemma4 ATOM Level 1 (torch.compile) Startup ==="

# Apply all patches
echo "[1] Syncing latest gemma4.py from host workspace..."
if [ -f /workspace/ATOM/atom/models/gemma4.py ]; then
    cp /workspace/ATOM/atom/models/gemma4.py /app/ATOM/atom/models/gemma4.py
fi
if [ -f /workspace/ATOM/atom/model_config/gemma4.py ]; then
    cp /workspace/ATOM/atom/model_config/gemma4.py /app/ATOM/atom/model_config/gemma4.py
fi

echo "[2] Applying crash patches..."
python3 /workspace/patch_all_crashes.py 2>/dev/null || true
python3 /workspace/patch_kv_cache_v2.py 2>/dev/null || true

echo "[3] Applying optimization patches..."
# RoPE 5D shape fix
python3 /tmp/fix_rope_shape.py 2>/dev/null || true
# AiterBackend.is_ssm
python3 /tmp/fix_is_ssm.py 2>/dev/null || true
# GemmaRMSNorm AITER kernel
python3 /tmp/fix_gemma_rmsnorm.py 2>/dev/null || true
python3 /tmp/fix_weight_cache_invalidate.py 2>/dev/null || true
# Fused QK-norm-RoPE
python3 /tmp/fix_fused_qk_rope.py 2>/dev/null || true

echo "[4] Clearing __pycache__..."
find /app -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

echo "[5] Starting ATOM server with --level 1..."
exec python3 -m atom.entrypoints.openai_server \
    --model /workspace/models/gemma-4-31b-it \
    --host 0.0.0.0 --server-port 8001 \
    --trust-remote-code \
    --level 1 \
    --kv_cache_dtype bf16 \
    --tensor-parallel-size 1
