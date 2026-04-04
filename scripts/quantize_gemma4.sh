#!/bin/bash
set -euo pipefail

# Gemma 4 MXFP4 Quantization Script (Quark 0.11.1)
# Reference: amd/Kimi-K2.5-MXFP4 quantization pattern
# Run inside container: podman exec atom-mxfp4 bash /workspace/scripts/quantize_gemma4.sh

echo "=== Gemma 4 MXFP4 Quantization ==="
echo "Quark version: $(pip show amd-quark 2>/dev/null | grep Version | awk '{print $2}')"

# Clone Quark if not present (for quantization scripts)
if [ ! -d /workspace/Quark ]; then
    echo "Cloning Quark repository..."
    cd /workspace && git clone --depth 1 --branch v0.11.1 https://github.com/amd/Quark.git
fi

cd /workspace/Quark/examples/torch/language_modeling/llm_ptq/

# Dense model exclude pattern: keep attention, gate, lm_head, vision layers in high precision
DENSE_EXCLUDE="*self_attn* *mlp.gate *lm_head *mm_projector* *vision_tower*"

# MoE model exclude pattern: also keep router weights in high precision
MOE_EXCLUDE="*self_attn* *router* *mlp.gate *lm_head *mm_projector* *vision_tower*"

# --- Quantize Gemma 4 31B Dense (instruct) ---
if [ -d /models/gemma-4-31b-it ] && [ ! -d /models/gemma-4-31b-it-MXFP4 ]; then
    echo ""
    echo ">>> Quantizing Gemma 4 31B Dense (instruct) to MXFP4..."
    python quantize_quark.py \
        --model_dir /models/gemma-4-31b-it \
        --quant_scheme mxfp4 \
        --exclude_layers $DENSE_EXCLUDE \
        --output_dir /models/gemma-4-31b-it-MXFP4 \
        --file2file_quantization \
        2>&1 | tee /workspace/quant_31b_it.log
    echo "  Done: /models/gemma-4-31b-it-MXFP4"
else
    echo "  Skipping 31B-it: source missing or output exists"
fi

# --- Quantize Gemma 4 26B-A4B MoE (instruct) ---
if [ -d /models/gemma-4-26b-a4b-it ] && [ ! -d /models/gemma-4-26b-a4b-it-MXFP4 ]; then
    echo ""
    echo ">>> Quantizing Gemma 4 26B-A4B MoE (instruct) to MXFP4..."
    python quantize_quark.py \
        --model_dir /models/gemma-4-26b-a4b-it \
        --quant_scheme mxfp4 \
        --exclude_layers $MOE_EXCLUDE \
        --output_dir /models/gemma-4-26b-a4b-it-MXFP4 \
        --file2file_quantization \
        2>&1 | tee /workspace/quant_26b_moe.log
    echo "  Done: /models/gemma-4-26b-a4b-it-MXFP4"
else
    echo "  Skipping 26B-A4B: source missing or output exists"
fi

echo ""
echo "=== Quantization Complete ==="
ls -lhd /models/gemma-4-*MXFP4 2>/dev/null || echo "No MXFP4 models produced yet (downloads may still be running)"
