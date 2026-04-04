#!/bin/bash
set -euo pipefail

# Master orchestration: Quantize -> Profile -> Benchmark -> Report
# Run inside container: bash /workspace/ATOM/scripts/run_all.sh
# Prerequisite: Model downloads complete in /workspace/models/

export PYTHONPATH=/sgl-workspace/aiter
MODELS_DIR="/workspace/models"
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  GEMMA 4 MXFP4 FULL PIPELINE"
echo "  $(date)"
echo "================================================================"

# Step 0: Check model availability
echo ""
echo "=== Step 0: Checking model availability ==="
for m in gemma-4-31b-it gemma-4-26b-a4b-it; do
    if [ -f "$MODELS_DIR/$m/config.json" ]; then
        echo "  OK: $m ($(du -sh $MODELS_DIR/$m | cut -f1))"
    else
        echo "  MISSING: $m -- waiting for download to complete"
        echo "  Check /workspace/dl_progress.log for status"
        exit 1
    fi
done

# Step 1: Install Quark and run quantization
echo ""
echo "=== Step 1: MXFP4 Quantization ==="
pip show amd-quark > /dev/null 2>&1 || pip install amd-quark==0.11.1

if [ ! -d /workspace/Quark ]; then
    git clone --depth 1 --branch v0.11.1 https://github.com/amd/Quark.git /workspace/Quark
fi

cd /workspace/Quark/examples/torch/language_modeling/llm_ptq/

DENSE_EXCLUDE="*self_attn* *mlp.gate *lm_head *mm_projector* *vision_tower*"
MOE_EXCLUDE="*self_attn* *router* *mlp.gate *lm_head *mm_projector* *vision_tower*"

if [ ! -d "$MODELS_DIR/gemma-4-31b-it-MXFP4" ]; then
    echo "  Quantizing 31B dense..."
    python quantize_quark.py \
        --model_dir "$MODELS_DIR/gemma-4-31b-it" \
        --quant_scheme mxfp4 \
        --exclude_layers $DENSE_EXCLUDE \
        --output_dir "$MODELS_DIR/gemma-4-31b-it-MXFP4" \
        --file2file_quantization \
        2>&1 | tee "$LOG_DIR/quant_31b.log"
fi

if [ ! -d "$MODELS_DIR/gemma-4-26b-a4b-it-MXFP4" ]; then
    echo "  Quantizing 26B MoE..."
    python quantize_quark.py \
        --model_dir "$MODELS_DIR/gemma-4-26b-a4b-it" \
        --quant_scheme mxfp4 \
        --exclude_layers $MOE_EXCLUDE \
        --output_dir "$MODELS_DIR/gemma-4-26b-a4b-it-MXFP4" \
        --file2file_quantization \
        2>&1 | tee "$LOG_DIR/quant_26b.log"
fi

# Step 2: MXFP4 tier micro-benchmark
echo ""
echo "=== Step 2: MXFP4 Tier Micro-Benchmark ==="
python3 /workspace/ATOM/scripts/bench_mxfp4_tiers.py 2>&1 | tee "$LOG_DIR/tier_bench.log"

# Step 3: Full benchmark suite
echo ""
echo "=== Step 3: Full Benchmark Suite ==="
bash /workspace/ATOM/scripts/full_benchmark.sh 2>&1 | tee "$LOG_DIR/full_bench.log"

# Step 4: Generate report
echo ""
echo "=== Step 4: Generate Report ==="
RESULT_DIR=$(ls -td /workspace/benchmark_results/mxfp4_* 2>/dev/null | head -1)
if [ -n "$RESULT_DIR" ]; then
    python3 /workspace/ATOM/scripts/generate_report.py --results-dir "$RESULT_DIR" 2>&1 | tee "$LOG_DIR/report.log"
fi

echo ""
echo "================================================================"
echo "  PIPELINE COMPLETE"
echo "  Logs: $LOG_DIR"
echo "  Results: $RESULT_DIR"
echo "================================================================"
