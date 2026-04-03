#!/bin/bash
# 3-Config Benchmark Comparison: SGLang vanilla vs SGLang+ATOM vs ATOM-only
# Usage: Run inside container on MI355X
# Env: PYTHONPATH=/sgl-workspace/aiter
set -euo pipefail

export PYTHONPATH=/sgl-workspace/aiter
RESULT_DIR="/workspace/benchmark_results/compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

MODEL_GEMMA4="google/gemma-4-31b-it"
MODEL_QWEN35="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=30000

echo "================================================================"
echo "  3-Config Benchmark: SGLang vs SGLang+ATOM vs ATOM-only"
echo "  MI355X @ $(hostname) | $(date)"
echo "  Results: $RESULT_DIR"
echo "================================================================"
echo ""

wait_ready() {
    local PORT=$1 TIMEOUT=${2:-300}
    for i in $(seq 1 $TIMEOUT); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo "  TIMEOUT: server not ready after ${TIMEOUT}s"
    return 1
}

run_bench() {
    local TAG=$1 PORT=$2 MODEL=$3 BACKEND=${4:-sglang}
    echo "  >> Benchmarking $TAG"
    for CONC in 4 16 64; do
        python3 -m sglang.bench_serving \
            --backend "$BACKEND" \
            --host localhost --port "$PORT" \
            --dataset-name random \
            --random-input-len 1024 --random-output-len 1024 \
            --random-range-ratio 0.8 \
            --num-prompts $((CONC * 10)) \
            --request-rate inf \
            --max-concurrency "$CONC" \
            --output-file "$RESULT_DIR/${TAG}_conc${CONC}.json" \
            2>&1 | tee -a "$RESULT_DIR/${TAG}.log" || true
    done
}

run_gsm8k() {
    local TAG=$1 PORT=$2 MODEL=$3
    echo "  >> GSM8K accuracy: $TAG"
    lm_eval --model local-completions \
        --model_args "model=$MODEL,base_url=http://localhost:${PORT}/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False" \
        --tasks gsm8k --num_fewshot 3 \
        2>&1 | tee "$RESULT_DIR/${TAG}_gsm8k.log" || true
}

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "atom.entrypoints" 2>/dev/null || true
    sleep 10
}

# ================================================================
# Config 1: SGLang vanilla (no ATOM)
# ================================================================
echo "[1/3] SGLang vanilla — Gemma 4 31B (TP=1)"
kill_server
python3 -m sglang.launch_server \
    --model-path "$MODEL_GEMMA4" \
    --host localhost --port $PORT \
    --attention-backend triton \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    2>&1 | tee "$RESULT_DIR/sglang_vanilla_gemma4_server.log" &
if wait_ready $PORT 300; then
    run_bench "sglang_vanilla_gemma4_31b" $PORT "$MODEL_GEMMA4"
    run_gsm8k "sglang_vanilla_gemma4_31b" $PORT "$MODEL_GEMMA4"
fi
kill_server

# ================================================================
# Config 2: SGLang + ATOM plugin
# ================================================================
echo "[2/3] SGLang + ATOM plugin — Gemma 4 31B (TP=1)"
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1
python3 -m sglang.launch_server \
    --model-path "$MODEL_GEMMA4" \
    --host localhost --port $PORT \
    --attention-backend triton \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --model-impl atom \
    2>&1 | tee "$RESULT_DIR/sglang_atom_gemma4_server.log" &
if wait_ready $PORT 300; then
    run_bench "sglang_atom_gemma4_31b" $PORT "$MODEL_GEMMA4"
    run_gsm8k "sglang_atom_gemma4_31b" $PORT "$MODEL_GEMMA4"
fi
kill_server

# ================================================================
# Config 3: ATOM standalone
# ================================================================
echo "[3/3] ATOM standalone — Gemma 4 31B (TP=1)"
python3 -m atom.entrypoints.openai_server \
    --model "$MODEL_GEMMA4" \
    --host localhost --port $PORT \
    --tp 1 \
    --kv-cache-dtype fp8 \
    2>&1 | tee "$RESULT_DIR/atom_standalone_gemma4_server.log" &
if wait_ready $PORT 300; then
    run_bench "atom_standalone_gemma4_31b" $PORT "$MODEL_GEMMA4"
    run_gsm8k "atom_standalone_gemma4_31b" $PORT "$MODEL_GEMMA4"
fi
kill_server

# ================================================================
# Summary
# ================================================================
echo ""
echo "================================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "  Results: $RESULT_DIR"
echo "================================================================"
echo ""
echo "Summary JSON files:"
ls -la "$RESULT_DIR"/*.json 2>/dev/null || echo "  (no JSON files — models may not have been downloaded)"
echo ""
echo "GSM8K logs:"
ls -la "$RESULT_DIR"/*gsm8k* 2>/dev/null || echo "  (no gsm8k results)"
