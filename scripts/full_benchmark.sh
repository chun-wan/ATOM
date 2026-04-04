#!/bin/bash
set -euo pipefail

# Full Gemma 4 MXFP4 Benchmark Suite for MI355X
# Runs: 9 configs x throughput + gsm8k accuracy
# Usage: Run inside container with PYTHONPATH=/sgl-workspace/aiter

export PYTHONPATH=/sgl-workspace/aiter
RESULT_DIR="/workspace/benchmark_results/mxfp4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
PORT=30000

echo "================================================================"
echo "  Gemma 4 MXFP4 Full Benchmark Suite"
echo "  MI355X @ $(hostname) | $(date)"
echo "  Results: $RESULT_DIR"
echo "================================================================"

wait_ready() {
    local PORT=$1 TIMEOUT=${2:-300}
    for i in $(seq 1 $TIMEOUT); do
        curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo "TIMEOUT: server not ready after ${TIMEOUT}s" && return 1
}

run_bench() {
    local TAG=$1 PORT=$2
    echo "  Throughput benchmark: $TAG"
    for CONC in 4 16 64; do
        python3 -m sglang.bench_serving \
            --backend sglang --host localhost --port "$PORT" \
            --dataset-name random \
            --random-input-len 1024 --random-output-len 1024 \
            --random-range-ratio 0.8 \
            --num-prompts $((CONC * 10)) \
            --request-rate inf --max-concurrency "$CONC" \
            --output-file "$RESULT_DIR/${TAG}_conc${CONC}.json" \
            2>&1 | tee -a "$RESULT_DIR/${TAG}.log" || true
    done
}

run_gsm8k() {
    local TAG=$1 PORT=$2 MODEL=$3
    echo "  gsm8k accuracy: $TAG"
    lm_eval --model local-completions \
        --model_args "model=$MODEL,base_url=http://localhost:${PORT}/v1/completions,tokenized_requests=False,num_concurrent=32" \
        --tasks gsm8k --num_fewshot 5 --batch_size 1 \
        2>&1 | tee "$RESULT_DIR/${TAG}_gsm8k.log" || true
}

kill_server() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "atom.entrypoints" 2>/dev/null || true
    sleep 10
}

# Optimization env vars
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1
export VLLM_ROCM_USE_AITER=1

# ================================================================
# A1: Gemma4-31B-it BF16, SGLang vanilla, TP=1
# ================================================================
echo "[A1] SGLang vanilla BF16 - Gemma 4 31B TP=1"
kill_server
python3 -m sglang.launch_server \
    --model-path google/gemma-4-31b-it \
    --host localhost --port $PORT \
    --attention-backend triton --tp 1 --mem-fraction-static 0.85 \
    2>&1 | tee "$RESULT_DIR/A1_server.log" &
if wait_ready $PORT; then
    run_bench "A1_sglang_bf16_31b" $PORT
    run_gsm8k "A1" $PORT "google/gemma-4-31b-it"
fi
kill_server

# ================================================================
# A3: Gemma4-31B-it MXFP4, SGLang+ATOM, TP=1
# ================================================================
echo "[A3] SGLang+ATOM MXFP4 - Gemma 4 31B TP=1"
python3 -m sglang.launch_server \
    --model-path /models/gemma-4-31b-it-MXFP4 \
    --host localhost --port $PORT \
    --attention-backend triton --tp 1 --mem-fraction-static 0.85 \
    --model-impl atom \
    2>&1 | tee "$RESULT_DIR/A3_server.log" &
if wait_ready $PORT; then
    run_bench "A3_atom_mxfp4_31b" $PORT
    run_gsm8k "A3" $PORT "/models/gemma-4-31b-it-MXFP4"
fi
kill_server

# ================================================================
# A5: Gemma4-31B-it MXFP4, ATOM, TP=1, spec_decode steps=3
# ================================================================
echo "[A5] ATOM MXFP4 + Spec Decode steps=3 - Gemma 4 31B"
python3 -m sglang.launch_server \
    --model-path /models/gemma-4-31b-it-MXFP4 \
    --host localhost --port $PORT \
    --attention-backend triton --tp 1 --mem-fraction-static 0.80 \
    --model-impl atom \
    --speculative-model google/gemma-4-e4b-it \
    --num-speculative-tokens 3 \
    2>&1 | tee "$RESULT_DIR/A5_server.log" &
if wait_ready $PORT 600; then
    run_bench "A5_atom_mxfp4_spec3_31b" $PORT
fi
kill_server

# ================================================================
# A6: Gemma4-31B-it MXFP4, ATOM, TP=2, Custom AllReduce INT4
# ================================================================
echo "[A6] ATOM MXFP4 TP=2 + Custom AllReduce INT4"
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_USE_CUSTOM_ALL_GATHER=1
python3 -m sglang.launch_server \
    --model-path /models/gemma-4-31b-it-MXFP4 \
    --host localhost --port $PORT \
    --attention-backend triton --tp 2 --mem-fraction-static 0.85 \
    --model-impl atom \
    2>&1 | tee "$RESULT_DIR/A6_server.log" &
if wait_ready $PORT; then
    run_bench "A6_atom_mxfp4_tp2_31b" $PORT
fi
kill_server

# ================================================================
# B1: Gemma4-26B-A4B-it BF16, SGLang vanilla, TP=1
# ================================================================
echo "[B1] SGLang vanilla BF16 - Gemma 4 26B MoE TP=1"
python3 -m sglang.launch_server \
    --model-path google/gemma-4-26b-a4b-it \
    --host localhost --port $PORT \
    --attention-backend triton --tp 1 --mem-fraction-static 0.85 \
    2>&1 | tee "$RESULT_DIR/B1_server.log" &
if wait_ready $PORT; then
    run_bench "B1_sglang_bf16_26b_moe" $PORT
    run_gsm8k "B1" $PORT "google/gemma-4-26b-a4b-it"
fi
kill_server

# ================================================================
# B2: Gemma4-26B-A4B-it MXFP4, ATOM, TP=1
# ================================================================
echo "[B2] ATOM MXFP4 - Gemma 4 26B MoE TP=1"
python3 -m sglang.launch_server \
    --model-path /models/gemma-4-26b-a4b-it-MXFP4 \
    --host localhost --port $PORT \
    --attention-backend triton --tp 1 --mem-fraction-static 0.85 \
    --model-impl atom \
    2>&1 | tee "$RESULT_DIR/B2_server.log" &
if wait_ready $PORT; then
    run_bench "B2_atom_mxfp4_26b_moe" $PORT
    run_gsm8k "B2" $PORT "/models/gemma-4-26b-a4b-it-MXFP4"
fi
kill_server

echo ""
echo "================================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "  Results: $RESULT_DIR"
echo "================================================================"
ls -la "$RESULT_DIR"/
