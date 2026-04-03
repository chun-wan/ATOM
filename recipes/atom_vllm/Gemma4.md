# Gemma 4 with ATOM vLLM Plugin Backend (Optimized)

This recipe shows how to run Gemma 4 models with full operator optimizations on MI355X.

**Optimization flags enabled:**
- `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1` — fused QK-norm + RoPE + cache quant
- `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1` — fused AllReduce + RMSNorm
- `AITER_QUICK_REDUCE_QUANTIZATION=INT4` — quantized AllReduce for TP
- `cudagraph_mode: FULL_AND_PIECEWISE` — piecewise CUDA graph for hybrid attention

## Step 1: Pull Docker Image

```bash
podman pull rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260403
```

## Step 2: Launch Container

```bash
podman run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  -v ~/workspace:/workspace \
  -v /home/models:/models \
  -p 30000:30000 -p 8000:8000 \
  --name atom-gemma4 \
  rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260403 bash
```

## Step 3: Launch with ATOM vLLM Plugin (Optimized)

### Gemma 4 31B Dense (TP=1) — Optimized

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1

vllm serve google/gemma-4-31b-it \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### Gemma 4 26B-A4B MoE (TP=1) — Optimized

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1

vllm serve google/gemma-4-26b-a4b-it \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### SGLang + ATOM Plugin Backend

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python3 -m sglang.launch_server \
    --model-path google/gemma-4-31b-it \
    --host localhost \
    --port 30000 \
    --attention-backend triton \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --model-impl atom \
    2>&1 | tee log.serve.log &
```

## Step 4: Benchmark Suite

### A. Throughput Benchmark

```bash
for CONC in 4 16 64; do
  python -m atom.benchmarks.benchmark_serving \
      --model=google/gemma-4-31b-it \
      --backend=vllm \
      --base-url=http://localhost:8000 \
      --dataset-name=random \
      --random-input-len=1024 --random-output-len=1024 \
      --random-range-ratio 0.8 \
      --num-prompts=$((CONC * 10)) \
      --max-concurrency=$CONC \
      --request-rate=inf --ignore-eos \
      --save-result --result-dir=./results \
      --percentile-metrics="ttft,tpot,itl,e2el"
done
```

### B. Accuracy Validation (gsm8k)

```bash
lm_eval --model local-completions \
    --model_args model=google/gemma-4-31b-it,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
    --tasks gsm8k \
    --num_fewshot 3
```

## Performance Baseline (MI355X)

| Model | TP | ISL | OSL | Conc | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 |
|-------|-----|-----|-----|------|-------------|-------------|----------|----------|
| Gemma4-31B | 1 | 1024 | 1024 | 4 | TBD | TBD | TBD | TBD |
| Gemma4-31B | 1 | 1024 | 1024 | 16 | TBD | TBD | TBD | TBD |
| Gemma4-31B | 1 | 1024 | 1024 | 64 | TBD | TBD | TBD | TBD |
| Gemma4-26B-MoE | 1 | 1024 | 1024 | 4 | TBD | TBD | TBD | TBD |
| Gemma4-26B-MoE | 1 | 1024 | 1024 | 64 | TBD | TBD | TBD | TBD |

## Operator Optimization Status

| Operator | Level | Notes |
|----------|-------|-------|
| GeluAndMul | AITER CUDA JIT | `gelu_tanh_and_mul` kernel |
| Logit softcapping | Fused Triton | Single kernel, in-place |
| PA Decode | Gluon (CDNA4) | gfx950-optimized Triton |
| RMSNorm | AITER CUDA | fused_add_rms_norm_cu |
| RoPE | AITER CUDA | per-layer-type config |
| GEMM | CK | FP8/MXFP4 blockscale |
| FusedMoE | ASM + CK | 128 experts, top-k=8 |
| TopK softmax | ASM | module_moe_asm |
| CUDA Graph | Piecewise | sliding/global mixed |
| AllReduce | Quick (INT4) | AITER custom |
