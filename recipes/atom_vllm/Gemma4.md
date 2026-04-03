# Gemma 4 with ATOM vLLM Plugin Backend

This recipe shows how to run Gemma 4 models with the ATOM vLLM plugin backend on AMD Instinct MI355X GPUs.

Supported variants:
- `google/gemma-4-31b-it` (30.7B dense, TP=1)
- `google/gemma-4-26b-a4b-it` (25.2B MoE, 3.8B active, TP=1)
- `google/gemma-4-e4b-it` (8B with PLE, 4.5B effective, TP=1)

## Step 1: Pull the Docker Image

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

## Step 3: Launch vLLM Server

### Gemma 4 31B Dense (TP=1)

```bash
vllm serve google/gemma-4-31b-it \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### Gemma 4 26B-A4B MoE (TP=1)

```bash
vllm serve google/gemma-4-26b-a4b-it \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### Using SGLang with ATOM Backend

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

python3 -m sglang.launch_server \
    --model-path google/gemma-4-31b-it \
    --host localhost \
    --port 30000 \
    --attention-backend triton \
    --tensor-parallel-size 1 \
    --model-impl atom \
    2>&1 | tee log.serve.log &
```

## Step 4: Performance Benchmark

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=google/gemma-4-31b-it \
    --backend=vllm \
    --base-url=http://localhost:8000 \
    --dataset-name=random \
    --random-input-len=1024 \
    --random-output-len=1024 \
    --random-range-ratio 0.8 \
    --num-prompts=640 \
    --max-concurrency=64 \
    --request-rate=inf --ignore-eos \
    --percentile-metrics="ttft,tpot,itl,e2el"
```

## Step 5: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=google/gemma-4-31b-it,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

## Performance Baseline

Performance numbers on MI355X GPU (to be updated after benchmarking):

| Model | ISL | OSL | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) |
|-------|-----|-----|-------------|-------------|--------------------------|--------------------------|
| Gemma 4 31B (TP=1) | 1024 | 1024 | 4 | 40 | TBD | TBD |
| Gemma 4 31B (TP=1) | 1024 | 1024 | 16 | 160 | TBD | TBD |
| Gemma 4 31B (TP=1) | 1024 | 1024 | 64 | 640 | TBD | TBD |
| Gemma 4 26B-A4B (TP=1) | 1024 | 1024 | 4 | 40 | TBD | TBD |
| Gemma 4 26B-A4B (TP=1) | 1024 | 1024 | 64 | 640 | TBD | TBD |

## Key Notes

- All Gemma 4 models require the Triton attention backend for bidirectional image-token attention when used with multimodal inputs.
- The `attention_k_eq_v=True` config means K and V share weights (unified K=V), which reduces KV cache memory.
- Different RoPE configurations are used for sliding-window vs global attention layers.
- `final_logit_softcapping=30.0` is applied to output logits.
