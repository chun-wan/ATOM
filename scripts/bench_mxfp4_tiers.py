#!/usr/bin/env python3
"""MXFP4 operator micro-benchmark across AITER tiers for MI355X.

Benchmarks GEMM and MoE operators at Triton, Gluon, CK, and ASM levels
using Gemma 4 model shapes.

Usage (inside container):
  PYTHONPATH=/sgl-workspace/aiter python3 /workspace/scripts/bench_mxfp4_tiers.py
"""

import torch
import time
import json
import sys
from pathlib import Path

# Gemma 4 31B Dense shapes
DENSE_SHAPES = {
    "gate_up_proj": {"M_values": [1, 4, 16, 32, 128], "N": 21504, "K": 5376},
    "down_proj":    {"M_values": [1, 4, 16, 32, 128], "N": 5376, "K": 10752},
    "qkv_proj":     {"M_values": [1, 4, 16, 32, 128], "N": 9216, "K": 5376},
    "o_proj":       {"M_values": [1, 4, 16, 32, 128], "N": 5376, "K": 8192},
}

# Gemma 4 26B-A4B MoE shapes
MOE_SHAPES = {
    "expert_gate_up": {"num_experts": 128, "top_k": 8, "N": 1408, "K": 2816},
    "expert_down":    {"num_experts": 128, "top_k": 8, "N": 2816, "K": 704},
}


def bench_fn(fn, warmup=10, iters=50):
    """Benchmark a function with GPU sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    return elapsed


def bench_gluon_gemm_afp4wfp4(M, N, K):
    """Benchmark Gluon-level MXFP4 dense GEMM."""
    try:
        from aiter.ops.triton.gluon.gemm_afp4wfp4 import gemm_afp4wfp4
        A = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda")
        B = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")
        A_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        B_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        A_mx = torch.randint(0, 255, (M, K // 32), dtype=torch.uint8, device="cuda")
        B_mx = torch.randint(0, 255, (N, K // 32), dtype=torch.uint8, device="cuda")
        C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        def fn():
            gemm_afp4wfp4(A, B, A_scale, B_scale, A_mx, B_mx, out=C)

        return bench_fn(fn)
    except Exception as e:
        return f"ERROR: {e}"


def bench_triton_moe_mxfp4(num_tokens, N, K, num_experts, top_k, activation="silu"):
    """Benchmark Triton-level MXFP4 MoE."""
    try:
        if activation == "gelu":
            sys.path.insert(0, "/workspace/ATOM/scripts")
            from moe_op_mxfp4_gelu_fused import fused_moe_mxfp4_gelu as moe_fn
        else:
            from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu as moe_fn
        import triton.language as tl

        A = torch.randint(0, 255, (num_tokens, K // 2), dtype=torch.uint8, device="cuda")
        B = torch.randint(0, 255, (num_experts, N, K // 2), dtype=torch.uint8, device="cuda")
        C = torch.empty(num_tokens * top_k, N // 2, dtype=torch.bfloat16, device="cuda")
        A_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        B_scale = torch.ones(num_experts, dtype=torch.float32, device="cuda")
        A_mx = torch.randint(0, 255, (num_tokens, K // 32), dtype=torch.uint8, device="cuda")
        B_mx = torch.randint(0, 255, (num_experts, N, K // 32), dtype=torch.uint8, device="cuda")
        topk_w = torch.ones(num_tokens, top_k, dtype=torch.float32, device="cuda") / top_k
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device="cuda")

        block_size = 32
        max_padded = num_tokens * top_k + num_experts * block_size
        sorted_ids = torch.arange(min(num_tokens * top_k, max_padded), dtype=torch.int32, device="cuda")
        expert_ids_t = torch.zeros(max_padded // block_size, dtype=torch.int32, device="cuda")
        ntpp = torch.tensor([num_tokens * top_k], dtype=torch.int32, device="cuda")

        config = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}

        def fn():
            moe_fn(A, B, C, A_scale, B_scale, A_mx, B_mx, topk_w, topk_ids,
                    sorted_ids, expert_ids_t, ntpp, True, top_k, False, False,
                    config, tl.bfloat16)

        return bench_fn(fn, warmup=5, iters=20)
    except Exception as e:
        return f"ERROR: {e}"


def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"{'='*90}")
    print(f"MXFP4 Operator Micro-Benchmark (Gemma 4 shapes on MI355X)")
    print(f"{'='*90}")

    results = {"gpu": props.name, "dense_gemm": {}, "moe": {}}

    # Dense GEMM benchmarks (Gluon tier)
    print(f"\n--- Dense GEMM (Gluon gemm_afp4wfp4) ---")
    print(f"{'Layer':>15}  {'M':>5}  {'N':>6}  {'K':>6}  {'Time(ms)':>10}  {'TFLOPS':>8}")
    for layer, params in DENSE_SHAPES.items():
        for M in params["M_values"]:
            N, K = params["N"], params["K"]
            t = bench_gluon_gemm_afp4wfp4(M, N, K)
            if isinstance(t, str):
                print(f"{layer:>15}  {M:>5}  {N:>6}  {K:>6}  {t}")
            else:
                flops = 2 * M * N * K / t / 1e9
                print(f"{layer:>15}  {M:>5}  {N:>6}  {K:>6}  {t:>10.3f}  {flops:>8.1f}")
                results["dense_gemm"][f"{layer}_M{M}"] = {"ms": round(t, 3), "tflops": round(flops, 1)}

    # MoE benchmarks
    print(f"\n--- MoE MXFP4 (Triton fused) ---")
    print(f"{'Layer':>20}  {'Tokens':>6}  {'Activation':>10}  {'Time(ms)':>10}")
    for layer, params in MOE_SHAPES.items():
        for num_tokens in [1, 4, 16, 32, 128]:
            for act in ["silu", "gelu"]:
                t = bench_triton_moe_mxfp4(
                    num_tokens, params["N"], params["K"],
                    params["num_experts"], params["top_k"], act
                )
                if isinstance(t, str):
                    print(f"{layer:>20}  {num_tokens:>6}  {act:>10}  {t}")
                else:
                    print(f"{layer:>20}  {num_tokens:>6}  {act:>10}  {t:>10.3f}")
                    results["moe"][f"{layer}_T{num_tokens}_{act}"] = round(t, 3)

    report_path = "/workspace/mxfp4_tier_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {report_path}")


if __name__ == "__main__":
    main()
