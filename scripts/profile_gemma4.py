#!/usr/bin/env python3
"""Profiling script for Gemma 4 on MI355X.

Captures torch.profiler trace and extracts top-N hotspot kernels
with AITER tier classification.

Usage (inside container):
  PYTHONPATH=/sgl-workspace/aiter python3 /workspace/scripts/profile_gemma4.py \
      --model /models/gemma-4-31b-it-MXFP4 --tp 1 --warmup 3 --profile-steps 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

TIER_PATTERNS = {
    "ASM": [".co", "hsaco", "fmoe_mxfp4", "asm_mla", "module_moe_asm"],
    "CK": ["ck_gemm", "ck2stages", "ck_tile", "composable_kernel", "ck_batched"],
    "Gluon": ["gluon", "gemm_afp4wfp4", "pa_decode_gluon", "pa_mqa_logits"],
    "Triton": ["triton", "_fused_moe_kernel", "fused_mxfp4_quant", "unified_attention"],
    "CUDA_JIT": ["module_activation", "module_rmsnorm", "module_rope", "module_moe_asm",
                  "gelu_tanh_and_mul", "silu_and_mul", "rmsnorm2d_fwd"],
    "PyTorch": ["aten::", "torch.", "autograd"],
}


def classify_tier(kernel_name: str) -> str:
    for tier, patterns in TIER_PATTERNS.items():
        for p in patterns:
            if p.lower() in kernel_name.lower():
                return tier
    return "Unknown"


def analyze_trace(trace_path: str, top_n: int = 30):
    """Parse Chrome trace JSON and extract hotspot kernels."""
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    kernel_stats = {}
    for e in events:
        if e.get("cat") in ("kernel", "gpu_memcpy", "cuda_runtime", "hip_runtime"):
            name = e.get("name", "unknown")
            dur = e.get("dur", 0)
            if name not in kernel_stats:
                kernel_stats[name] = {"total_us": 0, "count": 0, "max_us": 0}
            kernel_stats[name]["total_us"] += dur
            kernel_stats[name]["count"] += 1
            kernel_stats[name]["max_us"] = max(kernel_stats[name]["max_us"], dur)

    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)

    print(f"\n{'='*100}")
    print(f"Top-{top_n} Hotspot Kernels (from {trace_path})")
    print(f"{'='*100}")
    print(f"{'Rank':>4}  {'Total(ms)':>10}  {'Count':>6}  {'Avg(us)':>8}  {'Tier':>10}  Kernel Name")
    print(f"{'-'*100}")

    results = []
    for rank, (name, stats) in enumerate(sorted_kernels[:top_n], 1):
        tier = classify_tier(name)
        avg_us = stats["total_us"] / max(stats["count"], 1)
        total_ms = stats["total_us"] / 1000
        print(f"{rank:4d}  {total_ms:10.2f}  {stats['count']:6d}  {avg_us:8.1f}  {tier:>10}  {name[:80]}")
        results.append({
            "rank": rank, "kernel": name, "tier": tier,
            "total_ms": round(total_ms, 2), "count": stats["count"],
            "avg_us": round(avg_us, 1),
        })

    # Tier distribution
    tier_totals = {}
    for name, stats in sorted_kernels:
        tier = classify_tier(name)
        tier_totals[tier] = tier_totals.get(tier, 0) + stats["total_us"]
    total_us = sum(tier_totals.values())

    print(f"\n{'='*60}")
    print("AITER Tier Distribution:")
    for tier, us in sorted(tier_totals.items(), key=lambda x: -x[1]):
        pct = 100 * us / max(total_us, 1)
        print(f"  {tier:>10}: {us/1000:10.2f} ms ({pct:5.1f}%)")

    report_path = trace_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({"kernels": results, "tier_distribution": tier_totals}, f, indent=2)
    print(f"\nReport saved: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="/workspace/profiles",
                        help="Directory containing profiler trace JSON files")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Trace directory {trace_dir} does not exist.")
        print("Run the model with ATOM_TORCH_PROFILER_DIR set first.")
        return

    traces = sorted(trace_dir.glob("*.json"))
    if not traces:
        print(f"No JSON traces found in {trace_dir}")
        return

    for t in traces:
        analyze_trace(str(t), args.top_n)


if __name__ == "__main__":
    main()
