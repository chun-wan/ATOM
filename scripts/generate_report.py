#!/usr/bin/env python3
"""Generate final comparison report from benchmark results.

Usage: python3 /workspace/scripts/generate_report.py --results-dir /workspace/benchmark_results/mxfp4_*
"""

import argparse
import json
import glob
import os
from pathlib import Path


def parse_sglang_bench(json_path):
    """Parse SGLang bench_serving output JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        return {
            "output_tps": data.get("output_throughput", 0),
            "total_tps": data.get("total_throughput", 0),
            "ttft_p50": data.get("ttft_p50_ms", 0),
            "ttft_p99": data.get("ttft_p99_ms", 0),
            "tpot_p50": data.get("tpot_p50_ms", 0),
            "tpot_p99": data.get("tpot_p99_ms", 0),
        }
    except Exception:
        return None


def parse_gsm8k_log(log_path):
    """Parse lm_eval gsm8k log for accuracy numbers."""
    try:
        with open(log_path) as f:
            text = f.read()
        flex = strict = None
        for line in text.split("\n"):
            if "flexible-extract" in line and "exact_match" in line:
                parts = line.split("|")
                for p in parts:
                    try:
                        flex = float(p.strip())
                    except ValueError:
                        continue
            if "strict-match" in line and "exact_match" in line:
                parts = line.split("|")
                for p in parts:
                    try:
                        strict = float(p.strip())
                    except ValueError:
                        continue
        return {"gsm8k_flex": flex, "gsm8k_strict": strict}
    except Exception:
        return {"gsm8k_flex": None, "gsm8k_strict": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    configs = [
        ("A1", "Gemma4-31B-it", "BF16", "SGLang vanilla", "1", "None"),
        ("A3", "Gemma4-31B-it", "MXFP4", "SGLang+ATOM", "1", "None"),
        ("A5", "Gemma4-31B-it", "MXFP4", "ATOM+SpecDec3", "1", "steps=3"),
        ("A6", "Gemma4-31B-it", "MXFP4", "ATOM TP=2", "2", "None"),
        ("B1", "Gemma4-26B-MoE", "BF16", "SGLang vanilla", "1", "None"),
        ("B2", "Gemma4-26B-MoE", "MXFP4", "ATOM", "1", "None"),
    ]

    print(f"\n{'='*120}")
    print(f"Gemma 4 MXFP4 Optimization - Full Comparison Report")
    print(f"{'='*120}")

    # Throughput table (concurrency=64)
    print(f"\n--- Throughput (ISL=1024, OSL=1024, concurrency=64) ---")
    print(f"{'ID':>3}  {'Model':>16}  {'Quant':>5}  {'Backend':>16}  {'TP':>2}  {'SpecDec':>8}  "
          f"{'OutTPS':>8}  {'TotalTPS':>9}  {'TTFT_p50':>9}  {'TPOT_p50':>9}")
    print("-" * 120)

    for config_id, model, quant, backend, tp, spec in configs:
        bench_file = list(results_dir.glob(f"{config_id}_*_conc64.json"))
        if bench_file:
            data = parse_sglang_bench(str(bench_file[0]))
            if data:
                print(f"{config_id:>3}  {model:>16}  {quant:>5}  {backend:>16}  {tp:>2}  {spec:>8}  "
                      f"{data['output_tps']:>8.1f}  {data['total_tps']:>9.1f}  "
                      f"{data['ttft_p50']:>9.1f}  {data['tpot_p50']:>9.1f}")
                continue
        print(f"{config_id:>3}  {model:>16}  {quant:>5}  {backend:>16}  {tp:>2}  {spec:>8}  "
              f"{'N/A':>8}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}")

    # Accuracy table
    print(f"\n--- gsm8k Accuracy (5-shot) ---")
    print(f"{'ID':>3}  {'Model':>16}  {'Quant':>5}  {'gsm8k_flex':>11}  {'gsm8k_strict':>12}  {'Recovery':>9}")
    print("-" * 70)

    bf16_flex = {}
    for config_id, model, quant, *_ in configs:
        gsm8k_file = list(results_dir.glob(f"{config_id}_gsm8k.log"))
        if gsm8k_file:
            acc = parse_gsm8k_log(str(gsm8k_file[0]))
            flex_str = f"{acc['gsm8k_flex']:.4f}" if acc['gsm8k_flex'] else "N/A"
            strict_str = f"{acc['gsm8k_strict']:.4f}" if acc['gsm8k_strict'] else "N/A"
            if quant == "BF16" and acc['gsm8k_flex']:
                bf16_flex[model] = acc['gsm8k_flex']
            recovery = ""
            if quant == "MXFP4" and acc['gsm8k_flex'] and model in bf16_flex:
                recovery = f"{100*acc['gsm8k_flex']/bf16_flex[model]:.1f}%"
            print(f"{config_id:>3}  {model:>16}  {quant:>5}  {flex_str:>11}  {strict_str:>12}  {recovery:>9}")
        else:
            print(f"{config_id:>3}  {model:>16}  {quant:>5}  {'N/A':>11}  {'N/A':>12}  {'':>9}")

    # MXFP4 tier report
    tier_file = Path("/workspace/mxfp4_tier_benchmark.json")
    if tier_file.exists():
        print(f"\n--- MXFP4 Operator Tier Benchmark ---")
        with open(tier_file) as f:
            tier_data = json.load(f)
        if "dense_gemm" in tier_data:
            print("\nDense GEMM (Gluon gemm_afp4wfp4):")
            for k, v in tier_data["dense_gemm"].items():
                print(f"  {k:>30}: {v['ms']:.3f} ms  ({v['tflops']:.1f} TFLOPS)")
        if "moe" in tier_data:
            print("\nMoE MXFP4 (Triton fused):")
            for k, v in tier_data["moe"].items():
                print(f"  {k:>35}: {v:.3f} ms")

    # Profiling report
    profile_reports = list(Path("/workspace/profiles").glob("*_report.json")) if Path("/workspace/profiles").exists() else []
    if profile_reports:
        print(f"\n--- Top Kernel Hotspots (from profiler) ---")
        with open(profile_reports[0]) as f:
            prof_data = json.load(f)
        for k in prof_data.get("kernels", [])[:15]:
            print(f"  #{k['rank']:2d}  {k['total_ms']:8.2f}ms  {k['tier']:>10}  {k['kernel'][:60]}")

    report_path = results_dir / "FINAL_REPORT.txt"
    print(f"\nReport path: {report_path}")


if __name__ == "__main__":
    main()
