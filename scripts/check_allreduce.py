#!/usr/bin/env python3
"""Verify AITER custom AllReduce (quick_all_reduce) availability and configuration.

Usage: PYTHONPATH=/sgl-workspace/aiter python3 /workspace/scripts/check_allreduce.py
"""

import os
import sys
import importlib

print("=" * 70)
print("AITER Custom AllReduce Verification")
print("=" * 70)

# Check env vars
for var in ["AITER_QUICK_REDUCE_QUANTIZATION", "ATOM_USE_CUSTOM_ALL_GATHER",
            "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION"]:
    val = os.environ.get(var, "<not set>")
    print(f"  {var} = {val}")

# Check AITER quick_all_reduce module
print("\nModule availability:")
try:
    from aiter.ops import quick_all_reduce
    print(f"  aiter.ops.quick_all_reduce: AVAILABLE")
    print(f"    Functions: {[x for x in dir(quick_all_reduce) if not x.startswith('_')]}")
except ImportError as e:
    print(f"  aiter.ops.quick_all_reduce: MISSING ({e})")

try:
    from aiter.ops import custom_all_reduce
    print(f"  aiter.ops.custom_all_reduce: AVAILABLE")
except ImportError as e:
    print(f"  aiter.ops.custom_all_reduce: MISSING ({e})")

try:
    from aiter.dist import communication_op
    has_fused = hasattr(communication_op, "tensor_model_parallel_fused_allreduce_rmsnorm")
    print(f"  fused_allreduce_rmsnorm: {'AVAILABLE' if has_fused else 'MISSING'}")
except ImportError as e:
    print(f"  aiter.dist.communication_op: MISSING ({e})")

# Check csrc compilation
print("\nCompiled kernels:")
try:
    import aiter
    aiter_path = os.path.dirname(aiter.__file__)
    qar_path = os.path.join(aiter_path, "ops", "quick_all_reduce.py")
    if os.path.exists(qar_path):
        print(f"  quick_all_reduce.py: EXISTS at {qar_path}")
    else:
        print(f"  quick_all_reduce.py: NOT FOUND (checked {qar_path})")
except Exception as e:
    print(f"  Error checking paths: {e}")

print("\nRecommended env vars for TP > 1:")
print("  export AITER_QUICK_REDUCE_QUANTIZATION=INT4")
print("  export ATOM_USE_CUSTOM_ALL_GATHER=1")
print("  export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1")
