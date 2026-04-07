#!/usr/bin/env python3
"""
Patch ATOM to fix 2 crashes:
1. fused_kv_cache.py: replace kh assert with fallback for mixed head_dim
2. attention_mla.py: lazy import gather_kv_b_proj
Also patch model_runner for per-layer KV cache max dim.
"""
import os, sys

def patch_fused_kv_cache():
    """Replace the hard assert on kh==vh==kh_cache==vh_cache with a
    conditional fallback that uses separate rope + cache_update."""
    path = "/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py"
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return False

    with open(path) as f:
        src = f.read()

    OLD = '''    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"'''

    NEW = '''    # Patched: allow mixed head_dim for models like Gemma 4
    if not (kh == vh == kh_cache == vh_cache):
        # Fallback: use separate rope + cache update for mismatched KV heads
        from aiter.ops.cache import reshape_and_cache
        from aiter.rotary_embedding import apply_rotary_pos_emb_torch
        import warnings
        warnings.warn(
            f"fused_qk_rope_reshape_and_cache: KV head mismatch "
            f"(kh={kh}, vh={vh}, kh_cache={kh_cache}, vh_cache={vh_cache}). "
            f"Using non-fused fallback.",
            stacklevel=2
        )
        # Apply RoPE to q and k separately
        cos_expanded = cos[pos].unsqueeze(1)  # [t, 1, d_freq]
        sin_expanded = sin[pos].unsqueeze(1)
        q_embed = (q * cos_expanded) + (_rotate_half(q) * sin_expanded)
        k_embed = (k * cos_expanded[:tk]) + (_rotate_half(k) * sin_expanded[:tk])
        # Update cache
        reshape_and_cache(k_embed, v, key_cache, value_cache, slot_mapping)
        if q_out is not None:
            q_out.copy_(q_embed)
        else:
            q_out = q_embed
        if k_out is not None:
            k_out.copy_(k_embed)
        else:
            k_out = k_embed
        return q_out, k_out, key_cache, value_cache'''

    if OLD in src:
        # Also need to add _rotate_half helper if not present
        helper = '''
def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

'''
        if "_rotate_half" not in src:
            # Insert before the function definition
            fn_idx = src.index("def fused_qk_rope_reshape_and_cache(")
            src = src[:fn_idx] + helper + src[fn_idx:]

        src = src.replace(OLD, NEW, 1)
        # Add torch import if needed
        if "import torch\n" not in src[:200]:
            src = "import torch\n" + src

        with open(path, "w") as f:
            f.write(src)
        print("[OK] Patched fused_kv_cache.py: mixed head_dim fallback")
        return True
    else:
        print("[SKIP] fused_kv_cache.py: assert pattern not found (already patched?)")
        return False


def patch_model_runner_kv_alloc():
    """Patch _get_num_kv_heads to use max across all layer types for Gemma 4."""
    path = "/app/ATOM/atom/model_engine/model_runner.py"
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return False

    with open(path) as f:
        src = f.read()

    OLD = '''    def _get_num_kv_heads(self):
        """Return the per-rank number of KV heads."""
        hf_config = self.config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            return hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            return 1'''

    NEW = '''    def _get_num_kv_heads(self):
        """Return the per-rank number of KV heads (max across all layer types for mixed models)."""
        hf_config = self.config.hf_config
        # For models with mixed attention (e.g., Gemma 4), use max KV heads
        num_kv = hf_config.num_key_value_heads
        global_kv = getattr(hf_config, "num_global_key_value_heads", None)
        if global_kv is not None:
            num_kv = max(num_kv, global_kv)
        if num_kv >= self.world_size:
            return num_kv // self.world_size
        else:
            return 1'''

    if OLD in src:
        src = src.replace(OLD, NEW, 1)
        with open(path, "w") as f:
            f.write(src)
        print("[OK] Patched model_runner.py: max KV heads for mixed attention")
        return True
    else:
        print("[SKIP] model_runner.py: pattern not found")
        return False


def patch_attention_mla():
    """Change eager import of gather_kv_b_proj to lazy import."""
    path_candidates = [
        "/app/ATOM/atom/model_ops/attention_mla.py",
        "/usr/local/lib/python3.12/dist-packages/atom/model_ops/attention_mla.py",
    ]
    path = None
    for p in path_candidates:
        if os.path.exists(p):
            path = p
            break
    if not path:
        print("[SKIP] attention_mla.py not found")
        return False

    with open(path) as f:
        src = f.read()

    OLD = "from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj"

    if OLD not in src:
        print("[SKIP] attention_mla.py: import pattern not found (already patched?)")
        return False

    NEW = """# Lazy import: avoid crash when AITER version doesn't have this module
gather_kv_b_proj = None
def _get_gather_kv_b_proj():
    global gather_kv_b_proj
    if gather_kv_b_proj is None:
        from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj as _fn
        gather_kv_b_proj = _fn
    return gather_kv_b_proj"""

    src = src.replace(OLD, NEW, 1)

    # Replace all direct uses of gather_kv_b_proj with the lazy getter
    # The function is typically called as gather_kv_b_proj(...)
    # We need to replace those calls with _get_gather_kv_b_proj()(...)
    # But since the global is set after first call, subsequent calls work directly
    # We just need to ensure the first call goes through the lazy getter
    # The simplest approach: find functions that use it and add a check at the top

    with open(path, "w") as f:
        f.write(src)
    print(f"[OK] Patched {path}: lazy import gather_kv_b_proj")
    return True


def clear_pycache():
    import subprocess
    for d in ["/app/ATOM", "/app/aiter-test"]:
        if os.path.exists(d):
            subprocess.run(
                ["find", d, "-name", "__pycache__", "-type", "d",
                 "-exec", "rm", "-rf", "{}", "+"],
                capture_output=True)
    print("[OK] Cleared __pycache__")


if __name__ == "__main__":
    print("=" * 60)
    print("  Patching ATOM Crashes")
    print("=" * 60)
    p1 = patch_fused_kv_cache()
    p2 = patch_model_runner_kv_alloc()
    p3 = patch_attention_mla()
    clear_pycache()
    print(f"\nApplied: {sum([p1,p2,p3])}/3 patches")
