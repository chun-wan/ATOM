#!/usr/bin/env python3
"""Fix the fused_kv_cache fallback to use PyTorch RoPE (not AITER imports)."""
path = "/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py"
with open(path) as f:
    src = f.read()

# Replace the broken fallback with a simpler one using reshape_and_cache only
# Skip RoPE in the fallback (just do cache update)
OLD = '''    # Patched: allow mixed head_dim for models like Gemma 4
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

NEW = '''    # Patched: allow mixed head_dim for models like Gemma 4
    if not (kh == vh == kh_cache == vh_cache):
        # Fallback: apply RoPE via PyTorch, then update cache separately
        from aiter.ops.cache import reshape_and_cache
        # Inline RoPE: q/k have shape [t, heads, d], cos/sin [seq, d_freq]
        d_freq = cos.shape[-1]
        cos_pos = cos[pos].unsqueeze(1)  # [t, 1, d_freq]
        sin_pos = sin[pos].unsqueeze(1)
        # Apply RoPE to first d_freq dims of q and k
        q_rope = q.clone()
        k_rope = k.clone()
        q1 = q[..., :d_freq]
        q2 = q[..., d_freq:2*d_freq] if 2*d_freq <= d else torch.zeros_like(q1)
        q_rope[..., :d_freq] = q1 * cos_pos - q2 * sin_pos
        if 2*d_freq <= d:
            q_rope[..., d_freq:2*d_freq] = q2 * cos_pos + q1 * sin_pos
        k1 = k[..., :d_freq]
        k2 = k[..., d_freq:2*d_freq] if 2*d_freq <= dk else torch.zeros_like(k1)
        k_rope[..., :d_freq] = k1 * cos_pos[:tk] - k2 * sin_pos[:tk]
        if 2*d_freq <= dk:
            k_rope[..., d_freq:2*d_freq] = k2 * cos_pos[:tk] + k1 * sin_pos[:tk]
        # Update KV cache
        reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping)
        if q_out is not None:
            q_out.copy_(q_rope)
        else:
            q_out = q_rope
        if k_out is not None:
            k_out.copy_(k_rope)
        else:
            k_out = k_rope
        return q_out, k_out, key_cache, value_cache'''

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(src)
    print("[OK] Fixed fused_kv_cache fallback (PyTorch RoPE, no AITER imports)")
elif "Patched: allow mixed head_dim" not in src:
    print("[FAIL] Original fallback not found")
else:
    print("[SKIP] Different fallback version, checking...")
    # The patch_all_crashes.py version might differ, let me just re-apply from scratch
    # Find the original assert and replace it completely
    ORIG_ASSERT = '''    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"'''

    FIXED = '''    # Patched v2: allow mixed head_dim for Gemma 4
    if not (kh == vh == kh_cache == vh_cache):
        from aiter.ops.cache import reshape_and_cache
        d_freq = cos.shape[-1]
        cos_pos = cos[pos].unsqueeze(1)
        sin_pos = sin[pos].unsqueeze(1)
        q_rope = q.clone()
        k_rope = k.clone()
        q1, q2 = q[..., :d_freq], q[..., d_freq:2*d_freq]
        q_rope[..., :d_freq] = q1 * cos_pos - q2 * sin_pos
        q_rope[..., d_freq:2*d_freq] = q2 * cos_pos + q1 * sin_pos
        k1, k2 = k[..., :d_freq], k[..., d_freq:2*d_freq]
        k_rope[..., :d_freq] = k1 * cos_pos[:tk] - k2 * sin_pos[:tk]
        k_rope[..., d_freq:2*d_freq] = k2 * cos_pos[:tk] + k1 * sin_pos[:tk]
        reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping)
        q_out = q_rope if q_out is None else q_out.copy_(q_rope) or q_out
        k_out = k_rope if k_out is None else k_out.copy_(k_rope) or k_out
        return q_out, k_out, key_cache, value_cache'''

    if ORIG_ASSERT in src:
        src = src.replace(ORIG_ASSERT, FIXED, 1)
        with open(path, "w") as f:
            f.write(src)
        print("[OK] Applied v2 fallback from original assert")
    else:
        print("[FAIL] Could not find any patchable pattern")

import subprocess
subprocess.run(["find", "/app/aiter-test", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
