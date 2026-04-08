"""
Enable fused QK-norm-RoPE for Gemma 4 by:
1. Creating a NormWrapper that presents GemmaRMSNorm's (weight+1) as .weight
2. Adding cos_sin_cache property to RotaryEmbedding
3. Passing q_norm/k_norm to Attention for sliding layers
4. For k_eq_v full-attention layers, keep manual path but after fused norm+rope, copy K to V cache
"""
import sys

# --- Part 1: Add cos_sin_cache property to RotaryEmbedding ---
filepath1 = '/app/aiter-test/aiter/rotary_embedding.py'
with open(filepath1, 'r') as f:
    content1 = f.read()

# Add cos_sin_cache property after the existing cos_cache/sin_cache registers
insert_after = """        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)"""

insert_property = """        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
        # Combined cache for fused QK-norm-RoPE kernel
        cos_sin = torch.cat([cos.squeeze(-2).squeeze(-2),
                             sin.squeeze(-2).squeeze(-2)], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin, persistent=False)"""

if insert_after in content1 and "cos_sin_cache" not in content1[:content1.find(insert_after)+len(insert_after)+200]:
    content1 = content1.replace(insert_after, insert_property, 1)
    with open(filepath1, 'w') as f:
        f.write(content1)
    print('ROPE_PATCHED_OK')
else:
    if "cos_sin_cache" in content1[:content1.find(insert_after)+len(insert_after)+200] if insert_after in content1 else False:
        print('ROPE_ALREADY_PATCHED')
    else:
        print('ROPE_NOT_FOUND')

# --- Part 2: Add GemmaNormWrapper to gemma4.py for fused path ---
filepath2 = '/app/ATOM/atom/models/gemma4.py'
with open(filepath2, 'r') as f:
    content2 = f.read()

# Add NormWrapper class before Gemma4Attention
norm_wrapper = '''
class _GemmaNormWrapper:
    """Wraps GemmaRMSNorm to expose (weight+1) as .weight for fused kernels."""
    def __init__(self, gemma_norm):
        self._norm = gemma_norm
    @property
    def weight(self):
        return self._norm.weight.data + 1.0
    @property
    def eps(self):
        return self._norm.variance_epsilon

'''

marker = "class Gemma4Attention(nn.Module):"
if "_GemmaNormWrapper" not in content2 and marker in content2:
    content2 = content2.replace(marker, norm_wrapper + marker)
    with open(filepath2, 'w') as f:
        f.write(content2)
    print('NORM_WRAPPER_ADDED')
else:
    if "_GemmaNormWrapper" in content2:
        print('NORM_WRAPPER_ALREADY_EXISTS')
    else:
        print('NORM_WRAPPER_MARKER_NOT_FOUND')

# --- Part 3: Pass norms to Attention constructor in Gemma4Attention ---
with open(filepath2, 'r') as f:
    content2 = f.read()

old_attn_init = """        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            sliding_window=sw,
            prefix=f"{prefix}.attn",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)"""

new_attn_init = """        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        # Pass wrapped norms to Attention for fused QK-norm-RoPE path
        # Only enable fused path for non-k_eq_v layers (sliding attention)
        q_norm_for_attn = _GemmaNormWrapper(self.q_norm) if not attention_k_eq_v else None
        k_norm_for_attn = _GemmaNormWrapper(self.k_norm) if not attention_k_eq_v else None
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            sliding_window=sw,
            q_norm=q_norm_for_attn,
            k_norm=k_norm_for_attn,
            prefix=f"{prefix}.attn",
        )"""

if old_attn_init in content2:
    content2 = content2.replace(old_attn_init, new_attn_init)

    # Also update forward: skip manual norm for sliding layers (fused does it)
    old_fwd = """        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.attention_k_eq_v:
            v = k

        o = self.attn(q, k, v, positions, **model_kwargs)"""

    new_fwd = """        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        if self.attention_k_eq_v:
            # Full-attention layers: manual norm (fused path disabled for k_eq_v)
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = k
            o = self.attn(q, k, v, positions, **model_kwargs)
        else:
            # Sliding layers: norms handled by fused QK-norm-RoPE kernel
            o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs)"""

    if old_fwd in content2:
        content2 = content2.replace(old_fwd, new_fwd)
        with open(filepath2, 'w') as f:
            f.write(content2)
        print('ATTN_INIT_AND_FWD_PATCHED')
    else:
        with open(filepath2, 'w') as f:
            f.write(content2)
        print('ATTN_INIT_PATCHED_BUT_FWD_NOT_FOUND')
else:
    print('ATTN_INIT_NOT_FOUND')
