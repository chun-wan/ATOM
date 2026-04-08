"""
Complete rewrite of Gemma4Attention based on vLLM reference.
Key fixes:
1. Add v_norm (missing -- vLLM has it)
2. Unified forward path (no branching for k_eq_v)
3. Remove _GemmaNormWrapper (not needed for correctness-first)
4. Apply norms always, v_norm conditional on non-k_eq_v
"""
import os

filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# ===== FIX 1: Remove _GemmaNormWrapper class =====
old_wrapper = '''
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
if old_wrapper in content:
    content = content.replace(old_wrapper, '\n')
    changes += 1
    print('[1] Removed _GemmaNormWrapper')
else:
    print('[1] _GemmaNormWrapper not found (already removed?)')

# ===== FIX 2: Replace Attention __init__ norm/attn section =====
# Add v_norm, remove GemmaNormWrapper plumbing, simplify attn constructor
old_attn_init = """        sw = sliding_window if not is_global else None
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
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
            per_layer_sliding_window=sw,
            q_norm=q_norm_for_attn,
            k_norm=k_norm_for_attn,
            prefix=f"{prefix}.attn",
        )"""

new_attn_init = """        sw = sliding_window if not is_global else None
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.v_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
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
            per_layer_sliding_window=sw,
            prefix=f"{prefix}.attn",
        )"""

if old_attn_init in content:
    content = content.replace(old_attn_init, new_attn_init)
    changes += 1
    print('[2] Replaced attn init: added v_norm, removed GemmaNormWrapper')
else:
    print('[2] Attn init pattern not found')

# ===== FIX 3: Replace forward method with unified path =====
old_forward = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        if self.attention_k_eq_v:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = k
            o = self.attn(q, k, v, positions, **model_kwargs)
        else:
            o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs)
        output = self.o_proj(o)
        return output"""

new_forward = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # Apply per-head norms (AITER rmsnorm auto-reshapes to [-1, head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        if not self.attention_k_eq_v:
            v = self.v_norm(v)
        # For k_eq_v: V already equals K from weight loading (K weights copied to V slot)
        o = self.attn(q, k, v, positions, **model_kwargs)
        output = self.o_proj(o)
        return output"""

if old_forward in content:
    content = content.replace(old_forward, new_forward)
    changes += 1
    print('[3] Replaced forward with unified path + v_norm')
else:
    print('[3] Forward pattern not found')

with open(filepath, 'w') as f:
    f.write(content)

print(f'\nTotal changes: {changes}/3')
if changes == 3:
    print('ALL_FIXES_APPLIED')
else:
    print('PARTIAL_FIX')
