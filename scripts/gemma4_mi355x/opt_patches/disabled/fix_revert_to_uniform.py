"""Revert to uniform QKVParallelLinear for all layers.
K->V copy happens in load_weights for k_eq_v layers.
"""
filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

# Revert projection init to uniform QKVParallelLinear
old_proj = """        if is_global and attention_k_eq_v:
            # Full-attention k_eq_v layers: separate Q, K projections (no V)
            from atom.model_ops.linear import ColumnParallelLinear
            self.q_proj = ColumnParallelLinear(
                hidden_size, self.q_size, bias=False,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size, self.kv_size, bias=False,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.k_proj",
            )
            self.qkv_proj = None
        else:
            # Sliding-attention layers: merged QKV
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.q_proj = None
            self.k_proj = None"""

new_proj = """        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )"""

if old_proj in content:
    content = content.replace(old_proj, new_proj)
    print('PROJ_REVERTED')
else:
    print('PROJ_NOT_FOUND')

# Revert forward to uniform QKV split
old_fwd = """        if self.qkv_proj is not None:
            # Sliding layers: merged QKV, norms handled by fused kernel
            qkv = self.qkv_proj(hidden_states)
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs)
        else:
            # Full-attention k_eq_v layers: separate Q, K projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = k  # V reuses K
            o = self.attn(q, k, v, positions, **model_kwargs)"""

new_fwd = """        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        if self.attention_k_eq_v:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = k
            o = self.attn(q, k, v, positions, **model_kwargs)
        else:
            o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs)"""

if old_fwd in content:
    content = content.replace(old_fwd, new_fwd)
    print('FWD_REVERTED')
else:
    print('FWD_NOT_FOUND')

# Add K->V weight copy back in load_weights
old_load = """        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        return loaded_weights_record"""

new_load = """        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        # Copy K weights to V slot for k_eq_v layers
        for layer in self.model.layers:
            attn = layer.self_attn
            if getattr(attn, 'attention_k_eq_v', False):
                w = attn.qkv_proj.weight.data
                q_sz = attn.q_size
                kv_sz = attn.kv_size
                w[q_sz + kv_sz : q_sz + 2*kv_sz].copy_(w[q_sz : q_sz + kv_sz])
        return loaded_weights_record"""

if old_load in content:
    content = content.replace(old_load, new_load)
    print('LOAD_FIXED')
else:
    print('LOAD_NOT_FOUND')

with open(filepath, 'w') as f:
    f.write(content)
print('DONE')
