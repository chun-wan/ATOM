"""Rewrite Gemma4Attention for k_eq_v layers: use separate Q, K projections."""
import os

filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Replace the QKVParallelLinear block with conditional per-layer-type projections
old_proj = """        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )"""

new_proj = """        if is_global and attention_k_eq_v:
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
            self.k_proj = None
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )"""

if old_proj in content:
    content = content.replace(old_proj, new_proj)
    print('PROJ_REPLACED')
else:
    print('PROJ_NOT_FOUND')

# 2. Fix the forward method to handle separate projections
old_fwd = """        qkv = self.qkv_proj(hidden_states)
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

new_fwd = """        if self.qkv_proj is not None:
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

if old_fwd in content:
    content = content.replace(old_fwd, new_fwd)
    print('FWD_REPLACED')
else:
    print('FWD_NOT_FOUND')

# 3. Remove q/k/v from packed_modules_mapping (they don't work with per-layer variation)
# Keep gate_proj/up_proj packing (same for all layers)
old_packed = """    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }"""

new_packed = """    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }"""

if old_packed in content:
    content = content.replace(old_packed, new_packed)
    print('PACKED_REPLACED')
else:
    print('PACKED_NOT_FOUND')

# 4. Remove the old k_eq_v weight copy hack from load_weights
old_kev_fix = """        # Fix k_eq_v layers: copy K weights to V slot in qkv_proj
        for layer in self.model.layers:
            attn = layer.self_attn
            if getattr(attn, 'attention_k_eq_v', False):
                w = attn.qkv_proj.weight.data
                q_size = attn.q_size
                kv_size = attn.kv_size
                # qkv layout: [q_size + kv_size + kv_size, hidden_size]
                k_start = q_size
                v_start = q_size + kv_size
                w[v_start:v_start+kv_size] = w[k_start:k_start+kv_size].clone()"""

if old_kev_fix in content:
    content = content.replace(old_kev_fix, "")
    print('KEV_HACK_REMOVED')
else:
    print('KEV_HACK_NOT_FOUND')

with open(filepath, 'w') as f:
    f.write(content)
print('DONE')
