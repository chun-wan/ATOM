"""Fix QKV weight packing for k_eq_v layers.
After load_weights, for full-attention layers where attention_k_eq_v=True,
copy K weights to V slot in the qkv_proj weight matrix.
"""
import os
filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

# Find load_weights and add k_eq_v weight fix after loading
old = """        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        return loaded_weights_record"""

new = """        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        # Fix k_eq_v layers: copy K weights to V slot in qkv_proj
        for layer in self.model.layers:
            attn = layer.self_attn
            if getattr(attn, 'attention_k_eq_v', False):
                w = attn.qkv_proj.weight.data
                q_size = attn.q_size
                kv_size = attn.kv_size
                # qkv layout: [q_size + kv_size + kv_size, hidden_size]
                k_start = q_size
                v_start = q_size + kv_size
                w[v_start:v_start+kv_size] = w[k_start:k_start+kv_size].clone()
        return loaded_weights_record"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
    idx = content.find('loaded_weights_record')
    if idx >= 0:
        print('CTX:', content[idx-100:idx+200])
