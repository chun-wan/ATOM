filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

# Match with the blank line between invalidate and return
old = """                module._invalidate_weight_cache()

        return loaded_weights_record"""

new = """                module._invalidate_weight_cache()
        # Copy K weights to V slot for k_eq_v layers (V=K, no v_proj in checkpoint)
        for layer in self.model.layers:
            attn = layer.self_attn
            if getattr(attn, 'attention_k_eq_v', False) and hasattr(attn, 'qkv_proj'):
                w = attn.qkv_proj.weight.data
                q_sz = attn.q_size
                kv_sz = attn.kv_size
                w[q_sz + kv_sz : q_sz + 2*kv_sz].copy_(w[q_sz : q_sz + kv_sz])
        return loaded_weights_record"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
