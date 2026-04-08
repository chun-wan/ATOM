filepath = '/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py'
with open(filepath, 'r') as f:
    content = f.read()

old = '        reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto")'
new = """        try:
            reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto")
        except RuntimeError:
            # CK fails for head_dim > 256; use manual PyTorch KV cache update
            for i in range(k_rope.shape[0]):
                slot = slot_mapping[i].item()
                block_idx = slot // key_cache.shape[2]
                block_off = slot % key_cache.shape[2]
                if key_cache.dim() == 4:
                    key_cache[block_idx, :, block_off, :] = k_rope[i]
                    value_cache[block_idx, :, block_off, :] = v[i]
                else:
                    kk = k_rope[i].view(k_rope.shape[1], -1, key_cache.shape[-1])
                    vv = v[i].view(v.shape[1], -1, value_cache.shape[-1])
                    key_cache[block_idx, :, :, block_off, :] = kk
                    value_cache[block_idx, :, :, block_off, :] = vv"""

count = content.count(old)
content = content.replace(old, new)
with open(filepath, 'w') as f:
    f.write(content)
print(f'REPLACED {count} occurrences')
