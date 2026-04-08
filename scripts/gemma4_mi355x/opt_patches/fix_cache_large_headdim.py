filepath = '/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace CK reshape_and_cache with a PyTorch fallback that handles any head_dim
old = '        reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto")'

new = '''        # PyTorch fallback for reshape_and_cache (CK fails for head_dim > 256)
        try:
            reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto")
        except RuntimeError:
            # Manual KV cache update when CK kernel doesn't support the shape
            num_tokens = k_rope.shape[0]
            for i in range(num_tokens):
                slot = slot_mapping[i].item()
                block_idx = slot // key_cache.shape[-2] if key_cache.dim() == 4 else slot // key_cache.shape[2]
                block_off = slot % (key_cache.shape[-2] if key_cache.dim() == 4 else key_cache.shape[2])
                if key_cache.dim() == 4:  # [num_blocks, num_heads, block_size, head_dim]
                    key_cache[block_idx, :, block_off, :k_rope.shape[-1]] = k_rope[i]
                    value_cache[block_idx, :, block_off, :v.shape[-1]] = v[i]
                else:  # [num_blocks, num_heads, head_dim//x, block_size, x]
                    key_cache[block_idx, :, :, block_off, :] = k_rope[i].view(k_rope.shape[1], -1, key_cache.shape[-1])
                    value_cache[block_idx, :, :, block_off, :] = v[i].view(v.shape[1], -1, value_cache.shape[-1])'''

count = content.count(old)
if count > 0:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f'PATCHED_OK: fixed {count} calls')
else:
    print('NOT_FOUND')
    idx = content.find('reshape_and_cache')
    if idx >= 0:
        print('CTX:', content[max(0,idx-50):idx+100])
