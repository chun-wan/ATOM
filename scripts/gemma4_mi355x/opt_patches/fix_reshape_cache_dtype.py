filepath = '/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix all reshape_and_cache calls missing kv_cache_dtype
old = 'reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping)'
new = 'reshape_and_cache(k_rope, v, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto")'

count = content.count(old)
if count > 0:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f'PATCHED_OK: fixed {count} calls')
else:
    print('NOT_FOUND')
