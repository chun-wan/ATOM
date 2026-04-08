import sys

filepath = '/app/aiter-test/aiter/ops/triton/fusions/fused_kv_cache.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix the RoPE fallback: cos has shape [max_pos, 1, 1, d_freq],
# so cos[pos] gives [t, 1, 1, d_freq]. The old code did .unsqueeze(1)
# making it 5D [t, 1, 1, 1, d_freq]. Fix: flatten cos/sin to 2D first.
old = """        d_freq = cos.shape[-1]
        cos_pos = cos[pos].unsqueeze(1)  # [t, 1, d_freq]
        sin_pos = sin[pos].unsqueeze(1)"""

new = """        # cos/sin cache may be 4D [max_pos, 1, 1, d_freq] -- flatten to 2D
        cos_2d = cos.view(cos.shape[0], -1)  # [max_pos, d_freq]
        sin_2d = sin.view(sin.shape[0], -1)
        d_freq = cos_2d.shape[-1]
        cos_pos = cos_2d[pos].unsqueeze(1)  # [t, 1, d_freq]
        sin_pos = sin_2d[pos].unsqueeze(1)"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('OLD_NOT_FOUND')
    # Show context around cos_pos
    idx = content.find('cos_pos')
    if idx >= 0:
        print('CONTEXT:', content[idx-100:idx+200])
    else:
        print('cos_pos not found in file at all')
