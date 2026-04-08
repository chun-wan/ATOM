import os
filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'

with open(filepath, 'r') as f:
    content = f.read()

# The issue: get_rope() doesn't support rope_type="proportional".
# Fix: When building rope_scaling for get_rope(), convert "proportional" to "default"
# since Gemma 4 "proportional" is just standard RoPE with different theta + partial rotation.
# The rotary_dim and theta are already handled correctly.

old = """        rotary_dim = head_dim
        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0) if rope_scaling else 1.0
        if partial_rotary_factor < 1.0:
            rotary_dim = int(head_dim * partial_rotary_factor)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )"""

new = """        rotary_dim = head_dim
        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0) if rope_scaling else 1.0
        if partial_rotary_factor < 1.0:
            rotary_dim = int(head_dim * partial_rotary_factor)

        # AITER get_rope doesn't support rope_type="proportional" -- convert to "default"
        clean_rope_scaling = None
        if rope_scaling:
            rtype = rope_scaling.get("rope_type", "default")
            if rtype in ("proportional", "default"):
                clean_rope_scaling = None  # use plain RoPE with base=rope_theta
            else:
                clean_rope_scaling = dict(rope_scaling)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=clean_rope_scaling,
        )"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('OLD_NOT_FOUND')
    idx = content.find('self.rotary_emb = get_rope(')
    if idx >= 0:
        print('CTX:', content[idx-200:idx+200])
