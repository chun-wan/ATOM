"""Fix Gemma4Attention.forward to reshape Q/K/V per-head before norms (vLLM pattern)."""
filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

old = """        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # Apply per-head norms (AITER rmsnorm auto-reshapes to [-1, head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        if not self.attention_k_eq_v:
            v = self.v_norm(v)
        # For k_eq_v: V already equals K from weight loading (K weights copied to V slot)
        o = self.attn(q, k, v, positions, **model_kwargs)"""

new = """        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # Reshape to per-head, apply norm, reshape back (vLLM pattern)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)

        if not self.attention_k_eq_v:
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)

        o = self.attn(q, k, v, positions, **model_kwargs)"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
