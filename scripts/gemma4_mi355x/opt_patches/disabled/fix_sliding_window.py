filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
import os
os.chmod(filepath, 0o666) if os.stat(filepath).st_uid == os.getuid() else None
with open(filepath, 'r') as f:
    content = f.read()

# The Attention factory also passes sliding_window from config, causing duplicate.
# Use per_layer_sliding_window instead, or remove from Attention() call.
old = """        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            sliding_window=sw,
            q_norm=q_norm_for_attn,
            k_norm=k_norm_for_attn,
            prefix=f"{prefix}.attn",
        )"""

new = """        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            per_layer_sliding_window=sw,
            q_norm=q_norm_for_attn,
            k_norm=k_norm_for_attn,
            prefix=f"{prefix}.attn",
        )"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('OLD_NOT_FOUND')
    idx = content.find('self.attn = Attention(')
    if idx >= 0:
        print('CONTEXT:', content[idx:idx+400])
