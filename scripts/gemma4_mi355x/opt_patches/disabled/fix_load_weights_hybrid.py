"""Fix load_weights to handle hybrid QKV loading:
- Sliding layers: q_proj+k_proj+v_proj -> qkv_proj (manual packing)
- Full-attention k_eq_v layers: q_proj -> self.q_proj, k_proj -> self.k_proj (direct)
"""
import os

filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

# Restore q/k/v in packed_modules_mapping for sliding layers
old_packed = """    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }"""

new_packed = """    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        # QKV packing only applies to sliding layers (which have qkv_proj).
        # Full-attention layers have separate q_proj/k_proj (loaded directly).
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }"""

if old_packed in content:
    content = content.replace(old_packed, new_packed)
    print('PACKED_RESTORED')
else:
    print('PACKED_NOT_FOUND')

with open(filepath, 'w') as f:
    f.write(content)
print('DONE')
