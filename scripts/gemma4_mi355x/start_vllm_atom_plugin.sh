#!/bin/bash
set -e
# Fix safetensors index
python3 -c "
import json, glob, os
from safetensors import safe_open
model_dir = '/workspace/models/gemma-4-31b-it'
idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
wm = {}
for sf in sorted(glob.glob(os.path.join(model_dir, 'model-*.safetensors'))):
    fn = os.path.basename(sf)
    with safe_open(sf, framework='pt') as f:
        for k in f.keys():
            if k not in wm:
                wm[k] = fn
            else:
                with safe_open(os.path.join(model_dir, wm[k]), framework='pt') as f2:
                    if f.get_tensor(k).numel() > f2.get_tensor(k).numel():
                        wm[k] = fn
idx = {'metadata': {'total_size': 0}, 'weight_map': wm}
with open(idx_path, 'w') as f:
    json.dump(idx, f, indent=2)
print(f'Index rebuilt: {len(wm)} entries')
"

# Apply patches
python3 /workspace/patch_vllm_renderer.py

# Install ATOM + patch MLA
pip install -q git+https://github.com/ROCm/ATOM.git
pip install -q --no-deps git+https://github.com/huggingface/transformers.git
python3 /workspace/patch_all_crashes.py

# Start vLLM (ATOM plugin auto-detected via entry_points)
exec vllm serve \
    --model /workspace/models/gemma-4-31b-it \
    --host 0.0.0.0 --port 8000 \
    --attention-backend TRITON_ATTN \
    --trust-remote-code --max-model-len 4096
