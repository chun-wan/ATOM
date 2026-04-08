"""
Register Gemma 4 MTP model in ATOM's speculative decoding framework.
Creates placeholder registration and startup script for future use.
"""

# Part 1: Add Gemma 4 to eagle model dict
filepath = '/app/ATOM/atom/spec_decode/eagle.py'
with open(filepath, 'r') as f:
    content = f.read()

old = '''support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
}'''

new = '''support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
    # Gemma 4 MTP: placeholder for when Google releases the draft model
    # "Gemma4MTPModel": "atom.models.gemma4_mtp.Gemma4MTP",
}'''

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('EAGLE_DICT_UPDATED')
else:
    if 'Gemma4MTP' in content:
        print('EAGLE_DICT_ALREADY_UPDATED')
    else:
        print('EAGLE_DICT_NOT_FOUND')

# Part 2: Create startup script for speculative decoding
spec_script = '''#!/bin/bash
# Gemma 4 Speculative Decoding with MTP
# Requires: Gemma 4 MTP draft model (not yet released by Google)
# Usage: Start this after the MTP model is available
set -e

# Apply all patches
for f in /tmp/fix_*.py; do python3 "$f" 2>/dev/null || true; done
find /app -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

DRAFT_MODEL="/workspace/models/gemma-4-mtp"  # Update path when available

if [ ! -d "$DRAFT_MODEL" ]; then
    echo "ERROR: MTP draft model not found at $DRAFT_MODEL"
    echo "Gemma 4 MTP model has not been released yet."
    echo "Falling back to non-speculative mode..."
    exec python3 -m atom.entrypoints.openai_server \\
        --model /workspace/models/gemma-4-31b-it \\
        --host 0.0.0.0 --server-port 8001 \\
        --trust-remote-code --level 2 \\
        --kv_cache_dtype bf16 --tensor-parallel-size 1
fi

exec python3 -m atom.entrypoints.openai_server \\
    --model /workspace/models/gemma-4-31b-it \\
    --host 0.0.0.0 --server-port 8001 \\
    --trust-remote-code --level 2 \\
    --kv_cache_dtype bf16 --tensor-parallel-size 1 \\
    --method mtp --num-speculative-tokens 3
'''

with open('/workspace/start_atom_spec_decode.sh', 'w') as f:
    f.write(spec_script)
import os
os.chmod('/workspace/start_atom_spec_decode.sh', 0o755)
print('SPEC_DECODE_SCRIPT_CREATED')
print('DONE')
