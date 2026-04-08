filepath = '/app/ATOM/atom/model_ops/attention_mla.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

# Find and clean up the corrupted lazy import section (lines ~20-35)
# Replace everything from first "gather_kv_b_proj" to the end of the duplicated block
new_lines = []
skip_until_clean = False
inserted_lazy = False

for i, line in enumerate(lines):
    # Skip all corrupted lazy import lines
    if 'gather_kv_b_proj' in line and not inserted_lazy:
        if not skip_until_clean:
            skip_until_clean = True
            # Insert clean lazy import block
            new_lines.append('gather_kv_b_proj = None\n')
            new_lines.append('def _get_gather_kv_b_proj():\n')
            new_lines.append('    global gather_kv_b_proj\n')
            new_lines.append('    if gather_kv_b_proj is None:\n')
            new_lines.append('        from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj as _fn\n')
            new_lines.append('        gather_kv_b_proj = _fn\n')
            new_lines.append('    return gather_kv_b_proj\n')
            new_lines.append('\n')
            inserted_lazy = True
        continue
    if skip_until_clean and ('gather_kv_b_proj' in line or 'def _get_gather' in line or line.strip().startswith('global gather')):
        continue
    if skip_until_clean and line.strip() and not line.startswith(' ') and 'gather' not in line and 'def _get' not in line:
        skip_until_clean = False
    if not skip_until_clean:
        new_lines.append(line)

with open(filepath, 'w') as f:
    f.writelines(new_lines)
print(f'FIXED: wrote {len(new_lines)} lines (was {len(lines)})')
