filepath = '/app/ATOM/atom/model_ops/attention_mha.py'
with open(filepath, 'r') as f:
    content = f.read()
old = """        if ctx.is_prefill:
            if self.head_dim > 256:
                return self.prefill_attention_triton
            return self.prefill_attention"""
new = """        if ctx.is_prefill:
            return self.prefill_attention"""
if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('REVERTED_OK')
else:
    print('ALREADY_CLEAN')
