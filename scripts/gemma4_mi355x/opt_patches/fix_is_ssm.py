import sys

filepath = '/app/ATOM/atom/model_ops/attentions/aiter_attention.py'
with open(filepath, 'r') as f:
    content = f.read()

# Add is_ssm() classmethod to AiterBackend
old = """class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:"""

new = """class AiterBackend(AttentionBackend):
    @classmethod
    def is_ssm(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('OLD_NOT_FOUND')
    idx = content.find('class AiterBackend')
    if idx >= 0:
        print('CONTEXT:', content[idx:idx+300])
