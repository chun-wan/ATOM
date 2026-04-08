filepath = '/app/ATOM/atom/model_ops/layernorm.py'
with open(filepath, 'r') as f:
    content = f.read()

# Add a method to invalidate the cached weight after loading
old = """    def _get_weight_plus_one(self) -> torch.Tensor:
        if not hasattr(self, "_weight_plus_one") or self._weight_plus_one is None:
            self._weight_plus_one = self.weight.data + 1.0
        return self._weight_plus_one"""

new = """    def _get_weight_plus_one(self) -> torch.Tensor:
        if not hasattr(self, "_weight_plus_one") or self._weight_plus_one is None:
            self._weight_plus_one = (self.weight.data + 1.0).contiguous()
        return self._weight_plus_one

    def _invalidate_weight_cache(self):
        self._weight_plus_one = None"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('ALREADY_PATCHED_OR_NOT_FOUND')
