"""Fix GemmaRMSNorm: use pure native forward, no torch.compile, no AITER."""
filepath = '/app/ATOM/atom/model_ops/layernorm.py'
with open(filepath, 'r') as f:
    content = f.read()

old = """    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(self.forward_static)  # type: ignore
            self._is_compiled = True
        return self.forward_native(x, residual)"""

new = """    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
