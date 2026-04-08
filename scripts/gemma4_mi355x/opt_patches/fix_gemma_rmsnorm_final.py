"""Replace GemmaRMSNorm.forward_cuda: original torch.compile version -> AITER with dtype-safe weight+1.0"""
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
        return self.forward_native(x, residual)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)"""

new = """    def _get_weight_plus_one(self) -> torch.Tensor:
        if not hasattr(self, "_weight_plus_one") or self._weight_plus_one is None:
            # CRITICAL: must keep weight dtype (bf16). Without .to(), +1.0 promotes to f32
            # and AITER rmsnorm2d_fwd reads f32 bytes as bf16 -> corrupt output
            self._weight_plus_one = (self.weight.data.float() + 1.0).to(self.weight.dtype).contiguous()
        return self._weight_plus_one

    def _invalidate_weight_cache(self):
        self._weight_plus_one = None

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)
        w = self._get_weight_plus_one()
        dim = w.shape[0]
        if residual is None:
            x = rmsnorm2d_fwd_(x, w, self.variance_epsilon, dim)
            return x
        else:
            x, residual = rmsnorm2d_fwd_with_add_(
                x, w, residual, self.variance_epsilon, dim
            )
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
