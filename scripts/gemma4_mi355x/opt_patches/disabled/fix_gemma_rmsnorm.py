import sys

filepath = '/app/ATOM/atom/model_ops/layernorm.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace GemmaRMSNorm.forward_cuda to use AITER kernels
# The Gemma RMSNorm does x * (1 + weight) instead of x * weight.
# We pre-compute weight+1 and use rmsnorm2d_fwd_ / rmsnorm2d_fwd_with_add_

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
            self._weight_plus_one = self.weight.data + 1.0
        return self._weight_plus_one

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
    print('OLD_NOT_FOUND')
    idx = content.find('class GemmaRMSNorm')
    if idx >= 0:
        end = content.find('class ', idx + 20)
        if end == -1:
            end = idx + 2000
        print('FULL CLASS:', content[idx:end])
