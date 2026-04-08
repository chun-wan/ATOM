"""Restore GemmaRMSNorm AITER kernel with dtype-safe weight+1.0.

This replaces both:
1. The native-only forward_cuda (from fix_gemma_rmsnorm_native.py)
2. The old AITER path with f32 dtype bug (from fix_gemma_rmsnorm.py)
"""
filepath = '/app/ATOM/atom/model_ops/layernorm.py'
with open(filepath, 'r') as f:
    content = f.read()

# Check which version is currently active
if 'def _get_weight_plus_one' in content:
    print('AITER path already present, fixing dtype')
    # Apply the dtype fix from fix_gemma_rmsnorm_dtype.py
    old = """            self._weight_plus_one = (self.weight.data + 1.0).contiguous()"""
    new = """            self._weight_plus_one = (self.weight.data.float() + 1.0).to(self.weight.dtype).contiguous()"""
    if old in content:
        content = content.replace(old, new)
        with open(filepath, 'w') as f:
            f.write(content)
        print('DTYPE_FIXED')
    else:
        # Maybe already fixed?
        if '.to(self.weight.dtype)' in content:
            print('ALREADY_FIXED')
        else:
            print('PATTERN_NOT_FOUND')
else:
    print('AITER path not present, adding it with correct dtype')
    # Need to add _get_weight_plus_one + forward_cuda using AITER
    # Find the native-only forward_cuda and replace it
    
    old_native = """    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)"""
    
    new_aiter = """    def _get_weight_plus_one(self) -> torch.Tensor:
        if not hasattr(self, "_weight_plus_one") or self._weight_plus_one is None:
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
            return x, residual"""
    
    if old_native in content:
        content = content.replace(old_native, new_aiter)
        with open(filepath, 'w') as f:
            f.write(content)
        print('AITER_PATH_RESTORED_WITH_DTYPE_FIX')
    else:
        print('NATIVE_PATTERN_NOT_FOUND')
        # Show what's there
        idx = content.find('class GemmaRMSNorm')
        fwd = content.find('def forward_cuda', idx) if idx > 0 else -1
        if fwd > 0:
            print('CURRENT:', content[fwd:fwd+200])
