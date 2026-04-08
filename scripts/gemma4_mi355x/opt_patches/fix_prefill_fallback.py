filepath = '/app/ATOM/atom/model_ops/attention_mha.py'
with open(filepath, 'r') as f:
    content = f.read()

old = """    @mark_trace(prefix="prefill_attention", torch_compile=False)
    def prefill_attention(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        # variable lenth attention use key value as input
        attn_metadata = fwd_ctx.attn_metadata
        sliding_window = (
            (self.sliding_window, 0, 0)
            if self.sliding_window is not None
            else (-1, -1, 0)
        )
        o = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
            window_size=sliding_window,
            sink_ptr=self.sinks,
        )

        return o"""

new = """    @mark_trace(prefix="prefill_attention", torch_compile=False)
    def prefill_attention(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        attn_metadata = fwd_ctx.attn_metadata
        sliding_window = (
            (self.sliding_window, 0, 0)
            if self.sliding_window is not None
            else (-1, -1, 0)
        )
        try:
            o = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=attn_metadata.max_seqlen_q,
                max_seqlen_k=attn_metadata.max_seqlen_k,
                min_seqlen_q=attn_metadata.min_seqlen_q,
                dropout_p=attn_metadata.dropout_p,
                softmax_scale=self.scale,
                causal=True,
                window_size=sliding_window,
                sink_ptr=self.sinks,
            )
        except RuntimeError:
            import torch.nn.functional as F
            # CK flash attn fails for head_dim > 256; use PyTorch SDPA
            n = q.shape[0]
            q3 = q.unsqueeze(0).transpose(1, 2)  # [1, heads, seq, dim]
            k3 = k.unsqueeze(0).transpose(1, 2)
            v3 = v.unsqueeze(0).transpose(1, 2)
            o = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True, scale=self.scale)
            o = o.transpose(1, 2).squeeze(0)  # back to [seq, heads, dim]

        return o"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
