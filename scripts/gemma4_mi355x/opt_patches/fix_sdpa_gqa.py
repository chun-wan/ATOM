filepath = '/app/ATOM/atom/model_ops/attention_mha.py'
with open(filepath, 'r') as f:
    content = f.read()

old = """        except RuntimeError:
            import torch.nn.functional as F
            # CK flash attn fails for head_dim > 256; use PyTorch SDPA
            n = q.shape[0]
            q3 = q.unsqueeze(0).transpose(1, 2)  # [1, heads, seq, dim]
            k3 = k.unsqueeze(0).transpose(1, 2)
            v3 = v.unsqueeze(0).transpose(1, 2)
            o = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True, scale=self.scale)
            o = o.transpose(1, 2).squeeze(0)  # back to [seq, heads, dim]"""

new = """        except RuntimeError:
            import torch.nn.functional as F
            # CK flash attn fails for head_dim > 256; use PyTorch SDPA
            q3 = q.unsqueeze(0).transpose(1, 2)  # [1, q_heads, seq, dim]
            k3 = k.unsqueeze(0).transpose(1, 2)  # [1, kv_heads, seq, dim]
            v3 = v.unsqueeze(0).transpose(1, 2)
            # Handle GQA: expand KV heads to match Q heads
            if q3.shape[1] != k3.shape[1]:
                rep = q3.shape[1] // k3.shape[1]
                k3 = k3.repeat_interleave(rep, dim=1)
                v3 = v3.repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True, scale=self.scale)
            o = o.transpose(1, 2).squeeze(0)"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('NOT_FOUND')
