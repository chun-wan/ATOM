# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC and the ATOM contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma 4 model implementation for ATOM with AITER-optimized operators.

Operator optimization levels (MI355X / gfx950):
  - Attention decode: Gluon PA (Triton advanced, CDNA4-aware)
  - Attention prefill: Triton flash attention
  - GeluAndMul: AITER CUDA JIT kernel (gelu_tanh_and_mul)
  - RMSNorm: AITER CUDA (GemmaRMSNorm with +1 offset)
  - RoPE: AITER CUDA kernel per-layer-type config
  - GEMM: CK (Composable Kernel) for FP8/MXFP4
  - MoE: ASM + CK fused MoE (26B-A4B variant)
  - TopK softmax: ASM
  - Logit softcapping: fused Triton kernel (single launch)
"""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

import triton
import triton.language as tl

from aiter import gelu_tanh_and_mul
from aiter.dist.parallel_state import get_tp_group
from aiter.rotary_embedding import get_rope

from atom.config import Config, QuantizationConfig
from atom.model_config.gemma4 import Gemma4Config, Gemma4TextConfig
from atom.model_loader.loader import load_model_in_plugin_mode
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.models.utils import maybe_prefix
from atom.quant_spec import LayerQuantConfig
from atom.utils.decorators import support_torch_compile

from aiter import QuantType
from aiter.jit.utils.torch_guard import torch_compile_guard


# ---------------------------------------------------------------------------
# Fused logit softcapping Triton kernel
# Replaces 3 separate PyTorch ops (div + tanh + mul) with a single kernel.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_logit_softcap_kernel(
    logits_ptr,
    cap_inv,   # 1.0 / softcapping_value
    cap,       # softcapping_value
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(logits_ptr + offsets, mask=mask)
    x = tl.math.tanh(x * cap_inv) * cap
    tl.store(logits_ptr + offsets, x, mask=mask)


def fused_logit_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """In-place fused logit softcapping: tanh(x / cap) * cap."""
    n = logits.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _fused_logit_softcap_kernel[grid](logits, 1.0 / cap, cap, n, BLOCK_SIZE=BLOCK)
    return logits


# ---------------------------------------------------------------------------
# AITER-accelerated GeluAndMul activation
# Uses aiter.gelu_tanh_and_mul CUDA JIT kernel instead of PyTorch F.gelu
# ---------------------------------------------------------------------------

def _mxfp4_gelu_mul_quant_fuse_fake(
    x: torch.Tensor,
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N1 = x.shape
    N_half = N1 // 2
    out = torch.empty((M, N_half // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    MXFP4_QUANT_BLOCK_SIZE = 32
    SCALE_N_valid = (N_half + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if shuffle:
        SCALE_M = ((M + 255) // 256) * 256
        SCALE_N = ((SCALE_N_valid + 7) // 8) * 8
    else:
        SCALE_M = M
        SCALE_N = SCALE_N_valid
    scale = torch.empty((SCALE_M, SCALE_N), dtype=torch.float8_e8m0fnu, device=x.device)
    return out, scale


@torch_compile_guard(gen_fake=_mxfp4_gelu_mul_quant_fuse_fake, mutates_args=[])
def _mxfp4_gelu_mul_quant_fuse(
    x: torch.Tensor,
    shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_mxfp4_quant import (
        fused_reduce_act_mul_and_mxfp4_quant,
    )
    (x, x_scale), _ = fused_reduce_act_mul_and_mxfp4_quant(x, "gelu", shuffle=shuffle)
    return x, x_scale


class GeluAndMul(nn.Module):
    """AITER-accelerated GELU-gated activation for Gemma 4.

    Mirrors the SiluAndMul pattern but uses gelu_tanh_and_mul CUDA kernel.
    Supports FP8 and MXFP4 fused quantization paths.
    """

    def __init__(
        self,
        fused_quant: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fused_quant = fused_quant
        layer_quant_config = (
            LayerQuantConfig()
            if quant_config is None
            else quant_config.get_layer_quant_config(prefix)
        )
        self.quant_type = layer_quant_config.quant_type
        self.params_dtype = layer_quant_config.quant_dtype

    def forward_native(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up

    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x_scale is not None and self.fused_quant:
            from aiter.ops.triton.fused_fp8_quant import (
                fused_gelu_mul_fp8_per_tensor_static_quant,
            )
            import aiter as rocm_aiter
            x = fused_gelu_mul_fp8_per_tensor_static_quant(
                x, x_scale, dtype_quant=rocm_aiter.dtypes.fp8
            )
            return x, x_scale
        elif (
            x_scale is None
            and self.fused_quant
            and self.quant_type.value == QuantType.per_1x32.value
        ):
            return _mxfp4_gelu_mul_quant_fuse(x, shuffle=True)
        else:
            out = torch.empty(
                [*x.shape[:-1], x.shape[-1] // 2], device=x.device, dtype=x.dtype
            )
            gelu_tanh_and_mul(out, x)
            return out


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class _GemmaNormWrapper:
    """Wraps GemmaRMSNorm to expose (weight+1) as .weight for fused kernels."""
    def __init__(self, gemma_norm):
        self._norm = gemma_norm
    @property
    def weight(self):
        return self._norm.weight.data + 1.0
    @property
    def eps(self):
        return self._norm.variance_epsilon

class Gemma4Attention(nn.Module):
    """Multi-head attention for Gemma 4 with sliding/global window support.

    Gemma 4 uses different head_dim and num_kv_heads for global vs sliding
    attention layers, and applies per-type RoPE configurations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: dict | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        layer_num: int,
        atom_config: Config,
        is_global: bool = False,
        attention_k_eq_v: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_global = is_global
        self.attention_k_eq_v = attention_k_eq_v

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rotary_dim = head_dim
        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0) if rope_scaling else 1.0
        if partial_rotary_factor < 1.0:
            rotary_dim = int(head_dim * partial_rotary_factor)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        sw = sliding_window if not is_global else None
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        # Pass wrapped norms to Attention for fused QK-norm-RoPE path
        # Only enable fused path for non-k_eq_v layers (sliding attention)
        q_norm_for_attn = _GemmaNormWrapper(self.q_norm) if not attention_k_eq_v else None
        k_norm_for_attn = _GemmaNormWrapper(self.k_norm) if not attention_k_eq_v else None
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            sliding_window=sw,
            q_norm=q_norm_for_attn,
            k_norm=k_norm_for_attn,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        if self.attention_k_eq_v:
            # Full-attention layers: manual norm (fused path disabled for k_eq_v)
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = k
            o = self.attn(q, k, v, positions, **model_kwargs)
        else:
            # Sliding layers: norms handled by fused QK-norm-RoPE kernel
            o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs)
        output = self.o_proj(o)
        return output


class Gemma4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul(
            quant_config=quant_config,
            prefix=f"{prefix}.act_fn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Gemma4SparseMoeBlock(nn.Module):
    """Sparse MoE block for Gemma 4 26B-A4B variant.

    Uses AITER FusedMoE (ASM + CK backend) for high-throughput expert dispatch.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.moe_intermediate_size = config.moe_intermediate_size

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        self.router = nn.Linear(
            config.hidden_size, self.num_experts, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(hidden_states)
        return self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )


class Gemma4DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma4TextConfig,
        atom_config: Config,
        layer_num: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.layer_type = config.layer_types[layer_num]
        is_global = self.layer_type == "full_attention"

        if is_global:
            num_kv_heads = config.num_global_key_value_heads
            head_dim = config.global_head_dim
            rope_params = config.rope_parameters.get("full_attention", {}) if config.rope_parameters else {}
        else:
            num_kv_heads = config.num_key_value_heads
            head_dim = config.head_dim
            rope_params = config.rope_parameters.get("sliding_attention", {}) if config.rope_parameters else {}

        rope_theta = rope_params.get("rope_theta", 10000.0)

        self.self_attn = Gemma4Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_params,
            sliding_window=config.sliding_window,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=layer_num,
            atom_config=atom_config,
            is_global=is_global,
            attention_k_eq_v=getattr(config, "attention_k_eq_v", True),
            prefix=f"{prefix}.self_attn",
        )

        if config.enable_moe_block:
            self.mlp = Gemma4SparseMoeBlock(
                config=config,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Gemma4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **model_kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Gemma4Model(nn.Module):

    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = atom_config.hf_config
        if hasattr(config, "text_config"):
            config = config.text_config

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config=config,
                    atom_config=atom_config,
                    layer_num=layer_num,
                    prefix=f"{prefix}.layers.{layer_num}",
                )
                for layer_num in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hidden_size = config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.hidden_size**0.5)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **model_kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = self.atom_config.hf_config
        text_config = self.hf_config
        if hasattr(self.hf_config, "text_config"):
            text_config = self.hf_config.text_config

        self.model = Gemma4Model(
            atom_config=self.atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logit_softcapping = getattr(
            text_config, "final_logit_softcapping", None
        )

        if text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            **model_kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)

        if self.logit_softcapping is not None and self.logit_softcapping > 0:
            logits = fused_logit_softcap(logits, self.logit_softcapping)

        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
        )
        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        return loaded_weights_record
