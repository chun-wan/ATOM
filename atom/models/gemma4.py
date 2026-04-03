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
"""Gemma 4 model implementation for ATOM with AITER operators."""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from aiter.dist.parallel_state import get_tp_group
from aiter.rotary_embedding import get_rope

from atom.config import Config
from atom.model_config.gemma4 import Gemma4Config, Gemma4TextConfig
from atom.model_loader.loader import load_model_in_plugin_mode
from atom.model_ops.activation import SiluAndMul
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
from atom.utils.decorators import support_torch_compile


class Gemma4GeluAndMul(nn.Module):
    """GELU-gated MLP activation: GELU(gate) * up.

    Used by Gemma 4's hidden_activation='gelu_pytorch_tanh'.
    """

    def forward(
        self,
        x: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up


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
            prefix=f"{prefix}.attn",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

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
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.attention_k_eq_v:
            v = k

        o = self.attn(q, k, v, positions, **model_kwargs)
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
        self.act_fn = Gemma4GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Gemma4SparseMoeBlock(nn.Module):
    """Sparse MoE block for Gemma 4 26B-A4B variant."""

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
        # Gemma models scale embeddings by sqrt(hidden_size)
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
            logits = logits / self.logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.logit_softcapping

        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
        )
        return loaded_weights_record
