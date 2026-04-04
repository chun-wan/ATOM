# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted for Gemma 4 GELU activation from moe_op_mxfp4_silu_fused.py

"""Fused MoE MXFP4 kernel with GELU-tanh activation for Gemma 4.

Gemma 4 MoE (26B-A4B) uses gelu_pytorch_tanh activation instead of SiLU.
This kernel replaces _silu_exp2 with _gelu_tanh in the fused MoE computation
so that the activation is applied inside the GEMM epilogue rather than
requiring a separate kernel launch.

Operator hierarchy: Triton (upgradeable to FlyDSL/CK with GELU support)
"""

import torch
import triton
import triton.language as tl
from typing import Any, Dict
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.activation import _gelu_tanh, _tanh
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils._triton.moe_common import _write_zeros_to_output
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.types import torch_to_triton_dtype

_LOGGER = AiterTritonLogger()


def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


_fused_moe_kernel_mxfp4_gelu_repr = make_kernel_repr(
    "_fused_moe_kernel_mxfp4_gelu",
    [
        "A_DTYPE_FORMAT",
        "B_DTYPE_FORMAT",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "SWIZZLE_MX_A",
        "SWIZZLE_MX_B",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_kernel_mxfp4_gelu_repr)
def _fused_moe_kernel_mxfp4_gelu(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    a_mx_scale_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_amxm,
    stride_amxk,
    stride_bmxe,
    stride_bmxk,
    stride_bmxn,
    A_DTYPE_FORMAT: tl.constexpr,
    B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWIZZLE_MX_A: tl.constexpr,
    SWIZZLE_MX_B: tl.constexpr,
):
    """Fused MoE MXFP4 kernel with GELU-tanh activation (Gemma 4).

    Identical to _fused_moe_kernel_mxfp4_silu except the epilogue applies
    _gelu_tanh instead of _silu_exp2 to the gate half of the accumulator.
    """
    is_a_microscaled_format: tl.constexpr = a_mx_scale_ptr is not None
    is_b_microscaled_format: tl.constexpr = b_mx_scale_ptr is not None
    MX_PACK_DIVISOR: tl.constexpr = 32

    if is_a_microscaled_format:
        a_type: tl.constexpr = a_ptr.dtype.element_ty
        tl.static_assert(
            a_type == tl.uint8 or (a_type == tl.float8e4nv or a_type == tl.float8e5),
            "a must be 1 byte",
        )
        tl.static_assert(a_mx_scale_ptr.dtype.element_ty == tl.uint8, "a_mx_scale must be uint8")
        tl.static_assert(BLOCK_SIZE_K % MX_PACK_DIVISOR == 0, "BLOCK_SIZE_K must be multiple of 32")
    if is_b_microscaled_format:
        b_type: tl.constexpr = b_ptr.dtype.element_ty
        tl.static_assert(
            b_type == tl.uint8 or (b_type == tl.float8e4nv or b_type == tl.float8e5),
            "b must be 1 byte",
        )
        tl.static_assert(b_mx_scale_ptr.dtype.element_ty == tl.uint8, "b_mx_scale must be uint8")
        tl.static_assert(BLOCK_SIZE_K % MX_PACK_DIVISOR == 0, "BLOCK_SIZE_K must be multiple of 32")

    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    NUM_XCDS: tl.constexpr = 8

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        _write_zeros_to_output(
            c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask,
            BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type,
        )
        return

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    i_floor = i // 2
    offs_half = ((pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)).to(tl.int64)
    offs_b_n = ((offs_half + (i % 2) * (N // 2)) % N).to(tl.int64)

    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_expert)
    offs_b_n = tl.max_contiguous(tl.multiple_of(offs_b_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

    if is_a_microscaled_format:
        A_PACK_DIVISOR: tl.constexpr = 2 if a_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // A_PACK_DIVISOR
        MX_SCALE_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR
        offs_scale_ak = tl.arange(0, MX_SCALE_BLOCK_K_A)
        offs_scale_m = offs_token
        a_mx_scale_ptrs = (
            a_mx_scale_ptr
            + offs_scale_ak.to(tl.int64)[None, :] * stride_amxk
            + offs_scale_m.to(tl.int64)[:, None] // top_k * stride_amxm
        )
    else:
        a_mx_scale_ptrs = None
        A_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_A: tl.constexpr = 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K

    if is_b_microscaled_format:
        B_PACK_DIVISOR: tl.constexpr = 2 if b_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // B_PACK_DIVISOR
        MX_SCALE_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR
        b_mx_scale_ptr += off_expert * stride_bmxe
        offs_scale_bk = tl.arange(0, MX_SCALE_BLOCK_K_B)
        offs_scale_n = offs_b_n
        b_mx_scale_ptrs = (
            b_mx_scale_ptr
            + offs_scale_bk.to(tl.int64)[None, :] * stride_bmxk
            + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
        )
    else:
        b_mx_scale_ptrs = None
        B_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_B: tl.constexpr = 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K

    offs_a_k = tl.arange(0, PACKED_BLOCK_K_A)
    offs_b_k = tl.arange(0, PACKED_BLOCK_K_B)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_expert * stride_be + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PACKED_BLOCK_K_A)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_a_k[None, :] < (K - k * PACKED_BLOCK_K_A)), other=0.0)
            b = tl.load(b_ptrs, mask=offs_b_k[:, None] < (K - k * PACKED_BLOCK_K_B), other=0.0)

        if is_a_microscaled_format or is_b_microscaled_format:
            if is_a_microscaled_format:
                mask_ak_scale = offs_scale_ak < (K - k * PACKED_BLOCK_K_A) // (MX_PACK_DIVISOR // A_PACK_DIVISOR)
                a_mx_scales = tl.load(a_mx_scale_ptrs, mask=mask_ak_scale[None, :], other=0.0)
            else:
                a_mx_scales = None
            mask_bk_scale = offs_scale_bk < (K - k * PACKED_BLOCK_K_B) // (MX_PACK_DIVISOR // B_PACK_DIVISOR)
            b_mx_scales = tl.load(b_mx_scale_ptrs, mask=mask_bk_scale[None, :], other=0.0)

            accumulator = tl.dot_scaled(
                a, a_mx_scales, A_DTYPE_FORMAT,
                b, b_mx_scales, B_DTYPE_FORMAT,
                acc=accumulator, fast_math=True,
            )

            if is_a_microscaled_format:
                a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A * stride_amxk
            b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B * stride_bmxk

        a_ptrs += PACKED_BLOCK_K_A * stride_ak
        b_ptrs += PACKED_BLOCK_K_B * stride_bk

    accumulator *= a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    # ---- GELU-tanh activation (replacing SiLU) ----
    gelu_acc, mul_acc = (
        accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    )
    gelu_acc = _gelu_tanh(gelu_acc)
    accumulator = (gelu_acc * mul_acc).to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_mxfp4_gelu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    A_mx_scale: torch.Tensor,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    swizzle_mx_a: bool,
    swizzle_mx_b: bool,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """Fused MoE MXFP4 with GELU-tanh activation for Gemma 4.

    Same interface as fused_moe_mxfp4_silu but applies gelu_tanh instead of silu.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert A_scale is not None
    assert B_scale is not None

    if A.dtype == torch.uint8:
        assert A_mx_scale is not None
        A_mx_scale_strid_m, A_mx_scale_strid_k = A_mx_scale.stride()
    else:
        assert A_mx_scale is None
        A_mx_scale_strid_m, A_mx_scale_strid_k = None, None
    assert B_mx_scale is not None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    A_tl_dtype = torch_to_triton_dtype[A.dtype]
    A_DTYPE_FORMAT = get_scaled_dot_format_string(A_tl_dtype)
    B_tl_dtype = torch_to_triton_dtype[B.dtype]
    B_DTYPE_FORMAT = get_scaled_dot_format_string(B_tl_dtype)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4_gelu[grid](
        A, B, C, A_scale, B_scale, A_mx_scale, B_mx_scale,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.shape[1], A.shape[1], topk_ids.numel(),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        A_mx_scale_strid_m, A_mx_scale_strid_k,
        B_mx_scale.stride(0), B_mx_scale.stride(2), B_mx_scale.stride(1),
        A_DTYPE_FORMAT=A_DTYPE_FORMAT, B_DTYPE_FORMAT=B_DTYPE_FORMAT,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type,
        SWIZZLE_MX_A=swizzle_mx_a, SWIZZLE_MX_B=swizzle_mx_b,
        **config,
    )
