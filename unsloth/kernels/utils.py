# Unsloth Studio
# Copyright (C) 2023-present the Unsloth AI team. All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import triton
MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass


import bitsandbytes as bnb
get_ptr = bnb.functional.get_ptr
import ctypes
import torch
cdequantize_blockwise_fp32      = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4  = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4  = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)
pass


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None
    pass

    active_adapter = proj.active_adapters[0] if \
        hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A [active_adapter].weight
    B = proj.lora_B [active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s
pass


def fast_dequantize(W, quant_state = None, out = None):
    if quant_state is None: return W
    if type(quant_state) is not list:
        # New quant_state as a class
        # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        absmax     = quant_state.absmax
        shape      = quant_state.shape
        dtype      = quant_state.dtype
        blocksize  = quant_state.blocksize
        offset     = quant_state.offset
        state2     = quant_state.state2
        absmax2    = state2.absmax
        code2      = state2.code
        blocksize2 = state2.blocksize
    else:
        # Old quant_state as a list of lists
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass

    # Create weight matrix
    if out is None:
        out = torch.empty(shape, dtype = dtype, device = "cuda")
    else:
        assert(out.shape == shape)
        assert(out.dtype == dtype)

    # NF4 dequantization of statistics
    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(n_elements_absmax, dtype = torch.float32, device = "cuda")

    # Do dequantization
    ptr_out_absmax = get_ptr(out_absmax)
    cdequantize_blockwise_fp32(
        get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
        ctypes.c_int(blocksize2), ctypes.c_int(n_elements_absmax)
    )
    out_absmax += offset

    fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
         cdequantize_blockwise_bf16_nf4
    fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
       ctypes.c_int(blocksize), ctypes.c_int(out.numel()))

    # Careful returning transposed data
    is_transposed = (True if W.shape[0] == 1 else False)
    return out.t() if is_transposed else out
pass


def fast_gemv(X, W, quant_state, out = None):
    if quant_state is None: return torch.matmul(X, W, out = out)
    # For fast X @ W where seq_len == 1
    # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
    bsz, q_len, hd = X.shape
    # assert(q_len == 1)

    if type(quant_state) is not list:
        # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        absmax     = quant_state.absmax
        shape      = quant_state.shape
        dtype      = quant_state.dtype
        blocksize  = quant_state.blocksize
        stats      = quant_state.code
        offset     = quant_state.offset
        state2     = quant_state.state2
        absmax2    = state2.absmax
        code2      = state2.code
        blocksize2 = state2.blocksize
    else:
        absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass
    # assert(dtype == X.dtype)
    bout = shape[0]

    if out is None:
        out = torch.empty((bsz, 1, bout,), dtype = dtype, device = "cuda")
    else:
        assert(out.shape == (bsz, 1, bout,))
    pass

    n = 1
    m = shape[0]
    k = shape[1]
    lda = shape[0]
    ldc = shape[0]
    ldb = (hd+1)//2
    m = ctypes.c_int32(m)
    n = ctypes.c_int32(n)
    k = ctypes.c_int32(k)
    lda = ctypes.c_int32(lda)
    ldb = ctypes.c_int32(ldb)
    ldc = ctypes.c_int32(ldc)

    df = torch.empty(absmax.shape, dtype = torch.float32, device = "cuda")
    cdequantize_blockwise_fp32(
        get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), get_ptr(df),
        ctypes.c_int(blocksize2), ctypes.c_int(df.numel()),
    )
    df += offset
    absmax = df

    fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else \
        cgemm_4bit_inference_naive_bf16

    ptr_W      = get_ptr(W)
    ptr_absmax = get_ptr(absmax)
    ptr_stats  = get_ptr(stats)
    blocksize  = ctypes.c_int32(blocksize)

    for row in range(bsz):
        fx(m, n, k, get_ptr(X[row]), ptr_W, ptr_absmax, ptr_stats, get_ptr(out[row]),
           lda, ldb, ldc, blocksize)
    pass

    return out
pass


def fast_linear_forward(proj, X, temp_lora = None, out = None):

    W, W_quant, lora_A, lora_B, lora_S = get_lora_parameters(proj)

    bsz, _, in_dim = X.shape

    if W_quant is None:
        out = torch.matmul(X, W.t(), out = out)
    elif bsz <= 2:
        # Only batches of 2 are faster with Gemv
        out = fast_gemv(X, W, W_quant, out = out)
    else:
        W = fast_dequantize(W.t(), W_quant)
        out = torch.matmul(X, W, out = out)
    pass

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)
        pass

        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch.mv(lora_A._fast_lora, X.ravel(), out = temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha = lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch.mm(X.view(bsz, in_dim), lora_A._fast_lora.t(), out = temp_lora)
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha = lora_S)
        pass
        out = out.view(bsz, 1, out_dim)
    pass

    return out
pass
