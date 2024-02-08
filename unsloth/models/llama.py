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

import torch
from typing import Optional, Tuple, List, Union
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ..kernels import *
from ._utils import *
from ._utils import __version__
# if HAS_FLASH_ATTENTION:
#     from flash_attn import flash_attn_func

# Final patching code
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

# For Pytorch 2.1.1
try:
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention,
        LlamaFlashAttention2,
    )
except:
    LlamaSdpaAttention   = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention
pass

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers import set_seed as transformers_set_seed
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from peft import PeftModelForCausalLM
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
from ..save import patch_saving_functions


def original_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V
pass


def original_apply_o(self, X):
    O = self.o_proj(X)
    return O
pass


from math import sqrt as math_sqrt
KV_CACHE_INCREMENT = 128 # KV Cache update size

def LlamaAttention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L406
        Fast inference using KV cache.
        QK^T can be computed in 4 chunks

        [Q, q] @ [K, k].T where q, k are the new tokens.
        [QK^T, Qk^T]
        [qK^T, qk^T]

        Since the attention mask wipes Qk^T, we just get
        [QK^T,    0]
        [qK^T, qk^T]

        Since softmax is row-wise, we get
        softmax([QK^T,    0])
        softmax([qK^T, qk^T])

        We then multiply by   [V]
                              [v]
        softmax([QK^T,    0]) [softmax(QK^T)V] *
        softmax([qK^T, qk^T]) [softmax([qK^T, qk^T]) @ [V, v]]

        But notice * [softmax(QK^T)V] is just the last attention.
        We just need to compute the last final row.

        This means we can pass in a row of Q, but we need to
        remember K and V, which are called the KV cache.
    """
    Xn = hidden_states
    bsz, _, hd = hidden_states.size()
    K1, V1 = past_key_value
    dtype = Xn.dtype

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    # assert(n_kv_heads * n_groups == n_heads)
    seq_len = K1.shape[-2]
    kv_seq_len = seq_len + 1

    # Prefill phase
    # if not hasattr(self, "paged_attention"):
    if do_prefill:
        self.paged_attention = torch.empty((KV_CACHE_INCREMENT+seq_len+1, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = "cuda")
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, hd), dtype = dtype, device = "cuda")
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim), dtype = dtype, device = "cuda")
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = "cuda")
        self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len), dtype = dtype, device = "cuda")
        self.scalar = 1.0 / math_sqrt(self.head_dim)
    elif kv_seq_len >= self.paged_attention.shape[0]:
        self.paged_attention.resize_((self.paged_attention.shape[0]+KV_CACHE_INCREMENT, 2, bsz, n_kv_heads, head_dim))
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.attention.resize_((bsz, n_heads, 1, self.attention.shape[-1]+KV_CACHE_INCREMENT))
    pass

    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])
    Kn = fast_linear_forward(self.k_proj, Xn, out = self.temp_KV[0])
    Vn = fast_linear_forward(self.v_proj, Xn, out = self.temp_KV[1])
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)

    # cos, sin = self.rotary_emb(Vn, seq_len = kv_seq_len)
    # Qn, Kn = inplace_rope_embedding(Qn, Kn, cos, sin, position_ids)
    cos = self.rotary_emb.cos_cached[seq_len]
    sin = self.rotary_emb.sin_cached[seq_len]
    h = head_dim // 2

    RH_Q = self.RH_Q
    RH_Q[:,:,:,:h] = Qn[:,:,:,h:]; RH_Q[:,:,:,h:] = Qn[:,:,:,:h]; torch.neg(RH_Q[:,:,:,:h], out = RH_Q[:,:,:,:h]);
    Qn *= cos; Qn.addcmul_(RH_Q, sin);

    RH_K = RH_Q[:,:n_kv_heads,:,:] # torch.empty((n_kv_heads, 1, head_dim), dtype = dtype, device = "cuda")
    RH_K[:,:,:,:h] = Kn[:,:,:,h:]; RH_K[:,:,:,h:] = Kn[:,:,:,:h]; torch.neg(RH_K[:,:,:,:h], out = RH_K[:,:,:,:h]);
    Kn *= cos; Kn.addcmul_(RH_K, sin);
    
    # New KV cache
    # Kn = torch.cat([K1, Kn], dim = 2)
    # Vn = torch.cat([V1, Vn], dim = 2)
    self.paged_attention_K[seq_len] = Kn.permute(2, 0, 1, 3)
    self.paged_attention_V[seq_len] = Vn.permute(2, 0, 1, 3)
    Kn = self.paged_attention_K[:kv_seq_len].permute(1, 2, 0, 3)
    Vn = self.paged_attention_V[:kv_seq_len].permute(1, 2, 0, 3)

    # Handle sliding windows
    sliding_window = getattr(self.config, "sliding_window", None)
    if sliding_window is not None and kv_seq_len > sliding_window:
        # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L193
        slicing_tokens = 1 - sliding_window
        Knn = Kn[:, :, slicing_tokens:, :]#.contiguous()
        Vnn = Vn[:, :, slicing_tokens:, :]#.contiguous()
    else:
        Knn, Vnn = Kn, Vn
    pass

    # Grouped query attention
    _, _, cached_len, _ = Knn.shape
    if n_groups != 1:
        Knn = Knn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Vnn = Vnn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Knn = Knn.reshape(bsz, n_heads, cached_len, head_dim)
        Vnn = Vnn.reshape(bsz, n_heads, cached_len, head_dim)
    pass
    # else:
    #     Knn, Vnn = Knn, Vnn
    # pass

    # Attention
    A = torch.matmul(Qn, Knn.transpose(2, 3), out = self.attention[:,:,:,:cached_len])
    A *= self.scalar
    A[:] = torch.nn.functional.softmax(A, dim = -1, dtype = torch.float32)#.to(A.dtype)
    A = torch.matmul(A, Vnn, out = Qn)
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, self.hidden_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_QA[1])
    return A, (Kn, Vn)
pass


def fast_mlp_inference(self, X):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    mlp_size = self.config.intermediate_size
    temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda")

    gate = fast_linear_forward(self.gate_proj, X, out = temp[0])
    up   = fast_linear_forward(self.  up_proj, X, out = temp[1])
    gate = torch.nn.functional.silu(gate, inplace = True)
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])
    return down
pass


def fast_rms_layernorm_inference(self, X):
    old_dtype = X.dtype
    XX = X.to(torch.float32)
    variance = XX.square().mean(-1, keepdim = True)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()
    X = XX.to(old_dtype) # Must preserve due to residual
    X *= self.weight
    return X
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def LlamaAttention_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    bool = False,
    use_cache:            bool = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len = kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (attention_mask is None):
    # if (not HAS_FLASH_ATTENTION and attention_mask is None):
        # Xformers memory efficient attention
        # Also has Flash Attention v2 dispatching
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Group query attention
        if n_groups != 1:
            K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
            V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
            K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            if hidden_states.requires_grad:
                K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
        pass
        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    # elif HAS_FLASH_ATTENTION and attention_mask is None:
    #     Q = Q.transpose(1, 2)
    #     K = K.transpose(1, 2)
    #     V = V.transpose(1, 2)
    #     A = flash_attn_func(Q, K, V, causal = True)
    else:
        # Grouped query attention
        if n_groups != 1:
            K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
            V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
            K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
            V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
        pass
        # Must be contiguous or else results are False!
        # https://github.com/pytorch/pytorch/issues/112577
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        # Needs (batch_size, n_heads, seq_len, head_dim)
        # is_casual and attention_mask must not be both set!
        A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()
    pass
    attn_output = A.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if past_key_value is not None:
        do_prefill = not hasattr(self.self_attn, "paged_attention")

        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
            self.self_attn,
            hidden_states,
            past_key_value,
            position_ids,
            do_prefill = do_prefill,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_mlp_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
    self,
    input_ids:            torch.LongTensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_values:      Optional[List[torch.FloatTensor]] = None,
    inputs_embeds:        Optional[torch.FloatTensor] = None,
    use_cache:            Optional[bool] = None,
    output_attentions:    Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict:          Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    assert(output_attentions is False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("Unsloth: You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            logger.warning_once(
                f"Unsloth: Input IDs of length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"\
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:,:self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:,:self.max_seq_length,:]
        pass
    pass
    
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    pass

    # We already handle KV cache position_ids ourselves.
    if False:#(past_key_values_length != 0):
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype  = torch.int32,
            device = "cuda",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)#.long()
    else:
        position_ids = None
    pass

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))
    pass

    # embed positions
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Fix up attention mask by setting elements to 0
    # Specifically for DPO
    if self._has_no_labels and (attention_mask is not None) and (past_key_values is None):
        # Careful for inference the attention_mask is size (1, kv_seq_len)
        # Whilst the input_embeds is size (1, 1, 4096)
        inputs_requires_grad = inputs_embeds.requires_grad
        if inputs_requires_grad: inputs_embeds.requires_grad_(False)
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad: inputs_embeds.requires_grad_(True)
    pass

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        # if 0 in attention_mask:
        #     padding_mask = attention_mask
        # else:
        padding_mask = None

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
    pass

    hidden_states = inputs_embeds

    if past_key_values is None and self.gradient_checkpointing and self.training:
        use_cache = False
        # if use_cache:
        #     logger.warning_once(
        #         "Unsloth: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
        #     )
        #     use_cache = False
    pass

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                causal_mask,
                attention_mask,
                position_ids,
                use_reentrant=True,
                preserve_rng_state=False,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask=causal_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    pass
    
    hidden_states = fast_rms_layernorm(self.norm, hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
@torch.inference_mode
def LlamaModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
):
    # Fix out of bounds tokenization
    input_ids = input_ids[:,:self.max_seq_length]

    hidden_states = self.embed_tokens(input_ids)

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.layers):
        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(decoder_layer.input_layernorm, hidden_states)
        hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states,
            past_key_values[idx],
            None,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(decoder_layer.post_attention_layernorm, hidden_states)
        hidden_states = fast_mlp_inference(decoder_layer.mlp, hidden_states)
        hidden_states += residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference(self.norm, hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values   = next_decoder_cache,
        hidden_states     = [],
        attentions        = [],
    )
pass


def LlamaForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if causal_mask is None and past_key_values is None:
        causal_mask = xformers.attn_bias.LowerTriangularMask()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    self.model._has_no_labels = labels is None

    if past_key_values is not None and \
        hasattr(self.model.layers[0].self_attn, "paged_attention"):
        outputs = LlamaModel_fast_forward_inference(
            self.model,
            input_ids,
            past_key_values,
        )
    else:
        outputs = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    pass

    hidden_states = outputs[0]
    bsz, q_len, hd = hidden_states.shape
    if bsz == 1 and q_len == 1:
        logits = torch.mv(self.lm_head.weight, hidden_states.ravel())
        logits = logits.unsqueeze(0).unsqueeze(0)
    else:
        logits = self.lm_head(hidden_states)
    pass

    loss = None
    if labels is not None:
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda")
        pass
        
        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )
    pass

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
pass


def PeftModelForCausalLM_fast_forward(
    self,
    input_ids=None,
    causal_mask=None,
    attention_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    task_ids=None,
    **kwargs,
):
    return self.base_model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )
pass


class FastLlamaModel:

    @staticmethod
    def pre_patch():
        LlamaAttention      .forward = LlamaAttention_fast_forward
        LlamaSdpaAttention  .forward = LlamaAttention_fast_forward
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        LlamaDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        LlamaModel          .forward = LlamaModel_fast_forward
        LlamaForCausalLM    .forward = LlamaForCausalLM_fast_forward
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name     = "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length = 4096,
        dtype          = None,
        load_in_4bit   = True,
        token          = None,
        device_map     = "sequential",
        rope_scaling   = None,
        fix_tokenizer  = True,
        **kwargs,
    ):
        SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
           f"==((====))==  Unsloth Studio Free release {__version__}\n"\
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform = {platform_system}.\n"\
           f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit = {torch.version.cuda}.\n"\
           f"\        /    AGPLv3 license: http://github.com/unslothai/studio\n"\
           f' "-____-"     Downloading {model_name}..... Please wait.....'
        print(statistics)
        FastLlamaModel.pre_patch()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        # RoPE scaling
        model_max_seq_length = \
            AutoConfig.from_pretrained(model_name, token = token).max_position_embeddings

        if (rope_scaling is None) and (max_seq_length > model_max_seq_length):
            rope_scaling = max_seq_length / model_max_seq_length
            logger.warning_once(
                f"Unsloth: {model_name} can only handle sequence lengths of at most "\
                f"{model_max_seq_length}.\nBut with kaiokendev's RoPE scaling of "\
                f"{round(rope_scaling, 3)}, it can be magically be extended to "\
                f"{max_seq_length}!"
            )
            rope_scaling = {"type": "linear", "factor": rope_scaling,}
        pass

        bnb_config = None
        if load_in_4bit and not model_name.endswith("-bnb-4bit"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
            )
        pass

        # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/discussions/12
        # RoPE Scaling's max_position_embeddings must be updated
        max_position_embeddings = max(max_seq_length, model_max_seq_length)
        full_kwargs = kwargs | \
        {
            "device_map"              : device_map,
            "torch_dtype"             : dtype,
            "quantization_config"     : bnb_config,
            "token"                   : token,
            "rope_scaling"            : rope_scaling,
            "max_position_embeddings" : max_position_embeddings,
        }
        if bnb_config is None: del full_kwargs["quantization_config"]
        model = AutoModelForCausalLM.from_pretrained(model_name, **full_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length = max_position_embeddings,
            padding_side     = "right",
            token            = token,
        )

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = FastLlamaModel.post_patch(model)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o   = original_apply_o
        pass

        # Save max_seq_length
        model.max_seq_length = max_position_embeddings
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_position_embeddings
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_position_embeddings

        # We check the tokenizer first for errors
        if fix_tokenizer:
            tokenizer = check_tokenizer(
                model            = model,
                tokenizer        = tokenizer,
                model_name       = model_name,
                model_max_length = max_position_embeddings,
                padding_side     = "right",
                token            = token,
            )
        pass
        patch_saving_functions(tokenizer)

        # Fix up config for transformers uploading PEFT
        # Not necessary anymore since we require transformers>=4.37!
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[:len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path" : name})
            pass
        pass

        # Log Unsloth version for future fastpaths for inference
        model.config.update({"unsloth_version" : __version__})

        # Add save modules
        patch_saving_functions(model)

        return model, tokenizer
    pass


    @staticmethod
    def post_patch(model):
        # Patch model
        layers = model.model.layers

        # Torch.compile fails on embedding matrix??
        # Workaround randomnly fixes it for torch versions < 2.2
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(model.model.embed_tokens.weight)
        model.config.update({"unsloth_version" : __version__})

        # We also do this for the lm_head
        lm_head = torch.nn.Linear(1, 1, bias = None)
        del lm_head.weight
        lm_head.weight = model.lm_head.weight
        lm_head.in_features  = lm_head.weight.shape[1]
        lm_head.out_features = lm_head.weight.shape[0]
        model.lm_head = lm_head

        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        correct_dtype = lm_head.weight.dtype

        for name, module in model.named_modules():
            if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
                weight = module.weight
                quant_state = weight.quant_state

                if type(quant_state) is list:
                    # BnB seems to have float16 as default!
                    module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
                else:
                    # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                    quant_state.dtype = correct_dtype
                pass
            pass
        pass

        # Clear deleted GPU items
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model
    pass


    @staticmethod
    def get_peft_model(
        model,
        r                   = 16,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        lora_alpha          = 16,
        lora_dropout        = 0,
        bias                = "none",
        layers_to_transform = None,
        layers_pattern      = None,
        use_gradient_checkpointing = True,
        random_state        = 3407,
        max_seq_length      = 2048, # not used anymore
        use_rslora          = False,
        init_lora_weights   = True,
        loftq_config        = {},
        **kwargs,
    ):
        transformers_set_seed(random_state)

        if isinstance(model, PeftModelForCausalLM):
            raise TypeError(
                "Unsloth: Your model already has LoRA adapters. No need to run this again!"
            )
        pass

        import inspect
        signature = str(inspect.signature(LoraConfig))
        SUPPORTS_LOFTQ  = "loftq_config" in signature
        SUPPORTS_RSLORA = "use_rslora"   in signature

        assert(max_seq_length <= model.max_seq_length)

        if lora_dropout != 0:
            logger.warning_once(
                f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"\
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if bias != "none":
            logger.warning_once(
                f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"\
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if not (type(init_lora_weights) is bool or \
            init_lora_weights == "gaussian" or init_lora_weights == "loftq"):
            raise ValueError(
                'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq"].'
            )
        pass

        if init_lora_weights == "loftq":

            if not SUPPORTS_LOFTQ:
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"\
                    "Please install PEFT 0.7.2 or higher.\n"\
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass

            if loftq_config == {}:
                from peft import LoftQConfig
                logger.warning_once(
                    f"Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"\
                    f"We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
                )
                loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)
            pass
            
            if hasattr(model.config, "quantization_config"):
                raise ValueError(
                    "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"\
                    "Reload your model without any quantization by setting `load_in_4bit = False`."
                )
            pass
        pass

        assert(type(use_rslora) is bool)
        if use_rslora:
            if not SUPPORTS_RSLORA:
                # We manually check for PEFT
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"\
                    "Please install PEFT 0.7.2 or higher.\n"\
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass
        pass

        accepted_modules = frozenset(("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj",),)
        model.config.update({"unsloth_version" : __version__})
        for module in target_modules:
            assert(module in accepted_modules)
        pass

        # Get LoRA
        arguments = dict(
            r                   = r,
            lora_alpha          = lora_alpha,
            target_modules      = target_modules,
            lora_dropout        = lora_dropout,
            bias                = bias,
            task_type           = TaskType.CAUSAL_LM,
            layers_to_transform = layers_to_transform,
            init_lora_weights   = init_lora_weights,
            loftq_config        = loftq_config,
            use_rslora          = use_rslora,
            **kwargs,
        )
        if not SUPPORTS_LOFTQ:  del arguments["loftq_config"]
        if not SUPPORTS_RSLORA: del arguments["use_rslora"]

        lora_config = LoraConfig(**arguments)
        model = _get_peft_model(model, lora_config)

        model = FastLlamaModel.patch_peft_model(model, use_gradient_checkpointing)
        return model
    pass


    @staticmethod
    def patch_peft_model(
        model,
        use_gradient_checkpointing = True,
    ):
        if not isinstance(model, PeftModelForCausalLM):
            raise TypeError(
                "Unsloth: Your model needs to call `.get_peft_model` first!"
            )
        pass

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = True,
        )

        # Fix up config for transformers uploading PEFT
        for active_adapter in model.peft_config.keys():
            # Not necessary since we requires transformers >= 4.37
            if False:
                name = model.peft_config[active_adapter].base_model_name_or_path
                if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                    name = name[:len(name) - len("-bnb-4bit")]
                    model.peft_config[active_adapter].base_model_name_or_path = name
                pass
            # Add revision to enable future fast inference paths
            model.peft_config[active_adapter].revision = f"unsloth"
        pass

        # Do patching
        n_mlp = 0
        n_qkv = 0
        n_o   = 0
        import types

        active_adapter = model.active_adapters[0] if \
            hasattr(model, "active_adapters") else model.active_adapter

        # Get dropout and bias
        lora_dropout = model.peft_config[active_adapter].lora_dropout
        bias         = model.peft_config[active_adapter].bias

        if lora_dropout == 0 and bias == "none":
            for idx, layer in enumerate(model.model.model.layers):

                # MLP patching
                gate_proj = layer.mlp.gate_proj
                up_proj   = layer.mlp.  up_proj
                down_proj = layer.mlp.down_proj

                if  hasattr(gate_proj, "lora_A") and \
                    hasattr(  up_proj, "lora_A") and \
                    hasattr(down_proj, "lora_A") and \
                    (gate_proj.base_layer if hasattr(gate_proj, "base_layer") else gate_proj).bias is None and \
                    (  up_proj.base_layer if hasattr(  up_proj, "base_layer") else   up_proj).bias is None and \
                    (down_proj.base_layer if hasattr(down_proj, "base_layer") else down_proj).bias is None:

                    # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
                    layer.mlp.forward = types.MethodType(apply_lora_mlp, layer.mlp)
                    n_mlp += 1
                else:
                    logger.warning_once(
                        "Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters\n"\
                        "are not enabled or a bias term (like in Qwen) is used."
                    )
                pass

                # QKV attention patching
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                if  hasattr(q_proj, "lora_A") and \
                    hasattr(k_proj, "lora_A") and \
                    hasattr(v_proj, "lora_A") and \
                    (q_proj.base_layer if hasattr(q_proj, "base_layer") else q_proj).bias is None and \
                    (k_proj.base_layer if hasattr(k_proj, "base_layer") else k_proj).bias is None and \
                    (v_proj.base_layer if hasattr(v_proj, "base_layer") else v_proj).bias is None:

                    layer.self_attn.apply_qkv = apply_lora_qkv
                    n_qkv += 1
                else:
                    logger.warning_once(
                        "Unsloth cannot patch Attention layers with our manual autograd engine since either LoRA adapters\n"\
                        "are not enabled or a bias term (like in Qwen) is used."
                    )
                pass

                # O attention patching
                o_proj = layer.self_attn.o_proj
                if hasattr(o_proj, "lora_A") and \
                    (o_proj.base_layer if hasattr(o_proj, "base_layer") else o_proj).bias is None:

                    layer.self_attn.apply_o = apply_lora_o
                    n_o += 1
                else:
                    logger.warning_once(
                        "Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters\n"\
                        "are not enabled or a bias term (like in Qwen) is used."
                    )
                pass
            pass
        pass

        logger.warning_once(
            f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with "\
            f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
        )
        patch_saving_functions(model)

        # Patch cross entropy loss labels
        # Fixes https://github.com/unslothai/unsloth/issues/10
        max_seq_length = model.max_seq_length
        extra_ignored_labels = torch.full((max_seq_length, 1), -100, device = "cuda")
        model.model.extra_ignored_labels = extra_ignored_labels
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_seq_length
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_seq_length
        return model
    pass


    @staticmethod
    def for_inference(model):
        internal_model = model
        internal_model.gradient_checkpointing = False
        internal_model.training = False

        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
            internal_model.gradient_checkpointing = False
            internal_model.training = False
        pass
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        internal_model = model
        internal_model.gradient_checkpointing = use_gradient_checkpointing
        internal_model.training = True

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
            internal_model.gradient_checkpointing = use_gradient_checkpointing
            internal_model.training = True
        pass
    pass
pass
