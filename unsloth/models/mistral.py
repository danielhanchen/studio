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

from .llama import *
from ._utils import __version__

from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
)
# For Pytorch 2.1.1
try:
    from transformers.models.mistral.modeling_mistral import (
        MistralSdpaAttention,
        MistralFlashAttention2,
    )
except:
    MistralSdpaAttention   = MistralAttention
    MistralFlashAttention2 = MistralAttention
pass

from huggingface_hub.utils import (
    disable_progress_bars,
    enable_progress_bars,
)
import gc


def MistralAttention_fast_forward(
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
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        K_M = V_M = bsz * kv_seq_len
        Q_M = bsz * q_len

        has_swa = isinstance(causal_mask, xformers.attn_bias.BlockDiagonalCausalMask)

        # Group query attention
        K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        if hidden_states.requires_grad:
            K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
            V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)

            if has_swa:
                Q = Q.view(1, Q_M, n_heads, head_dim)
                K = K.view(1, K_M, n_heads, head_dim)
                V = V.view(1, V_M, n_heads, head_dim)
            pass
        else:
            # Xformers does support the forward pass though
            Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)

            if has_swa:
                Q = Q.view(1, Q_M, n_kv_heads, n_groups, head_dim)
                K = K.view(1, K_M, n_kv_heads, n_groups, head_dim)
                V = V.view(1, V_M, n_kv_heads, n_groups, head_dim)
            pass
        pass

        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    # elif HAS_FLASH_ATTENTION and attention_mask is None:
    #     Q = Q.transpose(1, 2)
    #     K = K.transpose(1, 2)
    #     V = V.transpose(1, 2)
    #     sw = getattr(self.config, "sliding_window", None)
    #     sw = kv_seq_len if (sw is None or sw == "null") else sw
    #     window = (-1, -1) if (kv_seq_len <= sw) else (sw, sw)
    #     A = flash_attn_func(Q, K, V, causal = True, window_size = window)
    else:
        # Grouped query attention
        # if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
        V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
        # pass
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


def MistralForCausalLM_fast_forward(
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
        bsz, q_len = input_ids.shape
        sliding_window = getattr(self.config, "sliding_window", None)
        if sliding_window is None or sliding_window == "null" or sliding_window <= 0:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
        elif q_len <= sliding_window:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
        else:
            # Fix from https://github.com/Rypo
            causal_mask = xformers.attn_bias.BlockDiagonalCausalMask\
                .from_seqlens([q_len]*bsz)\
                .make_local_attention(window_size = sliding_window)
    pass

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


class FastMistralModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        MistralAttention      .forward = MistralAttention_fast_forward
        MistralSdpaAttention  .forward = MistralAttention_fast_forward
        MistralFlashAttention2.forward = MistralAttention_fast_forward
        MistralDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        MistralModel          .forward = LlamaModel_fast_forward
        MistralForCausalLM    .forward = MistralForCausalLM_fast_forward
        PeftModelForCausalLM  .forward = PeftModelForCausalLM_fast_forward
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name     = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length = 4096,
        dtype          = None,
        load_in_4bit   = True,
        token          = None,
        device_map     = "sequential",
        rope_scaling   = None, # Mistral does not support RoPE scaling
        fix_tokenizer  = True,
        **kwargs,
    ):
        # Mistral does NOT support RoPE Scaling!
        if rope_scaling is not None:
            logger.warning_once("Unsloth: Mistral models do not support RoPE scaling.")
        pass

        SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
            f"{P_.GREEN}=={P_.END}(({P_.GREEN}===={P_.END})){P_.GREEN}=={P_.END}  🦥 "\
            f"{P_.BOLD}{P_.GREEN}Unsloth Studio{P_.END} Free release {__version__}\n"\
            f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform = {platform_system}.\n"\
            f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit = {torch.version.cuda}.\n"\
            f"\        /    AGPLv3 license: http://github.com/unslothai/studio\n"\
            f' "-____-"     Downloading {model_name}..... Please wait.....'
        print(statistics)
        FastMistralModel.pre_patch()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        # Check max sequence length
        model_config = AutoConfig.from_pretrained(model_name, token = token)
        model_max_seq_length = model_config.max_position_embeddings

        # Mistral does NOT support RoPE Scaling sadly so we have to error out.
        if max_seq_length > model_max_seq_length:
            raise RuntimeError(
                "Unsloth: Unfortunately Mistral type models do not support RoPE scaling!\n"\
                f"The maximum sequence length supported is {model_max_seq_length}.",
            )
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
        if not IS_GOOGLE_COLAB: raise RuntimeError("Unsloth Studio only works on Google Colab for now.")
        
        max_position_embeddings = max(max_seq_length, model_max_seq_length)
        full_kwargs = kwargs | \
        {
            "device_map"              : device_map,
            "torch_dtype"             : dtype,
            "quantization_config"     : bnb_config,
            "token"                   : token,
        }
        if bnb_config is None: del full_kwargs["quantization_config"]

        disable_progress_bars()
        with ProgressBar(
                desc = f"Unsloth: Downloading tokenizer for {model_name}",
                colour = "#14B789",
                total = 1,
            ) as progress_bar:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length = max_position_embeddings,
                padding_side     = "right",
                token            = token,
            )
            progress_bar.update(1)
        pass

        with ProgressBar(
                desc = f"Unsloth: Downloading model for {model_name}",
                colour = "#14B789",
                total = 1,
            ) as progress_bar:
            model = AutoModelForCausalLM.from_pretrained(model_name, **full_kwargs)
            progress_bar.update(1)
        pass
        enable_progress_bars()

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = FastMistralModel.post_patch(model)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o   = original_apply_o
        pass

        # Save max_seq_length
        max_position_embeddings = max(max_seq_length, model.config.max_position_embeddings)
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
        # Not necessary anymore since we require transformers>=4.37
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[:len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path" : name})
            pass
        
        # Log Unsloth version for future fastpaths for inference
        model.config.update({"unsloth_version" : __version__})

        # Add save modules
        patch_saving_functions(model)

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass

        return model, tokenizer
    pass
pass
