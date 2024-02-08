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

from .llama import FastLlamaModel, logger, IS_GOOGLE_COLAB
from .mistral import FastMistralModel
from transformers import AutoConfig
# from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel
from .mapper import INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER


# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
# major, minor = transformers_version.split(".")[:2]
# major, minor = int(major), int(minor)
SUPPORTS_FOURBIT = True #(major > 4) or (major == 4 and minor >= 37)
# del major, minor

from huggingface_hub.utils import (
    disable_progress_bars,
    enable_progress_bars,
)

def _get_model_name(model_name, load_in_4bit = True):

    if not SUPPORTS_FOURBIT and model_name in INT_TO_FLOAT_MAPPER:
        model_name = INT_TO_FLOAT_MAPPER[model_name]
        logger.warning_once(
            f"Unsloth: Your transformers version of {transformers_version} does not support native "\
            f"4bit loading.\nThe minimum required version is 4.37.\n"\
            f'Try `pip install --upgrade "transformers>=4.37"`\n'\
            f"to obtain the latest transformers build, then restart this session.\n"\
            f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
        )
    
    elif not load_in_4bit and model_name in INT_TO_FLOAT_MAPPER:
        new_model_name = INT_TO_FLOAT_MAPPER[model_name]
        logger.warning_once(
            f"Unsloth: You passed in `{model_name}` which is a 4bit model, yet you set\n"\
            f"`load_in_4bit = False`. We shall load `{new_model_name}` instead."
        )
        model_name = new_model_name

    elif load_in_4bit and SUPPORTS_FOURBIT and model_name in FLOAT_TO_INT_MAPPER:
        new_model_name = FLOAT_TO_INT_MAPPER[model_name]
        logger.warning_once(
            f"Unsloth: You passed in `{model_name}` and `load_in_4bit = True`.\n"\
            f"We shall load `{new_model_name}` for 4x faster loading."
        )
        model_name = new_model_name
    pass

    return model_name
pass


class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name     = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length = 4096,
        dtype          = None,
        load_in_4bit   = True,
        token          = None,
        device_map     = "sequential",
        rope_scaling   = None,
        fix_tokenizer  = True,
        use_gradient_checkpointing = True,
        *args, **kwargs,
    ):
        old_model_name = model_name
        model_name = _get_model_name(model_name, load_in_4bit)
        disable_progress_bars()
        if not IS_GOOGLE_COLAB: raise RuntimeError("Unsloth Studio only works on Google Colab for now.")
        
        # First check if it's a normal model via AutoConfig
        is_peft = False
        try:
            model_config = AutoConfig.from_pretrained(model_name, token = token)
            is_peft = False
        except:
            try:
                # Most likely a PEFT model
                peft_config = PeftConfig.from_pretrained(model_name, token = token)
            except:
                raise RuntimeError(f"Unsloth: `{model_name}` is not a full model or a PEFT model.")
            
            # Check base model again for PEFT
            model_name = _get_model_name(peft_config.base_model_name_or_path, load_in_4bit)
            model_config = AutoConfig.from_pretrained(model_name, token = token)
            is_peft = True
        pass

        model_type = model_config.model_type

        if   model_type == "llama":   dispatch_model = FastLlamaModel
        elif model_type == "mistral": dispatch_model = FastMistralModel
        else:
            raise NotImplementedError(
                f"Unsloth: {model_name} not supported yet!\n"\
                "Make an issue to https://github.com/unslothai/unsloth!",
            )
        pass

        model, tokenizer = dispatch_model.from_pretrained(
            model_name     = model_name,
            max_seq_length = max_seq_length,
            dtype          = dtype,
            load_in_4bit   = load_in_4bit,
            token          = token,
            device_map     = device_map,
            rope_scaling   = rope_scaling,
            fix_tokenizer  = fix_tokenizer,
            *args, **kwargs,
        )

        if load_in_4bit:
            # Fix up bitsandbytes config
            quantization_config = \
            {
                # Sometimes torch_dtype is not a string!!
                "bnb_4bit_compute_dtype"           : model.config.to_dict()["torch_dtype"],
                "bnb_4bit_quant_type"              : "nf4",
                "bnb_4bit_use_double_quant"        : True,
                "llm_int8_enable_fp32_cpu_offload" : False,
                "llm_int8_has_fp16_weight"         : False,
                "llm_int8_skip_modules"            : None,
                "llm_int8_threshold"               : 6.0,
                "load_in_4bit"                     : True,
                "load_in_8bit"                     : False,
                "quant_method"                     : "bitsandbytes",
            }
            model.config.update({"quantization_config" : quantization_config})
        pass

        if is_peft:
            # Now add PEFT adapters
            model = PeftModel.from_pretrained(model, old_model_name)
            # Patch it as well!
            model = dispatch_model.patch_peft_model(model, use_gradient_checkpointing)
        pass
        return model, tokenizer
    pass
pass
