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

from .cross_entropy_loss import fast_cross_entropy_loss
from .rms_layernorm import fast_rms_layernorm
from .rope_embedding import fast_rope_embedding, inplace_rope_embedding
from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
from .fast_lora import (
	get_lora_parameters,
	apply_lora_mlp,
	apply_lora_qkv,
	apply_lora_o,
)
from .utils import fast_dequantize, fast_gemv, QUANT_STATE, fast_linear_forward
