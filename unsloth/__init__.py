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

import os
import warnings
import importlib

# Currently only supports 1 GPU, or else seg faults will occur.
if "CUDA_VISIBLE_DEVICES" in os.environ:
    device = os.environ["CUDA_VISIBLE_DEVICES"]
    if not device.isdigit():
        warnings.warn(
            f"Unsloth: 'CUDA_VISIBLE_DEVICES' is currently {device} "\
             "but we require 'CUDA_VISIBLE_DEVICES=0'\n"\
             "We shall set it ourselves."
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif "CUDA_DEVICE_ORDER" not in os.environ:
        warnings.warn(
            f"Unsloth: 'CUDA_DEVICE_ORDER' is not set "\
             "but we require 'CUDA_DEVICE_ORDER=PCI_BUS_ID'\n"\
             "We shall set it ourselves."
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
else:
    # warnings.warn("Unsloth: 'CUDA_VISIBLE_DEVICES' is not set. We shall set it ourselves.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pass

# Check Colab
from huggingface_hub.utils._token import is_google_colab
if not is_google_colab():
    raise RuntimeError("Unsloth Studio only works on Google Colab for now.")

from .models import *
from .save import *
