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

__all__ = [
    "INT_TO_FLOAT_MAPPER",
    "FLOAT_TO_INT_MAPPER",
]

__INT_TO_FLOAT_MAPPER = \
{
    "unsloth/mistral-7b-bnb-4bit" : (
        "unsloth/mistral-7b",
        "mistralai/Mistral-7B-v0.1",
    ),
    "unsloth/llama-2-7b-bnb-4bit" : (
        "unsloth/llama-2-7b",
        "meta-llama/Llama-2-7b-hf",
    ),
    "unsloth/llama-2-13b-bnb-4bit" : (
        "unsloth/llama-13-7b",
        "meta-llama/Llama-2-13b-hf",
    ),
    "unsloth/codellama-34b-bnb-4bit" : (
        "codellama/CodeLlama-34b-hf",
    ),
    "unsloth/zephyr-sft-bnb-4bit" : (
        "unsloth/zephyr-sft",
        "HuggingFaceH4/mistral-7b-sft-beta",
    ),
    "unsloth/tinyllama-bnb-4bit" : (
        "unsloth/tinyllama",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ),
    "unsloth/mistral-7b-instruct-v0.1-bnb-4bit" : (
        "mistralai/Mistral-7B-Instruct-v0.1",
    ),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit" : (
        "mistralai/Mistral-7B-Instruct-v0.2",
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit" : (
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit" : (
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "unsloth/codellama-7b-bnb-4bit" : (
        "unsloth/codellama-7b",
        "codellama/CodeLlama-7b-hf",
    ),
    "unsloth/codellama-13b-bnb-4bit" : (
        "codellama/CodeLlama-13b-hf",
    ),
    "unsloth/yi-6b-bnb-4bit" : (
        "unsloth/yi-6b",
        "01-ai/Yi-6B",
    ),
    "unsloth/solar-10.7b-bnb-4bit" : (
        "upstage/SOLAR-10.7B-v1.0",
    ),
}

INT_TO_FLOAT_MAPPER = {}
FLOAT_TO_INT_MAPPER = {}

for key, values in __INT_TO_FLOAT_MAPPER.items():
    INT_TO_FLOAT_MAPPER[key] = values[0]

    for value in values:
        FLOAT_TO_INT_MAPPER[value] = key
    pass
pass
