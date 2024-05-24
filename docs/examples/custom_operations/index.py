# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from doc_index import doc, doc_entry, op_reference

doc(
    title="Custom Operations",
    underline_char="=",
    entries=[
        "custom_operator/create_a_custom_operator.ipynb",
        doc_entry(
            "python_operator.ipynb",
            [
                op_reference(
                    "fn.python_function",
                    "Running custom Python code with the family of python_function operators",
                ),
                op_reference(
                    "plugin.pytorch.fn.torch_python_function",
                    "Running custom Python code with the family of python_function operators",
                ),
                op_reference(
                    "fn.dl_tensor_python_function",
                    "Running custom Python code with the family of python_function operators",
                ),
            ],
        ),
        doc_entry(
            "gpu_python_operator.ipynb",
            [
                op_reference(
                    "fn.python_function",
                    "Processing GPU Data with Python Operators",
                ),
                op_reference(
                    "plugin.pytorch.fn.torch_python_function",
                    "Processing GPU Data with Python Operators",
                ),
                op_reference(
                    "fn.dl_tensor_python_function",
                    "Processing GPU Data with Python Operators",
                ),
            ],
        ),
        doc_entry(
            "numba_function.ipynb",
            op_reference(
                "plugin.numba.fn.experimental.numba_function",
                "Running custom operations written as Numba JIT-compiled functions",
            ),
        ),
        doc_entry(
            "jax_operator_basic.ipynb",
            op_reference(
                "plugin.jax.fn.jax_function",
                "Running custom JAX augmentations in DALI",
            ),
        ),
        doc_entry(
            "jax_operator_multi_gpu.ipynb",
            op_reference(
                "plugin.jax.fn.jax_function",
                "Running JAX augmentations on multiple GPUs",
            ),
        ),
    ],
)
