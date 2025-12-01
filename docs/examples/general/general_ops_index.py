# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    title="General Purpose",
    underline_char="=",
    entries=[
        "expressions/index.py",
        doc_entry(
            "reductions.ipynb",
            op_reference(
                "fn.reductions", "Tutorial describing how to use reductions"
            ),
        ),
        doc_entry(
            "tensor_join.ipynb",
            [
                op_reference("fn.cat", "Tutorial describing tensor joining"),
                op_reference("fn.stack", "Tutorial describing tensor joining"),
            ],
        ),
        doc_entry(
            "reinterpret.ipynb",
            [
                op_reference(
                    "fn.reshape", "Tutorial describing tensor reshaping"
                ),
                op_reference(
                    "fn.squeeze", "Tutorial describing tensor squeezing"
                ),
                op_reference(
                    "fn.expand_dims",
                    "Tutorial describing tensor dimensions expanding",
                ),
                op_reference(
                    "fn.reinterpret",
                    "Tutorial describing tensor reinterpreting",
                ),
            ],
        ),
        doc_entry(
            "normalize.ipynb",
            op_reference(
                "fn.normalize", "Tutorial describing tensor normalization"
            ),
        ),
        doc_entry(
            "../math/geometric_transforms.ipynb",
            [
                op_reference(
                    "fn.transforms",
                    (
                        "Tutorial describing tensor geometric transformations to transform"
                        " points and images"
                    ),
                ),
                op_reference(
                    "fn.warp_affine",
                    "Tutorial showing how to use afine transform",
                ),
                op_reference(
                    "fn.coord_transform",
                    "Tutorial describing how to transform points accompanying images",
                ),
            ],
        ),
        doc_entry(
            "erase.ipynb",
            op_reference("fn.erase", "Tutorial describing tensor erasing"),
        ),
    ],
)
