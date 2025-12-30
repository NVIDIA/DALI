# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    title="Augmentation Gallery",
    underline_char="=",
    options=":maxdepth: 1",
    entries=[
        doc_entry(
            "Pipeline Mode <pipeline_mode.ipynb>",
            [
                op_reference("fn.erase", "Augmentation gallery"),
                op_reference("fn.water", "Augmentation gallery"),
                op_reference("fn.sphere", "Augmentation gallery"),
                op_reference("fn.warp_affine", "Augmentation gallery"),
                op_reference(
                    "fn.jpeg_compression_distortion", "Augmentation gallery"
                ),
                op_reference("fn.paste", "Augmentation gallery"),
                op_reference("fn.flip", "Augmentation gallery"),
                op_reference("fn.rotate", "Augmentation gallery"),
                op_reference("fn.hsv", "Augmentation gallery"),
                op_reference("fn.brightness_contrast", "Augmentation gallery"),
            ],
        ),
        doc_entry(
            "Dynamic Mode <dynamic_mode.ipynb>",
            [
                op_reference("dynamic.erase", "Augmentation gallery"),
                op_reference("dynamic.water", "Augmentation gallery"),
                op_reference("dynamic.sphere", "Augmentation gallery"),
                op_reference("dynamic.warp_affine", "Augmentation gallery"),
                op_reference(
                    "dynamic.jpeg_compression_distortion",
                    "Augmentation gallery",
                ),
                op_reference("dynamic.paste", "Augmentation gallery"),
                op_reference("dynamic.flip", "Augmentation gallery"),
                op_reference("dynamic.rotate", "Augmentation gallery"),
                op_reference("dynamic.hsv", "Augmentation gallery"),
                op_reference(
                    "dynamic.brightness_contrast", "Augmentation gallery"
                ),
            ],
        ),
    ],
)
