# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    title="Image Processing",
    underline_char="=",
    entries=[
        doc_entry(
            "augmentation_gallery.ipynb",
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
        "brightness_contrast/index.py",
        "clahe/index.py",
        "color_space_conversion/index.py",
        "decoder/index.py",
        "hsv/index.py",
        doc_entry(
            "interp_types.ipynb",
            op_reference("fn.resize", "Interpolation methods", 1),
        ),
        "resize/index.py",
        "warp/index.py",
        doc_entry(
            "3d_transforms.ipynb",
            [
                op_reference("fn.resize", "3D transforms", 3),
                op_reference("fn.warp_affine", "3D transforms"),
                op_reference("fn.rotate", "3D transforms"),
            ],
        ),
    ],
)
