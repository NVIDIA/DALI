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
    title="Other",
    underline_char="=",
    entries=[
        doc_entry(
            "general/multigpu.ipynb",
            [
                op_reference(
                    "fn.readers.file",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.caffe",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.caffe2",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.coco",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.mxnet",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.nemo_asr",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.numpy",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.sequence",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.tfrecord",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.video",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.video_resize",
                    "Reading the data in the multi-GPU setup.",
                ),
                op_reference(
                    "fn.readers.webdataset",
                    "Reading the data in the multi-GPU setup.",
                ),
            ],
        ),
        doc_entry("general/conditionals.ipynb"),
        doc_entry("custom_operations/index.py"),
        doc_entry("advanced/serialization.ipynb"),
        doc_entry("legacy_getting_started.ipynb"),
        doc_entry("general/debug_mode.ipynb"),
    ],
)
