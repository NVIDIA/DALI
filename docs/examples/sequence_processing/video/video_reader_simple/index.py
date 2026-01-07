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
    title="Simple Video Pipeline Reading From Multiple Files",
    underline_char="=",
    options=":maxdepth: 1",
    entries=[
        doc_entry(
            "Pipeline Mode <pipeline_mode.ipynb>",
            op_reference(
                "fn.readers.video",
                "Tutorial describing how to use video reader",
            ),
        ),
        doc_entry(
            "Dynamic Mode <dynamic_mode.ipynb>",
            op_reference(
                "dynamic.readers.Video",
                "Tutorial describing how to use video reader",
            ),
        ),
    ],
)
