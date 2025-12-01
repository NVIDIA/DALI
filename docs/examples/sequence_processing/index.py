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
    title="Video Processing",
    underline_char="=",
    entries=[
        doc_entry(
            "video/video_reader_simple_example.ipynb",
            op_reference(
                "fn.readers.video",
                "Tutorial describing how to use video reader",
            ),
        ),
        doc_entry(
            "video/video_reader_label_example.ipynb",
            op_reference(
                "fn.readers.video",
                "Tutorial describing how to use video reader to output frames with labels",
            ),
        ),
        doc_entry(
            "video/video_file_list_outputs.ipynb",
            op_reference(
                "fn.readers.video",
                "Tutorial describing how to output frames with \
                                labels assigned to dedicated ranges of frame numbers/timestamps",
            ),
        ),
        doc_entry(
            "sequence_reader_simple_example.ipynb",
            op_reference(
                "fn.readers.sequence",
                "Tutorial describing how to read sequence of video frames stored as separate files",
            ),
        ),
        doc_entry(
            "video/video_processing_per_frame_arguments.ipynb",
            [
                op_reference(
                    "fn.readers.video", "Examples of processing video in DALI"
                ),
                op_reference(
                    "fn.per_frame",
                    "Using per-frame operator to specify arguments to video processing operators",
                ),
                op_reference(
                    "fn.gaussian_blur",
                    "Specifying per-frame arguments when processing video",
                ),
                op_reference(
                    "fn.laplacian",
                    "Specifying per-frame arguments when processing video",
                ),
                op_reference(
                    "fn.rotate",
                    "Specifying per-frame arguments when processing video",
                ),
                op_reference(
                    "fn.warp_affine",
                    "Specifying per-frame arguments when processing video",
                ),
                op_reference(
                    "fn.transforms",
                    "Specifying per-frame arguments when processing video",
                ),
            ],
        ),
        doc_entry(
            "optical_flow_example.ipynb",
            [
                op_reference(
                    "fn.readers.video",
                    "Tutorial describing how to calculate optical flow from video inputs",
                ),
                op_reference(
                    "fn.optical_flow",
                    "Tutorial describing how to calculate optical flow from sequence inputs",
                ),
            ],
        ),
    ],
)
