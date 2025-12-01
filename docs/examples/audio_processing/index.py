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
    title="Audio Processing",
    underline_char="=",
    entries=[
        doc_entry(
            "audio_decoder.ipynb",
            op_reference("fn.decoders.audio", "Audio decoder tutorial"),
        ),
        doc_entry(
            "spectrogram.ipynb",
            [
                op_reference("fn.spectrogram", "Audio spectrogram tutorial"),
                op_reference(
                    "fn.mel_filter_bank", "Audio spectrogram tutorial"
                ),
                op_reference("fn.to_decibels", "Audio spectrogram tutorial"),
                op_reference("fn.mfcc", "Audio spectrogram tutorial"),
            ],
        ),
    ],
)
