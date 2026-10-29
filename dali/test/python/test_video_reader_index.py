# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile

import numpy as np
from nose2.tools import params
from nvidia.dali import fn, pipeline_def

from test_utils import get_dali_extra_path

VIDEO_DIR = os.path.join(get_dali_extra_path(), "db", "video")

# Files that the CPU decoder supports (VP8, VP9, MJPEG).
SOURCE_FILES = [
    "cfr/test_1_vp9.mp4",
    "cfr/test_2_vp9.mp4",
    "vp9/vp9_0.mp4",
    "vp8/vp8.webm",
    "mjpeg/mjpeg.avi",
]
# Each source file is linked under several names. The index is built per distinct file name, so
# there are at least 8 distinct valid files: the index is built in parallel for the CPU backend
# above that.
COPIES = 3
VALID_FILES = [(name, c) for c in range(COPIES) for name in SOURCE_FILES]
INVALID_FILE = "corrupted/audio_only.mp4"


def make_file_list(root, tag, file_list_format):
    """Writes a file list with an invalid file in the middle, repeated files and ranges.

    The files are symlinked under names that are specific to `tag`, so that the index of each
    pipeline is built from scratch and not taken from the cache of the process.
    """
    links = {}
    for name, copy in VALID_FILES + [(INVALID_FILE, 0)]:
        base, ext = os.path.splitext(os.path.basename(name))
        link = os.path.join(root, f"{tag}_{base}_{copy}{ext}")
        os.symlink(os.path.join(VIDEO_DIR, name), link)
        links[(name, copy)] = link

    if file_list_format == "frames":
        ranges = ["", "2 14", "0 9", "3 -2", "-12 -1"]
    else:
        ranges = ["", "0.0 0.4", "0.05 0.3", "0.1 0.5", "0.2 0.6"]

    invalid = (INVALID_FILE, 0)
    order = VALID_FILES[:5] + [invalid] + VALID_FILES[5:] + VALID_FILES[:4] + [invalid]
    order += [VALID_FILES[2]] * 3  # a file repeated with different ranges
    lines = []
    for i, name in enumerate(order):
        rng = ranges[i % len(ranges)]
        lines.append(f"{links[name]} {i % 7} {rng}".rstrip())
    list_path = os.path.join(root, f"{tag}_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return list_path


def read_epoch(file_list, file_list_format, num_threads):
    @pipeline_def(batch_size=1, num_threads=num_threads, device_id=None, seed=42)
    def pipe_def():
        return fn.experimental.readers.video(
            device="cpu",
            file_list=file_list,
            file_list_format=file_list_format,
            sequence_length=3,
            stride=2,
            step=5,
            enable_frame_num=True,
            name="reader",
        )

    pipe = pipe_def()
    pipe.build()
    epoch_size = pipe.reader_meta("reader")["epoch_size"]
    results = []
    for _ in range(epoch_size):
        frames, label, frame_num = pipe.run()
        results.append(
            (
                np.array(frames[0]),
                np.array(label[0]),
                np.array(frame_num[0]),
            )
        )
    return epoch_size, results


@params("frames", "timestamps")
def test_parallel_index_matches_sequential(file_list_format):
    with tempfile.TemporaryDirectory() as root:
        list_seq = make_file_list(root, "seq", file_list_format)
        list_par = make_file_list(root, "par", file_list_format)
        size_seq, out_seq = read_epoch(list_seq, file_list_format, num_threads=1)
        size_par, out_par = read_epoch(list_par, file_list_format, num_threads=8)
    # every distinct valid file contributes at least one sample (plus the repeated ones)
    assert size_seq >= len(VALID_FILES)
    assert size_seq == size_par
    for i, (a, b) in enumerate(zip(out_seq, out_par)):
        for x, y in zip(a, b):
            assert np.array_equal(x, y), f"Mismatch in sample {i}"
