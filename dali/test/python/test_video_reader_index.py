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
from nose2.tools import cartesian_params
from nvidia.dali import fn, pipeline_def

from test_utils import get_dali_extra_path

VIDEO_DIR = os.path.join(get_dali_extra_path(), "db", "video")

# Files that each decoder supports: VP8, VP9 and MJPEG for the CPU one, H.264 for the GPU one.
SOURCE_FILES = {
    "cpu": [
        "cfr/test_1_vp9.mp4",
        "cfr/test_2_vp9.mp4",
        "vp9/vp9_0.mp4",
        "vp8/vp8.webm",
        "mjpeg/mjpeg.avi",
    ],
    "gpu": [
        "cfr/test_1.mp4",
        "cfr/test_2.mp4",
        "vfr/test_1.mp4",
        "vfr/test_2.mp4",
        "sintel/video_files/sintel_trailer-720p_0.mp4",
        "sintel/video_files/sintel_trailer-720p_1.mp4",
    ],
}
# Each source file is linked under several names. The index is built per distinct file name, so
# there are at least 8 distinct valid files: the index is built in parallel above that.
COPIES = 3
INVALID_FILE = "corrupted/audio_only.mp4"


def make_file_list(root, tag, file_list_format, device):
    """Writes a file list with an invalid file in the middle, repeated files and ranges.

    The files are symlinked under names that are specific to `tag`, so that the index of each
    pipeline is built from scratch and not taken from the cache of the process.
    """
    valid_files = [(name, c) for c in range(COPIES) for name in SOURCE_FILES[device]]
    links = {}
    for name, copy in valid_files + [(INVALID_FILE, 0)]:
        base, ext = os.path.splitext(name.replace("/", "_"))
        link = os.path.join(root, f"{tag}_{base}_{copy}{ext}")
        os.symlink(os.path.join(VIDEO_DIR, name), link)
        links[(name, copy)] = link

    if file_list_format == "frames":
        ranges = ["", "2 14", "0 9", "3 -2", "-12 -1"]
    else:
        ranges = ["", "0.0 0.4", "0.05 0.3", "0.1 0.5", "0.2 0.6"]

    invalid = (INVALID_FILE, 0)
    order = valid_files[:5] + [invalid] + valid_files[5:] + valid_files[:4] + [invalid]
    order += [valid_files[2]] * 3  # a file repeated with different ranges
    lines = []
    for i, name in enumerate(order):
        rng = ranges[i % len(ranges)]
        lines.append(f"{links[name]} {i % 7} {rng}".rstrip())
    list_path = os.path.join(root, f"{tag}_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return list_path


def read_epoch(file_list, file_list_format, num_threads, device):
    @pipeline_def(
        batch_size=1, num_threads=num_threads, device_id=None if device == "cpu" else 0, seed=42
    )
    def pipe_def():
        outputs = fn.experimental.readers.video(
            device=device,
            file_list=file_list,
            file_list_format=file_list_format,
            sequence_length=3,
            stride=2,
            step=5,
            enable_frame_num=True,
            name="reader",
        )
        return tuple(outputs)

    pipe = pipe_def()
    pipe.build()
    epoch_size = pipe.reader_meta("reader")["epoch_size"]
    results = []
    for _ in range(epoch_size):
        # with the GPU backend all the outputs are on the GPU
        frames, label, frame_num = (o.as_cpu() if device == "gpu" else o for o in pipe.run())
        results.append((np.array(frames[0]), np.array(label[0]), np.array(frame_num[0])))
    return epoch_size, results


@cartesian_params(("frames", "timestamps"), ("cpu", "gpu"))
def test_parallel_index_matches_sequential(file_list_format, device):
    with tempfile.TemporaryDirectory() as root:
        list_seq = make_file_list(root, "seq", file_list_format, device)
        list_par = make_file_list(root, "par", file_list_format, device)
        size_seq, out_seq = read_epoch(list_seq, file_list_format, 1, device)
        size_par, out_par = read_epoch(list_par, file_list_format, 8, device)
    # every distinct valid file contributes at least one sample (plus the repeated ones)
    assert size_seq >= len(SOURCE_FILES[device]) * COPIES
    assert size_seq == size_par
    for i, (a, b) in enumerate(zip(out_seq, out_par)):
        for x, y in zip(a, b):
            assert np.array_equal(x, y), f"Mismatch in sample {i}"
