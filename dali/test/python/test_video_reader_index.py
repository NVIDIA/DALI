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
from nose2.tools import cartesian_params, params
from nvidia.dali import fn, pipeline_def

from nose_utils import assert_raises
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
# there are more distinct files than the threads that index them.
COPIES = 4
INVALID_FILE = "corrupted/audio_only.mp4"  # no video stream
# A valid video whose codec the decoder does not support (H.264 for the CPU one, MJPEG for the GPU
# one): it is skipped.
UNSUPPORTED_FILE = {"cpu": "cfr/test_1.mp4", "gpu": "mjpeg/mjpeg.avi"}
# Files that, cut after TRUNCATED_SIZE bytes, still open (the header is intact) but have no frames,
# so building their index fails. VP8 in WebM is supported by the CPU decoder, H.264 in Matroska by
# the GPU one; the latter is not supported by the CPU decoder.
TRUNCATED_SIZE = 1000
BROKEN_SOURCE = {"cpu": "vp8/vp8.webm", "gpu": "containers/mkv/cfr.mkv"}
BROKEN_UNSUPPORTED_SOURCE = {"cpu": "containers/mkv/cfr.mkv"}


def write_truncated(source, path):
    with open(os.path.join(VIDEO_DIR, source), "rb") as f:
        data = f.read(TRUNCATED_SIZE)
    with open(path, "wb") as f:
        f.write(data)
    return path


def make_file_list(root, tag, file_list_format, device):
    """Writes a file list with an invalid file and a file with an unsupported codec in the middle,
    repeated files and ranges.

    The files are symlinked under names that are specific to `tag`, so that the index of each
    pipeline is built from scratch and not taken from the cache of the process.
    """
    valid_files = [(name, c) for c in range(COPIES) for name in SOURCE_FILES[device]]
    invalid = (INVALID_FILE, 0)
    unsupported = (UNSUPPORTED_FILE[device], 0)
    links = {}
    for name, copy in valid_files + [invalid, unsupported]:
        base, ext = os.path.splitext(name.replace("/", "_"))
        link = os.path.join(root, f"{tag}_{base}_{copy}{ext}")
        os.symlink(os.path.join(VIDEO_DIR, name), link)
        links[(name, copy)] = link

    if file_list_format == "frames":
        ranges = ["", "2 14", "0 9", "3 -2", "-12 -1"]
    else:
        ranges = ["", "0.0 0.4", "0.05 0.3", "0.1 0.5", "0.2 0.6"]

    order = valid_files[:5] + [invalid, unsupported] + valid_files[5:]
    order += valid_files[:4] + [invalid, unsupported]
    order += [valid_files[2]] * 3  # a file repeated with different ranges
    lines = []
    for i, name in enumerate(order):
        rng = ranges[i % len(ranges)]
        lines.append(f"{links[name]} {i % 7} {rng}".rstrip())
    list_path = os.path.join(root, f"{tag}_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return list_path


def read_epoch(file_list, file_list_format, num_threads, device, lazy_init):
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
            lazy_init=lazy_init,
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


@cartesian_params(("frames", "timestamps"), ("cpu", "gpu"), (False, True))
def test_parallel_index_matches_sequential(file_list_format, device, lazy_init):
    """With lazy_init, the index is built on the first use of the reader, on another thread."""
    with tempfile.TemporaryDirectory() as root:
        list_seq = make_file_list(root, "seq", file_list_format, device)
        list_par = make_file_list(root, "par", file_list_format, device)
        size_seq, out_seq = read_epoch(list_seq, file_list_format, 1, device, lazy_init)
        size_par, out_par = read_epoch(list_par, file_list_format, 8, device, lazy_init)
    # every distinct valid file contributes at least one sample (plus the repeated ones)
    assert size_seq >= len(SOURCE_FILES[device]) * COPIES
    assert size_seq == size_par
    for i, (a, b) in enumerate(zip(out_seq, out_par)):
        for x, y in zip(a, b):
            assert np.array_equal(x, y), f"Mismatch in sample {i}"


def build_reader(filenames, num_threads, device):
    @pipeline_def(
        batch_size=1, num_threads=num_threads, device_id=None if device == "cpu" else 0, seed=42
    )
    def pipe_def():
        return fn.experimental.readers.video(
            device=device, filenames=filenames, sequence_length=3, name="reader"
        )

    pipe = pipe_def()
    pipe.build()
    return pipe


@cartesian_params(("cpu", "gpu"), (1, 8))
def test_first_failing_file_is_reported(device, num_threads):
    """Of two files that fail to be indexed, the error of the first one in the list is reported,
    with 1 and with 8 threads. It checks the result, not a race: both files fail at once, so the
    order in which they fail is not forced."""
    with tempfile.TemporaryDirectory() as root:
        valid = [os.path.join(VIDEO_DIR, name) for name in SOURCE_FILES[device]]
        ext = os.path.splitext(BROKEN_SOURCE[device])[1]
        first = write_truncated(BROKEN_SOURCE[device], os.path.join(root, f"first{ext}"))
        second = write_truncated(BROKEN_SOURCE[device], os.path.join(root, f"second{ext}"))
        filenames = valid[:3] + [first] + valid[3:] + [second]
        with assert_raises(RuntimeError, glob=f'*No valid frames found in video file "{first}"*'):
            build_reader(filenames, num_threads, device)


@params(1, 8)
def test_broken_file_with_unsupported_codec_is_skipped(num_threads):
    """A file whose codec the decoder does not support is skipped, even if indexing it would
    fail."""
    with tempfile.TemporaryDirectory() as root:
        valid = [os.path.join(VIDEO_DIR, name) for name in SOURCE_FILES["cpu"]]
        broken = write_truncated(
            BROKEN_UNSUPPORTED_SOURCE["cpu"], os.path.join(root, f"broken_{num_threads}.mkv")
        )
        reference = build_reader(valid, num_threads, "cpu").reader_meta("reader")["epoch_size"]
        pipe = build_reader(valid[:2] + [broken] + valid[2:], num_threads, "cpu")
        assert pipe.reader_meta("reader")["epoch_size"] == reference
