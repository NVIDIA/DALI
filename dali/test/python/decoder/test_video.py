# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import numpy as np
import cv2
import nvidia.dali.types as types
import glob
import os
import random
from itertools import cycle
from test_utils import get_dali_extra_path, is_mulit_gpu, skip_if_m60
from nose2.tools import params
from nose_utils import SkipTest, attr, assert_raises


filenames = glob.glob(f"{get_dali_extra_path()}/db/video/[cv]fr/*.mp4")
# filter out HEVC because some GPUs do not support it
filenames = filter(lambda filename: "hevc" not in filename, filenames)
# mpeg4 is not yet supported in the CPU operator itself
filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
filenames = filter(lambda filename: "av1" not in filename, filenames)

files = [np.fromfile(filename, dtype=np.uint8) for filename in filenames]


@pipeline_def(device_id=0)
def video_decoder_pipeline(source, device="cpu", module=fn.experimental):
    data = fn.external_source(source=source, dtype=types.UINT8, ndim=1)
    return module.decoders.video(data, device=device)


def video_length(filename):
    cap = cv2.VideoCapture(filename)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def reference_pipeline(filename, device="cpu"):
    seq_length = video_length(filename)
    return fn.experimental.readers.video(
        filenames=[filename], sequence_length=seq_length, device=device
    )


def video_loader(batch_size, epochs):
    idx = 0
    while idx < epochs * len(files):
        batch = []
        for _ in range(batch_size):
            batch.append(files[idx % len(files)])
            idx = idx + 1
        yield batch


def video_decoder_iter(batch_size, epochs=1, device="cpu", module=fn.experimental):
    pipe = video_decoder_pipeline(
        batch_size=batch_size,
        device_id=0,
        num_threads=4,
        source=video_loader(batch_size, epochs),
        device=device,
        module=module,
    )
    for _ in range(int((epochs * len(files) + batch_size - 1) / batch_size)):
        (output,) = pipe.run()
        output = output.as_cpu()
        for i in range(batch_size):
            yield np.array(output[i])


def ref_iter(epochs=1, device="cpu"):
    for _ in range(epochs):
        for filename in filenames:
            pipe = reference_pipeline(filename, device=device)
            (output,) = pipe.run()
            output = output.as_cpu()
            yield np.array(output[0])


@params(("mixed", fn.experimental))
def test_video_decoder(device, module):
    batch_size = 3
    epochs = 3
    decoder_iter = video_decoder_iter(batch_size, epochs, device, module=module)
    ref_dec_iter = ref_iter(epochs, device="cpu" if device == "cpu" else "gpu")
    for seq, ref_seq in zip(decoder_iter, ref_dec_iter):
        assert seq.shape == ref_seq.shape
        assert np.array_equal(seq, ref_seq)


def test_full_range_video():
    skip_if_m60()

    @pipeline_def
    def test_pipeline():
        videos = fn.readers.video(
            device="gpu",
            filenames=[get_dali_extra_path() + "/db/video/full_dynamic_range/video.mp4"],
            sequence_length=1,
            initial_fill=10,
            normalized=False,
            dtype=types.UINT8,
        )
        return videos

    video_pipeline = test_pipeline(batch_size=1, num_threads=1, device_id=0)

    o = video_pipeline.run()
    out = o[0].as_cpu().as_array()
    ref = cv2.imread(get_dali_extra_path() + "/db/video/full_dynamic_range/0001.png")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    left = ref
    right = out
    absdiff = np.abs(left.astype(int) - right.astype(int))
    assert np.mean(absdiff) < 2


@params("cpu", "gpu")
def test_full_range_video_in_memory(device):
    skip_if_m60()

    @pipeline_def
    def test_pipeline():
        videos = fn.experimental.readers.video(
            device=device,
            filenames=[get_dali_extra_path() + "/db/video/full_dynamic_range/video.mp4"],
            sequence_length=1,
        )
        return videos

    video_pipeline = test_pipeline(batch_size=1, num_threads=1, device_id=0)

    o = video_pipeline.run()
    out = o[0]
    if device == "gpu":
        out = out.as_cpu()
    out = out.as_array()
    ref = cv2.imread(get_dali_extra_path() + "/db/video/full_dynamic_range/0001.png")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    left = ref
    right = out
    absdiff = np.abs(left.astype(int) - right.astype(int))
    assert np.mean(absdiff) < 2


@attr("multi_gpu")
@params("cpu", "mixed")
def test_multi_gpu_video(device):
    skip_if_m60()
    if not is_mulit_gpu():
        raise SkipTest()

    batch_size = 1

    def input_gen(batch_size):
        filenames = glob.glob(f"{get_dali_extra_path()}/db/video/[cv]fr/*.mp4")
        # test overflow of frame_buffer_
        filenames.append(f"{get_dali_extra_path()}/db/video/cfr_test.mp4")
        filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
        filenames = filter(lambda filename: "hevc" not in filename, filenames)
        filenames = filter(lambda filename: "av1" not in filename, filenames)
        filenames = cycle(filenames)
        while True:
            batch = []
            for _ in range(batch_size):
                batch.append(np.fromfile(next(filenames), dtype=np.uint8))
            yield batch

    @pipeline_def
    def test_pipeline():
        vid = fn.external_source(device="cpu", source=input_gen(batch_size))
        seq = fn.experimental.decoders.video(vid, device=device)
        return seq

    video_pipeline_0 = test_pipeline(batch_size=1, num_threads=1, device_id=0)
    video_pipeline_1 = test_pipeline(batch_size=1, num_threads=1, device_id=1)

    iters = 5
    for _ in range(iters):
        video_pipeline_0.run()
        video_pipeline_1.run()


@params("cpu", "gpu")
def test_source_info(device):
    skip_if_m60()
    filenames = glob.glob(f"{get_dali_extra_path()}/db/video/[cv]fr/*.mp4")
    # filter out HEVC because some GPUs do not support it
    filenames = filter(lambda filename: "hevc" not in filename, filenames)
    # mpeg4 is not yet supported in the CPU operator itself
    filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
    filenames = filter(lambda filename: "av1" not in filename, filenames)

    files = list(filenames)

    @pipeline_def
    def test_pipeline():
        videos = fn.experimental.readers.video(
            device=device,
            filenames=files,
            sequence_length=1,
            step=10000000,  # make sure that each video has only one valid sequence
        )
        return videos

    batch_size = 4
    p = test_pipeline(batch_size=batch_size, num_threads=1, device_id=0)

    samples_read = 0
    while samples_read < len(files):
        o = p.run()
        for idx, t in enumerate(o[0]):
            assert t.source_info() == files[(samples_read + idx) % len(files)]
        samples_read += batch_size


@params("cpu", "mixed")
def test_error_source_info(device):
    error_file = "README.txt"
    filenames = os.path.join(get_dali_extra_path(), "db/video/cfr/", error_file)

    @pipeline_def
    def test_pipeline():
        data, _ = fn.readers.file(files=filenames)
        return fn.experimental.decoders.video(data, device=device)

    batch_size = 4
    p = test_pipeline(batch_size=batch_size, num_threads=1, device_id=0)

    assert_raises(RuntimeError, p.run)


cfr_files = [
    f"{get_dali_extra_path()}/db/video/cfr/test_1.mp4",
    f"{get_dali_extra_path()}/db/video/cfr/test_2.mp4",
]
vfr_files = [
    f"{get_dali_extra_path()}/db/video/vfr/test_1.mp4",
    f"{get_dali_extra_path()}/db/video/vfr/test_2.mp4",
]


def idx_reflect_101(idx, lo, hi):
    """Reflects out-of-range indices until fits in range.

    Args:
        idx: The index to reflect
        lo: Low (inclusive) bound
        hi: High (exclusive) bound

    Returns:
        Reflected index value
    """
    if hi - lo < 2:
        return hi - 1  # make it obviously wrong if hi <= lo

    while True:
        if idx < lo:
            idx = 2 * lo - idx
        elif idx >= hi:
            idx = 2 * hi - 2 - idx
        else:
            break

    return idx


def idx_reflect_1001(idx, lo, hi):
    """Reflects out-of-range indices until fits in range.

    Args:
        idx: The index to reflect
        lo: Low (inclusive) bound
        hi: High (exclusive) bound

    Returns:
        Reflected index value

    This reflect flavor repeats the first and last element:

    lo--v     v--hi
         ABCDEF           is padded as
    DCBAABCDEFFEDCBAABCD
    """
    if hi - lo < 1:
        return hi - 1  # make it obviously wrong if hi <= lo

    while True:
        if idx < lo:
            idx = 2 * lo - 1 - idx
        elif idx >= hi:
            idx = 2 * hi - 1 - idx
        else:
            break

    return idx


@params(
    # Test case 1: Constant frame rate video, start_frame = 5, sequence_length = 3, stride = 1
    *[(device, 3, cfr_files, 5, 3, 1, None) for device in ["cpu", "mixed"]],
    # Test case 2: Variable frame rate video, start_frame = 0, sequence_length = 4, stride = 2
    *[(device, 3, vfr_files, 0, 4, 2, None) for device in ["cpu", "mixed"]],
    # Test case 3: Constant frame rate video, start_frame = 0, sequence_length = 100, stride = 11
    # pad_mode = "constant", "edge", "symmetric", "reflect"
    *[
        (device, 3, cfr_files, 0, 10, 11, pad_mode)
        for device in ["cpu", "mixed"]
        for pad_mode in ["constant", "edge", "symmetric", "reflect"]
    ],
    # Test case 4: Constant frame rate video, start_frame = sequence_length = stride = "random"
    *[(device, 3, cfr_files, "random", "random", "random", None) for device in ["cpu", "mixed"]],
)
def test_video_decoder_frame_start_end_stride(
    device, batch_size, filenames, start_frame, sequence_length, stride, pad_mode
):
    num_iters = 1
    batch = []
    fill_value = 111
    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        reference = fn.experimental.decoders.video(encoded, device=device)

        start_frame_arg = (
            fn.random.uniform(range=(0, 5), dtype=types.INT32)
            if start_frame == "random"
            else start_frame
        )
        sequence_length_arg = (
            fn.random.uniform(range=(3, 6), dtype=types.INT32)
            if sequence_length == "random"
            else sequence_length
        )
        stride_arg = (
            fn.random.uniform(range=(1, 4), dtype=types.INT32) if stride == "random" else stride
        )

        decoded = fn.experimental.decoders.video(
            encoded,
            device=device,
            start_frame=start_frame_arg,
            sequence_length=sequence_length_arg,
            stride=stride_arg,
            pad_mode=pad_mode,
            fill_value=fill_value,
        )

        return (reference, decoded, start_frame_arg, sequence_length_arg, stride_arg)

    pipe = test_pipeline(batch_size=batch_size, num_threads=3, device_id=0)

    for _ in range(num_iters):
        out0, out1, start_frame_arg, sequence_length_arg, stride_arg = (
            o.as_cpu() for o in pipe.run()
        )
        # Verify each sample in the batch
        for i in range(batch_size):
            out_reference_full = out0.at(i)
            out = out1.at(i)
            start_frame = start_frame_arg.at(i)
            sequence_length = sequence_length_arg.at(i)
            stride = stride_arg.at(i)
            orig_F, H, W, C = tuple(out_reference_full.shape)

            def constant_padding():
                return np.full((H, W, C), fill_value, dtype=np.uint8)

            # Generate indices for start:end:stride slice
            frame_indices = []
            for f in range(start_frame, start_frame + sequence_length * stride, stride):
                if f < orig_F:
                    frame_indices.append(f)
                else:
                    if pad_mode == "edge":
                        frame_indices.append(orig_F - 1)  # Repeat last frame
                    elif pad_mode == "symmetric":
                        # Reflect including boundary: abcd -> abccba
                        frame_indices.append(idx_reflect_1001(f, 0, orig_F))
                    elif pad_mode == "reflect":
                        # Reflect excluding boundary: abcd -> abcdcb
                        frame_indices.append(idx_reflect_101(f, 0, orig_F))
                    elif pad_mode == "constant":
                        frame_indices.append(-1)  # Special index for constant padding
                    else:
                        raise ValueError(f"Invalid pad_mode: {pad_mode}")

            out_reference = np.stack(
                [out_reference_full[f] if f >= 0 else constant_padding() for f in frame_indices]
            )
            np.testing.assert_array_equal(out_reference, out)


@params(
    # Test single constant frame rate video with simple frame indices
    *[(device, 3, cfr_files, [0, 5, 10]) for device in ["cpu", "mixed"]],
    # Test multiple variable frame rate videos with non-sequential frames
    *[(device, 3, vfr_files, [2, 4, 8]) for device in ["cpu", "mixed"]],
    # Test single constant frame rate video with non-monotonic frame indices
    *[(device, 3, cfr_files, [0, 5, 10, 8, 7, 6]) for device in ["cpu", "mixed"]],
    # Test single constant frame rate video with repeated frame indices
    *[(device, 3, cfr_files, [0, 5, 10, 8, 10, 5, 0, 0]) for device in ["cpu", "mixed"]],
    # Test multiple constant frame rate videos with random indices
    *[(device, 3, cfr_files, None) for device in ["cpu", "mixed"]],
)
def test_video_decoder_frame_indices(device, batch_size, filenames, frames):
    num_iters = 1
    batch = []
    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        reference = fn.experimental.decoders.video(encoded, device=device)
        if frames is None:
            frames_shape = fn.random.uniform(range=(1, 10), shape=(1,), dtype=types.INT32)
            frames_arg = fn.random.uniform(range=(0, 10), shape=frames_shape, dtype=types.INT32)
        else:
            frames_arg = frames
        decoded = fn.experimental.decoders.video(encoded, device=device, frames=frames_arg)
        return (reference, decoded, frames_arg)

    pipe = test_pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    for _ in range(num_iters):
        out0, out1, frames_arg = (o.as_cpu() for o in pipe.run())
        # Verify each sample in the batch
        for i in range(batch_size):
            out_reference = out0.at(i)
            out = out1.at(i)
            frames = frames_arg.at(i)
            # Get frames at specified indices for reference
            out_reference = out_reference[frames]
            np.testing.assert_array_equal(out_reference, out)


@params(
    # Test single constant frame rate video with start_frame near the end
    *[(device, 3, cfr_files, 1, 1000) for device in ["cpu", "mixed"]],
    # Test multiple variable frame rate videos with large stride
    *[(device, 3, vfr_files, 2, 1000) for device in ["cpu", "mixed"]],
)
def test_video_decoder_no_padding(device, batch_size, filenames, stride, sequence_length):
    batch = []
    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        # Get full video first as reference
        reference = fn.experimental.decoders.video(encoded, device=device, stride=stride)
        # Request frames that would exceed video length
        decoded = fn.experimental.decoders.video(
            encoded,
            device=device,
            stride=stride,
            sequence_length=sequence_length,
            pad_mode="none",
        )
        return (reference, decoded)

    pipe = test_pipeline(batch_size=batch_size, num_threads=1, device_id=0)
    out_ref, out = (o.as_cpu() for o in pipe.run())

    # Verify each sample in the batch
    for i in range(batch_size):
        ref = out_ref.at(i)
        actual = out.at(i)
        np.testing.assert_array_equal(ref, actual)


@params("cpu", "mixed")
def test_incompatible_args(device):
    batch = []
    with open(cfr_files[0], "rb") as f:
        batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def test_pipeline(frames=None, start_frame=None, stride=None, sequence_length=None):
        encoded = fn.external_source(source=get_batch, device="cpu")
        decoded = fn.experimental.decoders.video(
            encoded,
            device=device,
            frames=frames,
            start_frame=start_frame,
            stride=stride,
            sequence_length=sequence_length,
        )
        return decoded

    # Test that frames argument is incompatible with start_frame and stride
    for frames, start_frame, stride, sequence_length in [
        ([0, 1, 2], 0, None, None),
        ([0, 1, 2], None, 1, None),
        ([0, 1, 2], None, None, 3),
        ([0, 1, 2], 0, 1, 3),
    ]:
        with assert_raises(
            RuntimeError,
            glob="Cannot specify both `frames` and any of `start_frame`, "
            "`sequence_length`, `stride` arguments",
        ):
            pipe = test_pipeline(
                batch_size=1,
                num_threads=3,
                device_id=0,
                frames=frames,
                start_frame=start_frame,
                stride=stride,
                sequence_length=sequence_length,
            )
            pipe.build()


@params("cpu", "mixed")
def test_multichannel_fill_value(device):
    batch = []
    batch_size = 3
    fill_value = [118, 185, 0]  # RGB fill values

    for i in range(batch_size):
        with open(cfr_files[i % len(cfr_files)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        decoded0 = fn.experimental.decoders.video(
            encoded,
            stride=10,
            device=device,
            pad_mode="none",
        )
        decoded1 = fn.experimental.decoders.video(
            encoded,
            device=device,
            stride=10,
            sequence_length=10,  # Request more frames than available
            pad_mode="constant",
            fill_value=fill_value,
        )
        return decoded0, decoded1

    batch_size = 3
    pipe = test_pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    pipe.build()
    out = pipe.run()
    out0, out1 = (o.as_cpu() for o in out)

    for i in range(batch_size):
        frames = out0.at(i)
        frames_padded = out1.at(i)

        padded_F = frames_padded.shape[0]
        F, H, W, C = frames.shape
        padding = padded_F - F
        assert padding > 0

        # Create expected padded frames with RGB fill values
        padded_frame = np.full((padding, H, W, C), fill_value, dtype=np.uint8)
        np.testing.assert_array_equal(frames_padded[:F, :, :, :], frames)
        np.testing.assert_array_equal(frames_padded[F:, :, :, :], padded_frame)
