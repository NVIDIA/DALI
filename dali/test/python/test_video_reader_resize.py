# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import glob
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nose_utils import assert_raises

video_directory = "/tmp/labelled_videos/"
video_directory_multiple_resolutions = "/tmp/video_resolution/vp9/"

pipeline_params = {"num_threads": 8, "device_id": 0, "seed": 0}

video_reader_params = [
    {"device": "gpu", "file_root": video_directory, "sequence_length": 32, "random_shuffle": False},
    {
        "device": "gpu",
        "file_root": video_directory_multiple_resolutions,
        "sequence_length": 8,
        "random_shuffle": False,
    },
]

resize_params = [
    {
        "resize_x": 300,
        "resize_y": 200,
        "interp_type": types.DALIInterpType.INTERP_CUBIC,
        "minibatch_size": 8,
    },
    {"resize_x": 300, "interp_type": types.DALIInterpType.INTERP_CUBIC, "minibatch_size": 8},
    {
        "resize_x": 300,
        "resize_y": 200,
        "interp_type": types.DALIInterpType.INTERP_LANCZOS3,
        "minibatch_size": 8,
    },
    {"resize_shorter": 300, "interp_type": types.DALIInterpType.INTERP_CUBIC, "minibatch_size": 8},
    {"resize_longer": 500, "interp_type": types.DALIInterpType.INTERP_CUBIC, "minibatch_size": 8},
    {
        "resize_x": 300,
        "resize_y": 200,
        "min_filter": types.DALIInterpType.INTERP_CUBIC,
        "mag_filter": types.DALIInterpType.INTERP_TRIANGULAR,
        "minibatch_size": 8,
    },
    {
        "resize_x": 300,
        "resize_y": 200,
        "interp_type": types.DALIInterpType.INTERP_CUBIC,
        "minibatch_size": 4,
    },
]


def video_reader_pipeline_base(video_reader, batch_size, video_reader_params, resize_params={}):
    pipeline = dali.pipeline.Pipeline(batch_size=batch_size, **pipeline_params)
    with pipeline:
        outputs = video_reader(**video_reader_params, **resize_params)
        if type(outputs) is list:
            outputs = outputs[0]
        pipeline.set_outputs(outputs)

    return pipeline


def video_reader_resize_pipeline(batch_size, video_reader_params, resize_params):
    return video_reader_pipeline_base(
        dali.fn.readers.video_resize, batch_size, video_reader_params, resize_params
    )


def video_reader_pipeline(batch_size, video_reader_params):
    return video_reader_pipeline_base(dali.fn.readers.video, batch_size, video_reader_params)


def ground_truth_pipeline(batch_size, video_reader_params, resize_params):
    pipeline = video_reader_pipeline(batch_size, video_reader_params)

    def get_next_frame():
        (pipe_out,) = pipeline.run()
        sequences_out = pipe_out.as_cpu().as_array()
        for sample in range(batch_size):
            for frame in range(video_reader_params["sequence_length"]):
                yield [np.expand_dims(sequences_out[sample][frame], 0)]

    gt_pipeline = dali.Pipeline(batch_size=1, **pipeline_params)

    with gt_pipeline:
        resized_frame = dali.fn.external_source(source=get_next_frame, num_outputs=1)
        resized_frame = resized_frame[0].gpu()
        resized_frame = dali.fn.resize(resized_frame, **resize_params)
        gt_pipeline.set_outputs(resized_frame)

    return gt_pipeline


def compare_video_resize_pipelines(pipeline, gt_pipeline, batch_size, video_length):
    global_sample_id = 0
    (batch_gpu,) = pipeline.run()
    batch = batch_gpu.as_cpu()

    for sample_id in range(batch_size):
        global_sample_id = global_sample_id + 1
        sample = batch.at(sample_id)
        for frame_id in range(video_length):
            frame = sample[frame_id]
            gt_frame = gt_pipeline.run()[0].as_cpu().as_array()[0]
            if gt_frame.shape == frame.shape:
                assert (gt_frame == frame).all(), "Images are not equal"
            else:
                assert (
                    gt_frame.shape == frame.shape
                ), f"Shapes are not equal: {gt_frame.shape} != {frame.shape}"


def run_for_params(batch_size, video_reader_params, resize_params):
    pipeline = video_reader_resize_pipeline(batch_size, video_reader_params, resize_params)

    gt_pipeline = ground_truth_pipeline(batch_size, video_reader_params, resize_params)

    compare_video_resize_pipelines(
        pipeline, gt_pipeline, batch_size, video_reader_params["sequence_length"]
    )

    # The intermediate pipeline from ground_truth_pipeline gets entangled in cell objects
    # and is not automatically destroyed. The pipeline outputs are kept alive,
    # effectively leaking large amounts of GPU memory.
    gc.collect()


def test_video_resize(batch_size=2):
    for vp in video_reader_params:
        for rp in resize_params:
            yield run_for_params, batch_size, vp, rp


# ---------------------------------------------------------------------------
# experimental.readers.video_resize
# ---------------------------------------------------------------------------

experimental_video_files = sorted(glob.glob("/tmp/video_files/*.mp4"))

experimental_reader_params = [
    {"filenames": experimental_video_files, "sequence_length": 5},
    {"file_root": video_directory_multiple_resolutions, "sequence_length": 4},
]

experimental_resize_params = [
    {"resize_x": 100, "resize_y": 80},
    {"resize_x": 300, "resize_y": 200, "interp_type": types.DALIInterpType.INTERP_CUBIC},
    {"resize_shorter": 120, "interp_type": types.DALIInterpType.INTERP_LANCZOS3},
    {"resize_longer": 150, "antialias": False},
    {"size": (64, 96), "interp_type": types.DALIInterpType.INTERP_NN},
    {
        "resize_x": 300,
        "resize_y": 200,
        "min_filter": types.DALIInterpType.INTERP_CUBIC,
        "mag_filter": types.DALIInterpType.INTERP_TRIANGULAR,
        "minibatch_size": 4,
    },
]


def test_experimental_video_resize_basic_shape():
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video_resize(
            device="gpu",
            filenames=experimental_video_files,
            sequence_length=3,
            resize_x=100,
            resize_y=80,
        )

    p = pipe()
    p.build()
    (video,) = p.run()
    assert video.layout() == "FHWC"
    for i in range(len(video)):
        sample = np.array(video[i].as_cpu())
        assert sample.shape == (3, 80, 100, 3), sample.shape
        assert sample.dtype == np.uint8


def test_experimental_video_resize_emits_all_outputs():
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return tuple(
            fn.experimental.readers.video_resize(
                device="gpu",
                filenames=experimental_video_files,
                labels=list(range(len(experimental_video_files))),
                sequence_length=3,
                resize_x=64,
                resize_y=64,
                enable_frame_num=True,
                enable_timestamps=True,
            )
        )

    p = pipe()
    p.build()
    video, labels, frame_num, timestamps = p.run()
    for i in range(2):
        assert np.array(video[i].as_cpu()).shape == (3, 64, 64, 3)
        assert np.array(labels[i].as_cpu()).shape == (1,)
        assert np.array(frame_num[i].as_cpu()).shape == (1,)
        assert np.array(timestamps[i].as_cpu()).shape == (3,)


def test_experimental_video_resize_sequence_frame_num_shape():
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return tuple(
            fn.experimental.readers.video_resize(
                device="gpu",
                file_root=video_directory_multiple_resolutions,
                sequence_length=4,
                resize_x=50,
                resize_y=40,
                enable_frame_num="sequence",
                enable_timestamps=True,
            )
        )

    p = pipe()
    p.build()
    video, labels, frame_num, timestamps = p.run()
    for i in range(2):
        assert np.array(video[i].as_cpu()).shape == (4, 40, 50, 3)
        assert np.array(labels[i].as_cpu()).shape == (1,)
        assert np.array(frame_num[i].as_cpu()).shape == (4,)
        assert np.array(timestamps[i].as_cpu()).shape == (4,)


def _check_experimental_against_reference(batch_size, reader_params, resize_params, extra=None):
    """Compares the fused reader with experimental.readers.video followed by fn.resize.

    Both readers see the same files in the same order (no shuffling), so the fused output must
    be bit-exact with the two-step reference, and every metadata output must match.
    """
    extra = extra or {}

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0, seed=0)
    def pipe():
        fused = fn.experimental.readers.video_resize(
            device="gpu", **reader_params, **resize_params, **extra
        )
        plain = fn.experimental.readers.video(device="gpu", **reader_params, **extra)
        if not isinstance(fused, (list, tuple)):
            fused, plain = [fused], [plain]
        assert len(fused) == len(plain)
        reference = fn.resize(plain[0], **resize_params)
        return (*fused, reference, *plain[1:])

    p = pipe()
    p.build()
    for _ in range(2):
        out = p.run()
        assert len(out) % 2 == 0, len(out)
        half = len(out) // 2
        fused, reference = out[:half], out[half:]
        fused_video = fused[0].as_cpu()
        ref_video = reference[0].as_cpu()
        assert fused_video.layout() == "FHWC", fused_video.layout()
        # guard against a trivially passing comparison (e.g. both outputs all zeros)
        assert any(np.array(fused_video[i]).std() > 0 for i in range(batch_size))
        for i in range(batch_size):
            a = np.array(fused_video[i])
            b = np.array(ref_video[i])
            assert a.shape == b.shape, f"{a.shape} != {b.shape}"
            assert a.dtype == b.dtype, f"{a.dtype} != {b.dtype}"
            np.testing.assert_array_equal(a, b)
            assert fused_video[i].source_info() == ref_video[i].source_info()
        for fused_meta, ref_meta in zip(fused[1:], reference[1:]):
            fused_meta = fused_meta.as_cpu()
            ref_meta = ref_meta.as_cpu()
            for i in range(batch_size):
                np.testing.assert_array_equal(np.array(fused_meta[i]), np.array(ref_meta[i]))
    del p
    gc.collect()


def test_experimental_video_resize_matches_reader_plus_resize():
    for reader_params in experimental_reader_params:
        for rp in experimental_resize_params:
            yield _check_experimental_against_reference, 2, reader_params, rp


def test_experimental_video_resize_metadata_matches_reader():
    reader_params = {
        "filenames": experimental_video_files,
        "labels": list(range(len(experimental_video_files))),
        "sequence_length": 4,
        "stride": 2,
    }
    for frame_num in ["scalar", "sequence"]:
        yield (
            _check_experimental_against_reference,
            3,
            reader_params,
            {"resize_x": 96, "resize_y": 72},
            {"enable_frame_num": frame_num, "enable_timestamps": True},
        )


def test_experimental_video_resize_float():
    for normalized in [False, True]:
        yield (
            _check_experimental_against_reference,
            2,
            experimental_reader_params[0],
            {"resize_shorter": 100},
            {"dtype": types.FLOAT, "normalized": normalized},
        )


def test_experimental_video_resize_cpu_not_supported():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.experimental.readers.video_resize(
            device="cpu",
            filenames=experimental_video_files,
            sequence_length=2,
            resize_x=10,
            resize_y=10,
        )

    with assert_raises(RuntimeError, glob="*experimental__readers__VideoResize*not registered*"):
        p = pipe()
        p.build()
