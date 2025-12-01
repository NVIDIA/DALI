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
import numpy as np
import nvidia.dali as dali
import nvidia.dali.types as types

video_directory = "/tmp/labelled_videos/"
video_directory_multiple_resolutions = "/tmp/video_resolution/"

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
