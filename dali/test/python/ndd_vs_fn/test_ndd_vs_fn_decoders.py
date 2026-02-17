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


import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd
import os
from nose2.tools import params, cartesian_params
import numpy as np
from nvidia.dali.pipeline import pipeline_def
import test_utils
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    run_operator_test,
    feed_input,
    use_fn_api,
    use_ndd_api,
    ndd_device,
    flatten_operator_configs,
    generate_decoders_data,
    compare,
    sign_off,
)


@sign_off("decoders.audio")
@params("cpu")
def test_audio_decoders(device):
    audio_path = os.path.join(test_utils.get_dali_extra_path(), "db", "audio")
    data = generate_decoders_data(audio_path, ".wav")

    def operation(api, *inp, **operator_args):
        processed, _ = api.decoders.audio(*inp, downmix=True, sample_rate=12345)
        return processed

    ndd_op = use_ndd_api(operation)
    ndd_op._op_class = ndd.decoders.audio._op_class

    run_operator_test(
        input_epoch=data, fn_operator=use_fn_api(operation), ndd_operator=ndd_op, device=device
    )


IMAGE_DECODER_OPERATORS = [
    OperatorTestConfig("decoders.image", {"hw_decoder_load": 0.0}),
    OperatorTestConfig("decoders.image_crop", {"hw_decoder_load": 0.0}),
    OperatorTestConfig("peek_image_shape"),
    OperatorTestConfig("experimental.decoders.image", {"hw_decoder_load": 0.0}),
    OperatorTestConfig("experimental.decoders.image_crop", {"hw_decoder_load": 0.0}),
    OperatorTestConfig("decoders.image_random_crop", {"hw_decoder_load": 0.0}),
    OperatorTestConfig("experimental.decoders.image_random_crop", {"hw_decoder_load": 0.0}),
    OperatorTestConfig("peek_image_shape"),
]

image_decoders_test_configuration = flatten_operator_configs(IMAGE_DECODER_OPERATORS)


@params(*image_decoders_test_configuration)
def test_image_decoders(device, operator_name, fn_operator, ndd_operator, operator_args):
    image_decoder_extensions = ".jpg"
    exclude_subdirs = ["jpeg_lossless"]
    data_path = os.path.join(test_utils.get_dali_extra_path(), "db", "single")
    data = generate_decoders_data(
        data_path, image_decoder_extensions, exclude_subdirs=exclude_subdirs
    )

    run_operator_test(data, fn_operator, ndd_operator, device, operator_args)


@sign_off("decoders.video")
@params("cpu")
def test_video_decoder(device):
    batch_size = 1
    n_iterations = 3
    video_path = os.path.join(test_utils.get_dali_extra_path(), "db", "video", "cfr", "test_1.mp4")
    data = np.array([np.fromfile(video_path, dtype=np.uint8)] * batch_size)
    data = np.array([data] * n_iterations)

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.decoders.video,
        ndd_operator=ndd.decoders.video,
        device=device,
    )


@params(
    *flatten_operator_configs(
        [
            OperatorTestConfig("decoders.image_slice"),
            OperatorTestConfig("experimental.decoders.image_slice"),
        ]
    )
)
def test_image_slice_decoder(device, operator_name, fn_operator, ndd_operator, operator_args):
    image_decoder_extensions = ".jpg"
    exclude_subdirs = ["jpeg_lossless"]
    data_path = os.path.join(test_utils.get_dali_extra_path(), "db", "single")
    data = generate_decoders_data(
        data_path, image_decoder_extensions, exclude_subdirs=exclude_subdirs
    )

    @pipeline_def(
        batch_size=47, device_id=0, num_threads=ndd.get_num_threads(), prefetch_queue_depth=1
    )
    def image_slice_decoder_pipe():
        encoded = fn.external_source(name="INPUT0", device="cpu")
        decoded = fn_operator(encoded, 0.1, 0.4, axes=[0], hw_decoder_load=0.0, device=device)
        return decoded

    pipe = image_slice_decoder_pipe()
    pipe.build()
    for inp in data:
        feed_input(pipe, inp)
        pipe_out = pipe.run()
        ndd_out = ndd_operator(
            ndd.as_batch(inp, device="cpu"),
            0.1,
            0.4,
            axes=[0],
            hw_decoder_load=0.0,
            device=ndd_device(device),
        )
        compare(pipe_out, ndd_out)


# @params("cpu")
# def test_experimental_video_input(device):
#     batch_size = 1
# video_path = os.path.join(test_utils.get_dali_extra_path(),
#   "db", "video", "cfr", "test_1.mp4")
#     data = np.array([np.fromfile(video_path, dtype=np.uint8)] * batch_size)

#     @pipeline_def(
#         batch_size=batch_size,
#         device_id=0,
#         num_threads=ndd.get_num_threads(),
#         prefetch_queue_depth=1,
#     )
#     def pipeline():
#         video = fn.experimental.inputs.video(name="INPUT0", device=device, sequence_length=3)
#         return video

#     pipe = pipeline()
#     pipe.build()

#     pipe.feed_input("INPUT0", data)

#     ndd_video_input = ndd.experimental.inputs.video(
#         ndd.as_batch(data, device="cpu"), device=device, sequence_length=3
#     )

#     for _ in range(N_ITERATIONS):
#         pipe_out = pipe.run()
#         ndd_out = ndd.experimental.inputs.video(data, device=device, sequence_length=3)
#         compare(pipe_out, ndd_out)
