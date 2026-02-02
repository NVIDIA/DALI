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
from nose2.tools import params
from nvidia.dali.pipeline import pipeline_def
import test_utils
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    run_operator_test,
    feed_input,
    use_fn_api,
    use_ndd_api,
    flatten_operator_configs,
    generate_decoders_data,
    compare,
)


def test_audio_decoders():
    device = "cpu"
    audio_path = os.path.join(test_utils.get_dali_extra_path(), "db", "audio")
    data = generate_decoders_data(audio_path, ".wav")

    def operation(api, *inp, **operator_args):
        processed, _ = api.decoders.audio(*inp, downmix=True, sample_rate=12345)
        return processed

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
    )


IMAGE_DECODER_OPERATORS = [
    OperatorTestConfig("decoders.image", {"hw_decoder_load": 0.0}, devices=["cpu", "mixed"]),
    OperatorTestConfig("decoders.image_crop", {"hw_decoder_load": 0.0}, devices=["cpu", "mixed"]),
    OperatorTestConfig("peek_image_shape", devices=["cpu"]),
]

image_decoders_test_configuration = flatten_operator_configs(IMAGE_DECODER_OPERATORS)


@params(*image_decoders_test_configuration)
def test_image_decoders(device, fn_operator, ndd_operator, operator_args):
    image_decoder_extensions = ".jpg"
    exclude_subdirs = ["jpeg_lossless"]
    data_path = os.path.join(test_utils.get_dali_extra_path(), "db", "single")
    data = generate_decoders_data(
        data_path, image_decoder_extensions, exclude_subdirs=exclude_subdirs
    )

    @pipeline_def(
        batch_size=47, device_id=0, num_threads=ndd.get_num_threads(), prefetch_queue_depth=1
    )
    def image_decoder_pipe():
        encoded = fn.external_source(name="INPUT0", device="cpu")
        decoded = fn_operator(encoded, device=device, **operator_args)
        return decoded

    pipe = image_decoder_pipe()
    pipe.build()

    for inp in data:
        feed_input(pipe, inp)
        pipe_out = pipe.run()
        ndd_out = ndd_operator(ndd.as_batch(inp, device="cpu"), device=device, **operator_args)
        assert compare(pipe_out, ndd_out)
