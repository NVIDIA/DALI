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


import random
import os
import test_utils
import numpy as np
from nose2.tools import params
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    feed_input,
    image_like_shape_generator,
    run_operator_test,
    compare,
    flatten_operator_configs,
    _random_state_source_factory,
    array_1d_shape_generator,
    create_rngs,
    generate_data,
    generate_decoders_data,
    MAX_BATCH_SIZE,
    N_ITERATIONS,
)
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd


RANDOM_OPERATORS_1D_ARRAY = [
    OperatorTestConfig("random.choice", devices=["cpu"]),
    OperatorTestConfig("random.normal"),
    OperatorTestConfig("random.beta", devices=["cpu"]),
    OperatorTestConfig("random.coin_flip"),
    OperatorTestConfig("random.uniform"),
]

random_ops_1d_array_test_configuration = flatten_operator_configs(RANDOM_OPERATORS_1D_ARRAY)


@params(*random_ops_1d_array_test_configuration)
def test_random_1d_array(device, fn_operator, ndd_operator, operator_args):
    data = generate_data(array_1d_shape_generator, batch_sizes=MAX_BATCH_SIZE)

    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        random=True,
        operator_args=operator_args,
    )


RANDOM_OPERATORS_IMAGE_LIKE = [
    # OperatorTestConfig("random_resized_crop", {"size": (64)}),  # BUG
    OperatorTestConfig("jitter", devices=["gpu"]),
    OperatorTestConfig("noise.gaussian"),
    OperatorTestConfig("noise.shot"),
    OperatorTestConfig("noise.salt_and_pepper"),
    OperatorTestConfig("segmentation.random_mask_pixel", devices=["cpu"]),
    OperatorTestConfig(
        "roi_random_crop",
        {
            "crop_shape": [10, 15, 3],
            "roi_start": [25, 20, 0],
            "roi_shape": [40, 30, 3],
        },
        devices=["cpu"],
    ),
    # OperatorTestConfig("random_resized_crop", {"size": (64, 64)}),  # BUG
]

random_ops_image_like_test_configuration = flatten_operator_configs(RANDOM_OPERATORS_IMAGE_LIKE)


@params(*random_ops_image_like_test_configuration)
def test_random_image_like(device, fn_operator, ndd_operator, operator_args):
    data = generate_data(
        image_like_shape_generator, lo=0, hi=255, dtype=np.uint8, batch_sizes=MAX_BATCH_SIZE
    )

    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        random=True,
        operator_args=operator_args,
    )


# BUG (duplicating coin_flip test)
# def test_coin_flip():
#     fn_rng, ndd_rng = create_rngs()

#     @pipeline_def(
#         batch_size=MAX_BATCH_SIZE,
#         device_id=0,
#         num_threads=ndd.get_num_threads(),
#         prefetch_queue_depth=1,
#     )
#     def pipeline():
#         rs1 = fn.external_source(
#             source=_random_state_source_factory(fn_rng, MAX_BATCH_SIZE, 1),
#             num_outputs=1,
#         )[
#             0
#         ]  # [0], since external_source returns a tuple
#         return fn.random.coin_flip(_random_state=rs1)

#     pipe = pipeline()
#     pipe.build()
#     pipe_out = pipe.run()
#     ndd_out = ndd.random.coin_flip(rng=ndd_rng, batch_size=MAX_BATCH_SIZE)
#     assert compare(pipe_out, ndd_out)


def test_random_bbox_crop():
    device = "cpu"

    def get_data(batch_size):
        obj_num = random.randint(1, 20)
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num, 1]
        bboxes = [
            np.random.random(size=test_box_shape).astype(dtype=np.float32)
            for _ in range(batch_size)
        ]
        labels = [
            np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32)
            for _ in range(batch_size)
        ]
        return (bboxes, labels)

    data = [get_data(MAX_BATCH_SIZE) for _ in range(N_ITERATIONS)]

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.random_bbox_crop,
        ndd_operator=ndd.random_bbox_crop,
        device=device,
        random=True,
        num_inputs=2,
    )


RANDOM_IMAGE_DECODER_OPERATORS = [
    # OperatorTestConfig(
    #     "decoders.image_random_crop", {"hw_decoder_load": 0.0}, devices=["cpu", "mixed"]
    # ),  # BUG
    # OperatorTestConfig(
    #     "experimental.decoders.image_random_crop",
    #     {"hw_decoder_load": 0.0},
    #     devices=["cpu", "mixed"],
    # ),  # BUG
]

random_image_decoders_test_configuration = flatten_operator_configs(RANDOM_IMAGE_DECODER_OPERATORS)


@params(*random_image_decoders_test_configuration)
def test_image_decoders(device, fn_operator, ndd_operator, operator_args):
    image_decoder_extensions = ".jpg"
    exclude_subdirs = ["jpeg_lossless"]
    data_path = os.path.join(test_utils.get_dali_extra_path(), "db", "single")
    data = generate_decoders_data(
        data_path, image_decoder_extensions, exclude_subdirs=exclude_subdirs
    )
    fn_rng, ndd_rng = create_rngs()

    @pipeline_def(
        batch_size=MAX_BATCH_SIZE,
        device_id=0,
        num_threads=ndd.get_num_threads(),
        prefetch_queue_depth=1,
    )
    def pipeline():
        rs1 = fn.external_source(
            source=_random_state_source_factory(fn_rng, MAX_BATCH_SIZE, 1),
            num_outputs=1,
        )[
            0
        ]  # [0], since external_source returns a tuple
        inp = fn.external_source(name="INPUT0", device="cpu")
        processed = fn_operator(inp, _random_state=rs1, device=device, **operator_args)
        return processed

    pipe = pipeline()
    pipe.build()

    for inp in data:
        feed_input(pipe, inp)
        pipe_out = pipe.run()
        ndd_out = ndd_operator(
            ndd.as_batch(inp, device="cpu"), rng=ndd_rng, device=device, **operator_args
        )
        assert compare(pipe_out, ndd_out)


def test_random_crop_generator():
    device = "cpu"
    data = generate_data((2,), dtype=np.int64, batch_sizes=MAX_BATCH_SIZE)

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.random_crop_generator,
        ndd_operator=ndd.random_crop_generator,
        device=device,
        random=True,
    )


def test_random_object_bbox():
    device = "cpu"
    data = generate_data(
        image_like_shape_generator, lo=0, hi=255, dtype=np.uint8, batch_sizes=MAX_BATCH_SIZE
    )
    run_operator_test(
        input_epoch=data,
        fn_operator=fn.segmentation.random_object_bbox,
        ndd_operator=ndd.segmentation.random_object_bbox,
        device=device,
        random=True,
    )


# BUG
# @params("cpu", "gpu")
# def test_batch_permutation(device):
#     batch_size = MAX_BATCH_SIZE
#     data = generate_data(
#         image_like_shape_generator, lo=0, hi=255, dtype=np.uint8, batch_sizes=batch_size
#     )
#     fn_rng, ndd_rng = create_rngs()

#     @pipeline_def(
#         batch_size=MAX_BATCH_SIZE,
#         device_id=0,
#         num_threads=ndd.get_num_threads(),
#         prefetch_queue_depth=1,
#     )
#     def pipeline():
#         rs1 = fn.external_source(
#             source=_random_state_source_factory(fn_rng, MAX_BATCH_SIZE, 1),
#             num_outputs=1,
#         )[
#             0
#         ]  # [0], since external_source returns a tuple
#         inp = fn.external_source(name="INPUT0", device=device)
#         perm = fn.batch_permutation(_random_state=rs1, device="cpu")
#         processed = fn.permute_batch(inp, indices=perm)
#         return processed

#     pipe = pipeline()
#     pipe.build()

#     for inp in data:
#         feed_input(pipe, inp)
#         pipe_out = pipe.run()
#         perm = ndd.batch_permutation(rng=ndd_rng)
#         ndd_out = ndd.permute_batch(ndd.as_batch(inp, device=device), indices=perm, device=device)
#         assert compare(pipe_out, ndd_out)


tested_operators = [
    "random.choice",
    "random.normal",
    "random.beta",
    "random.uniform",
    "random.coin_flip",
    "jitter",
    "noise.gaussian",
    "noise.shot",
    "noise.salt_and_pepper",
    "segmentation.random_mask_pixel",
    "roi_random_crop",
    "random_bbox_crop",
    "random_crop_generator",
    "segmentation.random_object_bbox",
]
