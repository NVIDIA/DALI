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
import numpy as np
from nose2.tools import params
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    image_like_shape_generator,
    run_operator_test,
    flatten_operator_configs,
    array_1d_shape_generator,
    generate_data,
    sign_off,
    test_all_devices,
    create_rngs,
    compare,
    pipeline_es_feed_input_wrapper,
    MAX_BATCH_SIZE,
    N_ITERATIONS,
)
import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd


RANDOM_OPERATORS_1D_ARRAY = [
    OperatorTestConfig("random.choice"),
    OperatorTestConfig("random.normal"),
    OperatorTestConfig("random.beta"),
    OperatorTestConfig("random.coin_flip"),
    OperatorTestConfig("random.uniform"),
]

random_ops_1d_array_test_configuration = flatten_operator_configs(RANDOM_OPERATORS_1D_ARRAY)


@params(*random_ops_1d_array_test_configuration)
def test_random_1d_array(device, operator_name, fn_operator, ndd_operator, operator_args):
    data = generate_data(array_1d_shape_generator, batch_sizes=MAX_BATCH_SIZE)

    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        operator_args=operator_args,
    )


@sign_off("random_bbox_crop")
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
        num_inputs=2,
    )


@sign_off("random_crop_generator")
def test_random_crop_generator():
    device = "cpu"
    data = generate_data((2,), dtype=np.int64, batch_sizes=MAX_BATCH_SIZE)

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.random_crop_generator,
        ndd_operator=ndd.random_crop_generator,
        device=device,
    )


@sign_off("segmentation.random_object_bbox")
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
    )


@test_all_devices("batch_permutation")
def test_batch_permutation(device):
    batch_size = MAX_BATCH_SIZE
    fn_rng, ndd_rng = create_rngs()

    pipe = pipeline_es_feed_input_wrapper(
        fn.batch_permutation,
        device=device,
        input_device=device,
        max_batch_size=batch_size,
        rng=fn_rng,
        needs_input=False,
        batch_sizes=[batch_size],
    )

    for _ in range(10):
        pipe_out = pipe.run()
        ndd_out = ndd.batch_permutation(rng=ndd_rng, batch_size=batch_size)
        compare(pipe_out, ndd_out)
