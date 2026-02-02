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


from nose2.tools import params
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    run_operator_test,
    flatten_operator_configs,
    array_1d_shape_generator,
    generate_data,
    MAX_BATCH_SIZE,
)


RANDOM_OPERATORS = [
    OperatorTestConfig("random.choice", devices=["cpu"]),
    OperatorTestConfig("random.normal"),
    OperatorTestConfig("random.beta", devices=["cpu"]),
    OperatorTestConfig("random.coin_flip"),
    OperatorTestConfig("random.uniform"),
]

random_ops_test_configuration = flatten_operator_configs(RANDOM_OPERATORS)


@params(*random_ops_test_configuration)
def test_random_choice(device, fn_operator, ndd_operator, operator_args):
    data = generate_data(array_1d_shape_generator, batch_sizes=MAX_BATCH_SIZE)

    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        random=True,
        operator_args=operator_args,
    )


# def test_random_object_bbox():
#     device = "cpu"
#     data = [
#         np.int32([[1, 0, 0, 0], [1, 2, 2, 1], [1, 1, 2, 0], [2, 0, 0, 1]]),
#         np.int32([[0, 3, 3, 0], [1, 0, 1, 2], [0, 1, 1, 0], [0, 2, 0, 1], [0, 2, 2, 1]]),
#     ] * N_ITERATIONS
#     run_operator_test(
#         input_epoch=data,
#         fn_operator=fn.segmentation.random_object_bbox,
#         ndd_operator=ndd.segmentation.random_object_bbox,
#         device=device,
#         random=True,
#     )

# @pipeline_def(
#     batch_size=MAX_BATCH_SIZE,
#     device_id=0,
#     num_threads=ndd.get_num_threads(),
#     prefetch_queue_depth=1,
# )
# def pipeline():
#     (rstate,) = fn.external_source(
#         source=_random_state_source_factory(fn_rng, MAX_BATCH_SIZE, 1), num_outputs=1
#     )
#     inp = fn.external_source(name="INPUT0", device=device)
#     out = fn.random.choice(inp, _random_state=rstate)
#     return out

# pipe = pipeline()
# pipe.build()
# for inp in data:
#     feed_input(pipe, inp)
#     pipe_out = pipe.run()
#     ndd_out = ndd.random.choice(ndd.as_batch(inp, device=device), rng=ndd_rng)
#     assert compare(pipe_out, ndd_out)
