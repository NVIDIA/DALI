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

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd
import os
import random
from nose2.tools import params, cartesian_params
from nvidia.dali.pipeline import pipeline_def
from test_detection_pipeline import coco_anchors
from segmentation_test_utils import make_batch_select_masks
from nose_utils import SkipTest
import test_utils
from ndd_vs_fn_test_utils import (
    run_operator_test,
    feed_input,
    use_fn_api,
    use_ndd_api,
    get_nested_attr,
    custom_shape_generator,
    array_1d_shape_generator,
    generate_image_like_data,
    generate_data,
    compare,
    MAX_BATCH_SIZE,
    N_ITERATIONS,
)


"""
This module tests DALI operators by comparing fn API with ndd (dynamic) API outputs.

Test Structure:
- OperatorTestConfig: Dataclass for defining operator test configurations
- Operator configuration lists: IMAGE_LIKE_OPERATORS, ARRAY_1D_OPERATORS, etc.
- Test runner: run_operator_test() provides common test execution logic
- Individual tests: Parameterized functions for different operator categories

To add a new operator test:
1. Add OperatorTestConfig to the appropriate category list
2. Specify operator name, arguments, and devices
3. The test will automatically be generated and run
"""

tested_operators = [
    "expand_dims",
    "squeeze",
    "box_encoder",
    "experimental.remap",
    "bb_flip",
    "coord_flip",
    "lookup_table",
    "reductions.mean",
    "reductions.std_dev",
    "reductions.variance",
    "sequence_rearrange",
    "element_extract",
    "nonsilent_region",
    "spectrogram",
    "mel_filter_bank",
    "to_decibels",
    "mfcc",
    "segmentation.select_masks",
    "optical_flow",
    "debayer",
    "filter",
    "cast_like",
    "full_like",
]


@params("cpu", "gpu")
def test_squeeze_op(device):
    data = generate_image_like_data()

    def operation(api, *inp, **operator_args):
        out = api.expand_dims(*inp, axes=[0, 2], new_axis_names="YZ")
        out = api.squeeze(out, axis_names="Z")
        return out

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
        input_layout="HWC",
    )


@params("cpu")
def test_box_encoder_op(device):

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

    data = [get_data(random.randint(5, MAX_BATCH_SIZE)) for _ in range(N_ITERATIONS)]

    def operation(api, *inp, **operator_args):
        processed, _ = api.box_encoder(*inp, device=device, anchors=coco_anchors())
        return processed

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
        num_inputs=2,
    )


@params("gpu")
def test_remap(device):

    def get_data(batch_size):
        input_shape = [480, 640, 3]
        mapx_shape = mapy_shape = [480, 640]
        input = [
            np.random.randint(0, 255, size=input_shape, dtype=np.uint8) for _ in range(batch_size)
        ]
        mapx = [
            640 * np.random.random(size=mapx_shape).astype(np.float32)  # [0, 640) interval
            for _ in range(batch_size)
        ]
        mapy = [
            480 * np.random.random(size=mapy_shape).astype(np.float32)  # [0, 480) interval
            for _ in range(batch_size)
        ]
        return input, mapx, mapy

    data = [get_data(random.randint(5, MAX_BATCH_SIZE)) for _ in range(N_ITERATIONS)]

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.experimental.remap,
        ndd_operator=ndd.experimental.remap,
        device=device,
        num_inputs=3,
    )


@params("cpu", "gpu")
def test_bb_flip(device):
    data = generate_data(custom_shape_generator(150, 250, 4, 4))
    run_operator_test(
        input_epoch=data,
        fn_operator=fn.bb_flip,
        ndd_operator=ndd.bb_flip,
        device=device,
    )


# Multi output and boxes=inp[:,:4] syntax
# def test_bbox_rotate():
#     device = "cpu"
#     data = generate_data(custom_shape_generator(150, 250, 5, 5))

#     def operation(api, inp, device=None, **operator_args):
#         boxes = inp[:, :4]
#         labels = api.cast(inp[:, 4], dtype=DALIDataType.INT32)
#         return api.bbox_rotate(boxes, labels, angle=45.0, input_shape=[255, 255])

#     pipe = pipeline_es_feed_input_wrapper(use_fn_api(operation), device=device)
#     for inp in data:
#         feed_input(pipe, inp)
#         pipe_out = pipe.run()
#         ndd_out = use_ndd_api(operation)(ndd.as_batch(inp, device=device), device=device)
#         assert compare(pipe_out, ndd_out)


@params("cpu", "gpu")
def test_coord_flip(device):
    data = generate_data(custom_shape_generator(150, 250, 2, 2))
    run_operator_test(
        input_epoch=data,
        fn_operator=fn.coord_flip,
        ndd_operator=ndd.coord_flip,
        device=device,
    )


@params("cpu", "gpu")
def test_lookup_table(device):
    data = generate_data(array_1d_shape_generator, lo=0, hi=5, dtype=np.uint8)
    run_operator_test(
        input_epoch=data,
        fn_operator=fn.lookup_table,
        ndd_operator=ndd.lookup_table,
        device=device,
        operator_args={"keys": [1, 3], "values": [10, 50]},
    )


@cartesian_params(
    ["cpu", "gpu"],
    ["reductions.std_dev", "reductions.variance"],
)
def test_reduce(device, reduce_fn_name):
    data = generate_image_like_data()

    def operation(api, *inp, **operator_args):
        mean = api.reductions.mean(*inp)
        reduce_fn = get_nested_attr(api, reduce_fn_name)
        reduced = reduce_fn(*inp, mean)
        return reduced

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
    )


@params("cpu", "gpu")
def test_sequence_rearrange(device):
    data = generate_data(sample_shape=(5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8)
    run_operator_test(
        input_epoch=data,
        fn_operator=fn.sequence_rearrange,
        ndd_operator=ndd.sequence_rearrange,
        device=device,
        input_layout="FHWC",
        operator_args={"new_order": [0, 4, 1, 3, 2]},
    )


@params("cpu", "gpu")
def test_element_extract(device):
    data = generate_data(sample_shape=(5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8)
    run_operator_test(
        input_epoch=data,
        fn_operator=fn.element_extract,
        ndd_operator=ndd.element_extract,
        device=device,
        input_layout="FHWC",
        operator_args={"element_map": [3]},
    )


@params("cpu")
def test_nonsilent_region(device):
    data = generate_data(array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8)

    def operation(api, *inp, **operator_args):
        processed, _ = api.nonsilent_region(*inp)
        return processed

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
    )


@params("cpu", "gpu")
def test_mel_filter_bank(device):
    data = generate_data(array_1d_shape_generator)

    def operation(api, *inp, **operator_args):
        spectrum = api.spectrogram(*inp, nfft=60, window_length=50, window_step=25)
        processed = api.mel_filter_bank(spectrum)
        return processed

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
    )


@params("cpu", "gpu")
def test_mfcc(device):
    data = generate_data(array_1d_shape_generator)

    def operation(api, *inp, **operator_args):
        spectrum = api.spectrogram(*inp, nfft=60, window_length=50, window_step=25)
        mel = api.mel_filter_bank(spectrum)
        dec = api.to_decibels(mel)
        processed = api.mfcc(dec)
        return processed

    run_operator_test(
        input_epoch=data,
        fn_operator=use_fn_api(operation),
        ndd_operator=use_ndd_api(operation),
        device=device,
    )


def test_segmentation_select_masks():
    device = "cpu"

    def repacked_make_batch_select_masks(*args, **kwargs):
        """
        make_batch_select_masks returns data in (polygons, vertices, selected_masks) order,
        while segmentation.select_masks expects (mask_ids, polygons, vertices).
        """
        batch = make_batch_select_masks(*args, **kwargs)
        return batch[2], batch[0], batch[1]

    data = [
        repacked_make_batch_select_masks(
            random.randint(5, MAX_BATCH_SIZE),
            vertex_ndim=2,
            npolygons_range=(1, 5),
            nvertices_range=(3, 10),
        )
        for _ in range(N_ITERATIONS)
    ]

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.segmentation.select_masks,
        ndd_operator=ndd.segmentation.select_masks,
        device=device,
        num_inputs=3,
        operator_args={"reindex_masks": False},
    )


@params("gpu")
def test_optical_flow(device):
    if not test_utils.is_of_supported():
        raise SkipTest("Optical Flow is not supported on this platform")
    max_batch_size = 5
    n_iterations = 2

    data = generate_data(
        max_batch_size=max_batch_size,
        n_iter=n_iterations,
        sample_shape=(10, 480, 640, 3),
        lo=0,
        hi=255,
        dtype=np.uint8,
    )

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.optical_flow,
        ndd_operator=ndd.optical_flow,
        device=device,
        operator_args={"output_grid": 4},
    )


# @test_utils.has_operator("decoders.inflate")
# @test_utils.restrict_platform(min_compute_cap=6.0)
# def test_inflate():
#     import lz4.block

#     def sample_to_lz4(sample):
#         deflated_buf = lz4.block.compress(sample, store_size=False)
#         return np.frombuffer(deflated_buf, dtype=np.uint8)

#     def sample_gen():
#         j = 42
#         while True:
#             yield np.full((13, 7), j)
#             j += 1

#     def inflate_pipeline(max_batch_size, inputs, device):
#         input_data = [[sample_to_lz4(sample) for sample in batch] for batch in inputs]
#         input_shape = [
#             [np.array(sample.shape, dtype=np.int32) for sample in batch] for batch in inputs
#         ]

#     @pipeline_def
#     def piepline():
#         defalted = fn.external_source(name="INPUT0")
#         shape = fn.external_source(name="INPUT1")
#         return fn.decoders.inflate(defalted.gpu(), shape=shape)

#         return piepline(batch_size=max_batch_size, num_threads=4, device_id=0)

#     sample = sample_gen()
#     batches = [
#         [next(sample) for _ in range(5)],
#         [next(sample) for _ in range(13)],
#         [next(sample) for _ in range(2)],
#     ]

#     check_pipeline(batches, inflate_pipline, devices=["gpu"])


@params("cpu", "gpu")
def test_debayer(device):

    from debayer_test_utils import rgb2bayer, bayer_patterns, blue_position

    def sample_gen():
        rng = np.random.default_rng(seed=101)
        j = 0
        while True:
            pattern = bayer_patterns[j % len(bayer_patterns)]
            h, w = 2 * np.int32(rng.uniform(2, 3, 2))
            r, g, b = np.full((h, w), j), np.full((h, w), j + 1), np.full((h, w), j + 2)
            rgb = np.uint8(np.stack([r, g, b], axis=2))
            yield (
                rgb2bayer(rgb, pattern),
                np.array(blue_position(pattern), dtype=np.int32),
            )
            j += 1

    def iteration_gen(batch_size=MAX_BATCH_SIZE):
        """
        Generate complete iterations (batches) for testing.

        :param batch_size: Number of samples per iteration/batch
        :return: Yields tuples where each element is a list of numpy arrays for each input
        """
        gen = sample_gen()
        while True:
            # Collect batch_size samples
            samples = [next(gen) for _ in range(batch_size)]
            # Transpose: list of tuples -> tuple of lists
            # samples = [(inp0_s0, inp1_s0, inp2_s0), (inp0_s1, inp1_s1, inp2_s1), ...]
            # iteration = ([inp0_s0, inp0_s1, ...], [inp1_s0, inp1_s1, ...], [inp2_s0, inp2_s1, ...])
            iteration = tuple(zip(*samples))
            # Convert each input from tuple to list
            iteration = tuple(list(input_data) for input_data in iteration)
            yield iteration

    iteration_gen_instance = iteration_gen()
    data = [next(iteration_gen_instance) for _ in range(N_ITERATIONS)]

    @pipeline_def(
        batch_size=MAX_BATCH_SIZE,
        device_id=0,
        num_threads=ndd.get_num_threads(),
        prefetch_queue_depth=1,
    )
    def pipeline():
        bayered = fn.external_source(name="INPUT0")
        positions = fn.external_source(name="INPUT1")
        if device == "gpu":
            bayered = bayered.gpu()
        return fn.debayer(bayered, blue_position=positions)

    pipe = pipeline()
    pipe.build()
    for inp in data:
        feed_input(pipe, inp)
        pipe_out = pipe.run()
        ndd_out = ndd.debayer(
            ndd.as_batch(inp[0], device=device),
            blue_position=ndd.as_batch(inp[1], device=device),
        )
        assert compare(pipe_out, ndd_out)


@params("cpu", "gpu")
def test_filter(device):
    def sample_gen():
        rng = np.random.default_rng(seed=101)
        sample_shapes = [(300, 600, 3), (100, 100, 3), (500, 1024, 1), (40, 40, 20)]
        filter_shapes = [(5, 7), (3, 3), (60, 2)]
        j = 0
        while True:
            sample_shape = sample_shapes[j % len(sample_shapes)]
            filter_shape = filter_shapes[j % len(filter_shapes)]
            sample = np.uint8(rng.uniform(0, 255, sample_shape))
            filter = np.float32(rng.uniform(0, 255, filter_shape))
            yield sample, filter
            j += 1

    def iteration_gen(batch_size=MAX_BATCH_SIZE):
        """
        Generate complete iterations (batches) for testing.

        :param batch_size: Number of samples per iteration/batch
        :return: Yields tuples where each element is a list of numpy arrays for each input
        """
        gen = sample_gen()
        while True:
            # Collect batch_size samples
            samples = [next(gen) for _ in range(batch_size)]
            # Transpose: list of tuples -> tuple of lists
            # samples = [(inp0_s0, inp1_s0, inp2_s0), (inp0_s1, inp1_s1, inp2_s1), ...]
            # iteration = ([inp0_s0, inp0_s1, ...], [inp1_s0, inp1_s1, ...], [inp2_s0, inp2_s1, ...])
            iteration = tuple(zip(*samples))
            # Convert each input from tuple to list
            iteration = tuple(list(input_data) for input_data in iteration)
            yield iteration

    iteration_gen_instance = iteration_gen()
    data = [next(iteration_gen_instance) for _ in range(N_ITERATIONS)]

    @pipeline_def(
        batch_size=MAX_BATCH_SIZE,
        device_id=0,
        num_threads=ndd.get_num_threads(),
        prefetch_queue_depth=1,
    )
    def pipeline():
        samples = fn.external_source(name="INPUT0", layout="HWC", device=device)
        filters = fn.external_source(name="INPUT1", device=device)
        return fn.filter(samples, filters)

    pipe = pipeline()
    pipe.build()
    for inp in data:
        feed_input(pipe, inp)
        pipe_out = pipe.run()
        ndd_out = ndd.filter(
            ndd.as_batch(inp[0], layout="HWC", device=device),
            ndd.as_batch(inp[1], device=device),
        )
        assert compare(pipe_out, ndd_out)


@params("cpu", "gpu")
def test_cast_like(device):
    def get_data(batch_size):
        test_data_shape = [random.randint(5, 21), random.randint(5, 21), 3]
        data1 = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        data2 = [
            np.random.randint(1, 4, size=test_data_shape, dtype=np.int32) for _ in range(batch_size)
        ]
        return data1, data2

    data = [get_data(random.randint(5, MAX_BATCH_SIZE)) for _ in range(N_ITERATIONS)]

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.cast_like,
        ndd_operator=ndd.cast_like,
        device=device,
        num_inputs=2,
    )


@params("cpu")
def test_full_like(device):
    def get_data(batch_size):
        test_data_shape = [random.randint(5, 21), random.randint(5, 21), 3]
        data1 = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        data2 = [np.array([42]) for _ in range(batch_size)]
        return data1, data2

    data = [get_data(random.randint(5, MAX_BATCH_SIZE)) for _ in range(N_ITERATIONS)]

    run_operator_test(
        input_epoch=data,
        fn_operator=fn.full_like,
        ndd_operator=ndd.full_like,
        device=device,
        num_inputs=2,
    )


# BUG
# @params("cpu")
# def test_io_file_read(device):
#     def get_data(batch_size):
#         rel_fpaths = [
#             "db/single/png/0/cat-1046544_640.png",
#             "db/single/png/0/cat-111793_640.png",
#             "db/single/multichannel/with_alpha/cat-111793_640-alpha.jp2",
#             "db/single/jpeg2k/2/tiled-cat-300572_640.jp2",
#         ]
#         path_strs = [
#             os.path.join(test_utils.get_dali_extra_path(), rel_fpath) for rel_fpath in rel_fpaths
#         ]
#         data = []
#         for i in range(batch_size):
#             data.append(np.frombuffer(path_strs[i % len(rel_fpaths)].encode(), dtype=np.int8))
#         return data

#     data = [get_data(random.randint(3, 9)) for _ in range(N_ITERATIONS)]
#     pipe = pipeline_es_feed_input_wrapper(fn.io.file.read, device=device)
#     for inp in data:
#         feed_input(pipe, inp)
#         pipe_out = pipe.run()
#         ndd_out = ndd.io.file.read(ndd.as_batch(inp, device=device))
#         assert compare(pipe_out, ndd_out)
