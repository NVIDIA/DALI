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

import tensorflow as tf
import nvidia.dali.ops as ops
import nvidia.dali.pipeline as pipeline
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as dali_types
from test_utils_tensorflow import skip_for_incompatible_tf

import os
from nose2.tools import params
from nose_utils import raises, assert_equals
import itertools
import warnings

try:
    tf.compat.v1.enable_eager_execution()
except:  # noqa: E722
    pass

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

data_path = os.path.join(os.environ["DALI_EXTRA_PATH"], "db/single/jpeg/")
file_list_path = os.path.join(data_path, "image_list.txt")


def setup():
    skip_for_incompatible_tf()


def dali_pipe_batch_1(shapes, types, as_single_tuple=False):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.readers.File(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.decoders.Image(device="mixed")

        def define_graph(self):
            data, _ = self.reader()
            image = self.decoder(data)
            return image

    pipe = TestPipeline(batch_size=1, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=1, output_dtypes=types, output_shapes=shapes)
    # for clarity, we could have used the previous `pipe`
    pipe_ref = TestPipeline(batch_size=1, seed=0, device_id=0, num_threads=4)

    ds_iter = iter(ds)
    # See if the iteration over different images works
    if as_single_tuple:
        shapes = shapes[0]
    for _ in range(10):
        (image,) = ds_iter.next()
        (image_ref,) = pipe_ref.run()
        if shapes is None or len(shapes) == 4:
            assert_equals(image.shape, ([1] + image_ref[0].shape()))
        else:
            assert_equals(image.shape, image_ref[0].shape())


_batch_1_different_shapes_params = []
for shape in [
    None,
    (None, None, None, None),
    (None, None, None),
    (1, None, None, None),
    (1, None, None, 3),
    (None, None, 3),
]:
    _batch_1_different_shapes_params.append((shape, tf.uint8, False))
    _batch_1_different_shapes_params.append(((shape,), (tf.uint8,), True))


@params(*_batch_1_different_shapes_params)
def test_batch_1_different_shapes(shape, dtype, as_single_tuple):
    dali_pipe_batch_1(shape, dtype, as_single_tuple)


_batch_1_mixed_tuple_shapes = [
    (None, None, None, None),
    (None, None, None),
    (1, None, None, None),
    (1, None, None, 3),
    (None, None, 3),
]


@params(*_batch_1_mixed_tuple_shapes)
def test_batch_1_mixed_tuple_raises_value_error(shape):
    raises(ValueError, "The two structures don't have the same sequence length.")(
        dali_pipe_batch_1
    )(shape, (tf.uint8,))


@params(*_batch_1_mixed_tuple_shapes)
def test_batch_1_mixed_tuple_raises_type_error(shape):
    expected_msg = (
        "Dimension value must be integer or None * got value * " "with type '<class 'tuple'>'"
    )
    raises(TypeError, expected_msg)(dali_pipe_batch_1)((shape,), tf.uint8)


_batch_1_wrong_shapes = [(2, None, None, None), (None, None, 4), (2, None, None, 4), (None, 0, None, 3)]


@params(*_batch_1_wrong_shapes)
def test_batch_1_wrong_shape(shape):
    raises(
        tf.errors.InvalidArgumentError,
        "The shape provided for output `0` is not compatible with the "
        "shape returned by DALI Pipeline",
    )(dali_pipe_batch_1)(shape, tf.uint8)


def dali_pipe_batch_N(shapes, types, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.readers.File(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.decoders.Image(device="mixed")
            self.resize = ops.Resize(device="gpu", resize_x=200, resize_y=200)

        def define_graph(self):
            data, _ = self.reader()
            image = self.decoder(data)
            resized = self.resize(image)
            return resized

    pipe = TestPipeline(batch_size=batch, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for _ in range(10):
        (image,) = ds_iter.next()
        if shapes is None or len(shapes) == 4:
            assert_equals(image.shape, (batch, 200, 200, 3))
        else:
            assert_equals(image.shape, (200, 200, 3))


def _generate_batch_N_valid_shapes_params():
    result = []
    for batch in [1, 10]:
        # No shape
        result.append((None, tf.uint8, batch))
        # Full shape
        output_shape = (batch, 200, 200, 3)
        for i in range(2 ** len(output_shape)):
            noned_shape = tuple(
                (dim if i & (2**idx) else None) for idx, dim in enumerate(output_shape)
            )
            result.append((noned_shape, tf.uint8, batch))
    # Omitted batch of size `1`
    output_shape = (200, 200, 3)
    for i in range(2 ** len(output_shape)):
        noned_shape = tuple((dim if i & (2**idx) else None) for idx, dim in enumerate(output_shape))
        result.append((noned_shape, tf.uint8, 1))
    return result


@params(*_generate_batch_N_valid_shapes_params())
def test_batch_N_valid_shapes(shapes, types, batch):
    dali_pipe_batch_N(shapes, types, batch)


def dali_pipe_multiple_out(shapes, types, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.readers.File(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.decoders.Image(device="mixed")
            self.resize = ops.Resize(device="gpu", resize_x=200, resize_y=200)

        def define_graph(self):
            data, label = self.reader()
            image = self.decoder(data)
            resized = self.resize(image)
            return resized, label.gpu()

    pipe = TestPipeline(batch_size=batch, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for _ in range(10):
        image, label = ds_iter.next()
        if shapes is None or shapes[0] is None or len(shapes[0]) == 4:
            assert_equals(image.shape, (batch, 200, 200, 3))
        else:
            assert_equals(image.shape, (200, 200, 3))
        if shapes is None or shapes[1] is None or len(shapes[1]) == 2:
            assert_equals(label.shape, (batch, 1))
        else:
            assert_equals(label.shape, (batch,))


def _generate_multiple_input_valid_shapes_params():
    result = []
    for batch in [1, 10]:
        for shapes in [
            None,
            (None, None),
            ((batch, 200, 200, 3), None),
            (None, (batch, 1)),
            (None, (batch,)),
        ]:
            result.append((shapes, (tf.uint8, tf.int32), batch))
    return result


@params(*_generate_multiple_input_valid_shapes_params())
def test_multiple_input_valid_shapes(shapes, types, batch):
    dali_pipe_multiple_out(shapes, types, batch)


def _generate_multiple_input_invalid_params():
    result = []
    for batch in [1, 10]:
        for shapes in [(None,), (batch, 200, 200, 3, None), (None, None, None)]:
            result.append((shapes, (tf.uint8, tf.uint8), batch))
    return result


@params(*_generate_multiple_input_invalid_params())
def test_multiple_input_invalid(shapes, types, batch):
    raises(ValueError, "The two structures don't have the same sequence length.")(
        dali_pipe_multiple_out
    )(shapes, types, batch)


def dali_pipe_artificial_shape(shapes, tf_type, dali_type, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_type, idata=[1, 1], shape=[1, 2, 1])

        def define_graph(self):
            return self.constant().gpu()

    pipe = TestPipeline(batch_size=batch, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=tf_type, output_shapes=shapes)
    ds_iter = iter(ds)
    for _ in range(10):
        (out,) = ds_iter.next()
        if len(shapes) == 4:
            assert_equals(out.shape, (batch, 1, 2, 1))
        if len(shapes) == 3:
            assert_equals(out.shape, (batch, 1, 2))
        if len(shapes) == 2:
            assert_equals(
                out.shape,
                (
                    batch,
                    2,
                ),
            )
        if len(shapes) == 1:
            assert_equals(out.shape, (2,))


def _generate_artificial_match_params():
    result = []
    for batch in [1, 10]:
        for shape in [
            (None, None, None, None),
            (None, None, 2),
            (batch, None, None, None),
            (batch, None, 2),
        ]:
            result.append((shape, tf.uint8, dali_types.UINT8, batch))
    result.append(((10, 2), tf.uint8, dali_types.UINT8, 10))
    result.append(((2,), tf.uint8, dali_types.UINT8, 1))
    return result


@params(*_generate_artificial_match_params())
def test_artificial_match(shape, tf_type, dali_type, batch):
    dali_pipe_artificial_shape(shape, tf_type, dali_type, batch)


_artificial_no_match_shapes = [(11, None, None, None), (None, None, 3), (10, 2, 1, 1)]


@params(*_artificial_no_match_shapes)
def test_artificial_no_match(shape):
    batch = 10
    raises(
        tf.errors.InvalidArgumentError,
        "The shape provided for output `0` is not compatible with the "
        "shape returned by DALI Pipeline",
    )(dali_pipe_artificial_shape)(shape, tf.uint8, dali_types.UINT8, batch)


def dali_pipe_types(tf_type, dali_type):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_type, idata=[1, 1], shape=[2])

        def define_graph(self):
            return self.constant().gpu()

    pipe = TestPipeline(batch_size=1, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=1, output_dtypes=tf_type)
    ds_iter = iter(ds)
    (out,) = ds_iter.next()
    assert_equals(out.dtype, tf_type)


# float64 not tested because constant doesn't support it
tf_type_list = [
    tf.uint8,
    tf.uint16,
    tf.uint32,
    tf.uint64,
    tf.int8,
    tf.int16,
    tf.int32,
    tf.int64,
    tf.bool,
    tf.float16,
    tf.float32,
]
dali_type_list = [
    dali_types.UINT8,
    dali_types.UINT16,
    dali_types.UINT32,
    dali_types.UINT64,
    dali_types.INT8,
    dali_types.INT16,
    dali_types.INT32,
    dali_types.INT64,
    dali_types.BOOL,
    dali_types.FLOAT16,
    dali_types.FLOAT,
]
matching_types = list(zip(tf_type_list, dali_type_list))
all_types = itertools.product(tf_type_list, dali_type_list)
not_matching_types = list(set(all_types).difference(set(matching_types)))


@params(*matching_types)
def test_type_returns_matching(tf_t, dali_t):
    dali_pipe_types(tf_t, dali_t)


@params(*not_matching_types)
def test_type_returns_not_matching(tf_t, dali_t):
    raises(
        tf.errors.InvalidArgumentError,
        "The type provided for output `0` is not compatible with the type "
        "returned by DALI Pipeline",
    )(dali_pipe_types)(tf_t, dali_t)


def dali_pipe_deprecated(
    dataset_kwargs, shapes, tf_type, dali_type, batch, expected_warnings_count
):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_type, idata=[1, 1], shape=[2])

        def define_graph(self):
            return self.constant().gpu()

    pipe = TestPipeline(batch_size=batch, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = dali_tf.DALIDataset(pipe, batch_size=batch, **dataset_kwargs)
        assert_equals(len(w), expected_warnings_count)
        ds_iter = iter(ds)
        for _ in range(10):
            (out,) = ds_iter.next()
            if isinstance(shapes, int) or len(shapes) == 1:
                assert_equals(out.shape, (2,))
            else:
                assert_equals(out.shape, (batch, 2))
            assert_equals(out.dtype, tf_type)


def test_deprecated():
    dali_pipe_deprecated({"shapes": 2, "dtypes": tf.uint8}, 2, tf.uint8, dali_types.UINT8, 1, 2)
    dali_pipe_deprecated({"shapes": [4, 2], "dtypes": tf.uint8}, [4, 2], tf.uint8, dali_types.UINT8, 4, 2)
    dali_pipe_deprecated({"shapes": [[4, 2]], "dtypes": [tf.uint8]}, [4, 2], tf.uint8, dali_types.UINT8, 4, 2)
    dali_pipe_deprecated({"output_shapes": 2, "dtypes": tf.uint8}, 2, tf.uint8, dali_types.UINT8, 1, 1)
    dali_pipe_deprecated({"output_shapes": (4, 2), "dtypes": tf.uint8}, [4, 2], tf.uint8, dali_types.UINT8, 4, 1)
    dali_pipe_deprecated({"output_shapes": ((4, 2),), "dtypes": [tf.uint8]}, [4, 2], tf.uint8, dali_types.UINT8, 4, 1)
    dali_pipe_deprecated({"shapes": 2, "output_dtypes": tf.uint8}, 2, tf.uint8, dali_types.UINT8, 1, 1)
    dali_pipe_deprecated({"shapes": [4, 2], "output_dtypes": tf.uint8}, [4, 2], tf.uint8, dali_types.UINT8, 4, 1)
    dali_pipe_deprecated({"shapes": [[4, 2]], "output_dtypes": (tf.uint8,)}, [4, 2], tf.uint8, dali_types.UINT8, 4, 1)


def test_deprecated_double_def_shapes():
    error_msg = (
        "Usage of `{}` is deprecated in favor of `output_{}`*only `output_{}` "
        "should be provided."
    )
    shapes_error_msg = error_msg.format(*(("shapes",) * 3))
    raises(ValueError, shapes_error_msg)(dali_pipe_deprecated)(
        {"shapes": 2, "output_shapes": 2, "dtypes": tf.uint8},
        2, tf.uint8, dali_types.UINT8, 1, 2
    )


def test_deprecated_double_def_dtypes():
    error_msg = (
        "Usage of `{}` is deprecated in favor of `output_{}`*only `output_{}` "
        "should be provided."
    )
    dtypes_error_msg = error_msg.format(*(("dtypes",) * 3))
    raises(ValueError, dtypes_error_msg)(dali_pipe_deprecated)(
        {"shapes": 2, "dtypes": tf.uint8, "output_dtypes": tf.uint8},
        2, tf.uint8, dali_types.UINT8, 1, 2
    )


def test_no_output_dtypes():
    expected_msg = (
        "`output_dtypes` should be provided as single tf.DType value or a tuple of "
        "tf.DType values"
    )
    raises(TypeError, expected_msg)(dali_pipe_deprecated)(
        {"shapes": 2}, 2, tf.uint8, dali_types.UINT8, 1, 2
    )
