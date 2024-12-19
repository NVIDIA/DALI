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

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import numpy as np
from nose_utils import assert_raises, attr
import sys
import inspect
import nose


def _test_fn_rotate(device):
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=0)

    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint8)[
        :, :, np.newaxis
    ]
    batch = [image]

    input = fn.external_source([batch], layout="HWC")
    rotated = fn.rotate(input.gpu() if device == "gpu" else input, angle=90)
    pipe.set_outputs(rotated)

    outs = pipe.run()
    out = outs[0] if device == "cpu" else outs[0].as_cpu()
    arr = out.at(0)
    ref = np.array([[4, 8, 12], [3, 7, 11], [2, 6, 10], [1, 5, 9]])[:, :, np.newaxis]
    assert np.array_equal(arr, ref)


def test_set_outputs():
    data = [[[np.random.rand(1, 3, 2)], [np.random.rand(1, 4, 5)]]]
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=None)
    pipe.set_outputs(fn.external_source(data, num_outputs=2, cycle="quiet"))
    with assert_raises(
        TypeError, glob="Illegal pipeline output type. " "The output * contains a nested `DataNode`"
    ):
        pipe.build()


def test_set_outputs_err_msg_unpack():
    data = [[[np.random.rand(1, 3, 2)], [np.random.rand(1, 4, 5)]]]
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=None)
    pipe.set_outputs(fn.external_source(data, num_outputs=2, cycle="quiet"))
    with assert_raises(
        TypeError, glob="Illegal pipeline output type. " "The output * contains a nested `DataNode`"
    ):
        pipe.build()


def test_set_outputs_err_msg_random_type():
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=None)
    pipe.set_outputs("test")
    with assert_raises(
        TypeError, glob="Illegal output type. " "The output * is a `<class 'str'>`."
    ):
        pipe.build()


def test_fn_rotate():
    for device in ["cpu", "gpu"]:
        yield _test_fn_rotate, device


def test_fn_python_function():
    pipe = Pipeline(1, 1, 0, exec_pipelined=False, exec_async=False)

    batch1 = [np.array([1, 2, 3])]
    batch2 = [np.array([2, 3, 4])]
    # we need a context, because we use an operator with potential side-effects (python_function)
    with pipe:
        src = fn.external_source([batch1, batch2])
        out = fn.python_function(src, function=lambda x: x + 1)
        pipe.set_outputs(out)

    assert np.array_equal(pipe.run()[0].at(0), batch1[0] + 1)
    assert np.array_equal(pipe.run()[0].at(0), batch2[0] + 1)


def test_fn_multiple_input_sets():
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=0)

    image1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint8)[
        :, :, np.newaxis
    ]
    image2 = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8)[:, :, np.newaxis]
    batches = [[image1], [image2]]

    inputs = fn.external_source(lambda: batches, 2, layout="HWC")
    rotated = fn.rotate(inputs, angle=90)
    pipe.set_outputs(*rotated)

    outs = pipe.run()
    arr1 = outs[0].at(0)
    arr2 = outs[1].at(0)
    ref1 = np.array([[4, 8, 12], [3, 7, 11], [2, 6, 10], [1, 5, 9]])[:, :, np.newaxis]
    ref2 = np.array([[20, 40, 60], [10, 30, 50]], dtype=np.uint8)[:, :, np.newaxis]
    assert np.array_equal(arr1, ref1)
    assert np.array_equal(arr2, ref2)


def test_scalar_constant():
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=0)

    image1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint8)[
        :, :, np.newaxis
    ]
    image2 = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8)[:, :, np.newaxis]
    batches = [[image1], [image2]]

    inputs = fn.external_source(lambda: batches, 2, layout="HWC")
    rotated = fn.rotate(inputs, angle=types.ScalarConstant(90))
    pipe.set_outputs(*rotated, types.ScalarConstant(90))

    outs = pipe.run()
    arr1 = outs[0].at(0)
    arr2 = outs[1].at(0)
    arr3 = outs[2].at(0)
    ref1 = np.array([[4, 8, 12], [3, 7, 11], [2, 6, 10], [1, 5, 9]])[:, :, np.newaxis]
    ref2 = np.array([[20, 40, 60], [10, 30, 50]], dtype=np.uint8)[:, :, np.newaxis]
    ref3 = np.array(90)
    assert np.array_equal(arr1, ref1)
    assert np.array_equal(arr2, ref2)
    assert np.array_equal(arr3, ref3)


def test_to_snake_case_impl():
    fn_name_tests = [
        ("Test", "test"),
        ("OneTwo", "one_two"),
        ("TestXYZ", "test_xyz"),
        ("testX", "test_x"),
        ("TestXx", "test_xx"),
        ("testXX", "test_xx"),
        ("OneXYZTwo", "one_xyz_two"),
        ("MFCC", "mfcc"),
        ("RandomBBoxCrop", "random_bbox_crop"),
        ("STFT_CPU", "stft_cpu"),
        ("DOUBLE__UNDERSCORE", "double__underscore"),
        ("double__underscore", "double__underscore"),
        ("XYZ1ABC", "xyz1abc"),
        ("XYZ1abc", "xyz1abc"),
        ("trailing__", "trailing__"),
        ("TRAILING__", "trailing__"),
        ("Caffe2Reader", "caffe2_reader"),
        ("COCOReader", "coco_reader"),
        ("DLTensorPythonFunction", "dl_tensor_python_function"),
        ("TFRecordReader", "tfrecord_reader"),
        ("_Leading", "_leading"),
        ("_LEADing", "_lea_ding"),
        ("_LeAdIng", "_le_ad_ing"),
        ("_L_Eading", "_l_eading"),
    ]

    for inp, out in fn_name_tests:
        assert fn._to_snake_case(inp) == out, f"{fn._to_snake_case(inp)} != {out}"


def _test_schema_name_for_module(module_name, base_name=""):
    """Sanity test if we didn't miss the _schema_name for any op with custom wrapper"""
    if module_name.endswith("hidden"):
        return
    if base_name == "":
        base_name = module_name
    dali_module = sys.modules[module_name]
    for member_name in dir(dali_module):
        if member_name.startswith("_"):
            continue
        member = getattr(dali_module, member_name)
        if inspect.isfunction(member):
            # Check if we can reconstruct the name of the op from provided schema
            assert hasattr(member, "_schema_name")
            full_name = ops._op_name(member._schema_name)
            nose.tools.eq_(base_name + "." + full_name, module_name + "." + member_name)
        elif inspect.ismodule(member) and (module_name + "." + member_name) in sys.modules.keys():
            # Recurse on DALI submodule (filter out non-DALI reexported modules like `sys`)
            _test_schema_name_for_module(module_name + "." + member_name, base_name)


def test_schema_name():
    _test_schema_name_for_module("nvidia.dali.fn")


@attr("pytorch")
def test_schema_name_torch():
    import nvidia.dali.plugin.pytorch  # noqa: F401

    _test_schema_name_for_module("nvidia.dali.plugin.pytorch.fn")


@attr("numba")
def test_schema_name_numba():
    import nvidia.dali.plugin.numba  # noqa: F401

    _test_schema_name_for_module("nvidia.dali.plugin.numba.fn.experimental")
