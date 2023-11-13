# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import numpy as np

from typing import cast, Sequence

from nvidia.dali import fn, ops
from nvidia.dali import types, tensors
from nvidia.dali.data_node import DataNode
from nvidia.dali.pipeline import pipeline_def, Pipeline

from test_utils import get_dali_extra_path, check_numba_compatibility_cpu
from nose.plugins.attrib import attr  # type: ignore

_test_root = Path(get_dali_extra_path())


# Use annotated function for additional verification. Anything that is not the expected type
# passed here would fail mypy checks, as well as printing the runtime error
def expect_data_node(*inputs: DataNode) -> None:
    for input in inputs:
        assert isinstance(input, DataNode), f"Expected DataNode, got {input} of type {type(input)}"


def expect_pipeline(pipe: Pipeline) -> None:
    assert isinstance(pipe, Pipeline), f"Expected Pipeline, got {pipe} of type {type(pipe)}"


def test_rn50_pipe():

    @pipeline_def(batch_size=10, device_id=0, num_threads=4)
    def rn50_pipe():
        enc, label = fn.readers.file(
            files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")],
            name="FileReader")
        imgs = fn.decoders.image(enc, device="mixed")
        rng = fn.random.coin_flip(probability=0.5)
        resized = fn.random_resized_crop(imgs, size=[224, 224])
        normalized = fn.crop_mirror_normalize(resized, mirror=rng, dtype=types.DALIDataType.FLOAT16,
                                              output_layout="HWC", crop=(224, 224),
                                              mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                              std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        expect_data_node(enc, label, imgs, rng, resized, normalized, label.gpu())
        return normalized, label.gpu()

    pipe = rn50_pipe()
    expect_pipeline(pipe)
    pipe.build()
    imgs, labels = pipe.run()
    assert isinstance(imgs, tensors.TensorListGPU)
    assert imgs.dtype == types.DALIDataType.FLOAT16  # noqa: E721
    assert isinstance(labels, tensors.TensorListGPU)


def test_rn50_ops_pipe():

    @pipeline_def(batch_size=10, device_id=0, num_threads=4)
    def rn50_ops_pipe():
        Reader = ops.readers.File(
            files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")],
            name="FileReader")
        Decoder = ops.decoders.Image(device="mixed")
        Rng = ops.random.CoinFlip(probability=0.5)
        Rrc = ops.RandomResizedCrop(device="gpu", size=[224, 224])
        Cmn = ops.CropMirrorNormalize(
            mirror=Rng(),
            device="gpu",
            dtype=types.DALIDataType.FLOAT16,
            output_layout="HWC",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        enc, label = Reader()
        imgs = Decoder(enc)
        resized = Rrc(imgs)
        normalized = Cmn(resized)
        expect_data_node(enc, label, imgs, resized, normalized, label.gpu())
        return normalized, label.gpu()

    pipe = rn50_ops_pipe()
    expect_pipeline(pipe)
    pipe.build()
    imgs, labels = pipe.run()
    assert isinstance(imgs, tensors.TensorListGPU)
    assert imgs.dtype == types.DALIDataType.FLOAT16  # noqa: E721
    assert isinstance(labels, tensors.TensorListGPU)


def test_cond_pipe():

    @pipeline_def(batch_size=10, device_id=0, num_threads=4, enable_conditionals=True)
    def cond_pipe():
        enc, label = fn.readers.file(
            files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")],
            name="FileReader")
        imgs = fn.decoders.image(enc, device="mixed")
        resized = fn.crop(imgs, crop=[224, 224])
        if fn.random.uniform(range=[0, 1]) < 0.25:
            out = fn.rotate(resized, angle=fn.random.uniform(range=[30, 60]))
        else:
            out = resized
        expect_data_node(enc, label, imgs, resized, out, label.gpu())
        return out, label.gpu()

    pipe = cond_pipe()
    expect_pipeline(pipe)
    pipe.build()
    imgs, labels = pipe.run()
    assert isinstance(imgs, tensors.TensorListGPU)
    assert imgs.dtype == types.DALIDataType.UINT8  # noqa: E721
    assert isinstance(labels, tensors.TensorListGPU)


def test_es_pipe():

    @pipeline_def(batch_size=10, device_id=0, num_threads=4)
    def es_pipe():
        single_output = fn.external_source(source=lambda: np.array([0]), batch=False)
        out_1, out_2 = fn.external_source(source=lambda: (np.array([1]), np.array([2])),
                                          num_outputs=2, batch=False)
        out_3, out_4 = fn.external_source(
            source=lambda: [np.array([3]), np.array([4])], num_outputs=2, batch=False)
        expect_data_node(single_output, out_1, out_2, out_3, out_4)
        return single_output, out_1, out_2, out_3, out_4

    pipe = es_pipe()
    expect_pipeline(pipe)
    pipe.build()
    out0, out1, out2, out3, out4 = pipe.run()
    assert np.array_equal(np.array(out0.as_tensor()), np.full((10, 1), 0))
    assert np.array_equal(np.array(out1.as_tensor()), np.full((10, 1), 1))
    assert np.array_equal(np.array(out2.as_tensor()), np.full((10, 1), 2))
    assert np.array_equal(np.array(out3.as_tensor()), np.full((10, 1), 3))
    assert np.array_equal(np.array(out4.as_tensor()), np.full((10, 1), 4))


def test_python_function_pipe():

    @pipeline_def(batch_size=2, device_id=0, num_threads=4)
    def fn_pipe():
        ops_fn = ops.PythonFunction(lambda: np.full((10, 1), 0), num_outputs=1)
        zeros = ops_fn()
        # Do a narrowing assertion, as the actual type depends on the value of num_outputs
        # parameter and we do not provide overload resolution based on it
        assert isinstance(zeros, DataNode)
        # Here we try out a cast, as we don't provide overloads based on num_outputs values,
        # there is one if we don't touch the num_values.
        ones = fn.python_function(zeros, function=lambda x: x + np.full((10, 1), 1))
        twos, zeros_2 = cast(
            Sequence[DataNode],
            fn.python_function(zeros, function=lambda x: (x + np.full((10, 1), 2), x),
                               num_outputs=2))
        fn.python_function(zeros, function=print, num_outputs=0)
        expect_data_node(zeros, zeros_2, ones, twos)
        return zeros + twos - twos, zeros_2, ones

    pipe = fn_pipe()
    expect_pipeline(pipe)
    pipe.build()
    out0, out1, out2 = pipe.run()
    assert np.array_equal(np.array(out0.as_tensor()), np.full((2, 10, 1), 0))
    assert np.array_equal(np.array(out1.as_tensor()), np.full((2, 10, 1), 0))
    assert np.array_equal(np.array(out2.as_tensor()), np.full((2, 10, 1), 1))


@attr('pytorch')
def test_pytorch_plugin():
    import nvidia.dali.plugin.pytorch as dali_torch
    import torch  # type: ignore

    @pipeline_def(batch_size=2, device_id=0, num_threads=4)
    def torch_pipe():
        ops_fn = dali_torch.TorchPythonFunction(lambda: torch.full((10, 1), 0), num_outputs=1)
        zeros = ops_fn()
        # Do a narrowing assertion, as the actual type depends on the value of num_outputs
        # parameter and we do not provide overload resolution based on it
        assert isinstance(zeros, DataNode)
        # Here we try out a cast, as we don't provide overloads based on num_outputs values,
        # there is one if we don't touch the num_values.
        ones, zeros_2 = cast(
            Sequence[DataNode],
            dali_torch.fn.torch_python_function(zeros, function=lambda x: (x + torch.full(
                (10, 1), 1), x), num_outputs=2))
        twos = dali_torch.fn.torch_python_function(zeros, function=lambda x: x + torch.full(
            (10, 1), 2))
        dali_torch.fn.torch_python_function(zeros, function=print, num_outputs=0)
        expect_data_node(zeros, zeros_2, ones, twos)
        return zeros + twos - twos, ones

    pipe = torch_pipe()
    expect_pipeline(pipe)
    pipe.build()
    out0, out1 = pipe.run()
    assert np.array_equal(np.array(out0.as_tensor()), np.full((2, 10, 1), 0))
    assert np.array_equal(np.array(out1.as_tensor()), np.full((2, 10, 1), 1))

    dali_iter = dali_torch.DALIGenericIterator([torch_pipe()], ["zero", "one"])
    for _, res in zip(range(1), dali_iter):
        out_dict = res[0]
        out0_iter = out_dict["zero"]
        out1_iter = out_dict["one"]
        assert isinstance(out0_iter, torch.Tensor)
        assert isinstance(out1_iter, torch.Tensor)

        assert np.array_equal(out0_iter, torch.full((2, 10, 1), 0))
        assert np.array_equal(out1_iter, torch.full((2, 10, 1), 1))


@attr('numba')
def test_numba_plugin():
    import nvidia.dali.plugin.numba as dali_numba

    check_numba_compatibility_cpu()

    def double_sample(out_sample, in_sample):
        out_sample[:] = 2 * in_sample[:]

    @pipeline_def(batch_size=2, device_id=0, num_threads=4)
    def numba_pipe():
        forty_two = fn.external_source(source=lambda x: np.full((2, ), 42, dtype=np.uint8),
                                       batch=False)
        out = dali_numba.fn.experimental.numba_function(forty_two, run_fn=double_sample,
                                                        out_types=[types.DALIDataType.UINT8],
                                                        in_types=[types.DALIDataType.UINT8],
                                                        outs_ndim=[1], ins_ndim=[1],
                                                        batch_processing=False)
        return out

    pipe = numba_pipe()
    pipe.build()
    out, = pipe.run()
    assert np.array_equal(np.array(out.as_tensor()), np.full((2, 2), 84))
