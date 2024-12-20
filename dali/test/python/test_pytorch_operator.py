# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch as dalitorch
import nvidia.dali.types as types
import os
import torch
from nvidia.dali.pipeline import Pipeline

from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "single", "jpeg")

DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 32
NUM_WORKERS = 6


class CommonPipeline(Pipeline):
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        num_threads=NUM_WORKERS,
        device_id=DEVICE_ID,
        image_dir=images_dir,
    ):
        super().__init__(batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False)
        self.input = ops.readers.File(file_root=image_dir)
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        return decoded, labels


class BasicPipeline(CommonPipeline):
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        num_threads=NUM_WORKERS,
        device_id=DEVICE_ID,
        image_dir=images_dir,
    ):
        super().__init__(batch_size, num_threads, device_id, image_dir)

    def define_graph(self):
        images, labels = self.load()
        return images


class TorchPythonFunctionPipeline(CommonPipeline):
    def __init__(
        self,
        function,
        device,
        bp=False,
        batch_size=BATCH_SIZE,
        num_threads=NUM_WORKERS,
        device_id=DEVICE_ID,
        image_dir=images_dir,
    ):
        super().__init__(batch_size, num_threads, device_id, image_dir)
        self.device = device
        self.torch_function = dalitorch.TorchPythonFunction(
            function=function, num_outputs=2, device=device, batch_processing=bp
        )

    def define_graph(self):
        images, labels = self.load()
        return self.torch_function(images if self.device == "cpu" else images.gpu())


def torch_operation(tensor):
    tensor_n = tensor.double() / 255
    return tensor_n.sin(), tensor_n.cos()


def torch_batch_operation(tensors):
    out = [torch_operation(t) for t in tensors]
    return [p[0] for p in out], [p[1] for p in out]


def check_pytorch_operator(device):
    pipe = BasicPipeline()
    pt_pipe = TorchPythonFunctionPipeline(torch_operation, device)
    for it in range(ITERS):
        (preprocessed_output,) = pipe.run()
        output1, output2 = pt_pipe.run()
        if device == "gpu":
            output1 = output1.as_cpu()
            output2 = output2.as_cpu()
        for i in range(len(output1)):
            res1 = output1.at(i)
            res2 = output2.at(i)
            exp1_t, exp2_t = torch_operation(torch.from_numpy(preprocessed_output.at(i)))
            assert numpy.allclose(res1, exp1_t.numpy())
            assert numpy.allclose(res2, exp2_t.numpy())


def test_pytorch_operator():
    for device in {"cpu", "gpu"}:
        yield check_pytorch_operator, device


def check_pytorch_operator_batch_processing(device):
    pipe = BasicPipeline()
    pt_pipe = TorchPythonFunctionPipeline(torch_batch_operation, device, True)
    for it in range(ITERS):
        (preprocessed_output,) = pipe.run()
        tensors = [torch.from_numpy(preprocessed_output.at(i)) for i in range(BATCH_SIZE)]
        exp1, exp2 = torch_batch_operation(tensors)
        output1, output2 = pt_pipe.run()
        if device == "gpu":
            output1 = output1.as_cpu()
            output2 = output2.as_cpu()
        for i in range(len(output1)):
            res1 = output1.at(i)
            res2 = output2.at(i)
            assert numpy.allclose(res1, exp1[i].numpy())
            assert numpy.allclose(res2, exp2[i].numpy())


def test_pytorch_operator_batch_processing():
    for device in {"cpu", "gpu"}:
        yield check_pytorch_operator_batch_processing, device


def verify_pipeline(pipeline, input):
    assert pipeline is Pipeline.current()
    return input


def test_current_pipeline():
    pipe1 = Pipeline(13, 4, 0)
    with pipe1:
        dummy = types.Constant(numpy.ones((1)))
        output = dalitorch.fn.torch_python_function(
            dummy, function=lambda inp: verify_pipeline(pipe1, inp)
        )
        pipe1.set_outputs(output)

    pipe2 = Pipeline(6, 2, 0)
    with pipe2:
        dummy = types.Constant(numpy.ones((1)))
        output = dalitorch.fn.torch_python_function(
            dummy, function=lambda inp: verify_pipeline(pipe2, inp)
        )
        pipe2.set_outputs(output)

    pipe1.run()
    pipe2.run()
