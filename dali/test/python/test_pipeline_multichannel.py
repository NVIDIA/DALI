# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
import cv2
import glob

test_data_root = os.environ["DALI_EXTRA_PATH"]
multichannel_tiff_root = os.path.join(
    test_data_root, "db", "single", "multichannel", "tiff_multichannel"
)
multichannel_tiff_files = glob.glob(multichannel_tiff_root + "/*.tif*")


def crop_func_help(image, layout, crop_y=0.2, crop_x=0.3, crop_h=220, crop_w=224):
    if layout == "FHWC":
        assert len(image.shape) == 4
        H = image.shape[1]
        W = image.shape[2]
    elif layout == "HWC":
        assert len(image.shape) == 3
        H = image.shape[0]
        W = image.shape[1]

    assert H >= crop_h
    assert W >= crop_w

    start_y = int(np.float32(crop_y) * np.float32(H - crop_h) + np.float32(0.5))
    end_y = start_y + crop_h
    start_x = int(np.float32(crop_x) * np.float32(W - crop_w) + np.float32(0.5))
    end_x = start_x + crop_w

    if layout == "FHWC":
        return image[:, start_y:end_y, start_x:end_x, :]
    elif layout == "HWC":
        return image[start_y:end_y, start_x:end_x, :]
    else:
        assert False  # should not happen


def crop_NHWC_func(image):
    return crop_func_help(image, "HWC")


def resize_func_help(image, size_x=300, size_y=900):
    res = cv2.resize(image, (size_x, size_y))
    return res


def resize_func(image):
    return resize_func_help(image)


def transpose_func(image):
    return image.transpose((1, 0, 2))


def normalize_func(image):
    return np.float32(image) / np.float32(255.0)


def full_pipe_func(image):
    out = resize_func(image)
    out = crop_NHWC_func(out)
    out = transpose_func(out)
    out = normalize_func(out)
    return out


class MultichannelSynthPipeline(Pipeline):
    def __init__(
        self, device, batch_size, layout, iterator, num_threads=1, device_id=0, tested_operator=None
    ):
        super(MultichannelSynthPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.tested_operator = tested_operator
        if self.tested_operator == "resize" or not self.tested_operator:
            self.resize = ops.Resize(
                device=self.device,
                resize_y=900,
                resize_x=300,
                min_filter=types.DALIInterpType.INTERP_LINEAR,
                antialias=False,
            )
        if self.tested_operator == "crop" or not self.tested_operator:
            self.crop = ops.Crop(
                device=self.device, crop=(220, 224), crop_pos_x=0.3, crop_pos_y=0.2
            )
        if self.tested_operator == "transpose" or not self.tested_operator:
            self.transpose = ops.Transpose(
                device=self.device, perm=(1, 0, 2), transpose_layout=False
            )
        if self.tested_operator == "normalize" or not self.tested_operator:
            self.cmn = ops.CropMirrorNormalize(
                device=self.device, std=255.0, mean=0.0, output_layout="HWC", dtype=types.FLOAT
            )

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == "gpu" else self.data
        if self.tested_operator == "resize" or not self.tested_operator:
            out = self.resize(out)
        if self.tested_operator == "crop" or not self.tested_operator:
            out = self.crop(out)
        if self.tested_operator == "transpose" or not self.tested_operator:
            out = self.transpose(out)
        if self.tested_operator == "normalize" or not self.tested_operator:
            out = self.cmn(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


class MultichannelSynthPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(MultichannelSynthPythonOpPipeline, self).__init__(
            batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False
        )
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.oper = ops.PythonFunction(function=function, output_layouts=layout)

    def define_graph(self):
        self.data = self.inputs()
        out = self.oper(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def get_numpy_func(tested_operator):
    if not tested_operator:
        return full_pipe_func
    elif tested_operator == "resize":
        return resize_func
    elif tested_operator == "crop":
        return crop_NHWC_func
    elif tested_operator == "transpose":
        return transpose_func
    elif tested_operator == "normalize":
        return normalize_func
    else:
        assert False


def check_multichannel_synth_data_vs_numpy(tested_operator, device, batch_size, shape):
    eii1 = RandomDataIterator(batch_size, shape=shape)
    eii2 = RandomDataIterator(batch_size, shape=shape)
    mc_pipe = MultichannelSynthPipeline(
        device, batch_size, "HWC", iter(eii1), tested_operator=tested_operator
    )
    mc_pipe_python_op = MultichannelSynthPythonOpPipeline(
        get_numpy_func(tested_operator), batch_size, "HWC", iter(eii2)
    )
    compare_pipelines(mc_pipe, mc_pipe_python_op, batch_size=batch_size, N_iterations=3, eps=0.2)


def test_multichannel_synth_data_vs_numpy():
    full_pipeline_case = None
    for tested_operator in ["resize", "crop", "transpose", "normalize", full_pipeline_case]:
        # TODO(janton): remove when we implement CPU transpose
        supported_devices = ["gpu"] if tested_operator in [None, "transpose"] else ["cpu", "gpu"]
        for device in supported_devices:
            for batch_size in {3}:
                for shape in {(2048, 512, 8)}:
                    yield (
                        check_multichannel_synth_data_vs_numpy,
                        tested_operator,
                        device,
                        batch_size,
                        shape,
                    )


class MultichannelPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0):
        super(MultichannelPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device

        self.reader = ops.readers.File(files=multichannel_tiff_files)

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        self.decoder = ops.decoders.Image(device=decoder_device, output_type=types.ANY_DATA)

        self.resize = ops.Resize(
            device=self.device,
            resize_y=900,
            resize_x=300,
            min_filter=types.DALIInterpType.INTERP_LINEAR,
            antialias=False,
        )

        self.crop = ops.Crop(
            device=self.device, crop_h=220, crop_w=224, crop_pos_x=0.3, crop_pos_y=0.2
        )

        self.transpose = ops.Transpose(device=self.device, perm=(1, 0, 2), transpose_layout=False)

        self.cmn = ops.CropMirrorNormalize(
            device=self.device, std=255.0, mean=0.0, output_layout="HWC", dtype=types.FLOAT
        )

    def define_graph(self):
        encoded_data, _ = self.reader()
        decoded_data = self.decoder(encoded_data)
        out = decoded_data.gpu() if self.device == "gpu" else decoded_data
        out = self.resize(out)
        out = self.crop(out)
        out = self.transpose(out)
        out = self.cmn(out)
        return out


class MultichannelPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, num_threads=1, device_id=0):
        super(MultichannelPythonOpPipeline, self).__init__(
            batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False
        )
        self.reader = ops.readers.File(files=multichannel_tiff_files)
        self.decoder = ops.decoders.Image(device="cpu", output_type=types.ANY_DATA)
        self.oper = ops.PythonFunction(function=function, output_layouts="HWC")

    def define_graph(self):
        encoded_data, _ = self.reader()
        decoded_data = self.decoder(encoded_data)
        out = self.oper(decoded_data)
        return out


def check_full_pipe_multichannel_vs_numpy(device, batch_size):
    compare_pipelines(
        MultichannelPipeline(device, batch_size),
        MultichannelPythonOpPipeline(full_pipe_func, batch_size),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-03,
    )


def test_full_pipe_multichannel_vs_numpy():
    for device in {"cpu", "gpu"}:
        for batch_size in {1, 3}:
            yield check_full_pipe_multichannel_vs_numpy, device, batch_size
