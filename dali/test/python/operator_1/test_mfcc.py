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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np
from functools import partial
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
import librosa as librosa
from nose_utils import assert_raises, attr, SkipTest
from nose2.tools import cartesian_params


class MFCCPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        axis=0,
        dct_type=2,
        lifter=1.0,
        n_mfcc=20,
        norm=None,
        num_threads=1,
        device_id=0,
    ):
        super(MFCCPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.mfcc = ops.MFCC(
            device=self.device,
            axis=axis,
            dct_type=dct_type,
            lifter=lifter,
            n_mfcc=n_mfcc,
            normalize=norm,
        )

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == "gpu" else self.data
        out = self.mfcc(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)


def mfcc_func(axis, dct_type, lifter, n_mfcc, norm, input_data):
    # Librosa works with frequency-major mel-spectrograms
    if axis == 1:
        input_data = np.transpose(input_data)

    in_shape = input_data.shape
    assert len(in_shape) == 2

    norm_str = "ortho" if norm else None

    out = librosa.feature.mfcc(
        S=input_data, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm_str, lifter=lifter
    )

    # Scipy DCT (used by Librosa) without normalization is scaled by a factor of 2 when comparing
    # with Wikipedia's formula
    if not norm:
        out = out / 2

    # Transpose back the output if necessary
    if axis == 1:
        out = np.transpose(out)

    return out


class MFCCPythonPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        axis=0,
        dct_type=2,
        lifter=1.0,
        n_mfcc=20,
        norm=None,
        num_threads=1,
        device_id=0,
        func=mfcc_func,
    ):
        super(MFCCPythonPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False
        )
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(func, axis, dct_type, lifter, n_mfcc, norm)
        self.mfcc = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.mfcc(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)


def check_operator_mfcc_vs_python(
    device, batch_size, input_shape, axis, dct_type, lifter, n_mfcc, norm
):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        MFCCPipeline(
            device,
            batch_size,
            iter(eii1),
            axis=axis,
            dct_type=dct_type,
            lifter=lifter,
            n_mfcc=n_mfcc,
            norm=norm,
        ),
        MFCCPythonPipeline(
            device,
            batch_size,
            iter(eii2),
            axis=axis,
            dct_type=dct_type,
            lifter=lifter,
            n_mfcc=n_mfcc,
            norm=norm,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-03,
    )


@attr("sanitizer_skip")
@cartesian_params(
    ["cpu", "gpu"],  # device
    [1, 3],  # batch_size
    [1, 2, 3],  # dct_type
    [True, False],  # norm
    [
        (0, 17, 0.0, (17, 1)),
        (1, 80, 2.0, (513, 100)),
        (1, 90, 0.0, (513, 100)),
        (1, 20, 202.0, (513, 100)),
    ],  # axis, n_mfcc, lifter, shape
)
def test_operator_mfcc_vs_python(device, batch_size, dct_type, norm, axis_nmfcc_lifter_shape):
    axis, n_mfcc, lifter, shape = axis_nmfcc_lifter_shape
    if dct_type == 1 and norm is True:
        raise SkipTest()
    check_operator_mfcc_vs_python(
        device,
        batch_size,
        shape,
        axis,
        dct_type,
        lifter,
        n_mfcc,
        norm,
    )


def check_operator_mfcc_wrong_args(
    device, batch_size, input_shape, axis, dct_type, lifter, n_mfcc, norm, msg
):
    with assert_raises(RuntimeError, regex=msg):
        eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
        pipe = MFCCPipeline(
            device,
            batch_size,
            iter(eii1),
            axis=axis,
            dct_type=dct_type,
            lifter=lifter,
            n_mfcc=n_mfcc,
            norm=norm,
        )
        pipe.run()


def test_operator_mfcc_wrong_args():
    batch_size = 3
    for device in ["cpu", "gpu"]:
        for dct_type, norm, axis, n_mfcc, lifter, shape, msg in [
            # DCT-I ortho-normalization is not supported
            (
                1,
                True,
                0,
                20,
                0.0,
                (100, 100),
                "Ortho-normalization is not supported for DCT type I",
            ),
            # axis out of bounds
            (2, False, -1, 20, 0.0, (100, 100), "Provided axis cannot be negative"),
            # axis out of bounds
            (2, False, 2, 20, 0.0, (100, 100), "Axis [\\d]+ is out of bounds \\[[\\d]+,[\\d]+\\)"),
            # not supported DCT type
            (
                10,
                False,
                0,
                20,
                0.0,
                (100, 100),
                "Unsupported DCT type: 10. Supported types are: 1, 2, 3, 4",
            ),
        ]:
            yield (
                check_operator_mfcc_wrong_args,
                device,
                batch_size,
                shape,
                axis,
                dct_type,
                lifter,
                n_mfcc,
                norm,
                msg,
            )
