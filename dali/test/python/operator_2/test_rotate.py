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

import cv2
import numpy as np
import math
import os
import random

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from test_utils import compare_pipelines
from sequences_test_utils import (
    ArgData,
    ArgDesc,
    ArgCb,
    ParamsProvider,
    get_video_input_cases,
    sequence_suite_helper,
)


test_data_root = os.environ["DALI_EXTRA_PATH"]
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")


def get_output_size(angle, input_size, parity_correction=True):
    cosa = abs(math.cos(angle))
    sina = abs(math.sin(angle))
    (h, w) = input_size[0:2]
    eps = 1e-2
    out_w = int(math.ceil(w * cosa + h * sina - eps))
    out_h = int(math.ceil(h * cosa + w * sina - eps))
    if not parity_correction:
        return (out_h, out_w)

    if sina <= cosa:
        if out_w % 2 != w % 2:
            out_w += 1
        if out_h % 2 != h % 2:
            out_h += 1
    else:
        if out_w % 2 != h % 2:
            out_w += 1
        if out_h % 2 != w % 2:
            out_h += 1
    return (out_h, out_w)


def get_3d_lin_rotation(angle, axis):
    # mirrors transform.h:rotation3D
    if not angle:
        return np.eye((3, 3), dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    axis = [dim / axis_norm for dim in axis]
    u, v, w = axis
    cosa = math.cos(angle)
    sina = math.sin(angle)
    return np.array(
        [
            [
                u * u + (v * v + w * w) * cosa,
                u * v * (1 - cosa) - w * sina,
                u * w * (1 - cosa) + v * sina,
            ],
            [
                u * v * (1 - cosa) + w * sina,
                v * v + (u * u + w * w) * cosa,
                v * w * (1 - cosa) - u * sina,
            ],
            [
                u * w * (1 - cosa) - v * sina,
                v * w * (1 - cosa) + u * sina,
                w * w + (u * u + v * v) * cosa,
            ],
        ],
        dtype=np.float32,
    )


def get_3d_output_size(angle, axis, input_size, parity_correction=False):
    transform = np.abs(get_3d_lin_rotation(angle, axis))
    eps = 1e-2
    in_size = np.array(input_size[2::-1], dtype=np.int32)
    out_size = np.int32(np.ceil(np.matmul(transform, in_size) - eps))

    if parity_correction:
        dominant_axis = np.argmax(transform, axis=1)
        out_size += (out_size % 2) ^ (in_size[dominant_axis] % 2)

    return out_size[::-1]


def get_transform(angle, input_size, output_size):
    cosa = math.cos(angle)
    sina = math.sin(angle)
    (out_h, out_w) = output_size[0:2]
    (in_h, in_w) = input_size[0:2]
    t1 = np.array([[1, 0, -out_w * 0.5], [0, 1, -out_h * 0.5], [0, 0, 1]])
    r = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    t2 = np.array([[1, 0, in_w * 0.5], [0, 1, in_h * 0.5], [0, 0, 1]])

    return (np.matmul(t2, np.matmul(r, t1)))[0:2, 0:3]


def ToCVMatrix(matrix):
    offset = np.matmul(matrix, np.array([[0.5], [0.5], [1]]))
    result = matrix.copy()
    result[0][2] = offset[0] - 0.5
    result[1][2] = offset[1] - 0.5
    return result


def CVRotate(output_type, input_type, fixed_size):
    def warp_fn(img, angle):
        in_size = img.shape[0:2]
        angle = math.radians(angle)
        out_size = fixed_size if fixed_size is not None else get_output_size(angle, in_size)
        matrix = get_transform(angle, in_size, out_size)
        matrix = ToCVMatrix(matrix)
        if output_type == dali.types.FLOAT or input_type == dali.types.FLOAT:
            img = np.float32(img)
        out_size_wh = (out_size[1], out_size[0])
        out = cv2.warpAffine(
            img,
            matrix,
            out_size_wh,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[42, 42, 42],
            flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
        )
        if output_type == dali.types.UINT8 and input_type == dali.types.FLOAT:
            out = np.uint8(np.clip(out, 0, 255))
        return out

    return warp_fn


class RotatePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        output_type,
        input_type,
        fixed_size=None,
        num_threads=3,
        device_id=0,
        num_gpus=1,
    ):
        super(RotatePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False
        )
        self.name = device
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        if input_type != dali.types.UINT8:
            self.cast = ops.Cast(device=device, dtype=input_type)
        else:
            self.cast = None

        self.uniform = ops.random.Uniform(range=(-180.0, 180.0), seed=42)
        self.rotate = ops.Rotate(device=device, size=fixed_size, fill_value=42, dtype=output_type)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.rotate.device == "gpu":
            images = images.gpu()
        if self.cast:
            images = self.cast(images)

        outputs = self.rotate(images, angle=self.uniform())
        return outputs


class CVPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        output_type,
        input_type,
        fixed_size,
        num_threads=3,
        device_id=0,
        num_gpus=1,
    ):
        super(CVPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False
        )
        self.name = "cv"
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        self.rotate = ops.PythonFunction(
            function=CVRotate(output_type, input_type, fixed_size), output_layouts="HWC"
        )
        self.uniform = ops.random.Uniform(range=(-180.0, 180.0), seed=42)
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        angles = self.uniform()
        outputs = self.rotate(images, angles)
        return outputs


def compare(pipe1, pipe2, eps):
    epoch_size = pipe1.epoch_size("Reader")
    batch_size = pipe1.max_batch_size
    niter = 1 if batch_size >= epoch_size else 2
    compare_pipelines(pipe1, pipe2, batch_size, niter, eps)


io_types = [
    (dali.types.UINT8, dali.types.UINT8),
    (dali.types.UINT8, dali.types.FLOAT),
    (dali.types.FLOAT, dali.types.UINT8),
    (dali.types.FLOAT, dali.types.FLOAT),
]


def create_pipeline(backend, *args):
    if backend == "cv":
        return CVPipeline(*args)
    else:
        return RotatePipeline(backend, *args)


def run_cases(backend1, backend2, epsilon):
    for output_size in [None, (160, 240)]:
        for itype, otype in io_types:
            batch_size = np.random.choice([1, 4, 19])

            def run_case(backend1, backend2, *args):
                pipe1 = create_pipeline(backend1, *args)
                pipe2 = create_pipeline(backend2, *args)
                compare(pipe1, pipe2, epsilon)

            yield run_case, backend1, backend2, batch_size, otype, itype, output_size


def test_gpu_vs_cv():
    for test in run_cases("gpu", "cv", 8):
        yield test


def test_cpu_vs_cv():
    for test in run_cases("cpu", "cv", 8):
        yield test


def test_gpu_vs_cpu():
    for test in run_cases("gpu", "cpu", 1):
        yield test


def infer_sequence_size(input_shapes, angles, axes=None):
    assert len(input_shapes) == len(angles)
    assert axes is None or len(axes) == len(angles)
    if axes is None:
        no_correction_shapes = [
            np.array(get_output_size(math.radians(angle), shape, False), dtype=np.int32)
            for shape, angle in zip(input_shapes, angles)
        ]
        corrected_shapes = [
            np.array(get_output_size(math.radians(angle), shape, True), dtype=np.int32)
            for shape, angle in zip(input_shapes, angles)
        ]
    else:
        no_correction_shapes = [
            np.array(get_3d_output_size(math.radians(angle), axis, shape, False), dtype=np.int32)
            for shape, angle, axis in zip(input_shapes, angles, axes)
        ]
        corrected_shapes = [
            np.array(get_3d_output_size(math.radians(angle), axis, shape, True), dtype=np.int32)
            for shape, angle, axis in zip(input_shapes, angles, axes)
        ]
    max_shape = np.max(no_correction_shapes, axis=0)
    parity = np.sum(np.array(corrected_shapes, dtype=np.int32) % 2, axis=0)
    for i in range(len(max_shape)):
        if max_shape[i] % 2 != (2 * parity[i] > len(input_shapes)):
            max_shape[i] += 1
    return max_shape


def sequence_batch_output_size(unfolded_extents, input_batch, angle_batch, axis_batch=None):
    def iter_by_groups():
        assert sum(unfolded_extents) == len(input_batch)
        assert len(input_batch) == len(angle_batch)
        assert axis_batch is None or len(axis_batch) == len(angle_batch)
        offset = 0
        for group in unfolded_extents:
            yield input_batch[offset : offset + group], angle_batch[offset : offset + group], (
                None if axis_batch is None else axis_batch[offset : offset + group]
            )
            offset += group

    sequence_output_shape = [
        infer_sequence_size([frame.shape for frame in input_frames], angles, axes)
        for input_frames, angles, axes in iter_by_groups()
    ]
    return [
        output_shape
        for output_shape, num_frames in zip(sequence_output_shape, unfolded_extents)
        for _ in range(num_frames)
    ]


class RotatePerFrameParamsProvider(ParamsProvider):
    """
    Provides per frame angle argument input to the video rotate operator test.
    The expanded baseline pipeline must be provided with additional argument ``size``
    to make allowance for coalescing of inferred frames sizes
    """

    def __init__(self, input_params):
        super().__init__(input_params)

    def expand_params(self):
        assert self.input_data.desc.expandable_prefix == "F"
        expanded_params = super().expand_params()
        params_dict = {param_data.desc.name: param_data for param_data in expanded_params}
        expanded_angles = params_dict.get("angle")
        expanded_axis = params_dict.get("axis")
        assert (
            expanded_angles is not None
            and "size" not in self.fixed_params
            and "size" not in params_dict
        )
        sequence_extents = [
            [sample.shape[0] for sample in input_batch] for input_batch in self.input_data.data
        ]
        output_size_params = (sequence_extents, self.unfolded_input.data, expanded_angles.data)
        if expanded_axis is not None:
            output_size_params += (expanded_axis.data,)
        output_sizes = [sequence_batch_output_size(*args) for args in zip(*output_size_params)]
        expanded_params.append(ArgData(ArgDesc("size", "", "cpu"), output_sizes))
        return expanded_params

    def __repr__(self):
        return "{}({})".format(repr(self.__class__), repr(self.input_params))


def test_video():
    def small_angle(sample_desc):
        return np.array(sample_desc.rng.uniform(-44.0, 44.0), dtype=np.float32)

    def random_angle(sample_desc):
        return np.array(sample_desc.rng.uniform(-180.0, 180.0), dtype=np.float32)

    def random_output(sample_desc):
        return np.array([sample_desc.rng.randint(300, 400), rng.randint(300, 400)])

    video_test_cases = [
        (dali.fn.rotate, {"angle": 45.0}, []),
        (dali.fn.rotate, {}, [ArgCb("angle", small_angle, False)]),
        (dali.fn.rotate, {}, [ArgCb("angle", random_angle, False)]),
        (dali.fn.rotate, {}, RotatePerFrameParamsProvider([ArgCb("angle", small_angle, True)])),
        (dali.fn.rotate, {}, RotatePerFrameParamsProvider([ArgCb("angle", random_angle, True)])),
        (
            dali.fn.rotate,
            {},
            [ArgCb("angle", small_angle, True), ArgCb("size", random_output, False)],
        ),
    ]

    rng = random.Random(42)
    video_cases = get_video_input_cases("FHWC", rng, larger_shape=(512, 287))
    input_cases = [ArgData(ArgDesc(0, "F", "", "FHWC"), input_data) for input_data in video_cases]
    yield from sequence_suite_helper(rng, input_cases, video_test_cases)


def test_3d_sequence():
    rng = random.Random(42)
    num_batches = 4
    max_batch_size = 8
    max_frames_num = 32
    input_layout = "FDHWC"
    np_rng = np.random.default_rng(42)

    def get_random_sample():
        num_frames = rng.randint(1, max_frames_num)
        d, h, w = tuple(rng.randint(10, 50) for _ in range(3))
        return np.int32(np_rng.uniform(0, 255, (num_frames, d, h, w, 3)))

    def get_random_batch():
        return [get_random_sample() for _ in range(rng.randint(1, max_batch_size))]

    input_cases = [
        ArgData(
            desc=ArgDesc(0, "F", "", input_layout),
            data=[get_random_batch() for _ in range(num_batches)],
        )
    ]

    def random_angle(sample_desc):
        return np.array(sample_desc.rng.uniform(-180.0, 180.0), dtype=np.float32)

    def random_axis(sample_desc):
        return np.array([sample_desc.rng.uniform(-1, 1) for _ in range(3)], dtype=np.float32)

    test_cases = [
        (dali.fn.rotate, {"angle": 45.0, "axis": np.array([1, 0, 0], dtype=np.float32)}, []),
        (
            dali.fn.rotate,
            {"size": (50, 30, 20)},
            [ArgCb("angle", random_angle, True), ArgCb("axis", random_axis, True)],
        ),
        (
            dali.fn.rotate,
            {},
            RotatePerFrameParamsProvider(
                [ArgCb("angle", random_angle, True), ArgCb("axis", random_axis, True)]
            ),
        ),
    ]
    yield from sequence_suite_helper(rng, input_cases, test_cases)
