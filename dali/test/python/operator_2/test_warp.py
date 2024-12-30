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

import nose_utils  # noqa: F401
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import math
import os
import cv2
from sequences_test_utils import video_suite_helper, SampleDesc, ArgCb
from test_utils import compare_pipelines
import random

test_data_root = os.environ["DALI_EXTRA_PATH"]
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")


def gen_transform(angle, zoom, dst_cx, dst_cy, src_cx, src_cy):
    t1 = np.array([[1, 0, -dst_cx], [0, 1, -dst_cy], [0, 0, 1]])
    cosa = math.cos(angle) / zoom
    sina = math.sin(angle) / zoom
    r = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    t2 = np.array([[1, 0, src_cx], [0, 1, src_cy], [0, 0, 1]])
    return (np.matmul(t2, np.matmul(r, t1)))[0:2, 0:3]


def gen_transforms(n, step):
    a = 0.0
    step = step * (math.pi / 180)
    out = np.zeros([n, 2, 3])
    for i in range(n):
        out[i, :, :] = gen_transform(a, 2, 160, 120, 100, 100)
        a = a + step
    return out.astype(np.float32)


def ToCVMatrix(matrix):
    offset = np.matmul(matrix, np.array([[0.5], [0.5], [1]]))
    result = matrix.copy()
    result[0][2] = offset[0] - 0.5
    result[1][2] = offset[1] - 0.5
    return result


def CVWarp(output_type, input_type, warp_matrix=None, inv_map=False):
    def warp_fn(img, matrix):
        size = (320, 240)
        matrix = ToCVMatrix(matrix)
        if output_type == types.FLOAT or input_type == types.FLOAT:
            img = np.float32(img)

        fill = 12.5 if output_type == types.FLOAT else 42
        out = cv2.warpAffine(
            img,
            matrix,
            size,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[fill, fill, fill],
            flags=((cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) if inv_map else cv2.INTER_LINEAR),
        )
        if output_type == types.UINT8 and input_type == types.FLOAT:
            out = np.uint8(np.clip(out, 0, 255))
        return out

    if warp_matrix:
        m = np.array(warp_matrix)

        def warp_fixed(img):
            return warp_fn(img, m)

        return warp_fixed

    return warp_fn


class WarpPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        output_type,
        input_type,
        use_input,
        num_threads=3,
        device_id=0,
        num_gpus=1,
        inv_map=False,
    ):
        super(WarpPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False
        )
        self.use_input = use_input
        self.use_dynamic_size = use_input  # avoid Cartesian product
        self.name = device
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        if input_type != types.UINT8:
            self.cast = ops.Cast(device=device, dtype=input_type)
        else:
            self.cast = None

        static_size = None if self.use_dynamic_size else (240, 320)

        fill = 12.5 if output_type == types.FLOAT else 42
        output_type_arg = output_type if output_type != input_type else None

        if use_input:
            self.transform_source = ops.ExternalSource(
                lambda: gen_transforms(self.max_batch_size, 10)
            )
            self.warp = ops.WarpAffine(
                device=device,
                size=static_size,
                fill_value=fill,
                dtype=output_type_arg,
                inverse_map=inv_map,
            )
        else:
            warp_matrix = (0.1, 0.9, 10, 0.8, -0.2, -20)
            self.warp = ops.WarpAffine(
                device=device,
                size=static_size,
                matrix=warp_matrix,
                fill_value=fill,
                dtype=output_type_arg,
                inverse_map=inv_map,
            )

        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.warp.device == "gpu":
            images = images.gpu()
        if self.cast:
            images = self.cast(images)

        dynamic_size = (
            types.Constant(np.array([240, 320], dtype=np.float32))
            if self.use_dynamic_size
            else None
        )

        if self.use_input:
            transform = self.transform_source()
            outputs = self.warp(images, transform, size=dynamic_size)
        else:
            outputs = self.warp(images, size=dynamic_size)
        return outputs


class CVPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        output_type,
        input_type,
        use_input,
        num_threads=3,
        device_id=0,
        num_gpus=1,
        inv_map=False,
    ):
        super(CVPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False
        )
        self.use_input = use_input
        self.name = "cv"
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        if self.use_input:
            self.transform_source = ops.ExternalSource(
                lambda: gen_transforms(self.max_batch_size, 10)
            )
            self.warp = ops.PythonFunction(
                function=CVWarp(output_type, input_type, inv_map=inv_map), output_layouts="HWC"
            )
        else:
            self.warp = ops.PythonFunction(
                function=CVWarp(
                    output_type, input_type, [[0.1, 0.9, 10], [0.8, -0.2, -20]], inv_map
                ),
                output_layouts="HWC",
            )
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.use_input:
            self.transform = self.transform_source()
            outputs = self.warp(images, self.transform)
        else:
            outputs = self.warp(images)
        return outputs


def compare(pipe1, pipe2, max_err):
    epoch_size = pipe1.epoch_size("Reader")
    batch_size = pipe1.max_batch_size
    niter = (epoch_size + batch_size - 1) // batch_size
    compare_pipelines(pipe1, pipe2, batch_size, niter, max_allowed_error=max_err)


io_types = [
    (types.UINT8, types.UINT8),
    (types.UINT8, types.FLOAT),
    (types.FLOAT, types.UINT8),
    (types.FLOAT, types.FLOAT),
]


def test_vs_cv():
    def impl(device, batch_size, use_input, otype, itype, inv_map):
        cv_pipeline = CVPipeline(batch_size, otype, itype, use_input, inv_map=inv_map)
        cpu_pipeline = WarpPipeline(device, batch_size, otype, itype, use_input, inv_map=inv_map)
        compare(cv_pipeline, cpu_pipeline, 8)

    random.seed(1009)
    for device in ["cpu", "gpu"]:
        for use_input in [False, True]:
            for itype, otype in io_types:
                inv_map = random.choice([False, True])
                batch_size = random.choice([1, 4, 19])
                yield impl, device, batch_size, use_input, otype, itype, inv_map


def test_gpu_vs_cpu():
    def impl(batch_size, use_input, otype, itype, inv_map):
        cpu_pipeline = WarpPipeline("cpu", batch_size, otype, itype, use_input, inv_map=inv_map)
        cpu_pipeline.build()
        gpu_pipeline = WarpPipeline("gpu", batch_size, otype, itype, use_input, inv_map=inv_map)
        gpu_pipeline.build()

    random.seed(1006)
    for use_input in [False, True]:
        for itype, otype in io_types:
            inv_map = random.choice([False, True])
            batch_size = random.choice([1, 4, 19])
            yield impl, batch_size, use_input, otype, itype, inv_map


def _test_extremely_large_data(device):
    in_size = 30000
    out_size = 10
    channels = 3

    def get_data():
        out = np.full([in_size, in_size, channels], 42, dtype=np.uint8)
        for c in range(channels):
            out[in_size - 1, in_size - 1, c] = c
        return [out]

    pipe = Pipeline(1, 3, 0, prefetch_queue_depth=1)
    input = fn.external_source(source=get_data, device=device)

    rotated = fn.warp_affine(
        input,
        matrix=[-1, 0, in_size, 0, -1, in_size],
        fill_value=255.0,
        size=[out_size, out_size],
        interp_type=types.INTERP_NN,
    )
    pipe.set_outputs(rotated)

    out = None
    try:
        (out,) = tuple(out.as_cpu() for out in pipe.run())
    except RuntimeError as e:
        if "bad_alloc" in str(e):
            print("Skipping test due to out-of-memory error:", e)
            return
        raise
    except MemoryError as e:
        print("Skipping test due to out-of-memory error:", e)
        return
    out = out.at(0)
    assert out.shape == (out_size, out_size, channels)
    for c in range(channels):
        assert out[0, 0, c] == c


def test_extremely_large_data():
    for device in ["cpu", "gpu"]:
        yield _test_extremely_large_data, device


def test_video():
    rng = random.Random(42)

    def random_flip_mx(sample_desc):
        x, y = sample_desc.rng.choice([(-1, -1), (1, -1), (-1, 1)])
        _, h, w, _ = sample_desc.sample.shape  # assuming FHWC layout
        return np.array(
            [[x, 0, 0 if x == 1 else w], [0, y, 0 if y == 1 else h], [0, 0, 1]], dtype=np.float32
        )

    def random_translate_mx(sample_desc):
        _, h, w, _ = sample_desc.sample.shape  # assuming FHWC layout
        return np.array(
            [
                [1, 0, sample_desc.rng.uniform(-w / 2, w / 2)],
                [0, 1, rng.uniform(-h / 2, h / 2)],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def random_scale_mx(sample_desc):
        def rand_scale():
            return sample_desc.rng.uniform(0.25, 4)

        return np.array([[rand_scale(), 0, 0], [0, rand_scale(), 0], [0, 0, 1]], dtype=np.float32)

    def random_rotate_mx(sample_desc):
        angle = math.radians(sample_desc.rng.uniform(-90, 90))
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def random_mx(sample_desc):
        m = np.eye(3, dtype=np.float32)
        for transformation in [
            random_flip_mx,
            random_translate_mx,
            random_scale_mx,
            random_rotate_mx,
        ]:
            if sample_desc.rng.choice([0, 1]):
                m = np.matmul(m, transformation(sample_desc))
        return m[0:2, :]

    def output_size(sample_desc):
        _, h, w, _ = sample_desc.sample.shape  # assuming FHWC layout
        rng = sample_desc.rng
        return np.array([h * rng.uniform(0.5, 2), w * rng.uniform(0.5, 2)], dtype=np.float32)

    video_test_cases = [
        (fn.warp_affine, {"matrix": random_rotate_mx(SampleDesc(rng, 0, 0, 0, None))[0:2, :]}, []),
        (fn.warp_affine, {}, [ArgCb("matrix", random_mx, False)]),
        (fn.warp_affine, {}, [ArgCb("matrix", random_mx, True)]),
        (
            fn.warp_affine,
            {},
            [ArgCb("matrix", random_mx, False), ArgCb("size", output_size, False)],
        ),
        (fn.warp_affine, {}, [ArgCb("matrix", random_mx, True), ArgCb("size", output_size, False)]),
        (fn.warp_affine, {}, [ArgCb(1, random_mx, True, dest_device="cpu")]),
        (fn.warp_affine, {}, [ArgCb(1, random_mx, True, dest_device="gpu")], ["gpu"]),
        (fn.warp_affine, {}, [ArgCb(1, random_mx, False, dest_device="cpu")]),
        (fn.warp_affine, {}, [ArgCb(1, random_mx, False, dest_device="gpu")], ["gpu"]),
    ]

    yield from video_suite_helper(
        video_test_cases, test_channel_first=False, expand_channels=False, rng=rng
    )
