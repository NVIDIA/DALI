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

import cv2
import functools
import math
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os.path
from nvidia.dali import Pipeline, pipeline_def
from nvidia.dali.data_node import DataNode as _DataNode
from test_utils import check_batch, get_dali_extra_path, as_array
from nose2.tools import params

import PIL.Image

try:
    from PIL.Image.Resampling import NEAREST, BILINEAR, BICUBIC, LANCZOS
except Exception:
    # Deprecated import, needed for Python 3.6
    from PIL.Image import NEAREST, BILINEAR, BICUBIC, LANCZOS

resample_dali2pil = {
    types.INTERP_NN: NEAREST,
    types.INTERP_TRIANGULAR: BILINEAR,
    types.INTERP_CUBIC: BICUBIC,
    types.INTERP_LANCZOS3: LANCZOS,
}

test_data_root = get_dali_extra_path()
db_2d_folder = os.path.join(test_data_root, "db", "lmdb")
db_3d_folder = os.path.join(test_data_root, *"db/3D/MRI/Knee/Jpegs/STU00001".split("/"))


class random_3d_loader:
    def __init__(self, batch_size):
        np.random.seed(12345)
        self.subdirs = ["SER00004", "SER00006", "SER00008", "SER00009", "SER00011", "SER00015"]
        self.dirs = [os.path.join(db_3d_folder, x) for x in self.subdirs]
        self.batch_size = batch_size
        np.random.seed(1234)
        self.n = 0
        self.order = list(range(len(self.subdirs)))
        np.random.shuffle(self.order)

    def __iter__(self):
        return self

    def __next__(self):
        return [self.get_one() for _ in range(self.batch_size)]

    def get_one(self):
        idx = self.get_index()
        dir = self.dirs[idx]

        imgs = []
        i = 0
        path = os.path.join(dir, "%i.jpg")
        while True:
            fname = path % i
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is None:
                break
            i += 1
            imgs.append(img[::-1, :, np.newaxis])
        return np.stack(imgs, axis=0)

    def get_index(self):
        if self.n >= len(self.order):
            np.random.shuffle(self.order)
            self.n = 0
        idx = self.order[self.n]
        self.n += 1
        return idx


def layout_str(dim, channel_first):
    s = "DHW" if dim == 3 else "HW"
    s = "C" + s if channel_first else s + "C"
    return s


def resize2D_PIL(input, size, roi_start, roi_end, dtype, channel_first, resample):
    if channel_first:
        input = input.transpose([1, 2, 0])
    size = list(reversed(size.astype(np.int32).tolist()))
    roi_start = reversed(roi_start.tolist())
    roi_end = reversed(roi_end.tolist())

    box = list(roi_start) + list(roi_end)

    has_overshoot = resample in (LANCZOS, BICUBIC)
    if has_overshoot:
        # compress dynamic range to allow for overshoot
        input = (64 + input * 0.5).round().astype(np.uint8)

    out = PIL.Image.fromarray(input).resize(size, box=box, resample=resample)
    out = np.array(out)

    if channel_first:
        out = out.transpose([2, 0, 1])
    if has_overshoot:
        out = (out.astype(np.float32) - 64) * 2.0
    if dtype == np.uint8:
        out = out.round().clip(0, 255).astype(np.uint8)
    elif dtype == types.FLOAT:
        out = out.astype(np.float32)
    return out


def resize3D_PIL(input, size, roi_start, roi_end, dtype, channel_first, resample):
    size = list(size)
    if channel_first:
        input = input.transpose([1, 2, 3, 0])

    has_overshoot = resample in (LANCZOS, BICUBIC)
    if has_overshoot:
        # compress dynamic range to allow for overshoot
        input = (64 + input * 0.5).round().astype(np.uint8)

    mono = input.shape[3] == 1

    # First, slice along Z dimension and resize the XY slices
    sizeXY = [size[2], size[1]]
    boxXY = [roi_start[2], roi_start[1], roi_end[2], roi_end[1]]
    tmp = np.zeros([input.shape[0], size[1], size[2], input.shape[3]], dtype=np.uint8)
    for z in range(input.shape[0]):
        in_slice = input[z, :, :, 0] if mono else input[z]
        out_slice = np.array(
            PIL.Image.fromarray(in_slice).resize(sizeXY, box=boxXY, resample=resample)
        )
        tmp[z] = out_slice[:, :, np.newaxis] if mono else out_slice

    # Then, slice along Y and resize XZ slices
    sizeXZ = [size[2], size[0]]
    boxXZ = [0, roi_start[0], size[2], roi_end[0]]
    out = np.zeros(size + [input.shape[3]], dtype=np.uint8)
    for y in range(size[1]):
        in_slice = tmp[:, y, :, 0] if mono else tmp[:, y]
        out_slice = np.array(
            PIL.Image.fromarray(in_slice).resize(sizeXZ, box=boxXZ, resample=resample)
        )
        out[:, y, :, :] = out_slice[:, :, np.newaxis] if mono else out_slice

    # Restore dynamic range, losing some bit depth
    if has_overshoot:
        out = (out.astype(np.float32) - 64) * 2.0
    if dtype == np.uint8:
        out = out.round().clip(0, 255).astype(np.uint8)
    elif dtype == types.FLOAT:
        out = out.astype(np.float32)
    if channel_first:
        out = out.transpose([3, 0, 1, 2])
    return out


def resize_PIL(dim, channel_first, dtype, interp, data, size, roi_start, roi_end):
    pil_resample = resample_dali2pil[interp]

    assert dtype == types.UINT8 or dtype == types.FLOAT
    dtype = np.uint8 if dtype == types.UINT8 else np.float32

    base_func = resize3D_PIL if dim == 3 else resize2D_PIL
    f = functools.partial(
        base_func, channel_first=channel_first, dtype=dtype, resample=pil_resample
    )

    return dali.fn.python_function(
        data, size, roi_start, roi_end, function=f, batch_processing=False
    )


def resize_op(backend):
    if backend in ["cpu", "gpu"]:
        return fn.resize
    elif backend == "cvcuda":
        return fn.experimental.resize
    else:
        assert False


def backend_device(backend):
    if backend in ["cvcuda", "gpu"]:
        return "gpu"
    elif backend == "cpu":
        return "cpu"
    else:
        assert False


def resize_dali(
    backend,
    input,
    channel_first,
    dtype,
    interp,
    mode,
    size,
    w,
    h,
    d,
    roi_start,
    roi_end,
    minibatch_size,
    max_size,
):
    return resize_op(backend)(
        input,
        interp_type=interp,
        dtype=dtype,
        mode=mode,
        resize_x=w,
        resize_y=h,
        resize_z=d,
        size=size,
        roi_start=roi_start,
        roi_end=roi_end,
        minibatch_size=minibatch_size,
        max_size=max_size,
        subpixel_scale=False,
    )  # disable subpixel scale so we can use PIL as reference
    # Note: PIL supports ROI, but, unlike DALI, does not support overscan. DALI routinely overscans
    # on a subpixel level in about half of the cases, when adjusting ROI to keep subpixel aspect
    # ratio. This precludes the use of PIL as reference for subpixel_scale.


def ref_output_size(mode, requested_size, roi_size, max_size=None):
    """Returns ideal (non-rounded) output size that would result from the parameters.
    The result is not rounded, so we can check the real, rounded, value against this one and find
    rounding errors by setting epsilon <1."""
    roi_size = list(roi_size)
    dim = len(roi_size)
    if max_size is None:
        max_size = [math.inf] * dim
    elif not isinstance(max_size, (list, tuple, np.ndarray)):
        max_size = [max_size] * dim
    elif isinstance(max_size, np.ndarray) and max_size.shape == []:
        max_size = [float(max_size)] * dim

    if not isinstance(requested_size, (list, tuple, np.ndarray)):
        requested_size = [requested_size] * dim
    elif isinstance(requested_size, np.ndarray) and requested_size.shape == []:
        requested_size = [float(requested_size)] * dim

    requested_size = [abs(x) if x else None for x in requested_size]
    roi_size = [abs(x) for x in roi_size]

    if not any(requested_size):
        return roi_size

    if mode == "stretch":
        return [min(m, o or i) for o, i, m in zip(requested_size, roi_size, max_size)]
    elif mode == "not_smaller":
        max_scale = 0
        for o, i in zip(requested_size, roi_size):
            if not o:
                continue
            max_scale = max(max_scale, abs(o / i))

        for i in range(len(roi_size)):
            max_scale = min(max_scale, max_size[i] / abs(roi_size[i]))

        return [x * max_scale for x in roi_size]
    elif mode == "not_larger":
        min_scale = math.inf
        for o, i, m in zip(requested_size, roi_size, max_size):
            if not o:
                min_scale = min(min_scale, m / i)
                continue
            min_scale = min(min_scale, abs(min(m, o) / i))

        return [x * min_scale for x in roi_size]
    elif mode == "default" or mode is None:
        avg_scale = 1
        power = 0
        for o, i in zip(requested_size, roi_size):
            if o:
                avg_scale *= abs(o / i)
                power += 1
        if power == len(requested_size):
            return [min(o, m) for o, m in zip(requested_size, max_size)]
        if power > 1:
            avg_scale = math.pow(avg_scale, 1 / power)
        out = [min(m, o or avg_scale * i) for o, i, m in zip(requested_size, roi_size, max_size)]
        return out
    else:
        raise ValueError("Invalid mode '{}'".format(mode))


def test_ref_size():
    r = ref_output_size("not_smaller", [600, 600], [640, 480], 720)
    assert r == [720, 540]
    r = ref_output_size("not_larger", [600, 500], [640, 480], 720)
    assert r == [600, 450]
    r = ref_output_size("stretch", [600, 500], [640, 480], [1000, 300])
    assert r == [600, 300]
    r = ref_output_size("default", [600, 0], [640, 480])
    assert r == [600, 450]
    r = ref_output_size("default", [0, 600], [640, 480])
    assert r == [800, 600]
    r = ref_output_size("default", [80, 0, 20], [10, 10, 10])
    assert r == [80, 40, 20]


def max_size(dim):
    return 200 if dim == 3 else None


def build_pipes(
    backend,
    dim,
    batch_size,
    channel_first,
    mode,
    interp,
    dtype,
    w_input,
    h_input,
    d_input,
    use_size_arg,
    use_size_input,
    use_roi,
):
    dali_pipe = Pipeline(
        batch_size=batch_size,
        num_threads=8,
        device_id=0,
        seed=12345,
        exec_async=False,
        exec_pipelined=False,
    )
    with dali_pipe:
        if dim == 2:
            files, labels = dali.fn.readers.caffe(path=db_2d_folder, random_shuffle=True)
            images_cpu = dali.fn.decoders.image(files, device="cpu")
        else:
            images_cpu = dali.fn.external_source(source=random_3d_loader(batch_size), layout="DHWC")

        images_hwc = images_cpu if backend_device(backend) == "cpu" else images_cpu.gpu()

        if channel_first:
            images = dali.fn.transpose(
                images_hwc, perm=[3, 0, 1, 2] if dim == 3 else [2, 0, 1], transpose_layout=True
            )
        else:
            images = images_hwc

        roi_start = None
        roi_end = None
        w = None
        h = None
        d = None
        size = None

        minibatch_size = 2 if dim == 3 else 8

        if use_roi:
            # Calculate absolute RoI
            in_size = fn.slice(
                images_cpu.shape(),
                types.Constant(0, dtype=types.FLOAT, device="cpu"),
                types.Constant(dim, dtype=types.FLOAT, device="cpu"),
                axes=[0],
                normalized_shape=False,
            )
            roi_start = fn.random.uniform(range=(0, 0.4), shape=[dim]) * in_size
            roi_end = fn.random.uniform(range=(0.6, 1.0), shape=[dim]) * in_size

        size_range = (10, 200) if dim == 3 else (10, 1000)

        if use_size_arg:
            if use_size_input:
                mask = fn.cast(fn.random.uniform(range=(0.8, 1.9), shape=[dim]), dtype=types.INT32)
                size = fn.random.uniform(range=size_range, shape=[dim]) * mask
            else:
                size = [300, 400] if dim == 2 else [80, 100, 120]

            resized = resize_dali(
                backend,
                images,
                channel_first,
                dtype,
                interp,
                mode,
                size,
                None,
                None,
                None,
                roi_start,
                roi_end,
                minibatch_size=minibatch_size,
                max_size=max_size(dim),
            )
        else:
            if w_input:
                has_w = fn.random.coin_flip(probability=0.8)
                w = fn.random.uniform(range=size_range) * has_w
            else:
                w = 320  # some fixed value

            if h_input:
                has_h = fn.random.coin_flip(probability=0.8)
                h = fn.random.uniform(range=size_range) * has_h
            else:
                h = 240  # some other fixed value

            if dim >= 3:
                if d_input:
                    has_d = fn.random.coin_flip(probability=0.8)
                    d = fn.random.uniform(range=size_range) * has_d
                else:
                    d = 31  # some other fixed value

            resized = resize_dali(
                backend,
                images,
                channel_first,
                dtype,
                interp,
                mode,
                None,
                w,
                h,
                d,
                roi_start,
                roi_end,
                minibatch_size=minibatch_size,
                max_size=max_size(dim),
            )

        outputs = [images, resized]
        if roi_start is not None and roi_end is not None:
            outputs += [roi_start, roi_end]

        for x in (d, h, w, size):
            if x is not None:
                if isinstance(x, _DataNode):
                    outputs.append(x)
                else:
                    outputs.append(types.Constant(np.array(x, dtype=np.float32)))

        dali_pipe.set_outputs(*outputs)

    pil_pipe = Pipeline(
        batch_size=batch_size, num_threads=8, device_id=0, exec_async=False, exec_pipelined=False
    )
    with pil_pipe:
        images = fn.external_source(name="images", layout=layout_str(dim, channel_first))
        sizes = fn.external_source(name="size")
        roi_start = fn.external_source(name="roi_start")
        roi_end = fn.external_source(name="roi_end")
        resized = resize_PIL(dim, channel_first, dtype, interp, images, sizes, roi_start, roi_end)
        resized = fn.reshape(resized, layout=layout_str(dim, channel_first))
        pil_pipe.set_outputs(resized)

    return dali_pipe, pil_pipe


def interior(array, channel_first):
    array = np.array(array)
    channel_dim = 0 if channel_first else len(array.shape) - 1
    r = []
    for d in range(len(array.shape)):
        if d == channel_dim or array.shape[d] <= 2:
            r.append(slice(array.shape[d]))
        else:
            r.append(slice(1, -1))
    return array[tuple(r)]


def _test_ND(
    backend,
    dim,
    batch_size,
    channel_first,
    mode,
    interp,
    dtype,
    w_input,
    h_input,
    d_input,
    use_size_arg,
    use_size_input,
    use_roi,
):
    dali_pipe, pil_pipe = build_pipes(
        backend,
        dim,
        batch_size,
        channel_first,
        mode,
        interp,
        dtype,
        w_input,
        h_input,
        d_input,
        use_size_arg,
        use_size_input,
        use_roi,
    )

    first_spatial_dim = 1 if channel_first else 0

    max_iters = 3

    for iter in range(max_iters):
        o = dali_pipe.run()
        output_idx = 0

        def get_outputs(n):
            nonlocal output_idx
            start = output_idx
            output_idx += n
            return o[start:output_idx]

        def get_output():
            return get_outputs(1)[0]

        dali_in, dali_out = get_outputs(2)
        if use_roi:
            roi_start, roi_end = (np.array(x.as_tensor(), dtype=np.float32) for x in get_outputs(2))
        else:
            roi_end = np.stack(
                [
                    dali_in[i].shape()[first_spatial_dim : first_spatial_dim + dim]
                    for i in range(batch_size)
                ]
            ).astype(np.float32)
            roi_start = np.zeros([batch_size, dim], dtype=np.float32)
        if use_size_arg:
            size = np.array(get_output().as_tensor(), np.float32)
        else:
            size = np.stack([x.as_tensor() for x in get_outputs(dim)], axis=1)

        roi_size = roi_end - roi_start

        dali_out_size = np.stack(
            [
                dali_out[i].shape()[first_spatial_dim : first_spatial_dim + dim]
                for i in range(batch_size)
            ]
        )

        for i in range(batch_size):
            ref_size = ref_output_size(mode, size[i], roi_size[i], max_size(dim))
            real_size = dali_out_size[i]
            max_err = np.max(np.abs(ref_size - real_size))
            eps = 0.6  # allow for rounding errors - we'll use _real_ size when resizing with PIL
            if max_err > eps:
                print("Invalid output size!")
                print(dali_out[i].shape())
                print("Got:      ", real_size)
                print("Expected: ", ref_size)
                print("RoI", roi_size[i])
                print("Input size", dali_in[i].shape())
                print("Requested output", size[i])
                assert max_err <= eps

        ref_in = dali_in.as_cpu()
        pil_pipe.feed_input("images", ref_in, layout=layout_str(dim, channel_first))
        pil_pipe.feed_input("size", dali_out_size)
        pil_pipe.feed_input("roi_start", roi_start)
        pil_pipe.feed_input("roi_end", roi_end)
        ref = pil_pipe.run()

        dali_resized = o[1].as_cpu()
        ref_resized = ref[0]

        max_avg_err = 0.6 if dim == 3 else 0.4
        max_err = 12 if dim == 3 else 10
        if interp == types.INTERP_LANCZOS3:
            max_err *= 2

        dali_interior = [interior(x, channel_first) for x in dali_resized]
        ref_interior = [interior(x, channel_first) for x in ref_resized]
        check_batch(dali_interior, ref_interior, batch_size, max_avg_err, max_err)


def _tests(dim, backend):
    batch_size = 2 if dim == 3 else 10
    # - Cannot test linear against PIL, because PIL uses triangular filter when downscaling
    # - Cannot test Nearest Neighbor because rounding errors cause gross discrepancies (pixel shift)
    for mode in ["default", "stretch", "not_smaller", "not_larger"]:
        for (
            interp,
            dtype,
            channel_first,
            use_size_arg,
            use_size_input,
            w_input,
            h_input,
            d_input,
            use_roi,
        ) in [
            (0, types.UINT8, True, False, False, False, False, False, False),
            (1, types.FLOAT, False, False, False, False, True, True, True),
            (0, types.FLOAT, True, False, False, True, True, False, True),
            (1, types.FLOAT, False, False, False, True, False, True, False),
            (0, types.UINT8, True, True, False, False, False, False, True),
            (1, types.UINT8, False, True, True, False, False, False, False),
        ]:
            interp = [types.INTERP_TRIANGULAR, types.INTERP_LANCZOS3][interp]
            yield (
                _test_ND,
                backend,
                dim,
                batch_size,
                channel_first,
                mode,
                interp,
                dtype,
                w_input,
                h_input,
                d_input,
                use_size_arg,
                use_size_input,
                use_roi,
            )


def test_2D_gpu():
    for f, *args in _tests(2, "gpu"):
        yield (f, *args)


def test_3D_gpu():
    for f, *args in _tests(3, "gpu"):
        yield (f, *args)


def test_2D_cpu():
    for f, *args in _tests(2, "cpu"):
        yield (f, *args)


def test_3D_cpu():
    for f, *args in _tests(3, "cpu"):
        yield (f, *args)


def test_2D_cvcuda():
    for f, *args in _tests(2, "cvcuda"):
        yield (f, *args)


def test_3D_cvcuda():
    for f, *args in _tests(3, "cvcuda"):
        yield (f, *args)


def _test_stitching(backend, dim, channel_first, dtype, interp):
    batch_size = 1 if dim == 3 else 10
    pipe = dali.pipeline.Pipeline(
        batch_size=batch_size, num_threads=1, device_id=0, seed=1234, prefetch_queue_depth=1
    )
    with pipe:
        if dim == 2:
            files, labels = dali.fn.readers.caffe(path=db_2d_folder, random_shuffle=True)
            images_cpu = dali.fn.decoders.image(files, device="cpu")
        else:
            images_cpu = dali.fn.external_source(source=random_3d_loader(batch_size), layout="DHWC")

        images_hwc = images_cpu if backend_device(backend) == "cpu" else images_cpu.gpu()

        if channel_first:
            images = dali.fn.transpose(
                images_hwc, perm=[3, 0, 1, 2] if dim == 3 else [2, 0, 1], transpose_layout=True
            )
        else:
            images = images_hwc

        out_size_full = [32, 32, 32] if dim == 3 else [160, 160]
        out_size_half = [x // 2 for x in out_size_full]

        roi_start = [0] * dim
        roi_end = [1] * dim

        resized = resize_op(backend)(
            images, dtype=dtype, min_filter=interp, mag_filter=interp, size=out_size_full
        )

        outputs = [resized]

        for z in range(dim - 1):
            if dim == 3:
                roi_start[0] = z * 0.5
                roi_end[0] = (z + 1) * 0.5
            for y in [0, 1]:
                roi_start[-2] = y * 0.5
                roi_end[-2] = (y + 1) * 0.5
                for x in [0, 1]:
                    roi_start[-1] = x * 0.5
                    roi_end[-1] = (x + 1) * 0.5

                    part = fn.resize(
                        images,
                        dtype=dtype,
                        interp_type=interp,
                        size=out_size_half,
                        roi_start=roi_start,
                        roi_end=roi_end,
                        roi_relative=True,
                    )
                    outputs.append(part)

        pipe.set_outputs(*outputs)

    for iter in range(1):
        out = pipe.run()
        if backend_device(backend) == "gpu":
            out = [x.as_cpu() for x in out]
        whole = out[0]
        tiled = []
        for i in range(batch_size):
            slices = []
            for z in range(dim - 1):
                q00 = out[1 + z * 4 + 0].at(i)
                q01 = out[1 + z * 4 + 1].at(i)
                q10 = out[1 + z * 4 + 2].at(i)
                q11 = out[1 + z * 4 + 3].at(i)
                if channel_first:
                    slices.append(np.block([[q00, q01], [q10, q11]]))
                else:
                    slices.append(np.block([[[q00], [q01]], [[q10], [q11]]]))
            if dim == 3:
                if channel_first:
                    tiled.append(np.block([[[slices[0]]], [[slices[1]]]]))
                else:
                    tiled.append(np.block([[[[slices[0]]]], [[[slices[1]]]]]))
            else:
                tiled.append(slices[0])
        max_err = 1e-3 if type == types.FLOAT else 1
        check_batch(tiled, whole, batch_size, 1e-4, max_err, compare_layouts=False)


def test_stitching():
    for backend in ["cpu", "gpu", "cvcuda"]:
        for dim in [3]:
            for dtype in [types.UINT8, types.FLOAT]:
                for channel_first in [False, True]:
                    for interp in [
                        types.INTERP_LINEAR,
                        types.INTERP_CUBIC,
                        types.INTERP_TRIANGULAR,
                        types.INTERP_LANCZOS3,
                    ]:
                        yield _test_stitching, backend, dim, channel_first, dtype, interp


def _test_empty_input(dim, backend):
    batch_size = 8
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=0, seed=1234)
    if dim == 2:
        files, labels = dali.fn.readers.caffe(path=db_2d_folder, random_shuffle=True)
        images_cpu = dali.fn.decoders.image(files, device="cpu")
    else:
        images_cpu = dali.fn.external_source(source=random_3d_loader(batch_size), layout="DHWC")

    images = images_cpu if backend_device(backend) == "cpu" else images_cpu.gpu()

    in_rel_shapes = np.ones([batch_size, dim], dtype=np.float32)

    in_rel_shapes[::2, :] *= 0  # all zeros in every second sample

    degenerate_images = fn.slice(
        images, np.zeros([dim]), fn.external_source(lambda: in_rel_shapes), axes=list(range(dim))
    )

    sizes = np.random.randint(20, 50, [batch_size, dim], dtype=np.int32)
    size_inp = fn.external_source(lambda: [x.astype(np.float32) for x in sizes])

    resize_no_empty = resize_op(backend)(images, size=size_inp, mode="not_larger")
    resize_with_empty = resize_op(backend)(degenerate_images, size=size_inp, mode="not_larger")

    pipe.set_outputs(resize_no_empty, resize_with_empty)

    for it in range(3):
        out_no_empty, out_with_empty = pipe.run()
        if backend_device(backend) == "gpu":
            out_no_empty = out_no_empty.as_cpu()
            out_with_empty = out_with_empty.as_cpu()
        for i in range(batch_size):
            if i % 2 != 0:
                assert np.array_equal(out_no_empty.at(i), out_with_empty.at(i))
            else:
                assert np.prod(out_with_empty.at(i).shape) == 0


def test_empty_input():
    for backend in ["cpu", "gpu", "cvcuda"]:
        for dim in [2, 3]:
            yield _test_empty_input, dim, backend


def _test_very_small_output(dim, backend):
    batch_size = 8
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=0, seed=1234)
    if dim == 2:
        files, labels = dali.fn.readers.caffe(path=db_2d_folder, random_shuffle=True)
        images_cpu = dali.fn.decoders.image(files, device="cpu")
    else:
        images_cpu = dali.fn.external_source(source=random_3d_loader(batch_size), layout="DHWC")

    images = images_cpu if backend_device(backend) == "cpu" else images_cpu.gpu()

    resize_tiny = resize_op(backend)(images, size=1e-10)

    pipe.set_outputs(resize_tiny)

    for it in range(3):
        (out,) = pipe.run()
        ref_size = [1, 1, 1, 1] if dim == 3 else [1, 1, 3]
        for t in out:
            assert t.shape() == ref_size


def test_very_small_output():
    for backend in ["cpu", "gpu", "cvcuda"]:
        for dim in [2, 3]:
            yield _test_very_small_output, dim, backend


large_data = None
large_data_resized = None


@params((types.INTERP_NN, False), (types.INTERP_LINEAR, False), (types.INTERP_LINEAR, True))
def test_large_gpu(interp, antialias):
    def make_cube(d, h, w):
        z = np.arange(d)[:, np.newaxis, np.newaxis]
        z = (z * 256 / z.size).astype(np.uint8)
        z = np.stack([z, np.zeros_like(z), np.zeros_like(z)], axis=3)
        y = np.arange(h)[np.newaxis, :, np.newaxis]
        y = (y * 256 / y.size).astype(np.uint8)
        y = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=3)
        x = np.arange(w)[np.newaxis, np.newaxis, :]
        x = (x * 256 / x.size).astype(np.uint8)
        x = np.stack([np.zeros_like(x), np.zeros_like(x), x], axis=3)
        return x + y + z

    global large_data
    if large_data is None:
        large_data = make_cube(350, 1080, 1920)

    @pipeline_def(num_threads=3, batch_size=1, device_id=0)
    def resize_pipe():
        ext = fn.external_source(source=[[large_data]], layout="DHWC", cycle=True, device="gpu")
        return fn.resize(
            ext, size=(350, 224, 224), device="gpu", interp_type=interp, antialias=antialias
        )

    pipe = resize_pipe()
    (outs,) = pipe.run()
    out = np.array(outs.at(0).as_cpu())
    global large_data_resized
    if large_data_resized is None:
        large_data_resized = make_cube(350, 224, 224)
    assert np.max(np.abs(out - large_data_resized)) < 2


@params(("cpu", 0), ("cpu", 1), ("gpu", 0), ("gpu", 1))
def test_nn_on_one_axis(device, axis):
    # Checks whether having NN interpolation in one axis and full resampling in the other works
    data = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
        ],
        dtype=np.float32,
    )

    # magnification is NN, minification is triangular
    ref = np.array(
        [
            [0, 0, 0.3333334, 0.3333334, 0, 0],
            [0, 0, 1.6666667, 1.6666667, 0, 0],
        ],
        dtype=np.float32,
    )

    if axis == 1:
        data = np.transpose(data, (1, 0))
        ref = np.transpose(ref, (1, 0))

    # add channel
    data = data[..., np.newaxis]
    ref = ref[..., np.newaxis]

    @pipeline_def(batch_size=1, device_id=0, num_threads=1)
    def test_pipe():
        src = dali.types.Constant(data, device=device)
        return fn.resize(
            src,
            size=ref.shape[:-1],
            min_filter=dali.types.INTERP_LINEAR,
            mag_filter=dali.types.INTERP_NN,
            antialias=True,
        )

    pipe = test_pipe()
    (out,) = pipe.run()
    check_batch(out, [ref], 1, 1e-5, 1e-5, None, False)


def test_checkerboard_dali_vs_onnx_ref():
    improc_data_dir = os.path.join(test_data_root, "db", "imgproc")
    ref_dir = os.path.join(improc_data_dir, "ref", "resampling")

    # Checker board with shape (22, 22) with 2x2 squares
    checkerboard_file = os.path.join(improc_data_dir, "checkerboard_22_22.npy")
    checkerboard = np.load(checkerboard_file)
    assert checkerboard.shape == (22, 22)

    out_size = (17, 13)
    out_size_str = "_".join([str(n) for n in out_size])
    ref_resized_linear_filename = os.path.join(ref_dir, f"checkerboard_linear_{out_size_str}.npy")
    ref_resized_linear_antialias_filename = os.path.join(
        ref_dir, f"checkerboard_linear_antialias_{out_size_str}.npy"
    )
    ref_resized_cubic_filename = os.path.join(ref_dir, f"checkerboard_cubic_{out_size_str}.npy")
    ref_resized_cubic_antialias_filename = os.path.join(
        ref_dir, f"checkerboard_cubic_antialias_{out_size_str}.npy"
    )

    # Reference generated with ONNX reference code. To regenerate uncomment
    # from onnx.backend.test.case.node.resize import interpolate_nd, linear_coeffs, \
    #     linear_coeffs_antialias, cubic_coeffs, cubic_coeffs_antialias
    #
    # ref_resized_linear = interpolate_nd(checkerboard, lambda x, _: linear_coeffs(x),
    #                                     output_size=out_size)
    # np.save(ref_resized_linear_filename, ref_resized_linear)
    # ref_resized_linear_antialias = interpolate_nd(checkerboard, linear_coeffs_antialias,
    #                                               output_size=out_size)
    # np.save(ref_resized_linear_antialias_filename, ref_resized_linear_antialias)
    # ref_resized_cubic = interpolate_nd(checkerboard, lambda x, _: cubic_coeffs(x, A=-0.5),
    #                                    output_size=out_size)
    # np.save(ref_resized_cubic_filename, ref_resized_cubic)
    # ref_resized_cubic_antialias = interpolate_nd(checkerboard,
    #                                              lambda x, scale: cubic_coeffs_antialias(x, scale,
    #                                                                                      A=-0.5),
    #                                              output_size=out_size)
    # np.save(ref_resized_cubic_antialias_filename, ref_resized_cubic_antialias)

    ref_resized_linear = np.load(ref_resized_linear_filename)
    assert ref_resized_linear.shape == out_size

    ref_resized_linear_antialias = np.load(ref_resized_linear_antialias_filename)
    assert ref_resized_linear_antialias.shape == out_size

    ref_resized_cubic = np.load(ref_resized_cubic_filename)
    assert ref_resized_cubic.shape == out_size

    ref_resized_cubic_antialias = np.load(ref_resized_cubic_antialias_filename)
    assert ref_resized_cubic_antialias.shape == out_size

    antialias_ON = True
    antialias_OFF = False
    ref_data = {
        types.INTERP_LINEAR: {
            antialias_OFF: ref_resized_linear,
            antialias_ON: ref_resized_linear_antialias,
        },
        types.INTERP_CUBIC: {
            antialias_OFF: ref_resized_cubic,
            antialias_ON: ref_resized_cubic_antialias,
        },
    }

    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe(device, interp_type, antialias, test_data=checkerboard, out_size=out_size):
        data = types.Constant(test_data, device=device)
        data = fn.expand_dims(data, axes=[2])
        resized = fn.resize(
            data,
            dtype=types.FLOAT,
            min_filter=interp_type,
            mag_filter=interp_type,
            size=out_size,
            antialias=antialias,
        )
        resized = fn.squeeze(resized, axes=[2])
        return resized

    def impl(device, interp_type, antialias):
        assert interp_type in ref_data
        ref = ref_data[interp_type][antialias]

        p = pipe(device, interp_type, antialias)
        (out,) = p.run()

        out_dali = as_array(out[0])
        abs_diff = np.abs(ref - out_dali)
        max_error = np.max(abs_diff)

        if max_error > 1:
            suffix_str = "cubic" if interp_type == types.INTERP_CUBIC else "linear"
            img1 = PIL.Image.fromarray(np.clip(ref, 0, 255).astype(np.uint8))
            img1.save(f"ref_resized_{suffix_str}.png")

            img2 = PIL.Image.fromarray(np.clip(out_dali, 0, 255).astype(np.uint8))
            img2.save(f"dali_resized_{suffix_str}.png")

            img2 = PIL.Image.fromarray(np.clip(127 + abs_diff, 0, 255).astype(np.uint8))
            img2.save(f"diff_resized_{suffix_str}.png")

        np.testing.assert_allclose(out_dali, ref, atol=1)

    for device in ["cpu", "gpu"]:
        for interp_type in [types.INTERP_LINEAR, types.INTERP_CUBIC]:
            for antialias in [antialias_OFF, antialias_ON]:
                yield impl, device, interp_type, antialias
