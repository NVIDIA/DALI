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

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn

import test_utils


def close(a, b):
    if isinstance(a, np.float32):
        return np.isclose(a, b)
    absdiff = a - b if b < a else b - a
    return absdiff <= 1


def analyze_frame(image, channel_dim):
    def pixel(x, y):
        return image[:, y, x] if channel_dim == 0 else image[y, x, :]

    x0, y0, f0 = pixel(0, 0)
    x1, y1, f1 = pixel(-1, 0)
    x2, y2, f2 = pixel(0, -1)
    x3, y3, f3 = pixel(-1, -1)
    assert close(x0, x2), "x0 = {} != x2 = {}".format(x0, x2)
    assert close(x1, x3), "x1 = {} != x3 = {}".format(x1, x3)
    assert close(y0, y1), "y0 = {} != y1 = {}".format(y0, y1)
    assert close(y2, y3), "y2 = {} != y3 = {}".format(y2, y3)
    assert close(f0, f1) and close(f0, f2) and close(f0, f3)
    return x0, y0, x3, y3, int(np.round(f0))


def check_frame(
    image,
    frame_index,
    total_frames,
    channel_dim,
    roi,
    w,
    h,
    aspect_ratio_range,
    area_range,
    value_range,
):
    x0, y0, x1, y1, f = analyze_frame(image, channel_dim)
    assert f == frame_index * value_range // total_frames
    out_h, out_w = image.shape[:2] if channel_dim != 0 else image.shape[1:3]
    xeps = np.ceil(2 + 2 * w / out_w)
    yeps = np.ceil(2 + 2 * h / out_h)
    if frame_index == 0:
        roi_w_max = min((x1 - x0) * w / value_range + xeps, w)
        roi_w_min = max((x1 - x0) * w / value_range - xeps, 1)
        roi_h_max = min((y1 - y0) * h / value_range + yeps, h)
        roi_h_min = max((y1 - y0) * h / value_range - xeps, 1)
        ratio_min = roi_w_min / roi_h_max
        ratio_max = roi_w_max / roi_h_min
        area_min = roi_w_min * roi_h_min / (w * h)
        area_max = roi_w_max * roi_h_max / (w * h)

        assert (
            ratio_max >= aspect_ratio_range[0] and ratio_min <= aspect_ratio_range[1]
        ), "aspect ratio estimated at {}..{} outside valid range [{} .. {}]".format(
            ratio_min, ratio_min, *aspect_ratio_range
        )
        assert (
            area_max >= area_range[0] and area_min <= area_range[1]
        ), "area estimated at {}..{} outside valiid range [{} .. {}]".format(
            area_min, area_max, *area_range
        )
        return x0, y0, x1, y1
    else:
        assert (x0, y0, x1, y1) == roi
        return roi


def check_seq(seq, channel_dim, w, h, aspect_ratio_range, area_range, value_range):
    frame_dim = 1 if channel_dim == 0 else 0
    frame_channel_dim = -1 if channel_dim == -1 else 0
    roi = None
    total_frames = seq.shape[frame_dim]
    for f in range(total_frames):
        frame = seq[:, f] if frame_dim == 1 else seq[f]
        roi = check_frame(
            frame,
            f,
            total_frames,
            frame_channel_dim,
            roi,
            w,
            h,
            aspect_ratio_range,
            area_range,
            value_range,
        )


def check_output(output, channel_dim, input_shape, aspect_ratio_range, area_range, value_range):
    if len(input_shape) == 3:
        h, w = input_shape[1:3] if channel_dim == 0 else input_shape[0:2]
        check_frame(
            output, 0, 1, channel_dim, None, w, h, aspect_ratio_range, area_range, value_range
        )
    else:
        hidx = 1 if channel_dim == -1 else 2
        h, w = input_shape[hidx : hidx + 2]
        check_seq(output, channel_dim, w, h, aspect_ratio_range, area_range, value_range)


def type_range(type):
    if np.issubdtype(type, np.integer):
        return np.iinfo(type).max
    else:
        return 100000


def generate_data(frames, width, height, channel_dim, type):
    value_range = type_range(type)
    no_frames = frames is None
    if no_frames:
        frames = 1
    x = (np.arange(0, width) * value_range // width).astype(type)[np.newaxis, np.newaxis, :]
    y = (np.arange(0, height) * value_range // height).astype(type)[np.newaxis, :, np.newaxis]
    f = (np.arange(0, frames) * value_range // frames).astype(type)[:, np.newaxis, np.newaxis]
    x = np.broadcast_to(x, (frames, height, width))
    y = np.broadcast_to(y, (frames, height, width))
    f = np.broadcast_to(f, (frames, height, width))
    seq = np.stack([x, y, f], axis=channel_dim)
    if no_frames:
        seq = seq[:, 0] if channel_dim == 0 else seq[0]
    return seq


def generator(batch_size, max_frames, channel_dim, type):
    type = test_utils.dali_type_to_np(type)
    assert max_frames is not None or channel_dim != 1

    def generate():
        batch = []
        for _ in range(batch_size):
            frames = None if max_frames is None else np.random.randint(1, max_frames + 1)
            sz = np.random.randint(100, 2000 / (max_frames or 1))
            w, h = np.random.randint(sz, 2 * sz, [2])
            batch.append(generate_data(frames, w, h, channel_dim, type))
        return batch

    return generate


def _test_rrc(
    device, max_frames, layout, aspect_ratio_range, area_range, output_size, input_type, output_type
):
    batch_size = 4
    pipe = dali.pipeline.Pipeline(batch_size, 4, 0)
    channel_dim = layout.find("C")
    value_range = type_range(test_utils.dali_type_to_np(input_type))
    if channel_dim == len(layout) - 1:
        channel_dim = -1
    input = fn.external_source(
        source=generator(batch_size, max_frames, channel_dim, input_type), layout=layout
    )
    shape = input.shape()
    if device == "gpu":
        input = input.gpu()
    out = fn.random_resized_crop(
        input,
        random_aspect_ratio=aspect_ratio_range,
        random_area=area_range,
        size=output_size,
        interp_type=dali.types.INTERP_LINEAR,
        antialias=False,
        seed=12321,
        dtype=output_type,
    )
    pipe.set_outputs(out, shape)
    for iter in range(3):
        outputs, input_shapes = pipe.run()
        if device == "gpu":
            outputs = outputs.as_cpu()
        assert outputs.layout() == layout
        for i in range(batch_size):
            out = outputs.at(i)
            input_shape = input_shapes.at(i).tolist()
            check_output(out, channel_dim, input_shape, aspect_ratio_range, area_range, value_range)


def test_random_resized_crop():
    np.random.seed(12345)
    types = [dali.types.UINT8, dali.types.INT16, dali.types.FLOAT]
    sizes = [(100, 100), (320, 240)]
    for device in ["cpu", "gpu"]:
        for layout, max_frames in [
            ("FHWC", 8),
            ("FCHW", 1),
            ("CFHW", 1),
            ("HWC", None),
            ("CHW", None),
        ]:
            for aspect, area in [
                ((0.5, 2), (0.1, 0.8)),
                ((1, 2), (0.4, 1.0)),
                ((0.5, 1), (0.1, 0.5)),
            ]:
                input_type = types[np.random.randint(0, len(types))]
                output_type = dali.types.FLOAT if np.random.randint(0, 2) else None
                size = sizes[np.random.randint(0, len(sizes))]
                yield (
                    _test_rrc,
                    device,
                    max_frames,
                    layout,
                    aspect,
                    area,
                    size,
                    input_type,
                    output_type,
                )
