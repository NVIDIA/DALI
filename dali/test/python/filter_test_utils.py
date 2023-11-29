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

import numpy as np

border2scipy_border = {
    "101": "mirror",
    "1001": "reflect",
    "clamp": "nearest",
    "wrap": "wrap",
    "constant": "constant",
}


def make_slice(start, end):
    return slice(start, end if end < 0 else None)


def scipy_baseline_plane(sample, kernel, anchor, border, fill_value, mode):
    from scipy.ndimage import convolve as sp_convolve

    ndim = len(sample.shape)
    assert len(kernel.shape) == ndim, f"{kernel.shape}, {ndim}"
    in_dtype = sample.dtype

    if isinstance(anchor, int):
        anchor = (anchor,) * ndim
    assert len(anchor) == ndim, f"{anchor}, {ndim}"
    anchor = tuple(
        filt_ext // 2 if anch == -1 else anch for anch, filt_ext in zip(anchor, kernel.shape)
    )
    for anch, filt_ext in zip(anchor, kernel.shape):
        assert 0 <= anch < filt_ext
    # there are two ways (and none exact) to center the even filter
    # over the image; scipy does it the other way round
    origin = tuple((filt_ext - 1) // 2 - anch for anch, filt_ext in zip(anchor, kernel.shape))

    out = sp_convolve(
        np.float32(sample),
        np.float32(np.flip(kernel)),
        mode=border2scipy_border[border],
        origin=origin,
        cval=0 if fill_value is None else fill_value,
    )

    if np.issubdtype(in_dtype, np.integer):
        type_info = np.iinfo(in_dtype)
        v_min, v_max = type_info.min, type_info.max
        out = np.clip(out, v_min, v_max)

    if mode == "valid":
        slices = tuple(
            make_slice(anch, anch - filt_ext + 1) for anch, filt_ext in zip(anchor, kernel.shape)
        )
        out = out[slices]

    return out.astype(in_dtype)


def filter_baseline(
    sample, kernel, anchor, border, fill_value=None, mode="same", has_channels=False
):
    assert mode in ("same", "valid"), f"{mode}"

    def baseline_call(plane):
        return scipy_baseline_plane(plane, kernel, anchor, border, fill_value, mode)

    ndim = len(sample.shape)
    if not has_channels:
        assert ndim in (2, 3)
        return baseline_call(sample)
    assert ndim in (3, 4)

    ndim = len(sample.shape)
    channel_dim = ndim - 1
    channel_first = sample.transpose([channel_dim] + [i for i in range(channel_dim)])
    out = np.stack([baseline_call(plane) for plane in channel_first], axis=channel_dim)
    return out


def filter_baseline_layout(layout, sample, kernel, anchor, border, fill_value=None, mode="same"):
    ndim = len(sample.shape)
    if not layout:
        assert ndim in (2, 3), f"{sample.shape}"
        layout = "HW" if ndim == 2 else "DHW"
    assert len(layout) == ndim, f"{layout}, {sample.shape}"
    has_channels = layout[ndim - 1] == "C"

    def baseline_call(plane):
        return filter_baseline(
            plane, kernel, anchor, border, fill_value, mode, has_channels=has_channels
        )

    def get_seq_ndim():
        for i, c in enumerate(layout):
            if c not in "FC":
                return i
        assert False

    seq_ndim = get_seq_ndim()
    if seq_ndim == 0:
        return baseline_call(sample)
    else:
        seq_shape = sample.shape[:seq_ndim]
        spatial_shape = sample.shape[seq_ndim:]
        seq_vol = np.prod(seq_shape)
        sample = sample.reshape((seq_vol,) + spatial_shape)
        out = np.stack([baseline_call(plane) for plane in sample])
        return out.reshape(seq_shape + out.shape[1:])
