# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from scipy.signal import convolve2d


# note, it uses opencv's convention of naming the pattern after 2x2 tile
# that starts in the second row and column of the sensors' matrix
class BayerPattern:
    BGGR = 0
    GBRG = 1
    GRBG = 2
    RGGB = 3


bayer_patterns = [BayerPattern.BGGR, BayerPattern.GBRG, BayerPattern.GRBG, BayerPattern.RGGB]


def blue_position(pattern):
    assert 0 <= pattern <= 3
    return pattern // 2, pattern % 2


def blue_position2pattern(blue_position):
    y, x = blue_position
    assert 0 <= x <= 1 and 0 <= y <= 1
    return 2 * y + x


def rgb_bayer_masks(img_shape, pattern):
    h, w = img_shape
    assert h % 2 == 0 and w % 2 == 0, f"h: {h}, w: {w}"
    assert 0 <= pattern <= 3

    def sensor_matrix_00_is_green(pattern):
        return pattern in (BayerPattern.GRBG, BayerPattern.GBRG)

    def red_is_in_the_first_row(pattern):
        return pattern in (BayerPattern.BGGR, BayerPattern.GBRG)

    def vec(n, mod=2):
        return np.arange(0, n, dtype=np.uint8) % mod

    if sensor_matrix_00_is_green(pattern):
        top_right_mask = np.outer(1 - vec(h), vec(w))
        bottom_left_mask = np.outer(vec(h), 1 - vec(w))
        green = 1 - top_right_mask - bottom_left_mask
        if red_is_in_the_first_row(pattern):
            return top_right_mask, green, bottom_left_mask
        return bottom_left_mask, green, top_right_mask
    else:
        top_left_mask = np.outer(1 - vec(h), 1 - vec(w))
        bottom_right_mask = np.outer(vec(h), vec(w))
        green = 1 - top_left_mask - bottom_right_mask
        if red_is_in_the_first_row(pattern):
            return top_left_mask, green, bottom_right_mask
        return bottom_right_mask, green, top_left_mask


def rgb2bayer(img, pattern):
    h, w, c = img.shape
    assert c == 3
    h = h // 2 * 2
    w = w // 2 * 2
    r, g, b = rgb_bayer_masks((h, w), pattern)
    return img[:h, :w, 0] * r + img[:h, :w, 1] * g + img[:h, :w, 2] * b


def rgb2bayer_seq(seq, patterns):
    f, h, w, c = seq.shape
    assert f == len(patterns)
    assert c == 3
    h = h // 2 * 2
    w = w // 2 * 2
    bayer_masks = {pattern: rgb_bayer_masks((h, w), pattern) for pattern in bayer_patterns}
    seq_masks = [bayer_masks[pattern] for pattern in patterns]
    reds, greens, blues = [np.stack(channel) for channel in zip(*seq_masks)]
    return seq[:, :h, :w, 0] * reds + seq[:, :h, :w, 1] * greens + seq[:, :h, :w, 2] * blues


def conv2d_border101(img, filt):
    r, s = filt.shape
    assert r % 2 == 1 and s % 2 == 1
    padded = np.pad(img, ((r // 2, r // 2), (s // 2, s // 2)), "reflect")
    return convolve2d(padded, filt, mode="valid")


def conv2d_border101_seq(seq, filt):
    r, s = filt.shape
    assert r % 2 == 1 and s % 2 == 1
    padded = np.pad(seq, ((0, 0), (r // 2, r // 2), (s // 2, s // 2)), "reflect")
    debayered_frames = [convolve2d(frame, filt, mode="valid") for frame in padded]
    return np.stack(debayered_frames)


def debayer_bilinear_npp_masks(img, masks):
    """
    Computes the "bilinear with chroma correction for green channel" as
    defined by the NPP.
    """
    in_dtype = img.dtype
    ndim = len(img.shape)
    assert ndim in (2, 3)
    is_seq = ndim == 3
    conv = conv2d_border101 if not is_seq else conv2d_border101_seq
    red_mask, green_mask, blue_mask = masks
    red_signal = img * red_mask
    green_signal = img * green_mask
    blue_signal = img * blue_mask
    # When inferring red color for blue or green base, there are either
    # four red base neighbors at four corners or two base neighbors in
    # x or y axis. The blue color case is analogous.
    rb_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32)
    green_x_filter = np.array([[1, 0, 1]], dtype=np.int32)
    green_y_filter = green_x_filter.transpose()
    green_filter = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.int32)
    red = conv(red_signal, rb_filter) // 4
    blue = conv(blue_signal, rb_filter) // 4
    green_bilinear = conv(green_signal, green_filter) // 4
    green_x = conv(green_signal, green_x_filter) // 2
    green_y = conv(green_signal, green_y_filter) // 2

    def green_with_chroma_correlation(color_signal):
        # For red and blue based positions, there are always four
        # green neighbors (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1).
        # NPP does not simply average 4 of them to get green intensity.
        # Instead, it averages only two in either y or x axis as explained in
        # https://docs.nvidia.com/cuda/npp/group__image__color__debayer.html
        # I.e. the axis is chosen by looking at:
        # * abs(color(x, y), avg(color(y - 2, x), color(y + 2, x))) and
        # * abs(color(x, y), avg(color(y, x - 2), color(y, x - 2)))
        # and choosing the axis where the difference is smaller.
        # In other words, if we are inferring green color for blue(red)-based
        # position we check in which axis the blue(red) intensity changes less
        # and pick that axis.
        diff_filter_x = np.array([[1, 0, 0, 0, 1]], dtype=np.int32)
        diff_filter_y = diff_filter_x.transpose()
        # First compute the average, then the difference. Doing it with a single
        # conv yields different results (and as this servers as a mask,
        # it results in substantial differences in the end)
        x_avg = conv(color_signal, diff_filter_x) // 2
        y_avg = conv(color_signal, diff_filter_y) // 2
        diff_x = np.abs(color_signal - x_avg)
        diff_y = np.abs(color_signal - y_avg)
        return diff_x < diff_y, diff_x > diff_y

    pick_x_red_based, pick_y_red_based = green_with_chroma_correlation(red_signal)
    pick_x_blue_based, pick_y_blue_based = green_with_chroma_correlation(blue_signal)
    pick_x = pick_x_red_based + pick_x_blue_based
    pick_y = pick_y_red_based + pick_y_blue_based
    green = pick_x * green_x + pick_y * green_y + (1 - pick_x - pick_y) * green_bilinear
    return np.stack([red, green, blue], axis=ndim).astype(in_dtype)


def debayer_bilinear_npp_pattern(img, pattern):
    h, w = img.shape
    masks = rgb_bayer_masks((h, w), pattern)
    return debayer_bilinear_npp_masks(img, masks)


def debayer_bilinear_npp_pattern_seq(seq, patterns):
    f, h, w = seq.shape
    assert f == len(patterns)
    bayer_masks = {pattern: rgb_bayer_masks((h, w), pattern) for pattern in bayer_patterns}
    seq_masks = [bayer_masks[pattern] for pattern in patterns]
    reds, greens, blues = [np.stack(channel) for channel in zip(*seq_masks)]
    return debayer_bilinear_npp_masks(seq, (reds, greens, blues))


def debayer_opencv(img: np.ndarray, pattern: BayerPattern, algorithm: str):
    """Debayer image with OpenCV."""
    if pattern is not BayerPattern.BGGR:
        raise NotImplementedError("Only BGGR pattern is supported by at the moment.")
    if algorithm == "bilinear_ocv":
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB)
    if algorithm == "edgeaware_ocv":
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB_EA)
    if algorithm == "vng_ocv":
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB_VNG)
    if algorithm == "gray_ocv":
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2GRAY)[..., None]  # Make HWC
    raise ValueError(f"Unknown algorithm: {algorithm}")
