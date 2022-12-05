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

import itertools
import os
import unittest

import numpy as np
from scipy.signal import convolve2d

from nvidia.dali import pipeline_def, fn, types
from test_utils import get_dali_extra_path
from nose_utils import assert_raises
from nose2.tools import params

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
vid_dir = os.path.join(data_root, "db", "video", "sintel", "video_files")
vid_files = ["sintel_trailer-720p_3.mp4"]


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


def read_imgs(num_imgs, dtype, seed):

    @pipeline_def
    def pipeline():
        input, _ = fn.readers.file(file_root=images_dir, random_shuffle=True, seed=seed)
        return fn.decoders.image(input, device="cpu", output_type=types.RGB)

    pipe = pipeline(batch_size=num_imgs, device_id=0, num_threads=4)
    pipe.build()
    (batch, ) = pipe.run()
    return [np.array(img, dtype=dtype) for img in batch]


def read_video(num_sequences, num_frames, height, width, seed=42):
    roi_start = (90, 0)
    roi_end = (630, 1280)
    vid_filenames = [os.path.join(vid_dir, vid_file) for vid_file in vid_files]

    @pipeline_def
    def pipeline():
        video = fn.readers.video_resize(filenames=vid_filenames, name='video reader',
                                        sequence_length=num_frames,
                                        file_list_include_preceding_frame=True, device='gpu',
                                        roi_start=roi_start, roi_end=roi_end, seed=seed,
                                        resize_x=width, resize_y=height)
        return video

    pipe = pipeline(batch_size=num_sequences, device_id=0, num_threads=4)
    pipe.build()
    (batch, ) = pipe.run()
    return [np.array(seq) for seq in batch.as_cpu()]


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


# Computes the "bilinear with chroma correction for green channel" as
# defined by the NPP.
def debayer_bilinear_npp_pattern(img, pattern):
    h, w = img.shape
    masks = rgb_bayer_masks((h, w), pattern)
    return debayer_bilinear_npp_masks(img, masks)


def debayer_bilinear_npp_masks(img, masks):
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
    # four red base neigbours at four corners or two base neigbours in
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
        # green neighbours (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1).
        # NPP does not simply avarage 4 of them to get green intensity.
        # Insted, it avarages only two in either y or x axis.
        # The axis is chosen by looking at:
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


def debayer_bilinear_npp_pattern_seq(seq, patterns):
    f, h, w = seq.shape
    assert f == len(patterns)
    bayer_masks = {pattern: rgb_bayer_masks((h, w), pattern) for pattern in bayer_patterns}
    seq_masks = [bayer_masks[pattern] for pattern in patterns]
    reds, greens, blues = [np.stack(channel) for channel in zip(*seq_masks)]
    return debayer_bilinear_npp_masks(seq, (reds, greens, blues))


def prepare_test_imgs(num_samples, dtype):
    assert dtype in (np.uint8, np.uint16)
    rng = np.random.default_rng(seed=101)
    imgs = read_imgs(num_samples, dtype, seed=42 if dtype == np.uint8 else 13)
    if dtype == np.uint16:
        imgs = [
            np.uint16(img) * 256 + np.uint16(rng.uniform(0, 256, size=img.shape)) for img in imgs
        ]
    bayered_imgs = {
        pattern: [rgb2bayer(img, pattern) for img in imgs]
        for pattern in bayer_patterns
    }
    npp_baseline = {
        pattern: [debayer_bilinear_npp_pattern(img, pattern) for img in imgs]
        for pattern, imgs in bayered_imgs.items()
    }
    return bayered_imgs, npp_baseline


class DebayerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.num_samples = 7
        cls.bayered_imgs, cls.npp_baseline = prepare_test_imgs(cls.num_samples, dtype=np.uint8)
        cls.bayered_imgs16t, cls.npp_baseline16t = prepare_test_imgs(cls.num_samples,
                                                                     dtype=np.uint16)

    @classmethod
    def get_test_data(cls, dtype):
        assert dtype in (np.uint8, np.uint16)
        if dtype == np.uint8:
            return cls.bayered_imgs, cls.npp_baseline
        else:
            return cls.bayered_imgs16t, cls.npp_baseline16t

    @params(*enumerate(itertools.product([1, 64], bayer_patterns)))
    def test_debayer_fixed_pattern(self, i, args):
        (batch_size, pattern) = args
        num_iterations = 3
        test_hwc_single_channel_input = i % 2 == 1
        bayered_imgs, npp_baseline = self.get_test_data(np.uint8)

        def source(sample_info):
            idx = sample_info.idx_in_epoch % self.num_samples
            img = bayered_imgs[pattern][idx]
            assert len(img.shape) == 2
            if test_hwc_single_channel_input:
                h, w = img.shape
                img = img.reshape(h, w, 1)
            return img, np.array(idx, dtype=np.int32)

        @pipeline_def
        def debayer_pipeline():
            bayer_imgs, idxs = fn.external_source(source=source, batch=False, num_outputs=2)
            debayered_imgs = fn.experimental.debayer(bayer_imgs.gpu(),
                                                     blue_position=blue_position(pattern))
            return debayered_imgs, idxs

        pipe = debayer_pipeline(batch_size=batch_size, device_id=0, num_threads=4)
        pipe.build()

        out_batches = []
        for _ in range(num_iterations):
            debayered_imgs_dev, idxs = pipe.run()
            assert debayered_imgs_dev.layout() == "HWC"
            out_batches.append(
                ([np.array(img)
                  for img in debayered_imgs_dev.as_cpu()], [np.array(idx) for idx in idxs]))

        for debayered_imgs, idxs in out_batches:
            assert len(debayered_imgs) == len(idxs)
            for img_debayered, idx in zip(debayered_imgs, idxs):
                baseline = npp_baseline[pattern][idx]
                assert np.all(img_debayered == baseline)

    @params(*itertools.product([1, 11, 184], [np.uint8, np.uint16]))
    def test_debayer_per_sample_pattern(self, batch_size, dtype):
        num_iterations = 3
        num_patterns = len(bayer_patterns)
        rng = np.random.default_rng(seed=42 + batch_size)
        bayered_imgs, npp_baseline = self.get_test_data(dtype)

        def source(sample_info):
            idx = sample_info.idx_in_epoch % self.num_samples
            pattern_idx = np.int32(rng.uniform(0, num_patterns))
            pattern = bayer_patterns[pattern_idx]
            return bayered_imgs[pattern][idx], \
                np.array(blue_position(pattern), dtype=np.int32), \
                np.array(idx, dtype=np.int32)

        @pipeline_def
        def debayer_pipeline():
            bayer_imgs, blue_poses, idxs = fn.external_source(source=source, batch=False,
                                                              num_outputs=3)
            debayered_imgs = fn.experimental.debayer(bayer_imgs.gpu(), blue_position=blue_poses)
            return debayered_imgs, blue_poses, idxs

        pipe = debayer_pipeline(batch_size=batch_size, device_id=0, num_threads=4)
        pipe.build()

        out_batches = []
        for _ in range(num_iterations):
            debayered_imgs_dev, blue_poses, idxs = pipe.run()
            assert debayered_imgs_dev.layout() == "HWC"
            out_batches.append(
                ([np.array(img) for img in debayered_imgs_dev.as_cpu()],
                 [blue_position2pattern(np.array(blue_pos))
                  for blue_pos in blue_poses], [np.array(idx) for idx in idxs]))

        for debayered_imgs, patterns, idxs in out_batches:
            assert len(debayered_imgs) == len(patterns) == len(idxs)
            for img_debayered, pattern, idx in zip(debayered_imgs, patterns, idxs):
                baseline = npp_baseline[pattern][idx]
                assert np.all(img_debayered == baseline)


class DebayerVideoTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(seed=3)
        num_smaller, num_bigger = 4, 3
        cls.num_samples = num_smaller + num_bigger
        smaller = read_video(num_smaller, 60, 108, 192)
        bigger = read_video(num_bigger, 32, 216, 384)
        video = smaller + bigger
        rng.shuffle(video)
        patterns = [rng.choice(bayer_patterns, len(vid)) for vid in video]
        cls.blue_poses = [
            np.array([blue_position(pattern) for pattern in sample_patterns], dtype=np.int32)
            for sample_patterns in patterns
        ]
        cls.bayered_vid = [
            rgb2bayer_seq(vid, vid_patterns) for vid, vid_patterns in zip(video, patterns)
        ]
        cls.npp_baseline = [
            debayer_bilinear_npp_pattern_seq(vid, vid_patterns)
            for vid, vid_patterns in zip(cls.bayered_vid, patterns)
        ]

    def test_debayer_vid_per_frame_pattern(self):
        num_iterations = 2
        batch_size = (self.num_samples + 1) // 2

        def source(sample_info):
            idx = sample_info.idx_in_epoch % self.num_samples
            vid = self.bayered_vid[idx]
            return vid, self.blue_poses[idx], np.array(idx, dtype=np.int32)

        @pipeline_def
        def debayer_pipeline():
            bayered_vid, blue_positions, idxs = fn.external_source(source=source, batch=False,
                                                                   num_outputs=3,
                                                                   layout=["FHW", None, None])
            debayered_vid = fn.experimental.debayer(bayered_vid.gpu(),
                                                    blue_position=fn.per_frame(blue_positions))
            return debayered_vid, idxs

        pipe = debayer_pipeline(batch_size=batch_size, device_id=0, num_threads=4)
        pipe.build()

        out_batches = []
        for _ in range(num_iterations):
            debayered_dev, idxs = pipe.run()
            assert debayered_dev.layout() == "FHWC"
            out_batches.append(
                ([np.array(vid) for vid in debayered_dev.as_cpu()], [np.array(idx)
                                                                     for idx in idxs]))

        for debayered_videos, idxs in out_batches:
            assert len(debayered_videos) == len(idxs)
            for vid_debayered, idx in zip(debayered_videos, idxs):
                baseline = self.npp_baseline[idx]
                assert np.all(vid_debayered == baseline)


def source_full_array(shape, dtype):

    def source(sample_info):
        return np.full(shape, sample_info.idx_in_epoch, dtype=dtype)

    return source


def _test_shape_pipeline(shape, dtype):

    @pipeline_def
    def pipeline():
        bayer_imgs = fn.external_source(source_full_array(shape, dtype), batch=False)
        return fn.experimental.debayer(bayer_imgs.gpu(), blue_position=[0, 0])

    pipe = pipeline(batch_size=8, num_threads=4, device_id=0)
    pipe.build()
    pipe.run()


def test_odd_size_error():
    with assert_raises(RuntimeError,
                       glob="The height and width of the image to debayer must be even"):
        _test_shape_pipeline((20, 15), np.uint8)


def test_too_many_channels():
    with assert_raises(RuntimeError,
                       glob=" The debayer operator expects grayscale (i.e. single channel) images"):
        _test_shape_pipeline((20, 40, 2), np.uint8)


def test_wrong_sample_dim():
    with assert_raises(RuntimeError,
                       glob="The number of dimensions 4 does not match any of the allowed"):
        _test_shape_pipeline((20, 5, 5, 5), np.uint8)


def test_no_blue_position_specified():
    with assert_raises(RuntimeError, glob="Not all required arguments were specified"):

        @pipeline_def
        def pipeline():
            bayer_imgs = fn.external_source(source_full_array((20, 20), np.uint8), batch=False)
            return fn.experimental.debayer(bayer_imgs.gpu())

        pipe = pipeline(batch_size=8, num_threads=4, device_id=0)
        pipe.build()
        pipe.run()


@params(((2, 2), ), ((1, 2), ), ((-1, 0), ))
def test_blue_position_outside_of_2x2_tile(blue_position):
    with assert_raises(RuntimeError, glob="The `blue_position` position must lie within 2x2 tile"):

        @pipeline_def
        def pipeline():
            bayer_imgs = fn.external_source(source_full_array((20, 20), np.uint8), batch=False)
            return fn.experimental.debayer(bayer_imgs.gpu(), blue_position=blue_position)

        pipe = pipeline(batch_size=8, num_threads=4, device_id=0)
        pipe.build()
        pipe.run()
