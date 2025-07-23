# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import glob
import io
import itertools

import numpy as np
from nvidia.dali import fn, pipeline_def, types
from test_utils import (
    compare_pipelines,
    create_sign_off_decorator,
    get_dali_extra_path,
    check_numba_compatibility_cpu,
    has_operator,
    restrict_platform,
    is_of_supported,
)
from nose2.tools import params, cartesian_params
from nose_utils import assert_raises, SkipTest, attr

# Test configuration
batch_size = 8
test_data_shape = [40, 60, 3]
test_data_layout = "HWC"
test_data_frames = 24
test_sequence_shape = [test_data_frames, 426, 240, 3]  # 240p video


stateless_signed_off = create_sign_off_decorator()


def tensor_list_to_array(tensor_list):
    return tensor_list.as_cpu().as_array()


# Check whether a given pipeline is stateless
def check_is_pipeline_stateless(pipeline_factory, iterations=10):
    args = {
        "batch_size": batch_size,
        "num_threads": 4,
        "device_id": 0,
        "exec_async": True,
        "exec_pipelined": True,
    }

    pipe = pipeline_factory(**args)
    for _ in range(iterations):
        pipe.run()

    # Compare a pipeline that was already used with a fresh one
    compare_pipelines(pipe, pipeline_factory(**args), batch_size, iterations)


# Provides the same random batch each time
class RandomBatch:
    def __init__(self, data_shape=test_data_shape, dtype=np.uint8):
        rng = np.random.default_rng(1234)
        self.batch = [
            rng.integers(255, size=data_shape, dtype=np.uint8).astype(dtype)
            for _ in range(batch_size)
        ]

    def __call__(self):
        return self.batch


# Provides the same random batch of bounding boxes each time
class RandomBoundingBoxBatch:
    def __init__(self):
        rng = np.random.default_rng(1234)

        def random_sample():
            left = rng.uniform(0, 1, size=1)
            top = rng.uniform(0, 1, size=1)
            right = rng.uniform(left, 1)
            bottom = rng.uniform(top, 1)
            return np.vstack([left, top, right, bottom]).astype(np.float32).T

        self.batch = [random_sample() for _ in range(batch_size)]

    def __call__(self):
        return self.batch


def move_to(tensor, device):
    return tensor.gpu() if device == "gpu" else tensor


def check_single_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout=test_data_layout, batch=True)
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_sequence_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(
            source=RandomBatch(data_shape=test_sequence_shape), layout="FHWC", batch=True
        )
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_signal_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(
            source=RandomBatch(data_shape=[30, 40], dtype=np.float32), layout="ft", batch=True
        )
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_1d_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(
            source=RandomBatch(data_shape=[100], dtype=np.float32), batch=True
        )
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_filepath_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        path_str = os.path.join(get_dali_extra_path(), "db/single/jpeg/100/swan-3584559_640.jpg")
        path = np.frombuffer(path_str.encode(), dtype=np.int8)
        return op(move_to(path, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_encoded_jpeg_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        img = os.path.join(get_dali_extra_path(), "db/single/jpeg/100/swan-3584559_640.jpg")
        jpegs, _ = fn.readers.file(files=[img], pad_last_batch=True)
        return op(move_to(jpegs, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_encoded_audio_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        wav = os.path.join(get_dali_extra_path(), "db/audio/wav/237-134500-0000.wav")
        audio, _ = fn.readers.file(files=[wav], pad_last_batch=True)
        return op(move_to(audio, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_bbox_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBoundingBoxBatch(), batch=True)
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_no_input(op, device, **kwargs):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        return op(device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
def test_stateful(device):
    assert_raises(
        AssertionError,
        check_single_input,
        fn.random.coin_flip,
        device,
        glob="Mean error: *, Min error: *, Max error: *"
        "Total error count: *, Tensor size: *"
        "Index in batch: 0",
    )


@params("cpu", "gpu")
@stateless_signed_off("rotate")
def test_rotate_stateless(device):
    check_single_input(fn.rotate, device, angle=40)


@params("cpu", "gpu")
@stateless_signed_off("resize")
def test_resize_stateless(device):
    check_single_input(fn.resize, device, resize_x=50, resize_y=50)


@params("cpu", "gpu")
@stateless_signed_off("experimental.tensor_resize")
def test_tensor_resize_stateless(device):
    check_single_input(fn.experimental.tensor_resize, device, axes=[0, 1], sizes=[40, 40])


@params("cpu", "gpu")
@stateless_signed_off("flip")
def test_flip_stateless(device):
    check_single_input(fn.flip, device)


@params("cpu", "gpu")
@stateless_signed_off("crop")
def test_crop_stateless(device):
    check_single_input(fn.crop, device, crop=(20, 20))


@params("cpu", "gpu")
@stateless_signed_off("crop_mirror_normalize")
def test_crop_mirror_normalize_stateless(device):
    check_single_input(fn.crop_mirror_normalize, device, crop=(20, 20), mirror=True)


@params("cpu", "gpu")
@stateless_signed_off("warp_affine")
def test_warp_affine_stateless(device):
    check_single_input(fn.warp_affine, device, matrix=(0.1, 0.9, 10, 0.8, -0.2, -20))


@params("cpu", "gpu")
@stateless_signed_off("color_twist")
def test_color_twist_stateless(device):
    check_single_input(
        fn.color_twist,
        device,
        brightness=1.0,
        contrast=0.5,
        hue=90,
        saturation=1.2,
    )


@params("cpu", "gpu")
@stateless_signed_off("hsv")
def test_hsv_stateless(device):
    check_single_input(
        fn.hsv,
        device,
        hue=70,
        value=1.8,
        saturation=1.2,
    )


@params("cpu", "gpu")
@stateless_signed_off("hue")
def test_hue_stateless(device):
    check_single_input(fn.hue, device, hue=-90)


@params("cpu", "gpu")
@stateless_signed_off("saturation")
def test_saturation_stateless(device):
    check_single_input(fn.saturation, device)


@params("cpu", "gpu")
@stateless_signed_off("brightness_contrast", "brightness", "contrast")
def test_brightness_contrast_stateless(device):
    check_single_input(fn.brightness_contrast, device, brightness=0.7, contrast=1.7)


@params("cpu", "gpu")
@stateless_signed_off("reductions.min")
def test_reductions_min_stateless(device):
    check_single_input(fn.reductions.min, device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.max")
def test_reductions_max_stateless(device):
    check_single_input(fn.reductions.max, device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.sum")
def test_reductions_sum_stateless(device):
    check_single_input(fn.reductions.sum, device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.mean")
def test_reductions_mean_stateless(device):
    check_single_input(fn.reductions.mean, device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.mean_square")
def test_reductions_mean_square_stateless(device):
    check_single_input(fn.reductions.mean_square, device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.rms")
def test_reductions_rms_stateless(device):
    check_single_input(fn.reductions.rms, device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.std_dev")
def test_reductions_std_dev_stateless(device):
    check_single_input(lambda x, **kwargs: fn.reductions.std_dev(x, 0.0, **kwargs), device)


@params("cpu", "gpu")
@stateless_signed_off("reductions.variance")
def test_reductions_variance_stateless(device):
    check_single_input(lambda x, **kwargs: fn.reductions.variance(x, 5.0, **kwargs), device)


@params("cpu", "gpu")
@stateless_signed_off("experimental.equalize")
def test_equalize_stateless(device):
    check_single_input(fn.experimental.equalize, device)


@stateless_signed_off("transforms.crop")
def test_transforms_crop_stateless():
    check_no_input(fn.transforms.crop, "cpu")


@stateless_signed_off("transforms.rotation")
def test_transforms_rotation_stateless():
    check_no_input(fn.transforms.rotation, "cpu", angle=35)


@stateless_signed_off("transforms.shear")
def test_transforms_shear_stateless():
    check_no_input(fn.transforms.shear, "cpu", shear=(2, 2))


@stateless_signed_off("transforms.scale")
def test_transforms_scale_stateless():
    check_no_input(fn.transforms.scale, "cpu", scale=(3, 2))


@stateless_signed_off(
    "transforms.translation", "hidden.transform_translation", "transform_translation"
)
def test_transforms_translation_stateless():
    check_no_input(fn.transforms.translation, "cpu", offset=(4, 3))


@params("cpu", "gpu")
@stateless_signed_off("coord_transform")
def test_coord_transform(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch((5, 2)), layout="NX")
        if device == "gpu":
            data = data.gpu()
        return fn.coord_transform(data, M=(0.1, 0.9, 10, 0.8, -0.2, -20))

    check_is_pipeline_stateless(pipeline_factory)


@stateless_signed_off("transforms.combine")
def test_transforms_combine_stateless():
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        scale = fn.transforms.scale(scale=(3, 2))
        shear = fn.transforms.shear(shear=(2, 2))
        return fn.transforms.combine(scale, shear)

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("one_hot")
def test_one_hot_stateless(device):
    check_single_input(fn.one_hot, device)


@stateless_signed_off("experimental.median_blur")
def test_median_bluer_stateless():
    check_single_input(fn.experimental.median_blur, "gpu")


@params("cpu", "gpu")
@stateless_signed_off("erase")
def test_erase_stateless(device):
    check_single_input(fn.erase, device, anchor=(3, 4), shape=(5, 6))


@params("cpu", "gpu")
@stateless_signed_off("pad")
def test_pad_stateless(device):
    check_single_input(fn.pad, device, shape=(100, 100, 3))


@params("cpu", "gpu")
@stateless_signed_off("constant", "hidden.constant")
def test_constant_stateless(device):
    check_no_input(fn.constant, device, idata=[1, 2, 3])


@params("cpu", "gpu")
@stateless_signed_off("reshape", "reinterpret")
def test_reshape_stateless(device):
    check_single_input(fn.reshape, device, shape=[1, -1])


@params("cpu", "gpu")
@stateless_signed_off("lookup_table")
def test_lookup_table_stateless(device):
    check_single_input(fn.lookup_table, device, keys=[0], values=[1], default_value=123)


@params("cpu", "gpu")
@stateless_signed_off("transpose")
def test_transpose_stateless(device):
    check_single_input(fn.transpose, device, perm=[2, 0, 1])


@params("cpu", "gpu")
@stateless_signed_off("paste")
def test_paste_stateless(device):
    check_single_input(fn.paste, device, fill_value=0, ratio=2.0)


@params("cpu", "gpu")
@stateless_signed_off("laplacian")
def test_laplacian_stateless(device):
    check_single_input(fn.laplacian, device, window_size=3)


@params("cpu", "gpu")
@stateless_signed_off("gaussian_blur")
def test_gaussian_blur_stateless(device):
    check_single_input(fn.gaussian_blur, device, window_size=3)


@params("cpu", "gpu")
@stateless_signed_off("water")
def test_water_stateless(device):
    check_single_input(fn.water, device)


@params("cpu", "gpu")
@stateless_signed_off("sphere")
def test_sphere_stateless(device):
    check_single_input(fn.sphere, device)


@params("cpu", "gpu")
@stateless_signed_off("experimental.filter")
def test_filter_stateless(device):
    check_single_input(
        lambda x, **kwargs: fn.experimental.filter(x, np.full((3, 3), 1 / 9), **kwargs),
        device,
    )


@stateless_signed_off("experimental.remap")
def test_remap_stateless():
    rng = np.random.default_rng(42)
    np_map_x = 128 * rng.uniform(size=(100, 128))
    np_map_y = 100 * rng.uniform(size=(100, 128))

    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch((100, 128, 3)), layout="HWC")
        data = data.gpu()
        map_x = types.Constant(np_map_x).gpu()
        map_y = types.Constant(np_map_y).gpu()
        return fn.experimental.remap(data, map_x, map_y)

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("experimental.debayer")
def test_debayer_stateless(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch((40, 40)), layout="HW", batch=True)
        if device == "gpu":
            data = data.gpu()
        return fn.experimental.debayer(data, blue_position=[0, 0])

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("copy")
def test_copy_stateless(device):
    check_single_input(fn.copy, device)


@params("cpu", "gpu")
@stateless_signed_off("color_space_conversion")
def test_color_space_conversion_stateless(device):
    check_single_input(
        fn.color_space_conversion,
        device,
        image_type=types.DALIImageType.RGB,
        output_type=types.DALIImageType.YCbCr,
    )


@params("cpu", "gpu")
@stateless_signed_off("resize_crop_mirror", "fast_resize_crop_mirror")
def test_resize_crop_mirror_stateless(device):
    check_single_input(fn.resize_crop_mirror, device, size=(35, 55), crop=(20, 20), mirror=True)


@params("cpu", "gpu")
@stateless_signed_off("slice")
def test_slice_stateless(device):
    check_single_input(fn.slice, device, rel_start=(0.25, 0.25), rel_end=(0.75, 0.75))


@params("cpu", "gpu")
@stateless_signed_off("shapes")
def test_shapes_stateless(device):
    check_single_input(fn.shapes, device)


@params("cpu", "gpu")
@stateless_signed_off("per_frame")
def test_per_frame_stateless(device):
    check_single_input(fn.per_frame, device, replace=True)


@params("cpu", "gpu")
@stateless_signed_off("get_property")
def test_get_property_stateless(device):
    check_single_input(fn.get_property, device, key="layout")


@params("cpu", "gpu")
@stateless_signed_off("jpeg_compression_distortion")
def test_jpeg_compression_distortion_stateless(device):
    check_single_input(fn.jpeg_compression_distortion, device)


@params("cpu", "gpu")
@stateless_signed_off("multi_paste")
def test_multi_paste_stateless(device):
    check_single_input(
        fn.multi_paste, device, in_ids=list(range(batch_size)), output_size=[100, 100]
    )


@params("cpu", "gpu")
@stateless_signed_off("grid_mask")
def test_grid_mask_stateless(device):
    check_single_input(fn.grid_mask, device)


@params("cpu", "gpu")
@stateless_signed_off("preemphasis_filter")
def test_preemphasis_filter_stateless(device):
    check_single_input(fn.preemphasis_filter, device)


@stateless_signed_off("optical_flow")
def test_optical_flow_stateless():
    if not is_of_supported():
        raise SkipTest("Optical Flow is not supported on this platform")
    check_single_sequence_input(fn.optical_flow, "gpu")


@params("cpu", "gpu")
@stateless_signed_off("sequence_rearrange")
def test_sequence_rearrange_stateless(device):
    check_single_sequence_input(
        fn.sequence_rearrange, device, new_order=list(range(test_data_frames))
    )


@params("cpu", "gpu")
@stateless_signed_off("spectrogram")
def test_spectrogram_stateless(device):
    check_single_1d_input(fn.spectrogram, device)


@stateless_signed_off("power_spectrum")
def test_power_spectrum_stateless():
    check_single_signal_input(fn.power_spectrum, "cpu")


@params("cpu", "gpu")
@stateless_signed_off("dump_image")
def test_dump_image_stateless(device):
    suffix = "test_dump_image_stateless_tmp"
    check_single_input(fn.dump_image, device, suffix=suffix)
    for f in glob.glob(f"*-{suffix}-*.ppm"):
        os.remove(f)


@params("cpu", "gpu")
@stateless_signed_off("normalize")
def test_normalize_stateless(device):
    check_single_input(fn.normalize, device)


@params("cpu", "gpu")
@stateless_signed_off("mel_filter_bank")
def test_mel_filter_bank_stateless(device):
    check_single_signal_input(fn.mel_filter_bank, device)


@params("cpu", "gpu")
@stateless_signed_off("mfcc")
def test_mfcc_stateless(device):
    check_single_signal_input(fn.mfcc, device)


@params("cpu", "gpu")
@stateless_signed_off("nonsilent_region")
def test_nonsilent_region_stateless(device):
    check_single_1d_input(lambda *args, **kwargs: fn.nonsilent_region(*args, **kwargs)[0], device)


@params("cpu", "gpu")
@stateless_signed_off("audio_resample", "experimental.audio_resample")
def test_audio_resample_stateless(device):
    check_single_signal_input(fn.audio_resample, device, scale=0.5)


@params("cpu", "gpu")
@stateless_signed_off("element_extract")
def test_element_extract_stateless(device):
    check_single_sequence_input(fn.element_extract, device, element_map=[0])


@stateless_signed_off("bbox_paste")
def test_bbox_paste_stateless():
    check_single_bbox_input(fn.bbox_paste, "cpu", ratio=2)


@params("cpu", "gpu")
@stateless_signed_off("bb_flip")
def test_bb_flip_stateless(device):
    check_single_bbox_input(fn.bb_flip, device, ltrb=True)


@params("cpu", "gpu")
@stateless_signed_off("to_decibels")
def test_to_decibels_stateless(device):
    check_single_signal_input(fn.to_decibels, device)


@cartesian_params(("cpu", "gpu"), (fn.stack, fn.cat))
@stateless_signed_off("stack", "cat")
def test_tensor_join_stateless(device, join):
    def wrapper(x, **kwargs):
        return join(x, x, x, **kwargs)

    check_single_input(wrapper, device)


@params("cpu", "gpu")
@stateless_signed_off("tensor_subscript", "hidden.tensor_subscript")
def test_tensor_subscript_stateless(device):
    check_single_input(lambda x, **kwargs: x[0, :, 2:4:-1], device)


@params("cpu", "gpu")
@stateless_signed_off("subscript_dim_check", "hidden.subscript_dim_check")
def test_subscript_dim_check(device):
    check_single_input(lambda x, **kwargs: x[:], device)


@params("cpu", "gpu")
@stateless_signed_off("expand_dims")
def test_expand_dims(device):
    check_single_input(fn.expand_dims, device, axes=[0])


@params("cpu", "gpu")
@stateless_signed_off("squeeze")
def test_squeeze(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch((40, 1, 50, 1)), layout="DHWC")
        data = move_to(data, device)
        return fn.squeeze(data, axis_names="HC")

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("permute_batch")
def test_permute_batch_stateless(device):
    def wrapper(x, **kwargs):
        return fn.permute_batch(x, indices=[0] * batch_size, **kwargs)

    check_single_input(wrapper, device)


@stateless_signed_off("segmentation.select_masks")
def test_select_masks_stateless():
    n = 10
    polygons = np.asarray([[i, 0, i] for i in range(n)])
    vertices = np.asarray([[i, i + 1] for i in range(n)])
    mask_ids = np.asarray([i for i in range(n) if i % 2 == 0])

    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        return tuple(fn.segmentation.select_masks(mask_ids, polygons, vertices))

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("box_encoder")
def test_box_encoder_stateless(device):
    n = 10
    boxes = np.asarray([[float(i), float(i), float(i + 1), float(i + 1)] for i in range(n)])
    labels = np.asarray(list(range(n)))
    anchors = [float(i) for i in range(4)]

    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        return tuple(fn.box_encoder(boxes, labels, anchors=anchors, device=device))

    check_is_pipeline_stateless(pipeline_factory)


@attr("cupy")
@params("cpu", "gpu")
@stateless_signed_off("python_function")
def test_python_function_stateless(device):
    def wrapper(x, **kwargs):
        return fn.python_function(x, function=lambda x: x * 2, **kwargs)

    check_single_input(wrapper, device)


@params("cpu", "gpu")
@stateless_signed_off("dl_tensor_python_function")
def test_dl_tensor_python_function_stateless(device):
    def wrapper(x, **kwargs):
        return fn.dl_tensor_python_function(x, function=lambda x: x, **kwargs)

    check_single_input(wrapper, device)


@attr("numba")
@stateless_signed_off("experimental.numba_function")
def test_numba_function_stateless():
    import nvidia.dali.plugin.numba as dali_numba

    check_numba_compatibility_cpu()

    def double_sample(out_sample, in_sample):
        out_sample[:] = 2 * in_sample[:]

    @pipeline_def(batch_size=2, device_id=0, num_threads=4, enable_checkpointing=True)
    def numba_pipe():
        forty_two = fn.external_source(
            source=lambda x: np.full((2,), 42, dtype=np.uint8), batch=False
        )
        out = dali_numba.fn.experimental.numba_function(
            forty_two,
            run_fn=double_sample,
            out_types=[types.DALIDataType.UINT8],
            in_types=[types.DALIDataType.UINT8],
            outs_ndim=[1],
            ins_ndim=[1],
            batch_processing=False,
        )
        return out

    check_is_pipeline_stateless(numba_pipe)


@has_operator("experimental.inflate")
@restrict_platform(min_compute_cap=6.0)
@stateless_signed_off("experimental.inflate")
def test_inflate_stateless():
    import lz4.block

    def sample_to_lz4(sample):
        deflated_buf = lz4.block.compress(sample, store_size=False)
        return np.frombuffer(deflated_buf, dtype=np.uint8)

    source = RandomBatch()
    batch = source()
    input_data = [sample_to_lz4(sample) for sample in batch]

    input_shape = [np.array(sample.shape, dtype=np.int32) for sample in batch]

    @pipeline_def(enable_checkpointing=True)
    def pipeline():
        deflated = fn.external_source(source=itertools.repeat(input_data))
        shape = fn.external_source(source=itertools.repeat(input_shape))
        return fn.experimental.inflate(deflated.gpu(), shape=shape)

    check_is_pipeline_stateless(pipeline)


@stateless_signed_off("peek_image_shape")
def test_peek_image_shape_stateless():
    check_single_encoded_jpeg_input(fn.peek_image_shape, "cpu")


@stateless_signed_off("experimental.peek_image_shape")
def test_imgcodec_peek_image_shape_stateless():
    check_single_encoded_jpeg_input(fn.experimental.peek_image_shape, "cpu")


@stateless_signed_off("decoders.audio", "audio_decoder")
def test_audio_decoder_stateless():
    def audio_decoder_wrapper(*args, **kwargs):
        return fn.decoders.audio(*args, **kwargs)[0]

    check_single_encoded_audio_input(audio_decoder_wrapper, "cpu")


@params("cpu", "mixed")
@stateless_signed_off("decoders.image", "image_decoder")
def test_image_decoder_stateless(device):
    check_single_encoded_jpeg_input(fn.decoders.image, device)


@stateless_signed_off("decoders.numpy")
def test_numpy_decoder_stateless():

    class RandomEncode(RandomBatch):
        def encode_sample(self, data):
            buff = io.BytesIO()
            np.save(buff, data)
            buff.seek(0)
            return np.frombuffer(buff.read(), dtype=np.uint8)

        def __call__(self):
            data = super().__call__()
            return [self.encode_sample(sample) for sample in data]

    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        enc_data = fn.external_source(source=RandomEncode())
        dec_data = fn.decoders.numpy(enc_data)
        return dec_data

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "mixed")
@stateless_signed_off("experimental.decoders.image")
def test_experimental_image_decoder_stateless(device):
    check_single_encoded_jpeg_input(fn.experimental.decoders.image, device)


@params("cpu", "mixed")
@stateless_signed_off("decoders.image_crop", "image_decoder_crop")
def test_image_decoder_crop_stateless(device):
    check_single_encoded_jpeg_input(fn.decoders.image_crop, device, crop=(20, 50))


@params("cpu", "mixed")
@stateless_signed_off("experimental.decoders.image_crop")
def test_experimental_image_decoder_crop_stateless(device):
    check_single_encoded_jpeg_input(fn.experimental.decoders.image_crop, device, crop=(20, 50))


@params("cpu", "mixed")
@stateless_signed_off("decoders.image_slice", "image_decoder_slice")
def test_image_decoder_slice_stateless(device):
    check_single_encoded_jpeg_input(fn.decoders.image_slice, device, start=(5, 5), end=(45, 45))


@params("cpu", "mixed")
@stateless_signed_off("experimental.decoders.image_slice")
def test_experimental_image_decoder_slice_stateless(device):
    check_single_encoded_jpeg_input(
        fn.experimental.decoders.image_slice, device, start=(5, 5), end=(45, 45)
    )


@params("cpu", "gpu")
@stateless_signed_off("coord_flip")
def test_coord_flip_stateless(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        input = np.array([[1], [2], [3]], dtype=np.float32)
        return fn.coord_flip(input, flip_x=True, center_x=0, device=device)

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("cast_like")
def test_cast_like_stateless(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        return fn.cast_like(
            np.array([1, 2, 3], dtype=np.int32), np.array([1.0], dtype=np.float32), device=device
        )

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off("cast")
def test_cast_stateless(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        return fn.cast(
            np.array([1, 2, 3], dtype=np.int32),
            dtype=types.DALIDataType.INT16,
            device=device,
        )

    check_is_pipeline_stateless(pipeline_factory)


def arithm_ops_outputs(data):
    return (
        data * 2,
        data + 2,
        data - 2,
        data / 2,
        data // 2,
        data**2,
        data == 2,
        data != 2,
        data < 2,
        data <= 2,
        data > 2,
        data >= 2,
        data & 2,
        data | 2,
        data ^ 2,
    )


@params("cpu", "gpu")
@stateless_signed_off("hidden.arithmetic_generic_op", "arithmetic_generic_op")
def test_arithm_ops_stateless_cpu(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout="HWC")
        return arithm_ops_outputs(move_to(data, device))

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off(
    "_conditional.hidden.not_",
    "_conditional.hidden.validate_logical",
    "_conditional.not_",
    "_conditional.validate_logical",
)
def test_logic_ops(device):
    @pipeline_def(enable_conditionals=True, enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout="HWC")
        if device == "gpu":
            data = data.gpu()
        condition_1 = fn.external_source(source=RandomBatch(data_shape=())) < 125
        condition_2 = fn.external_source(source=RandomBatch(data_shape=())) >= 125
        condition_3 = fn.external_source(source=RandomBatch(data_shape=())) <= 100
        if condition_1 and not condition_2 or not condition_3:
            return data
        else:
            return data + 1

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
@stateless_signed_off(
    "_conditional.hidden.merge",
    "_conditional.hidden.split",
    "_conditional.merge",
    "_conditional.split",
)
def test_split_and_merge(device):
    @pipeline_def(enable_conditionals=True, enable_checkpointing=True)
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout="HWC")
        if device == "gpu":
            data = data.gpu()
        condition = fn.external_source(source=RandomBatch(data_shape=())) <= 130
        if condition:
            return data
        else:
            return data + types.Constant(1, dtype=types.DALIDataType.UINT8)

    check_is_pipeline_stateless(pipeline_factory)


@params("gpu")
@stateless_signed_off("experimental.dilate")
def test_dilate_stateless(device):
    check_single_input(fn.experimental.dilate, device)


@params("gpu")
@stateless_signed_off("experimental.erode")
def test_erode_stateless(device):
    check_single_input(fn.experimental.erode, device)


@params("gpu", "cpu")
@stateless_signed_off("experimental.warp_perspective")
def test_warp_perspective_stateless(device):
    check_single_input(fn.experimental.warp_perspective, device, matrix=np.eye(3))


@params("gpu")
@stateless_signed_off("experimental.resize")
def test_experimental_resize(device):
    check_single_input(fn.experimental.resize, device, resize_x=50, resize_y=50)


@params("cpu")
@stateless_signed_off("zeros", "ones", "full", "zeros_like", "ones_like", "full_like")
def test_full_operator_family(device):
    @pipeline_def(enable_checkpointing=True)
    def pipeline_factory():
        sh = np.array([2, 3], dtype=np.int32)
        fill_value = np.array([1.0, 0.4, 3.0], dtype=np.float32)
        zeros = fn.zeros(shape=sh)
        ones = fn.ones(shape=sh)
        full = fn.full(fill_value, shape=sh)
        zeros_like = fn.zeros_like(zeros)
        ones_like = fn.ones_like(zeros)
        full_like = fn.full_like(zeros, fill_value)
        return zeros, ones, full, zeros_like, ones_like, full_like

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu")
@stateless_signed_off("io.file.read")
def test_io_file_read(device):
    check_single_filepath_input(fn.io.file.read, device)
