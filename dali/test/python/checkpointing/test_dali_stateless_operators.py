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

import os
import glob
import nose
import numpy as np
import itertools
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from test_utils import (
    compare_pipelines,
    get_dali_extra_path,
    check_numba_compatibility_cpu,
    has_operator,
    restrict_platform,
)
from nose2.tools import params, cartesian_params
from nose_utils import assert_raises
from test_optical_flow import is_of_supported
from nose.plugins.attrib import attr

# Test configuration
batch_size = 8
test_data_shape = [40, 60, 3]
test_data_layout = "HWC"
test_data_frames = 24
test_sequence_shape = [test_data_frames, 426, 240, 3]  # 240p video


def tensor_list_to_array(tensor_list):
    if isinstance(tensor_list, dali.backend_impl.TensorListGPU):
        tensor_list = tensor_list.as_cpu()
    return tensor_list.as_array()


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
    pipe.build()
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
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout=test_data_layout, batch=True)
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_sequence_input(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(
            source=RandomBatch(data_shape=test_sequence_shape), layout="FHWC", batch=True
        )
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_signal_input(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(
            source=RandomBatch(data_shape=[30, 40], dtype=np.float32), layout="ft", batch=True
        )
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_1d_input(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(
            source=RandomBatch(data_shape=[100], dtype=np.float32), batch=True
        )
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_encoded_jpeg_input(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        img = os.path.join(get_dali_extra_path(), "db/single/jpeg/100/swan-3584559_640.jpg")
        jpegs, _ = fn.readers.file(files=[img], pad_last_batch=True)
        return op(move_to(jpegs, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_encoded_audio_input(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        wav = os.path.join(get_dali_extra_path(), "db/audio/wav/237-134500-0000.wav")
        audio, _ = fn.readers.file(files=[wav], pad_last_batch=True)
        return op(move_to(audio, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_single_bbox_input(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(source=RandomBoundingBoxBatch(), batch=True)
        return op(move_to(data, device), device=device, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_no_input(op, device, **kwargs):
    @pipeline_def
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
def test_rotate_stateless(device):
    check_single_input(fn.rotate, device, angle=40)


@params("cpu", "gpu")
def test_resize_stateless(device):
    check_single_input(fn.resize, device, resize_x=50, resize_y=50)


@params("cpu", "gpu")
def test_flip_stateless(device):
    check_single_input(fn.flip, device)


@params("cpu", "gpu")
def test_crop_mirror_normalize_stateless(device):
    check_single_input(fn.crop_mirror_normalize, device)


@params("cpu", "gpu")
def test_warp_affine_stateless(device):
    check_single_input(fn.warp_affine, device, matrix=(0.1, 0.9, 10, 0.8, -0.2, -20))


@params("cpu", "gpu")
def test_saturation_stateless(device):
    check_single_input(fn.saturation, device)


@params("cpu", "gpu")
def test_reductions_min_stateless(device):
    check_single_input(fn.reductions.min, device)


@params("cpu", "gpu")
def test_reductions_max_stateless(device):
    check_single_input(fn.reductions.max, device)


@params("cpu", "gpu")
def test_reductions_sum_stateless(device):
    check_single_input(fn.reductions.sum, device)


@params("cpu", "gpu")
def test_equalize_stateless(device):
    check_single_input(fn.experimental.equalize, device)


def test_transforms_crop_stateless():
    check_no_input(fn.transforms.crop, "cpu")


def test_transforms_rotation_stateless():
    check_no_input(fn.transforms.rotation, "cpu", angle=35)


def test_transforms_shear_stateless():
    check_no_input(fn.transforms.shear, "cpu", shear=(2, 2))


def test_transforms_scale_stateless():
    check_no_input(fn.transforms.scale, "cpu", scale=(3, 2))


def test_transforms_translation_stateless():
    check_no_input(fn.transforms.translation, "cpu", offset=(4, 3))


@params("cpu", "gpu")
def test_one_hot_stateless(device):
    check_single_input(fn.one_hot, device)


def test_median_bluer_stateless():
    check_single_input(fn.experimental.median_blur, "gpu")


@params("cpu", "gpu")
def test_erase_stateless(device):
    check_single_input(fn.erase, device, anchor=(3, 4), shape=(5, 6))


@params("cpu", "gpu")
def test_pad_stateless(device):
    check_single_input(fn.pad, device, shape=(100, 100, 3))


@params("cpu", "gpu")
def test_constant_stateless(device):
    check_no_input(fn.constant, device, idata=[1, 2, 3])


@params("cpu", "gpu")
def test_reshape_stateless(device):
    check_single_input(fn.reshape, device, shape=[1, -1])


@params("cpu", "gpu")
def test_lookup_table_stateless(device):
    check_single_input(fn.lookup_table, device, keys=[0], values=[1], default_value=123)


@params("cpu", "gpu")
def test_transpose_stateless(device):
    check_single_input(fn.transpose, device, perm=[2, 0, 1])


def test_paste_stateless():
    check_single_input(fn.paste, "gpu", fill_value=0, ratio=2)


@params("cpu", "gpu")
def test_color_space_conversion_stateless(device):
    check_single_input(
        fn.color_space_conversion,
        device,
        image_type=dali.types.DALIImageType.RGB,
        output_type=dali.types.DALIImageType.YCbCr,
    )


def test_resize_crop_mirror_stateless(device):
    check_single_input(fn.resize_crop_mirror, "cpu", crop=(2, 2, 3), mirror=True)


@params("cpu", "gpu")
def test_slice_stateless(device):
    check_single_input(fn.slice, device, rel_start=(0.25, 0.25), rel_end=(0.75, 0.75))


@params("cpu", "gpu")
def test_shapes_stateless(device):
    check_single_input(fn.shapes, device)


@params("cpu", "gpu")
def test_per_frame_stateless(device):
    check_single_input(fn.per_frame, device, replace=True)


@params("cpu", "gpu")
def test_get_property_stateless(device):
    check_single_input(fn.get_property, device, key="layout")


@params("cpu", "gpu")
def test_jpeg_compression_distortion_stateless(device):
    check_single_input(fn.jpeg_compression_distortion, device)


@params("cpu", "gpu")
def test_multi_paste_stateless(device):
    check_single_input(
        fn.multi_paste, device, in_ids=list(range(batch_size)), output_size=[100, 100]
    )


@params("cpu", "gpu")
def test_grid_mask_stateless(device):
    check_single_input(fn.grid_mask, device)


@params("cpu", "gpu")
def test_preemphasis_filter_stateless(device):
    check_single_input(fn.preemphasis_filter, device)


def test_optical_flow_stateless():
    if not is_of_supported():
        raise nose.SkipTest("Optical Flow is not supported on this platform")
    check_single_sequence_input(fn.optical_flow, "gpu")


@params("cpu", "gpu")
def test_sequence_rearrange_stateless(device):
    check_single_sequence_input(
        fn.sequence_rearrange, device, new_order=list(range(test_data_frames))
    )


@params("cpu", "gpu")
def test_spectrogram_stateless(device):
    check_single_1d_input(fn.spectrogram, device)


def test_power_spectrum_stateless():
    check_single_signal_input(fn.power_spectrum, "cpu")


@params("cpu", "gpu")
def test_dump_image_stateless(device):
    suffix = "test_dump_image_stateless_tmp"
    check_single_input(fn.dump_image, device, suffix=suffix)
    for f in glob.glob(f"*-{suffix}-*.ppm"):
        os.remove(f)


@params("cpu", "gpu")
def test_variance_stateless(device):
    check_single_1d_input(lambda x, **kwargs: fn.reductions.variance(x, 0.0, **kwargs), device)


@params("cpu", "gpu")
def test_normalize_stateless(device):
    check_single_input(fn.normalize, device)


@params("cpu", "gpu")
def test_mel_filter_bank_stateless(device):
    check_single_signal_input(fn.mel_filter_bank, device)


@params("cpu", "gpu")
def test_mfcc_stateless(device):
    check_single_signal_input(fn.mfcc, device)


@params("cpu", "gpu")
def test_nonsilent_region_stateless(device):
    check_single_1d_input(lambda *args, **kwargs: fn.nonsilent_region(*args, **kwargs)[0], device)


@params("cpu", "gpu")
def test_audio_resample_stateless(device):
    check_single_signal_input(fn.audio_resample, device, scale=0.5)


@params("cpu", "gpu")
def test_element_extract_stateless(device):
    check_single_sequence_input(fn.element_extract, device, element_map=[0])


def test_bbox_paste_stateless():
    check_single_bbox_input(fn.bbox_paste, "cpu", ratio=2)


@params("cpu", "gpu")
def test_bb_flip_stateless(device):
    check_single_bbox_input(fn.bb_flip, device, ltrb=True)


@params("cpu", "gpu")
def test_to_decibels_stateless(device):
    check_single_signal_input(fn.to_decibels, device)


@cartesian_params(("cpu", "gpu"), (fn.stack, fn.cat))
def test_tensor_join_stateless(device, join):
    def wrapper(x, **kwargs):
        return join(x, x, x, **kwargs)

    check_single_input(wrapper, device)


@params("cpu", "gpu")
def test_tensor_subscript_stateless(device):
    check_single_input(lambda x, **kwargs: x[0, :, 2:3], device)


@params("cpu", "gpu")
def test_permute_batch_stateless(device):
    def wrapper(x, **kwargs):
        return fn.permute_batch(x, indices=[0] * batch_size, **kwargs)

    check_single_input(wrapper, device)


def test_select_masks_stateless():
    n = 10
    polygons = np.asarray([[i, 0, i] for i in range(n)])
    vertices = np.asarray([[i, i + 1] for i in range(n)])
    mask_ids = np.asarray([i for i in range(n) if i % 2 == 0])

    @pipeline_def
    def pipeline_factory():
        return tuple(fn.segmentation.select_masks(mask_ids, polygons, vertices))

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
def test_box_encoder_stateless(device):
    n = 10
    boxes = np.asarray([[float(i), float(i), float(i + 1), float(i + 1)] for i in range(n)])
    labels = np.asarray(list(range(n)))
    anchors = [float(i) for i in range(4)]

    @pipeline_def
    def pipeline_factory():
        return tuple(fn.box_encoder(boxes, labels, anchors=anchors, device=device))

    check_is_pipeline_stateless(pipeline_factory)


@attr("cupy")
@params("cpu", "gpu")
def test_python_function_stateless(device):
    def wrapper(x, **kwargs):
        return fn.python_function(x, function=lambda x: x * 2, **kwargs)

    check_single_input(wrapper, device)


@attr("numba")
def test_numba_function_stateless():
    import nvidia.dali.plugin.numba as dali_numba

    check_numba_compatibility_cpu()

    def double_sample(out_sample, in_sample):
        out_sample[:] = 2 * in_sample[:]

    @pipeline_def(batch_size=2, device_id=0, num_threads=4)
    def numba_pipe():
        forty_two = fn.external_source(
            source=lambda x: np.full((2,), 42, dtype=np.uint8), batch=False
        )
        out = dali_numba.fn.experimental.numba_function(
            forty_two,
            run_fn=double_sample,
            out_types=[dali.types.DALIDataType.UINT8],
            in_types=[dali.types.DALIDataType.UINT8],
            outs_ndim=[1],
            ins_ndim=[1],
            batch_processing=False,
        )
        return out

    check_is_pipeline_stateless(numba_pipe)


@has_operator("experimental.inflate")
@restrict_platform(min_compute_cap=6.0, platforms=["x86_64"])
def test_inflate_stateless():
    import lz4.block

    def sample_to_lz4(sample):
        deflated_buf = lz4.block.compress(sample, store_size=False)
        return np.frombuffer(deflated_buf, dtype=np.uint8)

    source = RandomBatch()
    batch = source()
    input_data = [sample_to_lz4(sample) for sample in batch]

    input_shape = [np.array(sample.shape, dtype=np.int32) for sample in batch]

    @pipeline_def
    def pipeline():
        deflated = fn.external_source(source=itertools.repeat(input_data))
        shape = fn.external_source(source=itertools.repeat(input_shape))
        return fn.experimental.inflate(deflated.gpu(), shape=shape)

    check_is_pipeline_stateless(pipeline)


def test_peek_image_shape_stateless():
    check_single_encoded_jpeg_input(fn.peek_image_shape, "cpu")


def test_imgcodec_peek_image_shape_stateless():
    check_single_encoded_jpeg_input(fn.experimental.peek_image_shape, "cpu")


def test_audio_decoder_stateless():
    def audio_decoder_wrapper(*args, **kwargs):
        return fn.decoders.audio(*args, **kwargs)[0]

    check_single_encoded_audio_input(audio_decoder_wrapper, "cpu")


@params("cpu", "mixed")
def test_image_decoder_stateless(device):
    check_single_encoded_jpeg_input(fn.decoders.image, device)


@params("cpu", "mixed")
def test_image_decoder_crop_stateless(device):
    check_single_encoded_jpeg_input(fn.decoders.image_crop, device)


@params("cpu", "gpu")
def test_coord_flip_stateless(device):
    @pipeline_def
    def pipeline_factory():
        input = np.array([[1], [2], [3]], dtype=np.float32)
        return fn.coord_flip(input, flip_x=True, center_x=0, device=device)

    check_is_pipeline_stateless(pipeline_factory)


@params("cpu", "gpu")
def test_cast_like_stateless(device):
    @pipeline_def
    def pipeline_factory():
        return fn.cast_like(
            np.array([1, 2, 3], dtype=np.int32), np.array([1.0], dtype=np.float32), device=device
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
def test_arithm_ops_stateless_cpu(device):
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout="HWC")
        return arithm_ops_outputs(move_to(data, device))

    check_is_pipeline_stateless(pipeline_factory)
