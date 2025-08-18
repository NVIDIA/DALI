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

import glob
import math
import numpy as np
import nvidia.dali.backend
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import random
from nvidia.dali import pipeline_def

from nose_utils import assert_raises, SkipTest
from test_utils import compare_pipelines
from test_utils import get_dali_extra_path
from test_utils import to_array
from test_utils import get_arch
from test_utils import dump_as_core_artifacts
from nose2.tools import params


def get_img_files(data_path, subdir="*", ext=None):
    if subdir is None:
        subdir = ""
    if ext:
        if isinstance(ext, (list, tuple)):
            files = []
            for e in ext:
                files += glob.glob(data_path + f"/{subdir}/*.{e}")
        else:
            files = glob.glob(data_path + f"/{subdir}/*.{ext}")
        return files
    else:
        files = glob.glob(data_path + f"/{subdir}/*.*")
        txt_files = glob.glob(data_path + f"/{subdir}/*.txt")
        return list(set(files) - set(txt_files))


@pipeline_def
def decoder_pipe(data_path, device, use_fast_idct=False, jpeg_fancy_upsampling=False):
    inputs, labels = fn.readers.file(file_root=data_path, shard_id=0, num_shards=1, name="Reader")
    decoded = fn.experimental.decoders.image(
        inputs,
        device=device,
        output_type=types.RGB,
        use_fast_idct=use_fast_idct,
        jpeg_fancy_upsampling=jpeg_fancy_upsampling,
    )

    return decoded, labels


test_data_root = get_dali_extra_path()
good_path = "db/single"
misnamed_path = "db/single/missnamed"
test_good_path = ["jpeg", "mixed", "png", "tiff", "pnm", "bmp", "jpeg2k", "webp"]
test_misnamed_path = ["jpeg", "png", "tiff", "pnm", "bmp"]


def run_decode(data_path, batch, device, threads):
    pipe = decoder_pipe(
        data_path=data_path,
        batch_size=batch,
        num_threads=threads,
        device_id=0,
        device=device,
        prefetch_queue_depth=1,
    )
    iters = math.ceil(pipe.epoch_size("Reader") / batch)
    for iter in range(iters):
        pipe.run()


def test_image_decoder():
    for device in ["cpu", "mixed"]:
        for batch_size in [1, 10]:
            for img_type in test_good_path:
                for threads in [1, random.choice([2, 3, 4])]:
                    data_path = os.path.join(test_data_root, good_path, img_type)
                    yield run_decode, data_path, batch_size, device, threads
            for img_type in test_misnamed_path:
                for threads in [1, random.choice([2, 3, 4])]:
                    data_path = os.path.join(test_data_root, misnamed_path, img_type)
                    yield run_decode, data_path, batch_size, device, threads


@pipeline_def
def create_decoder_slice_pipeline(data_path, device):
    jpegs, _ = fn.readers.file(file_root=data_path, shard_id=0, num_shards=1, name="Reader")

    anchor = fn.random.uniform(range=[0.05, 0.15], shape=(2,))
    shape = fn.random.uniform(range=[0.5, 0.7], shape=(2,))
    images_sliced_1 = fn.experimental.decoders.image_slice(
        jpegs, anchor, shape, axes=(0, 1), device=device
    )

    images = fn.experimental.decoders.image(jpegs, device=device)
    images_sliced_2 = fn.slice(images, anchor, shape, axes=(0, 1))

    return images_sliced_1, images_sliced_2


@pipeline_def
def create_decoder_crop_pipeline(data_path, device):
    jpegs, _ = fn.readers.file(file_root=data_path, shard_id=0, num_shards=1, name="Reader")

    crop_pos_x = fn.random.uniform(range=[0.1, 0.9])
    crop_pos_y = fn.random.uniform(range=[0.1, 0.9])
    w = 242
    h = 230

    images_crop_1 = fn.experimental.decoders.image_crop(
        jpegs, crop=(w, h), crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, device=device
    )

    images = fn.experimental.decoders.image(jpegs, device=device)

    images_crop_2 = fn.crop(images, crop=(w, h), crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)

    return images_crop_1, images_crop_2


@pipeline_def
def create_decoder_random_crop_pipeline(data_path, device):
    seed = 1234
    jpegs, _ = fn.readers.file(file_root=data_path, shard_id=0, num_shards=1, name="Reader")

    w = 242
    h = 230
    images_random_crop_1 = fn.experimental.decoders.image_random_crop(
        jpegs, device=device, output_type=types.RGB, seed=seed
    )
    images_random_crop_1 = fn.resize(images_random_crop_1, size=(w, h))

    images = fn.experimental.decoders.image(jpegs, device=device)
    images_random_crop_2 = fn.random_resized_crop(images, size=(w, h), seed=seed)

    return images_random_crop_1, images_random_crop_2


def run_decode_fused(test_fun, path, img_type, batch, device, threads, validation_fun):
    data_path = os.path.join(test_data_root, path, img_type)
    pipe = test_fun(
        data_path=data_path,
        batch_size=batch,
        num_threads=threads,
        device_id=0,
        device=device,
        prefetch_queue_depth=1,
    )
    idxs = [i for i in range(batch)]
    iters = math.ceil(pipe.epoch_size("Reader") / batch)
    for it in range(iters):
        out_1, out_2 = pipe.run()
        for sample_idx, img_1, img_2 in zip(idxs, out_1, out_2):
            arr_1 = to_array(img_1)
            arr_2 = to_array(img_2)
            is_ok = validation_fun(arr_1, arr_2)
            if not is_ok:
                dump_as_core_artifacts(
                    img_1.source_info(), arr_1, arr_2, iter=it, sample_idx=sample_idx
                )
            assert is_ok, (
                f"{validation_fun.__name__}\n"
                + f"image: {img_1.source_info()} iter: {it} sample_idx: {sample_idx}"
            )


def test_image_decoder_fused():
    threads = 4
    batch_size = 10
    for test_fun in [
        create_decoder_slice_pipeline,
        create_decoder_crop_pipeline,
        create_decoder_random_crop_pipeline,
    ]:
        # before CUDA 11.4 HW decoder API doesn't support ROI so we get slightly different results
        # HW decoder + slice vs fused which in this case is executed by the hybrid backend
        if (
            test_fun == create_decoder_random_crop_pipeline
            or nvidia.dali.backend.GetNvjpegVersion() < 11040
        ):
            # random_resized_crop can properly handle border as it has pixels that are cropped out,
            # while plain resize following image_decoder_random_crop cannot do that
            # and must duplicate the border pixels
            def mean_close(x, y):
                return np.mean(np.abs(x - y) < 0.5)

            validation_fun = mean_close
        else:

            def mean_close(x, y):
                return np.allclose(x, y)

            validation_fun = mean_close
        for device in ["cpu", "mixed"]:
            for img_type in test_good_path:
                yield (
                    run_decode_fused,
                    test_fun,
                    good_path,
                    img_type,
                    batch_size,
                    device,
                    threads,
                    validation_fun,
                )


def check_FastDCT_body(batch_size, img_type, device):
    data_path = os.path.join(test_data_root, good_path, img_type)
    compare_pipelines(
        decoder_pipe(
            data_path=data_path,
            batch_size=batch_size,
            num_threads=3,
            device_id=0,
            device=device,
            use_fast_idct=False,
        ),
        decoder_pipe(
            data_path=data_path,
            batch_size=batch_size,
            num_threads=3,
            device_id=0,
            device="cpu",
            use_fast_idct=True,
        ),
        # average difference should be no bigger than off-by-3
        batch_size=batch_size,
        N_iterations=3,
        eps=3,
    )


def test_FastDCT():
    for device in ["cpu", "mixed"]:
        for batch_size in [1, 8]:
            for img_type in test_good_path:
                yield check_FastDCT_body, batch_size, img_type, device


def check_fancy_upsampling_body(batch_size, img_type, device):
    data_path = os.path.join(test_data_root, good_path, img_type)
    compare_pipelines(
        decoder_pipe(
            data_path=data_path,
            batch_size=batch_size,
            num_threads=3,
            device_id=0,
            device=device,
            jpeg_fancy_upsampling=True,
        ),
        decoder_pipe(
            data_path=data_path, batch_size=batch_size, num_threads=3, device_id=0, device="cpu"
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1,
    )


@params(1, 8)
def test_fancy_upsampling(batch_size):
    if nvidia.dali.backend.GetNvjpegVersion() < 12010:
        raise SkipTest("nvJPEG doesn't support fancy upsampling in this version")

    data_path = os.path.join(test_data_root, good_path, "jpeg")
    compare_pipelines(
        decoder_pipe(
            data_path=data_path,
            batch_size=batch_size,
            num_threads=3,
            device_id=0,
            device="mixed",
            jpeg_fancy_upsampling=True,
        ),
        decoder_pipe(
            data_path=data_path, batch_size=batch_size, num_threads=3, device_id=0, device="cpu"
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1,
    )


batch_size_test = 16


@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def img_decoder_pipe(device, out_type, files):
    encoded, _ = fn.readers.file(files=files)
    decoded = fn.experimental.decoders.image(encoded, device=device, output_type=out_type)
    return decoded


def _testimpl_image_decoder_consistency(img_out_type, file_fmt, path, subdir="*", ext=None):
    eps = 1
    if file_fmt == "jpeg" or file_fmt == "mixed":
        eps = 4
    if (file_fmt == "jpeg2k" or file_fmt == "mixed") and img_out_type == types.YCbCr:
        eps = 6
    files = get_img_files(os.path.join(test_data_root, path), subdir=subdir, ext=ext)
    compare_pipelines(
        img_decoder_pipe("cpu", out_type=img_out_type, files=files),
        img_decoder_pipe("mixed", out_type=img_out_type, files=files),
        batch_size=batch_size_test,
        N_iterations=3,
        eps=eps,
    )


def test_image_decoder_consistency():
    for out_img_type in [types.RGB, types.BGR, types.YCbCr, types.GRAY, types.ANY_DATA]:
        for file_fmt in test_good_path:
            path = os.path.join(good_path, file_fmt)
            yield _testimpl_image_decoder_consistency, out_img_type, file_fmt, path

        for file_fmt, path, ext in [
            ("tiff", "db/single/multichannel/tiff_multichannel", "tif"),
            ("jpeg2k", "db/single/multichannel/with_alpha", "jp2"),
            ("png", "db/single/multichannel/with_alpha", "png"),
        ]:
            subdir = None  # In those paths the images are not organized in subdirs
            yield _testimpl_image_decoder_consistency, out_img_type, file_fmt, path, subdir, ext


def _testimpl_image_decoder_tiff_with_alpha_16bit(device, out_type, path, ext):
    @pipeline_def(batch_size=1, device_id=0, num_threads=1)
    def pipe(device, out_type, files):
        encoded, _ = fn.readers.file(files=files)
        decoded = fn.experimental.decoders.image(encoded, device=device, output_type=out_type)
        peeked_shape = fn.experimental.peek_image_shape(encoded)
        return decoded, peeked_shape

    files = get_img_files(os.path.join(test_data_root, path), ext=ext, subdir=None)
    pipe = pipe(device, out_type=out_type, files=files)
    out, shape = pipe.run()
    if device == "mixed":
        out = out.as_cpu()
    out = np.array(out[0])
    shape = np.array(shape[0])
    expected_channels = 4 if out_type == types.ANY_DATA else 1 if out_type == types.GRAY else 3
    assert out.shape[2] == expected_channels, f"Expected {expected_channels} but got {out.shape[2]}"


def test_image_decoder_tiff_with_alpha_16bit():
    for device in ["cpu", "mixed"]:
        for out_type in [types.RGB, types.BGR, types.YCbCr, types.ANY_DATA]:
            path = "db/single/multichannel/with_alpha_16bit"
            for ext in [("png", "tiff", "jp2")]:
                yield _testimpl_image_decoder_tiff_with_alpha_16bit, device, out_type, path, ext


def _testimpl_image_decoder_crop_error_oob(device):
    file_root = os.path.join(test_data_root, good_path, "jpeg")

    @pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
    def pipe(device):
        encoded, _ = fn.readers.file(file_root=file_root)
        decoded = fn.experimental.decoders.image_crop(
            encoded, crop_w=10000, crop_h=100, device=device
        )
        return decoded

    p = pipe(device)
    assert_raises(
        RuntimeError, p.run, glob="cropping window*..*..*is not valid for image dimensions*[*x*]"
    )


def test_image_decoder_crop_error_oob():
    for device in ["cpu", "mixed"]:
        yield _testimpl_image_decoder_crop_error_oob, device


def _testimpl_image_decoder_slice_error_oob(device):
    file_root = os.path.join(test_data_root, good_path, "jpeg")

    @pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
    def pipe(device):
        encoded, _ = fn.readers.file(file_root=file_root)
        decoded = fn.experimental.decoders.image_slice(
            encoded, device=device, end=[10000], axes=[1]
        )
        return decoded

    p = pipe(device)
    assert_raises(
        RuntimeError, p.run, glob="cropping window*..*..*is not valid for image dimensions*[*x*]"
    )


def test_image_decoder_slice_error_oob():
    for device in ["cpu", "mixed"]:
        yield _testimpl_image_decoder_slice_error_oob, device


def test_tiff_palette():
    normal = os.path.join(test_data_root, good_path, "tiff", "0/cat-300572_640.tiff")
    palette = os.path.join(test_data_root, good_path, "tiff", "0/cat-300572_640_palette.tiff")

    @pipeline_def(batch_size=2, device_id=0, num_threads=1)
    def pipe():
        encoded, _ = fn.readers.file(files=[normal, palette])
        peeked_shapes = fn.experimental.peek_image_shape(encoded)
        decoded = fn.experimental.decoders.image(encoded, device="cpu")
        return decoded, peeked_shapes

    p = pipe()
    imgs, peeked_shapes = p.run()
    assert (
        peeked_shapes.at(0) == peeked_shapes.at(1)
    ).all(), "Invalid peeked shape of palette TIFF"

    delta = np.abs(imgs.at(0).astype("float") - imgs.at(1).astype("float")) / 256
    assert np.quantile(delta, 0.9) < 0.05, "Original and palette TIFF differ significantly"


def _testimpl_image_decoder_peek_shape(
    name, expected_shape, image_type=types.ANY_DATA, adjust_orientation=True
):
    file = os.path.join(test_data_root, good_path, name)

    @pipeline_def(batch_size=1, device_id=0, num_threads=1)
    def peek_shape_pipeline(file):
        encoded, _ = fn.readers.file(files=[file])
        return fn.experimental.peek_image_shape(
            encoded, image_type=image_type, adjust_orientation=adjust_orientation
        )

    pipe = peek_shape_pipeline(file)
    shape = tuple(np.asarray(pipe.run()[0][0]))
    assert shape == expected_shape, f"Expected shape {expected_shape} but got {shape}"


def test_peek_shape():
    tests = [
        ("bmp/0/cat-1245673_640.bmp", (423, 640, 3)),
        ("bmp/0/cat-111793_640_grayscale.bmp", (426, 640, 1)),
        ("jpeg/641/maracas-706753_1280.jpg", (1280, 919, 3)),
        ("jpeg2k/0/cat-3449999_640.jp2", (426, 640, 3)),
        ("tiff/0/cat-300572_640.tiff", (536, 640, 3)),
        ("png/0/cat-3591348_640.png", (427, 640, 3)),
        ("pnm/0/cat-3591348_640.pbm", (427, 640, 1)),
        ("tiff/0/kitty-2948404_640.tiff", (433, 640, 3)),
        ("tiff/0/cat-111793_640_gray.tiff", (475, 640, 1)),
        ("webp/lossless/cat-111793_640.webp", (426, 640, 3)),
        ("jpeg_lossless/0/cat-1245673_640_grayscale_16bit.jpg", (423, 640, 1)),
        ("multichannel/with_alpha/cat-111793_640-alpha.jp2", (426, 640, 4)),
        ("multichannel/with_alpha/cat-111793_640-alpha.png", (426, 640, 4)),
        ("multichannel/tiff_multichannel/cat-111793_640_multichannel.tif", (475, 640, 6)),
    ]

    for name, expected_shape in tests:
        yield _testimpl_image_decoder_peek_shape, name, expected_shape

    yield _testimpl_image_decoder_peek_shape, "tiff/0/kitty-2948404_640.tiff", (
        433,
        640,
        1,
    ), types.GRAY, True
    yield _testimpl_image_decoder_peek_shape, "bmp/0/cat-111793_640_grayscale.bmp", (
        426,
        640,
        3,
    ), types.RGB, True


def is_nvjpeg_lossless_supported(device_id):
    return get_arch(device_id) >= 6.0 and nvidia.dali.backend.GetNvjpegVersion() >= 12020


@params(
    ("cat-1245673_640_grayscale_16bit", types.ANY_DATA, types.UINT16, 16),
    ("cat-3449999_640_grayscale_16bit", types.ANY_DATA, types.UINT16, 16),
    ("cat-3449999_640_grayscale_12bit", types.ANY_DATA, types.UINT16, 12),
    ("cat-3449999_640_grayscale_16bit", types.ANY_DATA, types.FLOAT, 16),
    ("cat-3449999_640_grayscale_12bit", types.ANY_DATA, types.FLOAT, 12),
    ("cat-3449999_640_grayscale_16bit", types.GRAY, types.UINT16, 16),
    ("cat-3449999_640_grayscale_8bit", types.ANY_DATA, types.UINT8, 8),
)
def test_image_decoder_lossless_jpeg(img_name, output_type, dtype, precision):
    device_id = 0
    if not is_nvjpeg_lossless_supported(device_id=device_id):
        raise SkipTest("NVJPEG lossless supported on SM60+ capable devices only")

    data_dir = os.path.join(test_data_root, "db/single/jpeg_lossless/0")
    ref_data_dir = os.path.join(test_data_root, "db/single/reference/jpeg_lossless")

    @pipeline_def(batch_size=1, device_id=device_id, num_threads=1)
    def pipe(file):
        encoded, _ = fn.readers.file(files=[file])
        decoded = fn.experimental.decoders.image(
            encoded, device="mixed", dtype=dtype, output_type=output_type
        )
        return decoded

    p = pipe(data_dir + f"/{img_name}.jpg")
    (out,) = p.run()
    result = np.array(out[0].as_cpu())

    ref = np.load(ref_data_dir + f"/{img_name}.npy")
    kwargs = {}
    np_dtype = types.to_numpy_type(dtype)
    max_val = np_dtype(1.0) if dtype == types.FLOAT else np.iinfo(np_dtype).max
    need_scaling = max_val != np_dtype(2**precision - 1)
    if need_scaling:
        # numpy 2.x computes this division as float32 while numpy 1.x as float64
        # so we need to cast max_val to python float to get the same results
        multiplier = float(max_val) / float(2**precision - 1)
        ref = ref * multiplier
        if dtype != types.FLOAT:
            kwargs["atol"] = 0.5  # possible rounding error
    np.testing.assert_allclose(ref, result, **kwargs)


def test_image_decoder_lossless_jpeg_cpu_not_supported():
    @pipeline_def(batch_size=1, device_id=0, num_threads=1)
    def pipe(file):
        encoded, _ = fn.readers.file(files=[file])
        decoded = fn.experimental.decoders.image(
            encoded, device="cpu", dtype=types.UINT16, output_type=types.ANY_DATA
        )
        return decoded

    imgfile = "db/single/jpeg_lossless/0/cat-1245673_640_grayscale_16bit.jpg"
    p = pipe(os.path.join(test_data_root, imgfile))

    assert_raises(RuntimeError, p.run, glob="*Failed to decode*")
