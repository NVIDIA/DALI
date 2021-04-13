# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from test_utils import get_dali_extra_path, check_batch, RandomlyShapedDataIterator, dali_type
from segmentation_test_utils import make_batch_select_masks
from PIL import Image, ImageEnhance

import numpy as np
from nose.tools import assert_raises
import os
import glob
from math import ceil, sqrt
import tempfile

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
audio_dir = os.path.join(data_root, 'db', 'audio')
caffe_dir = os.path.join(data_root, 'db', 'lmdb')
caffe2_dir = os.path.join(data_root, 'db', 'c2lmdb')
recordio_dir = os.path.join(data_root, 'db', 'recordio')
tfrecord_dir = os.path.join(data_root, 'db', 'tfrecord')
coco_dir = os.path.join(data_root, 'db', 'coco', 'images')
coco_annotation = os.path.join(data_root, 'db', 'coco', 'instances.json')
sequence_dir = os.path.join(data_root, 'db', 'sequence', 'frames')

batch_size = 2
test_data_shape = [10, 20, 3]
def get_data():
    out = [np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
    return out

def test_move_to_device_end():
    test_data_shape = [1, 3, 0, 4]
    def get_data():
        out = [np.empty(test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    outs = fn.external_source(source = get_data)
    pipe.set_outputs(outs.gpu())
    assert_raises(RuntimeError, pipe.build)

def test_move_to_device_middle():
    test_data_shape = [1, 3, 0, 4]
    def get_data():
        out = [np.empty(test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source = get_data)
    outs = fn.rotate(data.gpu(), angle = 25)
    pipe.set_outputs(outs)
    assert_raises(RuntimeError, pipe.build)

def check_bad_device(device_id):
    test_data_shape = [1, 3, 0, 4]
    def get_data():
        out = [np.empty(test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=device_id)
    outs = fn.external_source(source = get_data, device = "gpu")
    pipe.set_outputs(outs)
    assert_raises(RuntimeError, pipe.build)

def test_gpu_op_bad_device():
    for device_id in [None, 0]:
        yield check_bad_device, device_id

def check_mixed_op_bad_device(device_id):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=device_id)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image(input, device="mixed", output_type=types.RGB)
    pipe.set_outputs(decoded)
    assert_raises(RuntimeError, pipe.build)

def test_mixed_op_bad_device():
    for device_id in [None, 0]:
        yield check_bad_device, device_id

def test_image_decoder_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image(input, output_type=types.RGB)
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def check_single_input(op, input_layout = "HWC", get_data = get_data, **kwargs):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_data, layout = input_layout)
    processed = op(data, **kwargs)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def check_no_input(op, get_data = get_data, **kwargs):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    processed = op(**kwargs)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_rotate_cpu():
    check_single_input(fn.rotate, angle = 25)

def test_brightness_contrast_cpu():
    check_single_input(fn.brightness_contrast)

def test_hue_cpu():
    check_single_input(fn.hue)

def test_brightness_cpu():
    check_single_input(fn.brightness)

def test_contrast_cpu():
    check_single_input(fn.contrast)

def test_hsv_cpu():
    check_single_input(fn.hsv)

def test_color_twist_cpu():
    check_single_input(fn.color_twist)

def test_saturation_cpu():
    check_single_input(fn.saturation)

def test_old_color_twist_cpu():
    check_single_input(fn.old_color_twist)

def test_shapes_cpu():
    check_single_input(fn.shapes)

def test_crop_cpu():
    check_single_input(fn.crop, crop = (5, 5))

def test_color_space_coversion_cpu():
    check_single_input(fn.color_space_conversion, image_type = types.BGR, output_type = types.RGB)

def test_cast_cpu():
    check_single_input(fn.cast, dtype  = types.INT32)

def test_resize_cpu():
    check_single_input(fn.resize, resize_x=50, resize_y=50)

def test_gaussian_blur_cpu():
    check_single_input(fn.gaussian_blur, window_size = 5)

def test_crop_mirror_normalize_cpu():
    check_single_input(fn.crop_mirror_normalize)

def test_flip_cpu():
    check_single_input(fn.flip, horizontal = True)

def test_jpeg_compression_distortion_cpu():
    check_single_input(fn.jpeg_compression_distortion, quality = 10)

def test_image_decoder_crop_device():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image_crop(input, output_type=types.RGB, crop = (10, 10))
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_image_decoder_random_crop_device():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image_random_crop(input, output_type=types.RGB)
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_coin_flip_device():
    check_no_input(fn.random.coin_flip)

def test_uniform_device():
    check_no_input(fn.random.uniform)

def test_reshape_cpu():
    new_shape = test_data_shape.copy()
    new_shape[0] //= 2
    new_shape[1] *= 2
    check_single_input(fn.reshape, shape = new_shape)

def test_reinterpret_cpu():
    check_single_input(fn.reinterpret, rel_shape = [0.5, 1, -1])

def test_water_cpu():
    check_single_input(fn.water)

def test_sphere_cpu():
    check_single_input(fn.sphere)

def test_erase_cpu():
    check_single_input(fn.erase, anchor=[0.3], axis_names="H", normalized_anchor = True, shape = [0.1],normalized_shape = True)

def test_random_resized_crop_cpu():
    check_single_input(fn.random_resized_crop, size = [5, 5])

def test_nonsilent_region_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200]
    def get_data():
        out = [np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        out[0][0] = 0
        out[1][0] = 0
        out[1][1] = 0
        return out
    data = fn.external_source(source = get_data)
    processed, _ = fn.nonsilent_region(data)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

test_audio_data_shape = [200]
def get_audio_data():
    out = [np.random.ranf(size = test_audio_data_shape).astype(dtype = np.float32) for _ in range(batch_size)]
    return out

def test_preemphasis_filter_cpu():
    check_single_input(fn.preemphasis_filter, get_data = get_audio_data, input_layout = None)

def test_power_spectrum_cpu():
    check_single_input(fn.power_spectrum, get_data = get_audio_data, input_layout = None)

def test_spectrogram_cpu():
    check_single_input(fn.spectrogram, get_data = get_audio_data, input_layout = None, nfft = 60, window_length = 50, window_step = 25)

def test_mel_filter_bank_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_audio_data)
    spectrum = fn.spectrogram(data, nfft = 60, window_length = 50, window_step = 25)
    processed = fn.mel_filter_bank(spectrum)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_to_decibels_cpu():
    check_single_input(fn.to_decibels, get_data = get_audio_data, input_layout = None)

def test_mfcc_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_audio_data)
    spectrum = fn.spectrogram(data, nfft = 60, window_length = 50, window_step = 25)
    mel = fn.mel_filter_bank(spectrum)
    dec = fn.to_decibels(mel)
    processed = fn.mfcc(dec)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_fast_resize_crop_mirror_cpu():
    check_single_input(fn.fast_resize_crop_mirror, crop = [5, 5], resize_shorter = 10)

def test_resize_crop_mirror_cpu():
    check_single_input(fn.resize_crop_mirror, crop = [5, 5], resize_shorter = 10)

def test_normal_distribution_cpu():
    check_no_input(fn.random.normal, shape = [5, 5])

def test_one_hot_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200]
    def get_data():
        out = [np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data)
    processed = fn.one_hot(data, num_classes = 256)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_transpose_cpu():
    check_single_input(fn.transpose, perm  = [2, 0, 1])

def test_audio_decoder_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=audio_dir, shard_id=0, num_shards=1)
    decoded, _ = fn.decoders.audio(input)
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_coord_flip_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200, 2]
    def get_data():
        out = [(np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data)
    processed = fn.coord_flip(data)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_bb_flip_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200, 4]
    def get_data():
        out = [(np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data)
    processed = fn.bb_flip(data)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_warp_affine_cpu():
    warp_matrix = (0.1, 0.9, 10, 0.8, -0.2, -20)
    check_single_input(fn.warp_affine, matrix = warp_matrix)

def test_normalize_cpu():
    check_single_input(fn.normalize, batch  = True)

def test_lookup_table_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [100]
    def get_data():
        out = [np.random.randint(0, 5, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data)
    processed = fn.lookup_table(data, keys = [1, 3], values = [10, 50])
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_slice_cpu():
    anch_shape = [2]
    def get_anchors():
        out = [(np.random.randint(1, 256, size = anch_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    def get_shape():
        out = [(np.random.randint(1, 256, size = anch_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_data, layout = "HWC")
    anchors = fn.external_source(source = get_anchors)
    shape = fn.external_source(source = get_shape)
    processed = fn.slice(data, anchors, shape, out_of_bounds_policy = "pad")
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_image_decoder_slice_cpu():
    anch_shape = [2]
    def get_anchors():
        out = [(np.random.randint(1, 128, size = anch_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    def get_shape():
        out = [(np.random.randint(1, 128, size = anch_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    anchors = fn.external_source(source = get_anchors)
    shape = fn.external_source(source = get_shape)
    processed = fn.decoders.image_slice(input, anchors, shape)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_pad_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [5, 4, 3]
    def get_data():
        out = [np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data, layout = "HWC")
    processed = fn.pad(data, fill_value = -1, axes = (0,), shape = (10,) )
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_mxnet_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    out, _ = fn.readers.mxnet(path = os.path.join(recordio_dir, "train.rec"),
                              index_path = os.path.join(recordio_dir, "train.idx"),
                              shard_id=0, num_shards=1)
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_tfrecord_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    tfrecord = sorted(glob.glob(os.path.join(tfrecord_dir, '*[!i][!d][!x]')))
    tfrecord_idx = sorted(glob.glob(os.path.join(tfrecord_dir, '*idx')))
    input = fn.readers.tfrecord(path = tfrecord,
                                index_path = tfrecord_idx,
                                shard_id=0, num_shards=1,
                                features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)})
    out = input["image/encoded"]
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_coco_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    out, _, _ = fn.readers.coco(file_root=coco_dir, annotations_file=coco_annotation, shard_id=0, num_shards=1)
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_caffe_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    out, _ = fn.readers.caffe(path = caffe_dir, shard_id=0, num_shards=1)
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_caffe2_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    out, _ = fn.readers.caffe2(path = caffe2_dir, shard_id=0, num_shards=1)
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_copy_cpu():
    check_single_input(fn.copy)

def test_element_extract_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [5, 10, 20, 3]
    def get_data():
        out = [np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data, layout = "FHWC")
    processed, _ = fn.element_extract(data, element_map=[0, 3])
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_bbox_paste_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200, 4]
    def get_data():
        out = [(np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data)
    paste_posx = fn.random.uniform(range=(0, 1))
    paste_posy = fn.random.uniform(range=(0, 1))
    paste_ratio = fn.random.uniform(range=(1, 2))
    processed = fn.bbox_paste(data, paste_x=paste_posx, paste_y=paste_posy, ratio=paste_ratio)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_random_bbox_crop_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_box_shape = [200, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size = test_box_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    test_lables_shape = [200, 1]
    def get_lables():
        out = [np.random.randint(0, 255, size = test_lables_shape, dtype = np.int32) for _ in range(batch_size)]
        return out
    boxes = fn.external_source(source = get_boxes)
    lables = fn.external_source(source = get_lables)
    processed, _, _, _ = fn.random_bbox_crop(boxes, lables,
                                             aspect_ratio=[0.5, 2.0],
                                             thresholds=[0.1, 0.3, 0.5],
                                             scaling=[0.8, 1.0],
                                             bbox_layout="xyXY")
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_ssd_random_crop_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_box_shape = [200, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size = test_box_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    test_lables_shape = [200]
    def get_lables():
        out = [np.random.randint(0, 255, size = test_lables_shape, dtype = np.int32) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data)
    boxes = fn.external_source(source = get_boxes)
    lables = fn.external_source(source = get_lables)
    processed, _, _ = fn.ssd_random_crop(data, boxes, lables)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_sequence_rearrange_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [5, 10, 20, 3]
    def get_data():
        out = [np.random.randint(0, 255, size = test_data_shape, dtype = np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source = get_data, layout = "FHWC")
    processed = fn.sequence_rearrange(data, new_order = [0, 4, 1, 3, 2])
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_box_encoder_cpu():
    def coco_anchors():
        anchors = []

        fig_size = 300
        feat_sizes = [38, 19, 10, 5, 3, 1]
        feat_count = len(feat_sizes)
        steps = [8., 16., 32., 64., 100., 300.]
        scales = [21., 45., 99., 153., 207., 261., 315.]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        fks = []
        for step in steps:
            fks.append(fig_size / step)

        anchor_idx = 0
        for idx in range(feat_count):
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)

            all_sizes = [[sk1, sk1], [sk3, sk3]]

            for alpha in aspect_ratios[idx]:
                w = sk1 * sqrt(alpha)
                h = sk1 / sqrt(alpha)
                all_sizes.append([w, h])
                all_sizes.append([h, w])

            for sizes in all_sizes:
                w, h = sizes[0], sizes[1]

                for i in range(feat_sizes[idx]):
                    for j in range(feat_sizes[idx]):
                        cx = (j + 0.5) / fks[idx]
                        cy = (i + 0.5) / fks[idx]

                        cx = max(min(cx, 1.), 0.)
                        cy = max(min(cy, 1.), 0.)
                        w = max(min(w, 1.), 0.)
                        h = max(min(h, 1.), 0.)

                        anchors.append(cx - 0.5 * w)
                        anchors.append(cy - 0.5 * h)
                        anchors.append(cx + 0.5 * w)
                        anchors.append(cy + 0.5 * h)

                        anchor_idx = anchor_idx + 1
        return anchors

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_box_shape = [20, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size = test_box_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    test_lables_shape = [20, 1]
    def get_lables():
        out = [np.random.randint(0, 255, size = test_lables_shape, dtype = np.int32) for _ in range(batch_size)]
        return out
    boxes = fn.external_source(source = get_boxes)
    lables = fn.external_source(source = get_lables)
    processed, _ = fn.box_encoder(boxes, lables, anchors=coco_anchors())
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_numpy_reader_cpu():
    with tempfile.TemporaryDirectory() as test_data_root:
        rng = np.random.RandomState(12345)
        def create_numpy_file(filename, shape, typ, fortran_order):
            # generate random array
            arr = rng.random_sample(shape) * 10.
            arr = arr.astype(typ)
            if fortran_order:
                arr = np.asfortranarray(arr)
            np.save(filename, arr)

        num_samples = 20
        filenames = []
        for index in range(0, num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)

        pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
        processed = fn.readers.numpy(file_root = test_data_root)
        pipe.set_outputs(processed)
        pipe.build()
        for _ in range(3):
            pipe.run()

def test_python_function_cpu():
    def resize(image):
        return np.array(Image.fromarray(image).resize((50, 10)))

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None, exec_async=False, exec_pipelined=False)
    with pipe:
        data = fn.external_source(source = get_data, layout = "HWC")
        processed = fn.python_function(data, function=resize)
        pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_constant_cpu():
    check_no_input(fn.constant, fdata = (1.25,2.5,3))

def test_dump_image_cpu():
    check_single_input(fn.dump_image)

def test_sequence_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    processed = fn.readers.sequence(file_root=sequence_dir, sequence_length=2, shard_id=0, num_shards=1)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_affine_translate_cpu():
    check_no_input(fn.transforms.translation, offset=(2, 3))

def test_affine_scale_cpu():
    check_no_input(fn.transforms.scale, scale=(2, 3))

def test_affine_rotate_cpu():
    check_no_input(fn.transforms.rotation, angle=30.0)

def test_affine_shear_cpu():
    check_no_input(fn.transforms.shear, shear=(2., 1.))

def test_affine_crop_cpu():
    check_no_input(fn.transforms.crop,
        from_start=(0., 1.), from_end=(1., 1.), to_start=(0.2, 0.3), to_end=(0.8, 0.5))

def test_combine_transforms_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    with pipe:
        t = fn.transforms.translation(offset=(1, 2))
        r = fn.transforms.rotation(angle=30.0)
        s = fn.transforms.scale(scale=(2, 3))
        out = fn.transforms.combine(t, r, s)
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_reduce_min_cpu():
    check_single_input(fn.reductions.min)

def test_reduce_max_cpu():
    check_single_input(fn.reductions.max)

def test_reduce_sum_cpu():
    check_single_input(fn.reductions.sum)

def test_segmentation_select_masks():
    def get_data_source(*args, **kwargs):
        return lambda: make_batch_select_masks(*args, **kwargs)
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None, seed=1234)
    with pipe:
        polygons, vertices, selected_masks = fn.external_source(
            num_outputs = 3, device='cpu',
            source = get_data_source(batch_size, vertex_ndim=2, npolygons_range=(1, 5),
                                     nvertices_range=(3, 10))
        )
        out_polygons, out_vertices = fn.segmentation.select_masks(
            selected_masks, polygons, vertices, reindex_masks=False
        )
    pipe.set_outputs(polygons, vertices, selected_masks, out_polygons, out_vertices)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_reduce_mean_cpu():
    check_single_input(fn.reductions.mean)

def test_reduce_mean_square_cpu():
    check_single_input(fn.reductions.mean_square)

def test_reduce_root_mean_square_cpu():
    check_single_input(fn.reductions.rms)

def test_reduce_std_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_data)
    mean = fn.reductions.mean(data)
    reduced = fn.reductions.std_dev(data, mean)
    pipe.set_outputs(reduced)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_reduce_variance_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_data)
    mean = fn.reductions.mean(data)
    reduced = fn.reductions.variance(data, mean)
    pipe.set_outputs(reduced)

def test_arithm_ops_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_data, layout = "HWC")
    processed = [data * 2, data + 2, data - 2, data / 2, data // 2]
    pipe.set_outputs(*processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_arithm_ops_cpu_gpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source = get_data, layout = "HWC")
    processed = [data * data.gpu(), data + data.gpu(), data - data.gpu(), data / data.gpu(), data // data.gpu()]
    pipe.set_outputs(*processed)
    assert_raises(RuntimeError, pipe.build)

def test_pytorch_plugin_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    outs = fn.external_source(source = get_data, layout = "HWC")
    pipe.set_outputs(outs)
    pii = DALIGenericIterator([pipe], ["data"])

def test_random_mask_pixel_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source = get_data, layout = "HWC")
    pixel_pos = fn.segmentation.random_mask_pixel(data)
    pipe.set_outputs(pixel_pos)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_cat_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source = get_data, layout = "HWC")
    data2 = fn.external_source(source = get_data, layout = "HWC")
    data3 = fn.external_source(source = get_data, layout = "HWC")
    pixel_pos = fn.cat(data, data2, data3)
    pipe.set_outputs(pixel_pos)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_stack_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source = get_data, layout = "HWC")
    data2 = fn.external_source(source = get_data, layout = "HWC")
    data3 = fn.external_source(source = get_data, layout = "HWC")
    pixel_pos = fn.stack(data, data2, data3)
    pipe.set_outputs(pixel_pos)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_separated_exec_setup():
    batch_size = 128
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None, prefetch_queue_depth = {"cpu_size": 5, "gpu_size": 3})
    inputs, labels = fn.readers.caffe(path = caffe_dir, shard_id = 0, num_shards = 1)
    images = fn.decoders.image(inputs, output_type = types.RGB)
    images = fn.resize(images, resize_x=224, resize_y=224)
    images_cpu = fn.dump_image(images, suffix = "cpu")
    pipe.set_outputs(images, images_cpu)

    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    assert(out[0].as_tensor().shape() == out[1].as_tensor().shape())
    a_raw = out[0]
    a_cpu = out[1]
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_raw)) == 0)

# ToDo add tests for DLTensorPythonFunction if easily possible
