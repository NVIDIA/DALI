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

from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
from test_utils import compare_pipelines, get_dali_extra_path
import os
from nose.tools import raises
import tempfile
import numpy as np

test_data_root = get_dali_extra_path()
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
train_annotations = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

test_data = {
    'car-race-438467_1280.jpg' : 17,
    'clock-1274699_1280.jpg' : 6,
    'kite-1159538_1280.jpg' : 21,
    'cow-234835_1280.jpg' : 59,
    'home-office-336378_1280.jpg' : 39,
    'suit-2619784_1280.jpg' : 0,
    'business-suit-690048_1280.jpg' : 5,
    'car-604019_1280.jpg' : 41
}

images = list(test_data.keys())
expected_ids = list(test_data.values())


def check_operator_coco_reader_custom_order(order=None, add_invalid_paths=False):
    batch_size = 2
    if not order:
        order = range(len(test_data))
    keys = list(test_data.keys())
    values = list(test_data.values())
    images = [keys[i] for i in order]
    images_arg = images.copy()
    if add_invalid_paths:
        images_arg += ["/invalid/path/image.png"]
    expected_ids = [values[i] for i in order]
    with tempfile.TemporaryDirectory() as annotations_dir:
        pipeline = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        with pipeline:
            inputs, _, _, ids = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                image_ids=True,
                images=images_arg,
                save_preprocessed_annotations=True,
                save_preprocessed_annotations_dir=annotations_dir)
            pipeline.set_outputs(ids)
        pipeline.build()

        i = 0
        assert len(images) % batch_size == 0
        while i < len(images):
            out = pipeline.run()
            for s in range(batch_size):
                assert out[0].at(s) == expected_ids[i], f"{i}, {expected_ids}"
                i = i + 1

        filenames_file = os.path.join(annotations_dir, 'filenames.dat')
        with open(filenames_file) as f:
            lines = f.read().splitlines()
        assert lines.sort() == images.sort()

def test_operator_coco_reader_custom_order():
    custom_orders = [
        None,  # natural order
        [0, 2, 4, 6, 1, 3, 5, 7],  # altered order
        [0, 1, 2, 3, 2, 1, 4, 1, 5, 2, 6, 7],  # with repetitions
        ]

    for order in custom_orders:
        yield check_operator_coco_reader_custom_order, order, False
    yield check_operator_coco_reader_custom_order, None, True  # Natural order plus an invalid path


def test_operator_coco_reader_same_images():
    file_root = os.path.join(test_data_root, 'db', 'coco_pixelwise', 'images')
    train_annotations = os.path.join(test_data_root, 'db', 'coco_pixelwise', 'instances.json')

    coco_dir = os.path.join(test_data_root, 'db', 'coco')
    coco_dir_imgs = os.path.join(coco_dir, 'images')
    coco_pixelwise_dir = os.path.join(test_data_root, 'db', 'coco_pixelwise')
    coco_pixelwise_dir_imgs = os.path.join(coco_pixelwise_dir, 'images')

    for file_root, annotations_file in [ \
        (coco_dir_imgs, os.path.join(coco_dir, 'instances.json')),
        (coco_pixelwise_dir_imgs, os.path.join(coco_pixelwise_dir, 'instances.json')),
        (coco_pixelwise_dir_imgs, os.path.join(coco_pixelwise_dir, 'instances_rle_counts.json'))]:
        pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
        with pipe:
            inputs1, boxes1, labels1, *other = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                name="reader1",
                seed=1234
            )
            inputs2, boxes2, labels2, *other = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                polygon_masks=True,
                name="reader2"
            )
            inputs3, boxes3, labels3, *other = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                pixelwise_masks=True,
                name="reader3"
            )
            pipe.set_outputs(
                inputs1, boxes1, labels1,
                inputs2, boxes2, labels2,
                inputs3, boxes3, labels3
            )
        pipe.build()

        epoch_sz = pipe.epoch_size("reader1")
        assert epoch_sz == pipe.epoch_size("reader2")
        assert epoch_sz == pipe.epoch_size("reader3")

        for i in range(epoch_sz):
            inputs1, boxes1, labels1, inputs2, boxes2, labels2, inputs3, boxes3, labels3 = \
                pipe.run()
            np.testing.assert_array_equal(inputs1.at(0), inputs2.at(0))
            np.testing.assert_array_equal(inputs1.at(0), inputs3.at(0))
            np.testing.assert_array_equal(labels1.at(0), labels2.at(0))
            np.testing.assert_array_equal(labels1.at(0), labels3.at(0))
            np.testing.assert_array_equal(boxes1.at(0), boxes2.at(0))
            np.testing.assert_array_equal(boxes1.at(0), boxes3.at(0))

@raises(RuntimeError)
def test_invalid_args():
    pipeline = Pipeline(batch_size=2, num_threads=4, device_id=0)
    with pipeline:
        inputs, _, _, ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            images=images,
            preprocessed_annotations_dir='/tmp')
        pipeline.set_outputs(ids)
    pipeline.build()


batch_size_alias_test=64

@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def coco_pipe(coco_op, file_root, annotations_file, polygon_masks, pixelwise_masks):
    inputs, boxes, labels, *other = coco_op(file_root=file_root, annotations_file=annotations_file,
            polygon_masks=polygon_masks, pixelwise_masks=pixelwise_masks)
    return inputs, boxes, labels

def test_coco_reader_alias():
    file_root = os.path.join(test_data_root, 'db', 'coco_pixelwise', 'images')
    train_annotations = os.path.join(test_data_root, 'db', 'coco_pixelwise', 'instances.json')
    for polygon_masks, pixelwise_masks in [(None, None), (True, None), (None, True)]:
        new_pipe = coco_pipe(fn.readers.coco, file_root, train_annotations, polygon_masks, pixelwise_masks)
        legacy_pipe = coco_pipe(fn.coco_reader, file_root, train_annotations, polygon_masks, pixelwise_masks)
        compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)
