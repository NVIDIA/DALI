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
import nvidia.dali.fn as fn
import os
import tempfile
import json
from nvidia.dali import Pipeline, pipeline_def

from nose_utils import raises
from nose2.tools import params
from test_utils import compare_pipelines, get_dali_extra_path

test_data_root = get_dali_extra_path()
file_root = os.path.join(test_data_root, "db", "coco", "images")
train_annotations = os.path.join(test_data_root, "db", "coco", "instances.json")


class sample_desc:
    def __init__(self, id, cls, mapped_cls):
        self.id = id
        self.cls = cls
        self.mapped_cls = mapped_cls


test_data = {
    "car-race-438467_1280.jpg": sample_desc(17, 5, 6),
    "clock-1274699_1280.jpg": sample_desc(6, 7, 8),
    "kite-1159538_1280.jpg": sample_desc(21, 12, 13),
    "cow-234835_1280.jpg": sample_desc(59, 8, 9),
    "home-office-336378_1280.jpg": sample_desc(39, 13, 14),
    "suit-2619784_1280.jpg": sample_desc(0, 16, 17),
    "business-suit-690048_1280.jpg": sample_desc(5, 16, 17),
    "car-604019_1280.jpg": sample_desc(41, 5, 6),
}

images = list(test_data.keys())
expected_ids = list(s.id for s in test_data.values())


def check_operator_coco_reader_custom_order(order=None, add_invalid_paths=False):
    batch_size = 2
    if not order:
        order = range(len(test_data))
    keys = list(test_data.keys())
    values = list(s.id for s in test_data.values())
    images = [keys[i] for i in order]
    images_arg = images.copy()
    if add_invalid_paths:
        images_arg += ["/invalid/path/image.png"]
    expected_ids = [values[i] for i in order]
    with tempfile.TemporaryDirectory() as annotations_dir:
        pipeline = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        with pipeline:
            _, _, _, ids = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                image_ids=True,
                images=images_arg,
                save_preprocessed_annotations=True,
                save_preprocessed_annotations_dir=annotations_dir,
            )
            pipeline.set_outputs(ids)

        i = 0
        assert len(images) % batch_size == 0
        while i < len(images):
            out = pipeline.run()
            for s in range(batch_size):
                assert out[0].at(s) == expected_ids[i], f"{i}, {expected_ids}"
                i = i + 1

        filenames_file = os.path.join(annotations_dir, "filenames.dat")
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


@params(True, False)
def test_operator_coco_reader_label_remap(avoid_remap):
    batch_size = 2
    images = list(test_data.keys())
    ids_map = {s.id: s.cls if avoid_remap else s.mapped_cls for s in test_data.values()}

    pipeline = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipeline:
        _, _, labels, ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            images=images,
            avoid_class_remapping=avoid_remap,
        )
        pipeline.set_outputs(ids, labels)

    i = 0
    assert len(images) % batch_size == 0
    while i < len(images):
        out = pipeline.run()
        for s in range(batch_size):
            print(out[0].at(s), out[1].at(s))
            assert ids_map[int(out[0].at(s))] == int(
                out[1].at(s)
            ), f"{i}, {ids_map[int(out[0].at(s))]} vs {out[1].at(s)}"
            i = i + 1


def test_operator_coco_reader_same_images():
    file_root = os.path.join(test_data_root, "db", "coco_pixelwise", "images")
    train_annotations = os.path.join(test_data_root, "db", "coco_pixelwise", "instances.json")

    coco_dir = os.path.join(test_data_root, "db", "coco")
    coco_dir_imgs = os.path.join(coco_dir, "images")
    coco_pixelwise_dir = os.path.join(test_data_root, "db", "coco_pixelwise")
    coco_pixelwise_dir_imgs = os.path.join(coco_pixelwise_dir, "images")

    for file_root, _ in [
        (coco_dir_imgs, os.path.join(coco_dir, "instances.json")),
        (coco_pixelwise_dir_imgs, os.path.join(coco_pixelwise_dir, "instances.json")),
        (coco_pixelwise_dir_imgs, os.path.join(coco_pixelwise_dir, "instances_rle_counts.json")),
    ]:
        pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
        with pipe:
            inputs1, boxes1, labels1, *_ = fn.readers.coco(
                file_root=file_root, annotations_file=train_annotations, name="reader1", seed=1234
            )
            inputs2, boxes2, labels2, *_ = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                polygon_masks=True,
                name="reader2",
            )
            inputs3, boxes3, labels3, *_ = fn.readers.coco(
                file_root=file_root,
                annotations_file=train_annotations,
                pixelwise_masks=True,
                name="reader3",
            )
            pipe.set_outputs(
                inputs1, boxes1, labels1, inputs2, boxes2, labels2, inputs3, boxes3, labels3
            )

        epoch_sz = pipe.epoch_size("reader1")
        assert epoch_sz == pipe.epoch_size("reader2")
        assert epoch_sz == pipe.epoch_size("reader3")

        for _ in range(epoch_sz):
            (
                inputs1,
                boxes1,
                labels1,
                inputs2,
                boxes2,
                labels2,
                inputs3,
                boxes3,
                labels3,
            ) = pipe.run()
            np.testing.assert_array_equal(inputs1.at(0), inputs2.at(0))
            np.testing.assert_array_equal(inputs1.at(0), inputs3.at(0))
            np.testing.assert_array_equal(labels1.at(0), labels2.at(0))
            np.testing.assert_array_equal(labels1.at(0), labels3.at(0))
            np.testing.assert_array_equal(boxes1.at(0), boxes2.at(0))
            np.testing.assert_array_equal(boxes1.at(0), boxes3.at(0))


@raises(
    KeyError,
    glob='Argument "preprocessed_annotations_dir" is not defined for operator *readers*COCO',
)
def test_invalid_args():
    pipeline = Pipeline(batch_size=2, num_threads=4, device_id=0)
    with pipeline:
        _, _, _, ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            images=images,
            preprocessed_annotations_dir="/tmp",
        )
        pipeline.set_outputs(ids)


batch_size_alias_test = 64


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def coco_pipe(coco_op, file_root, annotations_file, polygon_masks, pixelwise_masks):
    inputs, boxes, labels, *_ = coco_op(
        file_root=file_root,
        annotations_file=annotations_file,
        polygon_masks=polygon_masks,
        pixelwise_masks=pixelwise_masks,
    )
    return inputs, boxes, labels


def test_coco_reader_alias():
    def check_coco_reader_alias(polygon_masks, pixelwise_masks):
        new_pipe = coco_pipe(
            fn.readers.coco, file_root, train_annotations, polygon_masks, pixelwise_masks
        )
        legacy_pipe = coco_pipe(
            fn.coco_reader, file_root, train_annotations, polygon_masks, pixelwise_masks
        )
        compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 5)

    file_root = os.path.join(test_data_root, "db", "coco_pixelwise", "images")
    train_annotations = os.path.join(test_data_root, "db", "coco_pixelwise", "instances.json")

    for polygon_masks, pixelwise_masks in [(None, None), (True, None), (None, True)]:
        yield check_coco_reader_alias, polygon_masks, pixelwise_masks


@params(True, False)
def test_coco_include_crowd(include_iscrowd):
    @pipeline_def(batch_size=1, device_id=0, num_threads=4)
    def coco_pipe(include_iscrowd):
        _, boxes, _, image_ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            include_iscrowd=include_iscrowd,
        )
        return boxes, image_ids

    annotations = None
    with open(train_annotations) as file:
        annotations = json.load(file)

    pipe = coco_pipe(include_iscrowd=include_iscrowd)
    number_of_samples = pipe.epoch_size()
    for k in number_of_samples:
        # there is only one reader
        number_of_samples = number_of_samples[k]
        break

    anno_mapping = {}
    for elm in annotations["annotations"]:
        image_id = elm["image_id"]
        if not anno_mapping.get(image_id):
            anno_mapping[image_id] = {"bbox": [], "iscrowd": []}
        anno_mapping[image_id]["bbox"].append(elm["bbox"])
        anno_mapping[image_id]["iscrowd"].append(elm["iscrowd"])

    all_iscrowd = []
    for _ in range(number_of_samples):
        boxes, image_ids = pipe.run()
        image_ids = int(image_ids.as_array())
        boxes = boxes.as_array()[0]
        anno = anno_mapping[image_ids]
        idx = 0
        # it assumes that the coco reader reads annotations at the order of appearance inside JSON
        all_iscrowd += anno["iscrowd"]
        for j, iscrowd in enumerate(anno["iscrowd"]):
            if include_iscrowd or iscrowd == 0:
                assert np.all(boxes[idx] == np.array(anno["bbox"][j]))
                idx += 1
    assert any(all_iscrowd), "At least one annotation should include `iscrowd=1`"


def test_coco_empty_annotations_pix():
    file_root = os.path.join(test_data_root, "db", "coco_dummy", "images")
    train_annotations = os.path.join(test_data_root, "db", "coco_dummy", "instances.json")

    @pipeline_def(batch_size=1, device_id=0, num_threads=4)
    def coco_pipe():
        _, _, _, masks, ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            pixelwise_masks=True,
        )
        return masks, ids

    pipe = coco_pipe()
    number_of_samples = pipe.epoch_size()
    for k in number_of_samples:
        # there is only one reader
        number_of_samples = number_of_samples[k]
        break

    annotations = None
    with open(train_annotations) as file:
        annotations = json.load(file)

    anno_mapping = {}
    for elm in annotations["annotations"]:
        image_id = elm["image_id"]
        anno_mapping[image_id] = anno_mapping.get(image_id, False) or "segmentation" in elm

    for _ in range(number_of_samples):
        mask, image_ids = pipe.run()
        image_ids = int(image_ids.as_array())
        max_mask = np.max(np.array(mask.as_tensor()))
        assert (max_mask != 0 and image_ids in anno_mapping and anno_mapping[image_ids]) or (
            max_mask == 0 and not (image_ids in anno_mapping and anno_mapping[image_ids])
        )


def test_coco_empty_annotations_poly():
    file_root = os.path.join(test_data_root, "db", "coco_dummy", "images")
    train_annotations = os.path.join(test_data_root, "db", "coco_dummy", "instances.json")

    @pipeline_def(batch_size=1, device_id=0, num_threads=4)
    def coco_pipe():
        _, _, _, poly, vert, ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            polygon_masks=True,
        )
        return poly, vert, ids

    pipe = coco_pipe()
    number_of_samples = pipe.epoch_size()
    for k in number_of_samples:
        # there is only one reader
        number_of_samples = number_of_samples[k]
        break

    annotations = None
    with open(train_annotations) as file:
        annotations = json.load(file)

    anno_mapping = {}
    for elm in annotations["annotations"]:
        image_id = elm["image_id"]
        anno_mapping[image_id] = anno_mapping.get(image_id, False) or "segmentation" in elm

    for _ in range(number_of_samples):
        poly, vert, image_ids = pipe.run()
        image_ids = int(image_ids.as_array())
        poly = np.array(poly.as_tensor()).size
        vert = np.array(vert.as_tensor()).size
        assert (poly != 0 and image_ids in anno_mapping and anno_mapping[image_ids]) or (
            vert == 0 and not (image_ids in anno_mapping and anno_mapping[image_ids])
        )


def test_coco_pix_mask_ratio():
    file_root = os.path.join(test_data_root, "db", "coco_dummy", "images")
    train_annotations = os.path.join(test_data_root, "db", "coco_dummy", "instances.json")

    batch_size = 2

    @pipeline_def(batch_size=1, device_id=0, num_threads=4)
    def coco_pipe(ratio=False):
        _, _, _, masks = fn.readers.coco(
            file_root=file_root,
            annotations_file=train_annotations,
            pixelwise_masks=True,
            ratio=ratio,
        )
        return masks

    pipe_ref = coco_pipe(batch_size=batch_size, ratio=False)
    pipe_test = coco_pipe(batch_size=batch_size, ratio=True)
    compare_pipelines(pipe_ref, pipe_test, batch_size, 5)
