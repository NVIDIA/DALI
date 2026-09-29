# Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.fn as fn
import os
import tempfile
import json
from nvidia.dali import Pipeline, pipeline_def

from nose_utils import assert_raises, raises
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
            assert ids_map[int(out[0].at(s).item())] == int(
                out[1].at(s).item()
            ), f"{i}, {ids_map[int(out[0].at(s).item())]} vs {out[1].at(s).item()}"
            i = i + 1


@pipeline_def(batch_size=1, num_threads=1, device_id=None)
def coco_invalid_annotations_pipe(annotations_file, file_root):
    inputs, boxes, labels = fn.readers.coco(
        file_root=file_root,
        annotations_file=annotations_file,
    )
    return inputs, boxes, labels


@params([], [1, 2, 3], [1, 2, 3, 4, 5])
def test_operator_coco_reader_rejects_invalid_bbox_size(bbox):
    annotations = {
        "images": [
            {
                "id": 1,
                "width": 640,
                "height": 480,
                "file_name": "car-race-438467_1280.jpg",
            }
        ],
        "categories": [{"id": 1}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "iscrowd": 0,
            }
        ],
    }

    with tempfile.TemporaryDirectory() as annotations_dir:
        annotations_file = os.path.join(annotations_dir, "instances.json")
        with open(annotations_file, "w") as f:
            json.dump(annotations, f)

        pipe = coco_invalid_annotations_pipe(annotations_file, file_root)
        assert_raises(
            ValueError,
            pipe.run,
            glob="*Invalid COCO annotation: `bbox` must contain exactly 4 values*",
        )


@params((None, "null"), (123, "number"), (True, "boolean"), ({}, "object"), ([], "array"))
def test_operator_coco_reader_rejects_non_string_filename(filename, filename_type):
    annotations = {
        "images": [{"id": 1, "width": 640, "height": 480, "file_name": filename}],
        "categories": [],
        "annotations": [],
    }

    with tempfile.TemporaryDirectory() as annotations_dir:
        annotations_file = os.path.join(annotations_dir, "instances.json")
        with open(annotations_file, "w") as f:
            json.dump(annotations, f)

        pipe = coco_invalid_annotations_pipe(annotations_file, annotations_dir)
        assert_raises(
            ValueError,
            pipe.run,
            glob=f"*Invalid COCO annotation: `file_name` must be a string, got {filename_type}*",
        )


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
        image_ids = int(image_ids.as_array().item())
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
        image_ids = int(image_ids.as_array().item())
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
        image_ids = int(image_ids.as_array().item())
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


def _write_keypoints_dataset(root, num_keypoints=17, extra_annotations=()):
    """Writes a synthetic COCO keypoints dataset. Returns the parsed annotations dict."""
    rng = np.random.default_rng(1234)
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir)
    image_sizes = {1: (640, 480), 2: (800, 600), 3: (320, 240), 4: (1024, 768)}
    images = []
    for img_id, (w, h) in image_sizes.items():
        file_name = f"img_{img_id}.jpg"
        with open(os.path.join(images_dir, file_name), "wb") as f:
            f.write(bytes([img_id]) * 16)
        images.append({"id": img_id, "width": w, "height": h, "file_name": file_name})

    def person(ann_id, img_id):
        w, h = image_sizes[img_id]
        kps = []
        for _ in range(num_keypoints):
            v = int(rng.integers(0, 3))
            x, y = (float(rng.integers(0, w)), float(rng.integers(0, h))) if v else (0.0, 0.0)
            kps += [x, y, v]
        return {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "iscrowd": 0,
            "keypoints": kps,
            "num_keypoints": sum(1 for v in kps[2::3] if v > 0),
        }

    def other(ann_id, img_id):
        return {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 2,
            "bbox": [1.0, 2.0, 50.0, 60.0],
            "iscrowd": 0,
        }

    annotations = [
        person(1, 1),
        person(2, 1),
        other(3, 2),  # image with a mix of annotations with and without keypoints
        person(4, 2),
        other(5, 2),
        # image 3 has no annotations
        other(6, 4),  # image with annotations, none of which have keypoints
        *extra_annotations,
    ]
    data = {
        "images": images,
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "other"}],
        "annotations": annotations,
    }
    with open(os.path.join(root, "annotations.json"), "w") as f:
        json.dump(data, f)
    return data


def _ref_keypoints(data, image_id, num_keypoints, ratio):
    img = next(i for i in data["images"] if i["id"] == image_id)
    anns = [a for a in data["annotations"] if a["image_id"] == image_id]
    out = np.zeros((len(anns), num_keypoints, 3), dtype=np.float32)
    for i, a in enumerate(anns):
        if "keypoints" in a:
            out[i] = np.array(a["keypoints"], dtype=np.float32).reshape(num_keypoints, 3)
    if ratio:
        out[:, :, 0] /= img["width"]
        out[:, :, 1] /= img["height"]
    return out


@pipeline_def(batch_size=1, num_threads=1, device_id=None)
def coco_keypoints_pipe(**kwargs):
    _, boxes, labels, keypoints, ids = fn.readers.coco(keypoints=True, image_ids=True, **kwargs)
    return boxes, labels, keypoints, ids


def _run_keypoints_pipe(**kwargs):
    pipe = coco_keypoints_pipe(**kwargs)
    (epoch_size,) = pipe.epoch_size().values()
    results = {}
    for _ in range(epoch_size):
        boxes, labels, keypoints, ids = (np.array(o.at(0)) for o in pipe.run())
        results[int(ids.item())] = (boxes, labels, keypoints)
    return results


@params(*[(ratio, skip_empty) for ratio in (False, True) for skip_empty in (False, True)])
def test_coco_keypoints(ratio, skip_empty):
    num_keypoints = 17
    with tempfile.TemporaryDirectory() as root:
        data = _write_keypoints_dataset(root, num_keypoints)
        results = _run_keypoints_pipe(
            file_root=os.path.join(root, "images"),
            annotations_file=os.path.join(root, "annotations.json"),
            ratio=ratio,
            skip_empty=skip_empty,
        )
    expected_ids = {1, 2, 4} if skip_empty else {1, 2, 3, 4}
    assert set(results.keys()) == expected_ids, f"{results.keys()} vs {expected_ids}"
    for image_id, (boxes, labels, keypoints) in results.items():
        ref = _ref_keypoints(data, image_id, num_keypoints, ratio)
        assert keypoints.dtype == np.float32
        assert keypoints.shape == ref.shape, f"{keypoints.shape} vs {ref.shape}"
        assert keypoints.shape[0] == boxes.shape[0] == labels.shape[0]
        np.testing.assert_allclose(keypoints, ref, rtol=1e-6)


def test_coco_keypoints_none_defined():
    with tempfile.TemporaryDirectory() as root:
        _write_keypoints_dataset(root)
        data_file = os.path.join(root, "annotations.json")
        with open(data_file) as f:
            data = json.load(f)
        for a in data["annotations"]:
            a.pop("keypoints", None)
        with open(data_file, "w") as f:
            json.dump(data, f)
        results = _run_keypoints_pipe(
            file_root=os.path.join(root, "images"), annotations_file=data_file
        )
    for boxes, _, keypoints in results.values():
        assert keypoints.shape == (boxes.shape[0], 0, 3), keypoints.shape


def test_coco_keypoints_size_threshold():
    small_person = {
        "id": 100,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "iscrowd": 0,
        "keypoints": [1.0, 1.0, 2.0] * 17,
    }
    with tempfile.TemporaryDirectory() as root:
        data = _write_keypoints_dataset(root, extra_annotations=[small_person])
        results = _run_keypoints_pipe(
            file_root=os.path.join(root, "images"),
            annotations_file=os.path.join(root, "annotations.json"),
            size_threshold=5.0,
        )
    data["annotations"] = [a for a in data["annotations"] if a["id"] != small_person["id"]]
    boxes, _, keypoints = results[1]
    np.testing.assert_array_equal(keypoints, _ref_keypoints(data, 1, 17, False))
    assert keypoints.shape[0] == boxes.shape[0] == 2


@params(True, False)
def test_coco_keypoints_preprocessed_annotations(ratio):
    with tempfile.TemporaryDirectory() as root:
        _write_keypoints_dataset(root)
        preprocessed_dir = os.path.join(root, "preprocessed")
        os.makedirs(preprocessed_dir)
        file_root = os.path.join(root, "images")
        ref = _run_keypoints_pipe(
            file_root=file_root,
            annotations_file=os.path.join(root, "annotations.json"),
            ratio=ratio,
            save_preprocessed_annotations=True,
            save_preprocessed_annotations_dir=preprocessed_dir,
        )
        out = _run_keypoints_pipe(file_root=file_root, preprocessed_annotations=preprocessed_dir)
    assert ref.keys() == out.keys()
    for image_id in ref:
        for ref_arr, out_arr in zip(ref[image_id], out[image_id]):
            np.testing.assert_array_equal(ref_arr, out_arr)


def test_coco_keypoints_preprocessed_annotations_missing():
    with tempfile.TemporaryDirectory() as root:
        _write_keypoints_dataset(root)
        preprocessed_dir = os.path.join(root, "preprocessed")
        os.makedirs(preprocessed_dir)
        file_root = os.path.join(root, "images")

        @pipeline_def(batch_size=1, num_threads=1, device_id=None)
        def save_pipe():
            _, boxes, _, ids = fn.readers.coco(
                file_root=file_root,
                annotations_file=os.path.join(root, "annotations.json"),
                image_ids=True,
                save_preprocessed_annotations=True,
                save_preprocessed_annotations_dir=preprocessed_dir,
            )
            return boxes, ids

        save_pipe().run()
        pipe = coco_keypoints_pipe(file_root=file_root, preprocessed_annotations=preprocessed_dir)
        assert_raises(
            RuntimeError,
            pipe.run,
            glob="*Keypoints were requested, but the preprocessed annotations*",
        )


@params(
    ([1.0, 2.0], "*must be a multiple of 3*"),
    ("abc", "*`keypoints` must be an array, got string*"),
    ([1.0, None, 2.0], "*`keypoints` must contain only numbers, got null*"),
    ([1.0, 2.0, 2.0] * 5, "*must have the same number of keypoints, got 17 and 5*"),
)
def test_coco_keypoints_invalid(keypoints, error_glob):
    invalid = {
        "id": 100,
        "image_id": 3,
        "category_id": 1,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "iscrowd": 0,
        "keypoints": keypoints,
    }
    with tempfile.TemporaryDirectory() as root:
        _write_keypoints_dataset(root, extra_annotations=[invalid])
        pipe = coco_keypoints_pipe(
            file_root=os.path.join(root, "images"),
            annotations_file=os.path.join(root, "annotations.json"),
        )
        assert_raises(ValueError, pipe.run, glob=error_glob)


def test_coco_keypoints_ndd():
    with tempfile.TemporaryDirectory() as root:
        _write_keypoints_dataset(root)
        kwargs = dict(
            file_root=os.path.join(root, "images"),
            annotations_file=os.path.join(root, "annotations.json"),
            ratio=True,
        )
        ref = _run_keypoints_pipe(**kwargs)
        reader = ndd.readers.COCO(keypoints=True, image_ids=True, **kwargs)
        out = {}
        for _, boxes, labels, keypoints, ids in reader.next_epoch():
            out[int(np.asarray(ids).item())] = tuple(
                np.asarray(x) for x in (boxes, labels, keypoints)
            )
    assert ref.keys() == out.keys()
    for image_id in ref:
        for ref_arr, out_arr in zip(ref[image_id], out[image_id]):
            np.testing.assert_array_equal(ref_arr, out_arr)
