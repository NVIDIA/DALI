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

import nose_utils  # noqa:F401
import os
import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()


def check_bbox_random_crop_adjust_polygons(
    file_root, annotations_file, batch_size=3, num_iters=4, num_threads=4, device_id=0, seed=1234
):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    with pipe:
        # Read data from COCO
        # ratio=True means both bboxes and masks coordinates will be
        # relative to the image dimensions (range [0.0, 1.0])
        inputs, in_bboxes, labels, in_polygons, in_vertices = fn.readers.coco(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=0,
            num_shards=1,
            ratio=True,
            ltrb=True,
            polygon_masks=True,
        )

        # Generate a random crop. out_bboxes are adjusted to the crop window
        slice_anchor, slice_shape, out_bboxes, labels, bbox_indices = fn.random_bbox_crop(
            in_bboxes,
            labels,
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            bbox_layout="xyXY",
            output_bbox_indices=True,
        )
        # Crop the image
        _ = fn.decoders.image_slice(
            inputs, slice_anchor, slice_shape, device="mixed", axis_names="WH"
        )

        sel_polygons, sel_vertices = fn.segmentation.select_masks(
            bbox_indices, in_polygons, in_vertices
        )

        # Adjust masks coordinates to the coordinate space of the cropped image
        MT = fn.transforms.crop(from_start=slice_anchor, from_end=(slice_anchor + slice_shape))
        out_vertices = fn.coord_transform(sel_vertices, MT=MT)

        # Converting to absolute coordinates (demo purposes)
        image_shape = fn.peek_image_shape(inputs, dtype=types.FLOAT)
        h = fn.slice(image_shape, 0, 1, axes=[0])
        w = fn.slice(image_shape, 1, 1, axes=[0])

        # Original bboxes
        bbox_x = fn.slice(in_bboxes, 0, 1, axes=[1])
        bbox_y = fn.slice(in_bboxes, 1, 1, axes=[1])
        bbox_X = fn.slice(in_bboxes, 2, 1, axes=[1])
        bbox_Y = fn.slice(in_bboxes, 3, 1, axes=[1])
        in_bboxes_abs = fn.cat(bbox_x * w, bbox_y * h, bbox_X * w, bbox_Y * h, axis=1)

        # Transform to convert relative coordinates to absolute
        scale_rel_to_abs = fn.transforms.scale(scale=fn.cat(w, h))

        # Selected vertices (relative coordinates)
        sel_vertices_abs = fn.coord_transform(out_vertices, MT=scale_rel_to_abs)

        # Output bboxes
        bbox2_x = fn.slice(out_bboxes, 0, 1, axes=[1])
        bbox2_y = fn.slice(out_bboxes, 1, 1, axes=[1])
        bbox2_X = fn.slice(out_bboxes, 2, 1, axes=[1])
        bbox2_Y = fn.slice(out_bboxes, 3, 1, axes=[1])
        out_bboxes_abs = fn.cat(bbox2_x * w, bbox2_y * h, bbox2_X * w, bbox2_Y * h, axis=1)

        # Output vertices (absolute coordinates)
        out_vertices_abs = fn.coord_transform(out_vertices, MT=scale_rel_to_abs)

        # Clamped coordinates
        out_vertices_clamped = math.clamp(out_vertices, 0.0, 1.0)
        out_vertices_clamped_abs = fn.coord_transform(out_vertices_clamped, MT=scale_rel_to_abs)

    pipe.set_outputs(
        in_vertices,
        sel_vertices,
        sel_vertices_abs,
        out_vertices,
        out_vertices_clamped,
        out_vertices_abs,
        out_vertices_clamped_abs,
        in_bboxes,
        in_bboxes_abs,
        out_bboxes,
        out_bboxes_abs,
        in_polygons,
        sel_polygons,
        image_shape,
        slice_anchor,
        slice_shape,
        bbox_indices,
    )
    # Enough iterations to see an example with more than one bounding box
    for i in range(num_iters):
        outs = pipe.run()
        for j in range(batch_size):
            (
                in_vertices,
                sel_vertices,
                sel_vertices_abs,
                out_vertices,
                out_vertices_clamped,
                out_vertices_abs,
                out_vertices_clamped_abs,
                in_bboxes,
                in_bboxes_abs,
                out_bboxes,
                out_bboxes_abs,
                in_polygons,
                sel_polygons,
                image_shape,
                slice_anchor,
                slice_shape,
                bbox_indices,
            ) = (outs[k].at(j) for k in range(len(outs)))

            # Checking that the output polygon descriptors are the ones associated with the
            # selected bounding boxes
            expected_polygons_list = []
            expected_vertices_list = []
            ver_count = 0
            for k in range(in_polygons.shape[0]):
                mask_id = in_polygons[k][0]
                in_ver_start_idx = in_polygons[k][1]
                in_ver_end_idx = in_polygons[k][2]
                pol_nver = in_ver_end_idx - in_ver_start_idx
                if mask_id in bbox_indices:
                    expected_polygons_list.append([mask_id, ver_count, ver_count + pol_nver])
                    for j in range(in_ver_start_idx, in_ver_end_idx):
                        expected_vertices_list.append(in_vertices[j])
                    ver_count = ver_count + pol_nver
            expected_sel_polygons = np.array(expected_polygons_list)
            np.testing.assert_equal(expected_sel_polygons, sel_polygons)

            # Checking the selected vertices correspond to the selected masks
            expected_sel_vertices = np.array(expected_vertices_list)
            np.testing.assert_equal(expected_sel_vertices, sel_vertices)

            # Check that the vertices are correctly mapped to the cropping window
            expected_out_vertices = np.copy(expected_sel_vertices)
            crop_x, crop_y = slice_anchor
            crop_w, crop_h = slice_shape
            for v in range(expected_out_vertices.shape[0]):
                expected_out_vertices[v, 0] = (expected_out_vertices[v, 0] - crop_x) / crop_w
                expected_out_vertices[v, 1] = (expected_out_vertices[v, 1] - crop_y) / crop_h
            np.testing.assert_allclose(expected_out_vertices, out_vertices, rtol=1e-4)

            # Checking the conversion to absolute coordinates
            h, w, _ = image_shape
            wh = np.array([w, h])
            whwh = np.array([w, h, w, h])
            expected_out_vertices_abs = expected_out_vertices * wh
            np.testing.assert_allclose(expected_out_vertices_abs, out_vertices_abs, rtol=1e-4)

            # Checking clamping of the relative coordinates
            expected_out_vertices_clamped = np.clip(expected_out_vertices, a_min=0.0, a_max=1.0)
            np.testing.assert_allclose(
                expected_out_vertices_clamped, out_vertices_clamped, rtol=1e-4
            )

            # Checking clamping of the absolute coordinates
            expected_out_vertices_clamped_abs = np.clip(expected_out_vertices_abs, 0, wh)
            np.testing.assert_allclose(
                expected_out_vertices_clamped_abs, out_vertices_clamped_abs, rtol=1e-4
            )

            # Checking scaling of the bounding boxes
            expected_in_bboxes_abs = in_bboxes * whwh
            np.testing.assert_allclose(expected_in_bboxes_abs, in_bboxes_abs, rtol=1e-4)

            # Check box selection and mapping to the cropping window
            expected_out_bboxes = np.copy(in_bboxes[bbox_indices, :])
            for k in range(expected_out_bboxes.shape[0]):
                expected_out_bboxes[k, 0] = (expected_out_bboxes[k, 0] - crop_x) / crop_w
                expected_out_bboxes[k, 1] = (expected_out_bboxes[k, 1] - crop_y) / crop_h
                expected_out_bboxes[k, 2] = (expected_out_bboxes[k, 2] - crop_x) / crop_w
                expected_out_bboxes[k, 3] = (expected_out_bboxes[k, 3] - crop_y) / crop_h
            expected_out_bboxes = np.clip(expected_out_bboxes, a_min=0.0, a_max=1.0)
            np.testing.assert_allclose(expected_out_bboxes, out_bboxes, rtol=1e-4)

            expected_out_bboxes_abs = expected_out_bboxes * whwh
            np.testing.assert_allclose(expected_out_bboxes_abs, out_bboxes_abs, rtol=1e-4)


def test_bbox_random_crop_adjust_polygons():
    file_root = os.path.join(test_data_root, "db", "coco", "images")
    train_annotations = os.path.join(test_data_root, "db", "coco", "instances.json")
    check_bbox_random_crop_adjust_polygons(file_root, train_annotations, batch_size=3, num_iters=4)
