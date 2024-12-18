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

from nose_utils import assert_raises
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import random
from segmentation_test_utils import make_batch_select_masks

random.seed(1234)
np.random.seed(4321)


def check_select_masks(
    batch_size,
    npolygons_range=(1, 10),
    nvertices_range=(3, 40),
    vertex_ndim=2,
    vertex_dtype=np.float32,
    reindex_masks=False,
):
    def get_data_source(*args, **kwargs):
        return lambda: make_batch_select_masks(*args, **kwargs)

    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        polygons, vertices, mask_ids = fn.external_source(
            source=get_data_source(
                batch_size,
                npolygons_range=npolygons_range,
                nvertices_range=nvertices_range,
                vertex_ndim=vertex_ndim,
                vertex_dtype=vertex_dtype,
            ),
            num_outputs=3,
            device="cpu",
        )
        out_polygons, out_vertices = fn.segmentation.select_masks(
            mask_ids, polygons, vertices, reindex_masks=reindex_masks
        )
    pipe.set_outputs(polygons, vertices, mask_ids, out_polygons, out_vertices)
    for iter in range(3):
        outputs = pipe.run()
        for idx in range(batch_size):
            in_polygons = outputs[0].at(idx)
            in_vertices = outputs[1].at(idx)
            mask_ids = outputs[2].at(idx)
            out_polygons = outputs[3].at(idx)
            out_vertices = outputs[4].at(idx)

            in_polygons_dict = {}
            for k in range(in_polygons.shape[0]):
                mask_id = in_polygons[k, 0]
                in_polygons_dict[mask_id] = (in_polygons[k, 1], in_polygons[k, 2])

            if reindex_masks:
                index_map = {}
                for idx in range(len(mask_ids)):
                    index_map[mask_ids[idx]] = idx

            vertex_count = 0
            for m in range(len(mask_ids)):
                mask_id = mask_ids[m]
                in_vertex_start, in_vertex_end = in_polygons_dict[mask_id]
                in_nvertices = in_vertex_end - in_vertex_start

                expected_out_mask_id = index_map[mask_id] if reindex_masks else mask_id
                out_mask_id, out_vertex_start, out_vertex_end = out_polygons[m]
                assert out_mask_id == expected_out_mask_id
                assert out_vertex_start == vertex_count
                assert out_vertex_end == (vertex_count + in_nvertices)
                vertex_count = vertex_count + in_nvertices
                expected_out_vertex = in_vertices[in_vertex_start:in_vertex_end]
                out_vertex = out_vertices[out_vertex_start:out_vertex_end]
                assert (expected_out_vertex == out_vertex).all()


def test_select_masks():
    npolygons_range = (1, 10)
    nvertices_range = (3, 40)
    for batch_size in [1, 3]:
        for vertex_ndim in [2, 3, 6]:
            for vertex_dtype in [float, random.choice([np.int8, np.int16, np.int32, np.int64])]:
                reindex_masks = random.choice([False, True])
                yield (
                    check_select_masks,
                    batch_size,
                    npolygons_range,
                    nvertices_range,
                    vertex_ndim,
                    vertex_dtype,
                    reindex_masks,
                )


@dali.pipeline_def(batch_size=1, num_threads=4, device_id=0, seed=1234)
def wrong_input_pipe(data_source_fn, reindex_masks=False):
    polygons, vertices, mask_ids = fn.external_source(
        source=data_source_fn, num_outputs=3, device="cpu"
    )
    out_polygons, out_vertices = fn.segmentation.select_masks(
        mask_ids, polygons, vertices, reindex_masks=reindex_masks
    )
    return polygons, vertices, mask_ids, out_polygons, out_vertices


def _test_select_masks_wrong_input(data_source_fn, err_regex):
    p = wrong_input_pipe(data_source_fn=data_source_fn)
    with assert_raises(RuntimeError, regex=err_regex):
        _ = p.run()


def test_select_masks_wrong_mask_ids():
    def test_data():
        polygons = [np.array([[0, 0, 2], [1, 3, 5], [2, 6, 8]], dtype=np.int32)]
        vertices = [np.array(np.random.rand(9, 2), dtype=np.float32)]
        mask_ids = [np.array([10, 11], dtype=np.int32)]  # out of bounds ids
        return polygons, vertices, mask_ids

    _test_select_masks_wrong_input(
        lambda: test_data(), err_regex="Selected mask_id .* is not present in the input\\."
    )


def test_select_masks_wrong_mask_meta_dim():
    def test_data():
        # Expects 3 integers, not 4
        polygons = [np.array([[0, 0, 2, -1], [1, 3, 5, -1], [2, 6, 8, -1]], dtype=np.int32)]
        vertices = [np.array(np.random.rand(9, 2), dtype=np.float32)]
        mask_ids = [np.array([0], dtype=np.int32)]
        return polygons, vertices, mask_ids

    _test_select_masks_wrong_input(
        lambda: test_data(),
        err_regex="``polygons`` is expected to contain 2D tensors with 3 columns: "
        "``mask_id, start_idx, end_idx``\\. Got \\d* columns\\.",
    )


def test_select_masks_wrong_vertex_ids():
    def test_data():
        polygons = [np.array([[0, 0, 20]], dtype=np.int32)]  # Out of bounds vertex index
        vertices = [np.array(np.random.rand(3, 2), dtype=np.float32)]  # Only 3 vertices
        mask_ids = [np.array([0], dtype=np.int32)]
        return polygons, vertices, mask_ids

    _test_select_masks_wrong_input(
        lambda: test_data(),
        err_regex="Vertex index range for mask id .* is out of bounds\\. "
        "Expected to be within the range of available vertices .*\\.",
    )
