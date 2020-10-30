// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/operators/segmentation/select_masks.h"

#include <utility>
#include "dali/kernels/common/type_erasure.h"
#include "dali/core/static_switch.h"

namespace dali {

DALI_SCHEMA(segmentation__SelectMasks)
    .DocStr(R"(Selects a subset of polygons by their mask ids.

The operator expects three inputs describing multiple segmentation mask polygons belonging to different mask ids and 
a list of selected mask ids.

Each sample can contain several polygons belonging to different masks, and each polygon can be composed by an arbitrary 
number of vertices (at least 3). The masks polygons are described  by the inputs ``polygons`` and ``vertices`` and 
the operator produces output ``polygons`` and ``vertices`` where only the polygons associated with the selected 
masks are present.

.. note::
  
  The format of ``polygons`` and ``vertices`` is the same as produced by COCOReader.

**Examples:**

Let us assume the following input mask, where symbolic coordinates are used for a clearer example::

    polygons = [[0, 0, 3], [1, 3, 7], [2, 7, 10]]
    vertices = [[x0, y0], [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9]]

Example 1: Selecting a single mask with id ``1``, maintaining the original id::

    mask_ids = [1], ``reindex_masks`` = False
    out_polygons = [[1, 0, 4]]
    out_vertices = [[x3, y3], [x4, y4], [x5, y5], [x6, y6]]

Example 2: Selecting two out of the three masks, replacing the mask ids with the indices at which
they appeared in ``mask_ids`` input::

    mask_ids = [2, 0]
    reindex_masks = True
    out_polygons = [[0, 3, 6], [1, 0, 3]]
    out_vertices = [[x0, y0], [x1, y1], [x2, y2], [x7, y7], [x8, y8], [x9, y9]]
)")
    .NumInput(3)
    .NumOutput(2)
    .InputDox(0, "mask_ids", "1D TensorList of int",
              R"code(List of identifiers of the masks to be selected. The list should not contain duplicates.)code")
    .InputDox(1, "polygons", "2D TensorList of int",
              R"code(Polygons, described by 3 columns::

    [[mask_id0, start_vertex_idx0, end_vertex_idx0],
     [mask_id1, start_vertex_idx1, end_vertex_idx1],
     ...,
     [mask_idn, start_vertex_idxn, end_vertex_idxn],]

with ``mask_id`` being the identifier of the mask this polygon belongs to, and
``[start_vertex_idx, end_vertex_idx)`` describing the range of indices from ``vertices`` that belong to
this polygon.)code")
    .InputDox(2, "vertices", "2D TensorList",
      R"code(Vertex data stored in interleaved format::

    [[x0, y0, ...],
     [x1, y1, ...],
     ... ,
     [xn, yn, ...]]

The operator accepts vertices with arbitrary number of coordinates.)code")
    .AddOptionalArg<bool>("reindex_masks",
      R"code(If set to True, the output mask ids are replaced with the indices at which they appeared
in ``mask_ids`` input.)code",
      false);

bool SelectMasksCPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                              const workspace_t<CPUBackend> &ws) {
  const auto &in_mask_ids = ws.template InputRef<CPUBackend>(0);
  auto in_mask_ids_shape = in_mask_ids.shape();
  DALI_ENFORCE(in_mask_ids.type().id() == DALI_INT32, "``mask_ids`` input is expected to be int32");
  DALI_ENFORCE(in_mask_ids_shape.sample_dim() == 1, "``mask_ids`` input is expected to be 1D");

  const auto &in_polygons = ws.template InputRef<CPUBackend>(1);
  auto in_polygons_shape = in_polygons.shape();
  DALI_ENFORCE(in_polygons.type().id() == DALI_INT32,
               "``polygons`` input is expected to be int32");
  DALI_ENFORCE(in_polygons_shape.sample_dim() == 2,
               make_string("``polygons`` input is expected to be 2D. Got ",
                           in_polygons_shape.sample_dim(), "D"));

  const auto &in_vertices = ws.template InputRef<CPUBackend>(2);
  auto in_vertices_shape = in_vertices.shape();
  DALI_ENFORCE(in_vertices_shape.sample_dim() == 2,
               make_string("``vertices`` input is expected to be 2D. Got ",
                           in_vertices_shape.sample_dim(), "D"));

  int nsamples = in_polygons.size();
  DALI_ENFORCE(nsamples == in_polygons_shape.size() && nsamples == in_vertices_shape.size(),
               make_string("All the inputs should have the same number of samples. Got: ", nsamples,
                           ", ", in_polygons_shape.size(), ", ", in_vertices_shape.size()));

  if (nsamples == 0) {  // empty input
    output_desc.reserve(2);
    output_desc.push_back({in_polygons_shape, in_polygons.type()});
    output_desc.push_back({in_vertices_shape, in_vertices.type()});
    return true;
  }

  for (int i = 0; i < nsamples; i++) {
    auto sh = in_polygons_shape.tensor_shape_span(i);
    DALI_ENFORCE(3 == sh[1],
                 make_string("``polygons`` is expected to contain 2D tensors with 3 columns: "
                             "``mask_id, start_idx, end_idx``. Got ",
                             sh[1], " columns."));
  }

  const auto &in_mask_ids_view = view<const int32_t, 1>(in_mask_ids);
  const auto &in_polygons_view = view<const int32_t, 2>(in_polygons);

  auto out_polygons_shape = in_polygons_shape;
  auto out_vertices_shape = in_vertices_shape;

  samples_.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    samples_[i].clear();
    auto &selected_masks = samples_[i].selected_masks;
    auto &polygons = samples_[i].polygons;
    int64_t nselected = in_mask_ids_view.tensor_shape_span(i)[0];
    selected_masks = make_cspan(in_mask_ids_view.tensor_data(i), nselected);
    out_polygons_shape.tensor_shape_span(i)[0] = selected_masks.size();
    int idx = 0;
    for (auto mask_id : selected_masks) {
      if (polygons.find(mask_id) != polygons.end()) {
        DALI_FAIL(
            make_string("mask_ids should not have duplicated values. Got ", mask_id, " repeated."));
      }
      polygons[mask_id].new_mask_id = reindex_masks_ ? idx++ : mask_id;
    }

    int64_t npolygons = in_polygons_shape.tensor_shape_span(i)[0];
    int64_t in_nvertices = in_vertices_shape.tensor_shape_span(i)[0];
    for (int64_t k = 0; k < npolygons; k++) {
      const auto *poly_data = in_polygons_view.tensor_data(i) + k * 3;
      int mask_id = poly_data[0];
      auto it = polygons.find(mask_id);
      if (it == polygons.end())
        continue;
      auto &poly = it->second;
      poly.start_vertex = poly_data[1];
      poly.end_vertex = poly_data[2];

      DALI_ENFORCE(
        poly.start_vertex >= 0 && poly.end_vertex <= in_nvertices,
        make_string(
            "Vertex index range for mask id ", mask_id, " [", poly.start_vertex, ", ",
            poly.end_vertex,
            ") is out of bounds. Expected to be within the range of available vertices [0, ",
            in_nvertices, ")."));

      DALI_ENFORCE(poly.end_vertex >= poly.start_vertex,
                   make_string("Vertex start index can't be after end index. Got [",
                               poly.start_vertex, ", ", poly.end_vertex, ")"));
    }

    int64_t nvertices = 0;
    for (int k = 0; k < nselected; k++) {
      int mask_id = selected_masks[k];
      const auto &poly = polygons[mask_id];
      if (poly.start_vertex == -1 && poly.end_vertex == -1)
        DALI_FAIL(make_string("Selected mask_id ", mask_id, " is not present in the input."));
      nvertices += poly.end_vertex - poly.start_vertex;
    }
    out_vertices_shape.tensor_shape_span(i)[0] = nvertices;
  }

  output_desc.reserve(2);
  output_desc.push_back({std::move(out_polygons_shape), in_polygons.type()});
  output_desc.push_back({std::move(out_vertices_shape), in_vertices.type()});
  return true;
}

template <typename T>
void SelectMasksCPU::RunImplTyped(workspace_t<CPUBackend> &ws) {
  // Inputs were already validated and input 0 was already parsed in SetupImpl
  const auto &in_polygons = ws.template InputRef<CPUBackend>(1);
  const auto &in_polygons_view = view<const int32_t, 2>(in_polygons);
  auto &out_polygons = ws.template OutputRef<CPUBackend>(0);
  const auto &out_polygons_view = view<int32_t, 2>(out_polygons);

  const auto &in_vertices = ws.template InputRef<CPUBackend>(2);
  const auto &in_vertices_view = reinterpret_view<const T, 2>(in_vertices);
  auto &out_vertices = ws.template OutputRef<CPUBackend>(1);
  const auto &out_vertices_view = reinterpret_view<T, 2>(out_vertices);

  for (int i = 0; i < in_polygons_view.num_samples(); i++) {
    const auto &selected_masks = samples_[i].selected_masks;
    const auto &polygons = samples_[i].polygons;
    auto *out_polygons_data = out_polygons_view.tensor_data(i);
    auto *out_vertices_data = out_vertices_view.tensor_data(i);
    auto out_vertices_shape = out_vertices_view.tensor_shape_span(i);
    const auto *in_vertices_data = in_vertices_view.tensor_data(i);
    int64_t out_vertex_i = 0;
    for (int64_t k = 0; k < selected_masks.size(); k++) {
      int mask_id = selected_masks[k];
      auto it = polygons.find(mask_id);
      assert(it != polygons.end());
      const auto &poly = it->second;
      int64_t nvertices = poly.end_vertex - poly.start_vertex;
      *out_polygons_data++ = poly.new_mask_id;
      *out_polygons_data++ = out_vertex_i;  // start vertex
      *out_polygons_data++ = out_vertex_i + nvertices;  // end vertex
      auto vertex_ndim = out_vertices_shape[1];
      auto *in_vertex_data = in_vertices_data + poly.start_vertex * vertex_ndim;
      for (int64_t j = 0; j < nvertices * vertex_ndim; j++)
        *out_vertices_data++ = in_vertex_data[j];
      out_vertex_i += nvertices;
    }
  }
}

void SelectMasksCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in_vertices = ws.template InputRef<CPUBackend>(2);
  VALUE_SWITCH(in_vertices.type().size(), dtype_sz, (1, 2, 4, 8, 16), (
    using T = kernels::type_of_size<dtype_sz>;
    RunImplTyped<T>(ws);
  ), (  // NOLINT
    DALI_FAIL(make_string("Unexpected vertex data type: ", in_vertices.type().id()));
  ));  // NOLINT 
}

DALI_REGISTER_OPERATOR(segmentation__SelectMasks, SelectMasksCPU, CPU);

}  // namespace dali
