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
    .DocStr(R"(Selects a subset of mask by their mask ids.

The operator expects three inputs describing multiple segmentation mask polygons and a list of selected mask ids.

Each sample can contain several masks, and each mask can be composed by several polygons. The masks are described 
by the inputs ``mask_meta`` and ``mask_coords``. The format of this data is the same as the one produced by COCOReader.

The operator receives a list of mask ids that are to be selected for each sample, and produces
output ``mask_meta`` and ``mask_coords`` where only the masks corresponding to selected mask ids are present.

**Examples:**

Let us assume the following input mask, where symbolic coordinates are used for a clearer example::

    ``mask_coords`` = [[x0, y0], [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9]]
    ``mask_meta`` = [[0, 0, 2], [1, 3, 6], [2, 7, 9]

Example 1: Selecting a single mask with id ``1``, maintaining the original id::

    ``mask_ids`` = [1], ``reindex_masks`` = False
    ``out_mask_meta`` = [[1, 0, 4]]
    ``out_mask_coords`` = [[x3, y3], [x4, y4], [x5, y5], [x6, y6]]

Example 2: Selecting two out of the three mask, reindexing the mask ids to follow the order of the input mask ids::

    ``mask_ids`` = [2, 0]
    ``reindex_masks`` = True
    ``out_mask_meta`` = [[0, 3, 5], [1, 0, 2]]
    ``out_mask_coords`` = [[x0, y0], [x1, y1], [x2, y2], [x7, y7], [x8, y8], [x9, y9]]
)")
    .NumInput(3)
    .NumOutput(2)
    .InputDox(0, "mask_ids", "1D TensorList of int",
              R"code(List of mask identifiers that are to be selected.)code")
    .InputDox(
        1, "mask_meta", "2D TensorList of int",
        R"code(Each row describes a single mask with three values, ``mask_idx, start_idx, end_idx``,
where ``start_idx`` and ``end_idx`` refer to indices in ``mask_coords``.)code")
    .InputDox(2, "mask_coords", "2D TensorList of float or int",
              R"code(Each row contains a single ``x, y`` coordinate.)code")
    .AddOptionalArg<bool>(
        "reindex_masks",
        R"code(If set to True, new mask ids are reassigned to each of the selected mask, so that they
follow zero-based index of the mask ids in the input.)code",
        false);

bool SelectMasksCPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                              const workspace_t<CPUBackend> &ws) {
  const auto &in_mask_ids = ws.template InputRef<CPUBackend>(0);
  auto in_mask_ids_shape = in_mask_ids.shape();
  DALI_ENFORCE(in_mask_ids.type().id() == DALI_INT32, "``mask_ids`` input is expected to be int32");
  DALI_ENFORCE(in_mask_ids_shape.sample_dim() == 1, "``mask_ids`` input is expected to be 1D");

  const auto &in_mask_meta = ws.template InputRef<CPUBackend>(1);
  auto in_mask_meta_shape = in_mask_meta.shape();
  DALI_ENFORCE(in_mask_meta.type().id() == DALI_INT32,
               "``mask_meta`` input is expected to be int32");
  DALI_ENFORCE(in_mask_meta_shape.sample_dim() == 2,
               make_string("``mask_meta`` input is expected to be 2D. Got ",
                           in_mask_meta_shape.sample_dim(), "D"));

  const auto &in_mask_coords = ws.template InputRef<CPUBackend>(2);
  auto in_mask_coords_shape = in_mask_coords.shape();
  DALI_ENFORCE(in_mask_coords_shape.sample_dim() == 2,
               make_string("``mask_coords`` input is expected to be 2D. Got ",
                           in_mask_coords_shape.sample_dim(), "D"));

  int nsamples = in_mask_meta.size();
  DALI_ENFORCE(nsamples == in_mask_meta_shape.size() && nsamples == in_mask_coords_shape.size(),
               make_string("All the inputs should have the same number of samples. Got: ", nsamples,
                           ", ", in_mask_meta_shape.size(), ", ", in_mask_coords_shape.size()));

  if (nsamples == 0) {  // empty input
    output_desc.reserve(2);
    output_desc.push_back({in_mask_meta_shape, in_mask_meta.type()});
    output_desc.push_back({in_mask_coords_shape, in_mask_coords.type()});
    return true;
  }

  for (int i = 0; i < nsamples; i++) {
    auto sh = in_mask_meta_shape.tensor_shape_span(i);
    DALI_ENFORCE(3 == sh[1],
                 make_string("``mask_meta`` is expected to contain 3 element rows, containing "
                             "``mask_id, start_idx, end_idx``. Got ",
                             sh[1], " elements"));
  }

  int64_t coord_ndim = in_mask_coords_shape.tensor_shape_span(0)[1];
  for (int i = 1; i < nsamples; i++) {
    auto sh = in_mask_coords_shape.tensor_shape_span(i);
    DALI_ENFORCE(coord_ndim == sh[1],
      make_string("All coordinates are expected to have the same dimensionality. Got ",
                  coord_ndim, "D and ", sh[1], "D in the same batch"));
  }

  const auto &in_mask_ids_view = view<const int32_t, 1>(in_mask_ids);
  const auto &in_mask_meta_view = view<const int32_t, 2>(in_mask_meta);

  auto out_mask_meta_shape = in_mask_meta_shape;
  auto out_mask_coords_shape = in_mask_coords_shape;

  samples_meta_.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    auto &meta = samples_meta_[i];
    meta.clear();
    int64_t nselected = in_mask_ids_view.tensor_shape_span(i)[0];
    meta.selected_masks = make_cspan(in_mask_ids_view.tensor_data(i), nselected);
    out_mask_meta_shape.tensor_shape_span(i)[0] = meta.selected_masks.size();
    int idx = 0;
    for (auto mask_id : meta.selected_masks) {
      if (meta.masks_meta.find(mask_id) != meta.masks_meta.end()) {
        DALI_WARN(make_string("mask_id ", mask_id, " is duplicated. Ignoring..."));
        continue;
      }
      meta.masks_meta[mask_id].new_mask_id = reindex_masks_ ? idx++ : mask_id;
    }

    int64_t nmasks = in_mask_meta_shape.tensor_shape_span(i)[0];
    int64_t input_total_ncoords = in_mask_coords_shape.tensor_shape_span(i)[0];
    for (int64_t k = 0; k < nmasks; k++) {
      const auto *mask_data = in_mask_meta_view.tensor_data(i) + k * 3;
      int mask_id = mask_data[0];
      auto it = meta.masks_meta.find(mask_id);
      if (it == meta.masks_meta.end())
        continue;
      auto &mask = it->second;
      mask.start_coord = mask_data[1];
      mask.end_coord = mask_data[2];

      DALI_ENFORCE(
        mask.start_coord >= 0 && mask.end_coord < input_total_ncoords,
        make_string(
            "Coordinate index range for mask id ", mask_id, " [", mask.start_coord, ", ",
            mask.end_coord,
            "] is out of bounds. Expected to be within the range of available coordinates [0, ",
            input_total_ncoords - 1, "]."));
  }

    int64_t ncoords = 0;
    for (int k = 0; k < nselected; k++) {
      int mask_id = meta.selected_masks[k];
      const auto &mask = meta.masks_meta[mask_id];
      if (mask.start_coord >= mask.end_coord)
        DALI_FAIL(make_string("Selected mask_id ", mask_id, " is not present in the input."));
      ncoords += mask.end_coord - mask.start_coord + 1;
    }
    out_mask_coords_shape.tensor_shape_span(i)[0] = ncoords;
  }

  output_desc.reserve(2);
  output_desc.push_back({std::move(out_mask_meta_shape), in_mask_meta.type()});
  output_desc.push_back({std::move(out_mask_coords_shape), in_mask_coords.type()});
  return true;
}

template <typename T>
void SelectMasksCPU::RunImplTyped(workspace_t<CPUBackend> &ws) {
  // Inputs were already validated and input 0 was already parsed in SetupImpl
  const auto &in_mask_meta_view = view<const int32_t, 2>(ws.template InputRef<CPUBackend>(1));
  const auto &out_mask_meta_view = view<int32_t, 2>(ws.template OutputRef<CPUBackend>(0));
  const auto &in_mask_coords_view = view<const T, 2>(ws.template InputRef<CPUBackend>(2));
  const auto &out_mask_coords_view = view<T, 2>(ws.template OutputRef<CPUBackend>(1));

  for (int i = 0; i < in_mask_meta_view.num_samples(); i++) {
    const auto &meta = samples_meta_[i];
    auto *out_mask_meta_data = out_mask_meta_view.tensor_data(i);
    auto *out_mask_coords_data = out_mask_coords_view.tensor_data(i);
    auto out_mask_coords_shape = out_mask_coords_view.tensor_shape_span(i);
    const auto *in_mask_coords_data = in_mask_coords_view.tensor_data(i);
    int64_t out_coord_i = 0;
    for (int64_t k = 0; k < meta.selected_masks.size(); k++) {
      int mask_id = meta.selected_masks[k];
      auto it = meta.masks_meta.find(mask_id);
      assert(it != meta.masks_meta.end());
      const auto &mask_meta = it->second;
      int64_t ncoords = mask_meta.end_coord - mask_meta.start_coord + 1;
      *out_mask_meta_data++ = mask_meta.new_mask_id;
      *out_mask_meta_data++ = out_coord_i;  // start coord
      *out_mask_meta_data++ = out_coord_i + ncoords - 1;  // end coord
      auto coords_ndim = out_mask_coords_shape[1];
      auto *out_coords_data = out_mask_coords_data + out_coord_i * coords_ndim;
      auto *in_coords_data = in_mask_coords_data + mask_meta.start_coord * coords_ndim;
      for (int64_t j = 0; j < ncoords * coords_ndim; j++)
        out_coords_data[j] = in_coords_data[j];
      out_coord_i += ncoords;
    }
  }
}

void SelectMasksCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in_mask_coords = ws.template InputRef<CPUBackend>(2);
  VALUE_SWITCH(in_mask_coords.type().size(), dtype_sz, (1, 2, 4, 8, 16), (
    using T = kernels::type_of_size<dtype_sz>;
    RunImplTyped<T>(ws);
  ), (  // NOLINT
    DALI_FAIL(make_string("Unexpected data type for mask coordinates: ",
                          in_mask_coords.type().id()));
  ));  // NOLINT 
}

DALI_REGISTER_OPERATOR(segmentation__SelectMasks, SelectMasksCPU, CPU);

}  // namespace dali
