// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_STRUCTURE_CONNECTED_COMPONENTS_H_
#define DALI_KERNELS_IMGPROC_STRUCTURE_CONNECTED_COMPONENTS_H_

#include <set>
#include <unordered_map>
#include "dali/core/tensor_view.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/common/disjoint_set.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace connected_components {
namespace detail {

template <typename T, int ndim>
struct TensorSlice {
  T             *data;
  i64vec<ndim>   stride;
  i64vec<ndim>   size;

  DALI_HOST_DEV DALI_FORCEINLINE
  int64_t offset(i64vec<ndim> pos) const noexcept {
    return dot(pos, stride);
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  T &operator()(i64vec<ndim> pos) const noexcept {
    return data[offset(pos)];
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  TensorSlice<T, ndim-1> slice(int64_t outer_idx) const noexcept {
    return {
      data + stride[0] * outer_idx,
      sub<ndim-1>(stride, 1),
      sub<ndim-1>(size, 1)
    };
  }
};

template <typename OutLabel, typename InLabel>
void LabelRow(OutLabel *label_base,
              OutLabel *out_row,
              const InLabel *row, int length, InLabel background) {
  constexpr OutLabel bg_label = static_cast<OutLabel>(-1);
  OutLabel curr_label = bg_label;
  InLabel prev = background;
  for (int i = 0; i < length; i++) {
    if (row[i] != prev) {
      if (row[i] != background) {
        curr_label = out_row + i - label_base;
      } else {
        curr_label = bg_label;
      }
    }
    out_row[i] = curr_label;
    prev = row[i];
  }
}

template <typename OutLabel, typename InLabel>
void LabelSlice(OutLabel *label_base,
                TensorSlice<OutLabel, 1> out,
                TensorSlice<const InLabel, 1> slice,
                InLabel background) {
  LabelRow(label_base, out.data, slice.data, slice.size[0], background);
}

template <typename OutLabel, typename InLabel>
void MergeSlices(OutLabel *label_base,
                 TensorSlice<OutLabel, 1> out1,
                 TensorSlice<OutLabel, 1> out2,
                 TensorSlice<const InLabel, 1> in1,
                 TensorSlice<const InLabel, 1> in2) {
  int64_t n = in1.size[0];
  constexpr OutLabel bg_label = static_cast<OutLabel>(-1);
  OutLabel prev1 = bg_label;
  OutLabel prev2 = bg_label;
  disjoint_set<OutLabel, OutLabel> ds;
  for (int64_t i = 0; i < n; i++) {
    OutLabel o1 = out1(i);
    OutLabel o2 = out2(i);
    if (o1 != prev1 || o2 != prev2) {
      if (o1 != bg_label) {
        if (in1(i) == in2(i) && in1(i)) {
          ds.merge(label_base, o1, o2);
        }
      }
      prev1 = o1;
      prev2 = o2;
    }
  }
}

template <typename OutLabel, typename InLabel, int ndim>
void MergeSlices(OutLabel *label_base,
                 TensorSlice<OutLabel, ndim> out1,
                 TensorSlice<OutLabel, ndim> out2,
                 TensorSlice<const InLabel, ndim> in1,
                 TensorSlice<const InLabel, ndim> in2) {
  int64_t n = in1.size[0];
  for (int64_t i = 0; i < n; i++) {
    MergeSlices(label_base, out1.slice(i), out2.slice(i), in1.slice(i), in2.slice(i));
  }
}

template <typename OutLabel, typename InLabel, int ndim>
void LabelSlice(OutLabel *label_base,
                TensorSlice<OutLabel, ndim> out,
                TensorSlice<const InLabel, ndim> in,
                InLabel background) {
  int64_t n = in.size[0];
  TensorSlice<OutLabel, ndim-1> prev_out;
  TensorSlice<const InLabel, ndim-1> prev_in;
  for (int64_t i = 0; i < n; i++) {
    auto out_slice = out.slice(i);
    auto in_slice = in.slice(i);
    LabelSlice(label_base, out_slice, in_slice, background);

    if (i > 0)
      MergeSlices(label_base, prev_out, out_slice, prev_in, in_slice);

    prev_out = out_slice;
    prev_in  = in_slice;
  }
}

template <typename T, int ndim>
auto FullTensorSlice(const TensorView<StorageCPU, T, ndim> &tensor) {
  TensorSlice<T, ndim> slice;
  slice.data = tensor.data;
  for (int i = 0; i < ndim; i++)
    slice.size[i] = tensor.shape[i];
  CalcStrides(slice.stride, slice.size);
  return slice;
}

/**
 * @brief Compacts label indices in `labels`
 *
 * This function remaps the labels in `labels` so that the object labels are zero-based integers
 * with a gap for `bg_label` - for example, if bg_label is 0, the object indices would be
 * 1, 2, 3, ...; if bg_label is 2, the object labels will be 0, 1, 3, ...
 *
 * @param labels    Object labels; the structure must be usable as disjoint set structure.
 * @param volume    Number of elements in `labels`
 * @param bg_label  The output label assigned to background.
 */
template <typename OutLabel>
void CompactLabels(OutLabel *labels, int64_t volume, OutLabel bg_label = 0) {
  constexpr OutLabel old_bg_label = static_cast<OutLabel>(-1);
  OutLabel curr_label = 1;
  std::set<OutLabel> label_set;
  std::unordered_map<OutLabel, OutLabel> label_map;
  disjoint_set<OutLabel, OutLabel> ds;

  // Optimized label compaction algorithm:
  //
  // Labels are scanned and only when the label value changes are we invoking ds.find.
  // In the second pass, we only look up the map when the value changes, but this time we
  // can't call ds.find, because we're overwriting the labels array and the disjoint set
  // structure is effectively corrupted.

  OutLabel prev = old_bg_label;
  for (int64_t i = 0; i < volume; i++) {
    if (labels[i] != old_bg_label && labels[i] != prev) {
      // look up `ds` only when the value changes - this saves a lot of lookups
      label_set.insert(ds.find(labels, i));
      prev = labels[i];
    }
  }

  OutLabel next_label = 0;
  for (auto old : label_set) {
    if (next_label == bg_label)
      next_label++;
    label_map[old] = next_label++;
  }

  prev = old_bg_label;
  OutLabel remapped = 0;
  for (int64_t i = 0; i < volume; i++) {
    if (labels[i] != old_bg_label) {
      if (labels[i] != prev) {
        // We can't call ds.find, because the beginning of the labels array is overwritten
        // - but we don't have to, because we've already done that in the previous pass.
        auto it = label_map.find(labels[i]);
        assert(it != label_map.end());
        remapped = it->second;
        prev = labels[i];
      }
      labels[i] = remapped;
    } else {
      labels[i] = bg_label;
    }
  }
}

}  // namespace detail

/**
 * @brief Labels connected components in tensor `in` and stores the labels in `out`
 *
 * This function detects connected blobs having the same input label.
 * Elements equal to background_in are assigned the same special class index even if not connected.
 *
 * @tparam OutLabel type of the output label; must have enough capacity to hold offset to the last
 *                  element.
 *
 * @param out Tensor where each connected object is assigned a unique label
 * @param in  Tensor with objects, where one label value may be used to mark multiple objects
 * @param background_out Value in the output which will be set for background pixels
 * @param background_in  Value in the input which denotes background pixels
 */
template <typename OutLabel, typename InLabel, int ndim>
void LabelConnectedRegions(const OutTensorCPU<OutLabel, ndim> &out,
                           const InTensorCPU<InLabel, ndim> &in,
                           same_as_t<OutLabel> background_out = 0,
                           same_as_t<InLabel> background_in = 0) {
  auto out_slice = detail::FullTensorSlice(out);
  auto in_slice = detail::FullTensorSlice(in);
  detail::LabelSlice(out.data, out_slice, in_slice, background_in);
  detail::CompactLabels(out.data, out.num_elements(), background_out);
}

}  // namespace connected_components
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_STRUCTURE_CONNECTED_COMPONENTS_H_
