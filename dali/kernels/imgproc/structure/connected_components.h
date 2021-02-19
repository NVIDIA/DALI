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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include "dali/core/tensor_view.h"
#include "dali/core/geom/vec.h"
#include "dali/core/geom/box.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/disjoint_set.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/core/exec/engine.h"
#include "dali/core/spinlock.h"

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
        // The label is the offset of current element with respect to `label_base`.
        // This label is usable in union/find algorithms.
        curr_label = out_row + i - label_base;
      } else {
        curr_label = bg_label;
      }
    }
    out_row[i] = curr_label;
    prev = row[i];
  }
}

template <typename OutLabel, typename InLabel, typename ExecutionEngine>
void LabelSlice(OutLabel *label_base,
                TensorSlice<OutLabel, 1> out,
                TensorSlice<const InLabel, 1> slice,
                InLabel background,
                ExecutionEngine &engine) {
  LabelRow(label_base, out.data, slice.data, slice.size[0], background);
}

/**
 * @brief Merges corresponding labels in out1 and out2 slices when the corresponding
 *        input labels match.
 *
 * @param label_base Start of the output tensor.
 * @param out1 A slice of the output tensor. The tensor base pointer must be label_base.
 * @param out2 Another slice of the output tensor. The tensor base pointer must be label_base.
 * @param in1  A slice of the input tensor.
 * @param in2  Another slice of the input tensor (the same as in1).
 */
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
  int64_t in_stride  = in1.stride.x;  // manual strength reduction
  int64_t out_stride = out1.stride.x;
  for (int64_t i = 0, in_offset = 0, out_offset = 0; i < n; i++,
       in_offset += in_stride, out_offset += out_stride) {
    OutLabel o1 = out1.data[out_offset];
    OutLabel o2 = out2.data[out_offset];
    if (o1 != prev1 || o2 != prev2) {
      if (o1 != bg_label) {
        if (in1.data[in_offset] == in2.data[in_offset]) {
          ds.merge(label_base, o1, o2);
        }
      }
      prev1 = o1;
      prev2 = o2;
    }
  }
}

/**
 * @brief Merges corresponding labels in out1 and out2 slices when the corresponding
 *        input labels match.
 *
 * @param label_base Start of the output tensor.
 * @param out1 A slice of the output tensor. The tensor base pointer must be label_base.
 * @param out2 Another slice of the output tensor. The tensor base pointer must be label_base.
 * @param in1  A slice of the input tensor.
 * @param in2  Another slice of the input tensor (the same as in1).
 */
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
enable_if_t<(ndim > 1)> LabelSlice(OutLabel *label_base,
                                   TensorSlice<OutLabel, ndim> out,
                                   TensorSlice<const InLabel, ndim> in,
                                   InLabel background,
                                   SequentialExecutionEngine &engine) {
  int64_t n = in.size[0];
  TensorSlice<OutLabel, ndim-1> prev_out;
  TensorSlice<const InLabel, ndim-1> prev_in;
  for (int64_t i = 0; i < n; i++) {
    auto out_slice = out.slice(i);
    auto in_slice = in.slice(i);
    LabelSlice(label_base, out_slice, in_slice, background, engine);

    if (i > 0)
      MergeSlices(label_base, prev_out, out_slice, prev_in, in_slice);

    prev_out = out_slice;
    prev_in  = in_slice;
  }
}

template <typename OutLabel, typename InLabel, int ndim, typename ExecutionEngine>
void LabelSlice(OutLabel *label_base,
                TensorSlice<OutLabel, ndim> out,
                TensorSlice<const InLabel, ndim> in,
                InLabel background,
                ExecutionEngine &engine) {
  int64_t n = in.size[0];

  constexpr int kMinParallelInner = 1000;
  if (n < 3 || volume(sub<ndim-1>(in.size, 1)) < kMinParallelInner) {
    TensorSlice<OutLabel, ndim-1> prev_out;
    TensorSlice<const InLabel, ndim-1> prev_in;
    for (int64_t i = 0; i < n; i++) {
      auto out_slice = out.slice(i);
      auto in_slice = in.slice(i);
      LabelSlice(label_base, out_slice, in_slice, background, engine);

      if (i > 0)
        MergeSlices(label_base, prev_out, out_slice, prev_in, in_slice);

      prev_out = out_slice;
      prev_in  = in_slice;
    }
    return;
  }

  SequentialExecutionEngine seq_engn;

  for (int64_t i = 0; i < n; i++) {
    auto out_slice = out.slice(i);
    auto in_slice = in.slice(i);
    engine.AddWork([=, &seq_engn](int){
      LabelSlice(label_base, out_slice, in_slice, background, seq_engn);
    });
  }
  engine.RunAll();

  for (int64_t stride = 1; stride <= n; stride *= 2) {
    for (int64_t i = stride; i < n; i += 2*stride) {
      auto out_slice = out.slice(i);
      auto in_slice = in.slice(i);
      auto prev_out = out.slice(i-1);
      auto prev_in = in.slice(i-1);
      engine.AddWork([=](int){
        MergeSlices(label_base, prev_out, out_slice, prev_in, in_slice);
      });
    }
    engine.RunAll();
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

template <typename OutLabel, typename LabelMapping>
void RemapChunk(span<OutLabel> chunk, const LabelMapping &mapping) {
  assert(!chunk.empty());
  OutLabel prev = chunk[0];
  OutLabel remapped = mapping.find(prev)->second;
  for (auto &label : chunk) {
    if (label != prev) {
      prev = label;
      remapped = mapping.find(prev)->second;
    }
    label = remapped;
  }
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
 * @return Number of distinct non-background labels.
 */
template <typename OutLabel, typename ExecutionEngine>
int64_t CompactLabels(OutLabel *labels,
                      int64_t volume,
                      ExecutionEngine &engine,
                      OutLabel bg_label = 0) {
  constexpr OutLabel old_bg_label = static_cast<OutLabel>(-1);
  OutLabel curr_label = 1;
  std::unordered_map<OutLabel, OutLabel> label_map;
  disjoint_set<OutLabel, OutLabel> ds;

  const int64_t chunk_size = 16<<10;
  int num_chunks = std::min<int>(div_ceil(volume, chunk_size), engine.NumThreads());

  // Optimized label compaction algorithm:
  //
  // Labels are scanned and only when the label value changes are we invoking ds.find.
  // In the second pass, we only look up the map when the value changes, but this time we
  // can't call ds.find, because we're overwriting the labels array and the disjoint set
  // structure is effectively corrupted.

  std::set<OutLabel> label_set;

  if (num_chunks == 1) {
    std::unordered_set<OutLabel> tmp_set;
    OutLabel prev = old_bg_label;
    OutLabel remapped = old_bg_label;
    for (int64_t i = 0; i < volume; i++) {
      if (labels[i] != old_bg_label) {
        if (labels[i] != prev) {
          prev = labels[i];
          // look up `ds` only when the value changes - this saves a lot of lookups
          remapped = ds.find(labels, i);
          // no need to assign labels[i] = remapped; find did it
          tmp_set.insert(remapped);
        } else {
          labels[i] = remapped;
        }
      }
    }
    for (auto l : tmp_set)
      label_set.insert(l);
  } else {
    SmallVector<std::unordered_set<OutLabel>, 16> tmp_sets;
    tmp_sets.resize(engine.NumThreads());
    spinlock lock;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
      int64_t chunk_start = volume * chunk / num_chunks;
      int64_t chunk_end = volume * (chunk + 1) / num_chunks;

      engine.AddWork([=, &tmp_sets, &lock](int thread) {
        OutLabel prev = old_bg_label;
        OutLabel remapped = old_bg_label;
        for (int64_t i = chunk_start; i < chunk_end; i++) {
          OutLabel curr = labels[i];
          if (curr != old_bg_label) {
            if (curr != prev) {
              {
                std::lock_guard<spinlock> g(lock);
                prev = curr;
                // look up `ds` only when the value changes - this saves a lot of lookups
                remapped = ds.find(labels, i);
              }
              tmp_sets[thread].insert(remapped);
            } else {
              labels[i] = remapped;
            }
          }
        }
      });
    }
    engine.RunAll();

    for (auto &tmp_set : tmp_sets)
      for (OutLabel l : tmp_set)
        label_set.insert(l);
  }

  OutLabel next_label = 0;
  for (auto old : label_set) {
    if (next_label == bg_label)
      next_label++;
    label_map[old] = next_label++;
  }
  label_map[old_bg_label] = bg_label;

  OutLabel prev = old_bg_label;

  for (int chunk = 0; chunk < num_chunks; chunk++) {
    int64_t chunk_start = volume * chunk / num_chunks;
    int64_t chunk_end = volume * (chunk + 1) / num_chunks;
    engine.AddWork([=, &label_map](int) {
      RemapChunk(make_span(labels + chunk_start, chunk_end - chunk_start), label_map);
    });
  }
  engine.RunAll();
  return label_set.size();
}

template <typename OutLabel>
int64_t CompactLabels(OutLabel *labels,
                      int64_t volume,
                      OutLabel bg_label = 0) {
  SequentialExecutionEngine engine;
  return CompactLabels(labels, volume, engine, bg_label);
}

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
 *
 * @return Number of non-background connected detected.
 */
template <typename OutLabel, typename InLabel, int ndim, typename ExecutionEngine>
int64_t LabelConnectedRegionsImpl(const OutTensorCPU<OutLabel, ndim> &out,
                                  const InTensorCPU<InLabel, ndim> &in,
                                  ExecutionEngine &engine,
                                  same_as_t<OutLabel> background_out = 0,
                                  same_as_t<InLabel> background_in = 0) {
  auto out_slice = detail::FullTensorSlice(out);
  auto in_slice = detail::FullTensorSlice(in);
  detail::LabelSlice(out.data, out_slice, in_slice, background_in, engine);
  return detail::CompactLabels(out.data, out.num_elements(), engine, background_out);
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
 * @param engine  Execution engine for deferred execution (e.g. ThreadPool)
 * @param background_out Value in the output which will be set for background pixels
 * @param background_in  Value in the input which denotes background pixels
 *
 * @return Number of non-background connected detected.
 */
template <typename OutLabel, typename InLabel, int ndim, typename ExecutionEngine>
int64_t LabelConnectedRegions(const OutTensorCPU<OutLabel, ndim> &out,
                              const InTensorCPU<InLabel, ndim> &in,
                              ExecutionEngine &engine,
                              same_as_t<OutLabel> background_out = 0,
                              same_as_t<InLabel> background_in = 0) {
  assert(out.shape == in.shape);
  if (in.num_elements() == 0)
    return 0;
  TensorShape<> simplified;
  for (int i = 0; i < in.shape.size(); i++)
    if (in.shape[i] > 1)
      simplified.shape.push_back(in.shape[i]);
  if (simplified.empty()) {
    // The tensor has just one element.
    // If it's background, assign background label to it, otherwise assign the first
    // non-background label.
    bool is_foreground = in.data[0] != background_in;
    out.data[0] = is_foreground ? (background_in == 0 ? 1 : 0)
                                : background_out;
    return is_foreground ? 1 : 0;
  }
  int64_t ret = 0;
  VALUE_SWITCH(simplified.size(), simplified_ndim, (1, 2, 3, 4, 5, 6), (
      TensorShape<simplified_ndim> sh = simplified.to_static<simplified_ndim>();
      auto s_out = make_tensor_cpu(out.data, sh);
      auto s_in  = make_tensor_cpu(in.data, sh);
      ret = detail::LabelConnectedRegionsImpl(s_out, s_in, engine, background_out, background_in);
    ), (  // NOLINT
      throw std::invalid_argument(make_string(
          "Unsupported number of non-degenerate dimensions: ", simplified.size(),
          ". Valid range is 0..6."));
    )     // NOLINT
  );      // NOLINT
  return ret;
}

template <typename OutLabel, typename InLabel, int ndim>
int64_t LabelConnectedRegions(const OutTensorCPU<OutLabel, ndim> &out,
                              const InTensorCPU<InLabel, ndim> &in,
                              same_as_t<OutLabel> background_out = 0,
                              same_as_t<InLabel> background_in = 0) {
  SequentialExecutionEngine engine;
  return LabelConnectedRegions(out, in, engine, background_out, background_in);
}

}  // namespace connected_components
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_STRUCTURE_CONNECTED_COMPONENTS_H_
