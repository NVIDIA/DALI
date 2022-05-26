// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_UTILS_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_UTILS_H_

#include <tuple>
#include <utility>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/sequence_shape.h"

namespace dali {
namespace sequence_utils {

template <typename Storage, typename T, int ndim, int ndims_to_unfold>
class UnfoldedViewRange {
  static_assert(ndim == DynamicDimensions || ndim >= ndims_to_unfold);

 public:
  static constexpr int out_ndim =
      ndim == DynamicDimensions ? DynamicDimensions : ndim - ndims_to_unfold;
  using Self = UnfoldedViewRange<Storage, T, ndim, ndims_to_unfold>;
  using IndexType = int64_t;
  using ValueType = TensorView<Storage, T, out_ndim>;

  UnfoldedViewRange(TensorView<Storage, T, ndim> view)  // NOLINT
      : view_{std::move(view)},
        num_slices_{[&]() {
          assert(view_.shape.size() >= ndims_to_unfold);
          return volume(view_.shape.first(ndims_to_unfold));
        }()},
        slice_shape_{view_.shape.last(out_ndim)},
        slice_stride_{static_cast<size_t>(volume(slice_shape_))} {}

  RangeIterator<Self> begin() const {
    return {*this, 0};
  }

  RangeIterator<Self> end() const {
    return {*this, NumSlices()};
  }

  ValueType operator[](IndexType idx) const {
    return {view_.data + idx * SliceSize(), SliceShape()};
  }

  int64_t NumSlices() const {
    return num_slices_;
  }

  const auto &SliceShape() const {
    return slice_shape_;
  }

  size_t SliceSize() const {
    return slice_stride_;
  }

 private:
  TensorView<Storage, T, ndim> view_;
  int64_t num_slices_;
  TensorShape<out_ndim> slice_shape_;
  size_t slice_stride_;
};

template <int ndims_to_unfold, typename Storage, typename T, int ndim>
UnfoldedViewRange<Storage, T, ndim, ndims_to_unfold> unfolded_view_range(
    const TensorView<Storage, T, ndim> &view) {
  return {view};
}

template <typename Range, typename... Ranges>
class CombinedRange {
  static_assert(sizeof...(Ranges) >= 1);

 public:
  CombinedRange(Range &&range, Ranges &&...ranges)
      : ranges_{std::make_tuple(std::forward<Range>(range), std::forward<Ranges>(ranges)...)} {
    assert(((range.NumSlices() == ranges.NumSlices()) && ...));
  }
  using Self = CombinedRange<Range, Ranges...>;
  using IndexType = int64_t;
  using ValueType = std::tuple<typename Range::ValueType, typename Ranges::ValueType...>;

  ValueType operator[](IndexType idx) const {
    return std::apply([&idx](const auto &...ranges) { return std::make_tuple(ranges[idx]...); },
                      ranges_);
  }

  RangeIterator<Self> begin() const {
    return {*this, 0};
  }

  RangeIterator<Self> end() const {
    return {*this, std::get<0>(ranges_).NumSlices()};
  }

  template <int i>
  const auto &get() {
    return std::get<i>(ranges_);
  }

 private:
  std::tuple<Range, Ranges...> ranges_;
};

template <typename... Ranges>
CombinedRange<Ranges...> combine_ranges(Ranges &&...ranges) {
  return {std::forward<Ranges>(ranges)...};
}

template <int ndims_to_unfold, typename... Storages, typename... Ts, int... ndims>
CombinedRange<UnfoldedViewRange<Storages, Ts, ndims, ndims_to_unfold>...> unfolded_views_range(
    const TensorView<Storages, Ts, ndims> &...views) {
  return combine_ranges(unfolded_view_range<ndims_to_unfold>(views)...);
}

}  // namespace sequence_utils

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_UTILS_H_
