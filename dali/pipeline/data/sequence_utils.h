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

#ifndef DALI_PIPELINE_DATA_SEQUENCE_UTILS_H_
#define DALI_PIPELINE_DATA_SEQUENCE_UTILS_H_

#include <tuple>
#include <utility>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"

namespace dali {
namespace sequence_utils {

/**
 * @brief Facilitates usage of a type that supports operator[] (and provides IndexType and ValueType
 * types) in range-based for loops.
 */
template <typename Range>
class RangeIterator {
  using IndexType = typename Range::IndexType;
  using ValueType = typename Range::ValueType;

 public:
  RangeIterator(const Range &range, IndexType idx) : range_{range}, idx_{idx} {}

  ValueType operator*() const {
    return range_[idx_];
  }

  // For performance reasons the equality completely disregards Range instance at hand.
  // Comparing iterators over different range instances is UB.
  bool operator==(const RangeIterator &other) const {
    assert(&range_ == &other.range_);
    return idx_ == other.idx_;
  }

  bool operator!=(const RangeIterator &other) const {
    return !(*this == other);
  }

  void operator++() {
    ++idx_;
  }

 private:
  const Range &range_;
  IndexType idx_;
};


/**
 * @brief Unfolds `ndims_to_unfold` leading extents of a TensorView into a range of TensorViews.
 * For example, for `TensorView` of shape `{7, 2, 50, 100}` and `ndims_to_unfold` = 2,
 * results in a range of 14 contiguous TensorViews, each of shape {50, 100}.
 */
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
        slice_shape_{view_.shape.last(view_.shape.size() - ndims_to_unfold)},
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

#if __cplusplus >= 201703L

/**
 * @brief Zips two or more ranges of equal length into a single range of the same
 * length. The i-th element of the range is a tuple of length equal to the number
 * of input ranges, where k-th element of a tuple is the i-th element of the k-th range. */
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
    return std::apply([idx](const auto &...ranges) { return std::make_tuple(ranges[idx]...); },
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

/**
 * @brief Unfolds the `ndims_to_unfold` leading extents of two or more TensorViews and produces
 * the zipped range over the unfolded views ranges.
 * The volume of `ndims_to_unfold` leading extents of every TensorViews must be equal.
 * Consider the example: `in` is a TensorView of shape {50, 128, 128, 3}
 * and `out` is a TensorView {50, 64, 64, 3} (it may be a sequence of video frames and
 * we wish to resize each frame).
 * Then, the function can be used in range-based loop as follows:
 * for (auto &&[out_view, in_view] : unfolded_views_range<1>(out, in)) {
 *    // process each of the 50 frames
 * }
 *
 * */
template <int ndims_to_unfold, typename... Storages, typename... Ts, int... ndims>
CombinedRange<UnfoldedViewRange<Storages, Ts, ndims, ndims_to_unfold>...> unfolded_views_range(
    const TensorView<Storages, Ts, ndims> &...views) {
  return combine_ranges(unfolded_view_range<ndims_to_unfold>(views)...);
}

#endif  // __cplusplus >= 201703L

}  // namespace sequence_utils

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_SEQUENCE_UTILS_H_
