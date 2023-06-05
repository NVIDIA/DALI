// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_BINNING_RESOURCE_H_
#define DALI_CORE_MM_BINNING_RESOURCE_H_

#include <algorithm>
#include <cassert>
#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/traits.h"
#include "dali/core/util.h"
#include "dali/core/mm/memory_resource.h"

namespace dali {
namespace mm {

/** @brief A memory resource that distributes allocation requests to several other resources
 *         based on the allocation size.
 *
 * This meta-resource aggregates multiple resources of the same kind and directs alllocation
 * and deallocation requests to these resources based on the allocation size. The proper resource
 * is selected by running a binary search on a sorted array of split points.
 * The split point is an inclusive upper bound - e.g. split points of 16 and 64k would result
 * in allocation up to 16 bytes going to resource 0, 17..64k to resource 1 and 64k+1 and up to
 * resource 2.
 *
 */
template <typename Kind,
          int nstatic_bins = -1,
          typename Interface = mm::memory_resource<Kind>>
class binning_resource_base : public Interface {
 public:
  using memory_kind = Kind;
  static constexpr int static_num_bins = nstatic_bins;
  static_assert(static_num_bins != 0);

  /** @brief Constructs a binning resource from an array of split points, resources and extra data
   *
   * This constructor servers is used in a type deduction guide.
   *
   * @tparam SplitPointCollection   a collection of split points; it must have element type
   *                                convertible to size_t and support iterationa and obtaining size
   *                                with std::size
   *
   * @tparam ResourceCollection     a collection of pointers or iterators to memory_resource<Kind>
   *                                objects; the elements must be convertible to a plain pointer
   *                                by applying operators &* in succession
   *
   * @param split_points            a collection of split points; it must have one fewer element
   *                                than resources
   * @param resources               a collection of memory resource pointers or smart poitners or
   *                                iterators
   */
  template <typename SplitPointCollection, typename ResourceCollection>
  binning_resource_base(const SplitPointCollection &split_points,
                        const ResourceCollection &resources) {
    if constexpr (static_num_bins > 0) {
      assert(split_points.size() == static_num_bins - 1);
      assert(resources.size() == static_num_bins);
    }

    assert(dali::size(split_points) + 1 == dali::size(resources));
    resize_if_possible(bin_split_points_, dali::size(split_points));
    resize_if_possible(resources_, dali::size(resources));

    int i = 0;
    for (auto split_point : split_points) {
      bin_split_points_[i] = split_point;
      if (i && split_point <= bin_split_points_[i-1])
        throw std::invalid_argument("Bin split points must be strictly increasing.");
      i++;
    }
    assert(i == num_bins() - 1);

    i = 0;
    for (auto &&ptr : resources)
      resources_[i++] = &*ptr;

    assert(i == num_bins());
  }

  int num_bins() const {
    return resources_.size();
  }

  size_t split_point(int after_bin) const {
    assert(after_bin >= 0 && after_bin < num_bins() - 1);
    return bin_split_points_[after_bin];
  }

  auto resource(int bin) const {
    assert(bin >= 0 && bin < num_bins());
    return resources_[bin];
  }

 private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    int bin = find_bin(bytes);
    return resource(bin)->allocate(bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    int bin = find_bin(bytes);
    resource(bin)->deallocate(ptr, bytes, alignment);
  }

  int find_bin(size_t size) const {
    auto bin_it = std::lower_bound(bin_split_points_.begin(), bin_split_points_.end(), size);
    return bin_it - bin_split_points_.begin();
  }

  template <typename T, int n>
  using store_t = std::conditional_t<(static_num_bins < 0),
    std::vector<T>, std::array<T, (n < 0 ? 0 : n)>>;

  store_t<size_t, static_num_bins - 1> bin_split_points_;
  store_t<mm::memory_resource<Kind> *, static_num_bins> resources_;
};

template <typename Kind,
          int nstatic_bins = -1,
          typename ExtraArgs = std::tuple<>,
          typename Interface = mm::memory_resource<Kind>>
class binning_resource : public binning_resource_base<Kind, nstatic_bins, Interface> {
 public:
  using base = binning_resource_base<Kind, nstatic_bins, Interface>;
  using memory_kind = Kind;

  /** @brief Constructs a binning resource from an array of split points, resources and extra data
   *
   * This constructor servers is used in a type deduction guide.
   *
   * @tparam SplitPointCollection   a collection of split points; it must have element type
   *                                convertible to size_t and support iterationa and obtaining size
   *                                with std::size
   *
   * @tparam ResourceCollection     a collection of pointers or iterators to memory_resource<Kind>
   *                                objects; the elements must be convertible to a plain pointer
   *                                by applying operators &* in succession
   *
   * @param split_points            a collection of split points; it must have one fewer element
   *                                than resources
   * @param resources               a collection of memory resource pointers or smart poitners or
   *                                iterators
   * @param extra                   extra payload
   */
  template <typename SplitPointCollection, typename ResourceCollection, typename Extra>
  binning_resource(const SplitPointCollection &split_points,
                   const ResourceCollection &resources,
                   Extra &&extra)
  : base(split_points, resources), extra_args_(std::forward<ExtraArgs>(extra)) {
  }

  template <typename SplitPointCollection, typename ResourceCollection>
  binning_resource(const SplitPointCollection &split_points,
                   const ResourceCollection &resources)
  : base(split_points, resources) {}

 private:
  ExtraArgs extra_args_{};
};

template <typename SplitPointCollection,
          typename ResourceCollection,
          typename Extra>
binning_resource(const SplitPointCollection &split, const ResourceCollection &res, Extra &&extra) ->
binning_resource<
  typename std::remove_reference_t<decltype(*res[0])>::memory_kind,
  -1,
  Extra>;

template <typename SplitPointCollection,
          typename ResourceCollection>
binning_resource(const SplitPointCollection &split, const ResourceCollection &res) ->
binning_resource<
  typename std::remove_reference_t<decltype(*res[0])>::memory_kind,
  -1>;

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_BINNING_RESOURCE_H_
