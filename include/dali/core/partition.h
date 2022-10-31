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

#ifndef DALI_CORE_PARTITION_H_
#define DALI_CORE_PARTITION_H_

#include <type_traits>
#include <algorithm>
#include <tuple>
#include <utility>
#include "dali/core/util.h"

namespace dali {
namespace detail {

template <typename Iterator, typename Predicate>
auto multi_partition_impl(Iterator &&begin, Iterator &&end, Predicate &&pred) {
  auto it = std::stable_partition(begin, end, pred);
  return std::make_tuple(std::move(it));
}

template <typename Iterator, typename Predicate0, typename... Predicates>
auto
multi_partition_impl(Iterator &&begin, Iterator &&end, Predicate0 pred0, Predicates &&... preds) {
  auto second_group_begin = multi_partition_impl(
    begin,
    end,
    std::forward<Predicate0>(pred0));

  auto other_groups = multi_partition_impl(
    std::get<0>(second_group_begin),
    end,
    std::forward<Predicates>(preds)...);

  return std::tuple_cat(second_group_begin, other_groups);
}

}  // namespace detail

/**
 * @brief Partitions a range of elements according to predicates `preds`
 *
 * @tparam Iterator     a random-access iterator
 * @tparam Predicates   predicates callable on the elements of the collection
 * @param begin         iterator pointing to the first element in the range
 * @param end           iterator pointing to one-past the last element in the range
 * @param preds         the predicates by which to group the elements
 * @return A tuple of iterators pointing to the ends of regions satisfied by preds
 *
 * See the other overload for details.
 */
template <typename Iterator, typename... Predicates,
         typename = std::tuple<
            decltype(std::declval<Predicates>()(*std::declval<Iterator>()))...>>
auto
multi_partition(Iterator &&begin, Iterator &&end, Predicates &&... preds) {
    return detail::multi_partition_impl(
        std::forward<Iterator>(begin),
        std::forward<Iterator>(end),
        std::forward<Predicates>(preds)...);
}

/**
 * @brief Partitions a collection `c` according to predicates `preds`
 *
 * @tparam Collection   a random-accessible collection
 * @tparam Predicates   predicates callable on the elements of the collection
 * @param c             the colletion to partition
 * @param preds         the predicates by which to group the elements
 * @return A tuple of iterators pointing to the ends of regions satisfied by preds
 *
 * The collection is partitioned multiple times, by sequentially applying predicates `preds`.
 * After this function finishes, the collection is partitioned, so that the elements
 * that satisfy the first predicate go first, then elements that satisfy the second and so on.
 * The partitioning is stable, i.e. the elements within each group are not reordered.
 *
 * The predicates are applied sequentially and they don't need to establish ordering, i.e.
 * the grouping (not just the order) may be different if the predicates are reordered.
 *
 * The result is equivalent to applying an if ladder to each element:
 * ```
 * temp_collection group0, group1, ... groupN, tail;
 * for (auto &&item : c) {
 *   if (pred0(c))
 *     group0.push_back(std::move(c));
 *   else if (pred1(c))
 *     group1.push_back(std::move(c));
 *   ...
 *   else if (predN(c))
 *     groupN.push_back(std::move(c));
 *   else
 *     tail.push_back(std::move(c));
 * }
 * c = concatenate(group0, group1, ..., groupN, tail);
 *
 * return make_tuple(c.begin(), c.begin() + group0.size(), ..., c.begin() + groupN.size(), c.end());
 * ```
 *
 * Computational complexity: O(N * P) where:
 * N - the number of elements in c,
 * P - the number of predicates.
 */
template <typename Collection, typename... Predicates>
auto multi_partition(Collection &&c, Predicates &&... preds)
-> decltype(detail::multi_partition_impl(dali::begin(c), dali::end(c),
                                         std::forward<Predicates>(preds)...)) {
    return detail::multi_partition_impl(dali::begin(c), dali::end(c),
                                        std::forward<Predicates>(preds)...);
}

}  // namespace dali

#endif  // DALI_CORE_PARTITION_H_
