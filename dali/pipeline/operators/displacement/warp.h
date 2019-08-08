// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_H_

#include <tuple>
#include <utility>
#include "dali/pipeline/operators/operator.h"
#include "dali/core/tuple_helpers.h"

namespace dali {

template <typename... Pairs>
struct UnzipPairsHelper;

template <typename... Pairs>
using UnzipPairs = typename UnzipPairsHelper<Pairs...>::type;

template <>
struct UnzipPairsHelper<> {
  using type = std::tuple<>;
};

template <typename T1, typename T2, typename... Tail>
struct UnzipPairsHelper<T1, T2, Tail...> {
  static_assert(sizeof...(Tail) % 2 == 0, "Number of types for unzip must be even");
  using type = detail::tuple_cat_t<std::tuple<std::pair<T1, T2>>, UnzipPairs<Tail...>>;
};

template <typename Backend, typename Derived>
class Warp;

}  // namespace dali

#endif  //  DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_H_
