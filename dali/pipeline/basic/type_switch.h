// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_BASIC_TYPE_SWITCH_H_
#define DALI_PIPELINE_BASIC_TYPE_SWITCH_H_

#include <tuple>
#include <utility>

#include "dali/common.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace basic {

template <typename T>
struct tuple_head {};

template <typename T, typename... Ts>
struct tuple_head<std::tuple<T, Ts...>> {
  using type = T;
};

template <typename T>
using tuple_head_t = typename tuple_head<T>::type;

template <typename T>
struct tuple_tail {};

template <typename T, typename... Ts>
struct tuple_tail<std::tuple<T, Ts...>> {
  using type = std::tuple<Ts...>;
};

template <typename T>
using tuple_tail_t = typename tuple_tail<T>::type;

template <typename T, typename U>
using map_to = T;

template <typename FoundTypes, template <typename...> class Op, typename... Ts>
struct type_switch_impl_tuples {};

template <typename FoundTypes, template <typename...> class Op, typename T, typename... Ts>
struct type_switch_impl_tuples<FoundTypes, Op, T, Ts...> {
  template <typename... U>
  static void Run(map_to<DALIDataType, T> type_id, map_to<DALIDataType, Ts>... type_ids, U&&... u) {
    // Go recursively over contents of type list T
    if (type_id == TypeInfo::Create<tuple_head_t<T>>().id()) {
      // add type to list of found types
      using NewResultInst = decltype(
          std::tuple_cat(std::declval<FoundTypes>(), std::declval<std::tuple<tuple_head_t<T>>>()));
      // search
      type_switch_impl_tuples<NewResultInst, Op, Ts...>::Run(type_ids..., std::forward<U>(u)...);
    } else {
      type_switch_impl_tuples<FoundTypes, Op, tuple_tail_t<T>, Ts...>::Run(type_id, type_ids...,
                                                                           std::forward<U>(u)...);
    }
  }
};

// Type search end
template <typename FoundTypes, template <typename...> class Op, typename... Ts>
struct type_switch_impl_tuples<FoundTypes, Op, std::tuple<>, Ts...> {
  template <typename... U>
  static void Run(DALIDataType type_id, map_to<DALIDataType, Ts>... type_ids, U&&... u) {
    // Should not be called
    DALI_FAIL("Unknown type");
  }
};

template <typename... Ts, template <typename...> class Op>
struct type_switch_impl_tuples<std::tuple<Ts...>, Op> {
  template <typename... U>
  static void Run(U&&... u) {
    Op<Ts...>::Run(std::forward<U>(u)...);
  }
};

/**
 * @brief Templated type-switch
 *
 * @tparam Op Template class with static void Run() method TODO(klecki) forward the result
 * @tparam Ts list of tuples for each template argument of Op to be matched
 */
template <template <typename...> class Op, typename... Ts>
struct type_switch {
  /**
   * @brief Runs the Op<T...>::Run(u...) for types matching type_ids while forwarding arguments u.
   *
   * @tparam U Type of argument to forward
   * @param type_ids Type id to match from one of the tuples in Ts
   * @param u arguments to be forwarded
   */
  template <typename... U>
  static void Run(map_to<DALIDataType, Ts>... type_ids, U&&... u) {
    type_switch_impl_tuples<std::tuple<>, Op, Ts...>::Run(type_ids..., std::forward<U>(u)...);
  }
};

// template <size_t N, typename T, typename... Ts>
// struct repeat_t {
//   using type = typename repeat_t<N - 1, T, T, Ts...>::type;
// };

// template <typename... Ts>
// struct repeat_t<0, Ts...> {
//   using type = std::tuple<Ts...>;
// };

// TODO(klecki): Describe usage, allow to use one list (how to create argument pack of N tuples?)
/*
  template <typename A, typename B>
  struct aaa {

    static void Run(int aa) {
      std::cout << TypeInfo::Create<A>().id() " " << TypeInfo::Create<A>().id() << " " aa;
    }
  };
  {
    static int x =0;
    type_switch<aaa, std::tuple<int8_t, int16_t, int32_t>, std::tuple<float>>::Run(
      TypeInfo::Create<int16_t>().id(), TypeInfo::Create<float>().id(), x++);
  }

*/

}  // namespace basic

using basic::type_switch;

}  // namespace dali

#endif  // DALI_PIPELINE_BASIC_TYPE_SWITCH_H_
