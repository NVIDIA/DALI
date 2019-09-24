// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_TEST_KERNEL_TEST_UTILS_H_
#define DALI_KERNELS_TEST_KERNEL_TEST_UTILS_H_

#include <type_traits>
#include <tuple>
#include "dali/core/util.h"
#include "dali/kernels/kernel_traits.h"

namespace dali {
namespace testing {

template <typename Kernel_>
struct SimpleKernelTestBase {
  using Kernel = Kernel_;

  template <int i>
  using Input =  std::remove_reference_t<
          std::tuple_element_t<i, kernels::kernel_inputs<Kernel>>>;

  template <int i>
  using Output = std::remove_reference_t<std::tuple_element_t<i,
          kernels::kernel_outputs<Kernel>>>;

  template <int i>
  using Arg = std::tuple_element_t<i, kernels::kernel_args<Kernel>>;

  template <int i>
  using InputElement = std::remove_const_t<element_t<Input<i>>>;

  template <int i>
  using OutputElement = element_t<Output<i>>;
};

namespace detail {

template <template <typename A, typename B> class Pair, typename T1, typename... T>
using FixedFirstTypePairs = std::tuple<Pair<T1, T>...>;

template <template <typename A, typename B> class Pair, typename A, typename B>
struct AllPairsHelper;

template <template <typename A, typename B> class Pair, typename... A, typename... B>
struct AllPairsHelper<Pair, std::tuple<A...>, std::tuple<B...>> {
  using type = dali::detail::tuple_cat_t<FixedFirstTypePairs<Pair, A, B...>...>;
};

template <template <typename A, typename B> class Pair, typename TupleA, typename TupleB>
using AllPairs = typename AllPairsHelper<Pair, TupleA, TupleB>::type;

template <typename Tuple>
struct TupleToGTest;

template <typename... T>
struct TupleToGTest<std::tuple<T...>> {
  using type = ::testing::Types<T...>;
};

template <class InputType, class OutputType>
struct InputOutputTypes {
  using In = InputType;
  using Out = OutputType;
};

template <typename T>
struct is_tuple : std::false_type {
};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {
};


}  // namespace detail

/**
 * Registers GTest TYPED_TESTs, filling them up with a struct, that contains 2 types,
 * that are a carthesian product: `TupleWithTypes x TupleWithTypes`.
 *
 * This method is made to create tests for kernels,
 * that can accept different types for input and output.
 *
 * To use it, define a tuple with types (maximum number of types in the tuple is 7,
 * due to GTest limitations) and pass it as an argument to this macro:
 *
 * ```
 * template<class T>
 * class MyTestCase : public ::testing::Test {};
 *
 * using MyTestTypes = std::tuple<int, float, char>;
 * INPUT_OUTPUT_TYPED_TEST_SUITE(MyTestCase, MyTestTypes);
 * ```
 *
 * Snippet above will fill the TYPED_TESTs with following type pairs ( [input output] ):
 * [int int], [int float], [int char], [float int], [float float],
 * [float char], [char int], [char float], [char char]
 *
 * To reference given type, use regular GTest's `TypeParam` type, with `::In` and `::Out` suffixes:
 *
 * ```
 * TYPED_TEST(MyTestCase, test) {
 *   TypeParam::In in_value;
 *   std::vector<typename TypeParam::Out> out_vec;
 * }
 * ```
 *
 * @param CaseName Test case name (like in GTest)
 * @param TupleWithTypes std::tuple containing required types, e.g. std::tuple<int, char>
 */
#define INPUT_OUTPUT_TYPED_TEST_SUITE(CaseName, TupleWithTypes)                                    \
  static_assert(::dali::testing::detail::is_tuple<TupleWithTypes>::value,                          \
                "TupleWithTypes has to be a tuple");                                               \
  static_assert(std::tuple_size<TupleWithTypes>::value <= 7,                                       \
                "Maximum size of a tuple is 7 (enforced by GTest)");                               \
  static_assert(std::tuple_size<TupleWithTypes>::value >= 1,                                       \
                "TupleWithTypes has to contain at least 1 type");                                  \
  using MyTypesTuple = ::dali::testing::detail::AllPairs<                                          \
                       ::dali::testing::detail::InputOutputTypes, TupleWithTypes, TupleWithTypes>; \
  using GTestTypes = ::dali::testing::detail::TupleToGTest<MyTypesTuple>::type;                    \
  TYPED_TEST_SUITE(CaseName, GTestTypes)

}  // namespace testing
}  // namespace dali

#endif  // DALI_KERNELS_TEST_KERNEL_TEST_UTILS_H_
