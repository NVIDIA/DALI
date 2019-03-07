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
#include "dali/kernels/util.h"
#include "dali/kernels/kernel_traits.h"

namespace dali {
namespace testing {

template <typename Kernel_>
struct SimpleKernelTestBase {
  using Kernel = Kernel_;

  template <int i>
  using Input = typename std::remove_reference<
      typename std::tuple_element<i, kernels::kernel_inputs<Kernel>>::type>::type;

  template <int i>
  using Output = typename std::remove_reference<
      typename std::tuple_element<i, kernels::kernel_outputs<Kernel>>::type>::type;

  template <int i>
  using Arg = typename std::tuple_element<i, kernels::kernel_args<Kernel>>::type;

  template <int i>
  using InputElement = typename std::remove_const<element_t<Input<i>>>::type;
  template <int i>
  using OutputElement = element_t<Output<i>>;
};

}  // namespace testing
}  // namespace dali

#endif  // DALI_KERNELS_TEST_KERNEL_TEST_UTILS_H_
