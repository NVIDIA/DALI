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
  using Output = std::remove_reference_t<
      std::tuple_element_t<i, kernels::kernel_outputs<Kernel>>>;

  template <int i>
  using Arg = std::tuple_element_t<i, kernels::kernel_args<Kernel>>;

  template <int i>
  using InputElement = std::remove_const_t<element_t<Input<i>>>;
  template <int i>
  using OutputElement = element_t<Output<i>>;
};

}  // namespace testing
}  // namespace dali

#endif  // DALI_KERNELS_TEST_KERNEL_TEST_UTILS_H_
