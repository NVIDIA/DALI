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

#include <gtest/gtest.h>
#include <tuple>
#include "dali/kernels/kernel.h"
#include "dali/kernels/type_tag.h"
#include "dali/core/static_switch.h"
#include "dali/core/tuple_helpers.h"

namespace dali {
namespace kernels {

template <typename OutputType, typename Input1, typename Input2>
using ExampleKernel = examples::Kernel<OutputType, Input1, Input2>;

// Neither function present
struct Empty {
};

// Two run functions
struct TwoRuns {
  KernelRequirements Setup(KernelContext &conext, const InListCPU<float, 3>&);
  void Run();
  void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// No Setup
struct NoGetReq {
  void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// Setup returns wrong type
struct GetReqBadType {
  int Setup(KernelContext &conext, const InListCPU<float, 3>&);
  void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// Setup doesn't take KernelContext & as its first argument
struct GetReqBadParamsType {
  int Setup(const InListCPU<float, 3>&);
  void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// Run doesn't take KernelContext & as its first argument
struct RunBadParamsType {
  KernelRequirements Setup(KernelContext &conext, const InListCPU<float, 3> &);
  void Run(const OutListCPU<float, 3> &, const InListCPU<float, 3> &);
};

TEST(KernelAPI, InferIOArgs) {
  static_assert(std::is_same<
    kernel_inputs<ExampleKernel<float, float, float>>,
    std::tuple<const InListGPU<float, 3>&, const InTensorGPU<float, 4>&>
  >::value, "Wrong set of inputs detected");

  static_assert(std::is_same<
    kernel_outputs<ExampleKernel<int, float, int>>,
    std::tuple<const OutListGPU<int, 3>&>
  >::value, "Wrong set of outputs detected");

  static_assert(std::is_same<
    kernel_args<ExampleKernel<float, float, int>>,
    std::tuple<const std::vector<float>&>
  >::value, "Wrong set of arguments detected");
}

TEST(KernelAPI, EnforceConcept) {
  static_assert(detail::has_unique_member_function_Run<ExampleKernel<float, float, int>>::value,
                "ExampleKernel has Run function");

  static_assert(!detail::has_unique_member_function_Run<Empty>::value,
                "Empty has no Run function");
  static_assert(!detail::has_unique_member_function_Run<TwoRuns>::value,
                "TwoRuns has two Run functions");

  check_kernel<ExampleKernel<int, int, float>>();

  static_assert(!is_kernel<Empty>::value, "Empty has no Run function and cannot be a kernel");
  static_assert(!is_kernel<NoGetReq>::value,
                "Empty has no Setup function and cannot be a kernel");
  static_assert(!is_kernel<TwoRuns>::value, "ToRuns has two Run functions");
  static_assert(!is_kernel<RunBadParamsType>::value, "Run has bad parameters");
}

template <typename O, typename I1, typename I2>
KernelRequirements dali::kernels::examples::Kernel<O, I1, I2>::Setup(
  KernelContext &context,
  const InListGPU<I1, 3> &in1,
  const InTensorGPU<I2, 4> &in2,
  const std::vector<float> &aux) {
  return {};
}

template <typename O, typename I1, typename I2>
void dali::kernels::examples::Kernel<O, I1, I2>::Run(KernelContext &context,
  const OutListGPU<O, 3> &out,
  const InListGPU<I1, 3> &in1,
  const InTensorGPU<I2, 4> &in2,
  const std::vector<float> &aux) {}


TEST(KernelAPI, CallWithTuples) {
  InListGPU<float, 3> in1;
  InTensorGPU<int, 4> in2;
  OutListGPU<float, 3> out;
  std::vector<float> aux;

  examples::Kernel<float, float, int> K;
  KernelContext context;
  kernel::Run(K, context, std::tie(out), std::tie(in1, in2), std::tie(aux));
}

}  // namespace kernels
}  // namespace dali
