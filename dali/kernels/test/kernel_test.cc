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
#include "dali/kernels/static_switch.h"
#include "dali/kernels/tuple_helpers.h"

namespace dali {
namespace kernels {

template <typename Input1, typename Input2, typename OutputType>
using ExampleKernel = examples::Kernel<Input1, Input2, OutputType>;

// Neither function present
struct Empty {
};

// Two run functions
struct TwoRuns {
  static KernelRequirements GetRequirements(KernelContext &conext, const InListCPU<float, 3>&);
  void Run();
  static void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// No GetRequirements
struct NoGetReq {
  static void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// GetRequirements returns wrong type
struct GetReqBadType {
  static int GetRequirements(KernelContext &conext, const InListCPU<float, 3>&);
  static void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// GetRequirements doesn't take KernelContext & as its first argument
struct GetReqBadParamsType {
  static int GetRequirements(const InListCPU<float, 3>&);
  static void Run(KernelContext &conext, const OutListCPU<float, 3>&, const InListCPU<float, 3>&);
};

// Run doesn't take KernelContext & as its first argument
struct RunBadParamsType {
  static KernelRequirements GetRequirements(KernelContext &conext, const InListCPU<float, 3> &);
  static void Run(const OutListCPU<float, 3> &, const InListCPU<float, 3> &);
};

TEST(KernelAPI, InferIOArgs) {
  static_assert(std::is_same<
    kernel_inputs<ExampleKernel<float, float, float>>,
    std::tuple<const InListGPU<float, 3>&, const InTensorGPU<float, 4>&>
  >::value, "Wrong set of inputs detected");

  static_assert(std::is_same<
    kernel_outputs<ExampleKernel<float, int, int>>,
    std::tuple<const OutListGPU<int, 3>&>
  >::value, "Wrong set of outputs detected");

  static_assert(std::is_same<
    kernel_args<ExampleKernel<float, int, float>>,
    std::tuple<const std::vector<float>&>
  >::value, "Wrong set of arguments detected");
}

TEST(KernelAPI, EnforceConcept) {
  static_assert(detail::has_unique_function_Run<ExampleKernel<float, int, float>>::value,
                "ExampleKernel has Run function");

  static_assert(!detail::has_unique_function_Run<Empty>::value,
                "Empty has no Run function");
  static_assert(!detail::has_unique_function_Run<TwoRuns>::value,
                "TwoRuns has two Run functions");

  check_kernel<ExampleKernel<int, float, int>>();

  static_assert(!is_kernel<Empty>::value, "Empty has no Run function and cannot be a kernel");
  static_assert(!is_kernel<NoGetReq>::value,
                "Empty has no GetRequirements function and cannot be a kernel");
  static_assert(!is_kernel<TwoRuns>::value, "ToRuns has two Run functions");
  static_assert(!is_kernel<RunBadParamsType>::value, "Run has bad parameters");
}

template <typename I1, typename I2, typename O>
KernelRequirements dali::kernels::examples::Kernel<I1, I2, O>::GetRequirements(
  KernelContext &context,
  const InListGPU<I1, 3> &in1,
  const InTensorGPU<I2, 4> &in2,
  const std::vector<float> &aux) {
  return {};
}

template <typename I1, typename I2, typename O>
void dali::kernels::examples::Kernel<I1, I2, O>::Run(KernelContext &context,
  const OutListGPU<O, 3> &out,
  const InListGPU<I1, 3> &in1,
  const InTensorGPU<I2, 4> &in2,
  const std::vector<float> &aux) {}


TEST(KernelAPI, CallWithTuples) {
  InListGPU<float, 3> in1;
  InTensorGPU<int, 4> in2;
  OutListGPU<float, 3> out;
  std::vector<float> aux;

  examples::Kernel<float, int, float> K;
  KernelContext context;
  kernel::Run<decltype(K)>(context, std::tie(out), std::tie(in1, in2), std::tie(aux));
}

}  // namespace kernels
}  // namespace dali
