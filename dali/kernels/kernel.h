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

#ifndef DALI_KERNELS_KERNEL_H_
#define DALI_KERNELS_KERNEL_H_

#include <vector>
#include "dali/kernels/context.h"
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/kernel_traits.h"
#include "dali/kernels/tuple_helpers.h"
#include "dali/kernels/util.h"

namespace dali {

/// @brief Defines the DALI kernel API. See dali::kernels::examples::Kernel for details
namespace kernels {

namespace examples {

/// @brief  DALI Kernel example
///
/// This class represents a "concept" of a DALI kernel.
/// A kernel must provide two non-overloaded functions:
/// Run and GetRequirements.
///
/// The kernel can be run directly or its inputs, outputs and arguments can be tied
/// into tuples and then the kernel be configured and launched using:
///
/// `dali::kernels::kernel::GetRequirements`
///
/// `dali::kernels::kernel::Run`
///
/// Programmer can check whether their type satisfies conditions for being a kernel
/// through instantiating check_kernel<KernelType>. If the type does not meet requirements,
/// static_asserts should produce meaningful diagnostics that will help to rectify the problem.
template <typename Input1, typename Input2, typename OutputType>
struct Kernel {
  /// @brief Returns kernel output(s) shape(s) and additional memory requirements
  ///
  /// GetRequirements receives full input and output tensor lists and any extra arguments that
  /// are going to be passed to a subsequent call to Run.
  ///
  /// @remarks The inputs are provided mainly to inspect their shapes; actually looking at the
  /// data may degrade performance severely.
  ///
  /// @param context - environment of the kernel;, cuda stream, batch info, etc.
  ///                  At the time of call to GetRequirements, its scratch area is undefined.
  ///
  /// @param in1 - example input, consisting of a list of 3D tensors with element type Input1
  /// @param in2 - example input, consisting of a 4D tensor with element type Input2
  /// @param aux - some extra parameters (e.g. convolution kernel, mask)
  static KernelRequirements GetRequirements(
    KernelContext &context,
    const InListGPU<Input1, 3> &in1,
    const InTensorGPU<Input2, 4> &in2,
    const std::vector<float> &aux);

  /// @brief Runs the kernel
  ///
  /// Run processes the inputs and populates the pre-allocated output. Output shape is expected
  /// to match that returned by GetRequirements.
  ///
  /// @param context - environment; provides scratch memory, cuda stream, batch info, etc.
  ///                  Scratch area must satisfy requirements returned by GetRequirements.
  /// @param in1 - example input, consisting of a list of 3D tensors with element type Input1
/// @param in2 - example input, consisting of a 4D tensor with element type Input2
    /// @param aux - some extra parameters (e.g. convolution kernel, mask)
  static void Run(
    KernelContext &context,
    const OutListGPU<OutputType, 3> &out,
    const InListGPU<Input1, 3> &in1,
    const InTensorGPU<Input2, 4> &in2,
    const std::vector<float> &aux);
};

}  // namespace examples

/// @brief A collection of pseudo-methods to operate on Kernel classes/objects
namespace kernel {

// avoid retyping "Kernel" every second word...

template <typename Kernel>
using inputs = kernel_inputs<Kernel>;

template <typename Kernel>
using outputs = kernel_outputs<Kernel>;

template <typename Kernel>
using args = kernel_args<Kernel>;

using Context = KernelContext;
using Requirements = KernelRequirements;

/// @brief Gets requirements for given Kernel
/// @param context            - execution environment (without scratch memory)
/// @param input              - kernel inputs, convertible to kernel_inputs<Kernel>
/// @param args               - kernel extra arguments, convertible to kernel_args<Kernel>
template <typename Kernel>
Requirements GetRequirements(
      Context &context,
      const inputs<Kernel> &input,
      const args<Kernel> &args) {
  check_kernel<Kernel>();
  return apply_all(Kernel::GetRequirements, context, input, args);
}

/// @brief Executes a Kernel on an input set
/// @param context             - execution environment (with scratch memory)
/// @param input               - kernel inputs, convertible to kernel_inputs<Kernel>
/// @param outputs             - kernel outputs, convertible to kernel_outputs<Kernel>
/// @param args                - kernel extra arguments, convertible to kernel_args<Kernel>
template <typename Kernel>
void Run(
      Context &context,
      const outputs<Kernel> &output,
      const inputs<Kernel> &input,
      const args<Kernel> &args) {
  check_kernel<Kernel>();
  apply_all(Kernel::Run, context, output, input, args);
}

/// @brief Default implementation of requirements for multiple input sets
/// @remarks Reuse scratch, append output shapes
///
/// @TODO(michalz) remove references from inputs/args?
template <typename Kernel>
Requirements GetRequirements(
      Context &context,
      const std::vector<inputs<Kernel>> &input_sets,
      const args<Kernel> &args) {
  check_kernel<Kernel>();
  if (input_sets.empty())
    return {};

  Requirements req = GetRequirements<Kernel>(context, input_sets[0], args);
  for (size_t i = 1; i < input_sets.size(); i++) {
    Requirements newReq = GetRequirements<Kernel>(context, input_sets[i], args);
    req.AddInputSet(newReq, true);
  }
  return req;
}

/// @brief Default implementation of execution for multiple input sets
///
/// @TODO(michalz) remove references from inputs/outputs/args?
template <typename Kernel>
void Run(Context &context,
      const std::vector<outputs<Kernel> > &output_sets,
      const std::vector<inputs<Kernel> > &input_sets,
      const args<Kernel> &args) {
  check_kernel<Kernel>();
  assert(output_sets.size() == input_sets.size());
  for (size_t i = 0; i < input_sets.size(); i++) {
    Run<Kernel>(context, output_sets[i], input_sets[i], args);
  }
}

}  // namespace kernel

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_H_
