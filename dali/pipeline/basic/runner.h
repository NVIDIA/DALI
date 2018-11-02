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

#ifndef DALI_PIPELINE_BASIC_RUNNER_H_
#define DALI_PIPELINE_BASIC_RUNNER_H_

#include <utility>

#include "dali/pipeline/basic/tensor.h"
#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {
namespace basic {

/**
 * @brief Create Runner and Resize helpers for Operator
 *
 * AllowMIS - AllowMultipleInputSets
 * Conviniently the operator must heave one input and output.
 *
 * @tparam Op - operator which we automatically create runner and resizer
 */
template <typename Op>
struct SizeHelperAllowMIS {
  template <typename... U>
  static void Run(SampleWorkspace *ws, const int idx, U &&... u) {
    const auto &input = ws->Input<CPUBackend>(idx);
    // using Op = SequenceCrop<uint8_t, Out, Seq>;

    std::array<Index, Op::input_dim> in_shape = TensorShape<Op::input_dim>(input).GetShape();
    std::array<Index, Op::output_dim> out_shape;
    Op::CalcOutputSize(in_shape, {std::forward<U>(u)...}, out_shape);

    Dims out_dim;
    for (const auto &s : out_shape) {
      out_dim.push_back(s);
    }
    ws->Output<CPUBackend>(idx)->Resize(out_dim);
  }
};

template <typename Op>
struct RunHelperAllowMIS {
  template <typename... U>
  static void Run(SampleWorkspace *ws, const int idx, U &&... u) {
    const auto &input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);
    // using Op = SequenceCrop<uint8_t, Out, Seq>;

    // ValidateHelper not needed - TensorWrapper ensures that ptr != nullptr.
    // TODO(klecki) - Input and output allocations should already be hanlded at this stage.

    typename Op::InputType in_wrapper(input);
    typename Op::OutputType out_wrapper(*output);
    Op::Run(in_wrapper, {std::forward<U>(u)...}, out_wrapper);
  }
};

}  // namespace basic
}  // namespace dali

#endif  // DALI_PIPELINE_BASIC_RUNNER_H_
