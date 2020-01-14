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

#ifndef DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
#define DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_

#include <utility>
#include <vector>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace detail {

const std::string kCutoff = "cutoff_value";  // NOLINT
const int kNumOutputs = 2;
using OutputType = int;
static_assert(std::is_integral<OutputType>::value,
              "Operator return indices, thus OutputType shall be integral");

/**
 * Detects nonsilence region in provided 1-D audio buffer
 * @param cutoff Everything below this value will be reagrded as silence
 * @return (begin index, length)
 */
template<typename T>
std::pair<int, int> DetectNonsilenceRegion(span<const T> buffer, T cutoff) {
  int begin = -1;
  int end = buffer.size();
  while (begin < end && buffer[++begin] < cutoff);  // NOLINT
  if (begin == end) return {-1, 0};
  while (buffer[--end] < cutoff);  // NOLINT
  return {begin, end - begin + 1};
}

}  // namespace detail

template<typename Backend>
class NonsilenceOperator : public Operator<Backend> {
 public:
  ~NonsilenceOperator() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperator);

 protected:
  explicit NonsilenceOperator(const OpSpec &spec) :
          Operator<Backend>(spec),
          cutoff_(spec.GetArgument<float>(detail::kCutoff)) {}


  bool CanInferOutputs() const override {
    return true;
  }


  USE_OPERATOR_MEMBERS();
  const float cutoff_;
};


class NonsilenceOperatorCpu : public NonsilenceOperator<CPUBackend> {
 public:
  explicit NonsilenceOperatorCpu(const OpSpec &spec) : NonsilenceOperator<CPUBackend>(spec) {}


  ~NonsilenceOperatorCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorCpu);

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;
};


}  // namespace dali

#endif  // DALI_OPERATORS_AUDIO_NONSILENCE_OP_H_
