// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_COLOR_EQUALIZE_H_
#define DALI_OPERATORS_IMAGE_COLOR_EQUALIZE_H_

#include <vector>

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

template <typename Backend>
class Equalize : public SequenceOperator<Backend, StatelessOperator> {
 public:
  explicit Equalize(const OpSpec &spec) : SequenceOperator<Backend, StatelessOperator>(spec) {}

 protected:
  DISABLE_COPY_MOVE_ASSIGN(Equalize);
  USE_OPERATOR_MEMBERS();

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    auto input_type = ws.GetInputDataType(0);
    DALI_ENFORCE(input_type == type2id<uint8_t>::value,
                 make_string("Unsupported input type for equalize operator: ", input_type,
                             ". Expected input type: `uint8_t`."));

    output_desc[0].type = input_type;
    // output_desc[0].shape is set by ProcessOutputDesc
    return true;
  }

  bool CanInferOutputs() const override {
    return true;
  }

  bool ShouldExpandChannels(int input_idx) const override {
    (void)input_idx;
    return true;
  }

  // Overrides unnecessary shape coalescing for video/sequence inputs
  bool ProcessOutputDesc(std::vector<OutputDesc> &output_desc, const Workspace &ws,
                         bool is_inferred) override {
    assert(is_inferred && output_desc.size() == 1);
    const auto &input = ws.Input<Backend>(0);
    // The shape of data stays untouched
    output_desc[0].shape = input.shape();
    return true;
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_EQUALIZE_H_
