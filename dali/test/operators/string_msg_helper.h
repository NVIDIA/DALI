// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_OPERATORS_STRING_MSG_HELPER_H_
#define DALI_TEST_OPERATORS_STRING_MSG_HELPER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

/**
 * @brief Operator class that returns a precomputed string as an only sample.
 */
class StringMsgHelper : public Operator<CPUBackend> {
 public:
  inline explicit StringMsgHelper(const OpSpec &spec) : Operator<CPUBackend>(spec) {
  }

  inline ~StringMsgHelper() override = default;

  DISABLE_COPY_MOVE_ASSIGN(StringMsgHelper);
  USE_OPERATOR_MEMBERS();

 protected:
  virtual std::string GetMessage(const OpSpec &spec, const Workspace &ws) = 0;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    returned_message_ = GetMessage(spec_, ws);
    output_desc.resize(1);
    int bs = ws.GetRequestedBatchSize(0);
    TensorListShape<1> shape(bs);
    shape.set_tensor_shape(0, {static_cast<int64_t>(returned_message_.size())});
    output_desc[0].type = DALI_UINT8;
    output_desc[0].shape = shape;
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &out = ws.Output<CPUBackend>(0);
    auto *dst = out.template mutable_tensor<uint8_t>(0);
    memcpy(dst, returned_message_.data(), returned_message_.size());
  }
  std::string returned_message_;
};

}  // namespace dali

#endif  // DALI_TEST_OPERATORS_STRING_MSG_HELPER_H_
