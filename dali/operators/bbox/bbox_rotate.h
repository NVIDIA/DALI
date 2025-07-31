// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_BBOX_BBOX_ROTATE_H_
#define DALI_OPERATORS_BBOX_BBOX_ROTATE_H_

#include "dali/core/common.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

template <typename Backend>
class BBoxRotate : public StatelessOperator<Backend> {
  enum class Mode {
    Expand,
    Halfway,
    Fixed
  };

 public:
  explicit inline BBoxRotate(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        use_ltrb_(spec.GetArgument<bool>("ltrb")),
        keep_size_(spec.GetArgument<bool>("keep_size")) {
    const auto &mode_str = spec.GetArgument<std::string>("mode");
    if (mode_str == "expand") {
      mode_ = Mode::Expand;
    } else if (mode_str == "halfway") {
      mode_ = Mode::Halfway;
    } else if (mode_str == "fixed") {
      mode_ = Mode::Fixed;
    } else {
      DALI_FAIL("Unknown mode: ", mode_str, ". Supported modes are: expand, halfway, fixed.");
    }
  }

 protected:
  bool HasContiguousOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    TensorListShape buffer_shape(ws.GetInputShape(0));
    for (int i = 0; i < ws.GetRequestedBatchSize(1); i++) {
      buffer_shape.tensor_shape_span(i)[1] = 8;  // Need all four corners
    }
    bbox_rotate_buffer_.Resize(buffer_shape, DALI_FLOAT);
    return false;
  }

  void RunImpl(Workspace &ws) override;

  bool use_ltrb_;
  bool keep_size_;
  Mode mode_ = Mode::Expand;
  TensorList<Backend> bbox_rotate_buffer_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BBOX_ROTATE_H_
