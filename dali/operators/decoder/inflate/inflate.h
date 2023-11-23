// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_INFLATE_INFLATE_H_
#define DALI_OPERATORS_DECODER_INFLATE_INFLATE_H_

#include <memory>
#include <string>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/tensor_shape.h"
#include "dali/operators/decoder/inflate/inflate_params.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

namespace inflate {

template <typename Backend>
class InflateOpImplBase : public OpImplBase<Backend> {
 public:
  explicit InflateOpImplBase(const OpSpec &spec) : params_{spec} {
    dtype_ = spec.GetArgument<DALIDataType>(inflate::dTypeArgName);
    DALI_ENFORCE(
        IsFloatingPoint(dtype_) || IsIntegral(dtype_),
        make_string("The inflate output type must have floating point or integral type, got `",
                    dtype_, "` instead."));
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    auto input_type = ws.GetInputDataType(0);
    auto input_shape = ws.GetInputShape(0);
    DALI_ENFORCE(
        input_type == DALI_UINT8,
        make_string("Input must be a buffer with compressed data/data chunks represented as a 1D "
                    "tensor of bytes (uint8). Got tensor of type `",
                    input_type, "` instead."));
    DALI_ENFORCE(
        input_shape.sample_dim() == 1,
        make_string("Input must be a buffer with compressed data/data chunks represented as a 1D "
                    "tensor of uint8. Got input with ",
                    input_shape.sample_dim(), " dimensions instead."));
    params_.ProcessInputArgs(ws, input_shape.num_samples());
    output_desc.resize(1);
    output_desc[0].shape = params_.GetOutputShape();
    output_desc[0].type = dtype_;
    return true;
  }

 protected:
  DALIDataType dtype_;
  inflate::ShapeParams<Backend> params_;
};

}  // namespace inflate

template <typename Backend>
class Inflate : public StatelessOperator<Backend> {
 public:
  USE_OPERATOR_MEMBERS();

  explicit Inflate(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        alg_{inflate::parse_inflate_alg(spec.GetArgument<std::string>(inflate::algArgName))} {}

  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  void SetupOpImpl();

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    SetupOpImpl();
    assert(impl_ != nullptr);
    return impl_->SetupImpl(output_desc, ws);
  }

  void RunImpl(Workspace &ws) override {
    assert(impl_ != nullptr);
    impl_->RunImpl(ws);
  }

 protected:
  inflate::InflateAlg alg_;
  std::unique_ptr<inflate::InflateOpImplBase<Backend>> impl_ = nullptr;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_INFLATE_INFLATE_H_
