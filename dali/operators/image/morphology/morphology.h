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

#ifndef DALI_OPERATORS_IMAGE_MORPHOLOGY_MORPHOLOGY_H_
#define DALI_OPERATORS_IMAGE_MORPHOLOGY_MORPHOLOGY_H_

#include <vector>
#include <string>

#include <cvcuda/OpMorphology.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Tensor.hpp>

#include "dali/operators/nvcvop/nvcvop.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"

namespace dali {

class Morphology : public nvcvop::NVCVSequenceOperator<StatelessOperator> {
 public:
  Morphology(const OpSpec &spec, NVCVMorphologyType morph_type) :
    nvcvop::NVCVSequenceOperator<StatelessOperator>(spec),
    morph_type_(morph_type),
    border_mode_(nvcvop::GetBorderMode(spec.GetArgument<std::string>("border_mode"))),
    iteration_(spec.GetArgument<int>("iterations")) {
      DALI_ENFORCE(iteration_ >= 1, "iterations must be >= 1");
    }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  bool ShouldExpandChannels(int input_idx) const override {
    return true;
  }

  void RunImpl(Workspace &ws) override;

  USE_OPERATOR_MEMBERS();
  NVCVMorphologyType morph_type_;
  ArgValue<int32_t, 1> mask_arg_{"mask_size", spec_};
  ArgValue<int32_t, 1> anchor_arg_{"anchor", spec_};
  nvcv::ImageBatchVarShape op_workspace_{};
  NVCVBorderType border_mode_{NVCV_BORDER_CONSTANT};
  int32_t iteration_ = 1;
};

class Dilate : public Morphology {
 public:
  explicit Dilate(const OpSpec &spec): Morphology(spec, NVCVMorphologyType::NVCV_DILATE) {}
};

class Erode : public Morphology {
 public:
  explicit Erode(const OpSpec &spec): Morphology(spec, NVCVMorphologyType::NVCV_ERODE) {}
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_MORPHOLOGY_MORPHOLOGY_H_
