// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_

#include <vector>

#include "dali/pipeline/operators/attributes.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class NormalizePermute : public Operator<Backend>, protected NormalizeAttr<Backend>,
                         protected CastPermuteAttr {
 public:
  explicit inline NormalizePermute(const OpSpec &spec) : Operator<Backend>(spec),
             CastPermuteAttr(spec, false),
             H_(spec.GetArgument<int>("height")),
             W_(spec.GetArgument<int>("width")) {
    DALI_ENFORCE(H_ > 0);
    DALI_ENFORCE(W_ > 0);

    this->InitNormalizeAttr(spec, C_);
  }

  virtual inline ~NormalizePermute() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;
  void SetupSharedSampleParams(Workspace<Backend> *ws) override {
    CastPermuteAttr::SetupSharedSampleParams(ws);
  }

 private:
  void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  template <typename Out, class Converter>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  void CheckShape(const vector<Index> &shape) const {
    DALI_ENFORCE(shape.size() == 3,
                 "Expects 3-dim image input (v. " + std::to_string(shape.size()) + ")");
    DALI_ENFORCE(shape[0] == H_,
                 "Input image height does not match output height.");
    DALI_ENFORCE(shape[1] == W_,
                 "Input image width does not match output width.");
    DALI_ENFORCE(shape[2] == C_,
                 "Input image channels does not match output channels.");
  }

  int H_, W_;
  vector<Dims> output_shape_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_
