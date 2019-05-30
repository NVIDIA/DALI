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

#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class NormalizePermute : public Operator<Backend> {
 public:
  explicit inline NormalizePermute(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
    H_(spec.GetArgument<int>("height")),
    W_(spec.GetArgument<int>("width")),
    C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
    DALI_ENFORCE(H_ > 0);
    DALI_ENFORCE(W_ > 0);
    DALI_ENFORCE(C_ == 3 || C_ == 1);

    vector<float> mean, std;
    GetSingleOrRepeatedArg(spec, mean, "mean", C_);
    GetSingleOrRepeatedArg(spec, std, "std", C_);

    // Inverse the std-deviation
    for (int i = 0; i < C_; ++i) {
      std[i] = 1.f / std[i];
    }

    mean_.Copy(mean, 0);
    inv_std_.Copy(std, 0);

    output_shape_.resize(batch_size_);
    for (auto &shape : output_shape_) shape = {C_, H_, W_};
  }

  inline ~NormalizePermute() override = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  template <typename OUT>
  void CPURunHelper(const Tensor<CPUBackend> &input, Tensor<CPUBackend> &output);

  template <typename OUT>
  void GPURunHelper(DeviceWorkspace *ws, const int idx);

  Tensor<Backend> mean_, inv_std_;
  DALIDataType output_type_;
  int H_, W_, C_;
  vector<Dims> output_shape_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_
