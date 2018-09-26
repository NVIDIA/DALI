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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_

#include "dali/pipeline/operators/fused/crop_cast_permute.h"
#include <vector>

namespace dali {

template <typename Backend>
class CropMirrorNormalize : public CropCastPermute<Backend> {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec) :
    CropCastPermute<Backend>(spec) {
      InitParam(spec, batch_size_);
  }

  virtual inline ~CropMirrorNormalize() = default;

 protected:
  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

 private:
  template <typename OUT>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  virtual int GetPad() const                      { return pad_ ? 4 : this->C_; }

  void InitParam(const OpSpec &spec, int size) {
    pad_ = spec.GetArgument<bool>("pad_output");
    has_mirror_ = spec.HasTensorArgument("mirror");
    if (!has_mirror_) {
      vector<int> m(size, spec.GetArgument<int>("mirror"));
      mirror_.Copy(m, 0);
    }

    const auto C = this->C_;
    vector<float> mean_vec, inv_std_vec;
    GetSingleOrRepeatedArg(spec, &mean_vec, "mean", C);
    GetSingleOrRepeatedArg(spec, &inv_std_vec, "std", C);

    // Inverse the std-deviation
    for (int i = 0; i < C; ++i)
      inv_std_vec[i] = 1.f / inv_std_vec[i];

    mean_.Copy(mean_vec, 0);
    inv_std_.Copy(inv_std_vec, 0);
  }

  // Whether to pad output to 4 channels
  bool pad_;

  // Mirror?
  bool has_mirror_;

  Tensor<CPUBackend> mirror_;
  Tensor<GPUBackend> mirror_gpu_;

  // Tensor to store mean & stddiv
  Tensor<Backend> mean_, inv_std_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
