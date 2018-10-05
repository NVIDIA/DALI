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
#include "dali/pipeline/operators/fused/normalize_permute.h"

namespace dali {

template <typename Backend>
class CropMirrorNormalize : public CropCastPermute<Backend>,
                            public NormalizeAttr<Backend> {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec) :
    CropCastPermute<Backend>(spec) {
      InitParam(spec, batch_size_);
  }

  virtual inline ~CropMirrorNormalize() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

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

    this->InitNormalizeAttr(spec, this->C_);
  }

  // Whether to pad output to 4 channels
  bool pad_;

  // Mirror?
  bool has_mirror_;

  Tensor<CPUBackend> mirror_;
  Tensor<GPUBackend> mirror_gpu_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
