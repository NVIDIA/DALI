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

#ifndef DALI_OPERATORS_BBOX_BB_FLIP_CUH_
#define DALI_OPERATORS_BBOX_BB_FLIP_CUH_

#include <vector>
#include "dali/operators/bbox/bb_flip.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

class BbFlipGPU : public BbFlip<GPUBackend> {
 public:
  explicit BbFlipGPU(const OpSpec &spec)
    : BbFlip<GPUBackend>(spec) {
    vert_gpu_.set_type(TypeTable::GetTypeInfo(DALI_INT32));
    horz_gpu_.set_type(TypeTable::GetTypeInfo(DALI_INT32));
  }

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override;
  using BbFlip<GPUBackend>::RunImpl;

 private:
  using BbFlip<GPUBackend>::horz_;
  using BbFlip<GPUBackend>::vert_;
  using BbFlip<GPUBackend>::ltrb_;

  Tensor<GPUBackend> horz_gpu_;
  Tensor<GPUBackend> vert_gpu_;

  // contains a map from box index to sample index - used
  // for accessing per-sample horz/vert arguments.
  Tensor<GPUBackend> idx2sample_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BB_FLIP_CUH_
