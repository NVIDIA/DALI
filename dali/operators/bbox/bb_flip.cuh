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

template <>
class BbFlip<GPUBackend> : public Operator<GPUBackend> {
 public:
  explicit BbFlip(const OpSpec &spec)
      : Operator<GPUBackend>(spec),
        horz_("horizontal", spec),
        vert_("vertical", spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) override {
    auto &input = ws.Input<GPUBackend>(0);
    auto shape = input.shape();
    auto nsamples = shape.size();
    horz_.Acquire(spec_, ws, nsamples, true);
    vert_.Acquire(spec_, ws, nsamples, true);
    return false;
  }

  void RunImpl(Workspace<GPUBackend> &ws) override;

 private:
  ArgValue<int> horz_;
  Tensor<GPUBackend> horz_gpu_;
  ArgValue<int> vert_;
  Tensor<GPUBackend> vert_gpu_;

  // contains a map from box index to sample index - used
  // for accessing per-sample horz/vert arguments.
  Tensor<GPUBackend> idx2sample_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BB_FLIP_CUH_
