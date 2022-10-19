// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dev_buffer.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {

struct BbFlipSampleDesc {
  float *output;
  const float *input;
  bool horz;
  bool vert;
};

class BbFlipGPU : public BbFlip<GPUBackend> {
 public:
  explicit BbFlipGPU(const OpSpec &spec)
    : BbFlip<GPUBackend>(spec) {
    vert_gpu_.set_type<int32_t>();
    horz_gpu_.set_type<int32_t>();
  }

 protected:
  void RunImpl(Workspace &ws) override;
  using BbFlip<GPUBackend>::RunImpl;

 private:
  using BbFlip<GPUBackend>::horz_;
  using BbFlip<GPUBackend>::vert_;
  using BbFlip<GPUBackend>::ltrb_;

  Tensor<GPUBackend> horz_gpu_;
  Tensor<GPUBackend> vert_gpu_;

  using GpuBlockSetup = kernels::BlockSetup<1, 1>;

  GpuBlockSetup block_setup_;
  std::vector<BbFlipSampleDesc> samples_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BB_FLIP_CUH_
