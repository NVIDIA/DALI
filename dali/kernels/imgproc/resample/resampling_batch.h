// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_BATCH_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_BATCH_H_

#include "dali/kernels/imgproc/resample/resampling_setup.h"

namespace dali {
namespace kernels {

template <int which_pass, typename Output, typename Input>
void BatchedSeparableResample(Output *out, const Input *in,
  const SeparableResamplingSetup::SampleDesc *samples,
  int num_samples, const SampleBlockInfo *block2sample, int num_blocks,
  int2 block_size,
  cudaStream_t stream);

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_BATCH_H_
