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
namespace resampling {

template <int spatial_ndim, typename Output, typename Input>
void BatchedSeparableResample(
  int which_pass,
  const SampleDesc<spatial_ndim> *samples,
  const BlockDesc<spatial_ndim> *block2sample, int num_blocks,
  ivec3 block_size,
  cudaStream_t stream);

}  // namespace resampling
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_BATCH_H_
