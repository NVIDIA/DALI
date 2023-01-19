// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_LUT_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_LUT_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace lut {

struct SampleDesc {
  static constexpr int range_size = 256;
  uint8_t *out;
  const uint64_t *in;
};

struct DLL_PUBLIC LutKernelGpu {
  static constexpr int kBlockSize = 256;

  /**
   * @brief Computes per-channel lookup tables for sample equalization based on the sample
   * histograms.
   *
   * @param ctx Kernel context
   * @param lut The per-channel lookup table remapping the uint8 range to uint8 range, stored in
   * channel-first manner
   * @param histogram The per-channel histogram with 256 equal bins, stored in channel-first manner.
   */
  DLL_PUBLIC void Run(KernelContext &ctx, const TensorListView<StorageGPU, uint8_t, 2> &lut,
                      const TensorListView<StorageGPU, const uint64_t, 2> &histogram);

  std::vector<SampleDesc> sample_descs_;
};

}  // namespace lut
}  // namespace equalize
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_LUT_H_
