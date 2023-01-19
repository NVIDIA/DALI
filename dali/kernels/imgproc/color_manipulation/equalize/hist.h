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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_HIST_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_HIST_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace hist {

struct SampleDesc {
  static constexpr int range_size = 256;
  uint64_t *out;
  const uint8_t *in;
  uint64_t num_elements;
  fast_div<uint64_t> num_channels;
};

struct DLL_PUBLIC HistogramKernelGpu {
  static constexpr int64_t kBlockSize = 256;
  static constexpr int64_t kMaxGridSize = 128;
  static constexpr int64_t kShmPerChannelSize = SampleDesc::range_size * sizeof(uint64_t);

  HistogramKernelGpu() : shared_mem_limit_{GetSharedMemPerBlock()} {}

  /**
   * @brief Computes the per-channel histograms.
   *
   * @param ctx Kernel context
   * @param out The per-channel histogram with 256 equal bins, stored in channel-first manner.
   * @param in The flattened sample, stored in channel-last manner.
   */
  DLL_PUBLIC void Run(KernelContext &ctx, TensorListView<StorageGPU, uint64_t, 2> &out,
                      const TensorListView<StorageGPU, const uint8_t, 2> &in);

 protected:
  int shared_mem_limit_;
  std::vector<SampleDesc> sample_descs_;
};

}  // namespace hist
}  // namespace equalize
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_HIST_H_
