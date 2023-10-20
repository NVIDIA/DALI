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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_LOOKUP_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_LOOKUP_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace lookup {

struct SampleDesc {
  static constexpr int range_size = 256;
  uint8_t *out;
  const uint8_t *in;
  const uint8_t *lut;
  uint64_t num_elements;
  fast_div<uint64_t> num_channels;
};

struct DLL_PUBLIC LookupKernelGpu {
  static constexpr int64_t kBlockSize = 256;
  static constexpr int64_t kMaxGridSize = 1024;

  /**
   * @brief Performs per-channel remap using the lookup table.
   *
   * The operation simply is `out[i] = lut[i % num_channels][in[i]]`.
   *
   * @param ctx Kernel context
   * @param out The remapped sample, stored in channel-last manner.
   * @param in The input sample, stored in channel-last manner.
   * @param lut The per-channel lut, stored in channel-first manner.
   */
  DLL_PUBLIC void Run(KernelContext &ctx, const TensorListView<StorageGPU, uint8_t, 2> &out,
                      const TensorListView<StorageGPU, const uint8_t, 2> &in,
                      const TensorListView<StorageGPU, const uint8_t, 2> &lut);

 protected:
  std::vector<SampleDesc> sample_descs_;
};

}  // namespace lookup
}  // namespace equalize
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_EQUALIZE_LOOKUP_H_
