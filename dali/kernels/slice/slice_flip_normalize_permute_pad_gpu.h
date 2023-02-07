// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"

namespace dali {
namespace kernels {

template <typename OutputType, typename InputType, int Dims>
class SliceFlipNormalizePermutePadGpu {
 private:
  static constexpr size_t kBlockDim = 256;
  size_t block_size_ = 32 * kBlockDim;
  size_t block_count_ = 0;

  using ProcessedArgs = slice_impl::SliceFlipNormalizePermutePadProcessedArgs<Dims>;
  std::vector<ProcessedArgs> processed_args_;
  int norm_args_size_ = -1;
  bool has_channels_ = false;
  bool need_normalize_ = false;
  int nfill_values_ = 0;
  int channel_dim_ = -1;

 public:
  KernelRequirements Setup(KernelContext &context, const InListGPU<InputType, Dims> &in,
                           const std::vector<SliceFlipNormalizePermutePadArgs<Dims>> &args);

  void Run(KernelContext &context, const OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<SliceFlipNormalizePermutePadArgs<Dims>> &args);
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_
