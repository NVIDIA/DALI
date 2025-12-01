// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_PASTE_PASTE_GPU_INPUT_H_
#define DALI_KERNELS_IMGPROC_PASTE_PASTE_GPU_INPUT_H_

#include <vector>

namespace dali {
namespace kernels {
namespace paste {

template <int ndims>
struct MultiPasteSampleInput {
  struct InputPatch {
    ivec<ndims> out_anchor, in_anchor, size;
    int in_idx;  // which sample in the batch
    int batch_idx;  // which batch
  };
  std::vector<InputPatch> inputs;
  ivec<ndims> out_size;
  int channels;
};

}  // namespace paste
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_PASTE_PASTE_GPU_INPUT_H_
