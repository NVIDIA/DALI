// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_JPEG_JPEG_DISTORTION_CPU_KERNEL_H_
#define DALI_KERNELS_IMGPROC_JPEG_JPEG_DISTORTION_CPU_KERNEL_H_

#include "dali/core/api_helper.h"
#include "dali/core/span.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace jpeg {

// CPU port of the GPU lossy-only JPEG distortion kernel: RGB -> YCbCr (with
// optional 2:1 horizontal/vertical chroma subsampling) -> 8x8 forward DCT ->
// Annex-K quantization -> 8x8 inverse DCT -> YCbCr -> RGB. Skips the
// Huffman/bitstream stages of real JPEG, just like the GPU kernel.
//
// Threading is the caller's responsibility. RunSample processes a single
// image; the operator drives a thread pool across frames/samples.
class DLL_PUBLIC JpegCompressionDistortionCPU {
 public:
  // Apply the distortion to one image. `out` may alias `in`: the kernel
  // processes one macroblock at a time, completes all RGB reads (with
  // clamp-to-edge sampling on partial edge macroblocks) into per-macroblock
  // float scratch before producing any RGB output, and the partial-macroblock
  // clamp coordinates W-1 / H-1 always lie within the current macroblock's
  // own pixel span -- so a clamped read can never hit a pixel that an earlier
  // macroblock has already overwritten.
  //
  // @param out             interleaved RGB output, shape (H, W, 3)
  // @param in              interleaved RGB input,  shape (H, W, 3)
  // @param quality         JPEG quality factor in [1, 100] (clamped internally)
  // @param horz_subsample  if true, average chroma horizontally 2:1
  // @param vert_subsample  if true, average chroma vertically 2:1
  void RunSample(const TensorView<StorageCPU, uint8_t, 3> &out,
                 const TensorView<StorageCPU, const uint8_t, 3> &in,
                 int quality,
                 bool horz_subsample,
                 bool vert_subsample);
};

}  // namespace jpeg
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_JPEG_DISTORTION_CPU_KERNEL_H_
