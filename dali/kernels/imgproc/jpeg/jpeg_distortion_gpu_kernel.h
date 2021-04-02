// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_JPEG_JPEG_DISTORTION_GPU_KERNEL_H_
#define DALI_KERNELS_IMGPROC_JPEG_JPEG_DISTORTION_GPU_KERNEL_H_

#include <vector>
#include "dali/core/convert.h"
#include "dali/core/geom/mat.h"
#include "dali/core/geom/vec.h"
#include "dali/core/tensor_view.h"
#include "dali/core/util.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace jpeg {

inline float GetQualityFactorScale(int quality) {
  quality = clamp<int>(quality, 1, 100);
  float q_scale = 1.0f;
  if (quality < 50) {
    q_scale = 50.0f / quality;
  } else {
    q_scale = 2.0f - (2 * quality / 100.0f);
  }
  return q_scale;
}

// Quantization table coefficients that are suggested in the Annex K of the JPEG standard.

inline mat<8, 8, uint8_t> GetLumaQuantizationTable(int quality) {
  mat<8, 8, uint8_t> table = {{
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
  }};
  auto scale = GetQualityFactorScale(quality);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      table(i, j) = std::max<uint8_t>(ConvertSat<uint8_t>(scale * table(i, j)), 1);
    }
  }
  return table;
}

inline mat<8, 8, uint8_t> GetChromaQuantizationTable(int quality) {
  mat<8, 8, uint8_t> table = {{
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
  }};
  auto scale = GetQualityFactorScale(quality);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      table(i, j) = std::max<uint8_t>(ConvertSat<uint8_t>(scale * table(i, j)), 1);
    }
  }
  return table;
}

struct SampleDesc {
  const uint8_t *in;  // rgb
  uint8_t *out;  // rgb
  ivec<2> size;
  i64vec<2> strides;
  mat<8, 8, uint8_t> luma_Q_table;
  mat<8, 8, uint8_t> chroma_Q_table;
};

class DLL_PUBLIC JpegDistortionBaseGPU {
 public:
  KernelRequirements Setup(KernelContext &ctx, const TensorListShape<3> &in_shape,
                           bool horz_subsample, bool vert_subsample);

 protected:
  void SetupSampleDescs(const OutListGPU<uint8_t, 3> &out, const InListGPU<uint8_t, 3> &in,
                        span<const int> quality = {});

  using BlkSetup = BlockSetup<2, -1>;
  BlkSetup block_setup_;
  using BlockDesc = BlkSetup::BlockDesc;

  TensorListShape<2> chroma_shape_;
  std::vector<SampleDesc> sample_descs_;
  bool horz_subsample_ = true;
  bool vert_subsample_ = true;
};

class DLL_PUBLIC JpegCompressionDistortionGPU : public JpegDistortionBaseGPU {
 public:
  void Run(KernelContext &ctx, const OutListGPU<uint8_t, 3> &out, const InListGPU<uint8_t, 3> &in,
           span<const int> quality);

 private:
  using Base = JpegDistortionBaseGPU;
  using BlockDesc = typename Base::BlockDesc;
  using Base::SetupSampleDescs;
  using Base::block_setup_;
  using Base::sample_descs_;
};

class DLL_PUBLIC ChromaSubsampleDistortionGPU : public JpegDistortionBaseGPU {
 public:
  void Run(KernelContext &ctx, const OutListGPU<uint8_t, 3> &out, const InListGPU<uint8_t, 3> &in);

 private:
  using Base = JpegDistortionBaseGPU;
  using BlockDesc = typename Base::BlockDesc;
  using Base::SetupSampleDescs;
  using Base::block_setup_;
  using Base::sample_descs_;
};

}  // namespace jpeg
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_JPEG_DISTORTION_GPU_KERNEL_H_
