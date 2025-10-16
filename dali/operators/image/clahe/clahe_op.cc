// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <iostream>

#include "dali/core/backend_tags.h"
#include "dali/core/error_handling.h"
#include "dali/core/mm/memory.h"
#include "dali/core/tensor_layout.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

// External CUDA launcher prototypes (from clahe_op.cu)
void LaunchCLAHE_Grayscale_U8_NHWC(uint8_t *dst_gray, const uint8_t *src_gray, int H, int W,
                                   int tiles_x, int tiles_y, float clip_limit_rel,
                                   unsigned int *tmp_histograms, uint8_t *tmp_luts,
                                   cudaStream_t stream);

void LaunchCLAHE_RGB_U8_NHWC(uint8_t *dst_rgb, const uint8_t *src_rgb, uint8_t *y_plane, int H,
                             int W, int tiles_x, int tiles_y, float clip_limit_rel,
                             unsigned int *tmp_histograms, uint8_t *tmp_luts, cudaStream_t stream);

// Optimized version with fused kernels
void LaunchCLAHE_RGB_U8_NHWC_Optimized(uint8_t *dst_rgb, const uint8_t *src_rgb, uint8_t *y_plane,
                                       int H, int W, int tiles_x, int tiles_y, float clip_limit_rel,
                                       unsigned int *tmp_histograms, uint8_t *tmp_luts,
                                       cudaStream_t stream);

// -----------------------------------------------------------------------------
// Operator definition
// -----------------------------------------------------------------------------
class ClaheGPU : public Operator<GPUBackend> {
 public:
  explicit ClaheGPU(const OpSpec &spec)
      : Operator<GPUBackend>(spec),
        tiles_x_(spec.GetArgument<int>("tiles_x")),
        tiles_y_(spec.GetArgument<int>("tiles_y")),
        bins_(spec.GetArgument<int>("bins")),
        clip_limit_(spec.GetArgument<float>("clip_limit")),
        luma_only_(spec.GetArgument<bool>("luma_only")) {}

  bool SetupImpl(std::vector<OutputDesc> &outputs, const Workspace &ws) override {
    const auto &in = ws.Input<GPUBackend>(0);
    outputs.resize(1);
    outputs[0].type = in.type();
    outputs[0].shape = in.shape();  // same layout/shape as input
    return true;
  }

  void RunImpl(Workspace &ws) override {
    const auto &in = ws.Input<GPUBackend>(0);
    auto &out = ws.Output<GPUBackend>(0);
    auto stream = ws.stream();

    DALI_ENFORCE(in.type() == DALI_UINT8, "ClaheGPU currently supports only uint8 input.");

    const auto &shape = in.shape();
    int N = shape.num_samples();

    // Use DynamicScratchpad for automatic memory management
    kernels::DynamicScratchpad scratchpad(stream);

    for (int i = 0; i < N; i++) {
      auto sample_shape = shape.tensor_shape_span(i);
      DALI_ENFORCE(sample_shape.size() == 3, "ClaheGPU expects HWC input layout.");

      int H = sample_shape[0];
      int W = sample_shape[1];
      int C = sample_shape[2];
      DALI_ENFORCE(C == 1 || C == 3, "ClaheGPU supports 1 or 3 channels.");

      const uint8_t *in_ptr = in.tensor<uint8_t>(i);
      uint8_t *out_ptr = out.mutable_tensor<uint8_t>(i);

      // Allocate temporary buffers on demand using scratchpad
      int tiles_total = tiles_x_ * tiles_y_;
      unsigned int *histograms = scratchpad.AllocateGPU<unsigned int>(tiles_total * bins_);
      uint8_t *luts = scratchpad.AllocateGPU<uint8_t>(tiles_total * bins_);
      uint8_t *y_plane = (C == 3) ? scratchpad.AllocateGPU<uint8_t>(H * W) : nullptr;

      if (C == 1) {
        LaunchCLAHE_Grayscale_U8_NHWC(out_ptr, in_ptr, H, W, tiles_x_, tiles_y_, clip_limit_,
                                      histograms, luts, stream);
      } else {
        if (luma_only_) {
          // Use optimized fused kernel for better performance
          LaunchCLAHE_RGB_U8_NHWC_Optimized(out_ptr, in_ptr, y_plane, H, W, tiles_x_, tiles_y_,
                                            clip_limit_, histograms, luts, stream);
        } else {
          // Apply per-channel CLAHE (simple fallback: run per-channel grayscale)
          for (int c = 0; c < 3; ++c) {
            const uint8_t *src_ch = in_ptr + c;
            uint8_t *dst_ch = out_ptr + c;
            LaunchCLAHE_Grayscale_U8_NHWC(dst_ch, src_ch, H, W, tiles_x_, tiles_y_, clip_limit_,
                                          histograms, luts, stream);
          }
        }
      }
    }

    // Memory is automatically cleaned up when scratchpad goes out of scope
    // DALI handles stream synchronization automatically - no need to block here
  }

 private:
  int tiles_x_, tiles_y_, bins_;
  float clip_limit_;
  bool luma_only_;
};

// -----------------------------------------------------------------------------
// Schema and registration
// -----------------------------------------------------------------------------
DALI_SCHEMA(Clahe)
    .DocStr(R"code(Contrast Limited Adaptive Histogram Equalization (CLAHE) operator.
    
Performs local histogram equalization with clipping and bilinear blending 
of lookup tables (LUTs) between neighboring tiles. This technique enhances 
local contrast while preventing over-amplification of noise.
Attempts to use same algorithm as OpenCV 
(https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).
The input image is divided into rectangular tiles, and histogram equalization
is applied to each tile independently. To avoid artifacts at tile boundaries,
the lookup tables are bilinearly interpolated between neighboring tiles.
Supports both grayscale (1-channel) and RGB (3-channel) uint8 images in HWC layout.
For RGB images, by default CLAHE is applied to the luminance channel only (luma_only=True),
preserving color relationships. When luma_only=False, CLAHE is applied to each 
color channel independently.
**Performance**: The GPU variant of this operator includes automatic optimizations (kernel fusion, 
warp-privatized histograms, vectorized memory access) that provide 1.5-3x speedup 
while maintaining OpenCV algorithmic compatibility.
Example usage:
  # Grayscale image
  clahe_out = fn.clahe(grayscale_image, tiles_x=8, tiles_y=8, clip_limit=2.0)
  
  # RGB image with luminance-only processing (default)
  clahe_out = fn.clahe(rgb_image, tiles_x=8, tiles_y=8, clip_limit=3.0, luma_only=True)
  
  # RGB image with per-channel processing
  clahe_out = fn.clahe(rgb_image, tiles_x=8, tiles_y=8, clip_limit=2.0, luma_only=False)
)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("tiles_x", R"code(Number of tiles along the image width.
    
Higher values provide more localized enhancement but may introduce artifacts.
Typical values range from 4 to 16. Must be positive.)code",
            DALI_INT32)
    .AddArg("tiles_y", R"code(Number of tiles along the image height.
    
Higher values provide more localized enhancement but may introduce artifacts.
Typical values range from 4 to 16. Must be positive.)code",
            DALI_INT32)
    .AddArg("clip_limit", R"code(Relative clip limit multiplier for histogram bins.
    
Controls the contrast enhancement strength. The actual clip limit is calculated as:
clip_limit * (tile_area / bins). Values > 1.0 enhance contrast, while values 
close to 1.0 provide minimal enhancement. Typical values range from 1.5 to 4.0.
Higher values may cause over-enhancement and artifacts.)code",
            DALI_FLOAT)
    .AddOptionalArg("bins", R"code(Number of histogram bins for CLAHE computation.
    
Must be a power of 2. Higher values provide finer histogram resolution but 
increase computation cost. For uint8 images, 256 bins provide optimal results.)code",
                    256)
    .AddOptionalArg("luma_only", R"code(For RGB inputs, apply CLAHE to luminance channel only.
    
When True (default), CLAHE is applied to the luminance (Y) component of RGB images,
preserving color relationships. The RGB channels are then scaled proportionally.
When False, CLAHE is applied independently to each RGB channel, which may alter
color balance but provides stronger per-channel enhancement.)code",
                    true)
    .InputLayout("HWC");

DALI_REGISTER_OPERATOR(Clahe, ClaheGPU, GPU);

}  // namespace dali
