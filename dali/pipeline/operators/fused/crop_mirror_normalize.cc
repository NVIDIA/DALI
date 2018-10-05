// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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


#include "dali/pipeline/operators/fused/crop_mirror_normalize.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(CropMirrorNormalize)
  .DocStr(R"code(Perform fused cropping, normalization, format conversion
(NHWC to NCHW) if desired, and type casting.
Normalization takes input image and produces output using formula

..

   output = (input - mean) / std
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("pad_output",
      R"code(Whether to pad the output to number of channels being multiple of 4.)code",
      false)
  .AddOptionalArg("mirror",
      R"code(Mask for horizontal flip.

- `0` - do not perform horizontal flip for this image
- `1` - perform horizontal flip for this image.
)code", 0, true)

  .AddParent("NormalizeBase")
  .AddParent("CropCastPermute");


// Crop, mirror, mean sub, stddev div, NHWC->NCHW, Npp8u->fp32
template<typename Out>
void CropMirrorNormalizePermuteKernel(
  const int C,
  const int H,
  const int W,
  const bool pad,
  const int mirror_image,
  const float* mean,
  const float* inv_std,
  const uint8* input_ptr,
  const int in_step,
  DALITensorLayout layout,
  Out* output_ptr) {
  const int pad_C = pad ? 4 : C;
  const int nStride = pad_C*H*W;

  const int a = mirror_image? (W - 1) * C : 0;
  const int b = mirror_image? -C : C;
  if (layout == DALI_NCHW) {
    // Coalesced writes
    for (int c=0; c < C; ++c) {
      for (int h=0; h < H; ++h) {
        for (int w=0; w < W; ++w) {
          const int in_idx = a + c + b * w + in_step*h;   // HWC
          const int out_idx = (c*H + h)*W + w;            // CHW

          output_ptr[out_idx] = StaticCastGpu<Out>(
            (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c]);
        }
      }
    }

    // Pad to 4 channels with 0s
    if (pad) {
      const Out out = StaticCastGpu<Out>(0);
      for (int c=C; c < 4; ++c) {
        for (int h=0; h < H; ++h) {
          for (int w=0; w < W; ++w) {
            const int out_idx = (c*H + h)*W + w;  // CHW
            output_ptr[out_idx] = out;
          }
        }
      }
    }
  } else {
    for (int tid = 0; tid < nStride; ++tid) {
      const int c = tid % pad_C;
      const int w = (tid / pad_C) % W;
      const int h = tid / (pad_C * W);

      float input;
      if (pad && c == 3) {
        input = 0;
      } else {
        const int in_idx =  a + c + b * w + in_step * h;
        input = (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c];
      }

      const int out_idx = c + (w + h*W) * pad_C;
      output_ptr[out_idx] = StaticCastGpu<Out>(input);
    }
  }
}

template<>
template<typename Out>
void CropMirrorNormalize<CPUBackend>::RunHelper(SampleWorkspace *ws, const int idx) {
  const unsigned char *input_ptr;
  int stride;
  Out *output_ptr;
  const int mirror_image = mirror_.template data<int>()[ws->data_idx()];
  PrepareCropParam<Out>(ws, idx, &input_ptr, &stride, &output_ptr);
  CropMirrorNormalizePermuteKernel(C_, crop_[0], crop_[1],
                                   pad_, mirror_image,
                                   mean_.template data<float>(),
                                   inv_std_.template data<float>(),
                                   input_ptr,
                                   stride, output_layout_,
                                   output_ptr);
}


template<>
void CropMirrorNormalize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  Crop<CPUBackend>::SetupSharedSampleParams(ws);
  if (has_mirror_ && !ws->data_idx()) {
    const Tensor<CPUBackend> &mirror = ws->ArgumentInput("mirror");
    mirror_.Copy(mirror, 0);
  }
}

template<>
void CropMirrorNormalize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  RUN_IMPL_CPU(ws, idx);
}

// Register operator
DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<CPUBackend>, CPU);

}  // namespace dali
