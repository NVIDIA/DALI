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
Normalization takes input image and produces output using formula:

  output = (input - mean) / std
)code")
        .NumInput(1)
        .NumOutput(1)
        .AllowMultipleInputSets()
        .AddOptionalArg("output_dtype",
                        R"code(Output data type.)code", DALI_FLOAT)
        .AddOptionalArg("output_layout",
                        R"code(Output tensor data layout)code", DALI_NCHW)
        .AddOptionalArg("pad_output",
                        R"code(Whether to pad the output to number of channels being multiple of 4.)code",
                        false)
        .AddOptionalArg("mirror",
                        R"code(Mask for horizontal flip.
- `0` - do not perform horizontal flip for this image
- `1` - perform horizontal flip for this image.
)code", 0, true)
        .AddArg("mean",
                R"code(Mean pixel values for image normalization.)code",
                DALI_FLOAT_VEC)
        .AddArg("std",
                R"code(Standard deviation values for image normalization.)code",
                DALI_FLOAT_VEC)
        .AddParent("Crop");

// Crop, mirror, mean sub, stddev div, NHWC->NCHW, Npp8u->fp32
template <typename Out>
void CropMirrorNormalizePermuteKernel(const int C, const int H, const int W,
                                      const bool pad, const int mirror_image,
                                      const float *mean, const float *inv_std,
                                      const uint8 *input_ptr, const int in_step,
                                      DALITensorLayout layout,
                                      Out *output_ptr) {
  const int pad_C = C;
  const int nStride = pad_C * H * W;

  const int a = mirror_image ? (W - 1) * C : 0;
  const int b = mirror_image ? -C : C;
  if (layout == DALI_NCHW) {
    // Coalesced writes
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const int in_idx = a + c + b * w + in_step * h;  // HWC
          const int out_idx = (c * H + h) * W + w;         // CHW

          output_ptr[out_idx] = static_cast<Out>(
              (input_ptr[in_idx] - mean[c]) * inv_std[c]);
        }
      }
    }

    // Pad to 4 channels with 0s
    if (pad) {
      const Out out = static_cast<Out>(0);
      for (int c = C; c < 4; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            const int out_idx = (c * H + h) * W + w;  // CHW
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
        const int in_idx = a + c + b * w + in_step * h;
        input = (static_cast<float>(input_ptr[in_idx]) - mean[c]) * inv_std[c];
      }

      const int out_idx = c + (w + h * W) * pad_C;
      output_ptr[out_idx] = static_cast<Out>(input);
    }
  }
}

template <>
template <typename Out>
void CropMirrorNormalize<CPUBackend>::RunHelper(SampleWorkspace *ws,
                                                const int idx) {
  const auto &input = ws->Input<CPUBackend>(0);
  auto output = ws->Output<CPUBackend>(idx);

  Out *output_ptr = output->template mutable_data<Out>();
  const int stride = input.dim(1) * C_;
  const int mirror_image = mirror_.template data<int>()[ws->data_idx()];

  CropMirrorNormalizePermuteKernel(
      C_, crop_h_, crop_w_, pad_, mirror_image, mean_.template data<float>(),
      inv_std_.template data<float>(), input.template data<uint8>(), stride,
      output_layout_, output_ptr);
}

template <>
void CropMirrorNormalize<CPUBackend>::SetupSharedSampleParams(
    SampleWorkspace *ws) {
  if (has_mirror_ && !ws->data_idx()) {
    const Tensor<CPUBackend> &mirror = ws->ArgumentInput("mirror");
    mirror_.Copy(mirror, 0);
  }

  if (output_layout_ == DALI_SAME) {
    output_layout_ = ws->Input<CPUBackend>(0).GetLayout();
  }

  if (output_type_ == DALI_NO_TYPE) {
    output_type_ = ws->Input<CPUBackend>(0).type().id();
  }
}

template <>
void CropMirrorNormalize<CPUBackend>::DataDependentSetup(SampleWorkspace *ws,
                                                         const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);

  DALITensorLayout outLayout;
  output->Resize(GetOutShape(input.GetLayout(), &outLayout));
  output->SetLayout(outLayout);
}

template <>
void CropMirrorNormalize<CPUBackend>::RunImpl(SampleWorkspace *ws,
                                              const int idx) {
  DataDependentSetup(ws, idx);

  if (output_type_ == DALI_FLOAT16) {
    RunHelper<half_float::half>(ws, idx);
  } else if (output_type_ == DALI_FLOAT) {
    RunHelper<float>(ws, idx);
  } else if (output_type_ == DALI_UINT8) {
    RunHelper<unsigned char>(ws, idx);
  } else if (output_type_ == DALI_INT16) {
    RunHelper<int16>(ws, idx);
  } else if (output_type_ == DALI_INT32) {
    RunHelper<int>(ws, idx);
  } else if (output_type_ == DALI_INT64) {
    RunHelper<int64>(ws, idx);
  } else {
    DALI_FAIL("Unsupported output type.");
  }
}

// Register operator
DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<CPUBackend>,
                       CPU);

}  // namespace dali
