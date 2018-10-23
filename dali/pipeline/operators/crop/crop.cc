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

#include "dali/pipeline/operators/crop/crop.h"
#include "dali/image/transform.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(Crop)
    .DocStr(R"code(Perform a random crop.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("crop_pos_x",
                    R"code(Horizontal position of the crop in image coordinates (0.0 - 1.0))code",
                    0.5f, true)
    .AddOptionalArg("crop_pos_y",
                    R"code(Vertical position of the crop in image coordinates (0.0 - 1.0))code",
                    0.5f, true)
    .AddOptionalArg("image_type",
                    R"code(The color space of input and output image)code", DALI_RGB, false)
    .AddOptionalArg(
        "crop",
        R"code(Size of the cropped image. If only a single value `c` is provided,
        the resulting crop will be square with size `(c,c)`)code",
        std::vector<float>{0.f, 0.f})
    .EnforceInputLayout(DALI_NHWC);

std::pair<int, int> CropAttr::SetCropXY(const OpSpec &spec, const ArgumentWorkspace *ws,
    const Index dataIdx, int H, int W) {
    DALI_ENFORCE(H >= crop_height_[dataIdx]);
    DALI_ENFORCE(W >= crop_width_[dataIdx]);

    auto crop_x_norm = spec.GetArgument<float>("crop_pos_x", ws, dataIdx);
    auto crop_y_norm = spec.GetArgument<float>("crop_pos_y", ws, dataIdx);

    DALI_ENFORCE(crop_y_norm >= 0.f && crop_y_norm <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_x_norm >= 0.f && crop_x_norm <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");

    const int crop_y = crop_y_norm * (H - crop_height_[dataIdx]);
    const int crop_x = crop_x_norm * (W - crop_width_[dataIdx]);

    return std::make_pair(crop_y, crop_x);
}

const vector<Index> CropAttr::CheckShapes(const SampleWorkspace *ws) {
  const auto &input = ws->Input<CPUBackend>(0);

  // enforce that all shapes match
  for (int i = 1; i < ws->NumInput(); ++i) {
    DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
  }

  DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");

  return input.shape();
}


template<>
Crop<CPUBackend>::Crop(const OpSpec &spec) : Operator<CPUBackend>(spec), CropAttr(spec) {
  Init(num_threads_);
}

template<typename Out>
void CropKernel(
  const int C,
  const int H,
  const int W,
  const unsigned char *input_ptr,
  const int in_stride,
  DALITensorLayout output_layout,
  Out *output_ptr) {
  if (output_layout == DALI_NCHW) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // From HWC
          const int in_idx = h * in_stride + w * C + c;
          // To CHW
          const int out_idx = (c * H + h) * W + w;
          output_ptr[out_idx] = static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  } else {  // Layout == DALI_NHWC
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // From HWC
          const int in_idx = h * in_stride + w * C + c;
          // To HWC
          const int out_idx = (h * W + w) * C + c;
          output_ptr[out_idx] = static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  }
}

template<>
template<typename Out>
void Crop<CPUBackend>::RunHelper(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);

  const int threadIdx = ws->thread_idx();
  const int W = per_sample_dimensions_[threadIdx].second;

  const int crop_y = per_sample_crop_[threadIdx].first;
  const int crop_x = per_sample_crop_[threadIdx].second;

  const int dataIdx = ws->data_idx();
  CropKernel<Out>(C_, crop_height_[dataIdx], crop_width_[dataIdx],
                              input.template data<uint8>() + (crop_y * W + crop_x) * C_,
                              W * C_, output_layout_,
                              output->template mutable_data<Out>());
}

template<>
void Crop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto *output = ws->Output<CPUBackend>(idx);

  DALITensorLayout outLayout;
  output->Resize(GetOutShape(input.GetLayout(), &outLayout, ws->data_idx()));
  output->SetLayout(outLayout);

  // Check if we use u8, RGB or Greyscale
  CheckParam(input, "CropCPUBackend");
  if (output_type_ == DALI_FLOAT16)
    RunHelper<half_float::half>(ws, idx);
  else
    CallRunHelper(ws, idx);
}

template<>
void Crop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<CPUBackend>(0);
    output_type_ = input.type().id();
  }

  SetupSharedSampleParams(ws, CheckShapes(ws), ws->thread_idx(), ws->data_idx());
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<CPUBackend>, CPU);

}  // namespace dali
