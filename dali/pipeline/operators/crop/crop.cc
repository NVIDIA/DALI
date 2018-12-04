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

#include <tuple>

#include "dali/image/transform.h"
#include "dali/pipeline/operators/crop/kernel/coords.h"
#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"
#include "dali/pipeline/operators/crop/crop.h"
#include "dali/util/half.hpp"

namespace dali {

DALI_SCHEMA(Crop)
    .DocStr(R"code(Perform a random crop.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg(
        "crop_pos_x",
        R"code(Horizontal position of the crop in image coordinates (0.0 - 1.0))code",
        0.5f, true)
    .AddOptionalArg(
        "crop_pos_y",
        R"code(Vertical position of the crop in image coordinates (0.0 - 1.0))code",
        0.5f, true)
    .AddOptionalArg("image_type",
                    R"code(The color space of input and output image)code",
                    DALI_RGB, false)
    .AddOptionalArg(
        "crop",
        R"code(Size of the cropped image. If only a single value `c` is provided,
the resulting crop will be square with size `(c,c)`)code",
        std::vector<float>{0.f, 0.f})
    .EnforceInputLayout(DALI_NHWC);

std::pair<int, int> CropAttr::SetCropXY(const OpSpec &spec, const ArgumentWorkspace *ws,
    const Index dataIdx, int H, int W) {

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

template <>
Crop<CPUBackend>::Crop(const OpSpec &spec) : Operator<CPUBackend>(spec), CropAttr(spec) {
  Init(num_threads_);
}

template <>
void Crop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto *output = ws->Output<CPUBackend>(idx);

  DALITensorLayout out_layout = output_layout_ == DALI_SAME ? input.GetLayout() : output_layout_;
  output->SetLayout(out_layout);

  // Check if we use u8, RGB or Greyscale
  CheckParam(input, "CropCPUBackend");

  // Call AllocateAndRunKernel with detail::CropKernel<uint8_t, output_type_, out_layout>,
  // Note, that the last two template arguments are runtime values.
  if (out_layout == DALI_NHWC) {
    using nhwc_t = detail::dali_index_sequence<0, 1, 2>;
    if (output_type_ == DALI_FLOAT16) {
      using Kernel = detail::CropKernel<uint8_t, half_float::half, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_FLOAT) {
      using Kernel = detail::CropKernel<uint8_t, float, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_UINT8) {
      using Kernel = detail::CropKernel<uint8_t, uint8_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT16) {
      using Kernel = detail::CropKernel<uint8_t, int16_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT32) {
      using Kernel = detail::CropKernel<uint8_t, int32_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT64) {
      using Kernel = detail::CropKernel<uint8_t, int64_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else {
      DALI_FAIL("Unsupported output type.");
    }
  } else if (out_layout == DALI_NCHW) {
    using nchw_t = detail::dali_index_sequence<2, 0, 1>;
    if (output_type_ == DALI_FLOAT16) {
      using Kernel = detail::CropKernel<uint8_t, half_float::half, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_FLOAT) {
      using Kernel = detail::CropKernel<uint8_t, float, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_UINT8) {
      using Kernel = detail::CropKernel<uint8_t, uint8_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT16) {
      using Kernel = detail::CropKernel<uint8_t, int16_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT32) {
      using Kernel = detail::CropKernel<uint8_t, int32_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT64) {
      using Kernel = detail::CropKernel<uint8_t, int64_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else {
      DALI_FAIL("Unsupported output type.");
    }
  } else {
      DALI_FAIL("Unsupported output layout.");
  }
}

template <>
void Crop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<CPUBackend>(0);
    output_type_ = input.type().id();
  }

  SetupSharedSampleParams(ws, CheckShapes(ws), ws->thread_idx(),
                          ws->data_idx());
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<CPUBackend>, CPU);

}  // namespace dali
