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
#include <vector>
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
    .AllowSequences()
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
        std::vector<float>{0.f, 0.f});

template <>
Crop<CPUBackend>::Crop(const OpSpec &spec)
  : Operator<CPUBackend>(spec)
  , CropAttr(spec)
  , C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
  Init(num_threads_);
}

template <>
void Crop<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);

  DALITensorLayout in_layout = input.GetLayout();
  DALI_ENFORCE(in_layout == DALI_NHWC || in_layout == DALI_NFHWC);

  DALITensorLayout out_layout = output_layout_ == DALI_SAME ? in_layout : output_layout_;
  output.SetLayout(out_layout);

  // Check if we use u8, RGB or Greyscale
  CheckParam(input, "CropCPUBackend");

  // Call AllocateAndRunKernel with detail::CropKernel<uint8_t, output_type_, out_layout>,
  // Note, that the last two template arguments are runtime values.
  using nhwc_t = detail::dali_index_sequence<0, 1, 2>;
  using nchw_t = detail::dali_index_sequence<2, 0, 1>;

  if (out_layout == DALI_NHWC) {
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
    if (output_type_ == DALI_FLOAT16) {
      using Kernel = detail::CropKernel<uint8_t, float16_cpu, nchw_t>;
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
  } else if (out_layout == DALI_NFHWC) {
    if (output_type_ == DALI_FLOAT) {
      using Kernel = detail::SequenceCropKernel<uint8_t, float, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_UINT8) {
      using Kernel = detail::SequenceCropKernel<uint8_t, uint8_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT16) {
      using Kernel = detail::SequenceCropKernel<uint8_t, int16_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT32) {
      using Kernel = detail::SequenceCropKernel<uint8_t, int32_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT64) {
      using Kernel = detail::SequenceCropKernel<uint8_t, int64_t, nhwc_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else {
      DALI_FAIL("Unsupported output type.");
    }
  } else if (out_layout == DALI_NFCHW) {
    if (output_type_ == DALI_FLOAT) {
      using Kernel = detail::SequenceCropKernel<uint8_t, float, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_UINT8) {
      using Kernel = detail::SequenceCropKernel<uint8_t, uint8_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT16) {
      using Kernel = detail::SequenceCropKernel<uint8_t, int16_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT32) {
      using Kernel = detail::SequenceCropKernel<uint8_t, int32_t, nchw_t>;
      AllocateAndRunKernel<Kernel>(ws, idx);
    } else if (output_type_ == DALI_INT64) {
      using Kernel = detail::SequenceCropKernel<uint8_t, int64_t, nchw_t>;
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
