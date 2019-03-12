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
    .DocStr(R"code(Crops image with a given window dimensions and window position (upper left corner))code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AllowSequences()
    .AddOptionalArg(
        "image_type",
        R"code(The color space of input and output image)code",
        DALI_RGB, false)
    .AddParent("CropAttr");

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

  DALI_TYPE_SWITCH_WITH_FP16_CPU(output_type_, OType,
    switch (out_layout) {
      case DALI_NHWC:
      {
        using Kernel = detail::CropKernel<uint8_t, OType, nhwc_t>;
        AllocateAndRunKernel<Kernel>(ws, idx);
      }
      break;

      case DALI_NCHW:
      {
        using Kernel = detail::CropKernel<uint8_t, OType, nchw_t>;
        AllocateAndRunKernel<Kernel>(ws, idx);
      }
      break;

      case DALI_NFHWC:
      {
        using Kernel = detail::SequenceCropKernel<uint8_t, OType, nhwc_t>;
        AllocateAndRunKernel<Kernel>(ws, idx);
      }
      break;

      case DALI_NFCHW:
      {
        using Kernel = detail::SequenceCropKernel<uint8_t, OType, nchw_t>;
        AllocateAndRunKernel<Kernel>(ws, idx);
      }
      break;

      default:
        DALI_FAIL("output layout not supported");
    }
  ); // NOLINT
}

template <>
void Crop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  if (output_type_ == DALI_NO_TYPE) {
    const auto &input = ws->Input<CPUBackend>(0);
    output_type_ = input.type().id();
  }
  CropAttr::ProcessArguments(ws);
  SetupSharedSampleParams(ws, CheckShapes(ws), ws->thread_idx(),
                          ws->data_idx());
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<CPUBackend>, CPU);

}  // namespace dali
