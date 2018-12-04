// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"
#include "dali/pipeline/operators/crop/sequence_crop.h"

namespace dali {

DALI_SCHEMA(SequenceCrop)
    .DocStr(R"code(Perform a random crop on a sequecne.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddParent("Crop")
    .EnforceInputLayout(DALI_NHWC);

void SequenceCrop::RunImpl(Workspace<CPUBackend> *ws, const int idx) {
  // Ensure the layout
  auto *output = ws->Output<CPUBackend>(idx);
  DALITensorLayout out_layout =
      output_layout_ == DALI_SAME ? ws->Input<CPUBackend>(idx).GetLayout() : output_layout_;
  output->SetLayout(out_layout);

  // Check if we use u8, RGB or Greyscale
  // CheckParam(input, "CropCPUBackend");

  // TODO(klecki): simplification - do not handle float16

  // Call AllocateAndRunKernel with detail::SequenceCrop<uint8_t, output_type_, out_layout>,
  // Note, that the last two template arguments are runtime values.
  if (out_layout == DALI_NHWC) {
    using nhwc_t = detail::dali_index_sequence<0, 1, 2>;
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
  } else if (out_layout == DALI_NCHW) {
    using nchw_t = detail::dali_index_sequence<2, 0, 1>;
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

const std::vector<Index> SequenceCrop::CheckShapes(const SampleWorkspace *ws) {
  const auto &input = ws->Input<CPUBackend>(0);

  DALI_ENFORCE(input.ndim() == 4, "Operator expects 4-dimensional sequence image input.");

  // enforce that all shapes match
  for (int i = 1; i < ws->NumInput(); ++i) {
    const auto &other_input = ws->Input<CPUBackend>(i);
    DALI_ENFORCE(other_input.ndim() == 4, "Operator expects 4-dimensional sequence image input.");
    for (int j = 1; j < 4; j++) {
      DALI_ENFORCE(input.dim(j) == other_input.dim(j));
    }
  }

  std::vector<Index> frame_shape;
  for (int i = 1; i < 4; i++) {
    frame_shape.push_back(input.dim(i));
  }

  return frame_shape;
}

DALI_REGISTER_OPERATOR(SequenceCrop, SequenceCrop, CPU);

}  // namespace dali
