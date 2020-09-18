// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include "dali/operators/generic/flip.h"
#include "dali/kernels/imgproc/flip_cpu.h"
#include "dali/kernels/kernel_params.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/ocv.h"
#include "dali/operators/generic/flip_util.h"

namespace dali {

DALI_SCHEMA(Flip)
    .DocStr(R"code(Flips the images in selected dimensions (horizontal, vertical,
and depthwise).)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("horizontal", R"code(Flip the horizontal  dimension.)code", 1, true)
    .AddOptionalArg("vertical", R"code(Flip the vertical dimension.)code", 0, true)
    .AddOptionalArg("depthwise", R"code(Flip the depthwise dimension.)code", 0, true)
    .InputLayout({"FDHWC", "FHWC", "DHWC", "HWC", "FCDHW", "FCHW", "CDHW", "CHW"})
    .AllowSequences()
    .SupportVolumetric();


template <>
Flip<CPUBackend>::Flip(const OpSpec &spec)
    : Operator<CPUBackend>(spec) {}

void RunFlip(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
             const TensorLayout &layout,
             bool horizontal, bool vertical, bool depthwise) {
  DALI_TYPE_SWITCH(input.type().id(), DType,
      auto output_ptr = output.mutable_data<DType>();
      auto input_ptr = input.data<DType>();
      auto kernel = kernels::FlipCPU<DType>();
      kernels::KernelContext ctx;
      auto shape_dims = TensorListShape<>{{input.shape()}};
      auto shape = TransformShapes(shape_dims, layout)[0];
      auto in_view = kernels::InTensorCPU<DType, flip_ndim>(input_ptr, shape);
      auto reqs = kernel.Setup(ctx, in_view);
      auto out_shape = reqs.output_shapes[0][0].to_static<flip_ndim>();
      auto out_view = kernels::OutTensorCPU<DType, flip_ndim>(output_ptr, out_shape);
      kernel.Run(ctx, out_view, in_view, depthwise, vertical, horizontal);
  )
}

template <>
void Flip<CPUBackend>::RunImpl(Workspace<CPUBackend> &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  auto layout = input.GetLayout();
  output.SetLayout(layout);
  output.set_type(input.type());
  output.ResizeLike(input);
  auto _horizontal = GetHorizontal(ws, ws.data_idx());
  auto _vertical = GetVertical(ws, ws.data_idx());
  auto _depthwise = GetDepthwise(ws, ws.data_idx());
  if (!_horizontal && !_vertical && !_depthwise) {
    output.Copy(input, nullptr);
  } else {
    RunFlip(output, input, layout, _horizontal, _vertical, _depthwise);
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<CPUBackend>, CPU);

}  // namespace dali
