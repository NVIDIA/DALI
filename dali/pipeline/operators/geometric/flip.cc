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
#include "dali/pipeline/operators/geometric/flip.h"
#include "dali/kernels/imgproc/flip_cpu.h"
#include "dali/kernels/kernel_params.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/ocv.h"
#include "dali/pipeline/operators/geometric/flip_util.h"

namespace dali {

DALI_SCHEMA(Flip)
    .DocStr(R"code(Flip the image over the horizontal and/or vertical axes.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("horizontal", R"code(Perform a horizontal flip.)code", 1, true)
    .AddOptionalArg("vertical", R"code(Perform a vertical flip.)code", 0, true);


template <>
Flip<CPUBackend>::Flip(const OpSpec &spec)
    : Operator<CPUBackend>(spec) {}

void RunFlip(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
             bool horizontal, bool vertical) {
  DALI_TYPE_SWITCH(input.type().id(), DType,
      auto output_ptr = output.mutable_data<DType>();
      auto input_ptr = input.data<DType>();
      auto kernel = kernels::FlipCPU<DType>();
      kernels::KernelContext ctx;
      auto shape = TransformShapes({input.shape()}, input.GetLayout() == DALI_NHWC)[0];
      auto in_view = kernels::InTensorCPU<DType, 4>(input_ptr, shape);
      auto reqs = kernel.Setup(ctx, in_view);
      auto out_shape = reqs.output_shapes[0][0].to_static<4>();
      auto out_view = kernels::OutTensorCPU<DType, 4>(output_ptr, out_shape);
      kernel.Run(ctx, out_view, in_view, false, vertical, horizontal);
  )
}

template <>
void Flip<CPUBackend>::RunImpl(Workspace<CPUBackend> *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  DALI_ENFORCE(input.ndim() == 3);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  auto _horizontal = GetHorizontal(ws, ws->data_idx());
  auto _vertical = GetVertical(ws, ws->data_idx());
  if (!_horizontal && !_vertical) {
    output.Copy(input, nullptr);
  } else {
    RunFlip(output, input, _horizontal, _vertical);
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<CPUBackend>, CPU);

}  // namespace dali
