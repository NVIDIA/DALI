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


#include <vector>
#include "dali/util/npp.h"
#include "dali/operators/image/color/color_twist.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_gpu.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <>
bool ColorTwistBase<GPUBackend>::CanInferOutputs() const  {
  return true;
}

template <>
bool ColorTwistBase<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                           const DeviceWorkspace &ws) {
  const auto &input = ws.template InputRef<GPUBackend>(0);
  output_desc.resize(1);
  using Kernel = kernels::LinearTransformationGpu<uint8_t, uint8_t, 3, 3, 2>;
  kernel_manager_.Initialize<Kernel>();
  kernel_manager_.Resize<Kernel>(1, 1);
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  const auto tvin = view<const uint8_t, 3>(input);
  mats.resize(input.ntensor());
  vecs.resize(input.ntensor());
  for (size_t i = 0; i < input.ntensor(); ++i) {
    ColorAugment::mat_t m(1.);
    for (size_t j = 0; j < augments_.size(); ++j) {
      augments_[j]->Prepare(i, spec_, &ws);
      (*augments_[j])(m);
    }
    mats[i] = sub<3, 3>(m);
    vecs[i] = sub<3>(m.col(3));
  }
  const auto &reqs = kernel_manager_.Setup<Kernel>(0, ctx, tvin,
                                                   make_cspan(mats), make_cspan(vecs));
  auto &shapes = reqs.output_shapes[0];
  output_desc[0] = {shapes, TypeTable::GetTypeInfo(TypeTable::GetTypeID<uint8_t>())};
  return true;
}

template <>
void ColorTwistBase<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8_t>(input.type()),
      "Color augmentations accept only uint8 tensors");
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(InputLayout(ws, 0));
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto tvin = view<const uint8_t, 3>(input);
  auto tvout = view<uint8_t, 3>(output);
  using Kernel = kernels::LinearTransformationGpu<uint8_t, uint8_t, 3, 3, 2>;
  kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, tvout, tvin,
                              make_cspan(mats), make_cspan(vecs));
}

DALI_REGISTER_OPERATOR(Brightness, BrightnessAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Contrast, ContrastAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Hue, HueAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Saturation, SaturationAdjust<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(ColorTwist, ColorTwistAdjust<GPUBackend>, GPU);

}  // namespace dali
