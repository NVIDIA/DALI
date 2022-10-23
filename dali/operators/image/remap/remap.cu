// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/remap/remap.h"
#include "dali/operators/image/remap/remap.cuh"

namespace dali::remap {

namespace {

}  // namespace

class RemapGpu : public Remap<GPUBackend> {
  using B = GPUBackend;
 public:
  explicit RemapGpu(const OpSpec &spec) : Remap<B>(spec) {}


  void RunImpl(Workspace &ws) override {
    const auto &input = ws.template Input<B>(0);
    TYPE_SWITCH(input.type(), type2id, InputType, REMAP_SUPPORTED_TYPES, (//TODO indent
            {
              RunImplTyped<InputType>(ws);
            }
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
  }


 private:
  template<typename InputType>
  void RunImplTyped(Workspace &ws) {
    const auto &input = ws.template Input<B>(0);
    const auto &mapx = ws.template Input<B>(1);
    const auto &mapy = ws.template Input<B>(2);
    auto &output = ws.template Output<B>(0);
    using KernelBackend = StorageGPU;
    //    using KernelBackend = backend_to_storage_device<Backend>;
//    std::unique_ptr<kernels::remap::RemapKernel<KernelBackend, InputT>> kernel;
//    kernel = std::make_unique<kernels::remap::NppRemapKernel<KernelBackend, InputT>>();
    kernels::remap::NppRemapKernel<KernelBackend, InputType> kernel;
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();

    TensorList<B> mapx_shifted, mapy_shifted;
    if (shift_pixels_) {
      mapx_shifted.Copy(mapx);
      mapy_shifted.Copy(mapy);
      detail::ShiftPixelOrigin(view<float>(mapx_shifted), shift_value_, ws.stream());
      detail::ShiftPixelOrigin(view<float>(mapy_shifted), shift_value_, ws.stream());
    }
    cout<<"ROI\n"<<rois_<<endl;
    cout<<"MAPS\n"<<mapx.shape()<<endl<<mapy.shape()<<endl;
    kernel.Run(ctx, view<InputType, 3>(output), view<const InputType, 3>(input),
               view<const float, 2>(shift_pixels_ ? mapx_shifted : mapx),
               view<const float, 2>(shift_pixels_ ? mapy_shifted : mapy),
               make_span(rois_), {}, make_span(interps_), {/* Border (currently unsupported) */});
  }
};

DALI_REGISTER_OPERATOR(Remap, RemapGpu, GPU);


}  // namespace dali::remap