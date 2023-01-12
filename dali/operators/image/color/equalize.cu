// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <npp.h>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/equalized_lut.cuh"
#include "dali/kernels/imgproc/color_manipulation/equalize/hist.cuh"
#include "dali/kernels/imgproc/color_manipulation/equalize/lut_lookup.cuh"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/color/equalize.h"

namespace dali {

namespace equalize {

class EqualizeGPU : public Equalize<GPUBackend> {
 public:
  explicit EqualizeGPU(const OpSpec &spec) : Equalize<GPUBackend>(spec) {
    histogram_dev_.set_type<int32_t>();
    lut_dev_.set_type<uint8_t>();
  }

 protected:
  void RunImpl(Workspace &ws) override {
    auto input_type = ws.GetInputDataType(0);
    auto layout = GetInputLayout(ws, 0);
    // this could be an assert as it should be set by the executor
    DALI_ENFORCE(layout.size(), "The input of the equalize operator cannot have an empty layout");
    bool has_channels = layout[layout.size() - 1];
    TYPE_SWITCH(input_type, type2id, In, EQUALIZE_SUPPORTED_TYPES, (
      BOOL_SWITCH(has_channels, HasChannels, (
        RunImplTyped<In, HasChannels>(ws);
      ));
    ), DALI_FAIL(make_string("Unsupported input type: ", input_type, ".")));  // NOLINT
  }

  template <typename In, bool has_channels>
  void RunImplTyped(Workspace &ws) {
    static_assert(std::is_same<In, uint8>::value);
    DALI_ENFORCE(has_channels);
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());
    histogram_dev_.Resize(uniform_list_shape<>(input.num_samples(), {3, 256}));
    lut_dev_.Resize(uniform_list_shape<>(input.num_samples(), {3, 256}));
    kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    ctx.scratchpad = &scratchpad;
    auto out_view = view<uint8_t>(output);
    auto in_view = view<const uint8_t>(input);
    auto out_shape = collapse_dims<2>(out_view.shape, {{0, out_view.sample_dim() - 1}});
    auto in_shape = collapse_dims<2>(in_view.shape, {{0, in_view.sample_dim() - 1}});
    TensorListView<StorageGPU, uint8_t, 2> out_view_flat{out_view.data, out_shape};
    TensorListView<StorageGPU, const uint8_t, 2> in_view_flat{in_view.data, in_shape};
    auto hist_view = view<int32_t, 2>(histogram_dev_);
    auto lut_view = view<uint8_t, 2>(lut_dev_);
    hist_kernel_.Run(ctx, hist_view, in_view_flat);
    lut_kernel_.Run(ctx, lut_view, hist_view);
    lookup_kernel_.Run(ctx, out_view_flat, in_view_flat, lut_view);
  }

  TensorList<GPUBackend> histogram_dev_;
  TensorList<GPUBackend> lut_dev_;
  kernels::HistogramKernelGpu hist_kernel_;
  kernels::EqualizedLutKernelGpu lut_kernel_;
  kernels::LutLookupKernelGpu lookup_kernel_;
};

}  // namespace equalize

DALI_REGISTER_OPERATOR(experimental__Equalize, equalize::EqualizeGPU, GPU);

}  // namespace dali
