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

#include <vector>

#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/equalize.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/color/equalize.h"

namespace dali {

namespace equalize {

class EqualizeGPU : public Equalize<GPUBackend> {
  using Kernel = kernels::equalize::EqualizeKernelGpu;

 public:
  explicit EqualizeGPU(const OpSpec &spec) : Equalize<GPUBackend>(spec) {
    kmgr_.Resize<Kernel>(1);
  }

 protected:
  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    // by the check in Equalize::SetupImpl
    assert(input.type() == type2id<uint8_t>::value);
    auto layout = input.GetLayout();
    // enforced by the layouts specified in operator schema
    assert(layout.size() == 2 || layout.size() == 3);
    output.SetLayout(layout);
    kernels::DynamicScratchpad scratchpad(AccessOrder(ws.stream()));
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    ctx.scratchpad = &scratchpad;
    auto out_view = view<uint8_t>(output);
    auto in_view = view<const uint8_t>(input);
    auto out_shape = GetFlattenedShape(out_view.shape);
    auto in_shape = GetFlattenedShape(in_view.shape);
    TensorListView<StorageGPU, uint8_t, 2> out_view_flat{out_view.data, out_shape};
    TensorListView<StorageGPU, const uint8_t, 2> in_view_flat{in_view.data, in_shape};
    kmgr_.Run<Kernel>(0, ctx, out_view_flat, in_view_flat);
  }

  template <int ndim>
  TensorListShape<2> GetFlattenedShape(TensorListShape<ndim> shape) {
    if (shape.sample_dim() == 3) {  // has_channels
      return collapse_dims<2>(shape, {{0, shape.sample_dim() - 1}});
    } else {
      int batch_size = shape.num_samples();
      TensorListShape<2> ret{batch_size};
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        ret.set_tensor_shape(sample_idx, TensorShape<2>(shape[sample_idx].num_elements(), 1));
      }
      return ret;
    }
  }

  kernels::KernelManager kmgr_;
};

}  // namespace equalize

DALI_REGISTER_OPERATOR(experimental__Equalize, equalize::EqualizeGPU, GPU);

}  // namespace dali
