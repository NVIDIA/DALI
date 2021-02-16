// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/operators/image/paste/multipaste.h"
#include "dali/kernels/imgproc/paste/paste_gpu.h"
#include "dali/core/tensor_view.h"

namespace dali {

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteGPU, GPU)

bool MultiPasteGPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                              const workspace_t<GPUBackend> &ws) {
  AcquireArguments(spec_, ws);
  FillGPUInput(ws);

  const auto &images = ws.template InputRef<GPUBackend>(0);
  const auto &output = ws.template OutputRef<GPUBackend>(0);
  output_desc.resize(1);

  TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
            using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
            kernel_manager_.Initialize<Kernel>();

            TensorListShape<> sh = images.shape();
            TensorListShape<3> shapes(sh.num_samples(), sh.sample_dim());
            for (int i = 0; i < sh.num_samples(); i++) {
                const TensorShape<3> &out_sh = { output_size_[i].data[0],
                                                output_size_[i].data[1], sh[i][2] };
                shapes.set_tensor_shape(i, out_sh);
            }

            kernels::KernelContext ctx;
            ctx.gpu.stream = ws.stream();
            const auto tvin = view<const InputType, 3>(images);
            const auto &reqs = kernel_manager_.Setup<Kernel>(0, ctx, tvin,
                                                             samples, grid_cells, shapes);

            output_desc[0] = {shapes, TypeTable::GetTypeInfo(output_type_)};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
  return true;
}


void MultiPasteGPU::FillGPUInput(const workspace_t<GPUBackend> &ws) {
  auto &output = ws.template Output<GPUBackend>(0);
  auto out_shape = output.shape();
  int batch_size = out_shape.num_samples();
  for (int i = 0; i < batch_size; i++) {
      set<int> x_borders;
      x_borders.insert(0);
      x_borders.insert(out_shape[i].data[1]);
      int n_paste = in_idx_[i].shape[0];
      for (int j = 0; j < n_paste; j++) {

      }


  }
  DALI_FAIL("As for now, MultiPasteGUP does not support intersecting pastes");
}

template<typename InputType, typename OutputType>
void MultiPasteGPU::RunImplExplicitlyTyped(workspace_t<GPUBackend> &ws) {
  const auto &images = ws.template Input<GPUBackend>(0);
  auto &output = ws.template Output<GPUBackend>(0);

  output.SetLayout(images.GetLayout());
  auto out_shape = output.shape();
  using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
  auto in_view = view<const InputType, 3>(images);
  auto out_view = view<OutputType, 3>(output);

  kernels::KernelContext ctx;
  kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, out_view, in_view, samples, grid_cells);
}


void MultiPasteGPU::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto input_type_id = ws.template InputRef<GPUBackend>(0).type().id();
  TYPE_SWITCH(input_type_id, type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
              RunImplExplicitlyTyped<InputType, OutputType>(ws);
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input_type_id)))  // NOLINT
}

}  // namespace dali
