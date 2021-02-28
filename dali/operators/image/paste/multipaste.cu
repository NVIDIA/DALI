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

void MultiPasteGPU::FillGPUInput(const workspace_t<GPUBackend> &ws) {
  auto &output = ws.template OutputRef<GPUBackend>(0);
  int batch_size = output.shape().num_samples();
  int spatial_ndim = output_size_[0].shape[0];

  samples.resize(batch_size);
  for (int i = 0; i < batch_size; i++) {
    auto &sample = samples[i];
    int n = in_idx_[i].num_elements();

    sample.sizes.resize(n);
    sample.out_anchors.resize(n);
    sample.in_anchors.resize(n);
    sample.in_idx.resize(n);

    sample.channels = 3;
    memcpy(sample.out_size.begin(), output_size_[i].data, sizeof(int) * spatial_ndim);
    for (int j = 0; j < n; j++) {
      int from_sample = in_idx_[i].data[j];
      auto in_anchor_view = GetInAnchors(i, j);
      auto out_anchor_view = GetOutAnchors(i, j);
      auto shape_view = GetShape(i, j, Coords(
          raw_input_size_mem_.data() + 2 * from_sample,
          dali::TensorShape<>(spatial_ndim)));
      memcpy(sample.sizes[j].begin(), shape_view.data, sizeof(int) * spatial_ndim);
      memcpy(sample.in_anchors[j].begin(), in_anchor_view.data, sizeof(int) * spatial_ndim);
      memcpy(sample.out_anchors[j].begin(), out_anchor_view.data, sizeof(int) * spatial_ndim);
      sample.in_idx[j] = from_sample;
    }
  }
}

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
                                                 make_span(samples), shapes);

            output_desc[0] = {shapes, TypeTable::GetTypeInfo(output_type_)};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
  return true;
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
  ctx.gpu.stream = ws.stream();
  kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, out_view, in_view, make_span(samples));
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
