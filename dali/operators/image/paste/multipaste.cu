// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <int n, typename T, typename Source>
inline void to_vec(vec<n, T> &out, Source &&src) {
  for (int i = 0; i < n; i++)
    out[i] = src[i];
}

void MultiPasteGPU::InitSamples(const TensorListShape<> &out_shape) {
  assert(spatial_ndim_ == 2);
  int batch_size = out_shape.num_samples();
  samples_.resize(batch_size);
  for (int i = 0; i < batch_size; i++) {
    auto &sample = samples_[i];
    int n = in_idx_[i].num_elements();

    sample.inputs.resize(n);

    sample.channels = 3;

    to_vec(sample.out_size, out_shape[i]);
    for (int j = 0; j < n; j++) {
      int from_sample = in_idx_[i].data[j];
      auto in_anchor_view = GetInAnchors(i, j);
      auto out_anchor_view = GetOutAnchors(i, j);
      auto in_shape_view = GetInputShape(from_sample);
      auto region_shape = GetShape(i, j, in_shape_view, in_anchor_view);
      to_vec(sample.inputs[j].size,       region_shape);
      to_vec(sample.inputs[j].in_anchor,  in_anchor_view.data);
      to_vec(sample.inputs[j].out_anchor, out_anchor_view.data);
      sample.inputs[j].in_idx = from_sample;
    }
  }
}

template<typename OutputType, typename InputType>
void MultiPasteGPU::SetupTyped(const workspace_t<GPUBackend> &ws,
                               const TensorListShape<> &out_shape) {
  const auto &images = ws.template Input<GPUBackend>(0);
  const auto &in = view<const InputType, 3>(images);
  using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernel_manager_.Initialize<Kernel>();
  InitSamples(out_shape);
  const auto &reqs = kernel_manager_.Setup<Kernel>(
        0, ctx, make_span(samples_), out_shape.to_static<3>(), in.shape);
}

template<typename OutputType, typename InputType>
void MultiPasteGPU::RunTyped(workspace_t<GPUBackend> &ws) {
  const auto &images = ws.template Input<GPUBackend>(0);
  auto &output = ws.template Output<GPUBackend>(0);

  output.SetLayout(images.GetLayout());
  auto out_shape = output.shape();
  using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
  auto in_view = view<const InputType, 3>(images);
  auto out_view = view<OutputType, 3>(output);

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, out_view, in_view);
}

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteGPU, GPU)

}  // namespace dali
