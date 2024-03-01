// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void MultiPasteGPU::InitSamples(const Workspace &ws, const TensorListShape<> &out_shape) {
  assert(spatial_ndim_ == 2);
  assert(out_shape.sample_dim() == 3);
  int batch_size = out_shape.num_samples();
  samples_.resize(batch_size);
  bool hasInIdx = in_idx_.HasExplicitValue();
  for (int i = 0; i < batch_size; i++) {
    auto &sample = samples_[i];
    int n = GetPasteCount(ws, i);

    sample.inputs.resize(n);
    sample.channels = out_shape[i][spatial_ndim_];

    to_vec(sample.out_size, out_shape[i]);
    for (int j = 0; j < n; j++) {
      int batch_idx = hasInIdx ? 0 : j;
      int in_idx = hasInIdx ? in_idx_[i].data[j] : i;
      const auto &in_anchor = in_anchors_data_[i][j];
      const auto &out_anchor = out_anchors_data_[i][j];
      const auto &region_shape = region_shapes_data_[i][j];
      to_vec(sample.inputs[j].size,       region_shape);
      to_vec(sample.inputs[j].in_anchor,  in_anchor);
      to_vec(sample.inputs[j].out_anchor, out_anchor);
      sample.inputs[j].batch_idx = batch_idx;
      sample.inputs[j].in_idx = in_idx;
    }
  }
}

template<typename OutputType, typename InputType>
void MultiPasteGPU::SetupTyped(const Workspace &ws,
                               const TensorListShape<> &out_shape) {
  using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernel_manager_.Initialize<Kernel>();
  InitSamples(ws, out_shape);
  std::vector<TensorListShape<3>> in_shapes;
  in_shapes.reserve(ws.NumInput());
  for (int i = 0; i < ws.NumInput(); i++) {
    in_shapes.push_back(ws.Input<GPUBackend>(0).shape().to_static<3>());
  }
  const auto &reqs = kernel_manager_.Setup<Kernel>(
        0, ctx, make_span(samples_), out_shape.to_static<3>(), make_span(in_shapes));
}

template<typename OutputType, typename InputType>
void MultiPasteGPU::RunTyped(Workspace &ws) {
  auto &output = ws.Output<GPUBackend>(0);

  output.SetLayout(ws.Input<GPUBackend>(0).GetLayout());
  auto out_shape = output.shape();
  using Kernel = kernels::PasteGPU<OutputType, InputType, 3>;

  std::vector<TensorListView<StorageGPU, const InputType, 3>> in_views;
  in_views.reserve(ws.NumInput());
  for (int i = 0; i < ws.NumInput(); i++) {
    in_views.push_back(view<const InputType, 3>(ws.Input<GPUBackend>(i)));
  }
  auto out_view = view<OutputType, 3>(output);

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernel_manager_.Run<Kernel>(0, ctx, out_view, make_span(in_views));
}

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteGPU, GPU)

}  // namespace dali
