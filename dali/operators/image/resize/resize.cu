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

#include <cuda_runtime_api.h>

#include <utility>
#include <vector>

#include "dali/operators/image/resize/resize.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template<>
Resize<GPUBackend>::Resize(const OpSpec &spec)
    : Operator<GPUBackend>(spec)
    , ResizeAttr(spec)
    , ResizeBase(spec) {
  save_attrs_ = spec_.HasArgument("save_attrs");
  outputs_per_idx_ = save_attrs_ ? 2 : 1;

  ResizeAttr::SetBatchSize(batch_size_);
  InitializeGPU(batch_size_, spec_.GetArgument<int>("minibatch_size"));
  resample_params_.resize(batch_size_);
}

template<>
void Resize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);

  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");
  if (!input.GetLayout().empty()) {
    DALI_ENFORCE(ImageLayoutInfo::IsChannelLast(input.GetLayout()),
                 "Resize expects interleaved channel layout (aka channel-last or NHWC)");
  }

  for (int i = 0; i < batch_size_; ++i) {
    auto input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    per_sample_meta_[i] = GetTransformMeta(spec_, input_shape, &ws, i, ResizeInfoNeeded());
    resample_params_[i] = GetResamplingParams(per_sample_meta_[i]);
  }
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  RunGPU(output, input, ws.stream());
  output.SetLayout(InputLayout(ws, 0));

  // Setup and output the resize attributes if necessary
  if (save_attrs_) {
    TensorList<CPUBackend> attr_output_cpu;
    TensorListShape<> resize_shape(input.ntensor(), 1);

    for (size_t i = 0; i < input.ntensor(); ++i) {
      resize_shape.set_tensor_shape(i, {2});
    }

    attr_output_cpu.Resize(resize_shape);
    auto in_shape = input.shape().to_static<3>();

    for (int i = 0; i < in_shape.num_samples(); ++i) {
      int *t = attr_output_cpu.mutable_tensor<int>(i);
      auto sample_shape = in_shape.tensor_shape_span(i);
      t[0] = sample_shape[0];
      t[1] = sample_shape[1];
    }
    ws.Output<GPUBackend>(1).Copy(attr_output_cpu, ws.stream());
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
