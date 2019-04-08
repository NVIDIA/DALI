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

#include "dali/pipeline/operators/resize/resize.h"
#include "dali/kernels/static_switch.h"
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
void Resize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace* ws) {
  auto &input = ws->Input<GPUBackend>(0);

  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");
  if (input.GetLayout() != DALI_UNKNOWN) {
    DALI_ENFORCE(input.GetLayout() == DALI_NHWC,
                 "Resize expects interleaved channel layout (NHWC)");
  }

  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    per_sample_meta_[i] = GetTransformMeta(spec_, input_shape, ws, i, ResizeInfoNeeded());
    resample_params_[i] = GetResamplingParams(per_sample_meta_[i]);
  }
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(outputs_per_idx_ * idx);

  RunGPU(output, input, ws->stream());

  // Setup and output the resize attributes if necessary
  if (save_attrs_) {
    TensorList<CPUBackend> attr_output_cpu;
    vector<Dims> resize_shape(input.ntensor());

    for (size_t i = 0; i < input.ntensor(); ++i) {
      resize_shape[i] = Dims{2};
    }

    attr_output_cpu.Resize(resize_shape);
    auto in_shape = list_shape<3>(input);

    for (int i = 0; i < in_shape.num_samples(); ++i) {
      int *t = attr_output_cpu.mutable_tensor<int>(i);
      auto sample_shape = in_shape.tensor_shape_span(i);
      t[0] = sample_shape[0];
      t[1] = sample_shape[1];
    }
    ws->Output<GPUBackend>(outputs_per_idx_ * idx + 1).Copy(attr_output_cpu, ws->stream());
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
