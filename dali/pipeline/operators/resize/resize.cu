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

#include "dali/util/npp.h"
#include "dali/pipeline/operators/resize/resize.h"
#include "dali/kernels/static_switch.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template<>
Resize<GPUBackend>::Resize(const OpSpec &spec) : Operator<GPUBackend>(spec), ResizeAttr(spec) {
  save_attrs_ = spec_.HasArgument("save_attrs");
  outputs_per_idx_ = save_attrs_ ? 2 : 1;

  // Resize per-image data
  input_ptrs_.resize(batch_size_);
  output_ptrs_.resize(batch_size_);
  out_sizes.resize(batch_size_);
  in_sizes.resize(batch_size_);

  // Per set-of-sample TransformMeta
  per_sample_meta_.resize(batch_size_);
}

template<>
void Resize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace* ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  resample_params_.resize(batch_size_);

  DALIInterpType interp_min = DALIInterpType::DALI_INTERP_TRIANGULAR;
  DALIInterpType interp_mag = DALIInterpType::DALI_INTERP_LINEAR;

  if (spec_.HasArgument("min_filter"))
    interp_min = spec_.GetArgument<DALIInterpType>("min_filter");
  else if (spec_.HasArgument("interp_type"))
    interp_min = spec_.GetArgument<DALIInterpType>("interp_type");

  if (spec_.HasArgument("mag_filter"))
    interp_mag = spec_.GetArgument<DALIInterpType>("mag_filter");
  else if (spec_.HasArgument("interp_type"))
    interp_mag = spec_.GetArgument<DALIInterpType>("interp_type");

  kernels::ResamplingFilterType interp2resample[] = {
    kernels::ResamplingFilterType::Nearest,
    kernels::ResamplingFilterType::Linear,
    kernels::ResamplingFilterType::Cubic,
    kernels::ResamplingFilterType::Lanczos3,
    kernels::ResamplingFilterType::Triangular,
    kernels::ResamplingFilterType::Gaussian
  };

  kernels::FilterDesc min_filter = { interp2resample[interp_min], 0 };
  kernels::FilterDesc mag_filter = { interp2resample[interp_mag], 0 };

  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    per_sample_meta_[i] = GetTransformMeta(spec_, input_shape, ws, i, ResizeInfoNeeded());
    resample_params_[i][0].output_size = per_sample_meta_[i].rsz_h;
    resample_params_[i][1].output_size = per_sample_meta_[i].rsz_w;
    resample_params_[i][0].min_filter = resample_params_[i][1].min_filter = min_filter;
    resample_params_[i][0].mag_filter = resample_params_[i][1].mag_filter = mag_filter;
  }

  context_.gpu.stream = ws->stream();
  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;
  requirements_ = Kernel::GetRequirements(context_, view<const uint8_t, 3>(input), resample_params_);
  scratch_alloc_.Reserve(requirements_.scratch_sizes);
}

template <int ndim>
void ToDimsVec(std::vector<Dims> &dims_vec, const kernels::TensorListShape<ndim> &tls) {
  const int dim = tls.sample_dim();
  const int N = tls.num_samples();
  dims_vec.resize(N);

  for (int i = 0; i < N; i++) {
    dims_vec[i].resize(dim);

    for (int j = 0; j < dim; j++)
      dims_vec[i][j] = tls.tensor_shape_span(i)[j];
  }
}

template<>
void Resize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  context_.gpu.stream = ws->stream();
  const auto &input = ws->Input<GPUBackend>(idx);

  auto &output = ws->Output<GPUBackend>(outputs_per_idx_ * idx);
  ToDimsVec(out_shape_, requirements_.output_shapes[0]);
  output.Resize(out_shape_);

  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;

  auto scratchpad = scratch_alloc_.GetScratchpad();
  context_.scratchpad = &scratchpad;
  Kernel::Run(context_, view<uint8_t, 3>(output), view<const uint8_t, 3>(input), resample_params_);

  // Setup and output the resize attributes if necessary
  if (save_attrs_) {
    TensorList<CPUBackend> attr_output_cpu;
    vector<Dims> resize_shape(input.ntensor());

    for (size_t i = 0; i < input.ntensor(); ++i) {
      resize_shape[i] = Dims{2};
    }

    attr_output_cpu.Resize(resize_shape);

    for (size_t i = 0; i < input.ntensor(); ++i) {
      int *t = attr_output_cpu.mutable_tensor<int>(i);
      t[0] = in_sizes[i].height;
      t[1] = in_sizes[i].width;
    }
    ws->Output<GPUBackend>(outputs_per_idx_ * idx + 1).Copy(attr_output_cpu, ws->stream());
  }
}

DALI_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

}  // namespace dali
