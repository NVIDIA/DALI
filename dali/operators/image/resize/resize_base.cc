// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/data/views.h"

namespace dali {

inline kernels::ResamplingFilterType interp2resample(DALIInterpType interp) {
#define DALI_MAP_INTERP_TO_RESAMPLE(interp, resample) case DALI_INTERP_##interp:\
  return kernels::ResamplingFilterType::resample;

  switch (interp) {
    DALI_MAP_INTERP_TO_RESAMPLE(NN, Nearest);
    DALI_MAP_INTERP_TO_RESAMPLE(LINEAR, Linear);
    DALI_MAP_INTERP_TO_RESAMPLE(CUBIC, Cubic);
    DALI_MAP_INTERP_TO_RESAMPLE(LANCZOS3, Lanczos3);
    DALI_MAP_INTERP_TO_RESAMPLE(GAUSSIAN, Gaussian);
    DALI_MAP_INTERP_TO_RESAMPLE(TRIANGULAR, Triangular);
  default:
    DALI_FAIL("Unknown interpolation type");
  }
#undef DALI_MAP_INTERP_TO_RESAMPLE
}

DALI_SCHEMA(ResamplingFilterAttr)
  .DocStr(R"code(Resampling filter attribute placeholder)code")
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used. Use `min_filter` and `mag_filter` to specify
different filtering for downscaling and upscaling.)code",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("mag_filter", "Filter used when scaling up",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("min_filter", "Filter used when scaling down",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("temp_buffer_hint",
      "Initial size, in bytes, of a temporary buffer for resampling.\n"
      "Ingored for CPU variant.\n",
      0)
  .AddOptionalArg("minibatch_size", "Maximum number of images processed in a single kernel call",
      32);

ResamplingFilterAttr::ResamplingFilterAttr(const OpSpec &spec) {
  DALIInterpType interp_min = DALIInterpType::DALI_INTERP_LINEAR;
  DALIInterpType interp_mag = DALIInterpType::DALI_INTERP_LINEAR;

  if (spec.HasArgument("min_filter"))
    interp_min = spec.GetArgument<DALIInterpType>("min_filter");
  else if (spec.HasArgument("interp_type"))
    interp_min = spec.GetArgument<DALIInterpType>("interp_type");

  if (spec.HasArgument("mag_filter"))
    interp_mag = spec.GetArgument<DALIInterpType>("mag_filter");
  else if (spec.HasArgument("interp_type"))
    interp_mag = spec.GetArgument<DALIInterpType>("interp_type");

  min_filter_ = { interp2resample(interp_min), 0 };
  mag_filter_ = { interp2resample(interp_mag), 0 };

  temp_buffer_hint_ = spec.GetArgument<int64_t>("temp_buffer_hint");
}
void ResizeBase::SubdivideInput(const kernels::InListGPU<uint8_t, 3> &in) {
  for (auto &mb : minibatches_) {
    sample_range(mb.input, in, mb.start, mb.start + mb.count);
  }
}

void ResizeBase::SubdivideOutput(const kernels::OutListGPU<uint8_t, 3> &out) {
  for (auto &mb : minibatches_) {
    sample_range(mb.output, out, mb.start, mb.start + mb.count);
  }
}
void ResizeBase::RunGPU(TensorList<GPUBackend> &output,
                        const TensorList<GPUBackend> &input,
                        cudaStream_t stream) {
  output.set_type(input.type());

  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;

  auto in_view = view<const uint8_t, 3>(input);
  SubdivideInput(in_view);

  out_shape_ = TensorListShape<>();
  out_shape_.resize(in_view.num_samples(), in_view.sample_dim());
  int sample_idx = 0;

  kernels::KernelContext context;
  context.gpu.stream = stream;
  for (size_t b = 0; b < minibatches_.size(); b++) {
    MiniBatch &mb = minibatches_[b];

    auto &req = kmgr_.Setup<Kernel>(b, context,
        mb.input, make_span(resample_params_.data() + mb.start, mb.count));

    mb.out_shape = req.output_shapes[0];
    // minbatches have uniform dim
    for (int i = 0; i < mb.out_shape.size(); i++) {
      out_shape_.set_tensor_shape(i + sample_idx, mb.out_shape[i]);
    }
    sample_idx += mb.out_shape.size();
  }

  output.Resize(out_shape_);

  auto out_view = view<uint8_t, 3>(output);
  SubdivideOutput(out_view);
  kmgr_.ReserveMaxScratchpad(0);

  for (size_t b = 0; b < minibatches_.size(); b++) {
    MiniBatch &mb = minibatches_[b];

    kmgr_.Run<Kernel>(0, b, context,
        mb.output, mb.input, make_span(resample_params_.data() + mb.start, mb.count));
  }
}

void ResizeBase::Initialize(int num_threads) {
  using Kernel = kernels::ResampleCPU<uint8_t, uint8_t>;
  kmgr_.Resize<Kernel>(num_threads, num_threads);
  out_shape_.resize(num_threads, 3);
  resample_params_.resize(num_threads);
}

void ResizeBase::InitializeGPU(int batch_size, int mini_batch_size) {
  DALI_ENFORCE(batch_size > 0, "Batch size must be positive");
  DALI_ENFORCE(mini_batch_size > 0, "Mini-batch size must be positive");
  const int num_minibatches = div_ceil(batch_size, mini_batch_size);
  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;
  kmgr_.Resize<Kernel>(1, num_minibatches);
  minibatches_.resize(num_minibatches);

  for (int i = 0; i < num_minibatches; i++) {
    int start = batch_size * i / num_minibatches;
    int end = (batch_size * (i + 1)) / num_minibatches;

    minibatches_[i].start = start;
    minibatches_[i].count = end-start;
  }

  kmgr_.SetMemoryHint(kernels::AllocType::GPU, temp_buffer_hint_);
}

void ResizeBase::RunCPU(Tensor<CPUBackend> &output,
                        const Tensor<CPUBackend> &input,
                        int thread_idx) {
  using Kernel = kernels::ResampleCPU<uint8_t, uint8_t>;
  auto in_view = view<const uint8_t, 3>(input);
  kernels::KernelContext context;
  auto &req = kmgr_.Setup<Kernel>(
      thread_idx, context,
      in_view, resample_params_[thread_idx]);

  const auto &input_shape = input.shape();

  auto out_shape = req.output_shapes[0][0];
  out_shape_.set_tensor_shape(thread_idx, out_shape.shape);

  // Resize the output & run
  output.Resize(out_shape_[thread_idx]);
  auto out_view = view<uint8_t, 3>(output);
  kmgr_.Run<Kernel>(
      thread_idx, thread_idx, context,
      out_view, in_view, resample_params_[thread_idx]);
}

}  // namespace dali
