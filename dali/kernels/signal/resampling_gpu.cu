// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include "dali/core/dev_buffer.h"
#include "dali/core/mm/memory.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/resampling_gpu.cuh"
#include "dali/kernels/signal/resampling_gpu.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

template <typename Out, typename In>
void ResamplerGPU<Out, In>::Initialize(int lobes, int lookup_size) {
  windowed_sinc(window_cpu_, lookup_size, lobes);
  window_gpu_storage_.from_host(window_cpu_.storage);
  window_gpu_ = window_cpu_;
  window_gpu_.lookup = window_gpu_storage_.data();
  CUDA_CALL(cudaStreamSynchronize(0));
}

template <typename Out, typename In>
KernelRequirements ResamplerGPU<Out, In>::Setup(KernelContext &context, const InListGPU<In> &in,
                                                span<const Args> args) {
  KernelRequirements req;
  auto out_shape = in.shape;
  for (int i = 0; i < in.num_samples(); i++) {
    auto in_sh = in.shape.tensor_shape_span(i);
    auto out_sh = out_shape.tensor_shape_span(i);
    auto &arg = args[i];
    if (arg.out_begin > 0 || arg.out_end > 0) {
      out_sh[0] = arg.out_end - arg.out_begin;
    } else {
      out_sh[0] = resampled_length(in_sh[0], arg.in_rate, arg.out_rate);
    }
  }
  req.output_shapes = {out_shape};
  return req;
}

template <typename Out, typename In>
void ResamplerGPU<Out, In>::Run(KernelContext &context, const OutListGPU<Out> &out,
                                const InListGPU<In> &in, span<const Args> args) {
  if (window_gpu_storage_.empty())
    Initialize();

  assert(context.scratchpad);
  auto &scratch = *context.scratchpad;

  int nsamples = in.num_samples();
  auto samples_cpu =
      make_span(scratch.Allocate<mm::memory_kind::pinned, SampleDesc>(nsamples), nsamples);

  bool any_multichannel = false;
  for (int i = 0; i < nsamples; i++) {
    auto &desc = samples_cpu[i];
    auto in_sample = in[i];
    auto out_sample = out[i];
    desc.in = in_sample.data;
    desc.out = out_sample.data;
    desc.window = window_gpu_;
    const auto &in_sh = in_sample.shape;
    desc.in_len = in_sh[0];
    auto &arg = args[i];
    desc.out_begin = arg.out_begin > 0 ? arg.out_begin : 0;
    desc.out_end =
        arg.out_end > 0 ? arg.out_end : resampled_length(in_sh[0], arg.in_rate, arg.out_rate);
    assert((desc.out_end - desc.out_begin) == out_sample.shape[0]);
    desc.nchannels = in_sh.sample_dim() > 1 ? in_sh[1] : 1;
    desc.scale = arg.in_rate / arg.out_rate;
    any_multichannel |= desc.nchannels > 1;
  }

  auto samples_gpu = scratch.ToGPU(context.gpu.stream, samples_cpu);

  dim3 block(256, 1);
  int blocks_per_sample = std::max(32, 1024 / nsamples);
  dim3 grid(blocks_per_sample, nsamples);

  // window coefficients and temporary per channel out values
  size_t shm_size = (window_gpu_storage_.size() + (SHM_NCHANNELS + 1) * block.x) * sizeof(float);

  BOOL_SWITCH(!any_multichannel, SingleChannel,
              (ResampleGPUKernel<Out, In, SingleChannel>
               <<<grid, block, shm_size, context.gpu.stream>>>(samples_gpu);));  // NOLINT
  CUDA_CALL(cudaGetLastError());
}

DALI_INSTANTIATE_RESAMPLER_GPU();

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
