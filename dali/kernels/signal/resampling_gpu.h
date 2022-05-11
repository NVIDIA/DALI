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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_GPU_H_
#define DALI_KERNELS_SIGNAL_RESAMPLING_GPU_H_

#include <cuda_runtime.h>
#include "dali/kernels/signal/resampling.h"
#include "dali/kernels/signal/resampling_gpu.cuh"
#include "dali/kernels/kernel.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/mm/memory.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

template <typename OutputType = float, typename InputType = OutputType>
class ResamplerGPU {
 public:
  void Initialize(int lobes = 16, int lookup_size = 2048) {
    windowed_sinc(window_cpu_, lookup_size, lobes);
    window_gpu_storage_.from_host(window_cpu_.storage);
    window_gpu_ = window_cpu_;
    window_gpu_.lookup = window_gpu_storage_.data();
  }

  KernelRequirements Setup(KernelContext &context, const InListGPU<InputType> &in,
                           span<const float> in_rate, span<const float> out_rate) {
    KernelRequirements req;
    auto out_shape = in.shape;
    for (int i = 0; i < in.num_samples(); i++) {
      auto in_sh = in.shape.tensor_shape_span(i);
      auto out_sh = out_shape.tensor_shape_span(i);
      out_sh[0] = resampled_length(in_sh[0], in_rate[i], out_rate[i]);
    }
    req.output_shapes = {out_shape};
    return req;
  }

  void Run(KernelContext &context, const OutListGPU<OutputType> &out,
           const InListGPU<InputType> &in, span<const float> in_rates,
           span<const float> out_rates) {
    if (window_gpu_storage_.empty())
      Initialize();

    DynamicScratchpad dyn_scratchpad({}, AccessOrder(context.gpu.stream));
    if (!context.scratchpad)
      context.scratchpad = &dyn_scratchpad;
    auto &scratch = *context.scratchpad;

    int nsamples = in.num_samples();
    auto samples_cpu =
        make_span(scratch.Allocate<mm::memory_kind::pinned, SampleDesc>(nsamples), nsamples);

    bool any_multichannel = false;
    for (int i = 0; i < nsamples; i++) {
      auto &desc = samples_cpu[i];
      desc.in = in[i].data;
      desc.out = out[i].data;
      desc.window = window_gpu_;
      const auto &in_sh = in[i].shape;
      const auto &out_sh = out[i].shape;
      desc.in_len = in_sh[0];
      desc.out_len = resampled_length(in_sh[0], in_rates[i], out_rates[i]);
      assert(desc.out_len == out_sh[0]);
      desc.nchannels = in[i].shape.sample_dim() > 1 ? in_sh[1] : 1;
      desc.scale = static_cast<double>(in_rates[i]) / out_rates[i];
      any_multichannel |= desc.nchannels > 1;
    }

    auto samples_gpu = scratch.ToGPU(context.gpu.stream, samples_cpu);

    dim3 block(256, 1);
    int blocks_per_sample = std::max(32, 1024 / nsamples);
    dim3 grid(blocks_per_sample, nsamples);
    size_t shm_size = window_gpu_storage_.size() * sizeof(float);

    BOOL_SWITCH(!any_multichannel, SingleChannel, (
      ResampleGPUKernel<OutputType, InputType, SingleChannel>
        <<<grid, block, shm_size, context.gpu.stream>>>(samples_gpu);
    ));  // NOLINT
    CUDA_CALL(cudaGetLastError());
  }

 private:
  ResamplingWindowCPU window_cpu_;
  ResamplingWindow window_gpu_;
  DeviceBuffer<float> window_gpu_storage_;
};

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_GPU_H_
