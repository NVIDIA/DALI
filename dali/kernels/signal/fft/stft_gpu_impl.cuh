// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_FFT_STFT_GPU_IMPL_CUH_
#define DALI_KERNELS_SIGNAL_FFT_STFT_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include <cufft.h>
#include "dali/core/tensor_view.h"
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/signal/fft/stft_gpu.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

class StftGpuImpl {
 public:
  bool UpdateArgs(const StftArgs &args) {
    if (args.estimated_max_total_length != args_.estimated_max_total_length ||
        args.window_length != args_.window_length)
      return false;
    args_ = args;
    return true;
  }

  KernelRequirements Setup(span<int64_t> lengths) {
    int64_t windows = 0;
    int64_t total_length = 0;
    TensorListShape<> shape;
    shape.resize(lengths.size(), 2);
    int N = lengths.size();
    for (int i = 0; i < N; i++) {
      //
    }
    KernelRequirements req;
    req.output_shapes = { shape };
    return req;
  }

 private:
  cufftHandle plan_;
  StftArgs args_;
};


}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_STFT_GPU_IMPL_CUH_
