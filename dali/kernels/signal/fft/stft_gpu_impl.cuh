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
#include <vector>
#include <map>
#include "dali/core/tensor_view.h"
#include "dali/core/cuda_stream.h"
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/signal/fft/stft_gpu.h"
#include "dali/kernels/signal/fft/cufft_helper.h"
#include "dali/kernels/signal/window/extract_windows_gpu.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

class STFTImplGPU {
 public:

  KernelRequirements Setup(span<const int64_t> lengths, const STFTArgs &args);

  void RunR2C(KernelContext &ctx,
              const OutListGPU<complexf, 2> &out,
              const InListGPU<float, 1> &in,
              const InTensorGPU<float, 1> &window);

  void RunR2R(KernelContext &ctx,
              const OutListGPU<complexf, 2> &out,
              const InListGPU<float, 1> &in,
              const InTensorGPU<float, 1> &window);

 private:
  void Reset();
  void CreatePlans(int64_t nwindows);
  void ReserveTempStorage(ScratchpadEstimator &se, int64_t nwindows, int window_length);
  void CreateStreams(int new_num_streams);

  static constexpr int kMinSize = 1<<16;
  static constexpr int kMaxSize = 1<<26;

  int max_windows_ = 1, min_windows_ = 0;
  int64_t total_windows_ = 0;

  inline int transform_size() const {
    return args_.window_length;
  }

  struct PlanInfo {
    CUFFTHandle handle;
    size_t work_size = 0;
  };
  size_t total_work_size_ = 0;
  std::map<int, PlanInfo> plans_;
  std::vector<CUDAStream> streams_;
  STFTArgs args_;
  ExtractWindowsGPU<float, float> extract_windows_;
};

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_STFT_GPU_IMPL_CUH_
