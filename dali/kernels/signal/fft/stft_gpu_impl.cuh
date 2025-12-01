// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <map>
#include <memory>
#include <vector>
#include "dali/core/tensor_view.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_event.h"
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/signal/fft/stft_gpu.h"
#include "dali/kernels/signal/fft/cufft_helper.h"
#include "dali/kernels/signal/window/extract_windows_gpu.h"
#include "dali/kernels/signal/fft/fft_postprocess.cuh"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

class StftImplGPU {
 public:
  StftImplGPU() = default;
  StftImplGPU(StftImplGPU &&) = delete;
  StftImplGPU(const StftImplGPU &) = delete;


  KernelRequirements Setup(KernelContext &ctx, span<const int64_t> lengths, const StftArgs &args);

  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<1> &lengths,
                           const StftArgs &args) {
    return Setup(ctx, make_span(lengths.shapes), args);
  }

  void Run(KernelContext &ctx,
           const OutListGPU<complexf, 2> &out,
           const InListGPU<float, 1> &in,
           const InTensorGPU<float, 1> &window);

  void Run(KernelContext &ctx,
           const OutListGPU<float, 2> &out,
           const InListGPU<float, 1> &in,
           const InTensorGPU<float, 1> &window);

 private:
  void Reset();

  // setup functions

  void CreatePlans(int64_t nwindows);
  void CreateStreams(int new_num_streams);
  void ReserveTempStorage();
  void SetupWindowExtraction(KernelContext &ctx, span<const int64_t> input_lengths);
  void SetupPostprocessing(KernelContext &ctx);

  std::unique_ptr<fft_postprocess::FFTPostprocess<float2, float2>> post_complex_;
  std::unique_ptr<fft_postprocess::FFTPostprocess<float, float2>> post_real_;

  // execution functions

  struct ExecutionContext;

  void Run(ExecutionContext &ctx);
  void ExtractWindows(ExecutionContext &ctx);
  void RunTransform(ExecutionContext &ctx);
  void StoreComplexResult(ExecutionContext &ctx);
  void StoreRealResult(ExecutionContext &ctx);
  void ValidateParams(ExecutionContext &ctx);
  void StoreResult(ExecutionContext &ctx);

  static constexpr int kMinSize = 1<<16;
  static constexpr int kMaxSize = 1<<26;

  int max_windows_ = 1, min_windows_ = 0;
  int64_t total_windows_ = 0;

  inline int transform_size() const {
    return args_.nfft > 0 ? args_.nfft : args_.window_length;
  }
  inline int transform_in_size() const {
    // for in-place transform, it's be transform_size() + 2,
    // for out-of-place it's equal to transfrom_size()
    return transform_size();
  }
  inline int transform_out_size() const {
    return (transform_size() + 2) / 2;
  }
  inline int64_t num_temp_windows() const {
    assert(is_pow2(min_windows_));
    return align_up(total_windows_, min_windows_);
  }

  struct PlanInfo {
    CUFFTHandle handle;
    size_t work_size = 0;
  };
  std::map<int, PlanInfo> plans_;
  struct Stream {
    CUDAStream stream;
    CUDAEvent event;
  };
  CUDAEvent main_stream_ready_;
  std::vector<Stream> streams_;
  size_t max_work_size_ = 0;
  static constexpr int kMaxStreams = 4;

  StftArgs args_;

  ExtractWindowsGPU<float, float> window_extractor_;
  TensorListView<StorageGPU, float, 2> transform_in_;
  TensorListView<StorageGPU, float2, 2> transform_out_;

  struct ExecutionContext {
    struct Params {
      KernelContext *ctx;
      const OutListGPU<complexf, 2> *complex_out;
      const OutListGPU<float, 2> *real_out;
      const InListGPU<float, 1> *in;
      const InTensorGPU<float, 1> *window;
    } params;
    explicit ExecutionContext(const Params &params) : params(params) {}

    KernelContext &context() const { return *params.ctx; }
    Scratchpad *scratchpad() const { return params.ctx->scratchpad; }
    cudaStream_t stream() const { return params.ctx->gpu.stream; }
    const OutListGPU<float, 2> &real_out() const { return *params.real_out; }
    const OutListGPU<complexf, 2> &complex_out() const { return *params.complex_out; }
    const InListGPU<float, 1> &in() const { return *params.in; }
    const InTensorGPU<float, 1> &window() const { return *params.window; }

    bool has_window_func() const { return params.window && params.window->num_elements() != 0; }

    bool has_real_output() const { return params.real_out != nullptr; }

    const TensorListShape<2> &output_shape() const {
      return has_real_output() ? real_out().shape : complex_out().shape;
    }
  };
};

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_STFT_GPU_IMPL_CUH_
