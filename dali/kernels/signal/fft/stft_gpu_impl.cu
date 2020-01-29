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

#include "dali/kernels/signal/fft/stft_gpu_impl.cuh"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

KernelRequirements StftGpuImpl::Setup(span<const int64_t> lengths, const StftArgs &args) {
  if (args != args_) {
    plans_.clear();
    args_ = args;
  }
  int64_t windows = 0;
  TensorListShape<> shape;
  shape.resize(lengths.size(), 2);
  int N = lengths.size();
  for (int i = 0; i < N; i++) {
    int64_t l = lengths[i];
    int64_t n = args_.num_windows(l);
    windows += n;
    TensorShape<2> ts = { n, (args.window_length + 2) / 2 };
    shape.set_tensor_shape(i, ts);
  }
  KernelRequirements req;
  ScratchpadEstimator se;

  CreatePlans(windows);
  ReserveTempStorage(se, windows, args.window_length);

  req.output_shapes = { shape };
  return req;
}


void StftGpuImpl::CreatePlans(int64_t nwindows) {
  int64_t max_windows = kMaxSize;

  while (max_windows * transform_size() > kMaxSize)
    max_windows >>= 1;

  while (max_windows > nwindows)
    max_windows >>= 1;

  if (max_windows == 0)
    max_windows = 1;

  max_windows_ = max_windows;
  min_windows_ = std::min(max_windows_, next_pow2(kMinSize / transform_size()));

  int n[1] = { transform_size() };
  for (int w = max_windows_; w >= min_windows_; w >>= 1) {
    auto &plan = plans_[w];
    if (!plan.handle) {
      cufftHandle handle;
      CUDA_CALL(cufftCreate(&handle));
      plan.handle.reset(handle);
      CUDA_CALL(cufftSetAutoAllocation(handle, 0));
      plan.work_size = 0;
      CUDA_CALL(cufftMakePlanMany(handle, 1, n, 0, 0, 0, 0, 0, 0, CUFFT_R2C, w, &plan.work_size));
      total_work_size_ += plan.work_size;
    }
  }

  CreateStreams(plans_.size());
}

void StftGpuImpl::CreateStreams(int new_num_streams) {
  int num_streams = streams_.size();

  if (num_streams < new_num_streams) {
    streams_.resize(new_num_streams);
    for (int i = num_streams; i < new_num_streams; i++)
      streams_[i] = CUDAStream::Create(true);
  }
}

void StftGpuImpl::ReserveTempStorage(ScratchpadEstimator &se, int64_t nwindows, int window_length) {
  assert(is_pow2(min_windows_));
  int64_t wnd = align_up(nwindows, min_windows_);
  se.add<float>(AllocType::GPU, wnd * window_length, 8);
  se.add<float>(AllocType::GPU, total_work_size_);
}


void StftGpuImpl::RunR2C(KernelContext &ctx,
                         const OutListGPU<complexf, 2> &out,
                         const InListGPU<float, 1> &in) {
  int N = in.num_samples();
  assert(out.num_samples() == N);
  int nout = (args_.window_length + 2) / 2;
  for (int i = 0; i < N; i++) {
    auto length = in.shape[i][0];
    assert(out.shape[i] == (TensorShape<2>{ args_.num_windows(length), nout }));
  }


}


}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
