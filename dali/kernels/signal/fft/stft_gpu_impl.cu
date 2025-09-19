// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/format.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

void StftImplGPU::Reset() {
  plans_.clear();
  post_complex_.reset();
  post_real_.reset();
}

KernelRequirements StftImplGPU::Setup(
    KernelContext &ctx, span<const int64_t> lengths, const StftArgs &args) {
  if (args != args_) {
    Reset();
    args_ = args;
  }

  int64_t windows = 0;
  TensorListShape<> shape;
  shape.resize(lengths.size(), 2);

  transform_in_.data.clear();
  transform_out_.data.clear();
  transform_in_.resize(lengths.size(), 2);
  transform_out_.resize(lengths.size(), 2);
  int N = lengths.size();
  for (int i = 0; i < N; i++) {
    int64_t l = lengths[i];
    int64_t n = args_.num_windows(l);
    DALI_ENFORCE(n > 0, make_string("Signal is too short (", l, ") for sample ", i));
    TensorShape<2> ts_in = { n, transform_in_size() };
    TensorShape<2> ts_out = { n, transform_out_size() };
    transform_out_.shape.set_tensor_shape(i, ts_out);
    transform_in_.shape.set_tensor_shape(i, ts_in);

    if (!args_.time_major_layout)
      std::swap(ts_out[0], ts_out[1]);
    shape.set_tensor_shape(i, ts_out);
    windows += n;
  }
  total_windows_ = windows;

  KernelRequirements req;

  SetupWindowExtraction(ctx, lengths);
  CreatePlans(windows);
  ReserveTempStorage();
  SetupPostprocessing(ctx);

  req.output_shapes = { shape };
  return req;
}

void StftImplGPU::SetupWindowExtraction(
    KernelContext &ctx,
    span<const int64_t> input_lengths) {
  ExtractWindowsBatchedArgs extract_args;
  static_cast<ExtractWindowsArgs &>(extract_args) = static_cast<ExtractWindowsArgs &>(args_);
  extract_args.output_window_length = transform_in_size();  // include padding, if necessary

  KernelRequirements extract_req = window_extractor_.Setup(ctx, input_lengths, extract_args);
}

void StftImplGPU::SetupPostprocessing(KernelContext &ctx) {
  if (args_.spectrum_type == FFT_SPECTRUM_COMPLEX) {
    post_real_.reset();
    post_complex_ = fft_postprocess::GetSTFTPostprocessor(args_.time_major_layout);
    auto req = post_complex_->Setup(ctx, transform_out_.shape);
  } else {
    post_complex_.reset();
    post_real_ = fft_postprocess::GetSpectrogramPostprocessor(
        args_.time_major_layout, args_.spectrum_type);
    auto req = post_real_->Setup(ctx, transform_out_.shape);
  }
}


void StftImplGPU::CreatePlans(int64_t nwindows) {
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
      CUDA_CALL(cufftSetAutoAllocation(handle, false));
      plan.work_size = 0;
      CUDA_CALL(cufftMakePlanMany(
          handle, 1, n,
          0, 0, 0, 0, 0, 0,
          CUFFT_R2C, w, &plan.work_size));
    }
  }

  CreateStreams(std::min<int>(plans_.size(), kMaxStreams + 0 /* clang bug */));
}

void StftImplGPU::CreateStreams(int new_num_streams) {
  int num_streams = streams_.size();

  if (num_streams < new_num_streams) {
    streams_.resize(new_num_streams);
    for (int i = num_streams; i < new_num_streams; i++)
      streams_[i] = { CUDAStream::Create(true), CUDAEvent::Create() };
  }
}

void StftImplGPU::ReserveTempStorage() {
  // TODO(michalz) - try in-place transform to reduce memory footprint
  // extracted windows

  int windows_left = total_windows_;
  int max_plan = num_temp_windows();

  size_t max_work = 0;
  while (windows_left > 0) {
    auto it = plans_.upper_bound(max_plan);
    assert(it != plans_.begin());
    --it;
    int batch = it->first;
    max_work = std::max(max_work, it->second.work_size);
    windows_left -= batch;
    max_plan -= batch;
  }

  max_work_size_ = max_work;
}

void StftImplGPU::ValidateParams(ExecutionContext &ctx) {
  int N = ctx.in().num_samples();
  const TensorListShape<2> &out_shape = ctx.output_shape();

  DALI_ENFORCE(out_shape.num_samples() == N, make_string(
        "Unexpected number of samples in output list: ", out_shape.num_samples(), " vs ", N));

  for (int i = 0; i < N; i++) {
    TensorShape<2> ts = transform_out_.shape[i];

    if (!args_.time_major_layout)
      std::swap(ts[0], ts[1]);

    DALI_ENFORCE(out_shape[i] == ts,
        make_string("Unexpected output shape at sample ", i, ": ", out_shape[i], " expected ", ts));
  }
}

void StftImplGPU::Run(KernelContext &ctx,
                      const OutListGPU<complexf, 2> &out,
                      const InListGPU<float, 1> &in,
                      const InTensorGPU<float, 1> &window) {
  assert(args_.spectrum_type == FFT_SPECTRUM_COMPLEX);

  ExecutionContext ectx({ &ctx, &out, nullptr, &in, &window });
  Run(ectx);
}

void StftImplGPU::Run(KernelContext &ctx,
                      const OutListGPU<float, 2> &out,
                      const InListGPU<float, 1> &in,
                      const InTensorGPU<float, 1> &window) {
  assert(args_.spectrum_type != FFT_SPECTRUM_COMPLEX);

  ExecutionContext ectx({ &ctx, nullptr, &out, &in, &window });
  Run(ectx);
}

void StftImplGPU::Run(ExecutionContext &ctx) {
  ValidateParams(ctx);
  ExtractWindows(ctx);
  RunTransform(ctx);
  StoreResult(ctx);
}

void StftImplGPU::StoreResult(ExecutionContext &ctx) {
  if (ctx.has_real_output())
    StoreRealResult(ctx);
  else
    StoreComplexResult(ctx);
}

void StftImplGPU::StoreComplexResult(ExecutionContext &ctx) {
  static_assert(sizeof(ctx.complex_out().data[0][0]) == sizeof(float2),
                "Complex output should have element type with the same size as float2");
  // reinterpret the whole TensorListView reference to avoid copying the shape
  auto &out = reinterpret_cast<const OutListGPU<float2, 2> &>(ctx.complex_out());
  post_complex_->Run(ctx.context(), out, transform_out_);
}

void StftImplGPU::StoreRealResult(ExecutionContext &ctx) {
  post_real_->Run(ctx.context(), ctx.real_out(), transform_out_);
}

void StftImplGPU::RunTransform(ExecutionContext &ctx) {
  float2 *fft_out = ctx.scratchpad()->AllocateGPU<float2>(
      num_temp_windows() * transform_out_size());
  transform_out_.set_contiguous_data(fft_out);
  assert(transform_in_.is_contiguous());
  float *fft_in = transform_in_.data[0];

  SmallVector<char *, kMaxStreams> work;
  for (size_t i = 0; i < streams_.size(); i++)
    work[i] = ctx.scratchpad()->AllocateGPU<char>(max_work_size_, 16);

  int64_t windows_left = total_windows_;
  int64_t max_plan = num_temp_windows();

  int64_t in_ofs = 0, out_ofs = 0;

  // Max_plan may actually be greater than the number of windows taken from the input.
  // It's OK, since the intermediate buffers are overallocated.
  // This helps us reduce the number of cuFFT calls for batch sizes difficult to decompose
  // to powers of 2.

  int max_stream = -1;
  int stream_idx = 0;
  bool first_round = true;
  if (!main_stream_ready_)
    main_stream_ready_ = CUDAEvent::Create();
  CUDA_CALL(cudaEventRecord(main_stream_ready_, ctx.stream()));
  while (windows_left > 0) {
    auto it = plans_.upper_bound(max_plan);
    assert(it != plans_.begin());
    --it;
    int64_t batch = it->first;  // widen for multiplication

    max_stream = std::max(max_stream, stream_idx);
    PlanInfo &pi = it->second;
    if (first_round)
      CUDA_CALL(cudaStreamWaitEvent(streams_[stream_idx].stream, main_stream_ready_, 0));
    CUDA_CALL(cufftSetStream(pi.handle, streams_[stream_idx].stream));
    CUDA_CALL(cufftSetWorkArea(pi.handle, work[stream_idx]));
    CUDA_CALL(cufftExecR2C(pi.handle, fft_in + in_ofs, fft_out + out_ofs));
    windows_left -= batch;
    max_plan -= batch;
    in_ofs += batch * transform_in_size();
    out_ofs += batch * transform_out_size();
    stream_idx++;
    if (stream_idx >= static_cast<int>(streams_.size())) {
      stream_idx = 0;
      first_round = false;
    }
  }
  for (int i = 0; i <= max_stream; i++) {
    CUDA_CALL(cudaEventRecord(streams_[i].event, streams_[i].stream));
    CUDA_CALL(cudaStreamWaitEvent(ctx.stream(), streams_[i].event, 0));
  }
}

void StftImplGPU::ExtractWindows(ExecutionContext &ctx) {
  float *fft_in = ctx.scratchpad()->AllocateGPU<float>(
      num_temp_windows() * transform_in_size(), alignof(float2));
  transform_in_.set_contiguous_data(fft_in);

  window_extractor_.Run(ctx.context(), transform_in_, ctx.in(), ctx.window());

  int64_t ofs = transform_in_.num_elements();
  int64_t pad = num_temp_windows() * transform_in_size() - ofs;

  // 0-pad to avoid running FFT on garbage
  CUDA_CALL(cudaMemsetAsync(fft_in + ofs, 0, pad*sizeof(float), ctx.stream()));
}

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
