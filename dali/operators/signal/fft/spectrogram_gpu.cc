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

#include <memory>
#include <vector>
#include "dali/operators/signal/fft/spectrogram.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/signal/fft/stft_gpu.h"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace {

using kernels::KernelManager;
using kernels::KernelContext;
using kernels::signal::Padding;
using namespace kernels::signal::fft;  // NOLINT

struct SpectrogramOpImplGPU : public detail::OpImplBase<GPUBackend> {
  explicit SpectrogramOpImplGPU(const OpSpec &spec) {
    args.window_length = spec.GetArgument<int>("window_length");
    args.window_step = spec.GetArgument<int>("window_step");
    args.nfft = spec.GetArgument<int>("nfft");
    int power = spec.GetArgument<int>("power");

    DALI_ENFORCE(args.window_length > 0,
      make_string("Invalid window length: ", args.window_length));
    DALI_ENFORCE(args.window_step > 0, make_string("Invalid window step: ", args.window_step));
    DALI_ENFORCE(args.window_length <= args.nfft,
      make_string("`window_length` must not exceed ransform size (`nfft`). Got nfft = ", args.nfft,
      " window_length = ", args.window_length));
    DALI_ENFORCE(power >= 1 && power <= 2,
      make_string("`power` must be 1 (magnitude) or 2 (power), got: ", power));

    args.spectrum_type = static_cast<FftSpectrumType>(power);

    cpu_window = spec.GetRepeatedArgument<float>("window_fn");
    if (cpu_window.empty()) {
      cpu_window.resize(args.window_length);
      kernels::signal::HannWindow(make_span(cpu_window));
    }
    DALI_ENFORCE(cpu_window.size() == static_cast<size_t>(args.window_length),
      "Window function should match the specified `window_length`");

    bool center = spec.GetArgument<bool>("center_windows");
    if (center) {
      args.window_center = args.window_length / 2;
      args.padding = spec.GetArgument<bool>("reflect_padding") ? Padding::Reflect : Padding::Zero;
    } else {
      args.padding = Padding::None;
      args.window_center = 0;
    }

    kmgr.Resize<SpectrogramGPU>(1, 1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) override {
    auto &in = ws.InputRef<GPUBackend>(0);
    KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    auto in_shape = in.shape().to_static<1>();
    auto req = kmgr.Setup<SpectrogramGPU>(0, ctx, in_shape, args);
    CopyWindowToDevice(ctx.gpu.stream);
    output_desc.resize(1);
    output_desc[0] = { req.output_shapes[0], TypeTable::GetTypeInfo(DALI_FLOAT) };
    return true;
  }

  void RunImpl(DeviceWorkspace &ws) override {
    auto &in = ws.InputRef<GPUBackend>(0);
    auto &out = ws.OutputRef<GPUBackend>(0);
    assert(kmgr.GetRequirements(0).output_shapes[0] == out.shape());
    KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr.Run<SpectrogramGPU>(0, 0, ctx, view<float, 2>(out), view<float, 1>(in), gpu_window);
  }

  void CopyWindowToDevice(cudaStream_t stream) {
    if (gpu_window_ptr)
      return;  // already there
    gpu_window_ptr = kernels::memory::alloc_unique<float>(
                        kernels::AllocType::GPU, args.window_length);
    gpu_window = make_tensor_gpu<1>(gpu_window_ptr.get(), { args.window_length });
    CUDA_CALL(cudaMemcpyAsync(gpu_window_ptr.get(), cpu_window.data(),
                              args.window_length * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  }

  KernelManager kmgr;
  StftArgs args;
  vector<float> cpu_window;
  kernels::memory::KernelUniquePtr<float> gpu_window_ptr;
  TensorView<StorageGPU, float, 1> gpu_window;
};

}  // namespace


template <>
Spectrogram<GPUBackend>::Spectrogram(const OpSpec &spec)
    : Operator<GPUBackend>(spec)
    , impl_(std::make_unique<SpectrogramOpImplGPU>(spec)) {}

DALI_REGISTER_OPERATOR(Spectrogram, Spectrogram<GPUBackend>, GPU);

}  // namespace dali
