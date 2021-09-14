// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

struct SpectrogramOpImplGPU : public OpImplBase<GPUBackend> {
  explicit SpectrogramOpImplGPU(const OpSpec &spec) {
    args.window_length = spec.GetArgument<int>("window_length");
    args.window_step = spec.GetArgument<int>("window_step");
    args.nfft = spec.HasArgument("nfft") ? spec.GetArgument<int>("nfft") : args.window_length;
    int power = spec.GetArgument<int>("power");

    DALI_ENFORCE(args.window_length > 0,
      make_string("Invalid window length: ", args.window_length));
    DALI_ENFORCE(args.window_step > 0, make_string("Invalid window step: ", args.window_step));
    DALI_ENFORCE(args.window_length <= args.nfft,
      make_string("`window_length` must not exceed transform size (`nfft`). Got nfft = ", args.nfft,
      " window_length = ", args.window_length));
    DALI_ENFORCE(power >= 1 && power <= 2,
      make_string("`power` must be 1 (magnitude) or 2 (power), got: ", power));

    layout = spec.GetArgument<TensorLayout>("layout");
    DALI_ENFORCE(layout == "tf" || layout == "ft",
      make_string("Unsupported layout requested: ", layout,
                  ".\nSupported layouts are: \"tf\" and \"ft\"."));

    args.spectrum_type = static_cast<FftSpectrumType>(power);
    args.time_major_layout = layout == "tf";

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
    const auto &in_shape = in.shape();
    TensorListShape<> out_shape;
    in_shape_1D.resize(in_shape.num_samples());

    int axis = -1;
    if (in_shape.sample_dim() > 1) {
      for (int i = 0; i < in_shape.num_samples(); i++) {
        if (volume(in_shape.tensor_shape_span(i)) == 0) {
          in_shape_1D.tensor_shape_span(i)[0] = 0;
          continue;
        }
        if (axis < 0) {
          int max_extent = 0;
          // looking for non-degenerate dimension
          for (int d = 0; d < in_shape.sample_dim(); d++) {
            int extent = in_shape.tensor_shape_span(i)[d];
            if (extent > 1) {
              if (max_extent > 1) {
                DALI_FAIL("Spectogram can only be computed from 1D data. If the data is has more "
                  "dimensions, only one dimension can have extent greater than 1, e.g. "
                  "(length x 1), (1 x length), etc. The dimension with extent > 1 must be the same "
                  "one for all samples in the batch.");
              }
              axis = d;
              max_extent = extent;
            }
          }
          in_shape_1D.tensor_shape_span(i)[0] = max_extent;
        } else {
          for (int d = 0; d < in_shape.sample_dim(); d++) {
            if (d != axis && in_shape.tensor_shape_span(i)[d] > 1) {
                DALI_FAIL("Spectogram can only be computed from 1D data. If the data is has more "
                  "dimensions, only one dimension can have extent greater than 1, e.g. "
                  "(length x 1), (1 x length), etc. The dimension with extent > 1 must be the same "
                  "one for all samples in the batch.");
            }
          }
          in_shape_1D.tensor_shape_span(i)[0] = in_shape.tensor_shape_span(i)[axis];
        }
      }
      if (axis < 0)  // degenerate or true 1D case
        axis = 0;
    } else {
      in_shape_1D = in_shape.to_static<1>();
      axis = 0;
    }

    auto req = kmgr.Setup<SpectrogramGPU>(0, ctx, in_shape_1D, args);
    output_desc.resize(1);
    output_desc[0] = { req.output_shapes[0], TypeTable::GetTypeInfo(DALI_FLOAT) };

    CopyWindowToDevice(ctx.gpu.stream);
    return true;
  }

  void RunImpl(DeviceWorkspace &ws) override {
    const auto &in = ws.InputRef<GPUBackend>(0);
    auto &out = ws.OutputRef<GPUBackend>(0);
    out.SetLayout(layout);
    auto kernel_output_shape = kmgr.GetRequirements(0).output_shapes[0].to_static<2>();
    const auto in_view_1D = reshape(view<const float>(in), in_shape_1D);
    auto out_view_2D = view<float, 2>(out);
    KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr.Run<SpectrogramGPU>(0, 0, ctx, out_view_2D, in_view_1D, gpu_window);
  }

  void CopyWindowToDevice(cudaStream_t stream) {
    if (gpu_window_ptr)
      return;  // already there
    gpu_window_ptr = mm::alloc_raw_unique<float, mm::memory_kind::device>(args.window_length);
    gpu_window = make_tensor_gpu<1>(gpu_window_ptr.get(), { args.window_length });
    CUDA_CALL(cudaMemcpyAsync(gpu_window_ptr.get(), cpu_window.data(),
                              args.window_length * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  }

  KernelManager kmgr;
  StftArgs args;
  vector<float> cpu_window;
  mm::uptr<float> gpu_window_ptr;
  TensorView<StorageGPU, float, 1> gpu_window;
  TensorListShape<1> in_shape_1D;
  TensorLayout layout;
};

}  // namespace


template <>
Spectrogram<GPUBackend>::Spectrogram(const OpSpec &spec)
    : Operator<GPUBackend>(spec)
    , impl_(std::make_unique<SpectrogramOpImplGPU>(spec)) {}

DALI_REGISTER_OPERATOR(Spectrogram, Spectrogram<GPUBackend>, GPU);

}  // namespace dali
