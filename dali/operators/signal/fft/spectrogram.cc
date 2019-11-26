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
#include "dali/kernels/signal/window/extract_windows_cpu.h"
#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

#define SPECTROGRAM_SUPPORTED_NDIMS (1, 2)

namespace dali {

DALI_SCHEMA(Spectrogram)
  .DocStr(R"code(Produces a spectrogram from a 1D signal (e.g. audio). Input data is expected
to be single channel (1D shape `(time)`) or multi channel in planar layout (channel, time) 32 bit
float tensor)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("nfft",
    R"code(Size of the FFT. The number of bins created in the output is `nfft // 2 + 1`
(positive part of the spectrum only).)code",
    -1)
  .AddOptionalArg("window_length",
    R"code(Window size (in number of samples))code",
    512)
  .AddOptionalArg("window_step",
    R"code(Step betweeen the STFT windows (in number of samples))code",
    256)
  .AddOptionalArg("window_fn",
    R"code(Samples of the window function that will be multiplied to each extracted window when
  calculating the STFT. If provided it should be a list of floating point numbers of size
  `window_length`. If not provided, a Hann window will be used)code",
    std::vector<float>{})
  .AddOptionalArg("power",
    "Exponent of the magnitude of the spectrum. Supported values are 1 for energy and 2 for power)",
    2)
  .AddOptionalArg("center_windows",
    R"code(Indicates whether extracted windows should be padded so that window function is centered
at multiples of `window_step`)code",
    true)
  .AddOptionalArg("reflect_padding",
    R"code(Indicates the padding policy when sampling outside the bounds of the signal. If set to
true, the signal is mirrored with respecto to the boundary, otherwise the signal is padded with
zeros)code",
    true);

template <int Dims>
struct SpectrogramImplCpu : detail::OpImplBase<CPUBackend> {
  using OutputType = float;
  using InputType = float;

  static constexpr int InputDims = Dims;
  using WindowKernel =
    kernels::signal::window::ExtractWindowsCpu<InputType, InputType, InputDims>;

  static constexpr int WindowsDims = Dims+1;
  using FftKernel = kernels::signal::fft::Fft1DCpu<OutputType, InputType, WindowsDims>;

  explicit SpectrogramImplCpu(const OpSpec & spec);
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  int nfft_ = -1;
  int64_t window_length_ = -1;
  int64_t window_step_ = -1;
  int power_ = -1;
  bool reflect_padding_ = true;
  std::vector<float> window_fn_;
  int64_t window_center_ = -1;

  kernels::KernelManager kmgr_window_;
  kernels::signal::window::ExtractWindowsArgs window_args_;

  std::vector<OutputDesc> window_out_desc_;
  std::vector<Tensor<CPUBackend>> window_out_;

  kernels::KernelManager kmgr_fft_;
  kernels::signal::fft::FftArgs fft_args_;
};

namespace {
  void FillExtractWindowsArgs(kernels::signal::window::ExtractWindowsArgs& args,
                              int64_t window_length, int64_t window_step,
                              int64_t window_center, int ndim,
                              bool reflect_padding) {
    args.window_length = window_length;
    args.window_step = window_step;
    args.axis = ndim - 1;
    args.window_center = window_center;
    args.reflect_pad = reflect_padding;
  }

  void FillFftArgs(kernels::signal::fft::FftArgs& args,
                   int power, int window_length, int nfft, int ndims) {
    args.nfft = nfft;
    DALI_ENFORCE(window_length <= nfft, make_string(
      "Window length (", window_length, ") can't be bigger than the FFFT size (", nfft, ")"));
    switch (power) {
      case 1:
        args.spectrum_type = kernels::signal::fft::FFT_SPECTRUM_MAGNITUDE;
        break;
      case 2:
        args.spectrum_type = kernels::signal::fft::FFT_SPECTRUM_POWER;
        break;
      default:
        DALI_FAIL(make_string("`power` can be only 1 (energy) or 2 (power), received ", power));
    }

    // layout (.., freq, time)
    args.transform_axis = ndims - 2;
  }

  void CreateWindowHann(span<float> window_fn_data) {
    int N = window_fn_data.size();
    double a = (2*M_PI/N);
    for (int t = 0; t < N; t++) {
      double phase = a * (t+0.5);
      window_fn_data.data()[t] = static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
  }

}  // namespace

template <int Dims>
SpectrogramImplCpu<Dims>::SpectrogramImplCpu(const OpSpec & spec)
    : nfft_(spec.GetArgument<int>("nfft"))
    , window_length_(spec.GetArgument<int>("window_length"))
    , window_step_(spec.GetArgument<int>("window_step"))
    , power_(spec.GetArgument<int>("power"))
    , reflect_padding_(spec.GetArgument<bool>("reflect_padding"))
    , window_fn_(spec.GetRepeatedArgument<float>("window_fn")) {
  DALI_ENFORCE(window_length_ > 0, make_string("Invalid window length: ", window_length_));
  DALI_ENFORCE(window_step_ > 0, make_string("Invalid window step: ", window_step_));

  if (window_fn_.empty()) {
    window_fn_.resize(window_length_);
    CreateWindowHann(make_span(window_fn_));
  }
  DALI_ENFORCE(window_fn_.size() == static_cast<size_t>(window_length_),
    "Window function should match the specified `window_length`");

  window_center_ = spec.GetArgument<bool>("center_windows") ? window_length_ / 2 : 0;
}


template <int Dims>
bool SpectrogramImplCpu<Dims>::SetupImpl(std::vector<OutputDesc> &out_desc,
                                         const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().size();

  // Intermediate output buffers
  window_out_.resize(nthreads);

  kmgr_window_.Initialize<WindowKernel>();
  kmgr_window_.Resize<WindowKernel>(nthreads, nsamples);
  FillExtractWindowsArgs(window_args_, window_length_, window_step_, window_center_,
                         InputDims, reflect_padding_);

  kmgr_fft_.Initialize<FftKernel>();
  kmgr_fft_.Resize<FftKernel>(nthreads, nsamples);
  FillFftArgs(fft_args_, power_, window_length_, nfft_, WindowsDims);

  window_out_desc_.resize(1);
  window_out_desc_[0].type = TypeInfo::Create<InputType>();
  window_out_desc_[0].shape.resize(nsamples, WindowsDims);

  out_desc.resize(1);
  out_desc[0].type = TypeInfo::Create<OutputType>();
  out_desc[0].shape.resize(nsamples, WindowsDims);

  auto view_window_fn = make_tensor_cpu<1>(window_fn_.data(), window_length_);
  for (int i = 0; i < nsamples; i++) {
    auto &windows_req =
      kmgr_window_.Setup<WindowKernel>(
        i, ctx,
        view<const InputType, Dims>(input[i]),
        view_window_fn,
        window_args_);
    window_out_desc_[0].shape.set_tensor_shape(i, windows_req.output_shapes[0][0].shape);

    auto windows_shape = windows_req.output_shapes[0][0].template to_static<WindowsDims>();
    auto dummy_win_view = make_tensor_cpu<WindowsDims, InputType>(nullptr, windows_shape);
    auto &out_req =
      kmgr_fft_.Setup<FftKernel>(
        i, ctx,
        dummy_win_view,
        fft_args_);
    out_desc[0].shape.set_tensor_shape(i, out_req.output_shapes[0][0].shape);
  }
  return true;
}

template <int Dims>
void SpectrogramImplCpu<Dims>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();
  auto view_window_fn = make_tensor_cpu<1>(window_fn_.data(), window_length_);

  for (int i = 0; i < nsamples; i++) {
    thread_pool.DoWorkWithID(
      [this, &input, &output, view_window_fn, i](int thread_id) {
        kernels::KernelContext ctx;

        auto &win_out = window_out_[thread_id];
        win_out.set_type(TypeInfo::Create<InputType>());
        win_out.Resize(window_out_desc_[0].shape.tensor_shape(i));

        kmgr_window_.Run<WindowKernel>(
          thread_id, i, ctx,
          view<InputType, WindowsDims>(win_out),
          view<const InputType, InputDims>(input[i]),
          view_window_fn,
          window_args_);

        kmgr_fft_.Run<FftKernel>(
          thread_id, i, ctx,
          view<OutputType, WindowsDims>(output[i]),
          view<const InputType, WindowsDims>(win_out),
          fft_args_);
    });
  }

  thread_pool.WaitForWork();
}

template <>
Spectrogram<CPUBackend>::Spectrogram(const OpSpec &spec)
    : Operator<CPUBackend>(spec)
    , spec__(spec) {}

template <>
bool Spectrogram<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                        const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  VALUE_SWITCH(in_shape.sample_dim(), Dims, SPECTROGRAM_SUPPORTED_NDIMS,
    (impl_ = std::make_unique<SpectrogramImplCpu<Dims>>(spec__);),
    (DALI_FAIL(make_string("Not supported number of dimensions: ", in_shape.size()))));

  assert(impl_ != nullptr);
  return impl_->SetupImpl(output_desc, ws);
}

template <>
void Spectrogram<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Spectrogram, Spectrogram<CPUBackend>, CPU);

}  // namespace dali
