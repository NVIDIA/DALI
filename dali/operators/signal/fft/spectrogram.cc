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
#include <string>
#include <vector>
#include "dali/operators/signal/fft/spectrogram.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/signal/window/extract_windows_cpu.h"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

#define SPECTROGRAM_SUPPORTED_NDIMS (1, 2)

namespace dali {

DALI_SCHEMA(Spectrogram)
  .DocStr(R"(Produces a spectrogram from a 1D signal (for example, audio).

Input data is expected to be one channel (shape being ``(nsamples,)``, ``(nsamples, 1)``, or
``(1, nsamples)``) of type float32.)")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg<int>("nfft",
    R"(Size of the FFT.

The number of bins that are created in the output is ``nfft // 2 + 1``.

.. note::
  The output only represents the positive part of the spectrum.)",
  nullptr)
  .AddOptionalArg("window_length",
    R"(Window size in number of samples.)",
    512)
  .AddOptionalArg("window_step",
    R"(Step betweeen the STFT windows in number of samples.)",
    256)
  .AddOptionalArg("window_fn",
    R"(Samples of the window function that will be multiplied to each extracted window when
calculating the STFT.

If a value is provided, it should be a list of floating point numbers of size ``window_length``.
If a value is not provided, a Hann window will be used.)",
    std::vector<float>{})
  .AddOptionalArg("power",
    R"(Exponent of the magnitude of the spectrum.

Supported values:

- ``1`` - amplitude,
- ``2`` - power (faster to compute).
)",
    2)
  .AddOptionalArg("center_windows",
    R"(Indicates whether extracted windows should be padded so that the window function is
centered at multiples of ``window_step``.

If set to False, the signal will not be padded, that is, only windows within the input range
will be extracted.)",
    true)
  .AddOptionalArg("reflect_padding",
    R"(Indicates the padding policy when sampling outside the bounds of the signal.

If set to True, the signal is mirrored with respect to the boundary, otherwise the signal
is padded with zeros.

.. note::
  When ``center_windows`` is set to False, this option is ignored.
)",
    true)
  .AddOptionalArg("layout", R"(Output layout: "ft" (frequency-major) or "tf" (time-major).)",
    TensorLayout("ft"));

template <bool time_major>
struct SpectrogramImplCpu : OpImplBase<CPUBackend> {
  using OutputType = float;
  using InputType = float;

  static constexpr int InputDims = 1;
  static constexpr bool vertical_win = !time_major;
  using WindowKernel =
    kernels::signal::ExtractWindowsCpu<InputType, InputType, InputDims, vertical_win>;

  static constexpr int WindowsDims = 2;
  static constexpr int transform_dim = time_major ? 1 : 0;
  using FftKernel = kernels::signal::fft::Fft1DCpu<OutputType, InputType, WindowsDims>;

  explicit SpectrogramImplCpu(const OpSpec & spec);
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  int window_length_ = -1;
  int window_step_ = -1;
  int power_ = -1;
  std::vector<float> window_fn_;
  int window_center_ = -1;
  int nfft_ = -1;
  TensorLayout layout_;

  using Padding = kernels::signal::Padding;
  Padding padding_;

  kernels::KernelManager kmgr_window_;
  kernels::signal::ExtractWindowsArgs window_args_;

  std::vector<OutputDesc> window_out_desc_;
  std::vector<Tensor<CPUBackend>> window_out_;

  kernels::KernelManager kmgr_fft_;
  kernels::signal::fft::FftArgs fft_args_;
};

namespace {
  void FillFftArgs(kernels::signal::fft::FftArgs& args,
                   int power, int window_length, int nfft, int transform_axis) {
    args.nfft = nfft;
    DALI_ENFORCE(window_length <= nfft, make_string(
      "Window length (", window_length, ") can't be bigger than the FFT size (", nfft, ")"));
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

    // Time major (.., freq, time) or Freq major (.., time, freq)
    args.transform_axis = transform_axis;
  }

}  // namespace

template <bool time_major>
SpectrogramImplCpu<time_major>::SpectrogramImplCpu(const OpSpec &spec)
    : window_length_(spec.GetArgument<int>("window_length"))
    , window_step_(spec.GetArgument<int>("window_step"))
    , power_(spec.GetArgument<int>("power"))
    , window_fn_(spec.GetRepeatedArgument<float>("window_fn")) {
  DALI_ENFORCE(window_length_ > 0, make_string("Invalid window length: ", window_length_));
  DALI_ENFORCE(window_step_ > 0, make_string("Invalid window step: ", window_step_));
  nfft_ = spec.HasArgument("nfft") ? spec.GetArgument<int>("nfft") : window_length_;

  layout_ = spec.GetArgument<TensorLayout>("layout");
  assert((time_major && layout_ == "tf") || (!time_major && layout_ == "ft"));

  if (window_fn_.empty()) {
    window_fn_.resize(window_length_);
    kernels::signal::HannWindow(make_span(window_fn_));
  }
  DALI_ENFORCE(window_fn_.size() == static_cast<size_t>(window_length_),
    "Window function should match the specified `window_length`");

  if (spec.GetArgument<bool>("center_windows")) {
    window_center_ = window_length_ / 2;
    padding_ = spec.GetArgument<bool>("reflect_padding") ? Padding::Reflect : Padding::Zero;
  } else {
    padding_ = Padding::None;
    window_center_ = 0;
  }
}

template <bool time_major>
bool SpectrogramImplCpu<time_major>::SetupImpl(std::vector<OutputDesc> &out_desc,
                                               const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().NumThreads();

  // Check that input is 1-D (allowing having extra dims with extent 1)
  if (in_shape.sample_dim() > 1) {
    for (int i = 0; i < in_shape.num_samples(); i++) {
      auto shape = in_shape.tensor_shape(i);
      auto n = volume(shape);
      for (auto extent : shape) {
        DALI_ENFORCE(extent == 1 || extent == n, make_string("Input data must be 1D or all "
          "but one dimensions must be degenerate (extent 1). Got: ", shape));
      }
    }
  }

  // Intermediate output buffers
  window_out_.resize(nthreads);
  for (auto &w : window_out_) {
    if (!w.raw_data()) w.set_pinned(false);
  }

  kmgr_window_.Initialize<WindowKernel>();
  kmgr_window_.Resize<WindowKernel>(nthreads, nsamples);
  constexpr int axis = 0;
  window_args_ = {window_length_, window_center_, window_step_, axis, padding_};

  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    int64_t signal_length = in_shape[sample_id].num_elements();
    DALI_ENFORCE(window_args_.num_windows(signal_length) > 0,
      make_string("Signal is too short (", signal_length, ") for sample ", sample_id));
  }

  kmgr_fft_.Initialize<FftKernel>();
  kmgr_fft_.Resize<FftKernel>(nthreads, nsamples);
  FillFftArgs(fft_args_, power_, window_length_, nfft_, transform_dim);

  window_out_desc_.resize(1);
  window_out_desc_[0].type = TypeInfo::Create<InputType>();
  window_out_desc_[0].shape.resize(nsamples, WindowsDims);

  out_desc.resize(1);
  out_desc[0].type = TypeInfo::Create<OutputType>();
  out_desc[0].shape.resize(nsamples, WindowsDims);

  auto view_window_fn = make_tensor_cpu<1>(window_fn_.data(), window_length_);
  for (int i = 0; i < nsamples; i++) {
    auto view_signal_1d =
        make_tensor_cpu<1>(input[i].template data<const InputType>(), {input[i].size()});

    auto &windows_req =
      kmgr_window_.Setup<WindowKernel>(
        i, ctx,
        view_signal_1d,
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

template <bool time_major>
void SpectrogramImplCpu<time_major>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto out_shape = output.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();
  auto view_window_fn = make_tensor_cpu<1>(window_fn_.data(), window_length_);
  output.SetLayout(layout_);

  for (int i = 0; i < nsamples; i++) {
    thread_pool.AddWork(
      [this, &input, &output, view_window_fn, i](int thread_id) {
        kernels::KernelContext ctx;

        auto &win_out = window_out_[thread_id];
        win_out.set_type(TypeInfo::Create<InputType>());
        win_out.Resize(window_out_desc_[0].shape.tensor_shape(i));

        auto view_signal_1d =
            make_tensor_cpu<1>(input[i].data<const InputType>(), {input[i].size()});
        kmgr_window_.Run<WindowKernel>(
          thread_id, i, ctx,
          view<InputType, WindowsDims>(win_out),
          view_signal_1d,
          view_window_fn,
          window_args_);

        kmgr_fft_.Run<FftKernel>(
          thread_id, i, ctx,
          view<OutputType, WindowsDims>(output[i]),
          view<const InputType, WindowsDims>(win_out),
          fft_args_);
    }, out_shape.tensor_size(i));
  }

  thread_pool.RunAll();
}

template <>
Spectrogram<CPUBackend>::Spectrogram(const OpSpec &spec)
    : Operator<CPUBackend>(spec) {
  auto layout = spec.GetArgument<std::string>("layout");
  DALI_ENFORCE(layout == "tf" || layout == "ft",
               make_string("Unexpected layout: ", layout));
  bool time_major = layout == "tf";
  if (time_major)
    impl_ = std::make_unique<SpectrogramImplCpu<true>>(spec);
  else
    impl_ = std::make_unique<SpectrogramImplCpu<false>>(spec);
}

DALI_REGISTER_OPERATOR(Spectrogram, Spectrogram<CPUBackend>, CPU);

}  // namespace dali
