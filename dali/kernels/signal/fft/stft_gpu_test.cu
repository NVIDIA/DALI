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

#include <gtest/gtest.h>
#include <random>
#include <utility>
#include <vector>
#include <cmath>
#include <complex>
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/kernels/signal/fft/fft_test_ref.h"
#include "dali/kernels/signal/fft/stft_gpu.h"
#include "dali/kernels/signal/fft/fft_postprocess.cuh"
#include "dali/core/boundary.h"
#include "dali/test/test_sound_generator.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

void RefExtractWindow(float *out,
                      int window_index, const float *in, int n,
                      const StftArgs &args, const float *win_fn = nullptr) {
  int center = args.padding != Padding::None ? args.window_center : 0;
  if (center == -1)
    center = args.window_length / 2;

  int start = window_index * args.window_step - center;
  for (int i = 0; i < args.window_length; i++) {
    int idx = i + start;
    float v = 0;
    if (args.padding == Padding::Reflect) {
      idx = boundary::idx_reflect_101(idx, n);
    }
    if (idx >= 0 && idx < n) {
      v = in[idx];
    } else {
      assert(args.padding == Padding::Zero);
    }
    out[i] = win_fn ? v * win_fn[i] : v;
  }
}

inline void RefPostprocess(span<float> out, span<const complexf> in, FftSpectrumType type) {
  switch (type) {
    case FFT_SPECTRUM_MAGNITUDE:
    {
      fft_postprocess::norm2 pp;
      for (int i = 0; i < out.size(); i++)
        out[i] = pp(in[i]);
    } break;
    case FFT_SPECTRUM_POWER:
    {
      fft_postprocess::norm2square pp;
      for (int i = 0; i < out.size(); i++)
        out[i] = pp(in[i]);
     } break;
    case FFT_SPECTRUM_POWER_DECIBELS:
    {
      fft_postprocess::power_dB pp;
      for (int i = 0; i < out.size(); i++)
        out[i] = pp(in[i]);
    } break;
    default:
      assert(!"Unexpected postprocessing method");
  }
}

inline void RefPostprocess(span<complexf> out, span<const complexf> in, FftSpectrumType type) {
  assert(type == FFT_SPECTRUM_COMPLEX && "Unexpected postprocessing method");
  for (int i = 0; i < out.size(); i++)
    out[i] = in[i];
}

template <typename T>
void RefSpectrum(const OutTensorCPU<T, 2> &ref_out, const float *in, int n,
                 const StftArgs &args, span<const float> window) {
  int t_axis = args.time_major_layout ? 0 : 1;
  int f_axis = 1 - t_axis;
  int nout = args.num_windows(n);
  int nfft = args.nfft > 0 ? args.nfft : args.window_length;
  test::ReferenceFFT<float> fft(nfft);
  vector<float> ref_wnd(nfft);
  vector<complexf> ref_fft(nfft);
  assert(window.size() <= nfft);
  int out_nfft = ref_out.shape[f_axis];
  assert(out_nfft <= nfft);
  vector<T> ref(out_nfft);

  // When the nfft is larger than the window lenght, we center the window
  // (padding with zeros on both side)
  int in_win_start = args.window_length < nfft ? (nfft - args.window_length) / 2 : 0;
  for (int i = 0; i < nout; i++) {
    RefExtractWindow(ref_wnd.data() + in_win_start, i, in, n, args, window.data());

    fft(ref_fft.data(), ref_wnd.data());
    RefPostprocess(make_span(ref), make_span(ref_fft), args.spectrum_type);

    int pos[2] = {};
    pos[t_axis] = i;
    for (int j = 0; j < out_nfft; j++) {
      pos[f_axis] = j;
      *ref_out(pos) = ref[j];
    }
  }
}

template <typename T>
void RefSpectrum(const OutListCPU<T, 2> &ref_out, const InListCPU<float, 1> &in,
                 const StftArgs &args, span<const float> window) {
  for (int i = 0; i < in.num_samples(); i++) {
    RefSpectrum(ref_out[i], in.data[i], in.shape[i][0], args, window);
  }
}

template <typename T>
void RefSpectrum(TestTensorList<T, 2> &ref_out, const InListCPU<float, 1> &in,
                 const StftArgs &args, span<const float> window) {
  int t_axis = args.time_major_layout ? 0 : 1;
  int f_axis = 1 - t_axis;
  TensorListShape<2> ref_shape;
  ref_shape.resize(in.num_samples());
  int nfft = args.nfft > 0 ? args.nfft : args.window_length;
  int spectrum_size = (nfft + 2) / 2;
  for (int i = 0; i < in.num_samples(); i++) {
    TensorShape<2> ts;
    ts[t_axis] = args.num_windows(in.shape[i][0]);
    ts[f_axis] = spectrum_size;
    ref_shape.set_tensor_shape(i, ts);
  }
  ref_out.reshape(ref_shape);
  RefSpectrum(ref_out.cpu(), in, args, window);
}

TEST(StftGPU, Setup) {
  StftGPU stft;
  TensorListShape<1> lengths = {{ 100, 1000, 13, 45 }};
  StftArgs args;
  args.axis = 0;
  args.spectrum_type = FFT_SPECTRUM_COMPLEX;
  args.window_length = 1024;
  args.window_center = 512;
  args.window_step = 512;
  for (bool time_major : { false, true }) {
    args.time_major_layout = time_major;

    KernelContext ctx;
    ctx.gpu.stream = 0;
    KernelRequirements req = stft.Setup(ctx, lengths, args);
    ASSERT_EQ(req.output_shapes.size(), 1u);
    auto &o_shape = req.output_shapes[0];
    ASSERT_EQ(o_shape.num_samples(), 4);
    for (int i = 0; i < 4; i++) {
      TensorShape<2> ts = { args.num_windows(lengths.shapes[i]), args.window_length / 2 + 1 };
      if (!time_major)
        std::swap(ts[0], ts[1]);
      EXPECT_EQ(o_shape[i], ts);
    }
  }
}

template <typename Params>
class StftGPUTest;

template <typename OutputType, FftSpectrumType spectrum_type, bool time_major>
struct StftTestParams {};

template <typename OutputType, FftSpectrumType spectrum_type, bool time_major>
class StftGPUTest<StftTestParams<OutputType, spectrum_type, time_major>>
: public ::testing::Test {
 public:
  const FftSpectrumType spectrum_type_ = spectrum_type;

  template <typename Kernel>
  void Run(StftArgs args) {
    std::mt19937_64 rng(1234);

    Kernel stft;

    TensorListShape<1> in_shapes[] = {
      {{ 1000, 15, 35321, 2048, 11111, 20480 }},
      {{ 100, 150, 64, 20480, 3213 }},
      {{ 1, 2, 3, 4, 5, 6, 7, 8 }},
      {{ 0xffff, 0xfff, 0xff, 99, 77, 66, 55, 44 }}
    };
    int nfft = args.nfft < 0 ? args.window_length : args.nfft;

    for (auto &in_shape : in_shapes) {
      TestTensorList<float, 1> in;
      TestTensorList<float, 1> window;
      const int N = in_shape.num_samples();
      in.reshape(in_shape);
      const auto lengths = make_cspan(in_shape.shapes);
      TensorListView<StorageCPU, float, 1> in_cpu = in.cpu();
      for (int i = 0; i < N; i++) {
        testing::GenerateTestWave(rng, in_cpu.data[i], lengths[i], 30, lengths[i] / 5);
      }

      window.reshape({{TensorShape<1>{args.window_length}}});
      auto window_span = make_span(window.cpu().data[0], args.window_length);
      HannWindow(window_span);

      args.time_major_layout = time_major;
      TestTensorList<OutputType, 2> out;

      KernelContext ctx;
      ctx.gpu.stream = 0;
      KernelRequirements req = stft.Setup(ctx, in_shape, args);
      auto stream = ctx.gpu.stream;
      ASSERT_EQ(req.output_shapes.size(), 1u);
      auto &out_shape = req.output_shapes[0];
      ASSERT_EQ(out_shape.num_samples(), N);

      for (int i = 0; i < N; i++) {
        TensorShape<2> ts = { args.num_windows(lengths[i]), nfft / 2 + 1 };
        if (!time_major)
          std::swap(ts[0], ts[1]);
        EXPECT_EQ(out_shape[i], ts);
      }

      DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
      ctx.scratchpad = &dyn_scratchpad;

      auto window_gpu = window.gpu(stream)[0];
      out.reshape(convert_dim<2>(out_shape));
      stft.Run(ctx, out.gpu(stream), in.gpu(stream), window_gpu);
      TensorListView<StorageCPU, OutputType, 2> out_cpu = out.cpu(stream);

      TestTensorList<OutputType, 2> ref;
      RefSpectrum(ref, in_cpu, args, window_span);
      Check(out_cpu, ref.cpu(), EqualEpsRel(2e-5, 2e-4));
    }
  }

  void TestBatched(StftArgs args) {
    const bool is_float = std::is_same<OutputType, float>::value;
    using Kernel = std::conditional_t<is_float, SpectrogramGPU, StftGPU>;
    Run<Kernel>(std::move(args));
  }
};

using StftTypes = ::testing::Types<
  StftTestParams<complexf, FFT_SPECTRUM_COMPLEX, false>,
  StftTestParams<complexf, FFT_SPECTRUM_COMPLEX, true>,
  StftTestParams<float, FFT_SPECTRUM_MAGNITUDE, false>,
  StftTestParams<float, FFT_SPECTRUM_MAGNITUDE, true>,
  StftTestParams<float, FFT_SPECTRUM_POWER, false>,
  StftTestParams<float, FFT_SPECTRUM_POWER, true>
>;

TYPED_TEST_SUITE(StftGPUTest, StftTypes);

TYPED_TEST(StftGPUTest, TestBatched_DefaultNfft) {
  StftArgs args;
  args.axis = 0;
  args.spectrum_type = this->spectrum_type_;
  args.window_length = 64;
  args.nfft = -1;  // default
  args.window_center = 30;
  args.window_step = 40;
  args.padding = signal::Padding::Reflect;

  this->TestBatched(args);
}

TYPED_TEST(StftGPUTest, TestBatched_DifferentWindowLenAndNfft) {
  StftArgs args;
  args.axis = 0;
  args.spectrum_type = this->spectrum_type_;
  args.window_length = 200;
  args.nfft = 256;
  args.window_center = 100;
  args.window_step = 120;
  args.padding = signal::Padding::Reflect;

  this->TestBatched(args);
}


}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
