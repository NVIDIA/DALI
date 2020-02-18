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

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/kernels/signal/fft/stft_gpu_impl.cuh"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/kernels/signal/fft/fft_test_ref.h"
#include "dali/core/boundary.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

void RefExtractWindow(float *out,
                      int window_index, const float *in, int n,
                      const StftArgs &args) {
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
    out[i] = v;
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

  for (int i = 0; i < nout; i++) {
    RefExtractWindow(ref_wnd.data(), i, in, n, args);
    for (int j = 0; j < window.size(); j++)
      ref_wnd[j] *= window[j];

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

TEST(StftImplGPU, Setup) {
  StftImplGPU stft;
  int64_t lengths[] = { 100, 1000, 13, 45 };
  StftArgs args;
  args.axis = 0;
  args.spectrum_type = FFT_SPECTRUM_COMPLEX;
  args.window_length = 1024;
  args.window_center = 512;
  args.window_step = 512;
  for (bool time_major : { false, true }) {
    args.time_major_layout = time_major;

    KernelContext ctx;
    KernelRequirements req = stft.Setup(ctx, make_span(lengths), args);
    ASSERT_EQ(req.output_shapes.size(), 1u);
    auto &o_shape = req.output_shapes[0];
    ASSERT_EQ(o_shape.num_samples(), 4);
    for (int i = 0; i < 4; i++) {
      TensorShape<2> ts = { args.num_windows(lengths[i]), args.window_length / 2 + 1 };
      if (!time_major)
        std::swap(ts[0], ts[1]);
      EXPECT_EQ(o_shape[i], ts);
    }
  }
}

void GenerateTestSound(float *out, int length, float freq, float amplitude) {
  float m = M_PI * freq;
  int fade = length/4;
  float ampl[8];

  for (int h = 0; h < 8; h++)
    ampl[h] = pow(0.5, h);

  auto signal = [&](int i) {
    float v = 0;
    for (int h = 0; h < 8; h++) {
      float phase = i * m * (2*h+1);  // generate odd harmonics
      v += sin(phase) * ampl[h];
    }
    return v * amplitude;
  };

  int i = 0;
  for (; i < fade; i++) {
    float envelope = (1 - cos(M_PI*i/fade)) * 0.5f;
    out[i] += signal(i) * envelope;
  }

  for (; i < length - fade; i++) {
    out[i] += signal(i);
  }

  for (; i < length; i++) {
    float envelope = (1 - cos(M_PI*(length - i)/fade)) * 0.5f;
    out[i] += signal(i) * envelope;
  }
}

template <typename RNG>
void GenerateTestWave(RNG &rng, float *out, int length, int num_sounds, int max_sound_length,
                      float noise_level = 0.01f) {
  std::normal_distribution<float> noise(0, noise_level);
  std::uniform_int_distribution<int> lengths(max_sound_length/10, max_sound_length);
  std::uniform_real_distribution<float> freqs(1e-3f, 0.3f);
  std::uniform_real_distribution<float> ampls(0.1f, 1.0f);
  for (int i = 0; i < length; i++)
    out[i] = noise(rng);
  for (int i = 0; i < num_sounds; i++) {
    int l = lengths(rng);
    int pos = std::uniform_int_distribution<int>(0, length - l)(rng);
    GenerateTestSound(out + pos, l, freqs(rng), ampls(rng));
  }
}

template <typename Params>
class StftImplGPUTest;

template <typename OutputType, FftSpectrumType spectrum_type, bool time_major>
struct StftTestParams {};

template <typename OutputType, FftSpectrumType spectrum_type, bool time_major>
class StftImplGPUTest<StftTestParams<OutputType, spectrum_type, time_major>>
: public ::testing::Test {
 public:
  void Run() {
    std::mt19937_64 rng(1234);

    StftImplGPU stft;
    TestTensorList<float, 1> in;
    TestTensorList<float, 1> window;
    TensorListShape<1> in_shape = {{ 1000, 15, 35321, 2048, 11111, 20480 }};
    const int N = in_shape.num_samples();
    in.reshape(in_shape);
    const auto lengths = make_cspan(in_shape.shapes);
    TensorListView<StorageCPU, float, 1> in_cpu = in.cpu();
    for (int i = 0; i < N; i++) {
      GenerateTestWave(rng, in_cpu.data[i], lengths[i], 30, lengths[i] / 5);
    }

    StftArgs args;
    args.axis = 0;
    args.spectrum_type = spectrum_type;
    args.window_length = 128;
    args.window_center = 64;
    args.window_step = 64;
    args.padding = signal::Padding::Reflect;

    window.reshape({{TensorShape<1>{args.window_length}}});
    auto window_span = make_span(window.cpu().data[0], args.window_length);
    HannWindow(window_span);

    args.time_major_layout = time_major;
    TestTensorList<OutputType, 2> out;

    KernelContext ctx;
    KernelRequirements req = stft.Setup(ctx, make_span(lengths), args);
    ASSERT_EQ(req.output_shapes.size(), 1u);
    auto &out_shape = req.output_shapes[0];
    ASSERT_EQ(out_shape.num_samples(), N);

    for (int i = 0; i < N; i++) {
      TensorShape<2> ts = { args.num_windows(lengths[i]), args.window_length / 2 + 1 };
      if (!time_major)
        std::swap(ts[0], ts[1]);
      EXPECT_EQ(out_shape[i], ts);
    }

    ScratchpadAllocator sa;
    sa.Reserve(req.scratch_sizes);
    auto scratchpad = sa.GetScratchpad();
    ctx.scratchpad = &scratchpad;
    auto window_gpu = window.gpu()[0];
    out.reshape(convert_dim<2>(out_shape));
    stft.Run(ctx, out.gpu(), in.gpu(), window_gpu);
    TensorListView<StorageCPU, OutputType, 2> out_cpu = out.cpu();

    TestTensorList<OutputType, 2> ref;
    RefSpectrum(ref, in_cpu, args, window_span);
    if (spectrum_type == FFT_SPECTRUM_COMPLEX)
      Check(out_cpu, ref.cpu(), EqualEps(1e-5));
    else
      Check(out_cpu, ref.cpu(), EqualRelative(1e-3));
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

TYPED_TEST_SUITE(StftImplGPUTest, StftTypes);


TYPED_TEST(StftImplGPUTest, RunBatch) {
  this->Run();
}


}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
