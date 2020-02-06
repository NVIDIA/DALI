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

#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include "dali/kernels/signal/fft/stft_gpu_impl.cuh"
#include "dali/test/test_tensors.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/signal/window/window_functions.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

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

void GenerateTestWave(float *out, int length, int num_sounds, int max_sound_length,
                      float noise_level = 0.01f) {
  std::mt19937_64 rng(1234);
  std::normal_distribution<float> noise(0, noise_level);
  std::uniform_int_distribution<int> lengths(4, max_sound_length);
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

TEST(StftImplGPU, BatchOfOne) {
  StftImplGPU stft;
  TestTensorList<float, 1> in;
  TestTensorList<complexf, 2> out;
  TestTensorList<float, 1> window;
  TensorListShape<1> in_shape({ TensorShape<1>{35500} });
  in.reshape(in_shape);
  auto lengths = make_cspan(in_shape.shapes);
  int N = in_shape.num_samples();
  for (int i = 0; i < N; i++) {
    GenerateTestWave(in.cpu().data[i], lengths[i], 30, lengths[i] / 5);
  }

  {
    std::ofstream stft_in("stft_in.txt");
    for (auto x : make_span(in.cpu().data[0], lengths[0])) {
      stft_in << x << "\n";
    }
  }

  StftArgs args;
  args.axis = 0;
  args.spectrum_type = FFT_SPECTRUM_COMPLEX;
  args.window_length = 128;
  args.window_center = 64;
  args.window_step = 64;
  args.padding = signal::Padding::Reflect;

  window.reshape({{TensorShape<1>{args.window_length}}});
  HannWindow(make_span(window.cpu().data[0], args.window_length));


  for (bool time_major : { true }) {
    args.time_major_layout = time_major;

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
    out.reshape(convert_dim<2>(out_shape));
    stft.RunR2C(ctx, out.gpu(), in.gpu(), window.gpu()[0]);
    TensorListView<StorageCPU, complexf, 2> out_cpu = out.cpu();

    TensorView<StorageCPU, complexf, 2> tv = out_cpu[0];

    int time_axis = time_major ? 0 : 1;
    int freq_axis = 1 - time_axis;
    std::ofstream stft_out("stft_out.txt");
    stft_out << "[";
    for (int f = 0; f < tv.shape[freq_axis]; f++) {
      if (f)
        stft_out << ",\n";
      stft_out << "[";
      for (int t = 0; t < tv.shape[time_axis]; t++) {
        int i = time_major ? t : f;
        int j = time_major ? f : t;
        if (t)
          stft_out << ", ";
        stft_out << *tv(i, j);
      }
      stft_out << "]";
    }
    stft_out << "]";
  }
}


}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
