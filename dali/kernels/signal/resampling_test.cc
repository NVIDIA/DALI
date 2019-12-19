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
#include <vector>
#include <numeric>
#include "dali/kernels/signal/resampling.h"

namespace dali {
namespace kernels {
namespace signal {
namespace resampling {

namespace {

double HannWindow(int i, int n) {
  assert(n > 0);
  return Hann(2.0*i / n - 1);
}

template <typename T>
void TestWave(T *out, int n, int stride, float freq) {
  for (int i = 0; i < n; i++) {
    float x = i * freq;
    float f = std::sin(i* freq) * HannWindow(i, n);
    out[i*stride] = ConvertSatNorm<T>(f);
  }
}

}  // namespace

TEST(ResampleSinc, SingleChannel) {
  int n_in = 22050, n_out = 16000;  // typical downsampling
  std::vector<float> in(n_in);
  std::vector<float> out(n_out);
  std::vector<float> ref(out.size());
  float f_in = 0.1f;
  float f_out = f_in * n_in / n_out;
  double in_rate = n_in;
  double out_rate = n_out;
  TestWave(in.data(), n_in, 1, f_in);
  TestWave(ref.data(), n_out, 1, f_out);
  Resampler R;
  R.Initialize(16);
  R.Resample(out.data(), 0, n_out, out_rate, in.data(), n_in, in_rate);

  double err = 0, max_diff = 0;
  for (int i = 0; i < n_out; i++) {
    ASSERT_NEAR(out[i], ref[i], 1e-3) << "Sample error too big @" << i << std::endl;
    float diff = std::abs(out[i] - ref[i]);
    if (diff > max_diff)
      max_diff = diff;
    err += diff*diff;
  }
  err = std::sqrt(err/n_out);
  EXPECT_LE(err, 1e-3) << "Average error too big";
  std::cerr << "Resampling with Hann-windowed sinc filter and 16 zero crossings"
    "\n  max difference vs fresh signal: " << max_diff <<
    "\n  RMS error: " << err << std::endl;
}

TEST(ResampleSinc, MultiChannel) {
  int n_in = 22050, n_out = 22053;  // some weird upsampling
  int ch = 5;
  std::vector<float> in(n_in * ch);
  std::vector<float> out(n_out * ch);
  std::vector<float> ref(out.size());
  double in_rate = n_in;
  double out_rate = n_out;
  for (int c = 0; c < ch; c++) {
    float f_in = 0.1f * (1 + c * 0.012345);  // different signal in each channel
    float f_out = f_in * n_in / n_out;
    TestWave(in.data() + c, n_in, ch, f_in);
    TestWave(ref.data() + c, n_out, ch, f_out);
  }
  Resampler R;
  R.Initialize(16);
  R.Resample(out.data(), 0, n_out, out_rate, in.data(), n_in, in_rate, ch);

  double err = 0, max_diff = 0;
  for (int i = 0; i < n_out * ch; i++) {
    ASSERT_NEAR(out[i], ref[i], 2e-3) << "Sample error too big @" << i << std::endl;
    float diff = std::abs(out[i] - ref[i]);
    if (diff > max_diff)
      max_diff = diff;
    err += diff*diff;
  }
  err = std::sqrt(err/(n_out * ch));
  EXPECT_LE(err, 1e-3) << "Average error too big";
  std::cerr << "Resampling with Hann-windowed sinc filter and 16 zero crossings"
    "\n  max difference vs fresh signal: " << max_diff <<
    "\n  RMS error: " << err << std::endl;
}

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
