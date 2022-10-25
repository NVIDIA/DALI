// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/resampling_cpu.h"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "dali/core/cuda_error.h"
#include "dali/kernels/signal/resampling_test.h"

namespace dali {
namespace kernels {
namespace signal {
namespace resampling {
namespace test {

double HannWindow(int i, int n) {
  assert(n > 0);
  return Hann(2.0 * i / n - 1);
}

void ResamplingTest::PrepareData(span<const Args> args) {
  TensorListShape<> in_sh(nsamples_, nchannels_ > 1 ? 2 : 1);
  TensorListShape<> out_sh(nsamples_, nchannels_ > 1 ? 2 : 1);
  for (int s = 0; s < nsamples_; s++) {
    double in_rate = args[s].in_rate;
    double out_rate = args[s].out_rate;
    double scale = in_rate / out_rate;
    int n_in = nsec_ * in_rate + 12345 * s;  // different lengths
    int n_out = std::ceil(n_in / scale);
    int64_t out_begin = args[s].out_begin > 0 ? args[s].out_begin : 0;
    int64_t out_end = args[s].out_end > 0 ? args[s].out_end : n_out;
    ASSERT_GT(out_end, out_begin);
    in_sh.tensor_shape_span(s)[0] = n_in;
    out_sh.tensor_shape_span(s)[0] = out_end - out_begin;
    if (nchannels_ > 1) {
      in_sh.tensor_shape_span(s)[1] = nchannels_;
      out_sh.tensor_shape_span(s)[1] = nchannels_;
    }
  }
  ttl_in_.reshape(in_sh);
  ttl_out_.reshape(out_sh);
  ttl_outref_.reshape(out_sh);
  for (int s = 0; s < nsamples_; s++) {
    double in_rate = args[s].in_rate;
    double out_rate = args[s].out_rate;
    double scale = in_rate / out_rate;
    int64_t n_in = in_sh.tensor_shape_span(s)[0];
    int n_out = std::ceil(n_in / scale);
    int64_t out_begin = args[s].out_begin > 0 ? args[s].out_begin : 0;
    int64_t out_end = args[s].out_end > 0 ? args[s].out_end : n_out;
    for (int c = 0; c < nchannels_; c++) {
      double f_in = default_freq_in_ + 0.01 * s + 0.001 * c;
      double f_out = f_in * scale;
      // enough input samples for a given output region
      int64_t in_begin = std::max<int64_t>(out_begin * scale - 200, 0);
      int64_t in_end = std::min<int64_t>(out_end * scale + 200, n_in);
      TestWave(ttl_in_.cpu()[s].data + in_begin * nchannels_ + c, n_in, nchannels_, f_in,
               use_envelope_, in_begin, in_end);
      TestWave(ttl_outref_.cpu()[s].data + c, n_out, nchannels_, f_out, use_envelope_, out_begin,
               out_end);
    }
  }
}

void ResamplingTest::Verify(span<const Args> args) {
  auto in_sh = ttl_in_.cpu().shape;
  auto out_sh = ttl_outref_.cpu().shape;
  int nsamples = in_sh.num_samples();
  double err = 0, max_diff = 0;

  for (int s = 0; s < nsamples; s++) {
    float *out_data = ttl_out_.cpu()[s].data;
    float *out_ref = ttl_outref_.cpu()[s].data;
    int64_t out_len = out_sh.tensor_shape_span(s)[0];
    int nchannels = out_sh.sample_dim() == 1 ? 1 : out_sh.tensor_shape_span(s)[1];
    for (int64_t i = 0; i < out_len; i++) {
      ASSERT_NEAR(out_data[i], out_ref[i], eps_)
          << "Sample error too big @ sample=" << s << " pos=" << i << std::endl;
      float diff = std::abs(out_data[i] - out_ref[i]);
      if (diff > max_diff)
        max_diff = diff;
      err += diff * diff;
    }

    err = std::sqrt(err / out_len);
    EXPECT_LE(err, max_avg_err_) << "Average error too big";
    std::cerr << "Resampling with Hann-windowed sinc filter and 16 zero crossings"
                 "\n  max difference vs fresh signal: "
              << max_diff << "\n  RMS error: " << err << std::endl;
  }
}

void ResamplingTest::RunTest() {
  std::vector<Args> args_v;
  for (int i = 0; i < nsamples_; i++) {
    if (i % 2 == 0)
      args_v.push_back({22050.0f, 16000.0f, roi_start_, roi_end_});
    else
      args_v.push_back({44100.0f, 16000.0f, roi_start_, roi_end_});
  }
  auto args = make_cspan(args_v);

  PrepareData(args);

  RunResampling(args);

  Verify(args);
}

class ResamplingCPUTest : public ResamplingTest {
 public:
  void RunResampling(span<const Args> args) override {
    ResamplerCPU R;
    R.Initialize(16);

    ASSERT_EQ(args.size(), nsamples_);

    auto in_view = ttl_in_.cpu();
    auto out_view = ttl_out_.cpu();
    for (int s = 0; s < nsamples_; s++) {
      auto out_sh = out_view.shape[s];
      auto in_sh = in_view.shape[s];
      int n_in = in_sh[0];
      int n_out = resampled_length(n_in, args[s].in_rate, args[s].out_rate);
      int nchannels = in_sh.sample_dim() > 1 ? in_sh[1] : 1;
      int64_t out_begin = args[s].out_begin > 0 ? args[s].out_begin : 0;
      int64_t out_end = args[s].out_end > 0 ? args[s].out_end : n_out;
      ASSERT_EQ(out_sh[0], out_end - out_begin);
      R.Resample(out_view[s].data, out_begin, out_end, args[s].out_rate,
                 in_view[s].data, n_in, args[s].in_rate, nchannels);
    }
  }
};

TEST_F(ResamplingCPUTest, SingleChannel) {
  this->nchannels_ = 1;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, TwoChannel) {
  this->nchannels_ = 2;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, EightChannel) {
  this->nchannels_ = 8;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, ThirtyChannel) {
  this->nchannels_ = 30;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, OutBeginEnd) {
  this->roi_start_ = 100;
  this->roi_end_ = 8000;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, EightChannelOutBeginEnd) {
  this->roi_start_ = 100;
  this->roi_end_ = 8000;
  this->nchannels_ = 8;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, SingleChannelNeedHighPrecision) {
  this->default_freq_in_ = 0.49;
  this->nsec_ = 400;
  this->roi_start_ = 4000000;  // enough to look at the tail
  this->roi_end_ = -1;
  this->RunTest();
}

TEST_F(ResamplingCPUTest, ThreeChannelNeedHighPrecision) {
  this->default_freq_in_ = 0.49;
  this->nsec_ = 400;
  this->nchannels_ = 3;
  this->roi_start_ = 4000000;  // enough to look at the tail
  this->roi_end_ = -1;
  this->RunTest();
}

}  // namespace test
}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
