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

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "dali/kernels/signal/resampling.h"
#include "dali/kernels/signal/resampling_test.h"

namespace dali {
namespace kernels {
namespace signal {
namespace resampling {

void ResamplingTest::PrepareData(int nsamples, int nchannels, span<const float> in_rates,
                                 span<const float> out_rates) {
  TensorListShape<> in_sh(nsamples, nchannels > 1 ? 2 : 1);
  TensorListShape<> out_sh(nsamples, nchannels > 1 ? 2 : 1);
  for (int s = 0; s < nsamples; s++) {
    double in_rate = in_rates[s];
    double out_rate = out_rates[s];
    double scale = static_cast<double>(in_rate) / out_rate;
    int n_in = in_rate + 12345 * s;  // different lengths
    int n_out = std::ceil(n_in / scale);
    in_sh.tensor_shape_span(s)[0] = n_in;
    out_sh.tensor_shape_span(s)[0] = n_out;
    if (nchannels > 1) {
      in_sh.tensor_shape_span(s)[1] = nchannels;
      out_sh.tensor_shape_span(s)[1] = nchannels;
    }
  }
  ttl_in_.reshape(in_sh);
  ttl_out_.reshape(out_sh);
  ttl_outref_.reshape(out_sh);
  for (int s = 0; s < nsamples; s++) {
    double in_rate = in_rates[s];
    double out_rate = out_rates[s];
    double scale = static_cast<double>(in_rate) / out_rate;
    for (int c = 0; c < nchannels; c++) {
      float f_in = 0.1f + 0.01 * s + 0.001 * c;
      float f_out = f_in * scale;
      int n_in = in_sh.tensor_shape_span(s)[0];
      int n_out = out_sh.tensor_shape_span(s)[0];
      TestWave(ttl_in_.cpu()[s].data + c, n_in, nchannels, f_in);
      TestWave(ttl_outref_.cpu()[s].data + c, n_out, nchannels, f_out);
    }
  }
}

void ResamplingTest::Verify(bool downmix) {
  auto in_sh = ttl_in_.cpu().shape;
  auto out_sh = ttl_outref_.cpu().shape;
  int nsamples = in_sh.num_samples();
  double err = 0, max_diff = 0;

  for (int s = 0; s < nsamples; s++) {
    float *out_data = ttl_out_.cpu()[s].data;
    float *out_ref = ttl_outref_.cpu()[s].data;
    int n_out = out_sh.tensor_shape_span(s)[0];
    int nchannels = out_sh.sample_dim() == 1 ? 1 : out_sh.tensor_shape_span(s)[1];
    for (int i = 0; i < n_out; i++) {
      float ref_val = 0;
      if (downmix) {
        for (int c = 0; c < nchannels; c++) {
          ref_val += out_ref[i * nchannels + c];
        }
        ref_val /= nchannels;
      } else {
        ref_val = out_ref[i];
      }

      ASSERT_NEAR(out_data[i], ref_val, eps())
          << "Sample error too big @ sample=" << s << " pos=" << i << std::endl;
      float diff = std::abs(out_data[i] - ref_val);
      if (diff > max_diff)
        max_diff = diff;
      err += diff * diff;
    }

    err = std::sqrt(err / n_out);
    EXPECT_LE(err, max_avg_err()) << "Average error too big";
    std::cerr << "Resampling with Hann-windowed sinc filter and 16 zero crossings"
                 "\n  max difference vs fresh signal: "
              << max_diff << "\n  RMS error: " << err << std::endl;
  }
}

void ResamplingTest::RunTest(int nsamples, int nchannels, bool downmix) {
  std::vector<float> in_rates_v;
  for (int i = 0; i < nsamples; i++) {
    if (i % 2 == 0)
      in_rates_v.push_back(22050.0f);
    else
      in_rates_v.push_back(44100.0f);
  }
  auto in_rates = make_cspan(in_rates_v);

  std::vector<float> out_rates_v(nsamples, 16000.0f);
  auto out_rates = make_cspan(out_rates_v);

  PrepareData(nsamples, nchannels, in_rates, out_rates);

  RunResampling(in_rates, out_rates, downmix);

  Verify(downmix);
}

class ResamplingCPUTest : public ResamplingTest {
 public:
  void RunResampling(span<const float> in_rates, span<const float> out_rates, bool downmix) override {
    Resampler R;
    R.Initialize(16);

    int nsamples = in_rates.size();
    assert(nsamples == out_rates.size());

    auto in_view = ttl_in_.cpu();
    auto out_view = ttl_out_.cpu();
    for (int s = 0; s < nsamples; s++) {
      auto out_sh = out_view.shape[s];
      auto in_sh = in_view.shape[s];
      int n_out = out_sh[0];
      int n_in = in_sh[0];
      int nchannels = in_sh.sample_dim() > 1 ? in_sh[1] : 1;
      R.Resample(out_view[s].data, 0, n_out, out_rates[s], in_view[s].data, n_in, in_rates[s],
                 nchannels, downmix);
    }
  }
};

TEST_F(ResamplingCPUTest, SingleChannel) {
  this->RunTest(1, 1, false);
}

TEST_F(ResamplingCPUTest, TwoChannel) {
  this->RunTest(1, 2, false);
}

TEST_F(ResamplingCPUTest, EightChannel) {
  this->RunTest(1, 8, false);
}

TEST_F(ResamplingCPUTest, ThreeChannelDownmix) {
  this->RunTest(1, 3, true);
}

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
