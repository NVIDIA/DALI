// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_TEST_H_
#define DALI_KERNELS_SIGNAL_RESAMPLING_TEST_H_

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "dali/kernels/signal/resampling.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace signal {
namespace resampling {
namespace test {

double HannWindow(int i, int n);

template <typename T>
void TestWave(T *out, int n, int stride, double freq, bool envelope = true, int i_start = 0,
              int i_end = -1) {
  if (i_end <= 0)
    i_end = n;
  assert(i_start >= 0 && i_start <= n);
  assert(i_end >= 0 && i_end <= n);
  for (int i = i_start; i < i_end; i++) {
    float f;
    if (envelope)
      f = std::sin(i * freq) * HannWindow(i, n);
    else
      f = std::sin(i * freq);
    out[(i - i_start) * stride] = ConvertSatNorm<T>(f);
  }
}

class ResamplingTest : public ::testing::Test {
 public:
  void PrepareData(span<const Args> args);

  virtual float max_avg_err() const { return 1e-3; }
  void Verify(span<const Args> args);

  virtual void RunResampling(span<const Args> args) = 0;

  void RunTest();

 protected:
  int nsamples_ = 1;
  int nchannels_ = 1;
  double default_freq_in_ = 0.1;
  int nsec_ = 1;
  float eps_ = 2e-3;
  float max_avg_err_ = 1e-3;
  bool use_envelope_ = true;
  int64_t roi_start_ = 0;
  int64_t roi_end_ = -1;  // means end-of-signal

  TestTensorList<float> ttl_in_;
  TestTensorList<float> ttl_out_;
  TestTensorList<float> ttl_outref_;
};


}  // namespace test
}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_TEST_H_
