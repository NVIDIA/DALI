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
void TestWave(T *out, int n, int stride, float freq) {
  for (int i = 0; i < n; i++) {
    float f = std::sin(i* freq) * HannWindow(i, n);
    out[i*stride] = ConvertSatNorm<T>(f);
  }
}

class ResamplingTest : public ::testing::Test {
 public:
  void PrepareData(int nsamples, int nchannels,
                   span<const float> in_rates, span<const float> out_rates, int nsec = 1);

  virtual float eps() const { return 2e-3; }
  virtual float max_avg_err() const { return 1e-3; }
  void Verify();

  virtual void RunResampling(span<const float> in_rates, span<const float> out_rates) = 0;

  void RunTest(int nsamples, int nchannels);


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
