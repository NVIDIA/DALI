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
#include "dali/kernels/signal/downmixing.h"

namespace dali {
namespace kernels {
namespace signal {

TEST(SignalDownmixingTest, RawPointer_Weighted) {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int nchannels = 3;
  std::vector<float> weights = {3, 2, 1};
  float sum = std::accumulate(weights.begin(), weights.end(), 0);
  std::vector<float> ref = {
     (1 * 3 +  2 * 2 +  3) / sum,
     (4 * 3 +  5 * 2 +  6) / sum,
     (7 * 3 +  8 * 2 +  9) / sum,
    (10 * 3 + 11 * 2 + 12) / sum
  };
  std::vector<float> out;
  out.resize(ref.size());

  Downmix(out.data(), in.data(), in.size() / nchannels, nchannels, weights.data(), true);

  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_FLOAT_EQ(out[i], ref[i]);
  }
}

TEST(SignalDownmixingTest, Span_DefaultWeights) {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int nchannels = 3;
  std::vector<float> ref = {2, 5, 8, 11};

  Downmix(make_span(in), make_cspan(in), nchannels);

  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_FLOAT_EQ(in[i], ref[i]);
  }
}

}  // namespace signal
}  // namespace kernels
}  // namespace dali
