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
#include "dali/operators/decoder/audio/utils.h"

namespace dali {

TEST(AudioDecoderTest, DownmixingTest) {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int nchannels = 3;
  std::vector<int> weights = {3, 2, 1};
  std::vector<float> ref = {1.67, 4.67, 7.67, 10.67};
  std::vector<float> out;
  out.resize(ref.size());

  Downmixing(out.data(), in.data(), in.size(), weights);

  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_FLOAT_EQ(out[i], ref[i]);
  }
}


TEST(AudioDecoderTest, DownmixingInPlaceTest) {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int nchannels = 3;
  std::vector<float> ref = {2, 5, 8, 11};

  Downmixing(in.data(), in.data(), in.size(), nchannels);

  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_FLOAT_EQ(in[i], ref[i]);
  }
}


TEST(AudioDecoderTest, DownmixingSpan) {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int nchannels = 3;
  std::vector<float> ref = {2, 5, 8, 11};

  Downmixing(make_span(in), make_cspan(in), nchannels);

  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_FLOAT_EQ(in[i], ref[i]);
  }
}
}  // namespace dali