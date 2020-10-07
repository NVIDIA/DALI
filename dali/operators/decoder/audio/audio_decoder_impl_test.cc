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
#include <utility>
#include "dali/operators/decoder/audio/audio_decoder_impl.h"

namespace dali {
namespace test {

TEST(AudioDecoderImpl, ProcessOffsetAndLength) {
  int64_t start = 0;
  int64_t total_length = static_cast<int64_t>(3.4 * 16000);
  AudioMetadata meta{total_length, 16000, 1};  // 3.40s at 16kHz
  {
    int64_t offset = static_cast<int64_t>(0.5 * meta.sample_rate);
    int64_t length = static_cast<int64_t>(1.0 * meta.sample_rate);
    ASSERT_EQ(std::make_pair(offset, length), ProcessOffsetAndLength(meta, 0.5, 1.0));
  }
  {
    ASSERT_EQ(std::make_pair(start, total_length), ProcessOffsetAndLength(meta, -1.0, 5.0));
    ASSERT_EQ(std::make_pair(start, total_length), ProcessOffsetAndLength(meta, -1.0, -1.0));
  }

  {
    int64_t offset = static_cast<int64_t>(2.0 * meta.sample_rate);
    int64_t duration = total_length - offset;
    ASSERT_EQ(std::make_pair(offset, duration), ProcessOffsetAndLength(meta, 2.0, 3.4));
  }

  {
    int64_t offset = 0;
    int64_t duration = static_cast<int64_t>(2.4 * meta.sample_rate);
    ASSERT_EQ(std::make_pair(offset, duration), ProcessOffsetAndLength(meta, -1.0, 2.4));
  }

  {
    int64_t offset = static_cast<int64_t>(0.45 * meta.sample_rate);
    int64_t duration = total_length - offset;
    ASSERT_EQ(std::make_pair(offset, duration), ProcessOffsetAndLength(meta, 0.45, -1.0));
  }
}

}  // namespace test
}  // namespace dali
