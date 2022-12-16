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

#include <gtest/gtest.h>
#include <vector>
#include "dali/operators/input/video_input.h"

namespace dali {
namespace test {

namespace {
struct BatchOutlineTestData {
  int num_frames, frames_per_sequence, batch_size;
  int num_full_batches, num_full_sequences, frames_in_last_sequence;
};
std::vector<BatchOutlineTestData> test_data{
        {60,  7,  1,  8, 0, 4},
        {40,  7,  3,  1, 2, 5},
        {100, 50, 2,  1, 0, 0},
        {100, 49, 2,  1, 0, 2},
        {1,   10, 10, 0, 0, 1},
        {0,   1,  1,  0, 0, 0},
        {1,   1,  1,  1, 0, 0},
        {8,   2,  3,  1, 1, 0},
};


std::ostream &operator<<(std::ostream &os, const BatchOutlineTestData &td) {
  return os << "{" << td.num_frames << ", " << td.frames_per_sequence << ", " << td.batch_size
            << ", " << td.num_full_batches << ", " << td.num_full_sequences << ", "
            << td.frames_in_last_sequence << "}";
}
}  // namespace


TEST(VideoInputTest, DetermineBatchOutlineTest) {
  for (auto &td : test_data) {
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
            detail::DetermineBatchOutline(td.num_frames, td.frames_per_sequence, td.batch_size);
    EXPECT_EQ(num_full_batches, td.num_full_batches) << "Error in data: " << td;
    EXPECT_EQ(num_full_sequences, td.num_full_sequences) << "Error in data: " << td;
    EXPECT_EQ(frames_in_last_sequence, td.frames_in_last_sequence) << "Error in data: " << td;
  }
}

}  // namespace test
}  // namespace dali
