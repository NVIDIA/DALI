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
#include <random>
#include <algorithm>
#include "dali/operators/random/normal_distribution_op.cuh"

namespace dali {

class TestBlockDistribution : public ::testing::Test{
 protected:
  std::vector<int> distribute(const std::vector<int> &sizes) {
    TensorListShape<1> shape(sizes.size());
    for (int s = 0; s < shape.size(); ++s) shape.set_tensor_shape(s, {sizes[s]});
    return detail::DistributeBlocksPerSample(shape, block_size_, max_blocks_).first;
  }

  const int block_size_ = 256;
  const int max_blocks_ = 1024;
};

TEST_F(TestBlockDistribution, Uniform) {
  ASSERT_EQ(distribute(std::vector<int>(16, 70 * 256)), std::vector<int>(16, 64));
}

TEST_F(TestBlockDistribution, With_1) {
  ASSERT_EQ(distribute({128, 1, 1024 * 256}), (std::vector<int>{1, 1, 1022}));
}

TEST_F(TestBlockDistribution, Ratio) {
  std::vector<int> shape{1024 * 256, 2 * 1024 * 256, 5 * 1024 * 256};
  std::vector<int> dist{1 * 1024 / 8, 2 * 1024 / 8, 5 * 1024 / 8};
  ASSERT_EQ(distribute(shape), dist);
}

TEST_F(TestBlockDistribution, LessThanMax) {
  std::vector<int> shape{511 * 256 + 100, 128 * 256, 256 * 256};
  std::vector<int> dist{512, 128, 256};
  ASSERT_EQ(distribute(shape), dist);
}

TEST(DistributeBlocks, RandomTests) {
  std::mt19937 rnd(1234);
  const int block_size = 256;
  const int max_blocks = 1024;
  std::uniform_int_distribution<int> size_dist(32 * block_size, 2048 * block_size);
  TensorListShape<1> shape(31);
  for (int i = 0; i < 100; ++i) {
    for (int s = 0; s < shape.size(); ++s) shape.set_tensor_shape(s, {size_dist(rnd)});
    std::vector<int> blocks;
    int sum;
    std::tie(blocks, sum) =
      detail::DistributeBlocksPerSample(shape, block_size, max_blocks);
    EXPECT_EQ(std::accumulate(blocks.begin(), blocks.end(), 0), sum);
    EXPECT_EQ(sum, max_blocks);
  }
}

}  // namespace dali
