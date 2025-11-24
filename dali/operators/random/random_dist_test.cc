// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/random/random_dist.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include "dali/operators/random/philox.h"
#include "dali/core/int_literals.h"
#include "dali/core/span.h"
#include "dali/operators/random/random_dist_test.h"

namespace dali {
namespace random {
namespace test {

struct ConstantGenerator {
  uint32_t value;
  uint32_t operator()() const {
    return value;
  }
};

/** Check the mapping of the 32-bit RNG output to the desired integer range */
TEST(RandomDistTest, UniformIntDistMapping) {
  using dist_t = uniform_int_dist<uint32_t>;
  dist_t dist(0, 1);
  uint64_t full_range = 0x100000000_u64;
  ConstantGenerator gen;
  // Power-of-two range - use lower bits
  gen.value = 0;
  EXPECT_EQ(dist(gen), 0);
  gen.value = 1;
  EXPECT_EQ(dist(gen), 1);
  gen.value = 2;
  EXPECT_EQ(dist(gen), 0);
  gen.value = 0xfffffffd;
  EXPECT_EQ(dist(gen), 1);
  gen.value = 0xfffffffe;
  EXPECT_EQ(dist(gen), 0);
  gen.value = 0xffffffff;
  EXPECT_EQ(dist(gen), 1);

  // Non-power-of-two range - use range division
  dist = dist_t(0, 2);
  gen.value = 0;
  EXPECT_EQ(dist(gen), 0);
  gen.value = 1 * full_range / 3;
  EXPECT_EQ(dist(gen), 0);
  gen.value++;
  EXPECT_EQ(dist(gen), 1);
  gen.value = 2 * full_range / 3;
  EXPECT_EQ(dist(gen), 1);
  gen.value++;
  EXPECT_EQ(dist(gen), 2);

  // Non-power-of-two range - use range division
  dist = dist_t(0, 31337, true);
  for (uint32_t i = 0; i < 31337; i++) {
    gen.value = full_range * i / 31337 + 1;
    EXPECT_EQ(dist(gen), i);
    if (i > 0) {
      gen.value--;
      EXPECT_EQ(dist(gen), i - 1);
    }
  }
}

TEST(RandomDistTest, UniformIntDistMappingFullRange) {
  uniform_int_dist<uint32_t> udist(0, 0xffffffff_u32);
  uniform_int_dist<int32_t> sdist(0x80000000_i32, 0x7fffffff_i32);
  uint32_t values[] = {
    0u,
    1u,
    0x7ffffffeu,
    0x7fffffffu,
    0x80000000u,
    0x80000001u,
    0xfffffffeu,
    0xffffffffu
  };

  ConstantGenerator gen;
  for (uint32_t x : values) {
    gen.value = x;
    EXPECT_EQ(udist(gen), x);
    EXPECT_EQ(sdist(gen), int32_t(x - 0x80000000_u32));
  }
}

TEST(RandomDistTest, UniformRealDistMapping) {
  using dist_t = uniform_real_dist<float>;
  std::vector<std::pair<float, float>> values = {
    {-1.0f, 1.0f},
    {0.0f, 1.0f},
    {-1.0f, 0.0f},
    {1e+4f, 1e+8f},
  };
  for (auto [lo, hi] : values) {
    dist_t dist(lo, hi);
    ConstantGenerator gen;
    gen.value = 0;
    float tol = (std::abs(hi) + std::abs(lo)) * 0x1p-24f;
    EXPECT_GE(dist(gen), lo);  // inclusive bound
    EXPECT_NEAR(dist(gen), lo, tol);
    gen.value = 0xffffffffu;
    EXPECT_LT(dist(gen), hi);  // exclusive bound
    EXPECT_NEAR(dist(gen), std::nextafter(hi, lo), tol);
    gen.value = 0x7fffffffu;
    EXPECT_NEAR(dist(gen), (lo + hi) / 2, tol);
  }
}


TEST(RandomDistTest, UniformIntDistHistogramComparisonWithStdUniform) {
  using T = int32_t;
  T low = 10, high = 20;
  dali::random::uniform_int_dist<T> dali_dist(low, high);
  std::uniform_int_distribution<T> std_dist(low, high);

  Philox4x32_10 philox;
  philox.init(2024, 99, 123);
  std::mt19937 std_prng(2024);

  std::vector<T> dali_samples, std_samples;
  dali_samples.reserve(kSampleSize);
  std_samples.reserve(kSampleSize);

  for (int i = 0; i < kSampleSize; ++i) {
    dali_samples.push_back(dali_dist(philox));
    std_samples.push_back(std_dist(std_prng));
  }

  int nbins = high - low + 1;
  std::vector<int> dali_hist(nbins, 0), std_hist(nbins, 0);
  for (int i = 0; i < kSampleSize; ++i) {
    dali_hist[dali_samples[i] - low]++;
    std_hist[std_samples[i] - low]++;
  }
  CompareHistograms(dali_hist, std_hist);
}

TEST(RandomDistTest, UniformDiscreteDistHistogramComparisonWithStdDiscrete) {
  std::vector<int> values{3, 7, 11, 15};
  dali::random::uniform_discrete_dist<int> dali_dist(values.data(), values.size());
  // std::discrete_distribution with equal weights for uniformity
  std::discrete_distribution<> std_dist(values.size(), 0.0, 1.0, [](double) { return 1.0; });

  Philox4x32_10 philox;
  philox.init(100, 101, 102);
  std::mt19937 std_prng(100);

  std::vector<int> dali_hist(values.size(), 0);
  std::vector<int> std_hist(values.size(), 0);

  for (int i = 0; i < kSampleSize; ++i) {
    int v1 = dali_dist(philox);
    int idx1 = std::distance(values.begin(), std::find(values.begin(), values.end(), v1));
    ASSERT_LT(idx1, static_cast<int>(values.size()));
    dali_hist[idx1]++;

    int idx2 = std_dist(std_prng);
    std_hist[idx2]++;
  }
  CompareHistograms(dali_hist, std_hist);
}

template <typename T>
class RandomDistFPTest : public ::testing::Test {};

using FPTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RandomDistFPTest, FPTypes);


TYPED_TEST(RandomDistFPTest, UniformRealDistHistogramComparisonWithStdUniform) {
  using T = TypeParam;
  T low = -2.0f, high = 3.0f;
  dali::random::uniform_real_dist<T> dali_dist(low, high);
  std::uniform_real_distribution<T> std_dist(low, high);

  Philox4x32_10 philox;
  philox.init(1337, 42, 0);
  std::mt19937 std_prng(1337);

  std::vector<T> dali_samples, std_samples;
  dali_samples.reserve(kSampleSize);
  std_samples.reserve(kSampleSize);

  for (int i = 0; i < kSampleSize; ++i) {
    dali_samples.push_back(dali_dist(philox));
    std_samples.push_back(std_dist(std_prng));
  }

  constexpr int nbins = 20;
  auto dali_hist = ComputeHistogram(make_cspan(dali_samples), low, high, nbins);
  auto std_hist  = ComputeHistogram(make_cspan(std_samples), low, high, nbins);

  CompareHistograms(dali_hist, std_hist);
}

TYPED_TEST(RandomDistFPTest, NormalDistHistogramComparisonWithStdNormal) {
  using T = TypeParam;
  const int kSampleSize = 200000;
  const T mean = 2.5f;
  const T stddev = 1.2f;

  dali::random::normal_dist<T> dali_dist(mean, stddev);
  std::normal_distribution<T> std_dist(mean, stddev);

  Philox4x32_10 philox;
  philox.init(777, 888, 999);
  std::mt19937 std_prng(777);

  std::vector<T> dali_samples, std_samples;
  dali_samples.resize(kSampleSize);
  std_samples.resize(kSampleSize);

  for (int i = 0; i < kSampleSize; ++i) {
    dali_samples[i] = dali_dist(philox);
    std_samples[i] = std_dist(std_prng);
  }

  // Variable-width bins: finer near the mean, coarser in the tails
  std::vector<T> bin_edges = {
      mean - 4 * stddev,
      mean - 2 * stddev,
      mean - 1 * stddev,
      mean - 0.7f * stddev,
      mean - 0.4f * stddev,
      mean - 0.2f * stddev,
      mean,
      mean + 0.2f * stddev,
      mean + 0.4f * stddev,
      mean + 0.7f * stddev,
      mean + 1 * stddev,
      mean + 2 * stddev,
      mean + 4 * stddev
  };

  auto dali_hist = ComputeHistogram(make_cspan(dali_samples), make_cspan(bin_edges));
  auto std_hist  = ComputeHistogram(make_cspan(std_samples), make_cspan(bin_edges));

  CompareHistograms(dali_hist, std_hist);
}

TEST(RandomDistTest, PoissonWithPhilox) {
  std::poisson_distribution<int> std_dist(10);
  Philox4x32_10 philox;
  philox.init(100, 101, 102);
  std_dist(philox);
}

}  // namespace test
}  // namespace random
}  // namespace dali
