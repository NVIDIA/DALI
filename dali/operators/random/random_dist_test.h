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

#ifndef DALI_OPERATORS_RANDOM_RANDOM_DIST_TEST_H_
#define DALI_OPERATORS_RANDOM_RANDOM_DIST_TEST_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include "dali/operators/random/random_dist.h"
#include "dali/core/span.h"

namespace dali {
namespace random {
namespace test {

namespace {
constexpr int kSampleSize = 100000;
constexpr double kBinTolerance = 0.01;
constexpr double kCDFTolerance = 0.01;

template <typename T>
std::vector<int> ComputeHistogram(span<const T> samples, T min, T max, int nbins) {
    std::vector<int> hist(nbins, 0);
    double bin_width = static_cast<double>(max - min) / nbins;
    for (auto v : samples) {
    int bin = static_cast<int>((v - min) / bin_width);
    if (bin < 0)
        bin = 0;
    if (bin >= nbins)
        bin = nbins - 1;
    hist[bin]++;
    }
    return hist;
}

template <typename T>
std::vector<int> ComputeHistogram(span<const T> samples, span<const T> bin_edges) {
    std::vector<int> hist(bin_edges.size() + 1, 0);
    for (auto v : samples) {
    int bin = std::lower_bound(bin_edges.begin(), bin_edges.end(), v) - bin_edges.begin();
    hist[bin]++;
    }
    return hist;
}

void CompareHistograms(
    const std::vector<int> &a,
    const std::vector<int> &b,
    double bin_tol_frac = kBinTolerance,
    double cdf_tol_frac = kCDFTolerance) {
    int n = static_cast<int>(a.size());
    int total_a = std::accumulate(a.begin(), a.end(), 0);
    int total_b = std::accumulate(b.begin(), b.end(), 0);
    int partial_sum_a = 0, partial_sum_b = 0;
    double max_bin_diff = 0, max_cdf_diff = 0;
    for (int i = 0; i < n; ++i) {
    partial_sum_a += a[i];
    partial_sum_b += b[i];
    double fa = static_cast<double>(partial_sum_a) / total_a;
    double fb = static_cast<double>(partial_sum_b) / total_b;
    max_cdf_diff = std::max(max_cdf_diff, std::abs(fa - fb));
    EXPECT_NEAR(fa, fb, cdf_tol_frac)
        << "CDF mismatch at bin " << i << ": " << fa << " vs " << fb;
    fa = static_cast<double>(a[i]) / total_a;
    fb = static_cast<double>(b[i]) / total_b;
    max_bin_diff = std::max(max_bin_diff, std::abs(fa - fb));
    EXPECT_NEAR(fa, fb, bin_tol_frac)
        << "Histogram mismatch at bin " << i << ": " << fa << " vs " << fb;
    }
    std::cout << "Max bin diff: " << max_bin_diff << std::endl;
    std::cout << "Max cdf diff: " << max_cdf_diff << std::endl;
}

}  // namespace

}  // namespace test
}  // namespace random
}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RANDOM_DIST_TEST_H_
