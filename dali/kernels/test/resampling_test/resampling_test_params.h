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

#ifndef DALI_KERNELS_TEST_RESAMPLING_TEST_RESAMPLING_TEST_PARAMS_H_
#define DALI_KERNELS_TEST_RESAMPLING_TEST_RESAMPLING_TEST_PARAMS_H_

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "dali/kernels/imgproc/resample/params.h"

namespace dali {
namespace kernels {
namespace resample_test {

constexpr FilterDesc nearest(float radius = 0) {
  return { ResamplingFilterType::Nearest, radius };
}

constexpr FilterDesc tri(float radius = 0) {
  return { ResamplingFilterType::Triangular, radius };
}

constexpr FilterDesc lin() {
  return { ResamplingFilterType::Linear, 0 };
}

constexpr FilterDesc lanczos() {
  return { ResamplingFilterType::Lanczos3, 0 };
}

constexpr FilterDesc cubic() {
  return { ResamplingFilterType::Cubic, 0 };
}

constexpr FilterDesc gauss(float radius) {
  return { ResamplingFilterType::Gaussian, radius };
}

struct ResamplingTestEntry {
  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<int, 2> sizeWH,
                      FilterDesc filter,
                      double epsilon = 1)
    : ResamplingTestEntry(std::move(input)
    , std::move(reference), sizeWH, filter, filter, epsilon) {}

  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<int, 2> sizeWH,
                      FilterDesc fx,
                      FilterDesc fy,
                      double epsilon = 1)
    : input(std::move(input)), reference(std::move(reference)), epsilon(epsilon) {
    params[0].output_size = sizeWH[1];
    params[1].output_size = sizeWH[0];
    params[0].mag_filter = params[0].min_filter = fy;
    params[1].mag_filter = params[1].min_filter = fx;
  }

  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<float, 4> ROI_LTRB,
                      std::array<int, 2> sizeWH,
                      FilterDesc filter,
                      double epsilon = 1)
    : ResamplingTestEntry(
        std::move(input), std::move(reference),
        ROI_LTRB, sizeWH, filter, filter, epsilon) {}

  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<float, 4> ROI_LTRB,
                      std::array<int, 2> sizeWH,
                      FilterDesc fx,
                      FilterDesc fy,
                      double epsilon = 1)
    : input(std::move(input)), reference(std::move(reference)), epsilon(epsilon) {
    params[0].output_size = sizeWH[1];
    params[1].output_size = sizeWH[0];
    params[0].roi = { ROI_LTRB[1], ROI_LTRB[3] };
    params[0].mag_filter = params[0].min_filter = fy;
    params[1].mag_filter = params[1].min_filter = fx;
    params[1].roi = { ROI_LTRB[0], ROI_LTRB[2] };
  }

  std::string input, reference;
  ResamplingParams2D params;
  double epsilon = 1;
};


using ResamplingTestBatch = std::vector<ResamplingTestEntry>;

inline std::ostream &operator<<(std::ostream &os, const FilterDesc fd) {
  os << FilterName(fd.type);
  if (fd.radius) {
    switch (fd.type) {
    case ResamplingFilterType::Gaussian:
    case ResamplingFilterType::Triangular:
      os << "(r = " << fd.radius << ")";
      break;
    case ResamplingFilterType::Cubic:
      if (fd.radius != 4) os << "(custom radius: " << fd.radius << ")";
      break;
    case ResamplingFilterType::Lanczos3:
      if (fd.radius != 3) os << "(custom radius: " << fd.radius << ")";
      break;
    default:
      break;
    }
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ResamplingParams2D &params) {
  os  << "  Horizontal " << params[1].output_size << " px; "
      << " mag = " << params[1].mag_filter << " min = " << params[1].min_filter << "\n"
      << "  Vertical   " << params[0].output_size << " px; "
      << " mag = " << params[0].mag_filter << " min = " << params[0].min_filter << "\n";
  return os;
}

inline void PrintTo(const ResamplingTestEntry &entry, std::ostream *os) {
  *os << "Input: " << entry.input << "   ref:" << entry.reference << "\n  params:\n"
      << entry.params << "  Eps = " << entry.epsilon;
}

inline void PrintTo(const ResamplingTestBatch &batch, std::ostream *os) {
  *os << "{\n";
  bool first = true;
  for (auto &entry : batch) {
    if (first) { first = false;
    } else { *os << ",\n"; }
    PrintTo(entry, os);
  }
  *os << "\n}\n";
}

}  // namespace resample_test
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TEST_RESAMPLING_TEST_RESAMPLING_TEST_PARAMS_H_
