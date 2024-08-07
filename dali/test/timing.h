// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_TIMING_H_
#define DALI_TEST_TIMING_H_

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

namespace dali {
namespace test {

using perf_timer = std::chrono::high_resolution_clock;

inline std::ostream &print_time(std::ostream &os, double seconds) {
  if (seconds == 0) {
    os << "0";
  } else if (seconds < 1e-6) {
    os << seconds * 1e+9 << " ns";
  } else if (seconds < 1e-3) {
    os << seconds * 1e+6 << " µs";
  } else if (seconds < 1.0) {
    os << seconds * 1e+3 << " ms";
  } else if (seconds < 60.0) {
    os << seconds << " s";
  } else if (seconds < 3600.0) {
    int m = seconds / 60;
    double s = std::fmod(seconds, 60.0);
    char ss[32] = "00";
    // if `s` is <10, start at the second digit to add a leading zero
    snprintf(ss + (s < 10), sizeof(ss) - 1, "%g", s);
    char buf[64];
    snprintf(buf, sizeof(buf), "%d:%s", m, ss);
    os << buf;
  } else if (seconds < 3600*24) {
    int h, m;
    m = seconds / 60;
    double s = std::fmod(seconds, 60);
    h = m / 60;
    m = m % 60;
    char ss[32] = "00";
    // if `s` is <10, start at the second digit to add a leading zero
    snprintf(ss + (s < 10), sizeof(ss) -1, "%g", s);
    char buf[64];
    snprintf(buf, sizeof(buf), "%d:%02d:%s", h, m, ss);
    os << buf;
  } else {
    int days = seconds / (24 * 3600);
    os << days << " days";
    double rem = fmodf(seconds, 24*3600);
    if (rem) {
      os << " ";
      print_time(os, rem);
    }
  }
  return os;
}

template <typename Rep, typename Period>
double seconds(std::chrono::duration<Rep, Period> time) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(time).count();
}

template <typename Rep, typename Period>
std::ostream &print_time(std::ostream &os, std::chrono::duration<Rep, Period> time) {
  return print_time(os, seconds(time));
}

inline std::string format_time(double seconds) {
  std::stringstream ss;
  print_time(ss, seconds);
  return ss.str();
}

template <typename Rep, typename Period>
std::string format_time(std::chrono::duration<Rep, Period> time) {
  return format_time(seconds(time));
}

}  // namespace test
}  // namespace dali

#endif  // DALI_TEST_TIMING_H_
