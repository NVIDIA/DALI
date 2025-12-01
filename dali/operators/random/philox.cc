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

#include "dali/operators/random/philox.h"
#include <string>
#include <string_view>
#include <tuple>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "dali/core/format.h"

namespace dali {

#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u

namespace {

static inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
philox4x32round(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t kx, uint32_t ky) {
  uint64_t m0 = uint64_t(PHILOX_M4x32_0) * x;
  uint64_t m1 = uint64_t(PHILOX_M4x32_1) * z;
  uint32_t lo0 = uint32_t(m0);
  uint32_t hi0 = uint32_t(m0 >> 32);
  uint32_t lo1 = uint32_t(m1);
  uint32_t hi1 = uint32_t(m1 >> 32);
  return {hi1 ^ y ^ kx, lo1, hi0 ^ w ^ ky, lo0};
}

}  // namespace


void Philox4x32_10::recalc_output() {
  uint32_t x = state_.ctr[0];
  uint32_t y = state_.ctr[0] >> 32;
  uint32_t z = state_.ctr[1];
  uint32_t w = state_.ctr[1] >> 32;
  uint32_t kx = state_.key;
  uint32_t ky = state_.key >> 32;

  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 1
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 2
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 3
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 4
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 5
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 6
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 7
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 8
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 9
  kx += PHILOX_W32_0;
  ky += PHILOX_W32_1;
  std::tie(x, y, z, w) = philox4x32round(x, y, z, w, kx, ky);    // 10
  out_[0] = x;
  out_[1] = y;
  out_[2] = z;
  out_[3] = w;
}

constexpr const char *state_fmt_string() {
  if constexpr (sizeof(long) == 8) {  // NOLINT
    return "Philox_%016lX_%016lX:%016lX_%X";
  } else {
    static_assert(sizeof(long) == 8 || sizeof(long long) == 8,  // NOLINT
                  "Unsupported long/long long sizes");
    return "Philox_%016llX_%016llX:%016llX_%X";
  }
}

std::string Philox4x32_10::state_to_string(const State &state) {
  char state_str[64] = "";
  int n = std::snprintf(
      state_str, sizeof(state_str),
      state_fmt_string(),
      state.key, state.ctr[1], state.ctr[0], state.phase);
  assert(n > 0 && n < static_cast<int>(sizeof(state_str)));
  return state_str;
}

inline int parse_hex_char(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  } else {
    throw std::invalid_argument(make_string("\'", c, "\'is not a hexadecimal digit."));
  }
}

template <typename T>
T parse_hex(const char *str, size_t len) {
  T value = 0;
  for (size_t i = 0; i < len; i++) {
    value = (value << 4) | parse_hex_char(str[i]);
  }
  return value;
}

void Philox4x32_10::state_from_string(State &state, std::string_view str) {
  const char example[] = "Philox_0123456789ABCDEF_0123456789ABCDEF:0123456789ABCDEF_0";
  if (str.size() != sizeof(example) - 1) {
    throw std::invalid_argument(make_string(
      "Invalid Philox state string length: ", str.size()));
  }
  if (strncmp(str.data(), "Philox_", 7)) {
    throw std::invalid_argument(make_string(
      "Missing Philox_ prefix in: ", str.data()));
  }
  if (str[23] != '_' || str[40] != ':' || str[57] != '_') {
    throw std::invalid_argument(make_string(
      "Malformed Philox state string: ", str.data(), "\nShould be in format ", example));
  }

  uint64_t key = parse_hex<uint64_t>(str.data() + 7, 16);
  uint64_t ctr_hi = parse_hex<uint64_t>(str.data() + 24, 16);
  uint64_t ctr_lo = parse_hex<uint64_t>(str.data() + 41, 16);
  int phase = parse_hex<int>(str.data() + 58, 1);
  if (phase < 0 || phase > 3) {
    throw std::invalid_argument(make_string(
      "Invalid Philox state string phase: ", phase, " expected: 0..3"));
  }
  state.key = key;
  state.ctr[0] = ctr_lo;
  state.ctr[1] = ctr_hi;
  state.phase = phase;
}

}  // namespace dali
