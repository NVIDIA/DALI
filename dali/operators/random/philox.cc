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
#include <tuple>

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

}  // namespace dali
