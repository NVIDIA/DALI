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

#ifndef DALI_OPERATORS_RANDOM_PHILOX_H_
#define DALI_OPERATORS_RANDOM_PHILOX_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>
#include "dali/core/api_helper.h"

namespace dali {

class Philox4x32_10 {
 public:
  void skipahead(uint64_t n) {
    if (advance(n))
      recalc_output();
  }

  void skipahead_sequence(uint64_t n) {
    if (advance_sequence(n))
      recalc_output();
  }

  void init(uint64_t key, uint64_t sequence, uint64_t offset) {
    state_.key = key;
    state_.ctr[0] = offset >> 2;
    state_.ctr[1] = sequence;
    state_.phase = offset & 3;
    recalc_output();
  }

  void init(uint64_t key, uint64_t ctr_hi, uint64_t ctr_lo, unsigned phase) {
    assert(phase < 4);
    state_.key = key;
    state_.ctr[0] = ctr_lo;
    state_.ctr[1] = ctr_hi;
    state_.phase = phase & 3;
    recalc_output();
  }

  uint32_t next() {
    uint32_t ret = out_[state_.phase++];
    if (state_.phase >= 4) {
      state_.phase = 0;
      advance_counter(1);
      recalc_output();
    }
    return ret;
  }

  uint32_t operator()() { return next(); }

  using result_type = uint32_t;

  static constexpr uint32_t max() { return 0xffffffffu; }
  static constexpr uint32_t min() { return 0; }

  struct State {
    uint64_t key;
    uint64_t ctr[2];
    int phase;
  };

  static DLL_PUBLIC std::string state_to_string(const State &state);

  inline std::string state_to_string() const { return state_to_string(state_); }

  static DLL_PUBLIC void state_from_string(State &state, std::string_view str);

  inline void state_from_string(std::string_view str) {
    State s;
    state_from_string(s, str);
    set_state(s);
  }

  State get_state() const {
    return state_;
  }

  void set_state(const State &state) {
    state_ = state;
    recalc_output();
  }

 private:
  DLL_PUBLIC void recalc_output();

  bool advance(uint64_t n) {
    state_.phase = state_.phase + (n & 3);
    n >>= 2;
    if (state_.phase > 3) {
      n++;
      state_.phase -= 4;
    }
    return advance_counter(n);
  }

  bool advance_counter(uint64_t n) {
    if (!n)
      return false;
    state_.ctr[0] += n;
    if (state_.ctr[0] < n) {
      state_.ctr[1]++;
    }
    return true;
  }

  bool advance_sequence(uint64_t n) {
    state_.ctr[1] += n;
    return n != 0;
  }

  State state_ = {};
  uint32_t out_[4] = {};
};

inline std::ostream &operator<<(std::ostream &os, const Philox4x32_10::State &state) {
  return os << Philox4x32_10::state_to_string(state);
}

inline std::istream &operator>>(std::istream &is, Philox4x32_10::State &state) {
  std::string str;
  is >> str;
  Philox4x32_10::state_from_string(state, str);
  return is;
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_PHILOX_H_
