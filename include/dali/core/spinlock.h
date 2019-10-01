// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_SPINLOCK_H_
#define DALI_CORE_SPINLOCK_H_

#include <atomic>

namespace dali {

class spinlock {
 public:
  void lock() noexcept {
    while (flag_.test_and_set(std::memory_order_acquire)) {
      // busy-wait
    }
  }

  bool try_lock() noexcept {
      return !flag_.test_and_set(std::memory_order_acquire);
  }

  void unlock() noexcept {
    flag_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

}  // namespace dali

#endif  // DALI_CORE_SPINLOCK_H_
