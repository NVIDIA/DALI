// Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <thread>
#ifdef __x86_64
#include <emmintrin.h>
#endif

namespace dali {

/**
 * @brief A spinlock that purely busy-waits on an atomic flag
 */
class busy_spinlock {
 public:
  void lock() noexcept {
    for (;;) {
      bool was_locked = flag_.load(std::memory_order_relaxed);
      if (!was_locked && !flag_.exchange(true, std::memory_order_acquire))
        break;
      #ifdef __x86_64
      _mm_pause();  // Let other hyperthreads run
      #endif
    }
  }

  bool try_lock() noexcept {
    // First load the flag - if it's locked, just fail. This helps in high contention scenarios.
    bool was_locked = flag_.load(std::memory_order_relaxed);
    return !was_locked && !flag_.exchange(true, std::memory_order_acquire);
  }

  void unlock() noexcept {
    flag_.store(false, std::memory_order_release);
  }

 private:
  std::atomic<bool> flag_{false};
};

/**
 * @brief A spinlock that spins a predefined number of cycles and then yields.
 */
template <int yield_after_n_spins>
class yielding_spinlock {
 public:
  void lock() noexcept {
    int spin = yield_after_n_spins;
    for (;;) {
      bool was_locked = flag_.load(std::memory_order_relaxed);
      if (!was_locked && !flag_.exchange(true, std::memory_order_acquire))
        break;
      if (spin-- > 0) {
        #ifdef __x86_64
        _mm_pause();  // Let other hyperthreads run
        #endif
      } else {
        std::this_thread::yield();
        spin = yield_after_n_spins;
      }
    }
  }

  bool try_lock() noexcept {
    bool was_locked = flag_.load(std::memory_order_relaxed);
    return !was_locked && !flag_.exchange(true, std::memory_order_acquire);
  }

  void unlock() noexcept {
    flag_.store(false, std::memory_order_release);
  }

 private:
  std::atomic<bool> flag_{false};
};

using spinlock = yielding_spinlock<100>;

}  // namespace dali

#endif  // DALI_CORE_SPINLOCK_H_
