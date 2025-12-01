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

#ifndef DALI_CORE_SEMAPHORE_H_
#define DALI_CORE_SEMAPHORE_H_

#if __cpp_lib_semaphore >= 201907L && DALI_USE_STD_SEMAPHORE
#include <semaphore>
namespace dali {
using counting_semaphore = std::counting_semaphore<>;
}  // namespace dali

#else

#include <errno.h>
#include <limits.h>
#include <semaphore.h>
#include <time.h>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <system_error>

namespace dali {

class counting_semaphore {
 public:
  explicit counting_semaphore(ptrdiff_t initial_count) {
    assert(initial_count >= 0 && initial_count <= max());
    sem_init(&sem, 0, initial_count);
  }
  ~counting_semaphore() {
    sem_destroy(&sem);
  }

  counting_semaphore(const counting_semaphore &) = delete;
  counting_semaphore(counting_semaphore &&) = delete;
  counting_semaphore &operator=(const counting_semaphore &) = delete;
  counting_semaphore &operator=(counting_semaphore &&) = delete;

  static constexpr ptrdiff_t max() noexcept {
#ifdef SEM_VALUE_MAX
    return SEM_VALUE_MAX;
#else
    return _POSIX_SEM_VALUE_MAX;
#endif
  }

  void acquire() {
    for (;;) {
        if (sem_wait(&sem)) {
          if (errno == EINTR)
              continue;
          if (errno == ETIMEDOUT)
              continue;
          throw std::system_error(errno, std::generic_category());
        } else {
          break;
        }
    }
  }

  void release(ptrdiff_t update = 1) {
    assert(update >= 0 && update <= max());
    for (; update > 0; update--) {
      if (sem_post(&sem))
          throw std::system_error(errno, std::generic_category());
    }
  }

  bool try_acquire() noexcept {
    for (;;) {
      if (sem_trywait(&sem)) {
        if (errno == EINTR)
          continue;
        if (errno == EAGAIN)
          return false;
        std::terminate();
      } else {
        return true;
      }
    }
  }

  template <typename Clock, typename Duration>
  bool try_acquire_until(const std::chrono::time_point<Clock, Duration> &abs_time) {
    auto wait_until_time = convert_time(abs_time);
    auto s = std::chrono::time_point_cast<std::chrono::seconds>(wait_until_time);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_until_time - s);

    timespec ts = {
      static_cast<std::time_t>(s.time_since_epoch().count()),
      static_cast<long>(ns.count())  // NOLINT(runtime/int)
    };

    for (;;) {
      if (sem_timedwait(&sem, &ts)) {
        if (errno == EINTR)
            continue;
        if (errno == ETIMEDOUT || errno == EINVAL)
            return false;
        throw std::system_error(errno, std::generic_category());
      } else {
        return true;
      }
    }
  }

  template <typename Rep, typename Period>
  bool try_acquire_for(const std::chrono::duration<Rep, Period> &duration) {
    return try_acquire_until(std::chrono::system_clock::now() + duration);
  }

 private:
  template <typename Clock, typename Duration>
  static auto convert_time(const std::chrono::time_point<Clock, Duration> &time) {
    if constexpr (std::is_same_v<Clock, std::chrono::system_clock>) {
      return std::chrono::time_point_cast<std::chrono::system_clock::duration>(time);
    } else {
      return std::chrono::system_clock::now() + (time - Clock::now());
    }
  }
  sem_t sem;
};

}  // namespace dali

#endif

#endif  // DALI_CORE_SEMAPHORE_H_
