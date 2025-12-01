// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_CALL_ONCE_H_
#define DALI_CORE_CALL_ONCE_H_

#include <mutex>
#include <atomic>

namespace dali {

class once_flag {
 public:
  template <typename Callable>
  friend void call_once(once_flag &flag, Callable &&callable);

 private:
  bool was_called = false;
  std::mutex m;
};

/** Workaround for glibc bug 78184
 *
 * std::call_once hangs on second attempt when the first one throws
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=78184
 */
template <typename Callable>
void call_once(once_flag &flag, Callable &&callable) {
  if (!flag.was_called) {
    std::lock_guard g(flag.m);
    if (flag.was_called)
      return;
    callable();  // if this throws, there will be another attempt
    flag.was_called = true;
  }
}

}  // namespace dali

#endif  // DALI_CORE_CALL_ONCE_H_
