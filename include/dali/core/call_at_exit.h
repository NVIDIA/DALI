// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_CALL_AT_EXIT_H_
#define DALI_CORE_CALL_AT_EXIT_H_

#include <utility>

namespace dali {

namespace detail {
/**
 * @brief Executes the code provided in constructor when the object is destroyed
 */
template <typename Callable>
struct CallAtExit {
  explicit CallAtExit(Callable &&c) : callable(std::move(c)) {}
  ~CallAtExit() noexcept(false) {
    callable();
  }
  Callable callable;
};

}  // namespace detail

template <typename Callable>
detail::CallAtExit<Callable> AtScopeExit(Callable &&c) {
  return detail::CallAtExit<Callable>(std::forward<Callable>(c));
}

}  // namespace dali

#endif  // DALI_CORE_CALL_AT_EXIT_H_
