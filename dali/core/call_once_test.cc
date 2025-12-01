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

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include "dali/core/call_once.h"

namespace dali {

TEST(CallOnce, CountCalls) {
  dali::once_flag once;
  std::atomic_int calls{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 100; i++) {
    threads.emplace_back([&]() {
        dali::call_once(once, [&]() {
            ++calls;
        });
    });
  }
  for (auto &t : threads)
    t.join();
  EXPECT_EQ(calls.load(), 1);
}

TEST(CallOnce, RetryOnThrow) {
  dali::once_flag once;
  std::atomic_int failures{0}, successes{0};

  {
    std::vector<std::thread> threads;
    for (int i = 0; i < 100; i++) {
      threads.emplace_back([&]() {
        EXPECT_THROW((dali::call_once(once, [&]() {
            ++failures;
            throw std::runtime_error("test");
          })), std::runtime_error);
      });
    }
    for (auto &t : threads)
      t.join();
  }
  EXPECT_EQ(failures.load(), 100);

  {
    std::vector<std::thread> threads;
    for (int i = 0; i < 100; i++) {
      threads.emplace_back([&]() {
          dali::call_once(once, [&]() {
              ++successes;
          });
      });
    }
    for (auto &t : threads)
      t.join();
  }

  EXPECT_EQ(successes.load(), 1);
}

}  // namespace dali
