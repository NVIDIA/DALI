// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/util/thread_pool.h"
#include <gtest/gtest.h>
#include <atomic>

namespace dali {

namespace test {

TEST(ThreadPool, AddWork) {
  ThreadPool tp(16, 0, false);
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.AddWork(increase);
  }
  ASSERT_EQ(count, 0);
  tp.RunAll();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, AddWorkImmediateStart) {
  ThreadPool tp(16, 0, false);
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.AddWork(increase, 0, true);
  }
  tp.WaitForWork();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, AddWorkWithPriority) {
  ThreadPool tp(1, 0, false);  // only one thread to ensure deterministic behavior
  std::atomic<int> count{0};
  auto set_to_1 = [&count](int thread_id) {
    count = 1;
  };
  auto increase_by_1 = [&count](int thread_id) {
    count++;
  };
  auto mult_by_2 = [&count](int thread_id) {
    int val = count.load();
    while (!count.compare_exchange_weak(val, val * 2)) {}
  };
  tp.AddWork(increase_by_1, 2);
  tp.AddWork(mult_by_2, 7);
  tp.AddWork(mult_by_2, 9);
  tp.AddWork(mult_by_2, 8);
  tp.AddWork(increase_by_1, 100);
  tp.AddWork(set_to_1, 1000);

  tp.RunAll();
  ASSERT_EQ(((1+1) << 3) + 1, count);
}

}  // namespace test

}  // namespace dali
