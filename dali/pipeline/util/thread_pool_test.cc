// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "dali/core/call_at_exit.h"

namespace dali {

namespace test {

#if NVML_ENABLED
TEST(ThreadPoolTest, AffinityFailure) {
  const char *env_affinity = std::getenv("DALI_AFFINITY_MASK");
  bool had_affinity = env_affinity != nullptr;
  std::string old_affinity = had_affinity ? env_affinity : "";
  auto restore_affinity = AtScopeExit([&]() {
    if (had_affinity)
      setenv("DALI_AFFINITY_MASK", old_affinity.c_str(), 1);
    else
      unsetenv("DALI_AFFINITY_MASK");
  });
  // Force affinity setup to throw before calling NVML, without requiring a GPU.
  ASSERT_EQ(setenv("DALI_AFFINITY_MASK", "invalid", 1), 0);
  OldThreadPool tp(1, CPU_ONLY_DEVICE_ID, true, "Affinity failure test");
  int count = 0;
  for (int i = 0; i < 2; i++) {
    tp.AddWork([&]() { count++; });
    ASSERT_NO_THROW(tp.RunAll(false));
    EXPECT_NO_THROW(tp.WaitForWork());
    EXPECT_EQ(count, i + 1);
  }
}
#endif

TEST(ThreadPoolTest, WorkFailure) {
  OldThreadPool tp(1, CPU_ONLY_DEVICE_ID, false, "Work failure test");
  tp.AddWork([]() { throw std::runtime_error("Task failed"); });
  tp.RunAll(false);
  EXPECT_THROW(tp.WaitForWork(), std::runtime_error);

  int count = 0;
  tp.AddWork([&]() { count++; });
  EXPECT_NO_THROW(tp.RunAll());
  EXPECT_EQ(count, 1);
}

TEST(ThreadPool, AddWork) {
  OldThreadPool tp(16, 0, false, "OldThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.AddWork(increase);
  }
  ASSERT_EQ(count, 0);
  tp.RunAll();
  ASSERT_EQ(count, 64);
}


TEST(ThreadPool, AddWorkWithPriority) {
  // only one thread to ensure deterministic behavior
  OldThreadPool tp(1, 0, false, "OldThreadPool test");
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


TEST(ThreadPool, CheckName) {
  const char given_thread_pool_name[] = "OldThreadPool test";
  const char full_thread_pool_name[] = "[DALI][TP0]OldThreadPool test";
  // max len supported by pthread_getname_np is 16
  char read_thread_pool_name[16] = {0, };
  // only one thread to ensure deterministic behavior
  OldThreadPool tp(1, 0, false, given_thread_pool_name);
  auto set_name = [&read_thread_pool_name](int thread_id) {
    pthread_getname_np(pthread_self(), read_thread_pool_name, sizeof(read_thread_pool_name));
  };
  tp.AddWork(set_name, 1);

  tp.RunAll();
  // skip terminating \0 character
  ASSERT_EQ(0, memcmp(read_thread_pool_name, full_thread_pool_name,
                      std::min(sizeof(full_thread_pool_name), sizeof(read_thread_pool_name)) - 1));
}

}  // namespace test

}  // namespace dali
