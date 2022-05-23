// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <thread>

#include "dali/pipeline/util/worker_thread.h"

namespace dali {

namespace test {

TEST(WorkerThread, Destructing) {
  WorkerThread wt(0, false, "WorkerThread test");
  // check destruction of a running worker thread
}

TEST(WorkerThread, WaitForWorkErrorHandling) {
  WorkerThread wt(0, false, "WorkerThread test");
  ASSERT_TRUE(wt.WaitForInit());
  wt.DoWork([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    throw std::runtime_error("Asdf");
  });
  try {
    wt.WaitForWork();
  } catch (const std::runtime_error &e) {
    EXPECT_EQ(e.what(), std::string("Error in worker thread: Asdf"));
    return;
  }
  FAIL() << "Expected an exception";
}

TEST(WorkerThread, ShutdownErrorHandling) {
  WorkerThread wt(0, false, "WorkerThread test");
  ASSERT_TRUE(wt.WaitForInit());
  wt.DoWork([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    throw std::runtime_error("Worker thread exception message 2");
  });
  wt.Shutdown();  // assure it does not deadlock
}

TEST(WorkerThread, CheckName) {
  const char given_thread_name[] = "WorkerThread test";
  const char full_thread_name[] = "[DALI][WT]WorkerThread test";
  // max len supported by pthread_getname_np is 16
  char read_thread_name[16] = {0, };
  WorkerThread wt(0, false, given_thread_name);
  ASSERT_TRUE(wt.WaitForInit());
  wt.DoWork([&read_thread_name]() {
    pthread_getname_np(pthread_self(), read_thread_name, sizeof(read_thread_name));
  });
  wt.Shutdown();  // assure it does not deadlock
  // skip terminating \0 character
  ASSERT_EQ(0, memcmp(read_thread_name, full_thread_name,
                      std::min(sizeof(full_thread_name), sizeof(read_thread_name)) - 1));
}

}  // namespace test

}  // namespace dali
