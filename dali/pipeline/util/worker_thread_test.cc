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

#include "dali/pipeline/util/worker_thread.h"

namespace dali {

namespace test {

TEST(WorkerThread, Destructing) {
  WorkerThread wt(0, false);
  // check destruction of a running worker thread
}

TEST(WorkerThread, WaitForWorkErrorHandling) {
  WorkerThread wt(0, false);
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
  WorkerThread wt(0, false);
  ASSERT_TRUE(wt.WaitForInit());
  wt.DoWork([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    throw std::runtime_error("Worker thread exception message 2");
  });
  wt.Shutdown();  // assure it does not deadlock
}

}  // namespace test

}  // namespace dali
