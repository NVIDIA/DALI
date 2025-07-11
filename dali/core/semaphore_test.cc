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

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <chrono>
#include "dali/core/semaphore.h"

namespace dali {
namespace test {

class SemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(SemaphoreTest, BasicAcquireRelease) {
  counting_semaphore sem(1);

  // Should be able to acquire immediately
  EXPECT_TRUE(sem.try_acquire());

  // Should not be able to acquire again
  EXPECT_FALSE(sem.try_acquire());

  // Release and acquire again
  sem.release();
  EXPECT_TRUE(sem.try_acquire());
}

TEST_F(SemaphoreTest, InitialCount) {
  counting_semaphore sem(3);

  // Should be able to acquire 3 times
  EXPECT_TRUE(sem.try_acquire());
  EXPECT_TRUE(sem.try_acquire());
  EXPECT_TRUE(sem.try_acquire());

  // Should not be able to acquire 4th time
  EXPECT_FALSE(sem.try_acquire());
}

TEST_F(SemaphoreTest, ReleaseMultiple) {
  counting_semaphore sem(0);

  // Should not be able to acquire initially
  EXPECT_FALSE(sem.try_acquire());

  // Release multiple times
  sem.release();
  sem.release();
  sem.release();

  // Should be able to acquire 3 times
  EXPECT_TRUE(sem.try_acquire());
  EXPECT_TRUE(sem.try_acquire());
  EXPECT_TRUE(sem.try_acquire());

  // Should not be able to acquire 4th time
  EXPECT_FALSE(sem.try_acquire());
}

TEST_F(SemaphoreTest, AcquireTimeout) {
  counting_semaphore sem(0);

  auto start = std::chrono::steady_clock::now();
  bool acquired = sem.try_acquire_for(std::chrono::milliseconds(10));
  auto end = std::chrono::steady_clock::now();

  EXPECT_FALSE(acquired);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  EXPECT_GE(duration.count(), 10);
}

TEST_F(SemaphoreTest, AcquireUntilTimeout) {
  counting_semaphore sem(0);

  auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  bool acquired = sem.try_acquire_until(timeout);

  EXPECT_FALSE(acquired);
}

TEST_F(SemaphoreTest, MultiThreadedAcquire) {
  counting_semaphore sem(0);
  std::atomic<int> acquired_count{0};
  std::atomic<bool> should_stop{false};

  // Start consumer threads
  std::vector<std::thread> consumers;
  for (int i = 0; i < 3; ++i) {
    consumers.emplace_back([&sem, &acquired_count, &should_stop]() {
      while (!should_stop) {
        if (sem.try_acquire_for(std::chrono::milliseconds(10))) {
          acquired_count.fetch_add(1);
        }
      }
    });
  }

  // Producer thread
  std::thread producer([&sem]() {
    for (int i = 0; i < 10; ++i) {
      sem.release();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  producer.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  should_stop = true;

  for (auto& consumer : consumers) {
    consumer.join();
  }

  EXPECT_EQ(acquired_count.load(), 10);
}


TEST_F(SemaphoreTest, AcquireUntil) {
  counting_semaphore sem(0);

  // Consumer thread
  std::thread consumer([&sem]() {
    auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    bool acquired = sem.try_acquire_until(timeout);
    EXPECT_TRUE(acquired);
  });

  // Producer thread
  std::thread producer([&sem]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sem.release();
  });

  consumer.join();
  producer.join();
}

TEST_F(SemaphoreTest, AcquireFor) {
  counting_semaphore sem(0);

  // Consumer thread
  std::thread consumer([&sem]() {
    bool acquired = sem.try_acquire_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(acquired);
  });

  // Producer thread
  std::thread producer([&sem]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sem.release();
  });

  consumer.join();
  producer.join();
}

}  // namespace test
}  // namespace dali
