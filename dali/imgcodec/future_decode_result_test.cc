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

#include <gtest/gtest.h>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "dali/pipeline/util/thread_pool.h"
#include "dali/imgcodec/image_decoder_interfaces.h"

namespace dali {
namespace imgcodec {

TEST(FutureDecodeResultsTest, WaitNew) {
  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "FutureDecodeResultsTest");

  DecodeResultsPromise pro(3);
  auto fut = pro.get_future();
  tp.AddWork([pro](int) mutable {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pro.set(1, DecodeResult::Success());
    pro.set(0, DecodeResult::Success());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pro.set(2, DecodeResult::Success());
  }, 0, true);
  auto res1 = fut.wait_new();
  auto res2 = fut.wait_new();
  auto res3 = fut.wait_new();
  EXPECT_EQ(res1.size() + res2.size() + res3.size(), 3);
  // We can either get all results in one spans, in two spans or in three spans, depending
  // on timing - in any case, the end of the last non-empty span should point to the start of the
  // first span, offset by the number of entries (which is 3).
  ASSERT_TRUE(res1.begin() + 3 == res1.end() ||
              res1.begin() + 3 == res2.end() ||
              res1.begin() + 3 == res3.end());
  // now we know we can access all spans through the first one
  EXPECT_EQ(res1[0], 1);
  EXPECT_EQ(res1[1], 0);
  EXPECT_EQ(res1[2], 2);
}

TEST(FutureDecodeResultsTest, Benchmark) {
  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "FutureDecodeResultsTest");

  int num_iter = 1000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < num_iter; iter++) {
    DecodeResultsPromise res(100);
    tp.AddWork([&](int) {
      for (int i = 0; i < res.num_samples(); i++)
        res.set(i, DecodeResult::Success());
    }, 0, true);
    auto future = res.get_future();
    future.wait_all();
    future.get_all(true);
    tp.WaitForWork();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end-start);
  std::cout << time.count() / num_iter << " us/iter" << std::endl;
}

}  // namespace imgcodec
}  // namespace dali
