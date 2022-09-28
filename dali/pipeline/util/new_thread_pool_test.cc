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
#include "dali/pipeline/util/new_thread_pool.h"

namespace dali {
namespace experimental {

struct SerialExecutor {
  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void(int)>>>
  AddTask(Runnable &&runnable) {
    runnable(0);
  }
};

TEST(NewThreadPool, RunJobInSeries) {
  Job job;
  SerialExecutor tp;
  int a = 0, b = 0, c = 0;
  job.AddTask([&](int) {
    a = 1;
  });
  job.AddTask([&](int) {
    b = 2;
  });
  job.AddTask([&](int) {
    c = 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}

TEST(NewThreadPool, RunJobInThreadPool) {
  Job job;
  ThreadPoolBase tp(4);
  int a = 0, b = 0, c = 0;
  job.AddTask([&](int) {
    a = 1;
  });
  job.AddTask([&](int) {
    b = 2;
  });
  job.AddTask([&](int) {
    c = 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}


TEST(NewThreadPool, RethrowMultipleErrors) {
  Job job;
  ThreadPoolBase tp(4);
  job.AddTask([&](int) {
    throw std::runtime_error("Runtime");
  });
  job.AddTask([&](int) {
    // do nothing
  });
  job.AddTask([&](int) {
    throw std::logic_error("Logic");
  });
  EXPECT_THROW(job.Run(tp, true), MultipleErrors);
}


}  // namespace experimental
}  // namespace dali
