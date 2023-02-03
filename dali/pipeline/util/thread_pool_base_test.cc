// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/util/thread_pool_base.h"
#include "dali/core/format.h"

namespace dali {
namespace experimental {

struct SerialExecutor {
  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
  AddTask(Runnable &&runnable) {
    runnable();
  }
};

TEST(NewThreadPool, Scrap) {
  EXPECT_NO_THROW({
    Job job;
    job.AddTask([]() {});
    job.Scrap();
  });
}

TEST(NewThreadPool, ErrorNotStarted) {
  try {
    Job job;
    job.AddTask([]() {});
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "The job is not empty"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}


TEST(NewThreadPool, RunJobInSeries) {
  Job job;
  SerialExecutor tp;
  int a = 0, b = 0, c = 0;
  job.AddTask([&]() {
    a = 1;
  });
  job.AddTask([&]() {
    b = 2;
  });
  job.AddTask([&]() {
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
  job.AddTask([&]() {
    a = 1;
  });
  job.AddTask([&]() {
    b = 2;
  });
  job.AddTask([&]() {
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
  job.AddTask([&]() {
    throw std::runtime_error("Runtime");
  });
  job.AddTask([&]() {
    // do nothing
  });
  job.AddTask([&]() {
    throw std::logic_error("Logic");
  });
  EXPECT_THROW(job.Run(tp, true), MultipleErrors);
}

template <typename... Args>
void SyncPrint(Args&& ...args) {
  static std::mutex mtx;
  std::lock_guard guard(mtx);
  std::stringstream ss;
  print(ss, std::forward<Args>(args)...);
  auto &&str = ss.str();
  printf("%s", str.c_str());
}

TEST(NewThreadPool, Reentrant) {
  Job job;
  ThreadPoolBase tp(1);  // must not hang with just one thread
  std::atomic_int outer{0}, inner{0};
  for (int i = 0; i < 10; i++) {
    job.AddTask([&, i]() {
        outer |= (i << 10);
    });
  }

  job.AddTask([&]() {
    Job innerJob;

    for (int i = 0; i < 10; i++)
      innerJob.AddTask([&, i]() {
        inner |= (1 << i);
      });

    innerJob.Run(tp, false);
    innerJob.Wait();
    outer |= (1 << 11);
  });

  for (int i = 11; i < 20; i++) {
    job.AddTask([&, i]() {
        outer |= (1 << i);
    });
  }
  job.Run(tp, true);
}

}  // namespace experimental
}  // namespace dali
