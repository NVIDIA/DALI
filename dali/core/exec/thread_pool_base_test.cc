// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include "dali/core/exec/thread_pool_base.h"
#include "dali/core/format.h"
#include "dali/test/timing.h"

namespace dali {

struct SerialExecutor {
  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
  AddTask(Runnable &&runnable) {
    runnable();
  }
};

TEST(NewThreadPool, AddTask) {
  ThreadPoolBase tp(4);
  std::atomic_int flag{0};
  for (int i = 0; i < 16; i++)
    tp.AddTask([&, i]() {
      int f = (flag |= (1 << i));
      if (f == 0xffff)
        flag.notify_all();
    });

  int f = flag.load();
  while (f != 0xffff) {
    flag.wait(f);
    f = flag.load();
  }
  // No conditions - this test succeeds if it doesn't hang
}

TEST(NewThreadPool, BulkAddTask) {
  ThreadPoolBase tp(4);
  std::atomic_int flag{0};
  {
    ThreadPoolBase::TaskBulkAdd bulk = tp.BeginBulkAdd();
    for (int i = 0; i < 16; i++)
      bulk.Add([&, i]() {
        int f = (flag |= (1 << i));
        if (f == 0xffff)
          flag.notify_all();
      });
    EXPECT_EQ(bulk.Size(), 16);
    // submitted automatically on destruction
  }

  int f = flag.load();
  while (f != 0xffff) {
    flag.wait(f);
    f = flag.load();
  }
  // No conditions - this test succeeds if it doesn't hang
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

TEST(NewThreadPool, RunIncrementalJobInThreadPool) {
  ThreadPoolBase tp(4);
  IncrementalJob job;
  std::atomic_int a = 0, b = 0, c = 0;
  job.AddTask([&]() {
    a += 1;
  });
  job.AddTask([&]() {
    b += 2;
  });
  job.Run(tp, false);

  for (int i = 0; (a.load() != 1 || b.load() != 2) && i < 100000; i++)
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  ASSERT_TRUE(a.load() == 1 && b.load() == 2) << "The job didn't start.";

  job.AddTask([&]() {
    c += 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a.load(), 1);
  EXPECT_EQ(b.load(), 2);
  EXPECT_EQ(c.load(), 3);
}


TEST(NewThreadPool, RunLargeIncrementalJobInThreadPool) {
  ThreadPoolBase tp(4);
  const int max_attempts = 10;
  for (int attempt = 0; attempt < max_attempts; attempt++) {
    IncrementalJob job;
    std::atomic_int acc = 0;
    const int total_tasks = 40000;
    const int batch_size = 100;
    for (int i = 0; i < total_tasks; i += batch_size) {
      for (int j = i; j < i + batch_size; j++) {
        job.AddTask([&, j] {
          acc += j;
        });
      }
      job.Run(tp, false);
      if (i == 0) {
        for (int spin = 0; acc.load() == 0 && spin < 100000; spin++)
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        ASSERT_NE(acc.load(), 0) << "The job isn't running in the background.";
      }
    }
    int target_value = total_tasks * (total_tasks - 1) / 2;
    if (acc.load() == target_value) {
      if (attempt == max_attempts - 1) {
        FAIL() << "The job always finishes before a call to wait.";
      } else {
        std::cerr << "The job shouldn't have completed yet - retrying.\n";
      }
      job.Wait();
      continue;
    }
    job.Run(tp, true);
    EXPECT_EQ(acc.load(), target_value);
    break;
  }
}

template <typename JobType>
class NewThreadPoolJobTest : public ::testing::Test {};

using JobTypes = ::testing::Types<Job, IncrementalJob>;
TYPED_TEST_SUITE(NewThreadPoolJobTest, JobTypes);


TYPED_TEST(NewThreadPoolJobTest, RunJobInSeries) {
  TypeParam job;
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


TYPED_TEST(NewThreadPoolJobTest, Abandon) {
  EXPECT_NO_THROW({
    TypeParam job;
    job.AddTask([]() {});
    job.Abandon();
  });
}


TYPED_TEST(NewThreadPoolJobTest, ErrorIncrementalJobNotStarted) {
  try {
    TypeParam job;
    job.AddTask([]() {});
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "The job is not empty"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}


TYPED_TEST(NewThreadPoolJobTest, RethrowMultipleErrors) {
  TypeParam job;
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

TYPED_TEST(NewThreadPoolJobTest, Reentrant) {
  TypeParam job;
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

TYPED_TEST(NewThreadPoolJobTest, JobPerf) {
  using JobType = TypeParam;
  ThreadPoolBase tp(4);
  auto do_test = [&](int jobs, int tasks) {
    std::vector<int> v(tasks);
    auto start = test::perf_timer::now();
    std::optional<JobType> j;
    for (int i = 0; i < jobs; i++) {
      j.emplace();
      for (int t = 1; t < tasks; t++) {
        j->AddTask([&, t]() {
          v[t]++;
        });
      }
      j->Run(tp, false);
      v[0]++;
      j->Wait();
      j.reset();
    }
    auto end = test::perf_timer::now();

    for (int t = 0; t < tasks; t++)
      EXPECT_EQ(v[t], jobs) << "Tasks didn't do their job";
    print(
        std::cout, "Ran ", jobs, " jobs of ", tasks, " tasks each in ",
        test::format_time(end - start), "\n");

    return end - start;
  };

  int total_tasks = 100000;
  int jobs0 = 10000, tasks0 = total_tasks / jobs0;
  auto time0 = do_test(jobs0, tasks0);
  int jobs1 = 100, tasks1 = total_tasks / jobs1;
  auto time1 = do_test(jobs1, tasks1);

  // time0 = task_time * total_tasks + job_overhead * jobs0
  // time1 = task_time * total_tasks + job_overhead * jobs1
  // hence
  // time0 - time1 = job_overhead * (jobs0 - jobs1)
  // job_overhead = (time0 - time1) / (jobs0 - jobs1)

  double job_overhead = test::seconds(time0 - time1) / (jobs0 - jobs1);
  print(std::cout, "Job overhead ", test::format_time(job_overhead), "\n");
}

}  // namespace dali
