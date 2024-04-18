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
#include "dali/core/exec/tasking.h"

namespace dali::tasking::test {

inline auto t_now() {
  return std::chrono::high_resolution_clock::now();
}

template <typename T>
inline auto t_plus(T delta) {
  return std::chrono::high_resolution_clock::now() + std::chrono::duration<T>(delta);
}

TEST(TaskingTest, ExecutorShutdown) {
  EXPECT_NO_THROW({
    Executor ex(4);
    ex.Start();
  });
}


TEST(TaskingTest, IndependentTasksAreParallel) {
  int num_threads = 4;
  Executor ex(num_threads);
  ex.Start();

  std::atomic_int parallel;
  std::atomic_bool done = false;
  auto timeout = std::chrono::high_resolution_clock::now() + std::chrono::seconds(1);
  auto complete = Task::Create([](){});
  for (int i = 0; i < num_threads; i++) {
    auto task = Task::Create([&]() {
      if (++parallel)
        done = true;
      while (!done && std::chrono::high_resolution_clock::now() < timeout) {}
      parallel--;
    });
    complete->Succeed(task);
    ex.AddSilentTask(task);
  }
  ex.AddSilentTask(complete);
  ex.Wait(complete);
  EXPECT_TRUE(done);
}

TEST(TaskingTest, DependentTasksAreSequential) {
  int num_threads = 4;
  Executor ex(num_threads);
  ex.Start();

  int num_tasks = 10;

  std::atomic_int parallel = 0;
  std::atomic_int max_parallel = 0;
  SharedTask last_task;
  for (int i = 0; i < num_tasks; i++) {
    auto task = Task::Create([&]() {
      int p = ++parallel;
      int expected = max_parallel.load();
      while (!max_parallel.compare_exchange_strong(expected, std::max(p, expected))) {}
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      --parallel;
    });
    if (last_task)
      task->Succeed(last_task);
    ex.AddSilentTask(task);
    last_task = std::move(task);
  }
  ex.Wait(last_task);
  EXPECT_EQ(max_parallel, 1)
      << "The parallelism counter should not exceed 1 for a sequence of dependent tasks.";
}

TEST(TaskingTest, TaskArgumentValid) {
  Executor ex(4);
  ex.Start();
  const int N = 10;
  SharedTask tasks[N];
  std::atomic_int tested = 0, valid = 0;
  for (int i = 0; i < N; i++) {
    tasks[i] = Task::Create([&, i](Task *task) {
      ++tested;
      if (task == tasks[i].get())
        ++valid;
    });
    ex.AddSilentTask(tasks[i]);
  }
  auto timeout = t_plus(1.0);
  while (tested != N && t_now() < timeout) {}
  ASSERT_EQ(tested, N) << "Not all tasks finished - timeout expired.";
  EXPECT_EQ(valid, N) << "Invalid task identity inside task job.";
}


TEST(TaskingTest, ArgumentPassing) {
  Executor ex(4);
  ex.Start();
  auto producer1 = Task::Create([]() {
    return 42;
  });
  auto producer2 = Task::Create([]() {
    return 0.5;
  });
  auto consumer = Task::Create([&](Task *task) {
    return 2 * task->GetInputValue<int>(0) + task->GetInputValue<double>(1);
  });

  consumer->Consume(producer1)->Consume(producer2);
  ex.AddSilentTask(producer1);
  ex.AddSilentTask(producer2);
  double ret = ex.AddTask(consumer).Value<double>(ex);
  EXPECT_EQ(ret, 84.5);
}

TEST(TaskingTest, MultiOutput) {
  Executor ex(4);
  ex.Start();
  auto producer = Task::Create(2, []() {
    return std::vector<int>{1, 42};
  });

  auto consumer1 = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + 3;
  });
  consumer1->Consume(producer, 0);

  auto consumer2 = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + 5;
  });
  consumer2->Consume(producer, 1);

  auto apex = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + t->GetInputValue<int>(1) + 10;
  });
  apex->Consume(consumer1)->Consume(consumer2);

  ex.AddSilentTask(producer);
  ex.AddSilentTask(consumer1);
  ex.AddSilentTask(consumer2);
  int ret = ex.AddTask(apex).Value<int>(ex);
  EXPECT_EQ(ret, 1 + 3 + 42 + 5 + 10);
}

namespace {

template <typename T>
struct InstanceCounter {
  T payload;
  InstanceCounter() {
    ++num_instances;
  }
  ~InstanceCounter() {
    --num_instances;
  }
  InstanceCounter(T x) : payload(std::move(x)) {  // NOLINT
    ++num_instances;
  }
  InstanceCounter(const InstanceCounter &other) : payload(other.payload) {
    ++num_instances;
  }
  InstanceCounter(InstanceCounter &&other) : payload(std::move(other.payload)) {
    ++num_instances;
  }
  static std::atomic_int num_instances;
};

}  // namespace

template <typename T>
std::atomic_int InstanceCounter<T>::num_instances = 0;

/** This test makes sure that task results are disposed of as soon as possible */
TEST(TaskingTest, MultiOutputLifespan) {
  Executor ex(4);
  ex.Start();

  // These semaphores are used for delaying the launch of dependent tasks
  auto sem1 = std::make_shared<Semaphore>(1, 0);
  auto sem2 = std::make_shared<Semaphore>(1, 0);
  auto sem3 = std::make_shared<Semaphore>(1, 0);

  // This task creates 2 InstanceCounters - it has 2 separate outputs
  auto producer = Task::Create(2, []() {
    return std::vector<InstanceCounter<int>>{1, 42};
  });

  auto consumer1 = Task::Create([](Task *t) {
    return t->GetInputValue<InstanceCounter<int>>(0).payload;
  });
  // This task consumes the 1st output of producer
  consumer1->Consume(producer, 0)->Succeed(sem1);

  auto consumer2 = Task::Create([](Task *t) {
    return t->GetInputValue<InstanceCounter<int>>(0).payload;
  });
  // This task consumes the 2nd output of producer
  consumer2->Consume(producer, 1)->Succeed(sem2);

  auto consumer3 = Task::Create([](Task *t) {
    return t->GetInputValue<InstanceCounter<int>>(0).payload;
  });
  // This task consumes the (again) 2nd output of producer
  consumer3->Consume(producer, 1)->Succeed(sem3);


  ex.AddSilentTask(producer);
  ex.AddSilentTask(consumer1);
  ex.AddSilentTask(consumer2);
  ex.AddSilentTask(consumer3);
  ex.Wait(producer);

  // Once producer is finished we should still see both instances - they should be in the inputs
  // of the consumers.
  EXPECT_EQ(InstanceCounter<int>::num_instances, 2);

  // We trigger 1st consumer
  sem1->Release(ex);
  ex.Wait(consumer1);
  // After it's done, the 1st output of producer is no longer needed - it should be destroyed
  EXPECT_EQ(InstanceCounter<int>::num_instances, 1);
  sem2->Release(ex);
  ex.Wait(consumer2);
  // 2nd output is different - it has 2 consumers
  EXPECT_EQ(InstanceCounter<int>::num_instances, 1);

  sem3->Release(ex);
  ex.Wait(consumer3);
  // Both consumers are gone - the 2nd output should be gone now.
  EXPECT_EQ(InstanceCounter<int>::num_instances, 0);
}

}  // namespace dali::tasking::test
