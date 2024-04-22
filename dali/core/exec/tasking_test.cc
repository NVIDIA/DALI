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
#include <random>
#include <vector>
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

  std::atomic_int parallel = 0;
  std::atomic_bool done = false;
  auto timeout = std::chrono::high_resolution_clock::now() + std::chrono::seconds(1);
  auto complete = Task::Create([](){});
  for (int i = 0; i < num_threads; i++) {
    auto task = Task::Create([&]() {
      if (++parallel == num_threads)
        done = true;
      while (!done && std::chrono::high_resolution_clock::now() < timeout) {}
      --parallel;
    });
    complete->Succeed(task);
    ex.AddSilentTask(task);
  }
  ex.AddSilentTask(complete);
  ex.Wait(complete);
  EXPECT_TRUE(done);
  EXPECT_EQ(parallel, 0) << "The tasks didn't finish cleanly";
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
  EXPECT_EQ(parallel, 0) << "The tasks didn't finish cleanly";
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

  consumer->Subscribe(producer1)->Subscribe(producer2);
  ex.AddSilentTask(producer1);
  ex.AddSilentTask(producer2);
  double ret = ex.AddTask(consumer).Value<double>(ex);
  EXPECT_EQ(ret, 84.5);
}

TEST(TaskingTest, MultiOutputIterable) {
  Executor ex(4);
  ex.Start();
  auto producer = Task::Create(2, []() {
    return std::vector<int>{1, 42};
  });

  auto consumer1 = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + 3;
  });
  consumer1->Subscribe(producer, 0);

  auto consumer2 = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + 5;
  });
  consumer2->Subscribe(producer, 1);

  auto apex = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + t->GetInputValue<int>(1) + 10;
  });
  apex->Subscribe(consumer1)->Subscribe(consumer2);

  ex.AddSilentTask(producer);
  ex.AddSilentTask(consumer1);
  ex.AddSilentTask(consumer2);
  int ret = ex.AddTask(apex).Value<int>(ex);
  EXPECT_EQ(ret, 1 + 3 + 42 + 5 + 10);
}

TEST(TaskingTest, MultiOutputTuple) {
  Executor ex(4);
  ex.Start();
  auto producer = Task::Create(2, []() {
    return std::make_tuple(1.0, 42);
  });

  auto consumer1 = Task::Create([](Task *t) {
    return t->GetInputValue<double>(0) + 3;
  });
  consumer1->Subscribe(producer, 0);

  auto consumer2 = Task::Create([](Task *t) {
    return t->GetInputValue<int>(0) + 5;
  });
  consumer2->Subscribe(producer, 1);

  auto apex = Task::Create([](Task *t) {
    return t->GetInputValue<double>(0) + t->GetInputValue<int>(1) + 10;
  });
  apex->Subscribe(consumer1)->Subscribe(consumer2);

  ex.AddSilentTask(producer);
  ex.AddSilentTask(consumer1);
  ex.AddSilentTask(consumer2);
  double ret = ex.AddTask(apex).Value<double>(ex);
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
  consumer1->Subscribe(producer, 0)->Succeed(sem1);

  auto consumer2 = Task::Create([](Task *t) {
    return t->GetInputValue<InstanceCounter<int>>(0).payload;
  });
  // This task consumes the 2nd output of producer
  consumer2->Subscribe(producer, 1)->Succeed(sem2);

  auto consumer3 = Task::Create([](Task *t) {
    return t->GetInputValue<InstanceCounter<int>>(0).payload;
  });
  // This task consumes the (again) 2nd output of producer
  consumer3->Subscribe(producer, 1)->Succeed(sem3);


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

namespace {

template <typename RNG>
std::vector<int> subset(int n, int m, RNG &rng) {
  std::vector<int> selected;
  for (int i = 0; i < m; i++) {
    std::uniform_int_distribution<> dist(0, n - i - 1);
    int k = dist(rng);
    int j = 0;
    for (; j < static_cast<int>(selected.size()); j++) {
      if (selected[j] <= k)
        k++;
      else
        break;
    }
    selected.insert(selected.begin() + j, k);
  }
  return selected;
}

double slowfunc(double x) {
  return std::sqrt(x + std::sqrt(x) + std::sqrt(x + std::sqrt(x)));
}

// This test creates a large graph with several layers and randomly placed connections
// between nodes in the layers.
// Each node produces a value that is then consumed by the dependent tasks.
// Each task compares the value vs a reference computed at task creation.
// In the event of failure, obtaining the future throws an exception.
void GraphTest(Executor &ex, int num_layers, int layer_size, int prev_layer_conn) {
  std::vector<SharedTask> tasks;
  std::set<Task *> has_successor;
  std::mt19937_64 rng;
  tasks.reserve(num_layers * layer_size);
  std::vector<int> values(num_layers * layer_size);
  std::uniform_int_distribution<> vdist(0, 10000);
  for (auto &v : values)
    v = vdist(rng);
  for (int l = 0; l < num_layers; l++) {
    int prev_layer_start = l ? tasks.size() - layer_size : 0;
    int layer_start = tasks.size();
    int prev_layer_size = layer_start - prev_layer_start;
    for (int i = 0; i < layer_size; i++) {
      int conn = std::min<int>(prev_layer_size, prev_layer_conn);
      auto deps = subset(prev_layer_size, conn, rng);
      for (auto &dep : deps)
        dep += prev_layer_start;
      int initial = values[tasks.size()];
      double ref = initial;
      for (auto &dep : deps)
        ref += slowfunc(values[dep]);
      values[tasks.size()] = ref;
      auto task = Task::Create([initial, conn, ref](Task *t) {
        double sum = initial;
        for (int i = 0; i < conn; i++) {
          int inp = t->GetInputValue<int>(i);
          sum += slowfunc(inp);
        }
        if (sum != ref)
          throw std::runtime_error("Unexpected result.");
        return static_cast<int>(sum);
      });
      for (auto &dep : deps) {
        task->Subscribe(tasks[dep]);
        has_successor.insert(tasks[dep].get());
      }
      tasks.push_back(std::move(task));
    }
  }

  std::vector<TaskFuture> futures;
  futures.reserve(tasks.size() - has_successor.size());
  for (auto &t : tasks) {
    if (!has_successor.count(t.get()))
      futures.push_back(ex.AddTask(t));
    else
      ex.AddSilentTask(t);
  }

  for (auto &f : futures)
    (void)f.Value(ex);
}

}  // namespace

TEST(TaskingTest, HighLoad) {
  Executor ex(4);
  ex.Start();
  for (int i = 0; i < 10; i++)
    GraphTest(ex, 3, 1500, 50);
}

TEST(TaskingTest, Priority) {
  Scheduler sched;
  // add 4 tasks with shuffled order
  auto t1 = Task::Create([]() {}, 1);
  sched.AddSilentTask(t1);

  auto t4 = Task::Create([]() {}, 4);
  sched.AddSilentTask(t4);

  auto t3 = Task::Create([]() {}, 3);
  sched.AddSilentTask(t3);

  auto t2 = Task::Create([]() {}, 2);
  sched.AddSilentTask(t2);

  SharedTask t;
  EXPECT_EQ(t = sched.Pop(), t4) << "Expected task t4, got t" << t->Priority();
  EXPECT_EQ(t = sched.Pop(), t3) << "Expected task t3, got t" << t->Priority();
  EXPECT_EQ(t = sched.Pop(), t2) << "Expected task t2, got t" << t->Priority();
  EXPECT_EQ(t = sched.Pop(), t1) << "Expected task t1, got t" << t->Priority();
  // we need to run the tasks to avoid asserts
  t4->Run(sched);
  t3->Run(sched);
  t2->Run(sched);
  t1->Run(sched);
}

TEST(TaskingErrorTest, DoubleSubmit) {
  Scheduler sched;
  auto t1 = Task::Create([]() {});
  sched.AddSilentTask(t1);
  EXPECT_THROW(sched.AddSilentTask(t1), std::logic_error);
}

TEST(TaskingErrorTest, SuceedAfterSubmit) {
  Scheduler sched;
  auto t1 = Task::Create([]() {});
  auto t2 = Task::Create([]() {});
  sched.AddSilentTask(t1);
  EXPECT_THROW(t1->Succeed(t2), std::logic_error);
}

TEST(TaskingErrorTest, SubscribeToPending) {
  Scheduler sched;
  auto t1 = Task::Create([]() { return 0; });
  sched.AddSilentTask(t1);
  auto t2 = Task::Create([]() {});
  EXPECT_THROW(t2->Subscribe(t1), std::logic_error);
}

TEST(TaskingErrorTest, SubscribeInvalidIndex) {
  Scheduler sched;
  auto t1 = Task::Create(2, []() { return std::make_tuple(1, 2.0); });
  auto t2 = Task::Create([]() {});
  t2->Subscribe(t1, 0);
  t2->Subscribe(t1, 1);
  EXPECT_THROW(t2->Subscribe(t1, 2), std::out_of_range);
}

TEST(TaskingErrorTest, ScalarResult2Outputs) {
  Scheduler sched;
  EXPECT_THROW(Task::Create(2, []() { return 0; }), std::invalid_argument);
}

TEST(TaskingErrorTest, IncorrectTupleSize) {
  Scheduler sched;
  EXPECT_NO_THROW(Task::Create(3, []() { return std::make_tuple(1, 2, 3); }));
  EXPECT_THROW(Task::Create(2, []() { return std::make_tuple(1, 2, 3); }), std::invalid_argument);
}

TEST(TaskingErrorTest, TaskRun_IncorrectResultCountAtRunTime) {
  Scheduler sched;
  SharedTask task;
  EXPECT_NO_THROW(task = Task::Create(2, []() { return std::vector<int>{1, 2, 3 }; }));
  auto fut = sched.AddTask(task);
  sched.Pop()->Run(sched);
  EXPECT_THROW(fut.Value<int>(sched, 0), std::logic_error);
}

TEST(TaskFutureTest, IndexChecking) {
  Scheduler sched;
  SharedTask task;
  task = Task::Create(3, []() { return std::vector<int>{1, 2, 3 }; });
  auto fut = sched.AddTask(task);
  sched.Pop()->Run(sched);
  EXPECT_NO_THROW(fut.Value<int>(sched, 0));
  EXPECT_NO_THROW(fut.Value<int>(sched, 1));
  EXPECT_NO_THROW(fut.Value<int>(sched, 2));
  EXPECT_THROW(fut.Value<int>(sched, -1), std::out_of_range);
  EXPECT_THROW(fut.Value<int>(sched, 3), std::out_of_range);
}

TEST(TaskFutureTest, TypeChecking) {
  Scheduler sched;
  SharedTask task;
  task = Task::Create(2, []() { return std::vector<int>{1, 2, 3 }; });
  auto fut = sched.AddTask(task);
  sched.Pop()->Run(sched);
  EXPECT_NO_THROW(fut.Value<int>(sched, 0));
  EXPECT_THROW(fut.Value<double>(sched, 0), std::bad_any_cast);
}

TEST(TaskInputTest, InvalidResultIndex) {
  Executor ex(4);
  ex.Start();
  auto t1 = Task::Create(2, []() { return std::vector<int>{1, 2, 3 }; });
  auto t2 = Task::Create([](Task *t) { t->GetInputValue<int>(3); });
  t2->Subscribe(t1, 0);
  t2->Subscribe(t1, 1);
  t2->Subscribe(t1, 2);
  ex.AddSilentTask(t1);
  auto fut = ex.AddTask(t2);
  EXPECT_THROW(fut.Value<void>(ex), std::out_of_range);
}

TEST(TaskInputTest, InvalidResultType) {
  Executor ex(4);
  ex.Start();
  auto t1 = Task::Create([]() { return 42; });
  auto t2 = Task::Create([](Task *t) { t->GetInputValue<double>(1); });
  t2->Subscribe(t1);
  ex.AddSilentTask(t1);
  auto fut = ex.AddTask(t2);
  EXPECT_THROW(fut.Value<void>(ex), std::out_of_range);
}

}  // namespace dali::tasking::test
