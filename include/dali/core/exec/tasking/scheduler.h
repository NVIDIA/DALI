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

#ifndef DALI_CORE_EXEC_TASKING_SCHEDULER_H_
#define DALI_CORE_EXEC_TASKING_SCHEDULER_H_

#include <mutex>
#include <condition_variable>
#include <queue>
#include <utility>
#include <vector>
#include "dali/core/api_helper.h"
#include "dali/core/exec/tasking/task.h"
#include "dali/core/exec/tasking/sync.h"

namespace dali::tasking {

/** Represents a future result of a task.
 *
 * Like with std::future, this object can be used to wait for and obtain a result of a task.
 * The difference is that it needs a Scheduler object which can be used to determine task's
 * readiness.
 * TaskFuture prolongs the life ot a task object and its result.
 */
class TaskFuture {
 public:
  TaskFuture(SharedTask task, TaskResults results)
      : task_(std::move(task)), results_(std::move(results)) {}

  /** Waits for a result of type T
   *
   * If the task throws, the exception rethrown here.
   * If the task returns a value of a different type, std::bad_any_cast is thrown.
   */
  template <typename T>
  T Value(Scheduler &sched) & {
    Wait(sched);
    return results_.Value<T>();
  }

  /** Waits for a result of type T
   *
   * If the task throws, the exception rethrown here.
   * If the task returns a value of a different type, std::bad_any_cast is thrown.
   */
  template <typename T>
  T Value(Scheduler &sched) && {
    static_assert(!std::is_reference_v<T>, "Returning a reference to a temporary");
    Wait(sched);
    return results_.Value<T>();
  }

  /** Waits for a result and returns it as `std::any`
   *
   * If the task throws, the exception rethrown here.
   * If the task returns `void`, the returned `any` is empty.
   */
  decltype(auto) Value(Scheduler &sched) & {
    Wait(sched);
    return results_.Value();
  }

  /** Waits for a result and returns it as `std::any`
   *
   * If the task throws, the exception rethrown here.
   * If the task returns `void`, the returned `any` is empty.
   */
  auto Value(Scheduler &sched) && {
    Wait(sched);
    return results_.Value();
  }

  /** Waits for a result of type T
   *
   * If the task throws, the exception rethrown here.
   * If the task returns a value of a different type, std::bad_any_cast is thrown.
   *
   * @param index The index of the return value of the task. See `num_results` in `Task::Create`
   */
  template <typename T>
  T Value(Scheduler &sched, int index) & {
    Wait(sched);
    return results_.Value<T>(index);
  }

  /** Waits for a result of type T
   *
   * If the task throws, the exception rethrown here.
   * If the task returns a value of a different type, std::bad_any_cast is thrown.
   *
   * @param index The index of the return value of the task. See `num_results` in `Task::Create`
   */
  template <typename T>
  T Value(Scheduler &sched, int index) && {
    static_assert(!std::is_reference_v<T>, "Returning a reference to a temporary");
    Wait(sched);
    return results_.Value<T>(index);
  }

  /** Waits for a result and returns it as `std::any`
   *
   * If the task throws, the exception rethrown here.
   * If the task returns `void`, the returned `any` is empty.
   *
   * @param index The index of the return value of the task. See `num_results` in `Task::Create`
   */
  decltype(auto) Value(Scheduler &sched, int index) & {
    Wait(sched);
    return results_.Value(index);
  }

  /** Waits for a result and returns it as `std::any`
   *
   * If the task throws, the exception rethrown here.
   * If the task returns `void`, the returned `any` is empty.
   *
   * @param index The index of the return value of the task. See `num_results` in `Task::Create`
   */
  auto Value(Scheduler &sched, int index) && {
    Wait(sched);
    return results_.Value(index);
  }


 private:
  void Wait(Scheduler &sched);
  SharedTask task_;
  TaskResults results_;
  bool complete_ = false;  // optimize multiple calls to Wait / Value
};

/** Determines the readiness and execution order of tasks.
 *
 * The scheduler manages tasks by identifying ready tasks and moving them from Pending to Ready
 * state. The ready tasks can be Popped (at which point they transition to Running) and executed
 * by an external entity (e.g. Executor).
 *
 * The scheduler is also the central entity that participates in notifications about the change
 * of state of the objects. Whenever an object is signalled and may affect the readiness of
 * tasks, scheduler is involved.
 * Finally, scheduler is also used in determining task completion.
 *
 * In a typical scenario, it's used in the following calls:
 * `AddTask`
 * `AddSilentTask`
 * `Wait`
 * and in `Releasable::Release`.
 *
 * The scheduler implements deadlock mitigation by ensuring that objects are acquired only
 * when all preconditions of a task can be met.
 *
 * Once a task is submitted to the scheduler, its shared pointer reference is increased and the
 * caller doesn't need to maintain a copy of the task pointer.
 *
 * AddTask vs AddSilentTask
 * AddTask produces a TaskFuture object. This object can be used to wait for the task and get its
 * result. In case of tasks that do not produce final results, this future is not needed. In that
 * case the overhead of creating a future object can be avoided by using AddSilentTask.
 * The user can still Wait for tasks without a future object. The only difference is that
 * the results of silent tasks are only accessible by the registered consumers of its outputs.
 */
class Scheduler {
  struct TaskPriorityLess {
    bool operator()(const SharedTask &a, const SharedTask &b) const {
      return a->Priority() < b->Priority();
    }
  };

 public:
  /** Removes a ready task with the highest priorty or waits for one to appear or
   *  for a shutdown notification.
   */
  SharedTask Pop() {
    std::unique_lock lock(mtx_);
    task_ready_.wait(lock, [&]() { return !ready_.empty() || shutdown_requested_; });
    if (ready_.empty()) {
      assert(shutdown_requested_);
      return nullptr;
    }
    auto ret = std::move(ready_.top());
    assert(ret->state_ == TaskState::Ready);
    ready_.pop();
    ret->state_ = TaskState::Running;
    return ret;
  }

  /** Submits a task for execution
   */
  void AddSilentTask(SharedTask task) {
    if (task->state_ != TaskState::New)
      throw std::logic_error("A task can be submitted only once.");
    AddTaskImpl(std::move(task));
  }

  /** Submits a task for execution and gets a Future which can be used to get the output value
   */
  [[nodiscard("Use AddSilentTask if the result is not needed")]]
  TaskFuture AddTask(SharedTask task) {
    if (task->state_ != TaskState::New)
      throw std::logic_error("A task can be submitted only once.");
    auto res = task->results_;
    AddTaskImpl(task);
    return {std::move(task), std::move(res)};
  }

  /** Notifies the scheduler that a Waitable's state has changed and tasks waiting for it
   *  may become ready.
   */
  void DLL_PUBLIC Notify(Waitable *w);

  /** Waits for a task to complete. */
  void Wait(Task *task);

  /** Waits for a task to complete. */
  void Wait(const SharedTask &task) {
    Wait(task.get());
  }

  /** Makes all Pop functions return with an error value. */
  void Shutdown() {
    std::lock_guard g(mtx_);
    shutdown_requested_ = true;
    task_ready_.notify_all();
    task_done_.notify_all();
  }

  /** Checks whether a shutdown was requested. */
  bool ShutdownRequested() const {
    return shutdown_requested_;
  }

 private:
  /** Moves the task to the ready queue if all of its preconditions can be acquired.
   *
   * This function atomically checks that all preconditions can be met and if so, acquires them.
   * If the preconditions where met, the task is moved from the pending list to the ready queue.
   */
  bool DLL_PUBLIC AcquireAllAndMoveToReady(SharedTask &task) noexcept;

  void AddTaskImpl(SharedTask task) {
    assert(task->state_ == TaskState::New);
    if (task->Ready()) {  // if the task has no preconditions...
      {
        // ...then we add it directly to the read_ queue.
        std::lock_guard lock(mtx_);
        task->state_ = TaskState::Ready;
        ready_.push(task);
      }
      task_ready_.notify_one();
    } else {
      // Otherwise, the task is added to the pending list
      std::lock_guard lock(mtx_);
      task->state_ = TaskState::Pending;
      for (auto &pre : task->preconditions_) {
        bool added = pre->AddToWaiting(task);
        (void)added;
        assert(added);
      }
      pending_.PushFront(task);
      // ...and we check whether its preconditions are, in fact, met.
      if (AcquireAllAndMoveToReady(task))
        task_ready_.notify_one();
    }
  }

  friend class Task;

  std::mutex mtx_;
  std::condition_variable task_ready_, task_done_;

  detail::TaskList pending_;
  std::priority_queue<SharedTask, std::vector<SharedTask>, TaskPriorityLess> ready_;
  bool shutdown_requested_ = false;
};

inline void Waitable::Notify(Scheduler &sched) {
  sched.Notify(this);
}

inline void Task::Run(Scheduler &sched) {
  assert(state_ == TaskState::Running);
  wrapped_(this);
  results_.clear();
  inputs_.clear();
  wrapped_ = {};
  MarkAsComplete();
  Notify(sched);
  for (auto &r : release_) {
    r->Release(sched);
  }
  release_.clear();
  state_ = TaskState::Complete;
}

inline void TaskFuture::Wait(Scheduler &sched) {
  if (!complete_) {
    sched.Wait(task_.get());
    complete_ = true;
  }
}


void Scheduler::Wait(Task *task) {
  std::unique_lock lock(mtx_);
  if (task->state_ < TaskState::Pending)
    throw std::logic_error("Cannot wait for a task that has not been submitted");
  task_done_.wait(lock, [&]() { return task->IsAcquirable() || shutdown_requested_; });
  if (!task->IsAcquirable())
    throw std::runtime_error("The scheduler was shut down before the task was completed.");
}

}  // namespace dali::tasking


#endif  // DALI_CORE_EXEC_TASKING_SCHEDULER_H_
