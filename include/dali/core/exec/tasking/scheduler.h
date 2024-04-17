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
  TaskFuture(SharedTask task, SharedTaskResult result)
      : task_(std::move(task)), result_(std::move(result)) {}

  /** Waits for a result of type T
   *
   * If the task throws, the exception rethrown here.
   * If the task returns a value of a different type, std::bad_any_cast is thrown.
   */
  template <typename T>
  T Value(Scheduler &sched) & {
    Wait(sched);
    return result_->Value<T>();
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
    return result_->Value<T>();
  }

  /** Waits for a result and returns it as `std::any`
   *
   * If the task throws, the exception rethrown here.
   * If the task returns `void`, the returned `any` is empty.
   */
  decltype(auto) Value(Scheduler &sched) & {
    Wait(sched);
    return result_->Value();
  }

  /** Waits for a result and returns it as `std::any`
   *
   * If the task throws, the exception rethrown here.
   * If the task returns `void`, the returned `any` is empty.
   */
  auto Value(Scheduler &sched) && {
    Wait(sched);
    return result_->Value();
  }


 private:
  void Wait(Scheduler &sched);
  SharedTask task_;
  SharedTaskResult result_;
};

/** Determines the readiness and execution order of tasks.
 *
 * Task lifecycle:
 * 1. New
 *    - the task is created
 *    - the data dependencies are added
 * 2. Pending
 *    - the task has been submitted to a Scheduler via AddTask or AddSilentTask
 * 3. Ready
 *    - the task has been submitted and all its preconditions are met
 * 4. Running
 *    - the task has been Popped from the scheduler and is being run
 * 5. Complete
 *    - the task's payload has finished running
 *
 * The Scheduler steps in at stage 2 and drives the lifecycle of a task until completion.
 *
 */
class Scheduler {
  struct TaskPriorityLess {
    bool operator()(const SharedTask &a, const SharedTask &b) const {
      return a->Priority() < b->Priority();
    }
  };

 public:
  SharedTask Pop() {
    std::unique_lock lock(mtx_);
    task_ready_.wait(lock, [&]() { return !ready_.empty() || shutdown_requested_; });
    if (ready_.empty()) {
      assert(shutdown_requested_);
      return nullptr;
    }
    auto ret = std::move(ready_.top());
    assert(ret->state == TaskState::Ready);
    ready_.pop();
    ret->state = TaskState::Running;
    return ret;
  }

  void AddSilentTask(SharedTask task) {
    if (task->state != TaskState::New)
      throw std::logic_error("A task can be submitted only once.");
    AddTaskImpl(std::move(task));
  }

  [[nodiscard("Use AddSilentTask if the result is not needed")]] TaskFuture AddTask(
      SharedTask task) {
    if (task->state != TaskState::New)
      throw std::logic_error("A task can be submitted only once.");
    auto res = task->result_;
    assert(res);
    AddTaskImpl(task);
    return {std::move(task), std::move(res)};
  }

  void Notify(Waitable *w);

  void Wait(Task *task);

  void Shutdown() {
    std::lock_guard g(mtx_);
    shutdown_requested_ = true;
    task_ready_.notify_all();
  }

 private:
  bool CheckTaskReady(SharedTask &task) noexcept;

  void AddTaskImpl(SharedTask task) {
    if (task->state != TaskState::New)
      throw std::logic_error("A task can be submitted only once.");
    if (task->Ready()) {
      std::lock_guard lock(mtx_);
      {
        task->state = TaskState::Ready;
        ready_.push(task);
      }
    } else {
      std::lock_guard lock(mtx_);
      task->state = TaskState::Pending;
      for (auto &pre : task->preconditions_) {
        bool added = pre->AddToWaiting(task);
        (void)added;
        assert(added);
      }
      pending_.PushFront(task);
      CheckTaskReady(task);
    }
    task_ready_.notify_one();
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
  assert(state == TaskState::Running);
  wrapped_(this);
  result_.reset();
  MarkAsComplete();
  Notify(sched);
  for (auto &r : release_) {
    r->Release(sched);
  }
  release_.clear();
  state = TaskState::Complete;
}

inline void TaskFuture::Wait(Scheduler &sched) {
  sched.Wait(task_.get());
}


void Scheduler::Wait(Task *task) {
  std::unique_lock lock(mtx_);
  task_done_.wait(lock, [&]() { return task->CheckComplete() || shutdown_requested_; });
}

bool Scheduler::CheckTaskReady(SharedTask &task) noexcept {
  assert(task->state <= TaskState::Pending);

  for (auto &w : task->preconditions_)
    if (!w->CheckComplete())
      return false;
  for (auto &w : task->preconditions_)
    if (!w->TryAcquire(task)) {
      // this should be a fatal error, but terminate and abort don't have messages
      std::cerr
          << "Internal error - resource acquisition failed for a resource known to be available"
          << std::endl;
      std::abort();
    }

  task->preconditions_.clear();
  task->state = TaskState::Ready;
  pending_.Remove(task);
  ready_.push(std::move(task));
  return true;
}

void Scheduler::Notify(Waitable *w) {
  bool is_completion_event = dynamic_cast<CompletionEvent *>(w) != nullptr;
  bool is_task = is_completion_event && dynamic_cast<Task *>(w);

  int new_ready = 0;
  {
    std::lock_guard g(mtx_);
    if (is_task)
      task_done_.notify_all();

    SmallVector<SharedTask, 8> waiting;
    int n = w->waiting_.size();
    waiting.reserve(n);
    for (int i = 0; i < n; i++)
      waiting.emplace_back(w->waiting_[i]);

    for (auto &task : waiting) {
      if (task->Ready())
        continue;

      // If the task has only one precondition or the waitable is a completion event,
      // then we can just try to acquire that waitable on behalf of the task.
      if (is_completion_event ||
          (task->preconditions_.size() == 1 && task->preconditions_.begin()->get() == w)) {
        // try acquire - the only way this can fail is that the task was
        // re-checked in another thread and marked as ready...
        if (!w->TryAcquire(task)) {
          assert(task->preconditions_.size() != 1 || task->preconditions_.begin()->get() != w);
          continue;  // ... if so, nothing to do
        }
        auto it = std::find_if(task->preconditions_.begin(), task->preconditions_.end(),
                               [w](auto &pre) { return pre.get() == w; });
        assert(it != task->preconditions_.end());
        task->preconditions_.erase(it);
        if (task->Ready()) {
          pending_.Remove(task);
          task->state = TaskState::Ready;
          ready_.push(std::move(task));
          new_ready++;
          // OK, the task is ready, we're done with it
          continue;
        }
      }

      if (CheckTaskReady(task))
        new_ready++;
    }
  }


  if (new_ready == 1)
    this->task_ready_.notify_one();
  else if (new_ready > 1)
    this->task_ready_.notify_all();
}

}  // namespace dali::tasking


#endif  // DALI_CORE_EXEC_TASKING_SCHEDULER_H_
