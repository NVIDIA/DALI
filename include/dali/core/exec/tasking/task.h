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

#ifndef DALI_CORE_EXEC_TASKING_TASK_H_
#define DALI_CORE_EXEC_TASKING_TASK_H_

#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "dali/core/exec/tasking/sync.h"
#include "dali/core/small_vector.h"

namespace dali::tasking {

class Scheduler;


class TaskResult {
 public:
  template <typename F>
  void SetResultOf(F &&f) {
    Reset();
    try {
      if constexpr (std::is_void_v<decltype(f())>) {
        f();
        value_ = void_t();
      } else {
        value_ = f();
      }
    } catch (...) {
      exception_ = std::exception_ptr();
    }
  }

  void Reset() {
    value_.reset();
    exception_ = nullptr;
  }

  template <typename T>
  void Set(T &&t) {
    value_ = std::forward<T>(t);
  }

  void SetException(std::exception_ptr &&e) {
    exception_ = std::move(e);
  }

  bool Empty() {
    return !value_.has_value();
  }

  const std::any &Value() const {
    if (exception_)
      std::rethrow_exception(exception_);
    return value_;
  }

  template <typename T>
  T Value() const {
    if constexpr (std::is_void_v<T>) {
      (void)std::any_cast<void_t>(value_);
    }
  }

  const bool HasException() const {
    return exception_ != nullptr;
  }

 private:
  struct void_t {};
  std::any value_;
  std::exception_ptr exception_;
};

using SharedTaskResult = std::shared_ptr<TaskResult>;


enum class TaskState {
  New,
  Pending,
  Ready,
  Running,
  Complete,
  Destroyed
};


class Task : public CompletionEvent {
  template <typename F, typename... Args>
  void SetResult(F &&f, Args &&...args) {
    assert(result_);
    result_->SetResultOf([&]() { return f(std::forward<Args>(args)...); });
  }

 public:
  template <typename F>
  explicit Task(F &&function, double priority = 0) {
    priority_ = priority;
    wrapped_ = [f = std::forward<F>(function)](Task *t) {
      using Func = std::remove_reference_t<F>;
      if constexpr (std::is_invocable_v<Func, Task *>)
        t->SetResult(f, t);
      else if constexpr (std::is_invocable_v<Func>)
        t->SetResult(f);
    };
  }

  template <typename F>
  static SharedTask Create(F &&function, double priority = 0) {
    return std::make_shared<Task>(std::forward<F>(function), priority);
  }

  ~Task() {
    assert(prev == nullptr);
    assert(next == nullptr);
    assert(state != TaskState::Running && state != TaskState::Destroyed);
    state = TaskState::Destroyed;
  }

  TaskState state = TaskState::New;

  double Priority() const {
    return priority_;
  }

  bool Ready() {
    return preconditions_.empty();
  }

  Task *Succeed(const std::shared_ptr<Waitable> &w) {
    if (state != TaskState::New)
      throw std::logic_error(
          "Cannot add a new dependency to a task that has been submitted for execution.\n");
    preconditions_.push_back(w);
    return this;
  }

  Task *Consume(const SharedTask &producer) {
    if (producer->state != TaskState::New)
      throw std::logic_error(
          "Cannot subscribe to a result of a task that's been already submitted for execution.\n"
          "If only ordering is required, use Succeed instead.");
    inputs_.push_back(producer->result_);
    Succeed(producer);
    return this;
  }

  const std::any &GetProducerResult(int index) const {
    if (state != TaskState::Running)
      throw std::logic_error(
          "Obtaining a result of a producer task is only valid inside a task's payload function.");
    if (index < 0 || index >= inputs_.size())
      throw std::out_of_range("The specified producer index is out of range.");
    return inputs_[index]->Value();
  }

  template <typename T>
  T GetProducerResult(int index) const {
    const std::any &result = GetProducerResult(index);
    if constexpr (!std::is_void_v<T>)
      return std::any_cast<T>(inputs_[index]->Value());
  }

  void Run(Scheduler &sched);

  Task *ReleaseAfterRun(std::shared_ptr<Releasable> releasable) {
    if (state != TaskState::New)
      throw std::logic_error(
          "Cannot add a new postcondition to a task that's been submitted to execution.\n"
          "If you need to Release a releasable object after completion of an already "
          "submitted task, create an auxiliary dependent task.");
    release_.push_back(std::move(releasable));
    return this;
  }

  Task *GuardWith(std::shared_ptr<Releasable> releasable) {
    Succeed(releasable);
    ReleaseAfterRun(std::move(releasable));
    return this;
  }

 protected:
  double priority_ = 0;
  std::function<void(Task *)> wrapped_;
  SmallVector<std::shared_ptr<Waitable>, 4> preconditions_;
  SmallVector<SharedTaskResult, 4> inputs_;
  SmallVector<std::shared_ptr<Releasable>, 4> release_;

  friend class TaskList;
  friend class Scheduler;

  SharedTask next;        // pointer to the next task in an intrusive list
  Task *prev = nullptr;   // pointer to the previous task in an intrusive list
  SharedTaskResult result_ = std::make_shared<TaskResult>();
};

namespace detail {

/** An intrusive list of tasks
 *
 * This list uses tasks' built-in next and prev fields.
 */
class TaskList {
 public:
  void PushFront(SharedTask task) {
    assert(!task->next && !task->prev);
    if (!head) {
      assert(!tail);
      head = std::move(task);
      tail = head.get();
    } else {
      head->prev = task.get();
      task->next = std::move(head);
      head = std::move(task);
    }
  }

  void Remove(const SharedTask &task) {
    if (task == head)
      head = task->next;
    if (task.get() == tail)
      tail = task->prev;

    Task *p = task->prev;
    if (task->next)
      task->next->prev = p;
    if (p)
      p->next = std::move(task->next);
    else
      task->next.reset();
    assert(!task->next);
    task->prev = nullptr;
  }

 private:
  SharedTask head;
  Task *tail = nullptr;
};
}  // namespace detail

}  // namespace dali::tasking

#endif  // DALI_CORE_EXEC_TASKING_TASK_H_
