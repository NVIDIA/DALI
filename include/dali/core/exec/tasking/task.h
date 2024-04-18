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

#include <any>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "dali/core/exec/tasking/sync.h"
#include "dali/core/small_vector.h"

namespace dali::tasking {

namespace detail {
class TaskList;

template <typename T, typename U = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<
  decltype(*std::begin(std::declval<T>())),
  decltype(*std::end(std::declval<T>()))>>
: std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

}  // namespace detail

class Scheduler;


class TaskResult {
 public:
  template <typename F>
  void SetResultOf(F &&f) {
    try {
      if constexpr (std::is_void_v<decltype(f())>) {
        f();
        value_ = void_result();
      } else {
        value_ = f();
      }
    } catch (...) {
      exception_ = std::current_exception();
    }
  }

  template <typename T>
  void Set(T &&t) {
    value_ = std::forward<T>(t);
  }

  void SetException(std::exception_ptr e) {
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
    if (exception_)
      std::rethrow_exception(exception_);
    if constexpr (std::is_void_v<T>) {
      (void)std::any_cast<void_result>(value_);
    }
    return std::any_cast<T>(value_);
  }

  const bool HasException() const {
    return exception_ != nullptr;
  }

 private:
  struct void_result {};
  std::any value_;
  std::exception_ptr exception_;
};

using SharedTaskResult = std::shared_ptr<TaskResult>;

static constexpr int ScalarResult = -1;

class TaskResults : public SmallVector<SharedTaskResult, 4> {
 public:
  void Init(int num_results = ScalarResult) {
    assert(num_results == ScalarResult || num_results > 0);
    is_scalar_ = num_results < 0;
    resize(std::max(num_results, 1));
    for (auto &r : *this)
      r = std::make_shared<TaskResult>();
  }

  bool IsScalar() const noexcept { return is_scalar_; }

  template <typename T>
  decltype(auto) Value() const {
    if (!is_scalar_)
      throw std::logic_error("Cannot use argumentless Value to get a non-scalar value");
    return Value<T>(0);
  }

  template <typename T>
  decltype(auto) Value(int index) const {
    return GetChecked(index)->Value<T>();
  }

  decltype(auto) Value() const {
    if (!is_scalar_)
      throw std::logic_error("Cannot use argumentless Value to get a non-scalar value");
    return GetChecked(0)->Value();
  }

  decltype(auto) Value(int index) const {
    return GetChecked(index)->Value();
  }

  const SharedTaskResult &GetChecked(int index) const & {
    if (index < 0 || static_cast<size_t>(index) >= size())
      throw std::out_of_range(
          "The result index out of range. Valid range is [0.." +
          std::to_string(size() - 1) + "], got: " +
          std::to_string(index));
    return (*this)[index];
  }

 private:
  bool is_scalar_ = true;
};

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
    assert(state == TaskState::Running);
    assert(results_.size() == 1 && results_[0]);
    results_[0]->SetResultOf([&]() { return f(std::forward<Args>(args)...); });
  }

  template <typename F, typename... Args>
  auto SetResults(F &&f, Args &&...args) {
    assert(state == TaskState::Running);
    try {
      if constexpr (detail::is_iterable_v<decltype(f(std::forward<Args>(args)...))>) {
        auto &&results = f(std::forward<Args>(args)...);
        size_t n = 0;
        for (auto &&r : results) {
          if (n >= results_.size())
            throw std::logic_error("The function provided more results than "
                                  "the task was declared to have.");
          using T = std::remove_reference_t<decltype(r)>;
          results_[n]->Set(std::forward<T>(r));
          n++;
        }

        if (n < results_.size())
          throw std::logic_error("The function provided fewer results than "
                                  "the task was declared to have.");
      } else {
        throw std::invalid_argument("The result of the function is not iterable and cannot be "
                                    "used to obtain multiple output values.");
      }
    } catch (...) {
      auto ex = std::current_exception();
      for (auto &r : results_)
        r->SetException(ex);
    }
  }

  void InitResults(int n) {
    assert(n > 0);
    results_.resize(n);
    for (auto &r : results_)
      r = std::make_shared<TaskResult>();
  }

 public:
  /*template <typename F>
  explicit Task(F &&function, double priority = 0) {
    priority_ = priority;
    results_.Init(ScalarResult);

    wrapped_ = [f = std::forward<F>(function)](Task *t) {
      using Func = std::remove_reference_t<F>;
      if constexpr (std::is_invocable_v<Func, Task *>)
        t->SetResult(f, t);
      else if constexpr (std::is_invocable_v<Func>)
        t->SetResult(f);
    };
  }*/

  template <typename F>
  explicit Task(int num_results, F &&function, double priority = 0) {
    priority_ = priority;
    results_.Init(num_results);

    if (num_results == ScalarResult) {
      wrapped_ = [f = std::forward<F>(function)](Task *t) {
        using Func = std::remove_reference_t<F>;
        if constexpr (std::is_invocable_v<Func, Task *>)
          t->SetResult(f, t);
        else if constexpr (std::is_invocable_v<Func>)
          t->SetResult(f);
      };
    } else {
      using Func = std::remove_reference_t<F>;
      wrapped_ = [f = std::forward<F>(function)](Task *t) {
        if constexpr (std::is_invocable_v<Func, Task *>)
          t->SetResults(f, t);
        else if constexpr (std::is_invocable_v<Func>)
          t->SetResults(f);
      };
    }
  }


  template <typename F>
  static SharedTask Create(F &&function, double priority = 0) {
    return std::make_shared<Task>(ScalarResult, std::forward<F>(function), priority);
  }

  template <typename F>
  static SharedTask Create(int num_results, F &&function, double priority = 0) {
    return std::make_shared<Task>(num_results, std::forward<F>(function), priority);
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

  Task *Consume(const SharedTask &producer, int output_index = 0) {
    if (producer->state != TaskState::New)
      throw std::logic_error(
          "Cannot subscribe to a result of a task that's been already submitted for execution.\n"
          "If only ordering is required, use Succeed instead.");
    inputs_.push_back(producer->results_.GetChecked(output_index));
    Succeed(producer);
    return this;
  }

  const std::any &GetInputValue(int index) const {
    if (state != TaskState::Running)
      throw std::logic_error(
          "Obtaining a value of a task's input is only valid inside a task's payload function.");
    if (index < 0 || static_cast<size_t>(index) >= inputs_.size())
      throw std::out_of_range("The specified input index is out of range.");
    return inputs_[index]->Value();
  }

  template <typename T>
  T GetInputValue(int index) const {
    const std::any &result = GetInputValue(index);
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

  friend class detail::TaskList;
  friend class Scheduler;

  SharedTask next;        // pointer to the next task in an intrusive list
  Task *prev = nullptr;   // pointer to the previous task in an intrusive list
  TaskResults results_;
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
