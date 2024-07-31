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
#include <tuple>
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

template <typename T>
struct is_tuple : std::false_type {};

template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};

template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

}  // namespace detail

class Scheduler;

/** A result of a task.
 *
 * This class describes a single result of a task. A task can produce multiple results.
 * The result can store a value or an exception - attempting to access the value when an exception
 * is present will result in the exception being rethrown.
 */
class TaskResult {
 public:
  /** Sets the value (or exception) being a result of calling a parameterless function. */
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

  /** Sets the value. */
  template <typename T>
  void Set(T &&t) {
    value_ = std::forward<T>(t);
  }

  /** Sets an exception. */
  void SetException(std::exception_ptr e) {
    exception_ = std::move(e);
  }

  /** Checks whether there's a value or an exception. */
  bool Empty() {
    return !value_.has_value() && !HasException();
  }

  /** Gets the value as `std::any`.
   *
   * If an exception is stored, it's rethrown.
   */
  const std::any &Value() const {
    if (exception_)
      std::rethrow_exception(exception_);
    return value_;
  }

  /** Gets the value and casts it to the requested type.
   *
   * If an exception is stored, it's rethrown.
   * This function may additionally throw std::bad_any_cast if the value does not contain a value
   * of the requested type.
   */
  template <typename T>
  T Value() const {
    if (exception_)
      std::rethrow_exception(exception_);
    if constexpr (std::is_void_v<T>)
      return (void)std::any_cast<void_result>(value_);
    else
      return std::any_cast<T>(value_);
  }

  /** Checks whether the object stores an exception. */
  bool HasException() const {
    return exception_ != nullptr;
  }

 private:
  struct void_result {};
  std::any value_;
  std::exception_ptr exception_;
};

using SharedTaskResult = std::shared_ptr<TaskResult>;

/** A special value that indicates that the Task's return value is NOT a collection or tuple. */
static constexpr int ScalarResult = -1;

/** Represents all results of a task. */
class TaskResults : public SmallVector<SharedTaskResult, 4> {
 public:
  void Init(int num_results = ScalarResult) {
    assert(num_results == ScalarResult || num_results > 0);
    is_scalar_ = num_results < 0;
    resize(std::max(num_results, 1));
    for (auto &r : *this)
      r = std::make_shared<TaskResult>();
  }

  /** If true, the object represents a single, scalar result. */
  bool IsScalar() const noexcept { return is_scalar_; }

  /** Returns the scalar return value of a task. */
  template <typename T>
  decltype(auto) Value() const {
    if (!is_scalar_)
      throw std::logic_error("Cannot use argumentless Value to get a non-scalar value");
    return Value<T>(0);
  }

  /** Returns one of the return values of a task.
   *
   * @param index The index of the value in the collection or tuple returned by the task.
   */
  template <typename T>
  decltype(auto) Value(int index) const {
    return GetChecked(index)->Value<T>();
  }

  /** Returns the scalar return value of a task. */
  decltype(auto) Value() const {
    if (!is_scalar_)
      throw std::logic_error("Cannot use argumentless Value to get a non-scalar value");
    return GetChecked(0)->Value();
  }

  /** Returns one of the return values of a task.
   *
   * @param index The index of the value in the collection or tuple returned by the task.
   */
  decltype(auto) Value(int index) const {
    return GetChecked(index)->Value();
  }

  /** Returns the SharedTaskResult at the specified index or throws std::out_of_range. */
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

/** Describes a single task, considered as a unit by the Scheduler
 *
 * A Task is a function-like object that can carry some dependencies and preconditions
 * (via Succeed) and subscribe for results of other tasks (via Subscribe).
 *
 * A task function can produce no result (void), a scalar result (any type) or multiple results
 * (iterable object or a tuple).
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
 * A Task also manages its output values. The output values are managed via shared pointers.
 * When a task subscribes to another task's results, it gets a copy of the shared pointer to the
 * respective result. The producer task uses the shared pointer to populate the values and
 * clears the outputs and inputs once done. This guarantees that the outputs don't live longer
 * than necessary.
 *
 *
 * Succeed vs Subscribe
 * Succeed simply sets a dependency. It can take any Waitable (not just Task) and, importantly,
 * it can take tasks that have been already submitted or even completed.
 * Subscribe, by contrast, requires that the producer task is not yet submitted for execution. This
 * limitation is necessary because of the output lifecycle.
 */
class Task : public CompletionEvent {
  /** Sets a single (scalar) result of a task function.
   */
  template <typename F, typename... Args>
  void SetResult(F &&f, Args &&...args) {
    assert(state_ == TaskState::Running);
    assert(results_.size() == 1 && results_[0]);
    results_[0]->SetResultOf([&]() { return std::forward<F>(f)(std::forward<Args>(args)...); });
  }

  template <int i, typename... T>
  static void UnpackResults(TaskResults &r, std::tuple<T...> &&t) {
    if constexpr (i < sizeof...(T)) {
      r[i]->Set(std::move(std::get<i>(t)));
      UnpackResults<i + 1>(r, std::move(t));
    }
  }

  /** Sets mutliple results of a task function
   *
   * The task result must be a collection or a tuple.
   */
  template <typename F, typename... Args>
  auto SetResults(F &&f, Args &&...args) {
    assert(state_ == TaskState::Running);
    try {
      using result_t = std::remove_reference_t<decltype(std::forward<F>(f)(
        std::forward<Args>(args)...))>;
      if constexpr (detail::is_iterable_v<result_t>) {
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
      } else if constexpr (detail::is_tuple_v<result_t>) {  // NOLINT
        assert(std::tuple_size_v<result_t> == results_.size() &&
               "Internal error - incorrect tuple size should have been detected earlier.");
        auto &&results = f(std::forward<Args>(args)...);
        UnpackResults<0>(results_, std::move(results));
      } else {
        assert(!"Internal error - the output type should have been rejected earlier.");
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

  template <typename R>
  void CheckResultType(int num_results) {
    using Result = std::remove_reference_t<R>;
    if constexpr (detail::is_iterable_v<Result>) {
      // We don't know how many values the task will produce, so that is deferred until run-time.
      return;
    } else if constexpr (detail::is_tuple_v<Result>) {
      if (std::tuple_size_v<Result> != num_results)
        throw std::invalid_argument("The output tuple has a different size than "
                                    "the declared number of task's ouputs.");
    } else {
      throw std::invalid_argument("The result of the function is neither iterable nor a tuple "
                                  "and cannot be used to obtain multiple output values.");
    }
  }

  template <typename F>
  void CheckFuncResultType(int num_results) {
    using Func = std::remove_reference_t<F>;
    if constexpr (std::is_invocable_v<Func, Task *>)
      CheckResultType<decltype(std::declval<F>()(this))>(num_results);
    else
      CheckResultType<decltype(std::declval<F>()())>(num_results);
  }

 public:
  template <typename F>
  Task(int num_results, F &&function, double priority = 0) {
    using Func = std::remove_reference_t<F>;
    priority_ = priority;
    results_.Init(num_results);

    if (num_results == ScalarResult) {
      // A task with a scalar return value can return anything
      wrapped_ = [f = std::forward<F>(function)](Task *t) mutable {
        using Func = std::remove_reference_t<F>;
        if constexpr (std::is_invocable_v<Func, Task *>) {
          t->SetResult(std::forward<F>(f), t);
        } else {
          static_assert(std::is_invocable_v<Func>,
                        "The task function must take no arguments or a single Task * pointer.");
          t->SetResult(std::forward<F>(f));
        }
      };
    } else {
      // A task with a non-scalar return value must return a collection or a tuple.
      // We can check the return type early (i.e. now) to aid debugging.
      CheckFuncResultType<F>(num_results);
      wrapped_ = [f = std::forward<F>(function)](Task *t) mutable {
        if constexpr (std::is_invocable_v<Func, Task *>) {
          t->SetResults(std::forward<F>(f), t);
        } else {
          static_assert(std::is_invocable_v<Func>,
                        "The task function must take no arguments or a single Task * pointer.");
          t->SetResults(std::forward<F>(f));
        }
      };
    }
  }

  template <typename F>
  explicit Task(F &&function, double priority = 0)
  : Task(ScalarResult, std::forward<F>(function), priority) {}

  /** Creates a task with a scalar result.
   *
   * @param function    The callable object that defines the task; it can return any type.
   *                    The function can take either no arguments or a single argument
   *                    of type `Task *` which will point to the current task (the very pointer
   *                    that will be returned by the call to Create).
   * @param priority    the priority with which the task will be popped by the scheduler, once ready
   */
  template <typename F>
  static SharedTask Create(F &&function, double priority = 0) {
    return std::make_shared<Task>(std::forward<F>(function), priority);
  }

  /** Creates a task with a multiple results.
   *
   * @param num_results the number of results produced by the task function;
   *                    the special value `ScalarResult` changes the interpretation of the result.
   * @param function    the callable object that defines the task;
   *                    if num_result != ScalarValue, then the function can return
   *                    - an iterable type (one on which std::begin and std::end can be called)
   *                    - a tuple
   *                    if num_result == ScalarValue, the function can return anything
   *                    The function can take either no arguments or a single argument
   *                    of type `Task *` which will point to the current task (the very pointer
   *                    that will be returned by the call to Create).   *
   * @param priority    the priority with which the task will be popped by the scheduler, once ready
   */
  template <typename F>
  static SharedTask Create(int num_results, F &&function, double priority = 0) {
    return std::make_shared<Task>(num_results, std::forward<F>(function), priority);
  }

  ~Task() {
    assert(prev_ == nullptr);
    assert(next_ == nullptr);
    assert(state_ != TaskState::Running && state_ != TaskState::Destroyed);
    state_ = TaskState::Destroyed;
  }

  TaskState state_ = TaskState::New;

  /** The priority of the task; the higher, the sooner a task is picked. */
  double Priority() const {
    return priority_;
  }

  /** If true, the task can be immediately moved to execution. */
  bool Ready() {
    return preconditions_.empty();
  }

  /** Adds a precondition.
   *
   * Calling Succeed adds the waitable `w` to the task's list of preconditions.
   * Duplicates are detected and ignored.
   *
   * The Waitable can be a Task that's already submitted, running or even complete.
   */
  Task *Succeed(const std::shared_ptr<Waitable> &w) {
    if (state_ != TaskState::New)
      throw std::logic_error(
          "Cannot add a new dependency to a task that has been submitted for execution.");
    if (std::find(preconditions_.begin(), preconditions_.end(), w) == preconditions_.end())
      preconditions_.push_back(w);
    return this;
  }

  /** Subscribes to other task's output value.
   *
   * This function does two things:
   * - adds a dependency (Succeed) on the producer task
   * - creates a shared pointer to the producer task's output
   *
   * The producer must not have been submitted to the scheduler when Subscribe is called.
   */
  Task *Subscribe(const SharedTask &producer, int output_index = 0) {
    if (producer->state_ != TaskState::New)
      throw std::logic_error(
          "Cannot subscribe to a result of a task that's been already submitted for execution.\n"
          "If only ordering is required, use Succeed instead.");
    Succeed(producer);
    inputs_.push_back(producer->results_.GetChecked(output_index));
    return this;
  }

  /** Returns a value returned by one of the producers.
   *
   * @param index   The index of the corresponding call to Subscribe; NOT the output index.
   *                consumer->Subscribe(producer, 1);
   *                consumer->Subscribe(producer, 42);
   *
   *                GetInputValue(0)  // gets producer's output 1
   *                GetInputValue(1)  // gets producer's output 42
   */
  const std::any &GetInputValue(int index) const {
    if (state_ != TaskState::Running)
      throw std::logic_error(
          "Obtaining a value of a task's input is only valid inside a task's payload function.");
    if (index < 0 || static_cast<size_t>(index) >= inputs_.size())
      throw std::out_of_range("The specified input index is out of range.");
    return inputs_[index]->Value();
  }

  /** Returns the number of inputs that the task is subscribed to. */
  int NumInputs() const {
    return inputs_.size();
  }

  /** Returns a value returns by one of the producers and casts it to the specified type.
   */
  template <typename T>
  T GetInputValue(int index) const {
    const std::any &result = GetInputValue(index);
    if constexpr (!std::is_void_v<T>)
      return std::any_cast<T>(inputs_[index]->Value());
  }

  /** Ensures that the releasable object is released after the task completes.
   *
   * The releasable object will be released once the task is complete - even if it fails with
   * an exception.
   */
  Task *ReleaseAfterRun(std::shared_ptr<Releasable> releasable) {
    if (state_ != TaskState::New)
      throw std::logic_error(
          "Cannot add a new postcondition to a task that's been submitted to execution.\n"
          "If you need to Release a releasable object after completion of an already "
          "submitted task, create an auxiliary dependent task.");
    release_.push_back(std::move(releasable));
    return this;
  }

  /** Guards the execution of the task with a waitable/releasable object.
   *
   * Equivalent to Succeed + ReleaseAfterRun.
   */
  Task *GuardWith(std::shared_ptr<Releasable> releasable) {
    Succeed(releasable);
    ReleaseAfterRun(std::move(releasable));
    return this;
  }

  /** Executes the task. */
  void Run();

  /** Waits for the task to complete.
   *
   * The task must be already submitted for execution.
   */
  void Wait() const;

 private:
  SharedTask next_;       // pointer to the next task in an intrusive list
  Task *prev_ = nullptr;  // pointer to the previous task in an intrusive list

  Scheduler *sched_ = nullptr;  // the scheduler to which the task was submitted

  /** Associates the task with a scheduler and sets the state to Pending. */
  void Submit(Scheduler &sched) {
    if (state_ != TaskState::New)
      throw std::logic_error("The has already been submitted for execution.");
    sched_ = &sched;
    state_ = TaskState::Pending;
  }

  friend class detail::TaskList;
  friend class Scheduler;

  TaskResults results_;

  double priority_ = 0;
  std::function<void(Task *)> wrapped_;
  SmallVector<std::shared_ptr<Waitable>, 4> preconditions_;
  SmallVector<SharedTaskResult, 4> inputs_;
  SmallVector<std::shared_ptr<Releasable>, 4> release_;
};

namespace detail {

/** An intrusive doubly-linked list of tasks
 *
 * This list uses tasks' built-in next_ and prev_ fields for maintaining a doubly-link list.
 */
class TaskList {
 public:
  ~TaskList() {
    // remove iteratively to prevent long recursion and possible stack overflow
    while (head_) {
      assert(!head_->prev_);
      head_ = std::move(head_->next_);
    }
    tail_ = nullptr;
  }

  /** Places the task as the new head of the list.
   *
   * The task's prev/next pointers must be null.
   */
  void PushFront(SharedTask task) {
    assert(!task->next_ && !task->prev_);
    if (!head_) {
      assert(!tail_);
      head_ = std::move(task);
      tail_ = head_.get();
    } else {
      head_->prev_ = task.get();
      task->next_ = std::move(head_);
      head_ = std::move(task);
    }
  }

  /** Removes an element from the list.
   *
   * This function removes the element from the list by detaching it and reconnecting the
   * previous and next elements. If the task coincides with head or tail, then the respective
   * end is updated accordingly.
   */
  void Remove(const SharedTask &task) {
    if (task == head_)
      head_ = task->next_;
    if (task.get() == tail_)
      tail_ = task->prev_;

    Task *p = task->prev_;
    if (task->next_)
      task->next_->prev_ = p;
    if (p)
      p->next_ = std::move(task->next_);
    else
      task->next_.reset();
    assert(!task->next_);
    task->prev_ = nullptr;
  }

  /** Returns the head (front) of the list or null, if empty */
  SharedTask head() const { return head_; }
  /** Returns the tail (back) of the list or nullptr, if empty */
  Task *tail() const { return tail_; }

 private:
  SharedTask head_;
  Task *tail_ = nullptr;
};

}  // namespace detail

}  // namespace dali::tasking

#endif  // DALI_CORE_EXEC_TASKING_TASK_H_
