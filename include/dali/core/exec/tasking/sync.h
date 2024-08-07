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

#ifndef DALI_CORE_EXEC_TASKING_SYNC_H_
#define DALI_CORE_EXEC_TASKING_SYNC_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include "dali/core/small_vector.h"
#include "dali/core/spinlock.h"

namespace dali::tasking {

class Scheduler;
class Task;

using SharedTask = std::shared_ptr<Task>;
using WeakTask = std::weak_ptr<Task>;

/** Represents any object in the tasking framework that a task can wait for.
 *
 * A waitable object may be passed to Task::Succeed. This causes the execution of the task to
 * be deferred until all of its preceding conditions are ready.
 *
 * NOTE: The tasking framework implements "wait all" semantics - that is, a task acquires the
 *       waitable objects only after all of them are available, thus mitigating deadlocks.
 */
class Waitable : public std::enable_shared_from_this<Waitable> {
  friend class Scheduler;
  friend class Task;

 protected:
  /** A list of tasks waiting for this waitable object. */
  SmallVector<WeakTask, 8> waiting_;

  /** Checks whether the Waitable is ready to be acquired. */
  virtual bool IsAcquirable() const = 0;

  /** @brief Changes the state of the object to acquired (if necessary) and returns whether the
   *         operation was successful.
   */
  virtual bool AcquireImpl() = 0;

  /** Informs the scheduler that this Waitable is signalled and some tasks may become unblocked. */
  void Notify(Scheduler &sched);

  /** Fast comparison of weak pointers without calling `lock`. */
  static bool task_owner_equal(const WeakTask &w, const SharedTask &t) {
    return !w.owner_before(t) && !t.owner_before(w);
  }

  /** Checks wheter the task is waiting for this waitiable object. */
  bool IsWaitedForBy(const SharedTask &task) const {
    auto it = std::find_if(waiting_.begin(), waiting_.end(),
                           [&](auto &t) { return task_owner_equal(t, task); });
    return it != waiting_.end();
  }

  /** Tries to acquire the waitable object on behalf of a task.
   *
   * This function can fail for two reasons:
   * - the task is not waiting for this waitable
   * - AcquireImpl fails
   */
  bool TryAcquire(const SharedTask &task) {
    auto it = std::find_if(waiting_.begin(), waiting_.end(),
                           [&](auto &t) { return task_owner_equal(t, task); });
    if (it == waiting_.end())
      return false;
    if (AcquireImpl()) {
      waiting_.erase(it);
      return true;
    }
    return false;
  }

  /** Adds a task to the waiting list if it's not already there.
   *
   * @return `true`, if the task was added; `false` if it was already on the list.
   */
  bool AddToWaiting(const SharedTask &task) {
    if (IsWaitedForBy(task))
      return false;
    waiting_.push_back(task);
    return true;
  }

 public:
  virtual ~Waitable() = default;
};

/** A waitable object that changes the state at most once, from incomplete to complete.
 *
 * A completion event is any Waitable which cannot become "unsignalled" - for example, once
 * a task is complete it will ever remain so, it cannot go back to the incomplete state.
 */
class CompletionEvent : public Waitable {
 protected:
  bool AcquireImpl() override {
    // Nothing to acquire - just return true if the event is completed.
    return IsAcquirable();
  }

  void MarkAsComplete() {
    completed_.store(true, std::memory_order_release);
  }

  bool IsAcquirable() const override {
    return completed_.load(std::memory_order_acquire);
  }

 private:
  std::atomic_bool completed_{false};
};

/** A waitable object which has a Release method, which can be called from outside any task.
 */
class Releasable : public Waitable {
 public:
  /** Releases the object (see ReleaseImpl) and notifies the scheduler about the change. */
  void Release(Scheduler &sched) {
    ReleaseImpl();
    Notify(sched);
  }

 protected:
  /** Changes the internal state of the object.
   *
   * The state of the object is changed such that the next call to TryAcquire must suceed.
   */
  virtual void ReleaseImpl() = 0;
};

/** A releasable object which counts how many times it can be acquired.
 *
 * The object follows the semantics of Dijkstra's semaphore.
 * The TryAcquire is the P function and Release is the V function.
 * Unlike a mutex, a semaphore can be released by a task/thread other than one which acquired it.
 *
 * The maximum count is capped by `max_count` and `Release` will fail if it would result in
 * the semaphore exceeding the maximum count.
 */
class Semaphore : public Releasable {
 public:
  explicit Semaphore(int max_count) : Semaphore(max_count, max_count) {}
  Semaphore(int max_count, int initial_count) : count(initial_count), max_count(max_count) {}

  constexpr int MaxCount() const { return max_count; }

 protected:
  mutable spinlock lock_;

  bool IsAcquirable() const override {
    std::lock_guard g(lock_);
    return count > 0;
  }

  bool AcquireImpl() override {
    std::lock_guard g(lock_);
    if (count > 0) {
      count--;
      return true;
    }
    return false;
  }

  void ReleaseImpl() override {
    std::lock_guard g(lock_);
    if (count >= max_count)
      throw std::out_of_range("The semaphore exceeded its maximum count.");
    count++;
  }

 private:
  int count = 1;
  int max_count = 1;
};

}  // namespace dali::tasking


#endif  // DALI_CORE_EXEC_TASKING_SYNC_H_
