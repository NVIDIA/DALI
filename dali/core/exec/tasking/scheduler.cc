// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <mutex>
#include <iostream>
#include "dali/core/exec/tasking/scheduler.h"

namespace dali::tasking {

bool Scheduler::AcquireAllAndMoveToReady(SharedTask &task) noexcept {
  assert(task->state_ <= TaskState::Pending);

  // All or nothing - first we check that all preconditions are met
  for (auto &w : task->preconditions_)
    if (!w->IsAcquirable())
      return false;  // at least one unmet
  // If they are, we acquire them - this must succeed
  for (auto &w : task->preconditions_)
    if (!w->TryAcquire(task)) {
      std::cerr
          << "Internal error - resource acquisition failed for a resource known to be available"
          << std::endl;
      std::abort();
    }

  task->preconditions_.clear();
  task->state_ = TaskState::Ready;
  pending_.Remove(task);
  ready_.push(std::move(task));
  return true;
}

void Scheduler::Notify(Waitable *w) {
  bool is_completion_event = dynamic_cast<CompletionEvent *>(w) != nullptr;
  bool is_task = is_completion_event && dynamic_cast<Task *>(w);

  int new_ready = 0;
  int num_ready = 0;
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
      // If the waitable is a completion event, it will never become unacquirable again.
      // Otherwise, we have to re-check it.
      if (!is_completion_event && !w->IsAcquirable())
        break;
      if (task->Ready()) {
        new_ready++;
        continue;
      }

      // If the task has only one precondition or the waitable is a completion event,
      // then we can just try to acquire that waitable on behalf of the task.
      // A completion event, once complete, is never un-completed and all waiting threads
      // will be able to acquire it. This menas that we can eagerly acquire it without risking
      // deadlocks. This imposes less overhead than re-checking all preconditions each time.
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
          task->state_ = TaskState::Ready;
          ready_.push(std::move(task));
          new_ready++;
          // OK, the task is ready, we're done with it
          continue;
        }
      }

      if (AcquireAllAndMoveToReady(task))
        new_ready++;
    }
    num_ready = ready_.size();
  }

  if (new_ready == 1)
    this->task_ready_.notify_one();
  else if (new_ready > 1)
    this->task_ready_.notify_all();
}

}  // namespace dali::tasking
