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

#include <memory>
#include <algorithm>
#include "dali/core/small_vector.h"
#include "dali/core/spinlock.h"

namespace dali::tasking {


class Waitable : public std::enable_shared_from_this<Waitable>
{
    friend class Scheduler;
    friend class Task;
protected:
    SmallVector<std::weak_ptr<Task>, 8> waiting_;

    virtual bool CheckComplete() const = 0;

    virtual bool AcquireImpl() = 0;

    void Notify(Scheduler &sched);

    static bool task_owner_equal(const std::weak_ptr<Task> &w, const std::shared_ptr<Task> &t)
    {
        return !w.owner_before(t) && !t.owner_before(w);
    }

    bool IsWaitedForBy(const SharedTask &task) const
    {
        auto it = std::find_if(waiting_.begin(), waiting_.end(), [&](auto &t) { return task_owner_equal(t, task); });
        return it != waiting_.end();
    }

    bool TryAcquire(const SharedTask &task)
    {
        PROFILE_FUNCTION();
        auto it = std::find_if(waiting_.begin(), waiting_.end(), [&](auto &t) { return task_owner_equal(t, task); });
        if (it == waiting_.end())
            return false;
        if (AcquireImpl())
        {
            waiting_.erase(it);
            return true;
        }
        return false;
    }

    bool AddToWaiting(const SharedTask &task);

public:
    virtual ~Waitable() = default;
};

class CompletionEvent : public Waitable
{
protected:
    virtual bool AcquireImpl()
    {
        // Nothing to acquire - just return true if the event is completed.
        return CheckComplete();
    }

    void MarkAsComplete()
    {
        completed_ = true;
    }

    bool CheckComplete() const override
    {
        return completed_;
    }

private:
    bool completed_ = false;
};

class Releasable : public Waitable
{
public:
    bool Release(Scheduler &sched)
    {
        if (!ReleaseImpl())
            return false;
        Notify(sched);
        return true;
    }

protected:
    virtual bool ReleaseImpl() = 0;
};

class Semaphore : public Releasable
{
public:
    explicit Semaphore(int max_count) : Semaphore(max_count, max_count) {}
    Semaphore(int max_count, int initial_count) : max_count(max_count), count(initial_count) {}

protected:
    mutable spinlock lock_;

    bool CheckComplete() const override
    {
        std::lock_guard g(lock_);
        return count > 0;
    }

    bool AcquireImpl() override
    {
        std::lock_guard g(lock_);
        if (count > 0)
        {
            count--;
            return true;
        }
        return false;
    }

    bool ReleaseImpl() override
    {
        std::lock_guard g(lock_);
        if (count >= max_count)
            return false;
        count++;
        return true;
    }

private:
    int count = 1;
    int max_count = 1;
};

}  // namespace dali::tasking


#endif  // DALI_CORE_EXEC_TASKING_SYNC_H_
