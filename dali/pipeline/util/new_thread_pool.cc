// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>
#include "dali/core/call_at_exit.h"
#include "dali/core/small_vector.h"
#include "dali/pipeline/util/new_thread_pool.h"
#include "dali/core/device_guard.h"
#include "dali/util/nvml.h"
#include "dali/core/nvtx.h"

namespace dali {

NewThreadPool::NewThreadPool(
      int num_threads,
      std::optional<int> device_id,
      bool set_affinity,
      std::string name)
      : name_(name) {
  if (device_id.has_value() && *device_id == CPU_ONLY_DEVICE_ID)
    device_id = std::nullopt;
  device_id_ = device_id;
#if NVML_ENABLED
  // We use NVML only for setting thread affinity
  if (device_id.has_value() && set_affinity) {
    nvml_handle_ = nvml::NvmlInstance::CreateNvmlInstance();
  }
#endif
  Init(num_threads, [=, this](int thread_idx) {
    return OnThreadStart(thread_idx, set_affinity);
  });
}

std::any NewThreadPool::OnThreadStart(int thread_idx, bool set_affinity) {
  std::string name = make_string("[DALI][NT", thread_idx, "]", name_);
  SetThreadName(name.c_str());
  std::any dg;
  if (device_id_.has_value())
    dg.emplace<DeviceGuard>(*device_id_);
#if NVML_ENABLED
  try {
    if (set_affinity) {
      const char *env_affinity = std::getenv("DALI_AFFINITY_MASK");
      int core = -1;
      if (env_affinity) {
        const auto &vec = string_split(env_affinity, ',');
        if ((size_t)thread_idx < vec.size()) {
          core = std::stoi(vec[thread_idx]);
        } else {
          DALI_WARN("DALI_AFFINITY_MASK environment variable is set, "
                    "but does not have enough entries: thread_id (", thread_idx,
                    ") vs #entries (", vec.size(), "). Ignoring...");
        }
      }
      nvml::SetCPUAffinity(core);
    }
  } catch (const std::exception &e) {
    DALI_WARN("Couldn't set thread affinity in thread ", thread_idx, " of thread pool \"",
              name_, "\". Exception ", typeid(e).name(), ": ", e.what());
  } catch (...) {
    DALI_WARN("Couldn't set thread affinity in thread ", thread_idx, " of thread pool \"",
              name_, "\". Unknown error.");
  }
#endif
  return dg;
}

ThreadPoolFacade::~ThreadPoolFacade() noexcept {
  RunAll();
}

void ThreadPoolFacade::AddWork(std::function<void()> work, int64_t priority) {
  if (jobs_.empty() || jobs_.front().Started())
    jobs_.emplace_front();
  jobs_.front().AddTask(work, priority);
}

void ThreadPoolFacade::AddWork(std::function<void(int)> work, int64_t priority) {
  if (jobs_.empty() || jobs_.front().Started())
    jobs_.emplace_front();
  jobs_.front().AddTask([w = std::move(work)]() {
    w(ThreadPoolBase::this_thread_idx());
  }, priority);
}

void ThreadPoolFacade::RunAll(bool wait) {
  if (!jobs_.empty()) {
    if (!wait) {
      if (!jobs_.front().Started())  // all subsequent jobs_ must be started
        jobs_.front().Run(*tp_, false);
    } else {
      if (jobs_.size() == 1) {  // fast path for the common case
        auto atexit = AtScopeExit([&]() {
          jobs_.clear();
        });
        if (jobs_.front().Started())
          jobs_.front().Wait();
        else
          jobs_.front().Run(*tp_, true);
      } else {
        if (!jobs_.front().Started())
          jobs_.front().Run(*tp_, false);
        WaitForWork();
      }
    }
  }
}

void ThreadPoolFacade::WaitForWork() {
  if (!jobs_.empty()) {
    if (!jobs_.front().Started())
      throw std::logic_error("WaitForWork called without Run");
    auto atexit = AtScopeExit([&]() {
      jobs_.clear();
    });
    // This won't be allocated unless an exception was thrown
    std::vector<std::exception_ptr> errs;
    // The jobs in jobs_ are ordered from latest to oldest, so theres little chance that more than
    // one Wait would block.
    for (auto &job : jobs_) {
      try {
        job.Wait();
      } catch (MultipleErrors &e) {
        // unwrap MultipleErrors to avoid nesting
        errs.insert(errs.end(), e.errors().begin(), e.errors().end());
      } catch (...) {
        errs.push_back(std::current_exception());
      }
    }
    if (errs.size() == 1) {
      std::rethrow_exception(std::move(errs[0]));
    } else if (errs.size() > 1) {
      std::reverse(errs.begin(), errs.end());
      throw MultipleErrors(std::move(errs));
    }  // else no error
  }
}

int ThreadPoolFacade::NumThreads() const {
  return tp_->NumThreads();
}

std::vector<std::thread::id> ThreadPoolFacade::GetThreadIds() const {
  return tp_->GetThreadIds();
}

bool UseNewThreadPool() {
  static bool use_new_thread_pool = []() {
    const char *new_tp = getenv("DALI_USE_NEW_THREAD_POOL");
    return new_tp && atoi(new_tp);
  }();
  return use_new_thread_pool;
}


}  // namespace dali
