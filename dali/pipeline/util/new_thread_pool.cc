// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <typeinfo>
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
  std::string name = make_string("[DALI][NT", thread_idx, "]", name);
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

}  // namespace dali
