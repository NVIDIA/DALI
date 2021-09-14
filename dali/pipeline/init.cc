// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <signal.h>
#include <cstring>
#include <cstdlib>
#include "dali/pipeline/init.h"
#include "dali/core/error_handling.h"
#include "dali/core/math_util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {

#if DALI_DEBUG

namespace {

void signal_handler(int sig) {
  std::cerr << "Error: signal " << strsignal(sig) << ":"
            << GetStacktrace() << std::endl;
  exit(1);
}

void subscribe_signals() {
  signal(SIGTERM, signal_handler);
  signal(SIGKILL, signal_handler);
  signal(SIGSEGV, signal_handler);
}

}  // namespace

#endif

void InitializeBufferPolicies() {
  if (const char *threshold_str = std::getenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD")) {
    Buffer<CPUBackend>::SetShrinkThreshold(clamp(atof(threshold_str), 0.0, 1.0));
  }
  if (const char *factor = std::getenv("DALI_BUFFER_GROWTH_FACTOR")) {
    double max_factor = Buffer<CPUBackend>::kMaxGrowthFactor;
    Buffer<CPUBackend>::SetGrowthFactor(clamp(atof(factor), 1.0, max_factor));
    max_factor = Buffer<GPUBackend>::kMaxGrowthFactor;
    Buffer<GPUBackend>::SetGrowthFactor(clamp(atof(factor), 1.0, max_factor));
  }
  if (const char *factor = std::getenv("DALI_HOST_BUFFER_GROWTH_FACTOR")) {
    const double max_factor = Buffer<CPUBackend>::kMaxGrowthFactor;
    Buffer<CPUBackend>::SetGrowthFactor(clamp(atof(factor), 1.0, max_factor));
  }
  if (const char *factor = std::getenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR")) {
    const double max_factor = Buffer<GPUBackend>::kMaxGrowthFactor;
    Buffer<GPUBackend>::SetGrowthFactor(clamp(atof(factor), 1.0, max_factor));
  }
}

void DALIInit(const OpSpec &cpu_allocator,
              const OpSpec &pinned_cpu_allocator,
              const OpSpec &gpu_allocator) {
  (void)cpu_allocator;
  (void)pinned_cpu_allocator;
  (void)gpu_allocator;
#if DALI_DEBUG
  subscribe_signals();
#endif
  InitializeBufferPolicies();
}

}  // namespace dali
