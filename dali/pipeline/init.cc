// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/pipeline/init.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"

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

void DALIInit(const OpSpec &cpu_allocator,
              const OpSpec &pinned_cpu_allocator,
              const OpSpec &gpu_allocator) {
#if DALI_DEBUG
  subscribe_signals();
#endif
  InitializeBackends(cpu_allocator, pinned_cpu_allocator, gpu_allocator);
}

void DALISetCPUAllocator(const OpSpec& allocator) {
  SetCPUAllocator(allocator);
}

void DALISetPinnedCPUAllocator(const OpSpec& allocator) {
  SetPinnedCPUAllocator(allocator);
}

void DALISetGPUAllocator(const OpSpec& allocator) {
  SetGPUAllocator(allocator);
}

}  // namespace dali
