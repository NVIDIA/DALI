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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR_TYPE_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR_TYPE_H_

namespace dali {

enum class ExecutorFlags : int {
  None = 0,
  SetAffinity = 1,
  ConcurrencyMask = 0x0000000e,
  ConcurrencyDefault = 0,
  ConcurrencyNone = 1 << 1,
  ConcurrencyBackend = 2 << 1,
  ConcurrencyFull = 3 << 1,
  StreamPolicyMask = 0x00000070,
  StreamPolicyDefault = 0,
  StreamPolicySingle = 1 << 4,
  StreamPolicyPerBackend = 2 << 4,
  StreamPolicyPerOperator = 3 << 4,
};

constexpr ExecutorFlags operator|(ExecutorFlags a, ExecutorFlags b) {
  return static_cast<ExecutorFlags>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr ExecutorFlags operator&(ExecutorFlags a, ExecutorFlags b) {
  return static_cast<ExecutorFlags>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr ExecutorFlags operator~(ExecutorFlags flags) {
  return static_cast<ExecutorFlags>(~static_cast<int>(flags));
}

constexpr bool Test(ExecutorFlags all_flags, ExecutorFlags flags_to_test) {
  return (all_flags & flags_to_test) == flags_to_test;
}

enum class ExecutorType : int {
  Simple = 0,
  PipelinedFlag = 1,
  AsyncFlag = 2,
  SeparatedFlag = 4,
  DynamicFlag = 8,
  Pipelined = PipelinedFlag,
  AsyncPipelined = AsyncFlag | PipelinedFlag,
  SeparatedPipelined = PipelinedFlag | SeparatedFlag,  // TODO(michalz): I think it doesn't work
  AsyncSeparatedPipelined = AsyncPipelined | SeparatedFlag,
  Dynamic = AsyncPipelined | DynamicFlag,
};

constexpr ExecutorType operator|(ExecutorType a, ExecutorType b) {
  return static_cast<ExecutorType>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr ExecutorType operator&(ExecutorType a, ExecutorType b) {
  return static_cast<ExecutorType>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr bool Test(ExecutorType type, ExecutorType flags) {
  return (type & flags) == flags;
}

constexpr ExecutorType MakeExecutorType(bool pipelined, bool async, bool separated, bool dynamic) {
  ExecutorType type = ExecutorType::Simple;
  if (async) {
    pipelined = true;
  }
  if (dynamic) {
    async = true;
    pipelined = true;
  }
  if (pipelined) type = type | ExecutorType::PipelinedFlag;
  if (async) type = type | ExecutorType::AsyncFlag;
  if (separated) type = type | ExecutorType::SeparatedFlag;
  if (dynamic) type = type | ExecutorType::DynamicFlag;
  return type;
}


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_TYPE_H_
