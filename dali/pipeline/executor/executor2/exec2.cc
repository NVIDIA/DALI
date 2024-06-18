// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/executor/executor2/exec2.h"

namespace dali {
namespace exec2 {

void Executor2::Init() {
}

void Executor2::Run() {
}

void Executor2::Prefetch() {
}


void Executor2::Outputs(Workspace *ws) {
}

void Executor2::ShareOutputs(Workspace *ws) {
}

void Executor2::ReleaseOutputs() {
}

void Executor2::EnableMemoryStats(bool enable_memory_stats) {
}

void Executor2::EnableCheckpointing(bool checkpointing) {
}

ExecutorMetaMap Executor2::GetExecutorMeta() {
    return {};
}

void Executor2::Shutdown() {
}

Checkpoint &Executor2::GetCurrentCheckpoint() {
}

void Executor2::RestoreStateFromCheckpoint(const Checkpoint &cpt) {
}

int Executor2::InputFeedCount(std::string_view input_name) {
}

OperatorBase *Executor2::GetOperator(std::string_view name) {
}


}  // namespace exec2
}  // namespace dali
