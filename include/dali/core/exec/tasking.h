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

#ifndef DALI_CORE_EXEC_TASKING_H_
#define DALI_CORE_EXEC_TASKING_H_

#include "dali/core/exec/tasking/scheduler.h"
#include "dali/core/exec/tasking/task.h"
#include "dali/core/exec/tasking/sync.h"
#include "dali/core/exec/tasking/executor.h"

/**
 * @brief DALI tasking module
 *
 * The tasking module provides an abstraction for scheduling dependent tasks and a simple
 * thread-pool based executor.
 *
 * The tasks can have temporal dependencies (for side-effects) and data dependencies.
 * The tasks are single-use objects, passed around via shared pointer.
 *
 * Simple usage:
 * ```
 * Executor ex;
 * ex.Start();
 * auto task1 = Task::Create([]() {
 *     cout << "Foo" << endl;
 * });
 * auto task2 = Task::Create([]() {
 *     cout << "Bar" << endl;
 * });
 * auto task3 = Task::Create([]() {
 *     cout << "Baz" << endl;
 * });
 * ex.AddSilentTask(task1);  // we're not interested in the result
 * ex.AddSilentTask(task2);
 * task3->Succeed(task1)->Succeed(task2);
 * auto future = ex.AddTask(task3);
 * future.Value<void>();
 * ```
 *
 * Data dependencies
 * ```
 * Executor ex;
 * ex.Start();
 * auto task1 = Task::Create([]() {
 *     return 12;
 * });
 * auto task2 = Task::Create([]() {
 *     return 30;
 * });
 * auto task3 = Task::Create([](Task *t) {
 *     return t->GetInputValue<int>(0) + t->GetInputValue<int>(1);
 * });
 * task3->Subscribe(task1)->Subscribe(task2);
 * ex.AddSilentTask(task1);
 * ex.AddSilentTask(task2);
 * auto future = ex.AddTask(task3);
 * cout << future.Value<int>() << endl;  // prints 42
 * ```
 */
namespace dali::tasking {}

#endif  // DALI_CORE_EXEC_TASKING_H_
