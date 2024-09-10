// Copyright (c) 2019, 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR_FACTORY_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR_FACTORY_H_

#include <memory>

#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/queue_metadata.h"

namespace dali {

DLL_PUBLIC
std::unique_ptr<ExecutorBase> GetExecutor(bool pipelined, bool separated, bool async, bool dynamic,
                                          int batch_size, int num_thread, int device_id,
                                          size_t bytes_per_sample_hint, bool set_affinity = false,
                                          int max_num_stream = -1,
                                          int default_cuda_stream_priority = 0,
                                          QueueSizes prefetch_queue_depth = QueueSizes{2, 2});

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_FACTORY_H_
