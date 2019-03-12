// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"

namespace dali {

template <typename... Ts>
std::unique_ptr<ExecutorBase> GetExecutor(bool pipelined, bool separated, bool async,
                                          Ts... args) {
  if (async && separated && pipelined) {
    return std::unique_ptr<ExecutorBase>{new AsyncSeparatedPipelinedExecutor(args...)};
  } else if (async && !separated && pipelined) {
    return std::unique_ptr<ExecutorBase>{new AsyncPipelinedExecutor(args...)};
  } else if (!async && separated && pipelined) {
    return std::unique_ptr<ExecutorBase>{new SeparatedPipelinedExecutor(args...)};
  } else if (!async && !separated && pipelined) {
    return std::unique_ptr<ExecutorBase>{new PipelinedExecutor(args...)};
  } else if (!async && !separated && !pipelined) {
    return std::unique_ptr<ExecutorBase>{new SimpleExecutor(args...)};
  }
  std::stringstream error;
  error << std::boolalpha;
  error << "No supported executor selected for pipelined = " << pipelined
        << ", separated = " << separated << ", async = " << async << std::endl;
  DALI_FAIL(error.str());
}

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_FACTORY_H_
