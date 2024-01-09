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

#ifndef DALI_PIPELINE_EXECUTOR2_EXECUTOR2_H_
#define DALI_PIPELINE_EXECUTOR2_EXECUTOR2_H_

#include <memory>
#include "graph.h"

namespace dali {
namespace exec2 {

class Executor {
 public:
  virtual ~Executor() = default;

  virtual void Initialize(std::shared_ptr<Graph> graph) = 0;
  virtual void Run() = 0;
  virtual void GetOutputs(Workspace &ws) = 0;
 private:

};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_EXECUTOR2_H_

