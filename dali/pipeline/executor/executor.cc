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

#include <algorithm>
#include <condition_variable>
#include <iterator>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/graph/op_graph_storage.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

template class Executor<AOT_WS_Policy<UniformQueuePolicy>, UniformQueuePolicy>;

}  // namespace dali
