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


#include "dali/pipeline/operators/reader/loader/loader.h"

namespace dali {

DALI_SCHEMA(LoaderBase)
  .AddOptionalArg("random_shuffle",
      R"code(Whether to randomly shuffle data.)code", false)
  .AddOptionalArg("initial_fill",
      R"code(Size of the buffer used for shuffling.)code", 1024)
  .AddOptionalArg("num_shards",
      R"code(Partition the data into this many parts (used for multiGPU training).)code", 1)
  .AddOptionalArg("shard_id",
      R"code(Id of the part to read.)code", 0)
  .AddOptionalArg("tensor_init_bytes",
      R"code(Hint for how much memory to allocate per image.)code", 1048576);


}  // namespace dali
