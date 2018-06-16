// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/reader/loader/loader.h"

namespace dali {

DALI_SCHEMA(LoaderBase)
  .AddOptionalArg("random_shuffle",
      R"code(`bool`
      Whether to randomly shuffle data.)code", false)
  .AddOptionalArg("initial_fill",
      R"code(`int`
      Size of the buffer used for shuffling.)code", 1024)
  .AddOptionalArg("num_shards",
      R"code(`int`
      Partition the data into this many parts
      (used for multiGPU training).)code", 1)
  .AddOptionalArg("shard_id",
      R"code(`int`
      Id of the part to read)code", 0)
  .AddOptionalArg("tensor_init_bytes",
      R"code(`int`
      Hint for how much memory to allocate per image)code", 1048576);


}  // namespace dali
