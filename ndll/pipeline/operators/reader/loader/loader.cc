// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/reader/loader/loader.h"

namespace ndll {

NDLL_SCHEMA(LoaderBase)
  .AddOptionalArg("random_shuffle", "Whether to shuffle data", false)                              \
  .AddOptionalArg("initial_fill", "Size of the buffer used for shuffling", 1024)                   \
  .AddOptionalArg("num_shards", "Partition the data into this many parts", 1)                      \
  .AddOptionalArg("shard_id", "Id of the part to read", 0)                                         \
  .AddOptionalArg("tensor_init_bytes", "Hint for how much memory to allocate per image", 1048576);


}  // namespace ndll
