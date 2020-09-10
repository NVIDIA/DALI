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


#include "dali/operators/reader/loader/loader.h"

namespace dali {

DALI_SCHEMA(LoaderBase)
  .AddOptionalArg("random_shuffle",
      R"code(Determines whether to randomly shuffle data.

A Prefetch the buffer with a size equal to ``initial_fill`` is used to read data sequentially,
and then samples are selected randomly to form a batch.)code", false)
  .AddOptionalArg("initial_fill",
      R"code(Size of the buffer that is used for shuffling.

If ``random_shuffle`` is turned off,  this parameter is ignored.)code", 1024)
  .AddOptionalArg("num_shards",
      R"code(Partitions the data into the specified number of parts (shards). This is typically
used for multi-GPU or multi-node training.)code", 1)
  .AddOptionalArg("shard_id",
      R"code(Index of the shard to read.)code", 0)
  .AddOptionalArg("tensor_init_bytes",
      R"code(Hint for how much memory to allocate per image.)code", 1048576)
  .AddOptionalArg("stick_to_shard",
      R"code(Determines whether the reader should stick to a data shard instead of going through
the entire dataset.

If you use the decoder caching, it significantly reduces the amount of data to be cached, but
might affect accuracy.)code", false)
  .AddOptionalArg("read_ahead",
      R"code(Determines whether the accessed data should be read ahead.

For large files such as LMDB, RecordIO, or TFRecord, this argument slows down the first access but
decreases the time of all of the following accesses.)code", false)
  .AddOptionalArg("prefetch_queue_depth",
      R"code(Specifies the number of batches to be prefetched by the internal Loader.

This value should be increased when the pipeline is CPU-stage bound, trading memory
consumption for better interleaving with the Loader thread.)code", 1)
  .AddOptionalArg("skip_cached_images",
      R"code(If set to True, the loading data will be skipped when the sample is
in the decoder cache.

In this case, the output of the loader will be empty.)code", false)
  .AddOptionalArg("lazy_init",
      R"code(If set to True, the Loader parses and prepares the dataset metadata only during the
first run instead of in the constructor.)code", false)
  .AddOptionalArg("pad_last_batch",
      R"code(If set to True, when the batch size is not aligned with the shard size, the Loader
pads the last batch with the last image.

The rest of the batch, or even an entire batch, can be added when the dataset size is not equally
divisible by the number of shards, and the shard is not equally divisible by the batch size. The
shard size will ultimately be equalized between shards.)code", false)
.AddOptionalArg("dont_use_mmap",
      R"code(If set to True, instead of trying to map the file memory,
the Loader will use plain file I/O.

Mapping provides a small performance benefit when accessing a local file system, but most network file
systems, do not provide optimum performance.
)code", false);

size_t start_index(const size_t shard_id,
                   const size_t shard_num,
                   const size_t size) {
  return size * shard_id / shard_num;
}

Index num_samples(const size_t shard_num,
                  const size_t size) {
  return static_cast<Index>(std::ceil(size * 1.0 / shard_num));
}

}  // namespace dali
