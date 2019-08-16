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

#include <benchmark/benchmark.h>
#include <iostream>

#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/pipeline.h"
#include "dali/core/common.h"

inline int get_env_int(std::string env_var, int default_val) {
  const char *val = std::getenv(env_var.c_str());
  return  val ? std::stoi(val, nullptr) : default_val;
}

inline std::string get_env_str(std::string env_var, std::string default_val) {
  const char *val = std::getenv(env_var.c_str());
  return  val ? std::string(val) : default_val;
}

namespace dali {

class VideoReaderBench : public benchmark::Fixture {
};


BENCHMARK_DEFINE_F(VideoReaderBench, VideoPipe)(benchmark::State& st) { // NOLINT
  int executor = st.range(0);
  int batch_size = st.range(1);
  int num_thread = st.range(2);
  auto dtype = st.range(3);

  const bool pipelined = executor > 0;
  const bool async = executor > 1;
  const int64_t seed = 12;
  const int device_id = 0;
  const int prefetched_queue_depth = 2;

  // Skip if reader_type and file name is not set
  if (!(std::getenv("DALI_TEST_VIDEO_READER_TYPE") &&
        std::getenv("DALI_TEST_VIDEO_READER_FILES"))) {
    st.SkipWithError("DALI_TEST_VIDEO_READER_TYPE or DALI_TEST_VIDEO_READER_FILES "
                     "environment variable not set.\n"
                     "DALI_TEST_VIDEO_READER_TYPE expects filenames, file_root or file_list\n"
                     "DALI_TEST_VIDEO_READER_FILES expects the file or folder path corresponding"
                     " to the reader type");
    st.KeepRunning();
    return;
  }

  const string reader_type = std::getenv("DALI_TEST_VIDEO_READER_TYPE");
  const string files = std::getenv("DALI_TEST_VIDEO_READER_FILES");

  // Skip if reader_type is not one of expected types
  if (reader_type != "file_root" &&
      reader_type != "filenames" &&
      reader_type != "file_list") {
    st.SkipWithError("Reader type should be one of of \"file_root\", \"filenames\" "
                     "or \"file_list\"");
    st.KeepRunning();
    return;
  }

  const int seq_len = get_env_int("DALI_TEST_VIDEO_READER_SEQ_LEN", 5);
  const int step = get_env_int("DALI_TEST_VIDEO_READER_STEP", -1);
  const bool shuffle = get_env_int("DALI_TEST_VIDEO_READER_SHUFFLE", 0);
  const bool skip_cached = get_env_int("DALI_TEST_VIDEO_READER_SKIP_CACHE", 0);
  const int initial_fill = get_env_int("DALI_TEST_VIDEO_READER_INITIAL_FILL", 10);
  const string img_type = get_env_str("DALI_TEST_VIDEO_READER_IMAGE_TYPE", "rgb");
  const DALIImageType image_type_e = img_type == "rgb" ? DALI_RGB : DALI_YCbCr;

  // Create the pipeline
  Pipeline pipe(batch_size, num_thread, device_id, seed, pipelined,
                prefetched_queue_depth, async);

  OpSpec op_spec("VideoReader");
  op_spec = op_spec.AddArg("device", "gpu")
      .AddArg("sequence_length", seq_len)
      .AddArg("shard_id", 0)
      .AddArg("num_shards", 1)
      .AddArg("random_shuffle", shuffle)
      .AddArg("normalized", true)
      .AddArg("dtype", dtype)
      .AddArg("step", step)
      .AddArg("skip_cached_images", skip_cached)
      .AddArg("initial_fill", initial_fill)
      .AddArg("image_type", image_type_e)
      .AddOutput("sequences_out", "gpu");

  if (reader_type != "filenames") {
    op_spec.AddArg(reader_type, files);
    op_spec.AddOutput("labels", "gpu");
  } else {
    op_spec.AddArg(reader_type, std::vector<std::string>{files});
  }

  pipe.AddOperator(op_spec);

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"sequences_out", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.RunCPU();
      pipe.RunGPU();
    }
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations && pipelined) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches*seq_len,
      benchmark::Counter::kIsRate);
}

static void PipeArgs(benchmark::internal::Benchmark *b) {
  const int batch_size = get_env_int("DALI_TEST_VIDEO_READER_BATCH_SIZE", 4);
  const int nthreads = get_env_int("DALI_TEST_VIDEO_READER_NUM_THREADS", 2);
  for (int executor = 2; executor < 3; ++executor) {
    for (int num_batch = batch_size; num_batch <= batch_size; ++num_batch) {
      for (int num_thread = nthreads; num_thread <= nthreads; ++num_thread) {
        for (auto &dtype : {DALI_FLOAT/*, DALI_UINT8*/}) {
          b->Args({executor, batch_size, num_thread, dtype});
        }
      }
    }
  }
}

BENCHMARK_REGISTER_F(VideoReaderBench, VideoPipe)->Iterations(
  get_env_int("DALI_TEST_VIDEO_BENCH_ITER", 100))
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

}  // namespace dali
