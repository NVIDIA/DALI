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
#include "dali/pipeline/operators/crop/crop.h"
#include "dali/pipeline/operators/crop/new_crop.h"
#include "dali/benchmark/operator_bench.h"
#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

namespace dali {

class CropBench : public DALIBenchmark {
 public:
  void CropPipelineTest(benchmark::State& st,
                        int batch_size,
                        int num_thread,
                        std::string output_device,
                        OpSpec crop_operator,
                        std::function<void(Pipeline&)> add_other_inputs = {}) {
    DALIImageType img_type = DALI_RGB;

    // Create the pipeline
    Pipeline pipe(
        batch_size,
        num_thread,
        0, -1,
        true,   // pipelined
        2,      // pipe length
        true);  // async

    TensorList<CPUBackend> data;
    this->MakeJPEGBatch(&data, batch_size);
    pipe.AddExternalInput("raw_jpegs");
    pipe.SetExternalInput("raw_jpegs", data);

    if (add_other_inputs)
      add_other_inputs(pipe);

    if (output_device == "cpu") {
      pipe.AddOperator(
        OpSpec("HostDecoder")
          .AddArg("device", "cpu")
          .AddArg("output_type", img_type)
          .AddInput("raw_jpegs", "cpu")
          .AddOutput("decoded", "cpu"));
    } else {
      pipe.AddOperator(
        OpSpec("nvJPEGDecoder")
          .AddArg("device", "mixed")
          .AddArg("output_type", img_type)
          .AddArg("max_streams", num_thread)
          .AddArg("use_batched_decode", false)
          .AddArg("cache_size", 1000)  // megabytes
          .AddArg("cache_type", "largest")
          .AddArg("cache_debug", false)
          .AddInput("raw_jpegs", "cpu")
          .AddOutput("decoded", "gpu"));
    }

    pipe.AddOperator(crop_operator);

    // Build and run the pipeline
    vector<std::pair<string, string>> outputs = {{"images", output_device}};
    pipe.Build(outputs);

    // Run once to allocate the memory
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    while (st.KeepRunning()) {
      if (st.iterations() == 1) {
        // We will start the processing for the next batch
        // immediately after issueing work to the gpu to
        // pipeline the cpu/copy/gpu work
        pipe.RunCPU();
        pipe.RunGPU();
      }
      pipe.RunCPU();
      pipe.RunGPU();
      pipe.Outputs(&ws);

      if (st.iterations() == st.max_iterations) {
        // Block for the last batch to finish
        pipe.Outputs(&ws);
      }
    }

    // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
    int num_batches = st.iterations() + 1;
    st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
        benchmark::Counter::kIsRate);
  }
};

static void PipeArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
    for (int num_thread = 1; num_thread <= 4; ++num_thread) {
      b->Args({batch_size, num_thread});
    }
  }
}

BENCHMARK_DEFINE_F(CropBench, OldCropGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->CropPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("Crop")
      .AddArg("device", "gpu")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("decoded", "gpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(CropBench, OldCropGPU)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(CropBench, NewCropGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->CropPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("NewCrop")
      .AddArg("device", "gpu")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("decoded", "gpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(CropBench, NewCropGPU)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(CropBench, OldCropCPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->CropPipelineTest(
    st, batch_size, num_thread, "cpu",
    OpSpec("Crop")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("decoded", "cpu")
      .AddOutput("images", "cpu"));
}

BENCHMARK_REGISTER_F(CropBench, OldCropCPU)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(CropBench, NewCropCPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->CropPipelineTest(
    st, batch_size, num_thread, "cpu",
    OpSpec("NewCrop")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("decoded", "cpu")
      .AddOutput("images", "cpu"));
}

BENCHMARK_REGISTER_F(CropBench, NewCropCPU)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(OperatorBench, OldCropCPU_StandAlone)(benchmark::State& st) {
  this->RunCPU<uint8_t>(
    st,
    OpSpec("Crop")
      .AddArg("batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddArg("crop_pos_x", 0.5f)
      .AddArg("crop_pos_y", 0.5f)
  );
}

BENCHMARK_REGISTER_F(OperatorBench, OldCropCPU_StandAlone)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime();

BENCHMARK_DEFINE_F(OperatorBench, NewCropCPU_StandAlone)(benchmark::State& st) {
  this->RunCPU<uint8_t>(
    st,
    OpSpec("NewCrop")
      .AddArg("batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddArg("crop_pos_x", 0.5f)
      .AddArg("crop_pos_y", 0.5f)
  );
}

BENCHMARK_REGISTER_F(OperatorBench, NewCropCPU_StandAlone)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime();

}  // namespace dali
