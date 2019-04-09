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

#include <benchmark/benchmark.h>

#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

namespace dali {

class DecoderBench : public DALIBenchmark {
 public:
  void DecoderPipelineTest(benchmark::State& st,
                           int batch_size,
                           int num_thread,
                           std::string output_device,
                           OpSpec decoder_operator,
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

    pipe.AddOperator(decoder_operator);

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

BENCHMARK_DEFINE_F(DecoderBench, HostDecoder)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "cpu",
    OpSpec("HostDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoder)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoder)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("hybrid_huffman_threshold", std::numeric_limits<unsigned int>::max())
      .AddArg("use_batched_decode", false)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoder)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderSplitStages)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("hybrid_huffman_threshold", std::numeric_limits<unsigned int>::max())
      .AddArg("use_batched_decode", false)
      .AddArg("split_stages", true)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderSplitStages)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);


BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderCachedThreshold)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("use_batched_decode", false)
      .AddArg("cache_size", 1000)  // megabytes
      .AddArg("cache_threshold", 250*250*3)
      .AddArg("cache_type", "threshold")
      .AddArg("cache_debug", true)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderCachedThreshold)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderCachedLargest)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("use_batched_decode", false)
      .AddArg("cache_size", 1000)  // megabytes
      .AddArg("cache_type", "largest")
      .AddArg("cache_debug", true)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderCachedLargest)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderBatched)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("use_batched_decode", true)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderBatched)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, HostDecoderRandomCrop)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "cpu",
    OpSpec("HostDecoderRandomCrop")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoderRandomCrop)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, HostDecoderCrop)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "cpu",
    OpSpec("HostDecoderCrop")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoderCrop)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, HostDecoderSlice)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  vector<Dims> shape(batch_size, {2});
  TensorList<CPUBackend> begin_data;
  begin_data.set_type(TypeInfo::Create<float>());
  begin_data.Resize(shape);
  float crop_x = 0.25f, crop_y = 0.124f;
  for (int k = 0; k < batch_size; k++) {
    begin_data.mutable_tensor<float>(k)[0] = crop_x;
    begin_data.mutable_tensor<float>(k)[1] = crop_y;
  }

  TensorList<CPUBackend> crop_data;
  float crop_w = 0.5f, crop_h = 0.25f;
  crop_data.set_type(TypeInfo::Create<float>());
  crop_data.Resize(shape);
  for (int k = 0; k < batch_size; k++) {
    crop_data.mutable_tensor<float>(k)[0] = crop_w;
    crop_data.mutable_tensor<float>(k)[1] = crop_h;
  }

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "cpu",
    OpSpec("HostDecoderSlice")
      .AddArg("device", "cpu")
      .AddArg("output_type", DALI_RGB)
      .AddInput("raw_jpegs", "cpu")
      .AddInput("begin_data", "cpu")
      .AddInput("crop_data", "cpu")
      .AddOutput("images", "cpu"),
    [&begin_data, &crop_data](Pipeline& pipe) {
      pipe.AddExternalInput("begin_data");
      pipe.SetExternalInput("begin_data", begin_data);
      pipe.AddExternalInput("crop_data");
      pipe.SetExternalInput("crop_data", crop_data);
    });
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoderSlice)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderRandomCrop)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoderRandomCrop")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderRandomCrop)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderCrop)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoderCrop")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderCrop)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderSlice)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  vector<Dims> shape(batch_size, {2});
  TensorList<CPUBackend> begin_data;
  begin_data.set_type(TypeInfo::Create<float>());
  begin_data.Resize(shape);
  float crop_x = 0.25f, crop_y = 0.124f;
  for (int k = 0; k < batch_size; k++) {
    begin_data.mutable_tensor<float>(k)[0] = crop_x;
    begin_data.mutable_tensor<float>(k)[1] = crop_y;
  }

  TensorList<CPUBackend> crop_data;
  float crop_w = 0.5f, crop_h = 0.25f;
  crop_data.set_type(TypeInfo::Create<float>());
  crop_data.Resize(shape);
  for (int k = 0; k < batch_size; k++) {
    crop_data.mutable_tensor<float>(k)[0] = crop_w;
    crop_data.mutable_tensor<float>(k)[1] = crop_h;
  }

  this->DecoderPipelineTest(
    st, batch_size, num_thread, "gpu",
    OpSpec("nvJPEGDecoderSlice")
      .AddArg("device", "mixed")
      .AddArg("output_type", DALI_RGB)
      .AddInput("raw_jpegs", "cpu")
      .AddInput("begin_data", "cpu")
      .AddInput("crop_data", "cpu")
      .AddOutput("images", "gpu"),
    [&begin_data, &crop_data](Pipeline& pipe) {
      pipe.AddExternalInput("begin_data");
      pipe.SetExternalInput("begin_data", begin_data);
      pipe.AddExternalInput("crop_data");
      pipe.SetExternalInput("crop_data", crop_data);
    });
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderSlice)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

}  // namespace dali
