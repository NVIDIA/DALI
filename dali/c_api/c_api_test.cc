// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/c_api.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"

using namespace std::string_literals;  // NOLINT(build/namespaces)

namespace dali {

namespace {

using BACKEND = CPUBackend; // TODO remove

constexpr int batch_size = 12;
constexpr int num_thread = 4;
constexpr int device_id = 0;
constexpr int seed = 0;
constexpr bool pipelined = true;
constexpr int prefetch_queue_depth = 2;
constexpr bool async = true;
constexpr float output_size = 20.f;
const std::string input_name = "inputs"s;  // NOLINT

template<typename Backend>
struct backend_to_device_type {
  static constexpr device_type_t value = CPU;
};

template<>
struct backend_to_device_type<GPUBackend> {
  static constexpr device_type_t value = GPU;
};


template<typename Backend, device_type_t execution_device = backend_to_device_type<Backend>::value>
std::unique_ptr<Pipeline> GetTestPipeline(bool is_file_reader, const std::string &output_device) {
  auto pipe_ptr = std::make_unique<Pipeline>(batch_size, num_thread, device_id, seed, pipelined,
                                             prefetch_queue_depth, async);
  auto &pipe = *pipe_ptr;
  std::string exec_device = execution_device == CPU ? "cpu" : "gpu";
  TensorList<Backend> data;
  if (is_file_reader) {
    std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
    std::string file_list = file_root + "image_list.txt";
    pipe.AddOperator(OpSpec("FileReader")
                             .AddArg("device", execution_device == CPU ? "cpu"s : "mixed"s)
                             .AddArg("file_root", file_root)
                             .AddArg("file_list", file_list)
                             .AddOutput("compressed_images",exec_device)
                             .AddOutput("labels", exec_device));

    pipe.AddOperator(OpSpec("ImageDecoder")
                             .AddArg("device", exec_device)
                             .AddArg("output_type", DALI_RGB)
                             .AddInput("compressed_images", exec_device)
                             .AddOutput(input_name, exec_device));
  } else {
    pipe.AddExternalInput(input_name);
  }
  //  Some Op
  pipe.AddOperator(OpSpec("Resize")
                           .AddArg("device", exec_device)
                           .AddArg("image_type", DALI_RGB)
                           .AddArg("resize_x", output_size)
                           .AddArg("resize_y", output_size)
                           .AddInput(input_name, exec_device)
                           .AddOutput("outputs", exec_device));

  vector<std::pair<string, string>> outputs = {{"outputs", output_device}};

  pipe.SetOutputNames(outputs);
  return pipe_ptr;
}


// Takes Outptus from baseline and handle and compares them
// Allows only for uint8_t CPU/GPU output data to be compared
template<typename Backend>
void ComparePipelinesOutputs(daliPipelineHandle &handle, Pipeline &baseline) {
  dali::DeviceWorkspace ws;
  baseline.Outputs(&ws);
  daliOutput(&handle);

  EXPECT_EQ(daliGetNumOutput(&handle), ws.NumOutput());
  const int num_output = ws.NumOutput();
  TensorList<Backend> c_output;
  for (int output = 0; output < num_output; output++) {
    EXPECT_EQ(daliNumTensors(&handle, output), batch_size);
    for (int elem = 0; elem < batch_size; elem++) {
      auto *shape = daliShapeAtSample(&handle, output, elem);
      int idx = 0;
      auto ref_shape = ws.Output<Backend>(output).shape()[idx];
      for (; shape[idx] != 0; idx++) {
        EXPECT_EQ(shape[idx], ref_shape[idx]);
      }
      EXPECT_EQ(idx, ref_shape.sample_dim());
      free(shape);
    }

    TensorList<Backend> c_output;
    auto &regular_output = ws.Output<Backend>(0);
    c_output.Resize(regular_output.shape(), TypeInfo::Create<uint8_t>());
    daliCopyTensorListNTo(&handle, c_output.raw_mutable_data(), 0,
                          backend_to_device_type<Backend>::value, 0, false);
    Check(view<uint8_t>(c_output), view<uint8_t>(regular_output));
  }
}

}  // namespace

TEST(CApiTest, FileReaderPipe) {
  auto pipe_ptr = GetTestPipeline<BACKEND>(true, "cpu");
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();
  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth);
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<BACKEND>(handle, *pipe_ptr);
  }

  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<BACKEND>(handle, *pipe_ptr);
}

TEST(CApiTest, ExternalSourceSingleAllocPipe) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<BACKEND> input;
  input.Resize(input_shape, TypeInfo::Create<uint8_t>());
  auto pipe_ptr = GetTestPipeline<BACKEND>(false, "cpu");
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input), 42 * i);
    pipe_ptr->SetExternalInput(input_name, input);
    daliSetExternalInput(&handle, input_name.c_str(), backend_to_device_type<BACKEND>::value, input.raw_data(),
                         dali_data_type_t::DALI_UINT8, input_shape.data(), input_shape.sample_dim(),
                         nullptr);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<BACKEND>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input), 42 * prefetch_queue_depth);
  pipe_ptr->SetExternalInput(input_name, input);
  daliSetExternalInput(&handle, input_name.c_str(), backend_to_device_type<CPUBackend>::value, input.raw_data(),
                       dali_data_type_t::DALI_UINT8, input_shape.data(), input_shape.sample_dim(),
                       "HWC");
  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<BACKEND>(handle, *pipe_ptr);
}

TEST(CApiTest, ExternalSourceMultipleAllocPipe) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<BACKEND> input;
  input.Resize(input_shape, TypeInfo::Create<uint8_t>());
  std::vector<const void *> data_ptrs(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_ptrs[i] = input.raw_tensor(i);
  }
  auto pipe_ptr = GetTestPipeline<BACKEND>(false, "cpu");
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input), 42 * i);

    pipe_ptr->SetExternalInput(input_name, input);
    daliSetExternalInputTensors(&handle, input_name.c_str(), device_type_t::CPU, data_ptrs.data(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->RunCPU();
    pipe_ptr->RunGPU();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::DeviceWorkspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<BACKEND>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input), 42 * prefetch_queue_depth);
  pipe_ptr->SetExternalInput(input_name, input);
  daliSetExternalInputTensors(&handle, input_name.c_str(), device_type_t::CPU, data_ptrs.data(),
                              dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), "HWC");
  daliRun(&handle);
  pipe_ptr->RunCPU();
  pipe_ptr->RunGPU();

  ComparePipelinesOutputs<BACKEND>(handle, *pipe_ptr);
}

}  // namespace dali
