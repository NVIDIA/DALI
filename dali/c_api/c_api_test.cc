// Copyright (c) 2020-2022, 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"

using namespace std::string_literals;  // NOLINT(build/namespaces)

namespace dali {

namespace {

constexpr int batch_size = 12;
constexpr int num_thread = 4;
constexpr int device_id = 0;
constexpr int seed = 0;
constexpr bool pipelined = true;
constexpr int prefetch_queue_depth = 2;
constexpr bool async = true;
constexpr float output_size = 20.f;
constexpr cudaStream_t cuda_stream = 0;
const std::string input_name = "inputs"s;    // NOLINT
const std::string output_name = "outputs"s;  // NOLINT

template<typename Backend>
struct backend_to_device_type {
  static constexpr device_type_t value = CPU;
};

template<>
struct backend_to_device_type<GPUBackend> {
  static constexpr device_type_t value = GPU;
};

template<typename Backend>
struct the_other_backend {
  using type = GPUBackend;
};

template<>
struct the_other_backend<GPUBackend> {
  using type = CPUBackend;
};


std::string GetDeviceStr(device_type_t dev) {
  return dev == CPU ? "cpu" : "gpu";
}

// Allocates a buffer on the specified backend and, if necessary, another one for the CPU
template <typename Backend>
std::pair<shared_ptr<uint8_t>, shared_ptr<uint8_t>> AllocBufferPair(size_t bytes, bool pinned) {
  auto buffer = AllocBuffer<Backend>(bytes, pinned);
  if constexpr (std::is_same_v<Backend, CPUBackend>) {
    return std::make_pair(buffer, buffer);
  } else {
    return std::make_pair(buffer, AllocBuffer<CPUBackend>(bytes, pinned));
  }
}

void CopyIfDifferent(void *dest, const void *src, size_t bytes, cudaStream_t stream) {
  if (dest != src)
    MemCopy(dest, src, bytes, stream);
}


template<typename Backend, device_type_t execution_device = backend_to_device_type<Backend>::value>
std::unique_ptr<Pipeline> GetTestPipeline(bool is_file_reader, const std::string &output_device) {
  int dev = output_device == "cpu" ? CPU_ONLY_DEVICE_ID : device_id;
  auto pipe_ptr = std::make_unique<Pipeline>(batch_size, num_thread, dev, seed, pipelined,
                                             prefetch_queue_depth, async);
  auto &pipe = *pipe_ptr;
  std::string exec_device = GetDeviceStr(execution_device);
  if (is_file_reader) {
    std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
    std::string file_list = file_root + "image_list.txt";
    pipe.AddOperator(OpSpec("FileReader")
                             .AddArg("device", "cpu")
                             .AddArg("file_root", file_root)
                             .AddArg("file_list", file_list)
                             .AddOutput("compressed_images", "cpu")
                             .AddOutput("labels", "cpu"));

    pipe.AddOperator(OpSpec("ImageDecoder")
                             .AddArg("device", "cpu")
                             .AddArg("output_type", DALI_RGB)
                             .AddInput("compressed_images", "cpu")
                             .AddOutput(input_name, "cpu"));
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
                           .AddOutput(output_name, exec_device));

  std::vector<std::pair<std::string, std::string>> outputs = {{output_name, output_device}};

  pipe.SetOutputDescs(outputs);
  return pipe_ptr;
}


std::unique_ptr<Pipeline> GetExternalSourcePipeline(bool no_copy, const std::string &device) {
  int dev = device == "cpu" ? CPU_ONLY_DEVICE_ID : device_id;
  auto pipe_ptr = std::make_unique<Pipeline>(batch_size, num_thread, dev, seed, pipelined,
                                             prefetch_queue_depth, async);
  auto &pipe = *pipe_ptr;

  pipe.AddOperator(OpSpec("ExternalSource")
                       .AddArg("device", device)
                       .AddArg("name", input_name)
                       .AddArg("no_copy", no_copy)
                       .AddOutput(input_name, device), input_name);
  //  Some Op
  pipe.AddOperator(OpSpec("Resize")
                       .AddArg("device", device)
                       .AddArg("image_type", DALI_RGB)
                       .AddArg("resize_x", output_size)
                       .AddArg("resize_y", output_size)
                       .AddInput(input_name, device)
                       .AddOutput(output_name, device));

  std::vector<std::pair<std::string, std::string>> outputs = {{output_name, device}};

  pipe.SetOutputDescs(outputs);
  return pipe_ptr;
}


// Takes Outputs from baseline and handle and compares them
// Allows only for uint8_t CPU/GPU output data to be compared
template <typename Backend>
void ComparePipelinesOutputs(daliPipelineHandle &handle, Pipeline &baseline,
                             unsigned int copy_output_flags = DALI_ext_default,
                             int batch_size = dali::batch_size) {
  dali::Workspace ws;
  baseline.Outputs(&ws);
  daliOutput(&handle);

  EXPECT_EQ(daliGetNumOutput(&handle), ws.NumOutput());
  const int num_output = ws.NumOutput();
  for (int output = 0; output < num_output; output++) {
    EXPECT_EQ(daliNumTensors(&handle, output), batch_size);
    for (int elem = 0; elem < batch_size; elem++) {
      auto *shape = daliShapeAtSample(&handle, output, elem);
      auto ref_shape = ws.Output<Backend>(output).shape()[elem];
      int D = ref_shape.size();
      for (int d = 0; d < D; d++)
        EXPECT_EQ(shape[d], ref_shape[d]);
      EXPECT_EQ(shape[D], 0) << "Shapes in C API are 0-terminated";
      free(shape);
    }

    TensorList<CPUBackend> pipeline_output_cpu, c_api_output_cpu;
    pipeline_output_cpu.set_pinned(false);
    c_api_output_cpu.set_pinned(false);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    AccessOrder order = std::is_same_v<Backend, GPUBackend> ? AccessOrder(cuda_stream)
                                                            : AccessOrder::host();
    pipeline_output_cpu.Copy(ws.Output<Backend>(0), order);

    auto num_elems = pipeline_output_cpu.shape().num_elements();
    auto [backend_buf, cpu_buf] = AllocBufferPair<Backend>(num_elems, false);

    daliOutputCopy(&handle, backend_buf.get(), 0,
                   backend_to_device_type<Backend>::value, 0, copy_output_flags);

    CopyIfDifferent(cpu_buf.get(), backend_buf.get(), num_elems, cuda_stream);
    if (std::is_same_v<Backend, GPUBackend>)
      CUDA_CALL(cudaDeviceSynchronize());
    Check(view<uint8_t>(pipeline_output_cpu),
          TensorListView<StorageCPU, uint8_t>(cpu_buf.get(), pipeline_output_cpu.shape()));
  }
}

}  // namespace

template<typename Backend>
class CApiTest : public ::testing::Test {
 protected:
  CApiTest() {
    constexpr bool is_device = std::is_same_v<Backend, GPUBackend>;
    output_device_ = is_device ? "gpu" : "cpu";
    device_id_ = is_device ? device_id : CPU_ONLY_DEVICE_ID;
    order_ = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  }

  std::string output_device_;
  int device_id_;
  AccessOrder order_;
};

using Backends = ::testing::Types<CPUBackend, GPUBackend>;
TYPED_TEST_SUITE(CApiTest, Backends);


TYPED_TEST(CApiTest, GetOutputNameTest) {
  std::string output0_name = "compressed_images";
  std::string output1_name = "labels";
  auto pipe_ptr = std::make_unique<Pipeline>(batch_size, num_thread, this->device_id_,
                                            seed, pipelined, prefetch_queue_depth, async);
  auto &pipe = *pipe_ptr;
  std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
  std::string file_list = file_root + "image_list.txt";
  pipe.AddOperator(OpSpec("FileReader")
                       .AddArg("device", "cpu")
                       .AddArg("file_root", file_root)
                       .AddArg("file_list", file_list)
                       .AddOutput(output0_name, "cpu")
                       .AddOutput(output1_name, "cpu"));

  std::vector<std::pair<std::string, std::string>> outputs = {{output0_name, "cpu"},
                                                              {output1_name, "cpu"}};

  pipe.SetOutputDescs(outputs);

  auto serialized = pipe.SerializeToProtobuf();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  ASSERT_EQ(daliGetNumOutput(&handle), 2);
  EXPECT_STREQ(daliGetOutputName(&handle, 0), output0_name.c_str());
  EXPECT_STREQ(daliGetOutputName(&handle, 1), output1_name.c_str());

  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, FileReaderPipe) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();
  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  daliRun(&handle);
  pipe_ptr->Run();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  daliDeletePipeline(&handle);
}

TYPED_TEST(CApiTest, FileReaderDefaultPipe) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();
  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }

  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, serialized.c_str(), serialized.size());
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  daliRun(&handle);
  pipe_ptr->Run();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ExternalSourceSingleAllocPipe) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  auto num_elems = input_shape.num_elements();

  auto [input, input_cpu] = AllocBufferPair<TypeParam>(num_elems, false);
  TensorList<TypeParam> input_wrapper;

  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape), 42 * i);
    CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);
    input_wrapper.ShareData(std::static_pointer_cast<void>(input), num_elems,
                            false, input_shape, DALI_UINT8, this->device_id_);
    pipe_ptr->SetExternalInput(input_name, input_wrapper);
    daliSetExternalInputBatchSize(&handle, input_name.c_str(), input_shape.num_samples());
    daliSetExternalInputAsync(&handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                              input.get(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), nullptr, cuda_stream, DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }
  daliPrefetch(&handle);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape),
                 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);
  input_wrapper.ShareData(std::static_pointer_cast<void>(input), num_elems,
                          false, input_shape, DALI_UINT8, this->device_id_);
  pipe_ptr->SetExternalInput(input_name, input_wrapper);
  daliSetExternalInputAsync(&handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                            input.get(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                            input_shape.sample_dim(), "HWC", cuda_stream, DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->Run();

  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ExternalSourceSingleAllocVariableBatchSizePipe) {
  TensorListShape<> reference_input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                             {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                             {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  int max_batch_size = reference_input_shape.num_samples();
  std::vector<TensorListShape<>> trimmed_input_shapes = {
      sample_range(reference_input_shape, 0, max_batch_size / 2),
      sample_range(reference_input_shape, 0, max_batch_size / 4),
      sample_range(reference_input_shape, 0, max_batch_size),
  };

  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (auto &input_shape : trimmed_input_shapes) {
    pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
    pipe_ptr->Build();

    auto num_elems = input_shape.num_elements();

    auto [input, input_cpu] = AllocBufferPair<TypeParam>(num_elems, false);
    TensorList<TypeParam> input_wrapper;

    for (int i = 0; i < prefetch_queue_depth; i++) {
      SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape), 42 * i);
      // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
      CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);
      input_wrapper.ShareData(std::static_pointer_cast<void>(input), num_elems,
                              false, input_shape, DALI_UINT8, this->device_id_);
      pipe_ptr->SetExternalInput(input_name, input_wrapper);
      daliSetExternalInputBatchSize(&handle, input_name.c_str(), input_shape.num_samples());
      daliSetExternalInputAsync(&handle, input_name.c_str(),
                                backend_to_device_type<TypeParam>::value, input.get(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr, cuda_stream, DALI_ext_default);
    }

    for (int i = 0; i < prefetch_queue_depth; i++) {
      pipe_ptr->Run();
    }
    daliPrefetch(&handle);

    dali::Workspace ws;
    for (int i = 0; i < prefetch_queue_depth; i++) {
      ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr, DALI_ext_default,
                                         input_shape.num_samples());
    }
  }
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ExternalSourceMultipleAllocPipe) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  TensorList<TypeParam> input;
  input_cpu.set_pinned(false);
  input_cpu.Resize(input_shape, DALI_UINT8);
  input.set_pinned(false);
  std::vector<const void *> data_ptrs(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_ptrs[i] = input_cpu.raw_tensor(i);
  }
  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, this->order_);
    pipe_ptr->SetExternalInput(input_name, input, this->order_);
    daliSetExternalInputTensorsAsync(&handle, input_name.c_str(),
                                     backend_to_device_type<TypeParam>::value, data_ptrs.data(),
                                     dali_data_type_t::DALI_UINT8, input_shape.data(),
                                     input_shape.sample_dim(), nullptr, this->order_.stream(),
                                     DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }
  daliPrefetch(&handle);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input_cpu), 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  input.Copy(input_cpu, this->order_);
  pipe_ptr->SetExternalInput(input_name, input, this->order_);
  daliSetExternalInputTensorsAsync(&handle, input_name.c_str(),
                                   backend_to_device_type<TypeParam>::value, data_ptrs.data(),
                                   dali_data_type_t::DALI_UINT8, input_shape.data(),
                                   input_shape.sample_dim(), "HWC", cuda_stream, DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->Run();
  ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ExternalSourceSingleAllocDifferentBackendsTest) {
  using OpBackend = TypeParam;
  using DataBackend = typename the_other_backend<TypeParam>::type;
  if (std::is_same_v<OpBackend, CPUBackend> && std::is_same_v<DataBackend, GPUBackend>) {
    GTEST_SKIP();  // GPU data -> CPU op   is currently not supported. Might be added later.
  }
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8,  8,  3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  auto num_elems = input_shape.num_elements();

  auto [input, input_cpu] = AllocBufferPair<DataBackend>(num_elems, false);
  TensorList<DataBackend> input_wrapper;

  auto pipe_ptr = GetTestPipeline<OpBackend>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape), 42 * i);
    CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    input_wrapper.ShareData(std::static_pointer_cast<void>(input), num_elems,
                            false, input_shape, DALI_UINT8, this->device_id_);
    pipe_ptr->SetExternalInput(input_name, input_wrapper);
    daliSetExternalInput(&handle, input_name.c_str(), backend_to_device_type<DataBackend>::value,
                         input.get(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                         input_shape.sample_dim(), nullptr, DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }
  daliPrefetch(&handle);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
  }

  SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape),
                  42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  input_wrapper.ShareData(std::static_pointer_cast<void>(input), num_elems,
                          false, input_shape, DALI_UINT8, this->device_id_);
  pipe_ptr->SetExternalInput(input_name, input_wrapper);
  daliSetExternalInput(&handle, input_name.c_str(), backend_to_device_type<DataBackend>::value,
                        input.get(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                        input_shape.sample_dim(), "HWC", DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->Run();
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ExternalSourceMultipleAllocDifferentBackendsTest) {
  using OpBackend = TypeParam;
  using DataBackend = typename the_other_backend<TypeParam>::type;
  if (std::is_same_v<OpBackend, CPUBackend> && std::is_same_v<DataBackend, GPUBackend>) {
    GTEST_SKIP();  // GPU data -> CPU op   is currently not supported. Might be added later.
  }
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8,  8,  3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  TensorList<CPUBackend> input_cpu;
  TensorList<DataBackend> input;
  input_cpu.Resize(input_shape, DALI_UINT8);
  std::vector<const void *> data_ptrs(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_ptrs[i] = input_cpu.raw_tensor(i);
  }
  auto pipe_ptr = GetTestPipeline<OpBackend>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(view<uint8_t>(input_cpu), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    input.Copy(input_cpu, std::is_same_v<DataBackend, CPUBackend>
                          ? AccessOrder::host()
                          : this->order_);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    pipe_ptr->SetExternalInput(input_name, input, cuda_stream);
    daliSetExternalInputTensors(&handle, input_name.c_str(),
                                backend_to_device_type<DataBackend>::value, data_ptrs.data(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr, DALI_ext_default);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }
  daliPrefetch(&handle);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
  }

  SequentialFill(view<uint8_t>(input_cpu), 42 * prefetch_queue_depth);
  // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
  input.Copy(input_cpu, std::is_same_v<DataBackend, CPUBackend>
                        ? AccessOrder::host()
                        : this->order_);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
  pipe_ptr->SetExternalInput(input_name, input, cuda_stream);
  daliSetExternalInputTensors(&handle, input_name.c_str(),
                              backend_to_device_type<DataBackend>::value, data_ptrs.data(),
                              dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), "HWC", DALI_ext_default);
  daliRun(&handle);
  pipe_ptr->Run();
  ComparePipelinesOutputs<OpBackend>(handle, *pipe_ptr);
  daliDeletePipeline(&handle);
}

TYPED_TEST(CApiTest, TestExecutorMeta) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr.reset();
  daliPipelineHandle handle;
  daliCreatePipeline2(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                      this->device_id_, false, false, false,
                      prefetch_queue_depth, prefetch_queue_depth, prefetch_queue_depth, true);

  daliRun(&handle);
  daliOutput(&handle);
  if (std::is_same_v<TypeParam, GPUBackend>)
    CUDA_CALL(cudaDeviceSynchronize());

  size_t N;
  daliExecutorMetadata *meta;
  daliGetExecutorMetadata(&handle, &meta, &N);

  // File Reader -> Image Decoder -> [Copy to Gpu] -> Resize -> Make Contiguous (always for outputs)
  if (std::is_same_v<TypeParam, CPUBackend>) {
    EXPECT_EQ(N, 4);
  } else {
    EXPECT_EQ(N, 5);
  }

  for (size_t i = 0; i< N; ++i) {
    auto &meta_entry = meta[i];
    for (size_t j = 0; j < meta_entry.out_num; ++j) {
      EXPECT_LE(meta_entry.real_size[j], meta_entry.reserved[j]);
    }
  }
  daliFreeExecutorMetadata(meta, N);
  daliDeletePipeline(&handle);
}

TYPED_TEST(CApiTest, UseCopyKernel) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  auto num_elems = input_shape.num_elements();
  auto [input, input_cpu] = AllocBufferPair<TypeParam>(num_elems, true);

  TensorList<TypeParam> input_wrapper;
  if (std::is_same_v<TypeParam, CPUBackend>) {
    input_wrapper.set_pinned(true);
  }

  auto pipe_ptr = GetTestPipeline<TypeParam>(false, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();
  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  unsigned int flags = DALI_ext_default | DALI_ext_force_sync | DALI_use_copy_kernel;
  if (std::is_same_v<TypeParam, CPUBackend>)
    flags |= DALI_ext_pinned;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape), 42 * i);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);
    input_wrapper.ShareData(std::static_pointer_cast<void>(input), num_elems,
                            std::is_same_v<TypeParam, CPUBackend>, input_shape, DALI_UINT8,
                            this->device_id_);
    pipe_ptr->SetExternalInput(input_name, input_wrapper);
    daliSetExternalInputAsync(&handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                              input.get(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                              input_shape.sample_dim(), nullptr, cuda_stream, flags);
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr, flags);
  }
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ForceNoCopyFail) {
  this->device_id_ = device_id;  // we need both backends here
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  auto num_elems = input_shape.num_elements();

  auto [input, input_cpu] = AllocBufferPair<TypeParam>(num_elems, false);

  auto device = backend_to_device_type<TypeParam>::value;
  std::string device_str = GetDeviceStr(device);

  auto other_device = device == CPU ? GPU : CPU;
  std::string other_device_str = GetDeviceStr(other_device);

  auto pipe_ptr = GetExternalSourcePipeline(false, other_device_str);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     this->device_id_, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

    SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape), 42);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    CopyIfDifferent(input.get(), input_cpu.get(), num_elems, cuda_stream);

  // Try to fill the pipeline placed on "other_device" with data placed on the current "device"
  // while forcing NO COPY. It's not allowed to do a no copy across backends and it should error
  // out.
  ASSERT_THROW(daliSetExternalInputAsync(
                    &handle, input_name.c_str(), backend_to_device_type<TypeParam>::value,
                    input.get(), dali_data_type_t::DALI_UINT8, input_shape.data(),
                    input_shape.sample_dim(), nullptr, cuda_stream, DALI_ext_force_no_copy),
                std::runtime_error);
  daliDeletePipeline(&handle);
}


template <typename TypeParam>
void TestForceFlagRun(bool ext_src_no_copy, unsigned int flag_to_test, int device_id) {
  TensorListShape<> input_shape = {{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3},
                                   {64, 32, 3}, {32, 64, 3}, {20, 20, 3}, {64, 64, 3},
                                   {10, 10, 3}, {60, 50, 3}, {10, 15, 3}, {48, 48, 3}};
  auto num_elems = input_shape.num_elements();

  auto input_cpu = AllocBuffer<CPUBackend>(num_elems, false);

  auto device = backend_to_device_type<TypeParam>::value;
  std::string device_str = GetDeviceStr(device);

  auto pipe_ptr = GetExternalSourcePipeline(ext_src_no_copy, device_str);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  pipe_ptr->Build();

  daliPipelineHandle handle;
  daliCreatePipeline(&handle, serialized.c_str(), serialized.size(), batch_size, num_thread,
                     device_id, false, prefetch_queue_depth, prefetch_queue_depth,
                     prefetch_queue_depth, false);

  std::vector<std::shared_ptr<uint8_t>> data;
  data.reserve(prefetch_queue_depth);
  for (int i = 0; i < prefetch_queue_depth; i++) {
    data.push_back(AllocBuffer<TypeParam>(num_elems, false));
  }
  std::vector<TensorList<TypeParam>> input_wrapper(prefetch_queue_depth);

  for (int i = 0; i < prefetch_queue_depth; i++) {
    SequentialFill(TensorListView<StorageCPU, uint8_t>(input_cpu.get(), input_shape), 42 * i);
    if constexpr (std::is_same_v<TypeParam, CPUBackend>)
      memcpy(data[i].get(), input_cpu.get(), num_elems);
    else
      MemCopy(data[i].get(), input_cpu.get(), num_elems, cuda_stream);

    input_wrapper[i].ShareData(std::static_pointer_cast<void>(data[i]), num_elems,
                               false, input_shape, DALI_UINT8, device_id);
    pipe_ptr->SetExternalInput(input_name, input_wrapper[i]);
    if (flag_to_test == DALI_ext_force_no_copy) {
      // for no copy, we just pass the view to data
      daliSetExternalInputAsync(&handle, input_name.c_str(),
                                backend_to_device_type<TypeParam>::value, data[i].get(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr, cuda_stream, flag_to_test);
    } else {
      decltype(input_cpu) tmp_data;
      if constexpr (std::is_same_v<TypeParam, CPUBackend>) {
        tmp_data = data[i];
      } else {
        tmp_data = AllocBuffer<TypeParam>(num_elems, false);
        MemCopy(tmp_data.get(), data[i].get(), num_elems, cuda_stream);
      }
      // We pass a temporary TensorList as input and force the copy
      daliSetExternalInputAsync(&handle, input_name.c_str(),
                                backend_to_device_type<TypeParam>::value, tmp_data.get(),
                                dali_data_type_t::DALI_UINT8, input_shape.data(),
                                input_shape.sample_dim(), nullptr, cuda_stream, flag_to_test);
    }
  }

  for (int i = 0; i < prefetch_queue_depth; i++) {
    pipe_ptr->Run();
  }
  daliPrefetchUniform(&handle, prefetch_queue_depth);

  dali::Workspace ws;
  for (int i = 0; i < prefetch_queue_depth; i++) {
    ComparePipelinesOutputs<TypeParam>(handle, *pipe_ptr);
  }
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, ForceCopy) {
  TestForceFlagRun<TypeParam>(true, DALI_ext_force_copy, this->device_id_);
}


TYPED_TEST(CApiTest, ForceNoCopy) {
  TestForceFlagRun<TypeParam>(false, DALI_ext_force_no_copy, this->device_id_);
}


template <typename Backend>
void Clear(Tensor<Backend>& tensor);

template <>
void Clear(Tensor<CPUBackend>& tensor) {
  std::memset(tensor.raw_mutable_data(), 0, tensor.nbytes());
}

template <>
void Clear(Tensor<GPUBackend>& tensor) {
  CUDA_CALL(cudaMemset(tensor.raw_mutable_data(), 0, tensor.nbytes()));
}


TYPED_TEST(CApiTest, daliOutputCopySamples) {
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();

  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, serialized.c_str(), serialized.size());

  daliRun(&handle);
  daliOutput(&handle);
  const int num_output = daliGetNumOutput(&handle);
  for (int out_idx = 0; out_idx < num_output; out_idx++) {
    std::vector<int64_t> sample_sizes(batch_size, 0);
    EXPECT_EQ(daliNumTensors(&handle, out_idx), batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto *shape = daliShapeAtSample(&handle, out_idx, sample_idx);
      int ndim = 0;
      sample_sizes[sample_idx] = 1;
      for (int d = 0; shape[d] > 0; d++) {
        sample_sizes[sample_idx] *= shape[d];
      }
      free(shape);
    }

    DALIDataType type = static_cast<DALIDataType>(daliTypeAt(&handle, out_idx));
    auto type_info = dali::TypeTable::GetTypeInfo(type);
    int64_t out_size = daliNumElements(&handle, out_idx);
    Tensor<TypeParam> output1;
    output1.set_pinned(false);
    output1.Resize({out_size}, type_info.id());
    daliOutputCopy(&handle, output1.raw_mutable_data(), out_idx,
                   backend_to_device_type<TypeParam>::value, 0, DALI_ext_default);
    // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
    Tensor<CPUBackend> output1_cpu;
    output1_cpu.set_pinned(false);
    output1_cpu.Copy(output1, AccessOrder::host());

    for (bool use_copy_kernel : {false, true}) {
      bool pinned = use_copy_kernel;
      Tensor<TypeParam> output2;
      Tensor<CPUBackend> output2_cpu;
      output2_cpu.set_pinned(false);
      output2.set_pinned(pinned);
      output2.Resize({out_size}, type_info.id());
      // Making sure data is cleared
      // Somehow in debug mode it can get the same raw pointer which happen to have
      // the right data in the second iteration
      Clear(output2);

      std::vector<void*> sample_dsts(batch_size);
      int64_t offset = 0;
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        sample_dsts[sample_idx] = static_cast<uint8_t*>(output2.raw_mutable_data()) + offset;
        offset += sample_sizes[sample_idx] * type_info.size();
      }

      unsigned int flags = DALI_ext_default;
      if (use_copy_kernel)
        flags |= DALI_use_copy_kernel;
      if (pinned)
        flags |= DALI_ext_pinned;

      daliOutputCopySamples(&handle, sample_dsts.data(), out_idx,
                            backend_to_device_type<TypeParam>::value, cuda_stream, flags);

      // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
      output2_cpu.Copy(output2, this->order_);
      if (std::is_same_v<TypeParam, GPUBackend>)
        CUDA_CALL(cudaDeviceSynchronize());
      Check(view<uint8_t>(output1_cpu), view<uint8_t>(output2_cpu));
    }

    for (bool use_copy_kernel : {false, true}) {
      Tensor<TypeParam> output2;
      Tensor<CPUBackend> output2_cpu;
      output2_cpu.set_pinned(false);
      output2.set_pinned(std::is_same_v<TypeParam, CPUBackend>);
      output2.Resize({out_size}, type_info.id());
      // Making sure data is cleared
      // Somehow in debug mode it can get the same raw pointer which happen to have
      // the right data in the second iteration
      Clear(output2);

      std::vector<void*> sample_dsts_even(batch_size);
      std::vector<void*> sample_dsts_odd(batch_size);
      int64_t offset = 0;
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        auto sample_ptr = static_cast<uint8_t*>(output2.raw_mutable_data()) + offset;
        if (sample_idx % 2 == 0) {
          sample_dsts_even[sample_idx] = sample_ptr;
          sample_dsts_odd[sample_idx]  = nullptr;
        } else {
          sample_dsts_even[sample_idx] = nullptr;
          sample_dsts_odd[sample_idx]  = sample_ptr;
        }
        offset += sample_sizes[sample_idx] * type_info.size();
      }

      unsigned int flags = DALI_ext_default;
      if (use_copy_kernel)
        flags |= DALI_use_copy_kernel;
      if (std::is_same_v<TypeParam, CPUBackend>)
        flags |= DALI_ext_pinned;

      daliOutputCopySamples(&handle, sample_dsts_even.data(), out_idx,
                            backend_to_device_type<TypeParam>::value, cuda_stream, flags);
      daliOutputCopySamples(&handle, sample_dsts_odd.data(), out_idx,
                            backend_to_device_type<TypeParam>::value, cuda_stream, flags);

      // Unnecessary copy in case of CPUBackend, makes the code generic across Backends
      output2_cpu.Copy(output2, this->order_);
      if (std::is_same_v<TypeParam, GPUBackend>)
        CUDA_CALL(cudaDeviceSynchronize());
      Check(view<uint8_t>(output1_cpu), view<uint8_t>(output2_cpu));
    }
  }
  daliDeletePipeline(&handle);
}


TYPED_TEST(CApiTest, IsDeserializableTest) {
  using namespace std;  // NOLINT
  vector<tuple<string /* serialized pipeline */, bool /* is deserializable? */>> test_cases;
  auto pipe_ptr = GetTestPipeline<TypeParam>(true, this->output_device_);
  auto serialized = pipe_ptr->SerializeToProtobuf();
  test_cases.emplace_back(serialized, true);
  {
    dali::Pipeline pipe(1, 1, dali::CPU_ONLY_DEVICE_ID);
    auto ser = pipe.SerializeToProtobuf();
    test_cases.emplace_back(ser, true);
  }
  test_cases.emplace_back("", false);
  test_cases.emplace_back("This can't possibly be a valid DALI pipeline.", false);

  for (const auto &test_case : test_cases) {
    const auto &str = get<0>(test_case);
    const auto &res = get<1>(test_case);
    EXPECT_EQ(daliIsDeserializable(str.c_str(), str.length()), res ? 0 : 1) << str;
  }
}


TEST(CApiTest, CpuOnlyTest) {
  dali::Pipeline pipe(1, 1, dali::CPU_ONLY_DEVICE_ID);
  pipe.AddExternalInput("dummy");
  std::vector<std::pair<std::string, std::string>> outputs = {{"dummy", "cpu"}};
  pipe.SetOutputDescs(outputs);
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());
  daliDeletePipeline(&handle);
}

TEST(CApiTest, GetBackendTest) {
  dali::Pipeline pipe(1, 1, 0);
  std::string es_gpu_name = "es_gpu";
  pipe.AddExternalInput(es_gpu_name, "gpu");
  std::string es_cpu_name = "es_cpu";
  pipe.AddExternalInput(es_cpu_name, "cpu");
  std::string cont_name = "contiguous";
  pipe.AddOperator(OpSpec("MakeContiguous")
                          .AddArg("device", "mixed")
                          .AddArg("name", cont_name)
                          .AddInput(es_cpu_name, "cpu")
                          .AddOutput(cont_name, "gpu"), cont_name);
  std::vector<std::pair<std::string, std::string>> outputs = {{es_gpu_name, "gpu"},
                                                              {cont_name, "gpu"}};
  pipe.SetOutputDescs(outputs);
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());
  EXPECT_EQ(daliGetOperatorBackend(&handle, es_cpu_name.c_str()), DALI_BACKEND_CPU);
  EXPECT_EQ(daliGetOperatorBackend(&handle, es_gpu_name.c_str()), DALI_BACKEND_GPU);
  EXPECT_EQ(daliGetOperatorBackend(&handle, cont_name.c_str()), DALI_BACKEND_MIXED);
  daliDeletePipeline(&handle);
}

TEST(CApiTest, GetESDetailsTest) {
  dali::Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("INPUT3", "cpu", DALI_FLOAT16, 3, "HWC");
  pipe.AddExternalInput("INPUT1", "gpu", DALI_UINT32, -1, "NHWC");
  pipe.AddExternalInput("INPUT2", "cpu");

  pipe.SetOutputDescs({{"INPUT3", "cpu"}, {"INPUT1", "gpu"}, {"INPUT2", "cpu"}});
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());
  EXPECT_EQ(daliGetNumExternalInput(&handle), 3);

  EXPECT_EQ(daliGetExternalInputName(&handle, 0), std::string("INPUT1"));
  EXPECT_EQ(daliGetExternalInputLayout(&handle, "INPUT1"), std::string("NHWC"));
  EXPECT_EQ(daliGetExternalInputNdim(&handle, "INPUT1"), 4);
  EXPECT_EQ(daliGetExternalInputType(&handle, "INPUT1"), DALI_UINT32);

  EXPECT_EQ(daliGetExternalInputName(&handle, 1), std::string("INPUT2"));
  EXPECT_EQ(daliGetExternalInputLayout(&handle, "INPUT2"), std::string(""));
  EXPECT_EQ(daliGetExternalInputNdim(&handle, "INPUT2"), -1);
  EXPECT_EQ(daliGetExternalInputType(&handle, "INPUT2"), DALI_NO_TYPE);

  EXPECT_EQ(daliGetExternalInputName(&handle, 2), std::string("INPUT3"));
  EXPECT_EQ(daliGetExternalInputLayout(&handle, "INPUT3"), std::string("HWC"));
  EXPECT_EQ(daliGetExternalInputNdim(&handle, "INPUT3"), 3);
  EXPECT_EQ(daliGetExternalInputType(&handle, "INPUT3"), DALI_FLOAT16);

  daliDeletePipeline(&handle);
}

TEST(CApiTest, GetMaxBatchSizeTest) {
  const int BS = 13;
  dali::Pipeline pipe(BS, 1, 0);
  pipe.AddExternalInput("INPUT", "cpu", DALI_FLOAT16, 3, "HWC");
  pipe.SetOutputDescs({{"INPUT", "cpu"}});
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());

  EXPECT_EQ(daliGetMaxBatchSize(&handle), BS);

  daliDeletePipeline(&handle);
}

TEST(CApiTest, GetDeclaredOutputDtypeNdimTest) {
  const DALIDataType dtype = DALIDataType::DALI_UINT8;
  const dali_data_type_t ref_dtype = dali_data_type_t::DALI_UINT8;
  const int ndim = 2;
  dali::Pipeline pipe(13, 1, 0);
  pipe.AddExternalInput("INPUT", "cpu", DALI_FLOAT16, 3, "HWC");
  pipe.SetOutputDescs({{"INPUT", "cpu", dtype, ndim}});
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());

  EXPECT_EQ(daliGetDeclaredOutputDtype(&handle, 0), ref_dtype);
  EXPECT_EQ(daliGetDeclaredOutputNdim(&handle, 0), ndim);

  daliDeletePipeline(&handle);
}

daliPipelineHandle CreateCheckpointingTestPipe() {
  dali::Pipeline pipe(1, 1, 0, -1, true, 1);
  pipe.AddOperator(
    OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("dtype", DALI_FLOAT64)
      .AddOutput("OUTPUT", "cpu"));
  pipe.SetOutputDescs({{"OUTPUT", "cpu"}});
  pipe.EnableCheckpointing();
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());
  return handle;
}

TEST(CApiTest, CheckpointingTest) {
  // Create the first pipeline
  auto handle1 = CreateCheckpointingTestPipe();

  // Run it for a few iterations
  for (int i = 0; i < 3; i++) {
      daliRun(&handle1);
      daliOutput(&handle1);
  }

  // Save the checkpoint
  daliExternalContextCheckpoint mock_external_context{};

  const std::string pipeline_data = "Hello world";
  mock_external_context.pipeline_data.data = static_cast<char *>(daliAlloc(pipeline_data.size()));
  memcpy(mock_external_context.pipeline_data.data, pipeline_data.c_str(), pipeline_data.size());
  mock_external_context.pipeline_data.size = pipeline_data.size();

  const std::string iterator_data = "Iterator data!";
  mock_external_context.iterator_data.data = static_cast<char *>(daliAlloc(iterator_data.size()));
  memcpy(mock_external_context.iterator_data.data, iterator_data.c_str(), iterator_data.size());
  mock_external_context.iterator_data.size = iterator_data.size();

  char *cpt;
  size_t n;
  daliGetSerializedCheckpoint(&handle1, &mock_external_context, &cpt, &n);

  // Check pipeline's result
  double result1;
  daliRun(&handle1);
  daliOutput(&handle1);
  daliOutputCopy(&handle1, &result1, 0, device_type_t::CPU, 0, DALI_ext_default);

  // Create a new pipeline from the saved checkpoint
  auto handle2 = CreateCheckpointingTestPipe();
  daliExternalContextCheckpoint restored_external_context{};
  daliRestoreFromSerializedCheckpoint(&handle2, cpt, n, &restored_external_context);
  free(cpt);

  EXPECT_EQ(restored_external_context.pipeline_data.size,
            mock_external_context.pipeline_data.size);
  EXPECT_EQ(strncmp(
    restored_external_context.pipeline_data.data,
    mock_external_context.pipeline_data.data,
    mock_external_context.pipeline_data.size), 0);
  EXPECT_EQ(strncmp(
    restored_external_context.iterator_data.data,
    mock_external_context.iterator_data.data,
    mock_external_context.iterator_data.size), 0);
  daliDestroyExternalContextCheckpoint(&mock_external_context);
  daliDestroyExternalContextCheckpoint(&restored_external_context);

  // Check the result of the new pipeline
  double result2;
  daliRun(&handle2);
  daliOutput(&handle2);
  daliOutputCopy(&handle2, &result2, 0, device_type_t::CPU, 0, DALI_ext_default);
  EXPECT_EQ(result1, result2);

  // Delete the pipelines
  daliDeletePipeline(&handle1);
  daliDeletePipeline(&handle2);
}

}  // namespace dali
