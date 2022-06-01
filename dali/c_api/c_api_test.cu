// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/dev_buffer.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"


namespace dali {


namespace {

__global__ void hog(float *f, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    f[i] = sqrt(f[i]);
}

struct GPUHog {
  ~GPUHog() {
    if (mem) {
      CUDA_DTOR_CALL(cudaFree(mem));
      mem = nullptr;
    }
  }

  void init() {
    if (!mem)
      CUDA_CALL(cudaMalloc(&mem, size * sizeof(float)));
  }

  void run(cudaStream_t stream, int count = 1) {
    for (int i = 0; i < count; i++) {
      CUDA_CALL(cudaMemsetAsync(mem, 1, size * sizeof(float), stream));
      hog<<<div_ceil(size, 512), 512>>>(mem, size);
    }
    CUDA_CALL(cudaGetLastError());
  }

  float *mem = nullptr;
  size_t size = 16<<20;
};

}  // namespace

TEST(CApiTest, daliOutputCopy_Async) {
  int batch_size = 8;
  dali::Pipeline pipe(batch_size, 4, 0);
  std::string es_cpu_name = "pipe_in";
  pipe.AddExternalInput(es_cpu_name, "cpu");

  std::string cont_name = "pipe_out";
  pipe.AddOperator(OpSpec("MakeContiguous")
                          .AddArg("device", "mixed")
                          .AddArg("name", cont_name)
                          .AddInput(es_cpu_name, "cpu")
                          .AddOutput(cont_name, "gpu"), cont_name);
  std::vector<std::pair<std::string, std::string>> outputs = {{"pipe_out", "gpu"}};

  GPUHog hog;
  std::vector<std::vector<int>> in_data;
  int sample_size = 1000000;
  TensorListShape<> shape = uniform_list_shape(8, {sample_size});
  in_data.resize(2);
  for (int i = 0; i < 2; i++) {
    in_data[i].resize(batch_size*sample_size);
    for (int j = 0; j < batch_size*sample_size; j++)
      in_data[i][j] = i + j;
  }
  pipe.SetOutputDescs(outputs);

  std::vector<int> out_cpu(batch_size*sample_size);
  DeviceBuffer<int> out_gpu;
  out_gpu.resize(batch_size*sample_size);
  CUDAStreamLease stream = CUDAStreamPool::instance().Get();

  CUDA_CALL(cudaMemsetAsync(out_gpu.data(), -1, out_gpu.size() * sizeof(int), stream));
  std::string ser = pipe.SerializeToProtobuf();
  daliPipelineHandle handle;
  hog.init();
  hog.run(stream, 10);
  daliDeserializeDefault(&handle, ser.c_str(), ser.size());

  daliSetExternalInput(&handle, "pipe_in", CPU, in_data[0].data(), ::DALI_INT32,
                       shape.shapes.data(), 1, nullptr, 0);
  daliRun(&handle);

  // schedule an extra iteration
  daliSetExternalInput(&handle, "pipe_in", CPU, in_data[1].data(), ::DALI_INT32,
                       shape.shapes.data(), 1, nullptr, 0);
  daliRun(&handle);
  hog.run(stream, 1000);
  daliOutput(&handle);
  daliOutputCopy(&handle, out_gpu.data(), 0, GPU, stream, 0);
  daliSetExternalInput(&handle, "pipe_in", CPU, in_data[1].data(), ::DALI_INT32,
                       shape.shapes.data(), 1, nullptr, 0);
  daliRun(&handle);

  CUDA_CALL(cudaMemcpyAsync(out_cpu.data(), out_gpu.data(), batch_size*sample_size*sizeof(int),
                            cudaMemcpyDeviceToHost, stream));

  for (int i = 0; i < batch_size * sample_size; i++)
    ASSERT_EQ(out_cpu[i], in_data[0][i]) << " at index " << i;

  daliDeletePipeline(&handle);
}

}  // namespace dali
