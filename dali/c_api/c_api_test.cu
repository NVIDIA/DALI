// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/pipeline.h"


namespace dali {


namespace {

__global__ void hog(float *f, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    for (int k = 0; k < 100; k++)
      f[i] = sqrt(f[i]);
  }
}

struct GPUHog {
  void init() {
    if (!mem)
      mem = mm::alloc_raw_unique<float, mm::memory_kind::device>(size);
    CUDA_CALL(cudaMemset(mem.get(), 1, size * sizeof(float)));
  }

  void run(cudaStream_t stream, int count = 1) {
    for (int i = 0; i < count; i++) {
      hog<<<div_ceil(size, 512), 512, 0, stream>>>(mem.get(), size);
      CUDA_CALL(cudaGetLastError());
    }
  }

  mm::uptr<float> mem;
  size_t size = 16<<20;
};

}  // namespace

enum class Method {
  Contiguous, Samples
};

void TestCopyOutput(Method method) {
  constexpr int batch_size = 2;
  dali::Pipeline pipe(batch_size, 4, 0);
  std::string es_cpu_name = "pipe_in";
  pipe.AddExternalInput(es_cpu_name, "cpu");

  std::string cont_name = "pipe_out";
  pipe.AddOperator(OpSpec("MakeContiguous")
                          .AddArg("device", "mixed")
                          .AddArg("name", cont_name)
                          .AddInput(es_cpu_name, StorageDevice::CPU)
                          .AddOutput(cont_name, StorageDevice::GPU), cont_name);
  std::vector<std::pair<std::string, std::string>> outputs = {{"pipe_out", "gpu"}};

  GPUHog hog;
  hog.init();

  std::vector<std::vector<int>> in_data;
  int sample_size = 400000;
  TensorListShape<> shape = uniform_list_shape(8, {sample_size});
  in_data.resize(3);
  for (int i = 0; i < 3; i++) {
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
  hog.run(stream, 50);


  // This loop is tuned so that if the output buffer is recycled before the asynchronous copy
  // finishes, the buffer is clobbered and an error is detected.
  // In order to trigger a failure, remove the `wait_order.wait` at the end of
  // daliOutputCopy / daliOutputCopySamples
  for (int attempt = 0; attempt < 20; attempt++) {
    daliPipelineHandle handle;

    // create a new instance of the pipeline
    daliDeserializeDefault(&handle, ser.c_str(), ser.size());

    // feed the data & run - this is the iteration from which we want to see the data
    daliSetExternalInput(&handle, "pipe_in", CPU, in_data[0].data(), ::DALI_INT32,
                        shape.shapes.data(), 1, nullptr, 0);
    daliRun(&handle);

    // schedule an extra iteration
    daliSetExternalInput(&handle, "pipe_in", CPU, in_data[1].data(), ::DALI_INT32,
                        shape.shapes.data(), 1, nullptr, 0);
    daliRun(&handle);
    // ...and prepare for one more
    daliSetExternalInput(&handle, "pipe_in", CPU, in_data[2].data(), ::DALI_INT32,
                        shape.shapes.data(), 1, nullptr, 0);

    // get the outputs - this contains some synchronization, so it comes before dispatching the hog
    daliShareOutput(&handle);
    // hog the GPU on the stream on which we'll copy the output
    hog.run(stream, 10);

    // copy the output on our stream, without waiting on host
    if (method == Method::Contiguous) {
      daliOutputCopy(&handle, out_gpu.data(), 0, GPU, stream, 0);
    } else if (method == Method::Samples) {
      void *dsts[batch_size];
      for (int i = 0; i < batch_size; i++)
        dsts[i] = out_gpu.data() + i*sample_size;
      daliOutputCopySamples(&handle, dsts, 0, GPU, stream, 0);
    }

    // release the buffer - it can be immediately recycled (in appropriate stream order)
    daliOutputRelease(&handle);
    daliRun(&handle);

    // now, copy the buffer to host...
    CUDA_CALL(cudaMemcpyAsync(out_cpu.data(), out_gpu.data(), batch_size*sample_size*sizeof(int),
                              cudaMemcpyDeviceToHost, stream));

    // ...and verify the contents
    for (int i = 0; i < batch_size * sample_size; i++) {
      // check for race condition...
      ASSERT_TRUE(out_cpu[i] != in_data[1][i] && out_cpu[i] != in_data[2][i])
        << "Invalid value: " << out_cpu[i] << " - synchronization failed"
           " - data clobbered by next iteration; detected at index " << i;

      // ...and for any other corruption
      ASSERT_EQ(out_cpu[i], in_data[0][i]) << " data corrupted at index " << i;
    }

    daliDeletePipeline(&handle);
  }
}

TEST(CApiTest, daliOutputCopy_Async) {
  TestCopyOutput(Method::Contiguous);
}

TEST(CApiTest, daliOutputCopySamples_Async) {
  TestCopyOutput(Method::Samples);
}

}  // namespace dali
