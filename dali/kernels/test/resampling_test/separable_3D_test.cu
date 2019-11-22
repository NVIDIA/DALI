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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include "dali/core/math_util.h"
#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/test/resampling_test/resampling_test_params.h"

using std::cout;
using std::endl;

namespace dali {
namespace kernels {
namespace resample_test {

struct Bubble {
  vec3 centre;
  vec3 color;
  float frequency;
  float decay;
};

template <typename T>
__global__ void DrawBubblesKernel(T *data, ivec3 size, const Bubble *bubbles, int nbubbles) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  T *pixel = &data[3 * (x + size.x * (y + size.y * z))];

  vec3 pos(x + 0.5f, y + 0.5f, z + 0.5f);

  vec3 color(0, 0, 0);
  for (int i = 0; i < nbubbles; i++) {
    float dsq = (bubbles[i].centre - pos).length_square();
    float d = dsq*rsqrt(dsq);
    float magnitude = expf(bubbles[i].decay * dsq);
    float phase = bubbles[i].frequency * d;
    color += bubbles[i].color * (1 + cos(phase)) * magnitude * 0.5f;
  }
  pixel[0] = ConvertSatNorm<T>(color[0]);
  pixel[1] = ConvertSatNorm<T>(color[1]);
  pixel[2] = ConvertSatNorm<T>(color[2]);
}

template <typename T>
struct GPUBuf {
  void alloc(size_t n) {
    if (n <= capacity)
      return;
    if (2 * capacity > n)
      n = 2 * capacity;
    buf.reset();
    buf = memory::alloc_unique<T>(AllocType::GPU, n);
    capacity = n;
  }

  void copy_in_h2d(const T *src, size_t count) {
    alloc(count);
    cudaMemcpy(get(), src, count*sizeof(T), cudaMemcpyHostToDevice);
  }

  void copy_out_d2h(T *dst, size_t count) {
    assert(count <= capacity);
    cudaMemcpy(dst, get(), count*sizeof(T), cudaMemcpyDeviceToHost);
  }

  T *get() const { return buf.get(); }

  memory::KernelUniquePtr<T> buf;
  size_t capacity = 0;
};

template <typename T>
struct TestDataGenerator {
  GPUBuf<T> gpu_tensor;
  GPUBuf<Bubble> gpu_bubbles;

  void DrawBubbles(const TensorView<StorageCPU, T, 4> &tensor, span<const Bubble> bubbles) {
    gpu_tensor.alloc(volume(tensor.shape()));
    DrawBubbles(make_tensor_gpu(static_cast<T*>(gpu_tensor.get()), tensor.shape),
                bubbles);
  }

  void DrawBubbles(const TensorView<StorageGPU, T, 4> &tensor, span<const Bubble> bubbles) {
    gpu_bubbles.copy_in_h2d(bubbles.data(), bubbles.size());
    ivec3 size(tensor.shape[2], tensor.shape[1], tensor.shape[0]);
    assert(tensor.shape[3] == 3);
    dim3 block(32, 32, 1);
    dim3 grid(div_ceil(size.x, 32), div_ceil(size.y, 32), size.z);
    DrawBubblesKernel<<<grid, block>>>(tensor.data, size, gpu_bubbles.get(), bubbles.size());
  }

  template <typename Storage>
  void GenerateTestData(const TensorView<Storage, T, 4> &tensor, int num_bubbles = 5) {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<float> dist(0, 1);
    std::uniform_real_distribution<float> freq_dist(1/(40*M_PI), 1/(5*M_PI));
    std::uniform_real_distribution<float> sigma_dist(10, 100);

    auto shape = tensor.shape;
    assert(shape[3] == 3);  // only RGB

    std::vector<Bubble> bubbles(num_bubbles);
    for (int i = 0; i < num_bubbles; i++) {
      bubbles[i].centre = { shape[2] * dist(rng), shape[1] * dist(rng), shape[0] * dist(rng) };
      bubbles[i].color = { dist(rng), dist(rng), dist(rng) };
      bubbles[i].frequency = freq_dist(rng);
      bubbles[i].decay = -1/sqrt(2 * sigma_dist(rng));
    }
    DrawBubbles(tensor, bubbles);
  }
};

}  // namespace resample_test
}  // namespace kernels
}  // namespace dali
