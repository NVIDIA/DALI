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

#include "reduce_axes_gpu_impl.cuh"
#include <random>
#include "dali/kernels/alloc.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

TEST(ReduceGPU, ReduceInnerSmall) {
  std::mt19937_64 rng(1234);
  std::uniform_real_distribution<float> dist(0, 1);
  std::uniform_int_distribution<int> inner_shape_dist(1, 63);
  std::uniform_int_distribution<int> outer_shape_dist(100, 10000);

  int N = 10;
  TensorListShape<2> tls;
  tls.resize(N);
  for (int i = 0; i < N; i++) {
    int outer = outer_shape_dist(rng);
    int inner = inner_shape_dist(rng);
    tls.set_tensor_shape(i, { outer, inner });
  }
  TestTensorList<float, 2> in;
  in.reshape(tls);
  UniformRandomFill(tls.cpu(), rng, 0, 1);

  auto gpu_in = tls.gpu();

  auto gpu_pointers = memory::alloc_unique<float*>(AllocType::GPU, N);
  auto gpu_shapes = memory::alloc_unique<int64_t>(AllocType::GPU, N);
  cudaMemcpy(gpu_pointers.get(), gpu_in.data.data(), N * sizeof(float*));
  cudaMemcpy(gpu_shapes.get(), tls.shapes.data(), N * sizeof(int64_t));


}

}  // namespace kernels
}  // namespace dali
