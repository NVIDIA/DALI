// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>

#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/jpeg/chroma_subsample_gpu.cuh"

namespace dali {
namespace kernels {
namespace test {

template <typename GTestParams>
class ChromaSubsampleGPUTest : public ::testing::Test {
  using T = typename GTestParams::T;
  static constexpr bool vert_subsample = GTestParams::vert_subsample;
  static constexpr bool horz_subsample = GTestParams::horz_subsample;

 public:
  ChromaSubsampleGPUTest() {
    input_host_.resize(batch_volume(in_shapes_));
  }

  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_host_, rng, 0., 255.);
    calc_output();
    CUDA_CALL(cudaMalloc(&input_device_, sizeof(uint8_t) * batch_volume(in_shapes_)));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), input_host_.size() * sizeof(uint8_t),
                         cudaMemcpyDefault));

    // for simplicity we are allocating 3 planes of the same size, even though
    // we only need half or a fourth of the memory, due to subsampling.
    CUDA_CALL(cudaMalloc(&output_, batch_volume(in_shapes_) * sizeof(T)));
    cudaDeviceSynchronize();
  }

  void TearDown() final {
    CUDA_CALL(cudaFree(input_device_));
    CUDA_CALL(cudaFree(output_));
  }

  void RunTest() {
    TensorListShape<3> chroma_shape(in_shapes_.size(), 3);
    std::vector<SampleDesc<T>> samples_cpu;
    samples_cpu.resize(in_shapes_.size());
    auto *out_ptr = output_;
    auto *in_ptr = input_device_;

    for (size_t i = 0; i < in_shapes_.size(); i++) {
      auto &sample_desc = samples_cpu[i];
      const auto& in_sh = in_shapes_[i];

      auto chroma_sh = chroma_shape.tensor_shape_span(i);
      chroma_sh[0] = 3;
      chroma_sh[1] = in_sh[0] >> vert_subsample;
      chroma_sh[2] = in_sh[1] >> horz_subsample;

      sample_desc.in_size = {in_sh[0], in_sh[1]};
      sample_desc.in_strides = {sample_desc.in_size[1] * 3, 3};

      sample_desc.out_y_size = {in_sh[0], in_sh[1]};
      sample_desc.out_y_strides = {sample_desc.out_y_size[1], 1};

      sample_desc.out_chroma_size = {chroma_sh[1], chroma_sh[2]};
      sample_desc.out_chroma_strides = {sample_desc.out_chroma_size[1], 1};

      auto luma_vol = volume(sample_desc.out_y_size);
      auto chroma_vol = volume(sample_desc.out_chroma_size);

      sample_desc.in = in_ptr;
      sample_desc.out_y = out_ptr;
      out_ptr += luma_vol;
      sample_desc.out_cb = out_ptr;
      out_ptr += chroma_vol;
      sample_desc.out_cr = out_ptr;
      out_ptr += chroma_vol;
      in_ptr += volume(in_sh);
    }

    block_setup_.SetupBlocks(chroma_shape, true);
    auto blocks_cpu = block_setup_.Blocks();

    SampleDesc<T> *samples_gpu;
    CUDA_CALL(cudaMalloc(&samples_gpu, sizeof(SampleDesc<T>) * samples_cpu.size()));
    CUDA_CALL(cudaMemcpy(samples_gpu, samples_cpu.data(),
                         sizeof(SampleDesc<T>) * samples_cpu.size(),
                         cudaMemcpyDefault));
    BlockDesc *blocks_gpu;
    CUDA_CALL(cudaMalloc(&blocks_gpu, sizeof(BlockDesc) * blocks_cpu.size()));
    CUDA_CALL(cudaMemcpy(blocks_gpu, blocks_cpu.data(), sizeof(BlockDesc) * blocks_cpu.size(),
                         cudaMemcpyDefault));

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    cudaStream_t stream = 0;
    RGBToYCbCrChromaSubsample<horz_subsample, vert_subsample, T>
      <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
    CUDA_CALL(cudaGetLastError());
    cudaDeviceSynchronize();

    // TODO(janton): Check data correctness

    CUDA_CALL(cudaFree(blocks_gpu));
    CUDA_CALL(cudaFree(samples_gpu));
  }

  uint8_t *input_device_;
  T *output_;
  std::vector<uint8_t> input_host_;
  std::vector<float> ref_output_;
  std::vector<TensorShape<3>> out_shapes_;
  std::vector<TensorShape<3>> in_shapes_ = {{40, 50, 3}, {10, 20, 3}};

  BlockSetup<2, 0> block_setup_;
  using BlockDesc = BlockSetup<2, 0>::BlockDesc;

  void calc_output() {
    // TODO(janton)
  }

  size_t batch_volume(const std::vector<TensorShape<3>> &shapes) {
    int ret = 0;
    for (auto sh : shapes) {
      ret += volume(sh);
    }
    return ret;
  }
};

template <typename OutType, bool v, bool h>
struct chroma_subsample_params_t {
  using T = OutType;
  static constexpr bool vert_subsample = v;
  static constexpr bool horz_subsample = h;
};

using ChromaSubsampleTestParams = ::testing::Types<chroma_subsample_params_t<uint8_t, true, true>>;

TYPED_TEST_SUITE_P(ChromaSubsampleGPUTest);

TYPED_TEST_P(ChromaSubsampleGPUTest, RunKernel) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(ChromaSubsampleGPUTest, RunKernel);
INSTANTIATE_TYPED_TEST_SUITE_P(ChromaSubsample, ChromaSubsampleGPUTest, ChromaSubsampleTestParams);

}  // namespace test
}  // namespace kernels
}  // namespace dali

