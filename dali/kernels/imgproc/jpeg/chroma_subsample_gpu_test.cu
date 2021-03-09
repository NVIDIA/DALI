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
      sample_desc.in_strides = {in_sh[1] * 3, 3};

      sample_desc.out_y_size = {in_sh[0], in_sh[1]};
      sample_desc.out_y_strides = {in_sh[1], 1};

      sample_desc.out_chroma_size = {chroma_sh[1], chroma_sh[2]};
      sample_desc.out_chroma_strides = {chroma_sh[2], 1};

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
    int64_t out_total_len = out_ptr - output_;

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

    std::cout << "params " << grid_dim.x << " " << grid_dim.y << " " << grid_dim.z << "\n" 
              << " block_dim " <<  block_dim.x << " " << block_dim.y << " " << block_dim.z << "\n";
    RGBToYCbCrChromaSubsample<horz_subsample, vert_subsample, T>
      <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
    CUDA_CALL(cudaGetLastError());
    cudaDeviceSynchronize();

    std::vector<T> output_host_(out_total_len);
    CUDA_CALL(cudaMemcpy(output_host_.data(), output_, sizeof(T) * out_total_len,
                         cudaMemcpyDefault));
    cudaDeviceSynchronize();

    std::cout << "ref size is " << ref_output_.size() << "\nData:";
    for (int i = 0; i < ref_output_.size(); i++) std::cout << " " << (int) ref_output_[i]; std::cout << "\n"; 
    std::cout << "out size is " << output_host_.size() << "\nData:";
    for (int i = 0; i < output_host_.size(); i++) std::cout << " " << (int) output_host_[i]; std::cout << "\n"; 

//    TensorShape<1> flat_sh{out_total_len};
//    TensorView<StorageCPU, T> ref_tv(output_host_.data(), flat_sh);
//    TensorView<StorageCPU, T> out_tv(output_, flat_sh);
//    Check(out_tv, ref_tv, EqualUlp());

    CUDA_CALL(cudaFree(blocks_gpu));
    CUDA_CALL(cudaFree(samples_gpu));
  }

  uint8_t *input_device_;
  T *output_;
  std::vector<uint8_t> input_host_;
  std::vector<T> ref_output_;
  std::vector<TensorShape<3>> out_shapes_;
  std::vector<TensorShape<3>> in_shapes_ = {{2, 4, 3}, {2, 2, 3}};

  BlockSetup<2, 0> block_setup_;
  using BlockDesc = BlockSetup<2, 0>::BlockDesc;

  void calc_output() {
    // simplest implementation for test purposes. First convert to YCbCr, then subsample
    std::vector<uint8_t> out_y, tmp_cb, tmp_cr;
    int64_t comp_len = input_host_.size() / 3;
    out_y.reserve(comp_len);
    tmp_cb.reserve(comp_len);
    tmp_cr.reserve(comp_len);

    ref_output_.clear();
    ref_output_.reserve(input_host_.size());  // worst case size
    for (int sample = 0; sample < in_shapes_.size(); sample++) {
      out_y.clear();
      tmp_cb.clear();
      tmp_cr.clear();

      auto sh = in_shapes_[sample];
      int64_t chroma_height = sh[0] >> horz_subsample;
      int64_t chroma_width  = sh[1] >> vert_subsample;
      int64_t npixels = sh[0] * sh[1];
      for (int64_t i = 0; i < npixels; i++) {
        vec<3, uint8_t> rgb{input_host_[i*3], input_host_[i*3+1], input_host_[i*3+2]};
        out_y[i]  =  0.29900000f * rgb.x + 0.58700000f * rgb.y + 0.11400000f * rgb.z;
        tmp_cb[i] = -0.16873589f * rgb.x - 0.33126411f * rgb.y + 0.50000000f * rgb.z + 128.0f;
        tmp_cr[i] =  0.50000000f * rgb.x - 0.41868759f * rgb.y - 0.08131241f * rgb.z + 128.0f;

        ref_output_.push_back(ConvertSat<T>(out_y[i]));
      }

      auto subsample_f = [&](std::vector<T> &out, const std::vector<uint8_t> &component) {
        for (int64_t y = 0; y < chroma_height; y++) {
          for (int64_t x = 0; x < chroma_width; x++) {
            auto in_y = y << vert_subsample;
            auto in_x = x << horz_subsample;
            auto in_offset_1 = in_y * sh[1] + in_x;
            auto in_offset_2 = in_y * sh[1] + in_x + 1;
            auto in_offset_3 = (in_y + 1) * sh[1] + in_x;
            auto in_offset_4 = (in_y + 1) * sh[1] + in_x + 1;
            int c = component[in_offset_1];
            if (horz_subsample && vert_subsample) {
              c += component[in_offset_2];
              c += component[in_offset_3];
              c += component[in_offset_4];
              out.push_back(ConvertSat<T>(c / 4));
            } else if (horz_subsample) {
              c += component[in_offset_2];
              out.push_back(ConvertSat<T>(c / 2));
            } else if (vert_subsample) {
              c += component[in_offset_3];
              out.push_back(ConvertSat<T>(c / 2));
            } else {
              out.push_back(ConvertSat<T>(c));
            }
          }
        }
      };
      subsample_f(ref_output_, tmp_cb);
      subsample_f(ref_output_, tmp_cr);
    }
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

