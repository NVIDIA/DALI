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
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/jpeg/chroma_subsample_gpu.cuh"

#define DEBUG_LOGS 0
#define PERF_RUN 0

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
    in_shapes_ = {{200, 400, 3}, {2000, 20, 3}, {2, 2, 3}};
#if DEBUG_LOGS
    in_shapes_ = {{2, 4, 3}, {2, 2, 3}};
#elif PERF_RUN
    in_shapes_.resize(64, TensorShape<3>{600, 800, 3});
#endif
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
    int64_t output_bytes = batch_volume(in_shapes_) * sizeof(T);
    CUDA_CALL(cudaMalloc(&output_, output_bytes));
    CUDA_CALL(cudaMemset(output_, 0, output_bytes));
    cudaDeviceSynchronize();
  }

  void TearDown() final {
    CUDA_CALL(cudaFree(input_device_));
    CUDA_CALL(cudaFree(output_));
  }

  void RunTest() {
    CUDAStream stream = CUDAStream::Create(true);

    TensorListShape<2> chroma_shape(in_shapes_.size(), 2);
    std::vector<SampleDesc<T>> samples_cpu;
    samples_cpu.resize(in_shapes_.size());
    auto *out_ptr = output_;
    auto *in_ptr = input_device_;

    for (size_t i = 0; i < in_shapes_.size(); i++) {
      auto &sample_desc = samples_cpu[i];
      const auto& in_sh = in_shapes_[i];
      auto width = in_sh[1];
      auto height = in_sh[0];
      auto chroma_width = width >> horz_subsample;
      auto chroma_height = height >> vert_subsample;
      auto chroma_sh = chroma_shape.tensor_shape_span(i);
      chroma_sh[0] = chroma_height;
      chroma_sh[1] = chroma_width;

      sample_desc.in_size.x = width;
      sample_desc.in_size.y = height;
      sample_desc.in_strides.x = 3;
      sample_desc.in_strides.y = width * 3;

      sample_desc.out_y_size.x = width;
      sample_desc.out_y_size.y = height;
      sample_desc.out_y_strides.x = 1;
      sample_desc.out_y_strides.y = width;

      sample_desc.out_chroma_size.x = chroma_width;
      sample_desc.out_chroma_size.y = chroma_height;
      sample_desc.out_chroma_strides.x = 1;
      sample_desc.out_chroma_strides.y = chroma_width;

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

    block_setup_.SetBlockDim(dim3(32, 8, 1));
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

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);
    CUDA_CALL(cudaEventRecord(start, stream));

    RGBToYCbCrChromaSubsample<horz_subsample, vert_subsample, T>
      <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaEventRecord(end, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

#if PERF_RUN
    float time = 0;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;  // to nanoseconds
    int64_t size = batch_volume(in_shapes_) * sizeof(uint8_t) + out_total_len * sizeof(T);
    std::cerr << "Throughput: " << size / time << " GB/s\n";
#endif

    std::vector<T> output_host_(out_total_len);
    CUDA_CALL(cudaMemcpy(output_host_.data(), output_, sizeof(T) * out_total_len,
                         cudaMemcpyDefault));

    TensorShape<1> flat_sh{out_total_len};
    TensorView<StorageCPU, T> ref_tv(ref_output_.data(), flat_sh);
    TensorView<StorageCPU, T> out_tv(output_host_.data(), flat_sh);

#if DEBUG_LOGS
    auto print_out = [&](std::vector<T> &outvec) {
      int off = 0;
      for (size_t i = 0; i < in_shapes_.size(); i++) {
        const auto& in_sh = in_shapes_[i];
        auto width = in_sh[1];
        auto height = in_sh[0];
        auto chroma_width = width >> horz_subsample;
        auto chroma_height = height >> vert_subsample;

        std::cout << "\n\nSample " << i << "Y:\n";
        for (size_t k = 0; k < width * height; k++) {
          if (k > 0 && k%width==0) std::cout << "\n";
          std::cout << " " << (int) outvec[off++];
        }
        std::cout << "\nSample " << i << "Cb:\n";
        for (size_t k = 0; k < chroma_width * chroma_height; k++) {
          if (k > 0 && k%chroma_width==0) std::cout << "\n";
          std::cout << " " << (int) outvec[off++];
        }
        std::cout << "\nSample " << i << "Cr:\n";
        for (size_t k = 0; k < chroma_width * chroma_height; k++) {
          if (k > 0 && k%chroma_width==0) std::cout << "\n";
          std::cout << " " << (int) outvec[off++];
        }
        std::cout << "\n";
      }
    };
    std::cout << "\n\nReference:\n\n";
    print_out(ref_output_);
    std::cout << "\n\nOutput:\n\n";
    print_out(output_host_);
#endif

    // In the kernel we average the RGB values, then converto to YCbCr
    // while here we are first converting and then averaging
    Check(out_tv, ref_tv, EqualEps(2.99));

    CUDA_CALL(cudaFree(blocks_gpu));
    CUDA_CALL(cudaFree(samples_gpu));
  }

  uint8_t *input_device_;
  T *output_;
  std::vector<uint8_t> input_host_;
  std::vector<T> ref_output_;
  std::vector<TensorShape<3>> out_shapes_;
  CUDAStream stream_;
  std::vector<TensorShape<3>> in_shapes_;

  using BlkSetup = BlockSetup<2, -1>;
  BlkSetup block_setup_;
  using BlockDesc = BlkSetup::BlockDesc;

  void calc_output() {
    // simplest implementation for test purposes. First convert to YCbCr, then subsample
    std::vector<uint8_t> out_y, tmp_cb, tmp_cr;
    int64_t comp_len = input_host_.size() / 3;
    out_y.reserve(comp_len);
    tmp_cb.reserve(comp_len);
    tmp_cr.reserve(comp_len);

    ref_output_.clear();
    ref_output_.reserve(input_host_.size());  // worst case size
    int64_t in_offset = 0;
    for (int sample = 0; sample < in_shapes_.size(); sample++) {
      out_y.clear();
      tmp_cb.clear();
      tmp_cr.clear();

      auto sh = in_shapes_[sample];
      int64_t chroma_height = sh[0] >> vert_subsample;
      int64_t chroma_width  = sh[1] >> horz_subsample;
      int64_t npixels = sh[0] * sh[1];

      for (int64_t i = 0; i < npixels; i++) {
        uint8_t r = input_host_[in_offset++];
        uint8_t g = input_host_[in_offset++];
        uint8_t b = input_host_[in_offset++];
        out_y[i]  =  0.29900000f * r + 0.58700000f * g + 0.11400000f * b;
        tmp_cb[i] = -0.16873589f * r - 0.33126411f * g + 0.50000000f * b + 128.0f;
        tmp_cr[i] =  0.50000000f * r - 0.41868759f * g - 0.08131241f * b + 128.0f;
        ref_output_.push_back(ConvertSat<T>(out_y[i]));
      }

#if DEBUG_LOGS
      std::cout << "\nY ref:\n";
      for (int64_t i = 0; i < npixels; i++) {
        if (i > 0 && i%sh[1]==0) std::cout << "\n";
        std::cout << " " << (int) out_y[i];
      }

      std::cout << "\nCb ref:\n";
      for (int64_t i = 0; i < npixels; i++) {
        if (i > 0 && i%sh[1]==0) std::cout << "\n";
        std::cout << " " << (int) tmp_cb[i];
      }

      std::cout << "\nCr ref:\n";
      for (int64_t i = 0; i < npixels; i++) {
        if (i > 0 && i%sh[1]==0) std::cout << "\n";
        std::cout << " " << (int) tmp_cr[i];
      }
#endif
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

using ChromaSubsampleTestParams = ::testing::Types<
  chroma_subsample_params_t<uint8_t, true, true>,
  chroma_subsample_params_t<int32_t, true, true>,
  chroma_subsample_params_t<float, true, true>,
  chroma_subsample_params_t<uint8_t, false, true>,
  chroma_subsample_params_t<uint8_t, true, false>,
  chroma_subsample_params_t<uint8_t, false, false>
>;

TYPED_TEST_SUITE_P(ChromaSubsampleGPUTest);

TYPED_TEST_P(ChromaSubsampleGPUTest, Test) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(ChromaSubsampleGPUTest, Test);
INSTANTIATE_TYPED_TEST_SUITE_P(ChromaSubsample, ChromaSubsampleGPUTest, ChromaSubsampleTestParams);


}  // namespace test
}  // namespace kernels
}  // namespace dali

