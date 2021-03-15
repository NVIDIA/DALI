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
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu.cuh"

#define DEBUG_LOGS 0
#define PERF_RUN 0

namespace dali {
namespace kernels {
namespace test {

using KernelPtr = void(*)(const SampleDesc *, const kernels::BlockDesc<2> *);

template <typename GTestParams>
class JpegDistortionTestGPU : public ::testing::Test {
  using T = typename GTestParams::T;
  static constexpr bool vert_subsample = GTestParams::vert_subsample;
  static constexpr bool horz_subsample = GTestParams::horz_subsample;

 public:
  JpegDistortionTestGPU() {
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

    int64_t nbytes = input_host_.size() * sizeof(uint8_t);
    CUDA_CALL(cudaMalloc(&input_device_, nbytes));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), nbytes, cudaMemcpyDefault));

    // for simplicity we are allocating 3 planes of the same size, even though
    // we only need half or a fourth of the memory, due to subsampling.
    CUDA_CALL(cudaMalloc(&output_, nbytes));
    CUDA_CALL(cudaMemset(output_, 0, nbytes));
    cudaDeviceSynchronize();
  }

  void TearDown() final {
    CUDA_CALL(cudaFree(input_device_));
    CUDA_CALL(cudaFree(output_));
  }

  void TestKernel(KernelPtr kernel_fn_ptr) {
    CUDAStream stream = CUDAStream::Create(true);

    TensorListShape<2> chroma_shape(in_shapes_.size(), 2);
    std::vector<SampleDesc> samples_cpu;
    samples_cpu.resize(in_shapes_.size());
    uint8_t *out_ptr = output_;
    uint8_t *in_ptr = input_device_;

    for (size_t i = 0; i < in_shapes_.size(); i++) {
      auto &sample_desc = samples_cpu[i];
      const auto& in_sh = in_shapes_[i];
      auto chroma_sh = chroma_shape.tensor_shape_span(i);
      auto shape_vol = volume(in_sh);
      auto width = in_sh[1];
      auto height = in_sh[0];
      auto chroma_width = width >> horz_subsample;
      auto chroma_height = height >> vert_subsample;
      // used to generate logical blocks (one thread per chroma pixel)
      chroma_sh[0] = chroma_height;
      chroma_sh[1] = chroma_width;
      chroma_sh[2] = 3;

      sample_desc.in = in_ptr;
      sample_desc.out = out_ptr;
      sample_desc.size.x = width;
      sample_desc.size.y = height;
      sample_desc.strides.x = 3;
      sample_desc.strides.y = width * 3;
      out_ptr += shape_vol;
      in_ptr += shape_vol;
    }

    block_setup_.SetBlockDim(dim3(32, 16, 1));
    int xblock = 64*(2-horz_subsample);
    int yblock = 128;
    block_setup_.SetDefaultBlockSize({xblock, yblock});
    block_setup_.SetupBlocks(chroma_shape, true);
    auto blocks_cpu = block_setup_.Blocks();

    SampleDesc *samples_gpu;
    CUDA_CALL(cudaMalloc(&samples_gpu, sizeof(SampleDesc) * samples_cpu.size()));
    CUDA_CALL(cudaMemcpy(samples_gpu, samples_cpu.data(),
                         sizeof(SampleDesc) * samples_cpu.size(),
                         cudaMemcpyDefault));
    BlockDesc *blocks_gpu;
    CUDA_CALL(cudaMalloc(&blocks_gpu, sizeof(BlockDesc) * blocks_cpu.size()));
    CUDA_CALL(cudaMemcpy(blocks_gpu, blocks_cpu.data(), sizeof(BlockDesc) * blocks_cpu.size(),
                         cudaMemcpyDefault));

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
#if DEBUG_LOGS
    std::cout << "\ngrid dim " << grid_dim.x << " " << grid_dim.y << " " << grid_dim.z
    << "\nblock_dim " << block_dim.x << " " << block_dim.y << " " << block_dim.z << "\n";
    for (int i = 0; i < blocks_cpu.size(); i++) {
      auto &blk = blocks_cpu[i];
      std::cout << "block " << i << " sample idx " << blk.sample_idx
                << " from " << blk.start << " to " << blk.end << "\n";
    }
#endif
    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);
#if PERF_RUN  // warm-up
    kernel_fn_ptr<<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
#endif
    CUDA_CALL(cudaEventRecord(start, stream));

    kernel_fn_ptr<<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaEventRecord(end, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

#if PERF_RUN
    float time = 0;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6;  // to nanoseconds
    int64_t size = 2 * input_host_.size() * sizeof(uint8_t);
    std::cerr << "Throughput: " << size / time << " GB/s\n";
#endif

    std::vector<uint8_t> output_host_(input_host_.size());
    CUDA_CALL(cudaMemcpy(output_host_.data(), output_, input_host_.size() * sizeof(uint8_t),
                         cudaMemcpyDefault));

    int64_t total_vol = input_host_.size();
    TensorShape<1> flat_sh{total_vol};
    TensorView<StorageCPU, uint8_t> ref_tv(ref_output_.data(), flat_sh);
    TensorView<StorageCPU, uint8_t> out_tv(output_host_.data(), flat_sh);

#if DEBUG_LOGS
    auto print_out = [&](std::vector<T> &outvec) {
      int off = 0;
      for (size_t i = 0; i < in_shapes_.size(); i++) {
        const auto& in_sh = in_shapes_[i];
        auto width = in_sh[1];
        auto height = in_sh[0];
        std::cout << "\nSample " << i << "\n";
        for (size_t k = 0; k < width * height; k++) {
          if (k > 0 && k % width == 0) std::cout << "\n";
          std::cout << " " << static_cast<int>(outvec[off++])
                    << "-" << static_cast<int>(outvec[off++])
                    << "-" << static_cast<int>(outvec[off++]);
        }
        std::cout << "\n";
      }
    };
    std::cout << "\nInput:";
    print_out(input_host_);
    std::cout << "\nReference:";
    print_out(ref_output_);
    std::cout << "\nOutput:";
    print_out(output_host_);
#endif

    // In the kernel we average the RGB values, then converto to YCbCr
    // while here we are first converting and then averaging
    Check(out_tv, ref_tv, EqualEps(5.99));

    CUDA_CALL(cudaFree(blocks_gpu));
    CUDA_CALL(cudaFree(samples_gpu));
  }

  void CalcOut_ChromaSubsampleDistortion() {
    // simplest implementation for test purposes. First convert to YCbCr, then subsample

    ref_output_.clear();
    ref_output_.reserve(input_host_.size());  // worst case size

    std::vector<T> tmp_y;
    std::vector<T> tmp_cb;
    std::vector<T> tmp_cr;
    int64_t in_offset = 0;
    for (size_t sample = 0; sample < in_shapes_.size(); sample++) {
      auto sh = in_shapes_[sample];
      int64_t npixels = sh[0] * sh[1];
      tmp_y.resize(npixels);
      tmp_cb.resize(npixels);
      tmp_cr.resize(npixels);
      for (int64_t i = 0; i < npixels; i++) {
        uint8_t r = input_host_[in_offset++];
        uint8_t g = input_host_[in_offset++];
        uint8_t b = input_host_[in_offset++];
        tmp_y[i]  =  0.29900000f * r + 0.58700000f * g + 0.11400000f * b;
        tmp_cb[i] = -0.16873589f * r - 0.33126411f * g + 0.50000000f * b + 128.0f;
        tmp_cr[i] =  0.50000000f * r - 0.41868759f * g - 0.08131241f * b + 128.0f;
      }

#if DEBUG_LOGS
      std::cout << "\nYCbCr original ref:\n";
      for (int64_t i = 0; i < npixels; i++) {
        if (i > 0 && i % sh[1] == 0) std::cout << "\n";
        std::cout << " " << static_cast<int>(tmp_y[i])
                  << "-" << static_cast<int>(tmp_cb[i])
                  << "-" << static_cast<int>(tmp_cr[i]);
      }
#endif

      auto subsample_f = [&](std::vector<T> &component) {
        for (int64_t y = 0; y < sh[0]; y+=(1 << vert_subsample)) {
          for (int64_t x = 0; x < sh[1]; x+=(1 << horz_subsample)) {
            auto in_offset_1 = y * sh[1] + x;
            auto in_offset_2 = y * sh[1] + x + 1;
            auto in_offset_3 = (y + 1) * sh[1] + x;
            auto in_offset_4 = (y + 1) * sh[1] + x + 1;
            if (horz_subsample && vert_subsample) {
              T avg = ConvertSat<T>(0.25f * (component[in_offset_1] + component[in_offset_2] +
                                             component[in_offset_3] + component[in_offset_4]));
              component[in_offset_1] = avg;
              component[in_offset_2] = avg;
              component[in_offset_3] = avg;
              component[in_offset_4] = avg;
            } else if (horz_subsample) {
              T avg = ConvertSat<T>(0.5f * (component[in_offset_1] + component[in_offset_2]));
              component[in_offset_1] = avg;
              component[in_offset_2] = avg;
            } else if (vert_subsample) {
              T avg = ConvertSat<T>(0.5f * (component[in_offset_1] + component[in_offset_3]));
              component[in_offset_1] = avg;
              component[in_offset_3] = avg;
            }
          }
        }
      };
      if (horz_subsample || vert_subsample) {
        subsample_f(tmp_cb);
        subsample_f(tmp_cr);
      }

#if DEBUG_LOGS
      std::cout << "\nYCbCr subsampled ref:\n";
      for (int64_t i = 0; i < npixels; i++) {
        if (i > 0 && i % sh[1] == 0) std::cout << "\n";
        std::cout << " " << static_cast<int>(tmp_y[i])
                  << "-" << static_cast<int>(tmp_cb[i])
                  << "-" << static_cast<int>(tmp_cr[i]);
      }
      std::cout << "\n";
#endif

      for (int64_t i = 0; i < npixels; i++) {
        float y = static_cast<float>(tmp_y[i]);
        float cb = static_cast<float>(tmp_cb[i]) - 128.0f;
        float cr = static_cast<float>(tmp_cr[i]) - 128.0f;
        auto r = ConvertSat<T>(y + 1.402f * cr);
        auto g = ConvertSat<T>(y - 0.34413629f * cb - 0.71413629f * cr);
        auto b = ConvertSat<T>(y + 1.772f * cb);
        ref_output_.push_back(r);
        ref_output_.push_back(g);
        ref_output_.push_back(b);
      }
    }
  }

  void CalcOut_JpegCompressionDistortion() {
    // TODO(janton): Implement JPEG encode/decode to produce the reference
    CalcOut_ChromaSubsampleDistortion();
  }

  void TestJpegCompressionDistortion() {
    CalcOut_JpegCompressionDistortion();
    TestKernel(JpegCompressionDistortion<horz_subsample, vert_subsample>);
  }

  void TestChromaSubsampleDistortion() {
    CalcOut_ChromaSubsampleDistortion();
    TestKernel(ChromaSubsampleDistortion<horz_subsample, vert_subsample>);
  }

  size_t batch_volume(const std::vector<TensorShape<3>> &shapes) {
    int ret = 0;
    for (auto sh : shapes) {
      ret += volume(sh);
    }
    return ret;
  }

  uint8_t *input_device_;
  uint8_t *output_;
  std::vector<uint8_t> input_host_;
  std::vector<uint8_t> ref_output_;
  CUDAStream stream_;
  std::vector<TensorShape<3>> in_shapes_;

  using BlkSetup = BlockSetup<2, -1>;
  BlkSetup block_setup_;
  using BlockDesc = BlkSetup::BlockDesc;
};

template <typename OutType, bool v, bool h>
struct jpeg_distortion_params_t {
  using T = OutType;
  static constexpr bool vert_subsample = v;
  static constexpr bool horz_subsample = h;
};

using TestParams = ::testing::Types<
  jpeg_distortion_params_t<uint8_t, true, true>,
  jpeg_distortion_params_t<uint8_t, false, true>,
  jpeg_distortion_params_t<uint8_t, true, false>,
  jpeg_distortion_params_t<uint8_t, false, false>
>;

TYPED_TEST_SUITE_P(JpegDistortionTestGPU);

TYPED_TEST_P(JpegDistortionTestGPU, ChromaSubsampleDistortion) {
  this->TestChromaSubsampleDistortion();
}

TYPED_TEST_P(JpegDistortionTestGPU, JpegCompressionDistortion) {
  this->TestJpegCompressionDistortion();
}

REGISTER_TYPED_TEST_SUITE_P(JpegDistortionTestGPU, ChromaSubsampleDistortion,
                                                   JpegCompressionDistortion);
INSTANTIATE_TYPED_TEST_SUITE_P(JpegDistortionSuite, JpegDistortionTestGPU, TestParams);

}  // namespace test
}  // namespace kernels
}  // namespace dali

