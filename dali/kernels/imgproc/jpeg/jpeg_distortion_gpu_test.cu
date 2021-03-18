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
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/dali_test_config.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu.cuh"

#define DEBUG_LOGS 0

namespace dali {
namespace kernels {
namespace test {

using KernelPtr = void(*)(const SampleDesc *, const kernels::BlockDesc<2> *);
using testing::dali_extra_path;

template <typename GTestParams>
class JpegDistortionTestGPU : public ::testing::Test {
  using T = typename GTestParams::T;
  static constexpr bool vert_subsample = GTestParams::vert_subsample;
  static constexpr bool horz_subsample = GTestParams::horz_subsample;
  static constexpr bool use_real_images = true;
  static constexpr bool perf_run = false;
  static constexpr bool dump_images = false;

 public:
  void SetUp() final {
    if (use_real_images) {
      std::vector<std::string> paths{
        dali_extra_path() + "/db/single/bmp/0/cat-111793_640_palette_8bit.bmp",
        dali_extra_path() + "/db/single/bmp/0/cat-1046544_640.bmp",
        dali_extra_path() + "/db/single/bmp/0/cat-3591348_640.bmp",
      };
      in_shapes_.resize(paths.size(), 3);
      std::vector<cv::Mat> images(paths.size());
      for (size_t i = 0; i < paths.size(); i++) {
        images[i] = cv::imread(paths[i]);
        cv::cvtColor(images[i], images[i], cv::COLOR_BGR2RGB);
        TensorShape<> sh{images[i].rows, images[i].cols, images[i].channels()};
        in_shapes_.set_tensor_shape(i, sh);
      }
      in_.reshape(in_shapes_);
      auto in_cpu = in_.cpu();
      for (size_t i = 0; i < paths.size(); i++) {
        std::memcpy(in_cpu[i].data, images[i].data,
                    in_shapes_.tensor_size(i) * sizeof(uint8_t));
      }
    } else {
      if (perf_run) {
        in_shapes_ = uniform_list_shape(64, TensorShape<3>{600, 800, 3});
      } else {
        // TODO(janton): Fix blocks not aligned to 16 length
        in_shapes_ = {{16, 16, 3}, {16, 16, 3}};
      }
      in_.reshape(in_shapes_);
      std::mt19937_64 rng;
      UniformRandomFill(in_.cpu(), rng, 0., 255.);
    }

    out_.reshape(in_shapes_);
  }

  void TearDown() final {
  }

  using KernelLaunchFn = std::function<
    void(dim3, dim3, cudaStream_t, const SampleDesc*, const kernels::BlockDesc<2>*)>;

  void TestKernel(KernelLaunchFn kernel_launch_fn) {
    CUDAStream stream = CUDAStream::Create(true);

    TensorListShape<2> chroma_shape(in_shapes_.size(), 2);
    std::vector<SampleDesc> samples_cpu;
    samples_cpu.resize(in_shapes_.size());

    auto in_view_gpu = in_.gpu(stream);
    auto out_view_gpu = out_.gpu(stream);
    out_.invalidate_cpu();
    CUDA_CALL(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < in_shapes_.size(); i++) {
      auto &sample_desc = samples_cpu[i];
      const auto& in_sh = in_shapes_[i];
      auto chroma_sh = chroma_shape.tensor_shape_span(i);
      auto shape_vol = volume(in_sh);
      auto width = in_sh[1];
      auto height = in_sh[0];
      auto chroma_width = div_ceil(width, 1 + horz_subsample);
      auto chroma_height = div_ceil(height, 1 + vert_subsample);
      // used to generate logical blocks (one thread per chroma pixel)
      chroma_sh[0] = chroma_height;
      chroma_sh[1] = chroma_width;
      chroma_sh[2] = 3;

      sample_desc.in = in_view_gpu[i].data;
      sample_desc.out = out_view_gpu[i].data;
      sample_desc.size.x = width;
      sample_desc.size.y = height;
      sample_desc.strides.x = 3;
      sample_desc.strides.y = width * 3;
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
    for (size_t i = 0; i < blocks_cpu.size(); i++) {
      auto &blk = blocks_cpu[i];
      std::cout << "block " << i << " sample idx " << blk.sample_idx
                << " from " << blk.start << " to " << blk.end << "\n";
    }
#endif
    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);

    if (perf_run)  // warm up
      kernel_launch_fn(grid_dim, block_dim, stream, samples_gpu, blocks_gpu);

    CUDA_CALL(cudaEventRecord(start, stream));

    kernel_launch_fn(grid_dim, block_dim, stream, samples_gpu, blocks_gpu);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaEventRecord(end, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    if (perf_run) {
      float time = 0;
      CUDA_CALL(cudaEventElapsedTime(&time, start, end));
      time *= 1e+6;  // to nanoseconds
      int64_t size = 2 * in_shapes_.num_elements() * sizeof(uint8_t);
      std::cerr << "Throughput: " << size / time << " GB/s\n";
    }

    auto out_view_cpu = out_.cpu(stream);
    CUDA_CALL(cudaStreamSynchronize(stream));
    auto in_view_cpu = in_.cpu();
    auto out_ref_cpu = out_ref_.cpu();

#if DEBUG_LOGS
    auto print_out = [&](TensorListView<StorageCPU, uint8_t, 3> view) {
      int off = 0;
      auto shape = view.shape;
      auto nsamples = shape.size();
      for (size_t i = 0; i < nsamples; i++) {
        auto sample_view = view[i];
        auto sample_sh = sample_view.shape;
        auto width = sample_sh[1];
        auto height = sample_sh[0];
        assert(sample_sh[2] == 3);
        std::cout << "\nSample " << i << "\n";
        for (size_t k = 0; k < width * height; k++) {
          if (k > 0 && k % width == 0) std::cout << "\n";
          std::cout << " " << static_cast<int>(sample_view.data[off++])
                    << "-" << static_cast<int>(sample_view.data[off++])
                    << "-" << static_cast<int>(sample_view.data[off++]);
        }
        std::cout << "\n";
      }
    };

    std::cout << "\nInput:";
    print_out(in_view_cpu);
    std::cout << "\nReference:";
    print_out(out_ref_cpu);
    std::cout << "\nOutput:";
    print_out(out_view_cpu);
#endif

    // In the kernel we average the RGB values, then converto to YCbCr
    // while here we are first converting and then averaging
    // Check(out_view_cpu, out_ref_cpu, EqualEps(40));

    for (size_t i = 0; i < in_shapes_.size(); i++) {
      auto sh = in_shapes_[i];
      cv::Mat in_mat(sh[0], sh[1], CV_8UC3, (void *) in_view_cpu[i].data);
      cv::Mat out_mat(sh[0], sh[1], CV_8UC3, (void *) out_view_cpu[i].data);
      cv::Mat out_ref(sh[0], sh[1], CV_8UC3, (void *) out_ref_cpu[i].data);
      cv::Mat diff;
      cv::absdiff(out_mat, out_ref, diff);

      // Sanity check. Checking that the average pixel difference is small
      // It is always best to compare the diff image visually to look for
      // artifacts.
      auto mean = cv::mean(diff);
      // different subsampling -> bigger error
      int avg_diff_max = horz_subsample && vert_subsample ? 2 : 6;
      for (int d = 0; d < 3; d++)
        ASSERT_LT(mean[d], avg_diff_max);

      if (dump_images) {
        std::stringstream ss1, ss2, ss3, ss4;
        ss1 << "jpeg_distortion_test_" << i << "_in.bmp";
        ss2 << "jpeg_distortion_test_" << i << "_out_ref.jpg";
        ss3 << "jpeg_distortion_test_" << i << "_out.bmp";
        ss4 << "jpeg_distortion_test_" << i << "_diff.bmp";
        cv::cvtColor(in_mat, in_mat, cv::COLOR_RGB2BGR);
        cv::imwrite(ss1.str(), in_mat);
        cv::cvtColor(in_mat, in_mat, cv::COLOR_BGR2RGB);

        cv::cvtColor(out_ref, out_ref, cv::COLOR_RGB2BGR);
        cv::imwrite(ss2.str(), out_ref);
        cv::cvtColor(out_ref, out_ref, cv::COLOR_BGR2RGB);

        cv::cvtColor(out_mat, out_mat, cv::COLOR_RGB2BGR);
        cv::imwrite(ss3.str(), out_mat);
        cv::cvtColor(out_mat, out_mat, cv::COLOR_BGR2RGB);

        cv::cvtColor(diff, diff, cv::COLOR_RGB2BGR);
        cv::imwrite(ss4.str(), diff);
        cv::cvtColor(diff, diff, cv::COLOR_BGR2RGB);
      }
    }

    CUDA_CALL(cudaFree(blocks_gpu));
    CUDA_CALL(cudaFree(samples_gpu));
  }

  void CalcOut_ChromaSubsampleDistortion() {
    // simplest implementation for test purposes. First convert to YCbCr, then subsample
    out_ref_.reshape(in_shapes_);
    auto out_ref_view = out_ref_.cpu();

    std::vector<T> tmp_y;
    std::vector<T> tmp_cb;
    std::vector<T> tmp_cr;
    auto in_view = in_.cpu();
    for (size_t sample = 0; sample < in_shapes_.size(); sample++) {
      auto sh = in_shapes_[sample];
      int64_t npixels = sh[0] * sh[1];
      tmp_y.resize(npixels);
      tmp_cb.resize(npixels);
      tmp_cr.resize(npixels);
      for (int64_t i = 0; i < npixels; i++) {
        uint8_t r = in_view[sample].data[3*i];
        uint8_t g = in_view[sample].data[3*i + 1];
        uint8_t b = in_view[sample].data[3*i + 2];
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

      auto* sample_data = out_ref_view[sample].data;
      for (int64_t i = 0; i < npixels; i++) {
        float y = static_cast<float>(tmp_y[i]);
        float cb = static_cast<float>(tmp_cb[i]) - 128.0f;
        float cr = static_cast<float>(tmp_cr[i]) - 128.0f;
        auto r = ConvertSat<T>(y + 1.402f * cr);
        auto g = ConvertSat<T>(y - 0.34413629f * cb - 0.71413629f * cr);
        auto b = ConvertSat<T>(y + 1.772f * cb);
        sample_data[3*i]   = r;
        sample_data[3*i+1] = g;
        sample_data[3*i+2] = b;
      }
    }
  }

  void CalcOut_JpegCompressionDistortion() {
    out_ref_.reshape(in_shapes_);
    auto out_ref_view = out_ref_.cpu();
    auto in_view_cpu = in_.cpu();
    for (size_t i = 0; i < in_shapes_.size(); i++) {
      auto sh = in_shapes_[i];
      cv::Mat in_mat(sh[0], sh[1], CV_8UC3, (void *) in_view_cpu[i].data);

      std::vector<uint8_t> encoded;

      cv::cvtColor(in_mat, in_mat, cv::COLOR_RGB2BGR);
      cv::imencode(".jpg", in_mat, encoded, {cv::IMWRITE_JPEG_QUALITY, ConvertSat<int>(quality_factor)});
      cv::cvtColor(in_mat, in_mat, cv::COLOR_BGR2RGB);

      cv::Mat encoded_mat(1, encoded.size(), CV_8UC1, encoded.data());
      auto out_ref = cv::imdecode(encoded_mat, cv::IMREAD_COLOR);
      cv::cvtColor(out_ref, out_ref, cv::COLOR_BGR2RGB);
      std::memcpy(out_ref_view[i].data, out_ref.data,
                  in_shapes_.tensor_size(i) * sizeof(uint8_t));
    }
  }

  void TestJpegCompressionDistortion() {
    quality_factor = 5;
    luma_table = GetLumaQuantizationTable(quality_factor);
    chroma_table = GetChromaQuantizationTable(quality_factor);
    CalcOut_JpegCompressionDistortion();
    TestKernel(
      [this](dim3 gridDim, dim3 blockDim, cudaStream_t stream,
             const SampleDesc* samples, const kernels::BlockDesc<2>* blocks) {
        JpegCompressionDistortion<horz_subsample, vert_subsample>
          <<<gridDim, blockDim, 0, stream>>>(samples, blocks, luma_table, chroma_table);
      });
  }

  void TestChromaSubsampleDistortion() {
    quality_factor = 99;
    CalcOut_ChromaSubsampleDistortion();
    TestKernel(
      [](dim3 gridDim, dim3 blockDim, cudaStream_t stream,
         const SampleDesc* samples, const kernels::BlockDesc<2>* blocks) {
        ChromaSubsampleDistortion<horz_subsample, vert_subsample>
          <<<gridDim, blockDim, 0, stream>>>(samples, blocks);
      });
  }

  CUDAStream stream_;
  TensorListShape<> in_shapes_;
  TestTensorList<uint8_t> in_;
  TestTensorList<uint8_t>out_;
  TestTensorList<uint8_t> out_ref_;

  using BlkSetup = BlockSetup<2, -1>;
  BlkSetup block_setup_;
  using BlockDesc = BlkSetup::BlockDesc;

  float quality_factor = 20.0f;
  DeviceArray<uint8_t, 64> luma_table;
  DeviceArray<uint8_t, 64> chroma_table;
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

