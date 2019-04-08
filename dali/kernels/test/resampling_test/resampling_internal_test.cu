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
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include "dali/kernels/alloc.h"
#include "dali/kernels/test/mat2tensor.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/imgproc/resample/resampling_impl.cuh"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

template <typename Dst, typename Src>
__global__ void ResampleHorzTestKernel(
    Dst *out, int out_stride, int out_w,
    const Src *in, int in_stride, int in_w, int in_h, int channels,
    ResamplingFilter filter, int support) {
  float scale = static_cast<float>(in_w) / out_w;

  int x0 = blockIdx.x * out_w / gridDim.x;
  int x1 = (blockIdx.x + 1) * out_w / gridDim.x;
  int y0 = blockIdx.y * in_h / gridDim.y;
  int y1 = (blockIdx.y + 1) * in_h / gridDim.y;
  ResampleHorz(
    x0, x1, y0, y1, 0, scale,
    out, out_stride, in, in_stride, in_w,
    channels, filter, support);
}

template <typename Dst, typename Src>
__global__ void ResampleVertTestKernel(
    Dst *out, int out_stride, int out_h,
    const Src *in, int in_stride, int in_w, int in_h, int channels,
    ResamplingFilter filter, int support) {
  float scale = static_cast<float>(in_h) / out_h;

  int x0 = blockIdx.x * in_w / gridDim.x;
  int x1 = (blockIdx.x + 1) * in_w / gridDim.x;
  int y0 = blockIdx.y * out_h / gridDim.y;
  int y1 = (blockIdx.y + 1) * out_h / gridDim.y;
  ResampleVert(
    x0, x1, y0, y1, 0, scale,
    out, out_stride, in, in_stride, in_h,
    channels, filter, support);
}

TEST(Resample, HorizontalGaussian) {
  auto cv_img = testing::data::image("imgproc_test/checkerboard.png");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/resample_horz.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outW = W / 2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, H * outW * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { H, outW, channels } };

  copy(img_in, img);  // NOLINT

  auto filters = GetResamplingFilters();
  ResamplingFilter filter = (*filters)[1];

  int radius = 40;
  filter.rescale(2*radius+1);

  for (int i = 0; i < 100; i++) {
    ResampleHorzTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      filter, filter.support());
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(H, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);  // NOLINT
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(1))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("resample_horz_dif.png", diff);
    return "Test failed. Absolute difference image saved to resample_horz_dif.png";
  }();
}

TEST(Resample, VerticalGaussian) {
  auto cv_img = testing::data::image("imgproc_test/checkerboard.png");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/resample_vert.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H / 2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outH * W * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { outH, W, channels } };

  copy(img_in, img);  // NOLINT

  auto filters = GetResamplingFilters();
  ResamplingFilter filter = (*filters)[1];

  int radius = 40;
  filter.rescale(2*radius+1);

  for (int i = 0; i < 100; i++) {
    ResampleVertTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, W*channels, outH, img_in.data, W*channels, W, H, channels,
      filter, filter.support());
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(outH, W, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);  // NOLINT
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(1))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("resample_vert_dif.png", diff);
    return "Test failed. Absolute difference image saved to resample_hv_dif.png";
  }();
}

class ResamplingTest : public ::testing::Test {
 public:
  enum BlockConfig {
    BlockPerImage = 0,
    BlockPerSpan = 1,
  };

  void SetOutputSize(int w, int h) {
    out_w_ = w;
    out_h_ = h;
    scale_x_ = static_cast<double>(out_w_) / InputWidth();
    scale_y_ = static_cast<double>(out_h_) / InputWidth();
    fixed_size_ = true;
  }
  void SetScale(double sx, double sy) {
    scale_x_ = sx;
    scale_y_ = sy;
    out_w_ = sx * InputWidth();
    out_h_ = sy * InputHeight();
    fixed_size_ = false;
  }
  void SetBlockConfig(BlockConfig config) {
    block_config_ = config;
  }
  void SetSource(const char *input, const char *reference = nullptr) {
    input_ = testing::data::image(input);
    if (reference) {
      reference_ = testing::data::image(reference);
    } else {
      reference_ = cv::Mat();
    }
  }
  void SetProcessingOrder(bool vert_first = false) {
    vert_first_ = vert_first;
  }

  void SetFilters(ResamplingFilter filter_x, ResamplingFilter filter_y) {
    flt_x_ = filter_x;
    flt_y_ = filter_y;
  }

  void CopyOutputToCPU(bool force = false) {
    if (force || output_.empty()) {
      auto type = CV_MAKETYPE(CV_8U, img_out_.shape[2]);
      output_.create(img_out_.shape[0], img_out_.shape[1], type);
      auto img_out_cpu = view_as_tensor<uint8_t, 3>(output_);
      copy(img_out_cpu, img_out_);
      cudaDeviceSynchronize();
    }
  }

  void SaveOutput(const char *file) {
    CopyOutputToCPU();
    cv::imwrite(file, output_);
  }

  void Verify(double epsilon, const char *diff_image = nullptr) {
    ASSERT_FALSE(reference_.empty()) << "Cannot verify with empty refernce";
    CopyOutputToCPU();
    auto img_ref_cpu = view_as_tensor<uint8_t, 3>(reference_);
    auto img_out_cpu = view_as_tensor<uint8_t, 3>(output_);
    EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(epsilon))) <<
    [&]()->std::string {
      if (diff_image) {
        if (img_out_cpu.shape == img_ref_cpu.shape) {
          cv::Mat diff;
          cv::absdiff(output_, reference_, diff);
          cv::imwrite(diff_image, diff);
          return "Test failed. Absolute difference image saved to " + std::string(diff_image);
        } else {
          return "Test failed. Ouput and reference have different size - no difference written.";
        }
      } else {
        return "Test failed. Difference image not saved (no file name specified).";
      }
    }();
  }

  void VerifyWithMargin(int hmargin, int vmargin, double epsilon,
                        const char *diff_image = nullptr) {
    ASSERT_FALSE(reference_.empty()) << "Cannot verify with empty refernce";
    CopyOutputToCPU();
    cv::Mat tmp_out;
    output_.copyTo(tmp_out);
    {
      int W = tmp_out.cols;
      int H = tmp_out.rows;
      if (hmargin) {
        cv::Rect L(0, 0, hmargin, H);
        cv::Rect R(W-hmargin, 0, hmargin, H);
        reference_(L).copyTo(tmp_out(L));
        reference_(R).copyTo(tmp_out(R));
      }
      if (vmargin) {
        cv::Rect T(0, 0, W, vmargin);
        cv::Rect B(0, H-vmargin, W, vmargin);
        reference_(T).copyTo(tmp_out(T));
        reference_(B).copyTo(tmp_out(B));
      }
    }
    auto img_ref_cpu = view_as_tensor<uint8_t, 3>(reference_);
    auto img_out_cpu = view_as_tensor<uint8_t, 3>(tmp_out);
    EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(epsilon))) <<
    [&]()->std::string {
      if (diff_image) {
        if (img_out_cpu.shape == img_ref_cpu.shape) {
          cv::Mat diff;
          cv::absdiff(output_, reference_, diff);
          cv::imwrite(diff_image, diff);
          return "Test failed. Absolute difference image saved to " + std::string(diff_image);
        } else {
          return "Test failed. Ouput and reference have different size - no difference written.";
        }
      } else {
        return "Test failed. Difference image not saved (no file name specified).";
      }
    }();
  }

  void Prepare() {
    int W = InputWidth();
    int H = InputHeight();

    // update output size or scale
    if (fixed_size_) {
      scale_x_ = static_cast<double>(out_w_) / W;
      scale_y_ = static_cast<double>(out_h_) / H;
    } else {
      out_w_ = W * scale_x_;
      out_h_ = H * scale_y_;
    }

    PrepareTensors(W, H, out_w_, out_h_, input_.channels());
    copy(img_in_, view_as_tensor<uint8_t, 3>(input_));
    output_ = cv::Mat();
  }

  void PrepareTensors(int W, int H, int outW, int outH, int channels) {
    int tmpW = vert_first_ ? W : outW;
    int tmpH = vert_first_ ? outH : H;
    gpu_mem_in_ = memory::alloc_unique<uint8_t>(AllocType::GPU, W * H * channels);
    gpu_mem_tmp_ = memory::alloc_unique<float>(AllocType::GPU, tmpW * tmpH * channels);
    gpu_mem_out_ = memory::alloc_unique<uint8_t>(AllocType::GPU, outW * outH * channels);

    img_in_  = { gpu_mem_in_.get(),  { H,    W,    channels } };
    img_tmp_ = { gpu_mem_tmp_.get(), { tmpH, tmpW, channels } };
    img_out_ = { gpu_mem_out_.get(), { outH, outW, channels } };
  }

  void Run() {
    bool per_span = block_config_  == BlockPerSpan;
    int W = img_in_.shape[1];
    int H = img_in_.shape[0];
    int tmpW = img_tmp_.shape[1];
    int tmpH = img_tmp_.shape[0];
    int outW = img_out_.shape[1];
    int outH = img_out_.shape[0];
    int channels = img_in_.shape[2];
    assert(img_out_.shape[2] == img_in_.shape[2]);

    #if DALI_DEBUG
    dim3 block(32, 8);
    #else
    dim3 block(32, 24);
    #endif

    if (vert_first_) {
      assert(img_tmp_.shape == TensorShape<3>(outH, W, channels));

      dim3 grid = per_span ? dim3(div_ceil(tmpW, block.x), div_ceil(tmpH, block.y)) : dim3(1);
      ResampleVertTestKernel<<<grid, block, ResampleSharedMemSize>>>(
        img_tmp_.data, tmpW*channels, tmpH, img_in_.data, W*channels, W, H, channels,
        flt_y_, flt_y_.support());
      grid = per_span ? dim3(div_ceil(outW, block.x), div_ceil(outH, block.y)) : dim3(1);
      ResampleHorzTestKernel<<<grid, block, ResampleSharedMemSize>>>(
        img_out_.data, outW*channels, outW, img_tmp_.data, tmpW*channels, tmpW, tmpH, channels,
        flt_x_, flt_x_.support());
    } else {
      assert(img_tmp_.shape == TensorShape<3>(H, outW, channels));

      dim3 grid = per_span ? dim3(div_ceil(tmpW, block.x), div_ceil(tmpH, block.y)) : dim3(1);
      ResampleHorzTestKernel<<<grid, block, ResampleSharedMemSize>>>(
        img_tmp_.data, tmpW*channels, tmpW, img_in_.data, W*channels, W, H, channels,
        flt_x_, flt_x_.support());
      grid = per_span ? dim3(div_ceil(outW, block.x), div_ceil(outH, block.y)) : dim3(1);
      ResampleVertTestKernel<<<grid, block, ResampleSharedMemSize>>>(
        img_out_.data, outW*channels, outH, img_tmp_.data, tmpW*channels, tmpW, tmpH, channels,
        flt_y_, flt_y_.support());
    }
  }

  int InputWidth() const {
    return input_.cols;
  }
  int InputHeight() const {
    return input_.rows;
  }
  int OutputWidth() const {
    return fixed_size_ ? out_w_ : static_cast<int>(InputWidth() * scale_x_);
  }
  int OutputHeight() const {
    return fixed_size_ ? out_h_ : static_cast<int>(InputHeight() * scale_y_);
  }
  double ScaleX() const {
    return fixed_size_ ? static_cast<double>(out_w_) / InputWidth() : scale_x_;
  }
  double ScaleY() const {
    return fixed_size_ ? static_cast<double>(out_h_) / InputHeight() : scale_y_;
  }

  cv::Mat input_, reference_, output_;

  void SetUp() override {
    input_ = cv::Mat();
    reference_ = cv::Mat();
    output_ = cv::Mat();
    fixed_size_ = false;
    scale_x_ = scale_y_ = 1;
    gpu_mem_in_.reset();
    gpu_mem_tmp_.reset();
    gpu_mem_out_.reset();
    img_in_ = {};
    img_out_ = {};
    img_tmp_ = {};
    out_w_ = out_h_ = 0;
    block_config_ = BlockPerImage;
    vert_first_ = false;
    flt_x_ = flt_y_ = {};
  }

 private:
  int out_w_, out_h_;
  double scale_x_ = 1, scale_y_ = 1;
  bool fixed_size_ = false;

  ResamplingFilter flt_x_, flt_y_;

  TensorView<StorageGPU, uint8_t, 3> img_in_, img_out_;
  TensorView<StorageGPU, float, 3> img_tmp_;
  using deleter = std::function<void(void*)>;
  std::unique_ptr<uint8_t, deleter> gpu_mem_in_, gpu_mem_out_;
  std::unique_ptr<float, deleter> gpu_mem_tmp_;
  BlockConfig block_config_ = BlockPerImage;
  bool vert_first_ = false;
};

TEST_F(ResamplingTest, ResampleGauss) {
  SetSource("imgproc_test/moire2.png", "imgproc_test/ref_out/resample_out.png");
  SetOutputSize(InputWidth()-1, InputHeight()-3);
  auto filters = GetResamplingFilters();
  float sigmaX = (1/ScaleX() - 0.3f) / sqrt(2);
  float sigmaY = (1/ScaleY() - 0.3f) / sqrt(2);
  auto fx = filters->Gaussian(sigmaX);
  auto fy = filters->Gaussian(sigmaY);
  SetFilters(fx, fy);
  Prepare();
  Run();
  // SaveOutput("resample_hv_out.png");
  Verify(1, "resample_hv_dif.png");
}

TEST_F(ResamplingTest, ResampleVHGauss) {
  SetSource("imgproc_test/moire2.png", "imgproc_test/ref_out/resample_out.png");
  SetOutputSize(InputWidth()-1, InputHeight()-3);
  auto filters = GetResamplingFilters();
  float sigmaX = (1/ScaleX() - 0.3f) / sqrt(2);
  float sigmaY = (1/ScaleY() - 0.3f) / sqrt(2);
  auto fx = filters->Gaussian(sigmaX);
  auto fy = filters->Gaussian(sigmaY);
  SetProcessingOrder(true);
  SetFilters(fx, fy);
  Prepare();
  Run();
  // SaveOutput("resample_vh_out.png");
  Verify(1, "resample_vh_dif.png");
}

TEST_F(ResamplingTest, SeparableTriangular) {
  SetSource("imgproc_test/alley.png", "imgproc_test/ref_out/alley_tri_PIL.png");
  SetOutputSize(300, 300);

  auto filters = GetResamplingFilters();
  auto fx = filters->Triangular(1 / ScaleX());
  auto fy = filters->Triangular(1 / ScaleY());

  Prepare();
  SetFilters(fx, fy);
  Run();
  // SaveOutput("alley_tri.png");
  VerifyWithMargin(1, 1, 1, "alley_tri_dif.png");
}

TEST_F(ResamplingTest, GaussianBlur) {
  SetSource("imgproc_test/alley.png", "imgproc_test/ref_out/alley_blurred.png");
  auto filters = GetResamplingFilters();
  float sigmaX = 6.0f / sqrt(2);
  float sigmaY = 6.0f / sqrt(2);

  SetFilters(filters->Gaussian(sigmaX), filters->Gaussian(sigmaY));
  Prepare();
  Run();
  Verify(2, "blur_dif.png");
}

// DISABLED: this 'test' is only for producing images for manual assessment
TEST_F(ResamplingTest, DISABLED_ProgressiveOutputs) {
  SetSource("imgproc_test/alley.png", nullptr);

  auto filters = GetResamplingFilters();
  for (int i = 0; i < 10; i++) {
    float sigmaX = powf(1.10f, i) * 0.5f;
    float sigmaY = powf(1.10f, i) * 0.5f;

    ResamplingFilter fx = filters->Gaussian(sigmaX);
    ResamplingFilter fy = filters->Gaussian(sigmaY);
    SetFilters(fx, fy);
    Prepare();
    Run();
    char name[64] = {};
    snprintf(name, sizeof(name), "blur_%i.png", i);
    SaveOutput(name);
  }
}

TEST_F(ResamplingTest, Lanczos3) {
  SetSource("imgproc_test/score.png", "imgproc_test/ref_out/score_lanczos3.png");
  SetScale(5, 5);
  auto filters = GetResamplingFilters();
  ResamplingFilter f = filters->Lanczos3();
  SetFilters(f, f);
  Prepare();
  Run();
  Verify(1, "score_lanczos_dif.png");
}

// It passes. It's DISABLED until the test data propagates to CI
TEST_F(ResamplingTest, DISABLED_Cubic) {
  SetSource("imgproc_test/score.png", "imgproc_test/ref_out/score_cubic.png");
  SetOutputSize(200, 93);
  auto filters = GetResamplingFilters();
  ResamplingFilter f = filters->Cubic();
  SetFilters(f, f);
  Prepare();
  Run();
  Verify(1, "score_cubic_dif.png");
}

}  // namespace kernels
}  // namespace dali
