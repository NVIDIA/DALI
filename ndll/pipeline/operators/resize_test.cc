// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <utility>
#include <string>

#include "ndll/pipeline/operators/resize_crop_mirror.h"
#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

namespace {
// 440 & 410 not supported by npp
const vector<string> hybdec_images = {
  image_folder + "/411.jpg",
  image_folder + "/420.jpg",
  image_folder + "/422.jpg",
  image_folder + "/444.jpg",
  image_folder + "/gray.jpg",
  image_folder + "/411-non-multiple-4-width.jpg",
  image_folder + "/420-odd-height.jpg",
  image_folder + "/420-odd-width.jpg",
  image_folder + "/420-odd-both.jpg",
  image_folder + "/422-odd-width.jpg"
};
}  // namespace

template <typename ImgType>
class ResizeTest : public NDLLTest {
 public:
  void SetUp() {
    if (IsColor(img_type_)) {
      c_ = 3;
    } else if (img_type_ == NDLL_GRAY) {
      c_ = 1;
    } else {
      NDLL_FAIL("Unsupported image type.");
    }

    rand_gen_.seed(time(nullptr));
    LoadJPEGS(hybdec_images, &jpegs_, &jpeg_sizes_);
  }

  void TearDown() {
    NDLLTest::TearDown();
  }

  void VerifyImage(const uint8 *img, const uint8 *img2, int n,
      float mean_bound = 2.0, float std_bound = 3.0) {
    std::vector<uint8> host_img(n), host_img2(n);

    CUDA_CALL(cudaMemcpy(host_img.data(), img, n*sizeof(uint8), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(host_img2.data(), img2, n*sizeof(uint8), cudaMemcpyDefault));

    vector<int> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(static_cast<int>(host_img[i] - host_img2[i]));
    }
    double mean, std;
    MeanStdDev(abs_diff, &mean, &std);

#ifndef NDEBUG
    cout << "num: " << abs_diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif

    // Note: We allow a slight deviation from the ground truth.
    // This value was picked fairly arbitrarily to let the test
    // pass for libjpeg turbo
    ASSERT_LT(mean, mean_bound);
    ASSERT_LT(std, std_bound);
  }

  template <typename T>
  void MeanStdDev(const vector<T> &diff, double *mean, double *std) {
    // Avoid division by zero
    ASSERT_NE(diff.size(), 0);

    double sum = 0, var_sum = 0;
    for (auto &val : diff) {
      sum += val;
    }
    *mean = sum / diff.size();
    for (auto &val : diff) {
      var_sum += (val - *mean)*(val - *mean);
    }
    *std = sqrt(var_sum / diff.size());
  }

 protected:
  const NDLLImageType img_type_ = ImgType::type;
  int c_;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(ResizeTest, Types);

TYPED_TEST(ResizeTest, MultipleData) {
  int batch_size = this->jpegs_.size();
  int num_thread = 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, false);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("jpegs");
  pipe.SetExternalInput("jpegs", data);

  // Decode the images
  pipe.AddOperator(
      OpSpec("TJPGDecoder")
      .AddInput("jpegs", "cpu")
      .AddOutput("images", "cpu"));

  pipe.AddOperator(
      OpSpec("TJPGDecoder")
      .AddInput("jpegs", "cpu")
      .AddOutput("images2", "cpu"));


  // Resize + crop multiple sets of images
  pipe.AddOperator(
      OpSpec("Resize")
      .AddArg("device", "gpu")
      .AddInput("images", "gpu")
      .AddOutput("resized1", "gpu")
      .AddInput("images2", "gpu")
      .AddOutput("resized2", "gpu")
      .AddArg("resize_a", 384)
      .AddArg("resize_b", 384)
      .AddArg("loop_count", 2));

    // Build and run the pipeline
    vector<std::pair<string, string>> outputs = {{"resized1", "gpu"}, {"resized2", "gpu"}};

  pipe.Build(outputs);

  // Decode the images
  pipe.RunCPU();
  pipe.RunGPU();

  DeviceWorkspace results;
  pipe.Outputs(&results);

  // Verify the results
  auto output0 = results.Output<GPUBackend>(0);
  auto output1 = results.Output<GPUBackend>(1);

  // WriteHWCBatch(*output, "image");
  for (int i = 0; i < batch_size; ++i) {
    this->VerifyImage(
        output0->template tensor<uint8>(i),
        output1->template tensor<uint8>(i),
        output0->tensor_shape(i)[0]*output0->tensor_shape(i)[1]*output0->tensor_shape(i)[2]);
  }
}

}  // namespace ndll

