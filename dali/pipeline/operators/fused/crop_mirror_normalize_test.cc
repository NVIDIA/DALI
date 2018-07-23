// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <opencv2/opencv.hpp>

#include <utility>
#include <string>

#include "dali/pipeline/operators/fused/resize_crop_mirror.h"
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/image/jpeg.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test.h"

namespace dali {

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
class CropMirrorNormalizePermuteTest : public DALITest {
 public:
  void SetUp() {
    if (IsColor(img_type_)) {
      c_ = 3;
    } else if (img_type_ == DALI_GRAY) {
      c_ = 1;
    } else {
      DALI_FAIL("Unsupported image type.");
    }

    rand_gen_.seed(time(nullptr));
    LoadJPEGS(hybdec_images, &jpegs_, &jpeg_sizes_);
  }

  void TearDown() {
    DALITest::TearDown();
  }

  void VerifyImage(const float *img, const float *img2, int n,
      float mean_bound = 2.0, float std_bound = 3.0) {
    std::vector<float> host_img(n), host_img2(n);

    CUDA_CALL(cudaMemcpy(host_img.data(), img, n*sizeof(float), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(host_img2.data(), img2, n*sizeof(float), cudaMemcpyDefault));

    vector<int> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(host_img[i] - host_img2[i]);
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
  const DALIImageType img_type_ = ImgType::type;
  int c_;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(CropMirrorNormalizePermuteTest, Types);

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleData) {
  int batch_size = this->jpegs_.size();
  int num_thread = 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("jpegs");
  pipe.SetExternalInput("jpegs", data);

  // Decode the images
  pipe.AddOperator(
      OpSpec("HostDecoder")
      .AddArg("output_type", this->img_type_)
      .AddInput("jpegs", "cpu")
      .AddOutput("images", "cpu"));

  pipe.AddOperator(
      OpSpec("HostDecoder")
      .AddArg("output_type", this->img_type_)
      .AddInput("jpegs", "cpu")
      .AddOutput("images2", "cpu"));


  std::vector<float> mean_vec(this->c_);
  for (int i = 0; i < this->c_; ++i) {
    mean_vec[i] = 0.;
  }

  // CropMirrorNormalizePermute + crop multiple sets of images
  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddInput("images", "gpu")
      .AddOutput("cropped1", "gpu")
      .AddInput("images2", "gpu")
      .AddOutput("cropped2", "gpu")
      .AddArg("crop", vector<int>{64, 64})
      .AddArg("mean", mean_vec)
      .AddArg("std", mean_vec)
      .AddArg("image_type", this->img_type_)
      .AddArg("num_input_sets", 2));

    // Build and run the pipeline
    vector<std::pair<string, string>> outputs = {{"cropped1", "gpu"}, {"cropped2", "gpu"}};

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
        output0->template tensor<float>(i),
        output1->template tensor<float>(i),
        output0->tensor_shape(i)[0]*output0->tensor_shape(i)[1]*output0->tensor_shape(i)[2]);
  }
}

}  // namespace dali


