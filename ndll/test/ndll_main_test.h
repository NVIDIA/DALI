#ifndef NDLL_TEST_NDLL_MAIN_TEST_H_
#define NDLL_TEST_NDLL_MAIN_TEST_H_

#include <cstring>

#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/util/image.h"

namespace ndll {

// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/test/test_images";

struct DimPair { int h = 0, w = 0; };

// Some useful test 'types'
struct RGB {
  static const NDLLImageType type = NDLL_RGB;
};
struct BGR {
  static const NDLLImageType type = NDLL_BGR;
};
struct Gray {
  static const NDLLImageType type = NDLL_GRAY;
};

// Main testing fixture to provide common functionality across tests
class NDLLTest : public ::testing::Test {
public:
  virtual void SetUp() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  }

  virtual void TearDown() {
    for (auto &ptr : jpegs_) delete[] ptr;
    for (auto &ptr : images_) delete[] ptr;
  }
  
  int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }
  
  void DecodeJPEGS(NDLLImageType type) {
    images_.resize(jpegs_.size());
    image_dims_.resize(jpegs_.size());
    for (size_t i = 0; i < jpegs_.size(); ++i) {
      cv::Mat img;
      cv::Mat jpeg = cv::Mat(1, jpeg_sizes_[i], CV_8UC1, jpegs_[i]);
      
      ASSERT_TRUE(CheckIsJPEG(jpegs_[i], jpeg_sizes_[i]));
      int flag = IsColor(type) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
      cv::imdecode(jpeg, flag, &img);

      int h = img.rows;
      int w = img.cols;
      cv::Mat out_img(h, w, IsColor(type) ? CV_8UC3 : CV_8UC2);
      if (type == NDLL_RGB) {
        // Convert from BGR to RGB for verification
        cv::cvtColor(img, out_img, CV_BGR2RGB);
      } else {
        out_img = img;
      }
    
      // Copy the decoded image out & save the dims
      ASSERT_TRUE(out_img.isContinuous());
      c_ = IsColor(type) ? 3 : 1;
      images_[i] = new uint8[h*w*c_];
      std::memcpy(images_[i], out_img.ptr(), h*w*c_);

      image_dims_[i].h = h;
      image_dims_[i].w = w;
    }
  }

  // Builds a batch of HWC images
  void MakeImageBatch(int n, Batch<CPUBackend> *batch) {
    NDLL_ENFORCE(images_.size() > 0, "Images must be decoded to create batches");
    vector<Dims> shape(n);
    for (int i = 0; i < n; ++i) {
      shape[i] = {image_dims_[i % images_.size()].h,
                  image_dims_[i % images_.size()].w,
                  c_};
    }
    
    batch->template data<uint8>();
    batch->Resize(shape);
    
    for (int i = 0; i < n; ++i) {
      std::memcpy(batch->template datum<uint8>(i),
          images_[i % images_.size()],
          Product(batch->datum_shape(i)));
    }
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
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;

  // Decoded images
  vector<uint8*> images_;
  vector<DimPair> image_dims_;
  int c_;
}; 
} // namespace ndll

#endif // NDLL_TEST_NDLL_MAIN_TEST_H_
