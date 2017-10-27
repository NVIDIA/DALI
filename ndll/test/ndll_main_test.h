#ifndef NDLL_TEST_NDLL_MAIN_TEST_H_
#define NDLL_TEST_NDLL_MAIN_TEST_H_

#include <cstring>

#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
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
    
    batch->template mutable_data<uint8>();
    batch->Resize(shape);
    
    for (int i = 0; i < n; ++i) {
      std::memcpy(batch->template mutable_datum<uint8>(i),
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

  // From OCV example :
  // docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
  cv::Scalar MSSIM(uint8 *a, uint8 *b, int h, int w, int c) {
    cv::Mat i1 = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, a);
    cv::Mat i2 = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, b);
    
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
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
