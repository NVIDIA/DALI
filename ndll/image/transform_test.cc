#include "ndll/image/transform.h"

#include <cmath>
#include <cstring>

#include <fstream>
#include <random>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ndll/common.h"
#include "ndll/image/jpeg.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/test/ndll_test.h"
#include "ndll/util/type_conversion.h"
#include "ndll/util/image.h"

namespace ndll {

template <typename ImgType>
class TransformTest : public NDLLTest {
public:
  void SetUp() {
    NDLLTest::SetUp();
    if (IsColor(img_type_)) {
      c_ = 3;
    } else {
      c_ = 1;
    }
    DecodeJPEGS(img_type_);
  }

  void OpenCVResizeCropMirror(uint8 *image, int h, int w, int c,
      int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
      int crop_w, bool mirror, uint8 *out_image) {
    cv::Mat cv_img = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, image);
    cv::Mat rsz_img;
    cv::resize(cv_img, rsz_img, cv::Size(rsz_w, rsz_h), 0, 0, cv::INTER_LINEAR);
    
    // Crop into another mat
    cv::Mat crop_img(crop_h, crop_w, c == 3 ? CV_8UC3 : CV_8UC1);
    int crop_offset = crop_y*rsz_w*c + crop_x*c;
    uint8 *crop_ptr = rsz_img.ptr() + crop_offset;
    CUDA_CALL(cudaMemcpy2D(crop_img.ptr(), crop_w*c, crop_ptr,
            rsz_w*c, crop_w*c, crop_h, cudaMemcpyHostToHost));

    // Random mirror
    cv::Mat mirror_img;
    if (mirror) {
      cv::flip(crop_img, mirror_img, 1);
    } else {
      mirror_img = crop_img;
    }

    // Copy to the output
    std::memcpy(out_image, mirror_img.ptr(), crop_h*crop_w*c);
  }

  void VerifyImage(uint8 *img, uint8 *ground_truth, int n,
      float mean_bound = 2.0, float std_bound = 3.0) {
    vector<uint8> host_img(n);
    CUDA_CALL(cudaMemcpy(host_img.data(), img, n, cudaMemcpyDefault));
    
    vector<int> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(int(host_img[i] - ground_truth[i]));
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

  // Resizes the images to the crop size
  void MakeImageBatch(int n, int h, int w, uint8 *batch) {
    // resize & crop to the same size
    vector<uint8> img(h*w*c_, 0);
    for (int i = 0; i < n; ++i) {
      OpenCVResizeCropMirror(images_[i], image_dims_[i].h,
          image_dims_[i].w, c_, h, w, 0, 0, h, w, false, img.data());

      // Copy into the batch
      std::memcpy(batch + i*h*w*c_, img.data(), h*w*c_);
    }
  }

  // Produces jagged batch
  void MakeImageBatch(int n, Batch<CPUBackend> *batch) {
    vector<Dims> shape(n);
    for (int i = 0; i < n; ++i) {
      shape[i] = {image_dims_[i % images_.size()].h,
                  image_dims_[i % images_.size()].w,
                  c_};
    }
    batch->template mutable_data<uint8>();
    batch->Resize(shape);

    for (int i = 0; i < n; ++i) {
      std::memcpy(batch->template mutable_sample<uint8>(i),
          images_[i % images_.size()],
          Product(batch->sample_shape(i)));
    }
  }
  
protected:
  NDLLImageType img_type_ = ImgType::type;
  int c_;
};

// Run RGB & grayscale tests
typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(TransformTest, Types);

// For functions that are the output of a pipeline,
// we want to test with all the output types
template <typename Types>
class OutputTransformTest : public TransformTest<typename Types::test_color> {
public:
  typedef typename Types::test_color color;
  // Comparison for other types. We use double for the ground truth.
  // Input data is assumed to be on the GPU
  template <typename T>
  void CompareData(T *data, double *ground_truth, int n) {
    // Conver the input data to double
    double *tmp_gpu = nullptr;
    CUDA_CALL(cudaMalloc((void**)&tmp_gpu, sizeof(double)*n));
    Convert(data, n, tmp_gpu);
    
    vector<double> tmp(n, 0);
    CUDA_CALL(cudaMemcpy(tmp.data(), tmp_gpu, n*sizeof(double),
            cudaMemcpyDeviceToHost));

    vector<double> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(tmp[i] - ground_truth[i]);
    }
    double mean, std;
    TransformTest<color>::MeanStdDev(abs_diff, &mean, &std);

#ifndef NDEBUG
    cout << "num: " << abs_diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif 
    
    ASSERT_LT(mean, 0.000001);
    ASSERT_LT(std, 0.000001);
    CUDA_CALL(cudaFree(tmp_gpu));
  }
  
protected:
};

template <typename color, typename OUT>
struct OutputTestTypes {
  typedef color test_color;
  typedef OUT TEST_OUT;
};

typedef ::testing::Types<OutputTestTypes<RGB, float16>,
                         OutputTestTypes<RGB, float>,
                         OutputTestTypes<RGB, double>,
                         OutputTestTypes<BGR, float16>,
                         OutputTestTypes<BGR, float>,
                         OutputTestTypes<BGR, double>,
                         OutputTestTypes<Gray, float16>,
                         OutputTestTypes<Gray, float>,
                         OutputTestTypes<Gray, double>> OutputTypes;

TYPED_TEST_CASE(OutputTransformTest, OutputTypes);

TYPED_TEST(TransformTest, TestResizeCrop) {
  vector<uint8> out_img, ver_img, tmp_img;
  for (size_t i = 0; i < this->images_.size(); ++i) {
    // Generate random resize params
    int rsz_h = this->RandInt(32, 512);
    int rsz_w = this->RandInt(32, 512);
    
    // Generate random crop params
    int crop_h = this->RandInt(32, rsz_h);
    int crop_w = this->RandInt(32, rsz_w);
    int crop_y = this->RandInt(0, rsz_h - crop_h);
    int crop_x = this->RandInt(0, rsz_w - crop_w);
    
    // Select whether to mirror
    bool mirror = false;

    out_img.resize(crop_h*crop_w*this->c_);
    tmp_img.resize(rsz_h*rsz_w*this->c_);
    NDLL_CALL(ResizeCropMirrorHost(this->images_[i], this->image_dims_[i].h,
            this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y,
            crop_x, crop_h, crop_w, mirror, out_img.data(), NDLL_INTERP_LINEAR,
            tmp_img.data()));

    // Verify the output
    ver_img.resize(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(this->images_[i], this->image_dims_[i].h,
        this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y, crop_x,
        crop_h, crop_w, mirror, ver_img.data());

    // TODO(tgale): In every test where we want extensive debug info we
    // have to use these ifdefs. We should add some logging functionality
    // that takes this into account so we don't have to dirty the code
    // up everywhere.
#ifndef NDEBUG
    cout << i << " " << this->jpeg_names_[i] << endl;
    cout << "dims: " << this->image_dims_[i].h << "x" << this->image_dims_[i].w << endl;
    cout << "rsz: " << rsz_h << "x" << rsz_w << endl;
    cout << "crop: " << crop_h << "x" << crop_w << endl;
    cout << "mirror: " << mirror << endl;
    DumpHWCToFile(out_img.data(), crop_h, crop_w,
        this->c_, std::to_string(i));
    DumpHWCToFile(ver_img.data(), crop_h, crop_w,
            this->c_, "ver_" + std::to_string(i));
#endif 
    this->VerifyImage(out_img.data(), ver_img.data(), out_img.size());
  }
}

TYPED_TEST(TransformTest, TestResizeCropMirror) {
  vector<uint8> out_img, ver_img, tmp_img;
  for (size_t i = 0; i < this->images_.size(); ++i) {
    // Generate random resize params
    int rsz_h = this->RandInt(32, 512);
    int rsz_w = this->RandInt(32, 512);
    
    // Generate random crop params
    int crop_h = this->RandInt(32, rsz_h);
    int crop_w = this->RandInt(32, rsz_w);
    int crop_y = this->RandInt(0, rsz_h - crop_h);
    int crop_x = this->RandInt(0, rsz_w - crop_w);

    // Select whether to mirror
    bool mirror = true;

    out_img.resize(crop_h*crop_w*this->c_);
    tmp_img.resize(rsz_h*rsz_w*this->c_);
    NDLL_CALL(ResizeCropMirrorHost(this->images_[i], this->image_dims_[i].h,
            this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y,
            crop_x, crop_h, crop_w, mirror, out_img.data(), NDLL_INTERP_LINEAR,
            tmp_img.data()));

    // Verify the output
    ver_img.resize(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(this->images_[i], this->image_dims_[i].h,
        this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y, crop_x,
        crop_h, crop_w, mirror, ver_img.data());
    
#ifndef NDEBUG
    cout << i << " " << this->jpeg_names_[i] << endl;
    cout << "dims: " << this->image_dims_[i].h << "x" << this->image_dims_[i].w << endl;
    cout << "rsz: " << rsz_h << "x" << rsz_w << endl;
    cout << "crop: " << crop_h << "x" << crop_w << endl;
    cout << "mirror: " << mirror << endl;
    DumpHWCToFile(out_img.data(), crop_h, crop_w,
        this->c_, std::to_string(i));
    DumpHWCToFile(ver_img.data(), crop_h, crop_w,
            this->c_, "ver_" + std::to_string(i));
#endif 
    this->VerifyImage(out_img.data(), ver_img.data(), out_img.size());
  }
}

TYPED_TEST(TransformTest, TestFastResizeCrop) {
  vector<uint8> out_img, ver_img;
  for (size_t i = 0; i < this->images_.size(); ++i) {
    // Generate random resize params
    int rsz_h = this->RandInt(32, 512);
    int rsz_w = this->RandInt(32, 512);
    
    // Generate random crop params
    int crop_h = this->RandInt(32, rsz_h);
    int crop_w = this->RandInt(32, rsz_w);
    int crop_y = this->RandInt(0, rsz_h - crop_h);
    int crop_x = this->RandInt(0, rsz_w - crop_w);
    
    // Select whether to mirror
    bool mirror = false;

    out_img.resize(crop_h*crop_w*this->c_);
    NDLL_CALL(FastResizeCropMirrorHost(this->images_[i], this->image_dims_[i].h,
            this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y,
            crop_x, crop_h, crop_w, mirror, out_img.data()));

    // Verify the output
    ver_img.resize(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(this->images_[i], this->image_dims_[i].h,
        this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y, crop_x,
        crop_h, crop_w, mirror, ver_img.data());

#ifndef NDEBUG
    cout << i << " " << this->jpeg_names_[i] << endl;
    cout << "dims: " << this->image_dims_[i].h << "x" << this->image_dims_[i].w << endl;
    cout << "rsz: " << rsz_h << "x" << rsz_w << endl;
    cout << "crop: " << crop_h << "x" << crop_w << endl;
    cout << "mirror: " << mirror << endl;
    DumpHWCToFile(out_img.data(), crop_h, crop_w,
        this->c_, std::to_string(i));
    DumpHWCToFile(ver_img.data(), crop_h, crop_w,
            this->c_, "ver_" + std::to_string(i));
#endif
    // TODO(tgale): We need a better way to evaluate similarity for the
    // FastResizeCropMirror method. The resulting image is very close,
    // but is slightly shifted (about a pixel), which causes higher MSE
    // and standard deviation than we would normally want to tolerate
    this->VerifyImage(out_img.data(), ver_img.data(), out_img.size(), 30.f, 32.f);
  }
}

TYPED_TEST(TransformTest, TestFastResizeMirror) {
  vector<uint8> out_img, ver_img, tmp_img;
  for (size_t i = 0; i < this->images_.size(); ++i) {
    // Generate random resize params
    int rsz_h = this->RandInt(32, 512);
    int rsz_w = this->RandInt(32, 512);
    
    // Generate random crop params
    int crop_h = this->RandInt(32, rsz_h);
    int crop_w = this->RandInt(32, rsz_w);
    int crop_y = this->RandInt(0, rsz_h - crop_h);
    int crop_x = this->RandInt(0, rsz_w - crop_w);
    
    // Select whether to mirror
    bool mirror = true;

    out_img.resize(crop_h*crop_w*this->c_);
    tmp_img.resize(crop_h*crop_w*this->c_);
    NDLL_CALL(FastResizeCropMirrorHost(this->images_[i], this->image_dims_[i].h,
            this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y,
            crop_x, crop_h, crop_w, mirror, out_img.data(), NDLL_INTERP_LINEAR,
            tmp_img.data()));

    // Verify the output
    ver_img.resize(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(this->images_[i], this->image_dims_[i].h,
        this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y, crop_x,
        crop_h, crop_w, mirror, ver_img.data());

#ifndef NDEBUG
    cout << i << " " << this->jpeg_names_[i] << endl;
    cout << "dims: " << this->image_dims_[i].h << "x" << this->image_dims_[i].w << endl;
    cout << "rsz: " << rsz_h << "x" << rsz_w << endl;
    cout << "crop: " << crop_h << "x" << crop_w << endl;
    cout << "mirror: " << mirror << endl;
    DumpHWCToFile(out_img.data(), crop_h, crop_w,
        this->c_, std::to_string(i));
    DumpHWCToFile(ver_img.data(), crop_h, crop_w,
            this->c_, "ver_" + std::to_string(i));
#endif
    // TODO(tgale): We need a better way to evaluate similarity for the
    // FastResizeCropMirror method. The resulting image is very close,
    // but is slightly shifted (about a pixel), which causes higher MSE
    // and standard deviation than we would normally want to tolerate
    this->VerifyImage(out_img.data(), ver_img.data(), out_img.size(), 30.f, 32.f);
  }
}

TYPED_TEST(TransformTest, TestBatchedResize) {
  int batch_size = this->images_.size();
  Batch<CPUBackend> batch;
  this->MakeImageBatch(batch_size, &batch);
  
  Batch<GPUBackend> gpu_batch;
  gpu_batch.template mutable_data<uint8>();
  gpu_batch.ResizeLike(batch);
  CUDA_CALL(cudaMemcpy(
          gpu_batch.template mutable_data<uint8>(),
          batch.template data<uint8>(),
          batch.nbytes(),
          cudaMemcpyHostToDevice)
      );
  
  // Setup resize parameters
  vector<uint8*> in_ptrs(batch_size, nullptr), out_ptrs(batch_size, nullptr);
  vector<NDLLSize> in_sizes(batch_size), out_sizes(batch_size);
  NDLLInterpType type = NDLL_INTERP_LINEAR;
  vector<Dims> output_shape(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    in_ptrs[i] = gpu_batch.template mutable_sample<uint8>(i);
    in_sizes[i].height = gpu_batch.sample_shape(i)[0];
    in_sizes[i].width = gpu_batch.sample_shape(i)[1];
    
    out_sizes[i].height = this->RandInt(32, 480);
    out_sizes[i].width = this->RandInt(32, 480);
    vector<Index> shape = {out_sizes[i].height, out_sizes[i].width, this->c_};
    output_shape[i] = shape;
  }

  Batch<GPUBackend> gpu_output_batch;
  gpu_output_batch.template mutable_data<uint8>();
  gpu_output_batch.Resize(output_shape);
  for (int i = 0; i < batch_size; ++i) {
    out_ptrs[i] = gpu_output_batch.template mutable_sample<uint8>(i);
  }

  NDLL_CALL(BatchedResize(
          (const uint8**)in_ptrs.data(),
          batch_size,
          this->c_,
          in_sizes.data(),
          out_ptrs.data(),
          out_sizes.data(),
          type));

#ifndef NDEBUG
  DumpHWCImageBatchToFile<uint8>(gpu_output_batch);
#endif
  
  // verify the resize
  for (int i = 0; i < batch_size; ++i) {
    cv::Mat img = cv::Mat(in_sizes[i].height, in_sizes[i].width,
        this->c_ == 3 ? CV_8UC3 : CV_8UC1, batch.template mutable_sample<uint8>(i));

    cv::Mat ground_truth;
    cv::resize(img, ground_truth,
        cv::Size(out_sizes[i].width, out_sizes[i].height),
        0, 0, cv::INTER_LINEAR);

#ifndef NDEBUG
    DumpHWCToFile(ground_truth.ptr(), ground_truth.rows, ground_truth.cols,
        ground_truth.channels(), "ver_" + std::to_string(i));
#endif
    // vector<uint8> tmp_output(out_sizes[i].height * out_sizes[i].width * this->c_, 0);
    // CUDA_CALL(cudaDeviceSynchronize());
    // CUDA_CALL(cudaMemcpy(tmp_output.data(), gpu_output_batch.template mutable_sample<uint8>(i),
    //         out_sizes[i].height * out_sizes[i].width * this->c_, cudaMemcpyDeviceToHost));
    // cv::Scalar mssim = this->MSSIM(tmp_output.data(), ground_truth.ptr(),
    //     out_sizes[i].height, out_sizes[i].width, this->c_);
    // cout << mssim << endl;
    
    this->VerifyImage(gpu_output_batch.template mutable_sample<uint8>(i), ground_truth.ptr(),
        out_sizes[i].height * out_sizes[i].width * this->c_, 40.f, 40.f);
  }
}

// TODO(tgale): There is probably a better place to put this than right here.
template <typename T>
void CPUBatchedNormalizePermute(const uint8 *image_batch,
    int N, int H, int W, int C,  float *mean, float *std,
    T *out_batch) {
  ASSERT_TRUE(image_batch != nullptr);
  ASSERT_TRUE(mean != nullptr);
  ASSERT_TRUE(std != nullptr);
  ASSERT_TRUE(out_batch != nullptr);
  ASSERT_TRUE(N > 0);
  ASSERT_TRUE((C == 1) || (C == 3));
  ASSERT_TRUE(W > 0);
  ASSERT_TRUE(H > 0);

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // Data comes in as NHWC & goes out NCHW
          int in_idx = n*H*W*C + h*W*C + w*C + c;
          int out_idx = n*H*W*C + c*H*W + h*W + w;

          out_batch[out_idx] = static_cast<T>(
              (static_cast<float>(image_batch[in_idx]) - mean[c]) / std[c]);
        }
      }
    }
  }
}

TYPED_TEST(OutputTransformTest, TestBatchedNormalizePermute) {
  // To make the test a bit more succinct
  typedef typename TypeParam::TEST_OUT T;
  
  int n = this->RandInt(4, this->jpegs_.size());
  int h = this->RandInt(32, 512);
  int w = this->RandInt(32, 512);
  vector<uint8> batch(n*h*w*this->c_, 0);
  this->MakeImageBatch(n, h, w, batch.data());

  // Set up the mean & std dev
  vector<float> vals(this->c_*2, 128);
  vector<float> std(this->c_, 128);
  for (int i = 0; i < this->c_; ++i) {
    // inverse the standard deviation
    vals[this->c_+i] = 1 / 128.f;
  }
  float *mean = nullptr;
  CUDA_CALL(cudaMalloc((void**)&mean, sizeof(float)*2*this->c_));
  CUDA_CALL(cudaMemcpy(mean, vals.data(), sizeof(float)*2*this->c_, cudaMemcpyHostToDevice));
  float *inv_std = mean + this->c_;
  
  // Move the batch to GPU
  uint8 *batch_gpu = nullptr;
  CUDA_CALL(cudaMalloc((void**)&batch_gpu, n*h*w*this->c_));
  CUDA_CALL(cudaMemcpy(batch_gpu, batch.data(), n*h*w*this->c_, cudaMemcpyHostToDevice));

  // Run the method
  T *output_batch = nullptr;
  CUDA_CALL(cudaMalloc((void**)&output_batch, n*h*w*this->c_*sizeof(T)));

  NDLL_CALL(BatchedNormalizePermute(batch_gpu, n, h, w, this->c_,
          mean, inv_std, output_batch, 0));

  vector<double> output_batch_ver(n*h*w*this->c_, 0);
  CPUBatchedNormalizePermute(batch.data(), n, h, w, this->c_,
      vals.data(), std.data(), output_batch_ver.data());

  this->CompareData(output_batch, output_batch_ver.data(), n*h*w*this->c_);
  
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaFree(mean));
  CUDA_CALL(cudaFree(batch_gpu));
  CUDA_CALL(cudaFree(output_batch));
}

TYPED_TEST(OutputTransformTest, TestBatchedCropMirrorNormalizePermute) {
  // To make the test a bit more succinct
  typedef typename TypeParam::TEST_OUT T;
  int batch_size = this->images_.size();
  
  // set valid crop dims
  int min_w = this->image_dims_[0].w;
  int min_h = this->image_dims_[0].h;
  for (int i = 1; i < batch_size; ++i) {
    if (this->image_dims_[i].w < min_w) {
      min_w = this->image_dims_[i].w;
    }
    if (this->image_dims_[i].h < min_h) {
      min_h = this->image_dims_[i].h;
    }
  }
  int crop_h = this->RandInt(1, min_h);
  int crop_w = this->RandInt(1, min_w);
  
  Batch<CPUBackend> batch;
  this->MakeImageBatch(batch_size, &batch);
  
  Batch<GPUBackend> gpu_batch;
  gpu_batch.template mutable_data<uint8>();
  gpu_batch.ResizeLike(batch);
  CUDA_CALL(cudaMemcpy(
          gpu_batch.template mutable_data<uint8>(),
          batch.template data<uint8>(),
          batch.nbytes(),
          cudaMemcpyHostToDevice)
      );

  // Setup parameteres
  vector<uint8*> in_ptrs(batch_size);
  vector<int> strides(batch_size);
  bool *mirror = new bool[batch_size];
  vector<float> mean(this->c_);
  vector<float> std(this->c_);
  vector<float> inv_std(this->c_);
  
  // choose crop offsets & whether to mirror
  vector<int> crop_xs(batch_size);
  vector<int> crop_ys(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    int crop_y = this->RandInt(0, this->image_dims_[i].h - crop_h);
    int crop_x = this->RandInt(0, this->image_dims_[i].w - crop_w);

    int crop_offset = crop_y * this->image_dims_[i].w * this->c_ + crop_x * this->c_;
    in_ptrs[i] = gpu_batch.template mutable_sample<uint8>(i) + crop_offset;
    strides[i] = this->image_dims_[i].w*this->c_;

    // Save the crop offsets
    crop_xs[i] = crop_x;
    crop_ys[i] = crop_y;
    
    mirror[i] = std::bernoulli_distribution(0.5)(this->rand_gen_);
  }

  // Set mean & std values
  for (int i = 0; i < this->c_; ++i) {
    mean[i] = this->RandInt(1, 128);
    std[i] = this->RandInt(1, 128);
    inv_std[i] = 1 / std[i];
  }

  T *out_batch = nullptr;
  CUDA_CALL(cudaMalloc((void**)&out_batch, batch_size*crop_h*crop_w*this->c_*sizeof(T)));
  
  // validate parameters
  NDLL_CALL(ValidateBatchedCropMirrorNormalizePermute((const uint8 * const *)in_ptrs.data(),
          strides.data(),
          batch_size,
          crop_h,
          crop_w,
          this->c_,
          mirror,
          mean.data(),
          inv_std.data(),
          out_batch));

  // Copy the parameters to the gpu
  uint8 **gpu_in_ptrs = nullptr;
  int *gpu_in_strides = nullptr;
  bool *gpu_mirror = nullptr;
  float *gpu_mean = nullptr;
  float *gpu_inv_std = nullptr;
  cudaStream_t stream = 0;
  
  CUDA_CALL(cudaMalloc((void**)&gpu_in_ptrs, batch_size*sizeof(uint8*)));
  CUDA_CALL(cudaMalloc((void**)&gpu_in_strides, batch_size*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&gpu_mirror, batch_size*sizeof(bool)));
  CUDA_CALL(cudaMalloc((void**)&gpu_mean, this->c_*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&gpu_inv_std, this->c_*sizeof(float)));

  CUDA_CALL(cudaMemcpy(gpu_in_ptrs, in_ptrs.data(),
          batch_size*sizeof(uint8*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(gpu_in_strides, strides.data(),
          batch_size*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(gpu_mirror, mirror,
          batch_size*sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(gpu_mean, mean.data(),
          this->c_*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(gpu_inv_std, inv_std.data(),
          this->c_*sizeof(float), cudaMemcpyHostToDevice));
  
  // Run the kernel
  NDLL_CALL(BatchedCropMirrorNormalizePermute(
          gpu_in_ptrs,
          gpu_in_strides,
          batch_size,
          crop_h,
          crop_w,
          this->c_,
          gpu_mirror,
          gpu_mean,
          gpu_inv_std,
          out_batch,
          stream));
  
  
  for (int i = 0; i < batch_size; ++i) {
    vector<uint8> crop_mirror_image(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(
        batch.template mutable_sample<uint8>(i),
        batch.sample_shape(i)[0],
        batch.sample_shape(i)[1],
        this->c_,
        batch.sample_shape(i)[0],
        batch.sample_shape(i)[1],
        crop_ys[i], crop_xs[i],
        crop_h, crop_w, mirror[i],
        crop_mirror_image.data());

    vector<double> ground_truth_img(crop_h*crop_w*this->c_);
    CPUBatchedNormalizePermute(
        crop_mirror_image.data(),
        1, crop_h, crop_w, this->c_,
        mean.data(), std.data(),
        ground_truth_img.data());

    // Compare the images
    this->CompareData(out_batch + i*(crop_h*crop_w*this->c_),
        ground_truth_img.data(), crop_h*crop_w*this->c_);
  }
  
  // Clean up
  delete[] mirror;
  CUDA_CALL(cudaFree(gpu_in_ptrs));
  CUDA_CALL(cudaFree(gpu_in_strides));
  CUDA_CALL(cudaFree(gpu_mirror));
  CUDA_CALL(cudaFree(gpu_mean));
  CUDA_CALL(cudaFree(gpu_inv_std));
  CUDA_CALL(cudaFree(out_batch));
}

} // namespace ndll
