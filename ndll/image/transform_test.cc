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
#include "ndll/test/ndll_main_test.h"
#include "ndll/test/type_conversion.h"

namespace ndll {

namespace {
// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/image/testing_jpegs";
const string image_list = image_folder + "/image_list.txt";
} // namespace

// Our test "types"
struct RGB {};
struct Gray {};

struct Dims { int h = 0, w = 0; };

// TODO(tgale): Move the methods used by common test fixtures
// into parent class for all NDLL tests, then derive from that
template <typename color>
class TransformTest : public ::testing::Test {
public:
  void SetUp() {
    if (std::is_same<color, RGB>::value) {
      color_ = true;
      c_ = 3;
    } else {
      color_ = false;
      c_ = 1;
    }
    LoadJPEGS();
    DecodeJPEGS();
  }

  virtual void TearDown() {
    for (auto ptr : jpegs_) delete[] ptr;
    for (auto ptr : images_) delete[] ptr;
  }
  
  void LoadJPEGS() {
    std::ifstream file(image_list);
    ASSERT_TRUE(file.is_open());
    
    string img;
    while(file >> img) {
      ASSERT_TRUE(img.size());
      jpeg_names_.push_back(image_folder + "/" + img);
    }

    for (auto img_name : jpeg_names_) {
      std::ifstream img_file(img_name);
      ASSERT_TRUE(img_file.is_open());

      img_file.seekg(0, std::ios::end);
      int img_size = (int)img_file.tellg();
      img_file.seekg(0, std::ios::beg);

      jpegs_.push_back(new uint8[img_size]);
      jpeg_sizes_.push_back(img_size);
      img_file.read(reinterpret_cast<char*>(jpegs_[jpegs_.size()-1]), img_size);
    }
  }

  void DecodeJPEGS() {
    images_.resize(jpegs_.size());
    image_dims_.resize(jpegs_.size());
    for (int i = 0; i < jpegs_.size(); ++i) {
      cv::Mat img;
      cv::Mat jpeg = cv::Mat(1, jpeg_sizes_[i], CV_8UC1, jpegs_[i]);
      
      ASSERT_TRUE(CheckIsJPEG(jpegs_[i], jpeg_sizes_[i]));
      int flag = color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
      cv::imdecode(jpeg, flag, &img);

      int h = img.rows;
      int w = img.cols;
      cv::Mat out_img(h, w, color_ ? CV_8UC3 : CV_8UC2);
      if (color_) {
        // Convert from BGR to RGB for verification
        cv::cvtColor(img, out_img, CV_BGR2RGB);
      } else {
        out_img = img;
      }
    
      // Copy the decoded image out & save the dims
      ASSERT_TRUE(out_img.isContinuous());
      images_[i] = new uint8[h*w*c_];
      std::memcpy(images_[i], out_img.ptr(), h*w*c_);

      image_dims_[i].h = h;
      image_dims_[i].w = w;
    }
  }
  
  // Image is assumed to be stored HWC in memory. Data-type is cast to unsigned int before
  // being written to file.
  template <typename T>
  void DumpToFile(T *img, int h, int w, int c, int stride, string file_name) {
    CHECK_CUDA(cudaDeviceSynchronize());
    T *tmp = new T[h*w*c];

    CHECK_CUDA(cudaMemcpy2D(tmp, w*c*sizeof(T), img, stride*sizeof(T),
            w*c*sizeof(T), h, cudaMemcpyDefault));
    std::ofstream file(file_name + ".jpg.txt");
    ASSERT_TRUE(file.is_open());

    file << h << " " << w << " " << c << endl;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        for (int k = 0; k < c; ++k) {
          file << unsigned(tmp[i*w*c + j*c + k]) << " ";
        }
      }
      file << endl;
    }
    delete[] tmp;
  }

  // Loads image dumped by "DumpToFile()"
  void LoadFromFile(string file_name, uint8 **image, int *h, int *w, int *c) {
    std::ifstream file(file_name + ".jpg.txt");
    ASSERT_TRUE(file.is_open());

    file >> *h;
    file >> *w;
    file >> *c;

    // lol at this multiplication
    int size = (*h)*(*w)*(*c);
    *image = new uint8[size];
    int tmp = 0;
    for (int i = 0; i < size; ++i) {
      file >> tmp;
      (*image)[i] = (uint8)tmp;
    }
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
    CHECK_CUDA(cudaMemcpy2D(crop_img.ptr(), crop_w*c, crop_ptr,
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

  void VerifyImage(uint8 *img, uint8 *ground_truth, int n) {
    vector<int> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(int(img[i] - ground_truth[i]));
    }
    double mean, std;
    MeanStdDev(abs_diff, &mean, &std);

#ifdef DUMP_IMAGES
    cout << "num: " << abs_diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif // DUMP_IMAGES
    
    // Note: We allow a slight deviation from the ground truth.
    // This value was picked fairly arbitrarily to let the test
    // pass for libjpeg turbo
    ASSERT_LT(mean, 2.0);
    ASSERT_LT(std, 3.0);
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
  
protected:
  bool color_;
  int c_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;

  vector<uint8*> images_;
  vector<Dims> image_dims_;

  uint8* image_batch_;
  int n_, h_, w_;
};

// Run RGB & grayscale tests
typedef ::testing::Types<RGB, Gray> Types;
TYPED_TEST_CASE(TransformTest, Types);

// For functions that are the output of a pipeline,
// we want to test with all the output types
template <typename Types>
// class OutputTransformTest : public TransformTest<typename Types::test_color> {
class OutputTransformTest : public TransformTest<typename Types::test_color> {
public:
  typedef typename Types::test_color color;
  // Comparison for other types. We use double for the ground truth.
  // Input data is assumed to be on the GPU
  template <typename T>
  void CompareData(T *data, double *ground_truth, int n) {
    // Conver the input data to double
    double *tmp_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&tmp_gpu, sizeof(double)*n));
    Convert(data, n, tmp_gpu);
    
    vector<double> tmp(n, 0);
    CHECK_CUDA(cudaMemcpy(tmp.data(), tmp_gpu, n*sizeof(double),
            cudaMemcpyDeviceToHost));

    vector<double> abs_diff(n, 0);
    for (int i = 0; i < n; ++i) {
      abs_diff[i] = abs(tmp[i] - ground_truth[i]);
    }
    double mean, std;
    TransformTest<color>::MeanStdDev(abs_diff, &mean, &std);

#ifdef DUMP_IMAGES
    cout << "num: " << abs_diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif // DUMP_IMAGES
    
    ASSERT_LT(mean, 0.000001);
    ASSERT_LT(std, 0.000001);
    CHECK_CUDA(cudaFree(tmp_gpu));
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
                         OutputTestTypes<Gray, float16>,
                         OutputTestTypes<Gray, float>,
                         OutputTestTypes<Gray, double>> OutputTypes;
TYPED_TEST_CASE(OutputTransformTest, OutputTypes);

TYPED_TEST(TransformTest, TestResizeCrop) {
  std::mt19937 rand_gen(time(nullptr));
  vector<uint8> out_img, ver_img;
  for (int i = 0; i < this->images_.size(); ++i) {
    // Generate random resize params
    int rsz_h = std::uniform_int_distribution<>(32, 512)(rand_gen);
    int rsz_w = std::uniform_int_distribution<>(32, 512)(rand_gen);
    
    // Generate random crop params
    int crop_h = std::uniform_int_distribution<>(32, rsz_h)(rand_gen);
    int crop_w = std::uniform_int_distribution<>(32, rsz_w)(rand_gen);
    int crop_y = std::uniform_int_distribution<>(0, rsz_h - crop_h)(rand_gen);
    int crop_x = std::uniform_int_distribution<>(0, rsz_w - crop_w)(rand_gen);
    
    // Select whether to mirror
    bool mirror = false;

    out_img.resize(crop_h*crop_w*this->c_);
    CHECK_NDLL(ResizeCropMirrorHost(this->images_[i], this->image_dims_[i].h,
            this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y,
            crop_x, crop_h, crop_w, mirror, out_img.data()));

    // Verify the output
    ver_img.resize(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(this->images_[i], this->image_dims_[i].h,
        this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y, crop_x,
        crop_h, crop_w, mirror, ver_img.data());

    // TODO(tgale): In every test where we want extensive debug info we
    // have to use these ifdefs. We should add some logging functionality
    // that takes this into account so we don't have to dirty the code
    // up everywhere.
#ifdef DUMP_IMAGES
    cout << i << " " << this->jpeg_names_[i] << endl;
    cout << "dims: " << this->image_dims_[i].h << "x" << this->image_dims_[i].w << endl;
    cout << "rsz: " << rsz_h << "x" << rsz_w << endl;
    cout << "crop: " << crop_h << "x" << crop_w << endl;
    cout << "mirror: " << mirror << endl;
    this->DumpToFile(out_img.data(), crop_h, crop_w,
        this->c_, crop_w*this->c_, std::to_string(i));
    this->DumpToFile(ver_img.data(), crop_h, crop_w,
            this->c_, crop_w*this->c_, "ver_" + std::to_string(i));
#endif // DUMP_IMAGES
    this->VerifyImage(out_img.data(), ver_img.data(), out_img.size());
  }
}

TYPED_TEST(TransformTest, TestResizeCropMirror) {
  std::mt19937 rand_gen(time(nullptr));
  vector<uint8> out_img, ver_img;
  for (int i = 0; i < this->images_.size(); ++i) {
    // Generate random resize params
    int rsz_h = std::uniform_int_distribution<>(32, 512)(rand_gen);
    int rsz_w = std::uniform_int_distribution<>(32, 512)(rand_gen);
    
    // Generate random crop params
    int crop_h = std::uniform_int_distribution<>(32, rsz_h)(rand_gen);
    int crop_w = std::uniform_int_distribution<>(32, rsz_w)(rand_gen);
    int crop_y = std::uniform_int_distribution<>(0, rsz_h - crop_h)(rand_gen);
    int crop_x = std::uniform_int_distribution<>(0, rsz_w - crop_w)(rand_gen);

    // Select whether to mirror
    bool mirror = true;

    out_img.resize(crop_h*crop_w*this->c_);
    CHECK_NDLL(ResizeCropMirrorHost(this->images_[i], this->image_dims_[i].h,
            this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y,
            crop_x, crop_h, crop_w, mirror, out_img.data()));

    // Verify the output
    ver_img.resize(crop_h*crop_w*this->c_);
    this->OpenCVResizeCropMirror(this->images_[i], this->image_dims_[i].h,
        this->image_dims_[i].w, this->c_, rsz_h, rsz_w, crop_y, crop_x,
        crop_h, crop_w, mirror, ver_img.data());
    
#ifdef DUMP_IMAGES
    cout << i << " " << this->jpeg_names_[i] << endl;
    cout << "dims: " << this->image_dims_[i].h << "x" << this->image_dims_[i].w << endl;
    cout << "rsz: " << rsz_h << "x" << rsz_w << endl;
    cout << "crop: " << crop_h << "x" << crop_w << endl;
    cout << "mirror: " << mirror << endl;
    this->DumpToFile(out_img.data(), crop_h, crop_w,
        this->c_, crop_w*this->c_, std::to_string(i));
    this->DumpToFile(ver_img.data(), crop_h, crop_w,
            this->c_, crop_w*this->c_, "ver_" + std::to_string(i));
#endif // DUMP_IMAGES
    this->VerifyImage(out_img.data(), ver_img.data(), out_img.size());
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
  
  std::mt19937 rand_gen(time(nullptr));
  int n = std::uniform_int_distribution<>(4, this->jpegs_.size())(rand_gen);
  int h = std::uniform_int_distribution<>(32, 512)(rand_gen);
  int w = std::uniform_int_distribution<>(32, 512)(rand_gen);
  vector<uint8> batch(n*h*w*this->c_, 0);
  this->MakeImageBatch(n, h, w, batch.data());

  // Set up the mean & std dev
  vector<float> vals(this->c_*2, 128);
  float *mean = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&mean, sizeof(float)*2*this->c_));
  CHECK_CUDA(cudaMemcpy(mean, vals.data(), sizeof(float)*2*this->c_, cudaMemcpyHostToDevice));
  float *std = mean + this->c_;
    
  // Move the batch to GPU
  uint8 *batch_gpu = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&batch_gpu, n*h*w*this->c_));
  CHECK_CUDA(cudaMemcpy(batch_gpu, batch.data(), n*h*w*this->c_, cudaMemcpyHostToDevice));

  // Run the method
  T *output_batch = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&output_batch, n*h*w*this->c_*sizeof(T)));
  
  CHECK_NDLL(BatchedNormalizePermute(batch_gpu, n, h, w, this->c_,
          mean, std, output_batch, 0));

  vector<double> output_batch_ver(n*h*w*this->c_, 0);
  CPUBatchedNormalizePermute(batch.data(), n, h, w, this->c_,
      vals.data(), vals.data()+this->c_, output_batch_ver.data());

  this->CompareData(output_batch, output_batch_ver.data(), n*h*w*this->c_);
  
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(mean));
  CHECK_CUDA(cudaFree(batch_gpu));
  CHECK_CUDA(cudaFree(output_batch));
}

} // namespace ndll
