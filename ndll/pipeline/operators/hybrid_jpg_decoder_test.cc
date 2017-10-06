#include "ndll/pipeline/operators/hybrid_jpg_decoder.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/test/ndll_main_test.h"

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
} // namespace

template <typename ImgType>
class HybridDecoderTest : public NDLLTest {
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

  void VerifyDecode(const uint8 *img, int h, int w, int img_id) {
    // Load the image to host
    uint8 *host_img = new uint8[h*w*c_];
    CUDA_CALL(cudaMemcpy(host_img, img, h*w*c_, cudaMemcpyDefault));
      
    // Compare w/ opencv result
    cv::Mat ver;
    cv::Mat jpeg = cv::Mat(1, jpeg_sizes_[img_id], CV_8UC1, jpegs_[img_id]);

    ASSERT_TRUE(CheckIsJPEG(jpegs_[img_id], jpeg_sizes_[img_id]));    
    int flag = IsColor(img_type_) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    cv::imdecode(jpeg, flag, &ver);

    cv::Mat ver_img(h, w, IsColor(img_type_) ? CV_8UC3 : CV_8UC2);
    if (img_type_ == NDLL_RGB) {
      // Convert from BGR to RGB for verification
      cv::cvtColor(ver, ver_img, CV_BGR2RGB);
    } else {
      ver_img = ver;
    }
    
#ifndef NDEBUG
    // Dump the opencv image
    DumpHWCToFile(ver.ptr(), h, w, c_, "ver_" + std::to_string(img_id));
#endif
    
    ASSERT_EQ(h, ver_img.rows);
    ASSERT_EQ(w, ver_img.cols);
    vector<int> diff(h*w*c_, 0);
    for (int i = 0; i < h*w*c_; ++i) {
      diff[i] = abs(int(ver_img.ptr()[i] - host_img[i]));
    }

#ifndef NDEBUG
    // Dump the absolute differences
    DumpHWCToFile(diff.data(), h, w, c_, "diff_" + std::to_string(img_id));
#endif 
    
    // calculate the MSE
    float mean, std;
    MeanStdDev(diff, &mean, &std);

#ifndef NDEBUG
    cout << "num: " << diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif 

    // Note: We allow a slight deviation from the ground truth.
    // This value was picked fairly arbitrarily to let the test
    // pass for libjpeg turbo
    ASSERT_LT(mean, 2.f);
    ASSERT_LT(std, 3.f);
  }

  void MeanStdDev(const vector<int> &diff, float *mean, float *std) {
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
TYPED_TEST_CASE(HybridDecoderTest, Types);

TYPED_TEST(HybridDecoderTest, JPEGDecode) {
  int batch_size = this->jpegs_.size();
  int num_thread = 1;
  cudaStream_t main_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
 
  // Create the pipeline
  Pipeline<CPUBackend, GPUBackend> pipe(
      batch_size,
      num_thread,
      main_stream,
      0);

  shared_ptr<Batch<CPUBackend>> batch(CreateJPEGBatch<CPUBackend>(
          this->jpegs_, this->jpeg_sizes_, batch_size));
  shared_ptr<Batch<GPUBackend>> output_batch(new Batch<GPUBackend>);

  // Add the data reader
  BatchDataReader<CPUBackend> reader(batch);
  pipe.AddDataReader(reader);
  
  // Add a hybrid jpeg decoder
  shared_ptr<HybridJPEGDecodeChannel> decode_channel(new HybridJPEGDecodeChannel);
  HuffmanDecoder<CPUBackend> huffman_decoder(decode_channel);
  pipe.AddDecoder(huffman_decoder);

  DCTQuantInvOp<GPUBackend> idct_op(this->img_type_, decode_channel);
  pipe.AddForwardOp(idct_op);
    
  // Build and run the pipeline
  pipe.Build(output_batch);

  // Decode the images
  pipe.RunPrefetch();
  pipe.RunCopy();
  pipe.RunForward();
  CUDA_CALL(cudaDeviceSynchronize());

#ifndef NDEBUG
  DumpHWCImageBatchToFile<uint8>(*output_batch);
#endif
  
  // Verify the results
  for (int i = 0; i < batch_size; ++i) {
    this->VerifyDecode(output_batch->template datum<uint8>(i),
        output_batch->datum_shape(i)[0],
        output_batch->datum_shape(i)[1], i);
  }
}

} // namespace ndll
