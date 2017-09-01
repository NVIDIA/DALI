#include "ndll/image/jpeg.h"

#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ndll/common.h"
#include "ndll/test/ndll_main_test.h"

namespace ndll {

namespace {
// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/image/testing_jpegs";
const string image_list = image_folder + "/image_list.txt";
} // namespace

// Our test "types"
struct RGB {};
struct Gray {};

// Fixture for jpeg decode testing. Templated
// to make googletest run our tests grayscale & rgb
template <typename color>
class JpegDecodeTest : public ::testing::Test {
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
  }

  void TearDown() {
    for (auto ptr : jpegs_) delete[] ptr;
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

  // Image is assumed to be stored HWC in memory
  void DumpToFile(uint8 *img, int h, int w, int c, int stride, string file_name) {
    CHECK_CUDA(cudaDeviceSynchronize());
    uint8 *tmp = new uint8[h*w*c];

    CHECK_CUDA(cudaMemcpy2D(tmp, w*c, img, stride, w*c, h, cudaMemcpyDefault));
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

  cv::Mat OpenCVDecode(uint8 *jpeg, int size) {
    cv::Mat jpeg_mat = cv::Mat(1, size, CV_8UC1, reinterpret_cast<unsigned char*>(jpeg));
    cv::Mat img;

    cv::Mat in = cv::imread(image_folder + "/cat.jpg");
    if (in.data == nullptr) {
      cout << "IMREADNULLPTR" << endl;
    }
    
    cv::imdecode(jpeg_mat, color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE, &img);

    if (img.data == nullptr) {
      cout << "NULLPTR" << endl;
    }
    return img;
  }
  
  void VerifyDecode(const uint8 *img, int h, int w, int img_id) {
    // Compare w/ opencv result
    cv::Mat ver;
    cv::Mat jpeg = cv::Mat(1, jpeg_sizes_[img_id], CV_8UC1, jpegs_[img_id]);

    cout << jpeg.rows << " " << jpeg.cols << endl;
    cout << (long long)jpeg.ptr() << endl;
    cout << (long long)jpegs_[img_id] << endl;
    ASSERT_TRUE(CheckIsJPEG(jpegs_[img_id], jpeg_sizes_[img_id]));
    
    int flag = color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    cv::imdecode(jpeg, flag, &ver);

    // Note: Need to do BGR->RGB here
    ASSERT_EQ(h, ver.rows);
    ASSERT_EQ(w, ver.cols);
    for (int i = 0; i < h*w*c_; ++i) {
      // Check the difference
      cout << int(ver.ptr()[i] - img[i]) << endl;
    }
  }


protected:
  bool color_;
  int c_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;
};

// Run RGB & grayscale tests
typedef ::testing::Types<RGB, Gray> Types;
TYPED_TEST_CASE(JpegDecodeTest, Types);

TYPED_TEST(JpegDecodeTest, DecodeOpenCV) {
  for (int img = 0; img < this->jpegs_.size(); ++img) {
    cv::Mat image = this->OpenCVDecode(this->jpegs_[img], this->jpeg_sizes_[img]);
  }
}

TYPED_TEST(JpegDecodeTest, DecodeJPEGHost) {
  // Decode all jpegs and see what they look like!
  vector<uint8> image;
  for (int img = 0; img < this->jpegs_.size(); ++img) {
    int h, w;

    CHECK_NDLL(GetJPEGImageDims(this->jpegs_[img],
            this->jpeg_sizes_[img], &h, &w));
    
    image.resize(h * w * this->c_);
    CHECK_NDLL(DecodeJPEGHost(this->jpegs_[img],
            this->jpeg_sizes_[img],
            this->color_, h, w,
            image.data()));

#ifdef DUMP_IMAGES
    cout << img << " " << this->jpeg_names_[img] << " " << this->jpeg_sizes_[img] << endl;
    cout << "dims: " << w << "x" << h << endl;
    this->DumpToFile(image.data(), h, w, this->c_, w*this->c_, std::to_string(img));
#endif // DUMP_IMAGES
    this->VerifyDecode(image.data(), h, w, img);
    break;
  }
}

} // namespace ndll
