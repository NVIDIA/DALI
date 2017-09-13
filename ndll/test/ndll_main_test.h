#ifndef NDLL_TEST_NDLL_MAIN_TEST_H_
#define NDLL_TEST_NDLL_MAIN_TEST_H_

#include <cassert>
#include <fstream>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

// Error checking for the test suite
#define TEST_NDLL(error)                        \
  do {                                          \
    assert(!error);                             \
  } while (0)

#define TEST_CUDA(code)                             \
  do {                                              \
    cudaError_t status = code;                      \
    if (status != cudaSuccess) {                    \
      string file = __FILE__;                       \
      string line = std::to_string(__LINE__);       \
      string error = "[" + file + ":" + line +      \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(status) + "\"";          \
      cout << error << endl;                        \
      assert(false);                                \
    }                                               \
  } while (0)


// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/image/testing_jpegs";
const string image_list = image_folder + "/image_list.txt";

// Main testing fixture to provide common functionality across tests
class NDLLTest : public ::testing::Test {
public:
  virtual void SetUp() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS();
  }

  virtual void TearDown() {

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

  // Image is assumed to be stored HWC in memory. Data-type is cast to unsigned int before
  // being written to file.
  template <typename T>
  void DumpToFile(T *img, int h, int w, int c, int stride, string file_name) {
    TEST_CUDA(cudaDeviceSynchronize());
    T *tmp = new T[h*w*c];

    TEST_CUDA(cudaMemcpy2D(tmp, w*c*sizeof(T), img, stride*sizeof(T),
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
  
  int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }
  
protected:
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;
}; 
} // namespace ndll

#endif // NDLL_TEST_NDLL_MAIN_TEST_H_
