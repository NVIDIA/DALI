#ifndef NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_
#define NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_

#include <benchmark/benchmark.h>

namespace ndll {

class NDLLBenchmark : public benchmark::Fixture {
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
    CUDA_CALL(cudaDeviceSynchronize());
    T *tmp = new T[h*w*c];

    CUDA_CALL(cudaMemcpy2D(tmp, w*c*sizeof(T), img, stride*sizeof(T),
            w*c*sizeof(T), h, cudaMemcpyDefault));
    std::ofstream file(file_name + ".jpg.txt");
    ASSERT_TRUE(file.is_open());

    file << h << " " << w << " " << c << endl;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        for (int k = 0; k < c; ++k) {
          file << float(tmp[i*w*c + j*c + k]) << " ";
        }
      }
      file << endl;
    }
    delete[] tmp;
  }

  // Dump CHW image to file as HWC
  template <typename T>
  void DumpCHWToFile(T *img, int h, int w, int c, string file_name) {
    CUDA_CALL(cudaDeviceSynchronize());
    T *tmp = new T[h*w*c];
    
    CUDA_CALL(cudaMemcpy(tmp, img, h*w*c*sizeof(T), cudaMemcpyDefault));
    std::ofstream file(file_name + ".jpg.txt");
    ASSERT_TRUE(file.is_open());

    // write the image as HWC for our scripts
    file << h << " " << w << " " << c << endl;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        for (int k = 0; k < c; ++k) {
          file << float(tmp[k*h*w + i*w +j]) << " ";
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

#endif // NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_
