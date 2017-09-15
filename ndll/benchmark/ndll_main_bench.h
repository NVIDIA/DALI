#ifndef NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_
#define NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_

#include <benchmark/benchmark.h>

namespace ndll {

class NDLLBenchmark : public benchmark::Fixture {
public:
    virtual void SetUp() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  }

  virtual void TearDown() {
    for (auto &ptr : jpegs_) delete[] ptr;
  }
    
  int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }

  // Load a batch of the input size from our jpegs. If we don't
  // have enough images, loop and add duplicates
  
protected:
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;
};

} // namespace ndll

#endif // NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_
