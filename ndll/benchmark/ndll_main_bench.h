#ifndef NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_
#define NDLL_BENCHMARK_NDLL_MAIN_BENCH_H_

#include <random>

#include <benchmark/benchmark.h>

#include "ndll/common.h"
#include "ndll/util/image.h"

namespace ndll {

// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/image/testing_jpegs";

class NDLLBenchmark : public benchmark::Fixture {
public:
  NDLLBenchmark() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  }

  virtual ~NDLLBenchmark() {
    for (auto &ptr : jpegs_) {
      cout << "deleting: " << (long long)ptr << endl;
      delete[] ptr;
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
