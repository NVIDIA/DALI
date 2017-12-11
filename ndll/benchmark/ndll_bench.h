// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_BENCHMARK_NDLL_BENCH_H_
#define NDLL_BENCHMARK_NDLL_BENCH_H_

#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "ndll/common.h"
#include "ndll/util/image.h"

namespace ndll {

// Note: this is setup for the binary to be executed from "build"
const string image_folder = "../ndll/benchmark/benchmark_images";  // NOLINT

class NDLLBenchmark : public benchmark::Fixture {
 public:
  NDLLBenchmark() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  }

  virtual ~NDLLBenchmark() {
    for (auto &ptr : jpegs_) {
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

  inline void MakeJPEGBatch(TensorList<CPUBackend> *tl, int n) {
    NDLL_ENFORCE(jpegs_.size() > 0, "jpegs must be loaded to create batches");
    vector<Dims> shape(n);
    for (int i = 0; i < n; ++i) {
      shape[i] = {jpeg_sizes_[i % jpegs_.size()]};
    }

    tl->template mutable_data<uint8>();
    tl->Resize(shape);

    for (int i = 0; i < n; ++i) {
      std::memcpy(tl->template mutable_tensor<uint8>(i),
          jpegs_[i % jpegs_.size()],
          jpeg_sizes_[i % jpegs_.size()]);
    }
  }

 protected:
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;
};

}  // namespace ndll

#endif  // NDLL_BENCHMARK_NDLL_BENCH_H_
