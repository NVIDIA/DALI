// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_BENCHMARK_DALI_BENCH_H_
#define DALI_BENCHMARK_DALI_BENCH_H_

#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "dali/common.h"
#include "dali/util/image.h"

namespace dali {

// Note: this is setup for the binary to be executed from "build"
const string image_folder = "/data/dali/benchmark/benchmark_images";  // NOLINT

class DALIBenchmark : public benchmark::Fixture {
 public:
  DALIBenchmark() {
    rand_gen_.seed(time(nullptr));
    jpeg_names_ = ImageList(image_folder, {".jpg"});
    LoadImages(jpeg_names_, &jpegs_);
  }

  ~DALIBenchmark() override = default;

  int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }

  inline void MakeJPEGBatch(TensorList<CPUBackend> *tl, int n) {
    const auto nImgs = jpegs_.nImages();
    DALI_ENFORCE(nImgs > 0, "jpegs must be loaded to create batches");
    vector<Dims> shape(n);
    for (int i = 0; i < n; ++i) {
      shape[i] = {jpegs_.sizes_[i % nImgs]};
    }

    tl->template mutable_data<uint8>();
    tl->Resize(shape);

    for (int i = 0; i < n; ++i) {
      std::memcpy(tl->template mutable_tensor<uint8>(i),
          jpegs_.data_[i % nImgs], jpegs_.sizes_[i % nImgs]);
      tl->SetSourceInfo(i, jpeg_names_[i % nImgs] + "_" + std::to_string(i));
    }
  }

 protected:
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  ImgSetDescr jpegs_;
};

}  // namespace dali

#endif  // DALI_BENCHMARK_DALI_BENCH_H_
