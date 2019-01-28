// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_TENSOR_LIST_WRAPPER_H_
#define DALI_TEST_TENSOR_LIST_WRAPPER_H_

#include <gtest/gtest.h>
#include <string>
#include "dali/pipeline/data/tensor_list.h"

namespace dali {
namespace testing {

class TensorListWrapper {
 public:
  TensorListWrapper(const TensorList<CPUBackend> *tensor_list)  // NOLINT non-explicit ctor
          : cpu_(tensor_list) {}


  TensorListWrapper(const TensorList<GPUBackend> *tensor_list)  // NOLINT non-explicit ctor
          : gpu_(tensor_list) {}


  TensorListWrapper() = default;


  template<typename Backend>
  const TensorList<Backend> *get() const {
    FAIL() << "Backend type not supported. You may want to write your own specialization", nullptr;
  }


  constexpr bool has_cpu() const noexcept {
    return cpu_ != nullptr;
  }


  constexpr bool has_gpu() const noexcept {
    return gpu_ != nullptr;
  }


  std::string backend() const noexcept {
    if (cpu_) {
      return "cpu";
    } else if (gpu_) {
      return "gpu";
    } else {
      FAIL() << "Unknown backend. If you are here, something went terribly wrong", std::string();
    }
  }


  const TensorList<CPUBackend>& cpu() const noexcept {
    ASSERT_TRUE(cpu_) << "This wrapper doesn't contain TensorList<CPUBackend>", *cpu_;
    return *cpu_;
  }


  const TensorList<GPUBackend>& gpu() const noexcept {
    ASSERT_TRUE(gpu_) << "This wrapper doesn't contain TensorList<CPUBackend>", *gpu_;
    return *gpu_;
  }


  explicit constexpr operator bool() const noexcept {
    return cpu_ || gpu_;
  }


 private:
  const TensorList<GPUBackend> *gpu_ = nullptr;
  const TensorList<CPUBackend> *cpu_ = nullptr;
};


template<>
inline const TensorList<CPUBackend> *TensorListWrapper::get() const {
  ASSERT_TRUE(cpu_) << "This wrapper doesn't contain TensorList<CPUBackend>", nullptr;
  return cpu_;
}


template<>
inline const TensorList<GPUBackend> *TensorListWrapper::get() const {
  ASSERT_TRUE(gpu_) << "This wrapper doesn't contain TensorList<CPUBackend>", nullptr;
  return gpu_;
}


}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_TENSOR_LIST_WRAPPER_H_
