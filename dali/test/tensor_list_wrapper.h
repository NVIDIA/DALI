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
#include <memory>
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

  template<typename Backend>
  constexpr bool has() const {
    FAIL() << "Backend type not supported. You may want to write your own specialization", false;
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
    ASSERT_TRUE(gpu_) << "This wrapper doesn't contain TensorList<GPUBackend>", *gpu_;
    return *gpu_;
  }

  template <typename DestinationBackend>
  std::unique_ptr<TensorList<DestinationBackend>> CopyTo() const {
    std::unique_ptr<TensorList<DestinationBackend>> result(new TensorList<DestinationBackend>());
    ASSERT_NE(has_cpu(), has_gpu())
        << "Should contain TensorList from exactly one backend", nullptr;
    if (has_cpu()) {
      result->Copy(*cpu_, 0);
    } else {
      result->Copy(*gpu_, 0);
    }
    CUDA_CALL(cudaStreamSynchronize(0));
    return result;
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
  ASSERT_TRUE(gpu_) << "This wrapper doesn't contain TensorList<GPUBackend>", nullptr;
  return gpu_;
}

template <>
inline bool TensorListWrapper::has<CPUBackend>() const {
  return has_cpu();
}

template <>
inline bool TensorListWrapper::has<GPUBackend>() const {
  return has_gpu();
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_TENSOR_LIST_WRAPPER_H_
