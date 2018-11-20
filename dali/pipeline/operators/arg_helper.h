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

#ifndef DALI_PIPELINE_OPERATORS_ARG_HELPER_H_
#define DALI_PIPELINE_OPERATORS_ARG_HELPER_H_

#include <dali/pipeline/operators/argument.h>
#include <dali/pipeline/operators/op_spec.h>
#include <dali/pipeline/data/tensor.h>
#include <memory>
#include <string>

namespace dali {

template <typename T>
class ArgValue {
 public:
  ArgValue() = default;
  ArgValue(ArgValue &&) = default;
  inline ArgValue(const ArgValue &other) { *this = other; }
  inline ArgValue(const std::string &name, const OpSpec &spec, ArgumentWorkspace *ws) {
    if (spec.HasTensorArgument(name)) {
      tensor_ = &ws->ArgumentInput(name);
      data_ = tensor_->data<T>();
    } else {
      value_ = spec.GetArgument<T>(name, ws);
    }
  }

  ArgValue &operator=(ArgValue &&) = default;
  inline ArgValue &operator=(const ArgValue &other) {
    gpu_.reset();
    value_  = other.value_;
    tensor_ = other.tensor_;
    data_   = other.data_;
    return *this;
  }

  inline bool IsTensor() const { return data_ != nullptr; }

  inline const T &operator[](Index index) {
    if (IsTensor()) {
#if DALI_DEBUG
      DALI_ENFORCE(index < tensor_->size());
#endif
      return data_[index];
    } else {
      return  value_;
    }
  }

  inline const Tensor<GPUBackend> *AsGPU(cudaStream_t stream) {
    DALI_ENFORCE(IsTensor());
    if (!gpu_) {
      gpu_.reset(new Tensor<GPUBackend>());
      gpu_->Copy(*tensor_, stream);
    }
    return gpu_.get();
  }

 private:
  T value_;
  const T *data_ = nullptr;
  const Tensor<CPUBackend> *tensor_ = nullptr;
  std::unique_ptr<Tensor<GPUBackend>> gpu_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_ARG_HELPER_H_
