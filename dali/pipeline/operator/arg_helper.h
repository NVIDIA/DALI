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

#ifndef DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
#define DALI_PIPELINE_OPERATOR_ARG_HELPER_H_

#include <dali/pipeline/operator/argument.h>
#include <dali/pipeline/operator/op_spec.h>
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
      tensor_vector_ = &ws->ArgumentInput(name);
    } else {
      value_ = spec.GetArgument<T>(name, ws);
    }
  }

  ArgValue &operator=(ArgValue &&) = default;
  inline ArgValue &operator=(const ArgValue &other) {
    gpu_.reset();
    value_  = other.value_;
    tensor_vector_ = other.tensor_vector_;
    return *this;
  }

  inline bool IsTensor() const { return tensor_vector_ != nullptr; }

  inline const T &operator[](Index index) {
    if (IsTensor()) {
#if DALI_DEBUG
      DALI_ENFORCE(index < static_cast<Index>(tensor_vector_->ntensor()));
#endif
      return (*tensor_vector_)[index].data<T>()[0];
    } else {
      return  value_;
    }
  }

  inline const TensorList<GPUBackend> *AsGPU(cudaStream_t stream) {
    DALI_ENFORCE(IsTensor());
    if (!gpu_) {
      gpu_.reset(new TensorList<GPUBackend>());
      gpu_->Copy(*tensor_vector_, stream);
    }
    return gpu_.get();
  }

 private:
  T value_;
  const TensorVector<CPUBackend> *tensor_vector_ = nullptr;
  std::unique_ptr<TensorList<GPUBackend>> gpu_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
