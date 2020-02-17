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
#include <dali/pipeline/data/views.h>
#include <memory>
#include <string>

namespace dali {

template <typename T>
class ArgValue {
 public:
  ArgValue() = default;
  ArgValue(ArgValue &&) = default;
  inline ArgValue(const ArgValue &other) { *this = other; }
  explicit inline ArgValue(const std::string &name) : name_(name) {
  }

  void Update(const OpSpec &spec, ArgumentWorkspace &ws, bool use_default = true) {
    if (spec.HasTensorArgument(name_)) {
      tensor_vector_ = &ws.ArgumentInput(name_);
      gpu_dirty_ = true;
    } else {
      is_set_ = spec.HasArgument(name_);
      if (use_default || is_set_)
        value_ = spec.GetArgument<T>(name_);
    }
  }

  ArgValue &operator=(ArgValue &&) = default;
  inline ArgValue &operator=(const ArgValue &other) {
    gpu_.reset();
    value_  = other.value_;
    tensor_vector_ = other.tensor_vector_;
    return *this;
  }

  inline bool IsInput() const { return tensor_vector_ != nullptr; }
  inline bool IsSet() const { return is_set_; }

  template <int ndim = DynamicDimensions>
  inline const TensorView<StorageCPU, T, ndim> Tensor(int sample_index) {
    ASSERT(IsInput());
    return view<T, ndim>(tensor_vector_[sample_index]);
  }

  inline const T &Value() {
    return value_;
  }

  inline const T &operator[](int sample_index) {
    if (IsInput()) {
#if DALI_DEBUG
      DALI_ENFORCE(sample_index < static_cast<int>(tensor_vector_->ntensor()));
#endif
      return (*tensor_vector_)[sample_index].data<T>()[0];
    } else {
      return  value_;
    }
  }

  inline const TensorList<GPUBackend> &AsGPU(cudaStream_t stream) {
    DALI_ENFORCE(IsInput());
    if (!gpu_)
      gpu_.reset(new TensorList<GPUBackend>());
    if (gpu_dirty_) {
      gpu_->Copy(*tensor_vector_, stream);
      gpu_dirty_ = false;
    }
    return *gpu_;
  }

 private:
  string name_;
  bool is_input_;
  bool is_set_;
  T value_;
  const TensorVector<CPUBackend> *tensor_vector_ = nullptr;
  std::unique_ptr<TensorList<GPUBackend>> gpu_;
  bool gpu_dirty_ = true;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
