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
      arg_input_ = &ws.ArgumentInput(name_);
      gpu_dirty_ = true;
      is_set_ = true;
    } else {
      is_set_ = spec.HasArgument(name_);
      arg_input_ = nullptr;
      if (use_default || is_set_)
        value_ = spec.GetArgument<T>(name_);
    }
  }

  ArgValue &operator=(ArgValue &&) = default;
  inline ArgValue &operator=(const ArgValue &other) {
    name_ = other.name_;
    is_set_ = other.is_set_;
    gpu_dirty_ = true;
    gpu_.reset();
    value_  = other.value_;
    arg_input_ = other.arg_input_;
    return *this;
  }

  inline bool IsInput() const noexcept { return arg_input_ != nullptr; }
  inline bool IsSet() const noexcept { return is_set_; }

  template <int ndim = DynamicDimensions>
  inline const TensorView<StorageCPU, T, ndim> Tensor(int sample_index) const {
    ASSERT(IsInput());
    return view<T, ndim>(arg_input_[sample_index]);
  }

  inline const T &Value() const {
    return value_;
  }

  inline const string &Name() const noexcept {
    return name_;
  }

  inline const T &operator[](int sample_index) const {
    if (IsInput()) {
#if DALI_DEBUG
      DALI_ENFORCE(sample_index < static_cast<int>(arg_input_->ntensor()));
#endif
      return (*arg_input_)[sample_index].data<T>()[0];
    } else {
      return  value_;
    }
  }

  inline const TensorList<GPUBackend> &AsGPU(cudaStream_t stream) {
    DALI_ENFORCE(IsInput());
    if (!gpu_)
      gpu_.reset(new TensorList<GPUBackend>());
    if (gpu_dirty_) {
      gpu_->Copy(*arg_input_, stream);
      gpu_dirty_ = false;
    }
    return *gpu_;
  }

 private:
  string name_;
  T value_;
  bool is_set_ = false;
  bool gpu_dirty_ = true;
  const TensorVector<CPUBackend> *arg_input_ = nullptr;
  std::unique_ptr<TensorList<GPUBackend>> gpu_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
