// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_OPERATOR_IMPL_UTILS_H_
#define DALI_PIPELINE_UTIL_OPERATOR_IMPL_UTILS_H_

#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

/**
 * @brief Utility interface to be used as a base for argument/type specific operator implementations
 */
template <typename Backend>
class OpImplBase {
 public:
  virtual ~OpImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const workspace_t<Backend> &ws) = 0;
  virtual void RunImpl(workspace_t<Backend> &ws) = 0;
};

template <>
class OpImplBase<CPUBackend> {
 public:
  virtual ~OpImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const workspace_t<CPUBackend> &ws) = 0;
  virtual void RunImpl(HostWorkspace &ws) {
    assert(false);
  }
  virtual void RunImpl(SampleWorkspace &ws) {
    assert(false);
  }
};


namespace detail {

template <typename T>
void ReadArgConstant(std::vector<T> &out, const string &arg_name, const OpSpec &spec) {
  out.resize(1);
  out[0] = spec.GetArgument<T>(arg_name);
}

template <typename T>
void ReadArgInput(std::vector<T> &out, const std::string &arg_name,
                  const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
  const auto& arg_in = ws.ArgumentInput(arg_name);
  auto arg_in_view = view<const float>(arg_in);
  DALI_ENFORCE(is_uniform(arg_in_view.shape) && volume(arg_in_view.shape[0]) == 1,
    make_string("``", arg_name, "`` must be a scalar or a 1D tensor with a single element"));
  if (arg_in_view.shape.sample_dim() > 0) {
    DALI_WARN_ONCE("Warning: \"", arg_name, "\""
                   " expected a scalar but received a 1D tensor with a single "
                   "element. Please use a scalar instead.");
  }

  auto nsamples = arg_in_view.size();
  out.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    out[i] = arg_in_view[i].data[0];
  }
}

template <typename T>
void ReadArgInput(std::vector<std::vector<T>> &out, const std::string &arg_name,
                  const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
  const auto& arg_in = ws.ArgumentInput(arg_name);
  auto arg_in_view = view<const float>(arg_in);
  DALI_ENFORCE(is_uniform(arg_in_view.shape),
    make_string("All samples in argument ``", arg_name, "`` should have the same shape"));
  DALI_ENFORCE(arg_in_view.shape.sample_dim() == 1,
    make_string("``", arg_name, "`` must be a 1D tensor"));

  auto nsamples = arg_in_view.size();
  out.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    auto ndim = arg_in_view[i].shape[0];
    out[i].resize(ndim);
    for (int d = 0; d < ndim; d++) {
      out[i][d] = arg_in_view[i].data[d];
    }
  }
}

}  // namespace detail


template <typename ArgType>
class ArgHelper {
 public:
  ArgHelper(const std::string &arg_name, const OpSpec &spec)
      : arg_name_(arg_name),
        has_arg_const_(spec.HasArgument(arg_name)),
        has_arg_input_(spec.HasTensorArgument(arg_name)) {
    assert(!(has_arg_const_ && has_arg_input_));
  }
  bool IsDefined() const {
    return has_arg_const_ || has_arg_input_;
  }
  bool IsConstant() const {
    return has_arg_const_;
  }
  bool IsArgInput() const {
    return has_arg_input_;
  }

  void Read(const OpSpec &spec, const workspace_t<CPUBackend> &ws, int repeat = 0) {
    if (has_arg_input_) {
      detail::ReadArgInput(data_, arg_name_, spec, ws);
    } else {
      detail::ReadArgConstant(data_, arg_name_, spec);
    }

    if (repeat > 1 && data_.size() == 1) {
      data_.resize(repeat, data_[0]);
    }
  }

  const std::string &name() const {
    return arg_name_;
  }

  ArgType &operator[](size_t idx) {
    return data_[idx];
  }
  const ArgType &operator[](size_t idx) const {
    return data_[idx];
  }

  void resize(size_t new_sz) {
    assert(new_sz >= 0);
    data_.resize(new_sz);
  }
  span<const ArgType> data() const {
    return make_cspan(data_);
  }
  span<ArgType> data() {
    return make_span(data_);
  }

  size_t size() const {
    return data_.size();
  }

 private:
  std::string arg_name_;
  std::vector<ArgType> data_;
  bool has_arg_const_ = false;
  bool has_arg_input_ = false;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_OPERATOR_IMPL_UTILS_H_
