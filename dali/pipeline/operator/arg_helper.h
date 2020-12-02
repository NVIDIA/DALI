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

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/geom/mat.h"
#include "dali/core/geom/vec.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

template <typename T, int ndim = 0>
class ArgValue {
 public:
  using TLV = TensorListView<StorageCPU, const T, ndim>;
  using TV = TensorView<StorageCPU, const T, ndim>;

  ArgValue(std::string arg_name, const OpSpec &spec)
      : arg_name_(std::move(arg_name)) {
    has_arg_const_ = spec.HasArgument(arg_name_);
    has_arg_input_ = spec.HasTensorArgument(arg_name_);
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

  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               const TensorShape<ndim> &expected_shape) {
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      DALI_ENFORCE(is_uniform(view_.shape) && expected_shape == view_.shape[0],
        make_string("Expected uniform shape for argument \"", arg_name_,
                    "\" but got shape ", view_.shape));
    } else if (has_arg_const_) {
      TensorShape<ndim> shape{};
      if (ndim == 0) {
        data_ = {spec.GetArgument<T>(arg_name_)};
      } else {
        assert (ndim <= 1);
        data_ = spec.GetRepeatedArgument<T>(arg_name_);
        int64_t len = data_.size();
        int64_t expected_len = volume(expected_shape);
        if (len == 1 && expected_shape > 1) {
          data_.resize(expected_len, data_[0]);
        } else {
          DALI_ENFORCE(len == volume(expected_shape),
            make_string("Argument \"", arg_name_, "\" expected shape ", expected_shape,
                        " but got ", len, " values, which can't be interpreted as the expected shape."));
        }
      }
      view_ = constant_view(nsamples, data_.data(), shape);
    }
  }

  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               bool enforce_uniform = false) {
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      if (enforce_uniform) {
        DALI_ENFORCE(is_uniform(view_.shape),
          make_string("Expected uniform shape for argument \"", arg_name_,
                      "\" but got shape ", view_.shape));
      }
    } else {
      TensorShape<ndim> shape{};
      if (ndim == 0) {
        data_ = {spec.GetArgument<T>(arg_name_)};
      } else if (ndim == 1) {
        data_ = spec.GetRepeatedArgument<T>(arg_name_);
        int64_t len = data_.size();
        shape = std::array<int64_t, 1>{len};
      } else {
        // ndim > 1 but we don't have information about the expected shape.
        // An overload with ``expected_shape`` should have been used.
        assert(false);
      }
      view_ = constant_view(nsamples, data_.data(), shape);
    }
  }

  const std::string &name() const {
    return arg_name_;
  }

  const TLV& get() const {
    return view_;
  }

  TV operator[](size_t idx) const {
    assert(idx < static_cast<size_t>(size()));
    return view_[idx];
  }

  int size() const {
    return view_.size();
  }

 private:
  TLV constant_view(int nsamples, const T* sample, const TensorShape<ndim>& shape) {
    std::vector<const T*> ptrs(nsamples, sample);
    return TLV(std::move(ptrs), uniform_list_shape(nsamples, shape));
  }

  std::string arg_name_;
  std::vector<T> data_;
  TLV view_;

  bool has_arg_const_ = false;
  bool has_arg_input_ = false;
};

namespace detail {

template <int N>
vec<N> as_vec(TensorView<StorageCPU, const float, 1> view) {
  if (view.num_elements() == 1) {
    return vec<N>(view.data[0]);
  }
  assert(N == view.num_elements());
  return *reinterpret_cast<const vec<N>*>(view.data);
}

template <int N>
vec<N> as_vec(TensorView<StorageCPU, const float, DynamicDimensions> view) {
  return as_vec<N>(view.to_static<1>());
}

template <int N, int M>
mat<N, M> as_mat(TensorView<StorageCPU, const float, 2> view) {
  assert(N * M == view.num_elements());
  return *reinterpret_cast<const mat<N, M>*>(view.data);
}

template <int N, int M>
mat<N, M> as_mat(TensorView<StorageCPU, const float, DynamicDimensions> view) {
  return as_mat<N, M>(view.to_static<2>());
}

}  // namespace detail

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
