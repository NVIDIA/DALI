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
#include "dali/core/geom/geom_utils.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

/**
 * @brief Infers the data shape from a flat size
 * @{
 */
template <int ndim>
struct ArgShapeFromSize {
  TensorShape<ndim> operator()(int64_t size) const {
    throw std::logic_error(make_string("Cannot infer a ", ndim, "D shape from a flat size."));
  }
};

template <>
struct ArgShapeFromSize<0> {
  TensorShape<0> operator()(int64_t size) const {
    DALI_ENFORCE(size == 1, make_string("Expected a scalar argument, got ", size, " values"));
    return {};
  }
};

template <>
struct ArgShapeFromSize<1> {
  TensorShape<1> operator()(int64_t size) const {
    return { size };
  }
};

/**
 * @}
 */

/**
 * @brief Helper to access operator argument data, regardless of whether the data was provided
 * as a build-time constant or a tensor input.
 * 
 * There are two ways to acquire arguments:
 * - Explicitly providing the expected shape of the data
 * - Inferring the shape of the data from a flat size, either by a default or with a custom callable object.
 * 
 * @tparam T    Underlying data type.
 * @tparam ndim Number of dimensions of the argument. By default, scalar.
 *              Higher dimensions are expected for arguments that are passed as vector<T>.
 */
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

  /**
   * @brief true if the argument was provided explicitly
   */
  bool IsDefined() const {
    return has_arg_const_ || has_arg_input_;
  }

  /**
   * @brief true if the argument is a build-time constant
   */
  bool IsConstant() const {
    return has_arg_const_;
  }

  /**
   * @brief true if the argument is a tensor input
   */
  bool IsArgInput() const {
    return has_arg_input_;
  }

  /**
   * @brief Acquires argument data, enforcing that the shape of the data matches the
   *        expected shape or it is a scalar, which can also be broadcasted to the expected shape
   */
  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               const TensorShape<ndim> &expected_shape) {
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      DALI_ENFORCE(is_uniform(view_.shape) && expected_shape == view_.shape[0],
        make_string("Expected uniform shape for argument \"", arg_name_,
                    "\" but got shape ", view_.shape));
    } else {
      if (ndim == 0) {
        data_.resize(1);
        if (!spec.TryGetArgument<T>(data_[0], arg_name_)) {
          // something went bad - call GetArgument and let it throw
          (void) spec.GetArgument<T>(arg_name_);
        }
      } else {
        if (!spec.TryGetRepeatedArgument<T>(data_, arg_name_)) {
          // something went bad - call GetRepeatedArgument and let it throw
          (void) spec.GetRepeatedArgument<T>(arg_name_);
        }
        int64_t len = data_.size();
        int64_t expected_len = volume(expected_shape);
        if (len == 1 && expected_len > 1) {
          data_.resize(expected_len, data_[0]);
        } else {
          DALI_ENFORCE(len == volume(expected_shape),
                       make_string("Argument \"", arg_name_, "\" expected shape ", expected_shape,
                                   " but got ", len,
                                   " values, which can't be interpreted as the expected shape."));
        }
      }
      view_ = constant_view(nsamples, data_.data(), expected_shape);
    }
  }

  /**
   * @brief Acquires argument data, inferring the data shape in case of non-tensor arguments.
   *        The shape of scalar and 1D arguments is inferred by default. For 2 or more dimensions,
   *        a custom callable ``shape_from_size`` is expected.
   */
  template <typename ShapeFromSizeFn = ArgShapeFromSize<ndim>>
  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               bool enforce_uniform = false, ShapeFromSizeFn &&shape_from_size = {}) {
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      if (enforce_uniform) {
        DALI_ENFORCE(is_uniform(view_.shape),
          make_string("Expected uniform shape for argument \"", arg_name_,
                      "\" but got shape ", view_.shape));
      }
    } else {
      if (ndim == 0) {
        data_.resize(1);
        if (!spec.TryGetArgument<T>(data_[0], arg_name_)) {
          // something went bad - call GetArgument and let it throw
          (void) spec.GetArgument<T>(arg_name_);
        }
      } else {
        if (!spec.TryGetRepeatedArgument<T>(data_, arg_name_)) {
          // something went bad - call GetRepeatedArgument and let it throw
          (void) spec.GetRepeatedArgument<T>(arg_name_);
        }
      }
      auto sh = shape_from_size(static_cast<int64_t>(data_.size()));
      view_ = constant_view(nsamples, data_.data(), std::move(sh));
    }
  }

  /**
   * @brief Argument name
   */
  const std::string &name() const {
    return arg_name_;
  }

  /**
   * @brief Get a tensor list view to the argument data
   */
  const TLV& get() const {
    return view_;
  }

  /**
   * @brief Get a tensor view to the argument data for a given sample index
   */
  TV operator[](size_t idx) const {
    assert(idx < static_cast<size_t>(size()));
    return view_[idx];
  }

  /**
   * @brief Number of samples
   */
  int size() const {
    return view_.size();
  }

 private:
  /**
   * @brief Creates a TensorListView out of a constant arguments by assigning the same
   *        data pointer to all the samples. This way, the user code can be shared regardless
   *        of whether the source of the data was a build time constant or an argument input.
   */
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

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
