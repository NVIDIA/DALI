// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief ArgValue flags for acquire
 *
 */
enum ArgValueFlags : unsigned {
  ArgValue_EnforceUniform = 0b0001,  // Enforces a uniform shape
  ArgValue_AllowEmpty = 0b0010,      // Allows empty samples
  ArgValue_Default = 0               // Default behavior
};

static constexpr ArgValueFlags operator |(ArgValueFlags a, ArgValueFlags b) {
  return ArgValueFlags(unsigned(a) | unsigned(b));  //  NOLINT
}


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
    has_explicit_const_ = spec.HasArgument(arg_name_);
    has_arg_input_ = spec.HasTensorArgument(arg_name_);
    assert(!(has_explicit_const_ && has_arg_input_));

    ReadConstant(spec, false);  // not raising errors here
  }

  /**
   * @brief true if there is a value available (explicit or default)
   */
  bool HasValue() const {
    return has_arg_input_ || has_constant_value_;
  }

  /**
   * @brief true if there is a value explicitly provided (constant or argument input)
   */
  bool HasExplicitValue() const {
    return has_arg_input_ || has_explicit_const_;
  }

  /**
   * @brief true if there is a constant explicitly provided
   */
  bool HasExplicitConstant() const {
    return has_explicit_const_;
  }

  /**
   * @brief true if there is an argument input
   */
  bool HasArgumentInput() const {
    return has_arg_input_;
  }

  explicit operator bool() const {
    return HasValue();
  }

  /**
   * @brief Acquires argument data, enforcing that the shape of the data matches the
   *        expected shape or it is a scalar, which can also be broadcasted to the expected shape
   *
   * @param spec
   * @param ws
   * @param nsamples
   * @param expected_shape
   * @param flags bit flags controlling the behavior (see ArgValue)
   */
  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               const TensorListShape<ndim> &expected_shape,
               ArgValueFlags flags = ArgValue_Default) {
    assert(!(flags & ArgValue_EnforceUniform) || is_uniform(expected_shape));
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      if (flags & ArgValue_AllowEmpty) {
        for (int i = 0; i < nsamples; i++) {
          auto sh_span = view_.shape.tensor_shape_span(i);
          auto expected_sh_span = expected_shape.tensor_shape_span(i);
          DALI_ENFORCE(
              volume(sh_span) == 0 || sh_span == expected_sh_span,
              make_string("Unexpected shape for argument \"", arg_name_, "\". Expected ",
                          expected_shape, " or empty, but got ", view_.shape));
        }
      } else {
        DALI_ENFORCE(expected_shape == view_.shape,
          make_string("Unexpected shape for argument \"", arg_name_,
                      "\". Expected ", expected_shape, ", but got ", view_.shape));
      }
    } else {
      if (!has_constant_value_)
        ReadConstant(spec);  // just to raise the appropriate error

      int64_t expected_len = expected_shape.num_elements();
      if (orig_constant_sz_ == 1 && expected_len != 1) {
        // broadcast single values to whatever shape, including empty tensors
        data_.resize(std::max(expected_len, 1_i64), data_[0]);
        view_ = TLV(data_.data(), expected_shape);
      } else if (orig_constant_sz_ == 0 && (flags & ArgValue_AllowEmpty)) {
        view_ = constant_view(nsamples, data_.data(), TensorShape<ndim>{});
      } else {
        if (!is_uniform(expected_shape)) {
          DALI_FAIL(make_string("Can't interpret argument ", arg_name_,
                                ". Provided an constant argument with ", orig_constant_sz_,
                                " elements but the expected shape is not uniform."));
        }
        // at this point we know the expected shape is uniform
        auto expected_sample_sh = expected_shape[0];
        DALI_ENFORCE(orig_constant_sz_ == volume(expected_sample_sh),
              make_string("Argument \"", arg_name_, "\" expected shape ", expected_sample_sh,
                          " but got ", orig_constant_sz_,
                          " values, which can't be interpreted as the expected shape."));
        view_ = constant_view(nsamples, data_.data(), expected_sample_sh);
      }
    }
  }

  /**
   * @brief Acquires argument data, enforcing that the shape of the data matches the
   *        expected shape or it is a scalar, which can also be broadcasted to the expected shape
   *
   * @param spec
   * @param ws
   * @param nsamples
   * @param expected_shape
   * @param flags bit flags controlling the behavior (see ArgValue)
   */
  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               const TensorShape<ndim> &expected_shape,
               ArgValueFlags flags = ArgValue_Default) {
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      span<const int64_t> expected_sh_span(&expected_shape[0], expected_shape.size());
      if (flags & ArgValue_AllowEmpty) {
        for (int i = 0; i < nsamples; i++) {
          auto sh_span = view_.shape.tensor_shape_span(i);
          DALI_ENFORCE(
              volume(sh_span) == 0 || sh_span == expected_sh_span,
              make_string("Unexpected shape for argument \"", arg_name_, "\". Expected ",
                          expected_shape, " or empty, but got ", view_.shape));
        }
      } else {
        DALI_ENFORCE(
            is_uniform(view_.shape) && expected_sh_span == view_.shape.tensor_shape_span(0),
            make_string("Expected uniform shape for argument \"", arg_name_, "\" but got shape ",
                        view_.shape));
      }
    } else {
      if (!has_constant_value_)
        ReadConstant(spec);  // just to raise the appropriate error

      int64_t expected_len = volume(expected_shape);
      if (orig_constant_sz_ == 1 && expected_len != 1) {
        // broadcast single values to whatever shape, including empty tensors
        data_.resize(std::max(expected_len, 1_i64), data_[0]);
        view_ = constant_view(nsamples, data_.data(), expected_shape);
      } else if (orig_constant_sz_ == 0 && (flags & ArgValue_AllowEmpty)) {
        view_ = constant_view(nsamples, data_.data(), TensorShape<ndim>{});
      } else {
        DALI_ENFORCE(orig_constant_sz_ == volume(expected_shape),
              make_string("Argument \"", arg_name_, "\" expected shape ", expected_shape,
                          " but got ", orig_constant_sz_,
                          " values, which can't be interpreted as the expected shape."));
        view_ = constant_view(nsamples, data_.data(), expected_shape);
      }
    }
  }

  /**
   * @brief Acquires argument data, inferring the data shape in case of non-tensor arguments.
   *        The shape of scalar and 1D arguments is inferred by default. For 2 or more dimensions,
   *        a custom callable ``shape_from_size`` is expected.
   *
   * @tparam ShapeFromSizeFn
   * @param spec
   * @param ws
   * @param nsamples
   * @param flags bit flags controlling the behavior (see ArgValue)
   * @param shape_from_size
   */
  template <typename ShapeFromSizeFn = ArgShapeFromSize<ndim>>
  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples,
               ArgValueFlags flags = ArgValue_Default,
               ShapeFromSizeFn &&shape_from_size = {}) {
    if (has_arg_input_) {
      view_ = view<const T, ndim>(ws.ArgumentInput(arg_name_));
      if (flags & ArgValue_EnforceUniform) {
        DALI_ENFORCE(is_uniform(view_.shape),
          make_string("Expected uniform shape for argument \"", arg_name_,
                      "\" but got shape ", view_.shape));
      }
    } else {
      if (!has_constant_value_)
        ReadConstant(spec);  // just to raise the appropriate error

      auto sh = shape_from_size(orig_constant_sz_);
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
  TV operator[](int idx) const {
    assert(idx >= 0 && idx < size());
    return view_[idx];
  }

  /**
   * @brief true if the argument is empty or the particular sample
   *        has a 0-volume shape
   */
  bool IsEmpty(int idx) const {
    assert(idx >= 0 && idx < size());
    return volume(view_.shape.tensor_shape_span(idx)) == 0;
  }

  /**
   * @brief Number of samples
   */
  int size() const {
    return view_.size();
  }

 private:
  /**
   * @brief Read constant argument data
   */
  void ReadConstant(const OpSpec &spec, bool error_if_no_value = true) {
    data_.clear();
    if (ndim == 0) {
      data_.resize(1);
      has_constant_value_ = spec.TryGetArgument<T>(data_[0], arg_name_);
      if (!has_constant_value_) {
        data_.clear();
        if (error_if_no_value)  // call GetArgument and let it throw
          (void) spec.GetArgument<T>(arg_name_);
      }
    } else {
      has_constant_value_ = spec.TryGetRepeatedArgument(data_, arg_name_);
      if (!has_constant_value_) {
        if (error_if_no_value)  // call GetRepeatedArgument and let it throw
          (void) spec.GetRepeatedArgument<T>(arg_name_);
      }
    }
    orig_constant_sz_ = data_.size();
  }

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

  bool has_explicit_const_ = false;  // explicit constant
  bool has_arg_input_ = false;  // tensor input

  int64_t orig_constant_sz_ = -1;  // Original size of the constant value (-1 -> not-read)
  bool has_constant_value_ = false;  // has a constant (explicit or default) defined
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ARG_HELPER_H_
