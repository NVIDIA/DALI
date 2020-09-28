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

#ifndef DALI_OPERATORS_GEOMETRY_TRANSFORM_BASE_OP_H_
#define DALI_OPERATORS_GEOMETRY_TRANSFORM_BASE_OP_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/mat.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/operator/operator.h"

#define TRANSFORM_INPUT_TYPES (float)

namespace dali {

template <int... values>
using dims = std::integer_sequence<int, values...>;

template <typename T, int ndim>
using affine_mat_t = mat<ndim+1, ndim+1, T>;

/**
 * @brief Base CRTP class for affine transform generators.
 * The matrix definition comes from the actual TransformImpl implementation.
 * As with any CRTP-based system, any non-private method can be shadowed by the TransformImpl class.
 */
template <typename Backend, typename TransformImpl>
class TransformBaseOp : public Operator<Backend> {
 public:
  explicit TransformBaseOp(const OpSpec &spec)
      : Operator<Backend>(spec),
        spec_(spec),
        nsamples_(spec.GetArgument<int>("batch_size")) {
    matrix_data_.set_pinned(false);
    matrix_data_.set_type(TypeTable::GetTypeInfo(dtype_));
  }

  bool CanInferOutputs() const override { return true; }

  TransformImpl &This() noexcept { return static_cast<TransformImpl&>(*this); }
  const TransformImpl &This() const noexcept { return static_cast<const TransformImpl&>(*this); }

 protected:
  void CheckInputShape(const workspace_t<CPUBackend> &ws) {
    if (has_input_) {
      auto in_t_ndim = input_transform_ndim(ws);
      DALI_ENFORCE(in_t_ndim == ndim_,
        make_string("The input describes a ", in_t_ndim,
                    "D transform but other arguments suggest a ", ndim_, "D transform"));
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) override {
    has_input_ = ws.template NumInput() > 0;
    if (has_input_) {
      auto &input = ws.template InputRef<Backend>(0);
      const auto &shape = input.shape();
      DALI_ENFORCE(is_uniform(shape), "All matrices must have the same shape.");
      DALI_ENFORCE(input.type().id() == dtype_,
        make_string("Unexpected input data type. Expected ", dtype_, ", got: ", input.type().id()));

      DALI_ENFORCE(shape.sample_dim() == 2 &&
                   shape.size() > 0 &&
                   shape[0][1] == (shape[0][0] + 1),
        make_string(
          "The input, if provided, is expected to be a 2D tensor with dimensions "
          "(ndim, ndim+1) representing an affine transform. Got: ", shape));
      nsamples_ = shape.num_samples();
    }

    This().ProcessArgs(spec_, ws);
    CheckInputShape(ws);

    output_descs.resize(1);  // only one output
    output_descs[0].type = TypeTable::GetTypeInfo(dtype_);
    output_descs[0].shape = uniform_list_shape(nsamples_, {ndim_, ndim_+1});
    return true;
  }

  template <typename T>
  void RunImplTyped(workspace_t<Backend> &ws, dims<>) {
    DALI_FAIL(make_string("Unsupported number of dimensions ", ndim_));
  }

  template <typename T, int ndim, int... ndims>
  void RunImplTyped(workspace_t<Backend> &ws, dims<ndim, ndims...>) {
    if (ndim_ != ndim) {
      RunImplTyped<T>(ws, dims<ndims...>());
      return;
    }

    auto &out = ws.template OutputRef<Backend>(0);
    out.SetLayout({});  // no layout

    span<affine_mat_t<T, ndim>> matrices{
        reinterpret_cast<affine_mat_t<T, ndim> *>(matrix_data_.mutable_data<T>()), nsamples_};
    auto out_view = view<T>(out);
    if (has_input_) {
      auto &in = ws.template InputRef<Backend>(0);
      auto in_view = view<T>(in);
      for (int i = 0; i < nsamples_; i++) {
        ApplyTransform<T, ndim>(out_view[i].data, in_view[i].data, matrices[i]);
      }
    } else {
      for (int i = 0; i < nsamples_; i++) {
        ApplyTransform<T, ndim>(out_view[i].data, matrices[i]);
      }
    }
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, TRANSFORM_INPUT_TYPES, (
      using SupportedDims = typename TransformImpl::SupportedDims;
      RunImplTyped<T>(ws, SupportedDims());
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

  int input_transform_ndim(const workspace_t<Backend> &ws) const {
    assert(has_input_);
    auto &input = ws.template InputRef<Backend>(0);
    const auto& shape = input.shape();
    int ndims = shape[0][0];
    assert(shape[0][1] == ndims + 1);
    return ndims;
  }

 private:
  template <typename T, int ndim>
  void ApplyTransform(T *transform_out, const affine_mat_t<T, ndim> &M) {
    for (int i = 0, k = 0; i < ndim; i++) {
      for (int j = 0; j < ndim + 1; j++, k++) {
        transform_out[k] = M(i, j);
      }
    }
  }

  template <typename T, int ndim>
  void ApplyTransform(T *transform_out, const T* transform_in, const affine_mat_t<T, ndim> &M) {
    auto mat_in = affine_mat_t<T, ndim>::identity();
    for (int i = 0, k = 0; i < ndim; i++) {
      for (int j = 0; j < ndim + 1; j++, k++) {
        mat_in(i, j) = transform_in[k];
      }
    }

    // matrix multiplication
    auto mat_out = M * mat_in;

    for (int i = 0, k = 0; i < ndim; i++) {
      for (int j = 0; j < ndim + 1; j++, k++) {
        transform_out[k] = mat_out(i, j);
      }
    }
  }

 protected:
  const OpSpec& spec_;
  DALIDataType dtype_ = DALI_FLOAT;
  int ndim_ = -1;  // will be inferred from the arguments or the input
  int nsamples_ = -1;
  bool has_input_ = false;

  Tensor<CPUBackend> matrix_data_;
};



}  // namespace dali

#endif  // DALI_OPERATORS_GEOMETRY_TRANSFORM_BASE_OP_H_
