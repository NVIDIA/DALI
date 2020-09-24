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

#define TRANSFORM_FACTORY_INPUT_TYPES (float)
#define TRANSFORM_FACTORY_DIMS (2, 3)

namespace dali {

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
        batch_size_(spec.GetArgument<int>("batch_size")) {
  }

  bool CanInferOutputs() const override { return true; }

  TransformImpl &This() noexcept { return static_cast<TransformImpl&>(*this); }
  const TransformImpl &This() const noexcept { return static_cast<const TransformImpl&>(*this); }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) override {
    int nsamples = batch_size_;
    has_input_ = ws.template NumInput() > 0;
    if (has_input_) {
      auto &input = ws.template InputRef<Backend>(0);
      nsamples = input.shape().num_samples();
    }

    ndim_ = This().ndim(ws);

    output_descs.resize(1);  // only one output
    output_descs[0].type = TypeTable::GetTypeInfo(dtype_);
    output_descs[0].shape = uniform_list_shape(nsamples, {ndim_, ndim_+1});
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    // auto &in = ws.template InputRef<Backend>(0);
    // DALIDataType in_type = in.type().id();
    auto &out = ws.template OutputRef<Backend>(0);
    out.SetLayout("");  // TODO(janton): Decide what layout we want for transforms
    int nsamples = batch_size_;

    TYPE_SWITCH(dtype_, type2id, T, TRANSFORM_FACTORY_INPUT_TYPES, (
      VALUE_SWITCH(ndim_, ndim, TRANSFORM_FACTORY_DIMS, (
        auto M = mat<ndim+1, ndim+1, T>::identity();
        This().template DefineTransform<T, ndim>(M, ws);
        auto out_view = view<T>(out);
        if (has_input_) {
          auto &in = ws.template InputRef<Backend>(0);
          DALI_ENFORCE(in.type().id() == dtype_,
            make_string("Unexpected input data type. Expected float, got: ", in.type().id()));
          nsamples = in.shape().num_samples();
          auto in_view = view<T>(in);
          for (int i = 0; i < nsamples; i++) {
            ApplyTransform<T, ndim>(out_view[i].data, in_view[i].data, M);
          }
        } else {
          for (int i = 0; i < nsamples; i++) {
            ApplyTransform<T, ndim>(out_view[i].data, M);
          }
        }
      ), DALI_FAIL(make_string("Unsupported number of dimensions ", ndim_)));  // NOLINT
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

  /**
   * @brief Infers number of dimension.
   * Transform implementations are expected to override (shadow) this
   */
  int ndim(const workspace_t<Backend> &ws) const {
    if (has_input_) {
      auto &input = ws.template InputRef<Backend>(0);
      // TODO(janton): infer dimensionality here ?
    }
    return 2;  // default
  }

 private:
  template <typename T, int ndim>
  void ApplyTransform(T *transform_out, const mat<ndim+1, ndim+1, T> &M) {
    for (int i = 0, k = 0; i < ndim; i++) {
      for (int j = 0; j < ndim + 1; j++, k++) {
        transform_out[k] = M(i, j);
      }
    }
  }

  template <typename T, int ndim>
  void ApplyTransform(T *transform_out, const T* transform_in, const mat<ndim+1, ndim+1, T> &M) {
    auto mat_in = mat<ndim+1, ndim+1, T>::identity();
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
  DALIDataType dtype_ = DALI_FLOAT;
  int ndim_ = 2; // TODO(janton): Fix  // will be inferred from the arguments
  int batch_size_ = -1;
  bool has_input_ = false;
};


/**
 * @brief Identity transformation. Just an example of a transform implementation
 */
class IdentityTransformCPU : public TransformBaseOp<CPUBackend, IdentityTransformCPU> {
 public:
  explicit IdentityTransformCPU(const OpSpec &spec)
      : TransformBaseOp<CPUBackend, IdentityTransformCPU>(spec) {}

  template <typename T, int ndim>
  void DefineTransform(mat<ndim+1, ndim+1, T> &M, workspace_t<CPUBackend> &ws) {
    (void) ws;
    M = mat<ndim+1, ndim+1, T>::identity();
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_GEOMETRY_TRANSFORM_BASE_OP_H_