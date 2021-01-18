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

#include "dali/operators/geometry/affine_transforms/transform_base_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(transforms__Scale)
  .DocStr(R"code(Produces a scale affine transform matrix.

If another transform matrix is passed as an input, the operator applies scaling to the matrix provided.

.. note::
    The output of this operator can be fed directly to ``CoordTransform`` and ``WarpAffine`` operators.
)code")
  .AddArg(
    "scale",
    R"code(The scale factor, per dimension.

The number of dimensions of the transform is inferred from this argument.)code",
    DALI_FLOAT_VEC, true)
  .AddOptionalArg<std::vector<float>>(
    "center",
    R"code(The center of the scale operation.

If provided, the number of elements should match the one of ``scale`` argument.)code",
    nullptr, true)
  .AddOptionalArg<int>(
    "ndim",
    R"code(Number of dimensions.

It should be provided when the number of dimensions can't be inferred. For example,
when `scale` is a scalar value and there's no input transform.
)code",
    nullptr, false)
  .NumInput(0, 1)
  .NumOutput(1)
  .AddParent("TransformAttr");

/**
 * @brief Scale transformation.
 */
class TransformScaleCPU
    : public TransformBaseOp<CPUBackend, TransformScaleCPU> {
 public:
  using SupportedDims = dims<1, 2, 3, 4, 5, 6>;

  explicit TransformScaleCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, TransformScaleCPU>(spec),
      scale_("scale", spec),
      center_("center", spec) {
    assert(scale_.IsDefined());
    if (spec.HasArgument("ndim"))
      ndim_arg_ = spec.GetArgument<int>("ndim");
  }

  template <typename T, int mat_dim>
  void DefineTransforms(span<affine_mat_t<T, mat_dim>> matrices) {
    constexpr int ndim = mat_dim - 1;
    assert(matrices.size() <= static_cast<int>(scale_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      auto scale = as_vec<ndim>(scale_[i]);
      mat = affine_mat_t<T, mat_dim>::identity();
      for (int d = 0; d < ndim; d++) {
        mat(d, d) = scale[d];
      }

      if (center_.IsDefined()) {
        auto center = center_[i].data;
        for (int d = 0; d < ndim; d++) {
          mat(d, ndim) = center[d] * (T(1) - scale[d]);
        }
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    assert(scale_.IsDefined());
    scale_.Acquire(spec, ws, nsamples_, true);
    int scale_ndim = scale_[0].num_elements();

    if (scale_ndim > 1) {
      ndim_ = scale_ndim;
    } else if (has_input_) {
      ndim_ = input_transform_ndim(ws);
    } else if (ndim_arg_ > 0) {
      ndim_ = ndim_arg_;
    } else {
      ndim_ = 1;
    }

    if (center_.IsDefined()) {
      center_.Acquire(spec, ws, nsamples_, TensorShape<1>{ndim_});
    }
  }

  bool IsConstantTransform() const {
    return !scale_.IsArgInput() && !center_.IsArgInput();
  }

 private:
  ArgValue<float, 1> scale_;
  ArgValue<float, 1> center_;
  int ndim_arg_ = -1;
};

DALI_REGISTER_OPERATOR(transforms__Scale, TransformScaleCPU, CPU);

}  // namespace dali
