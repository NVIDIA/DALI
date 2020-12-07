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
  }

  template <typename T, int mat_dim>
  void DefineTransforms(span<affine_mat_t<T, mat_dim>> matrices) {
    constexpr int ndim = mat_dim - 1;
    assert(matrices.size() == static_cast<int>(scale_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      auto scale = scale_[i];
      mat = affine_mat_t<T, mat_dim>::identity();
      for (int d = 0; d < ndim; d++) {
        mat(d, d) = scale[d];
      }

      if (center_.IsDefined()) {
        auto center = center_[i];
        for (int d = 0; d < ndim; d++) {
          mat(d, ndim) = center[d] * (T(1) - scale[d]);
        }
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    int repeat = IsConstantTransform() ? 0 : nsamples_;
    assert(scale_.IsDefined());
    scale_.Read(spec, ws, repeat);
    ndim_ = scale_[0].size();

    if (center_.IsDefined()) {
      center_.Read(spec, ws, repeat);
      DALI_ENFORCE(ndim_ == static_cast<int>(center_[0].size()),
        make_string("Unexpected number of dimensions for ``center`` argument. Got: ",
                    center_[0].size(), " but ``scale`` argument suggested ", ndim_,
                    " dimensions."));
    }
  }

  bool IsConstantTransform() const {
    return !scale_.IsArgInput() && !center_.IsArgInput();
  }

 private:
  Argument<std::vector<float>> scale_;
  Argument<std::vector<float>> center_;
};

DALI_REGISTER_OPERATOR(transforms__Scale, TransformScaleCPU, CPU);

}  // namespace dali
