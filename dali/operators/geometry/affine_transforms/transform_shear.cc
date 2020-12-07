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
#include "dali/core/geom/transform.h"
#include "dali/core/math_util.h"

namespace dali {

DALI_SCHEMA(transforms__Shear)
  .DocStr(R"code(Produces a shear affine transform matrix.

If another transform matrix is passed as an input, the operator applies the shear mapping to the matrix provided.

.. note::
    The output of this operator can be fed directly to ``CoordTransform`` and ``WarpAffine`` operators.
)code")
  .AddOptionalArg<std::vector<float>>(
    "shear",
    R"code(The shear factors.

For 2D, ``shear`` contains two elements: shear_x, shear_y.

For 3D, ``shear`` contains six elements: shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy.

A shear factor value can be interpreted as the offset to be applied in the first axis when moving in the
direction of the second axis.

.. note::
    This argument is mutually exclusive with ``angles``. 
    If provided, the number of dimensions of the transform is inferred from this argument.
)code",
    nullptr, true)
  .AddOptionalArg<std::vector<float>>(
    "angles",
    R"code(The shear angles, in degrees.

This argument is mutually exclusive with ``shear``.

For 2D, ``angles`` contains two elements: angle_x, angle_y.

For 3D, ``angles`` contains six elements: angle_xy, angle_xz, angle_yx, angle_yz, angle_zx, angle_zy.

A shear angle is translated to a shear factor as follows::

    shear_factor = tan(deg2rad(shear_angle))

.. note::
    The valid range of values is between -90 and 90 degrees.
    This argument is mutually exclusive with ``shear``. 
    If provided, the number of dimensions of the transform is inferred from this argument.
)code",
    nullptr, true)
  .AddOptionalArg<std::vector<float>>(
    "center",
    R"code(The center of the shear operation.

If provided, the number of elements should match the dimensionality of the transform.)code",
    nullptr, true)
  .NumInput(0, 1)
  .NumOutput(1)
  .AddParent("TransformAttr");

/**
 * @brief Scale transformation.
 */
class TransformShearCPU
    : public TransformBaseOp<CPUBackend, TransformShearCPU> {
 public:
  using SupportedDims = dims<2, 3>;

  explicit TransformShearCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, TransformShearCPU>(spec),
      shear_("shear", spec),
      angles_("angles", spec),
      center_("center", spec) {
    DALI_ENFORCE(shear_.IsDefined() + angles_.IsDefined() == 1,
      "One and only one of the following arguments is expected: ``shear`` or ``angles``");
  }

 /**
   * @brief 2D shear
   */
  template <typename T>
  void DefineTransforms(span<affine_mat_t<T, 3>> matrices) {
    constexpr int ndim = 2;
    assert(matrices.size() == static_cast<int>(shear_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      vec2 &shear_factors = *reinterpret_cast<vec2*>(shear_[i].data());
      mat = shear(shear_factors);

      if (center_.IsDefined()) {
        const vec2 &center = *reinterpret_cast<const vec2*>(center_[i].data());
        mat.set_col(ndim, cat(sub<ndim, ndim>(mat) * -center + center, 1.0f));
      }
    }
  }

  /**
   * @brief 3D shear
   */
  template <typename T>
  void DefineTransforms(span<affine_mat_t<T, 4>> matrices) {
    constexpr int ndim = 3;
    assert(matrices.size() == static_cast<int>(shear_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      const mat3x2 &shear_factors = *reinterpret_cast<const mat3x2*>(shear_[i].data());
      mat = shear(shear_factors);
      if (center_.IsDefined()) {
        const vec3 &center = *reinterpret_cast<const vec3*>(center_[i].data());
        mat.set_col(ndim, cat(sub<ndim, ndim>(mat) * -center + center, 1.0f));
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    int repeat = IsConstantTransform() ? 0 : nsamples_;
    if (shear_.IsDefined()) {
      shear_.Read(spec, ws, repeat);
      ndim_ = InferNumDims(shear_);
    } else {
      assert(angles_.IsDefined());
      angles_.Read(spec, ws, repeat);
      ndim_ = InferNumDims(angles_);
      shear_.resize(angles_.size());
      for (size_t i = 0; i < angles_.size(); i++) {
        auto &shear = shear_[i];
        auto &angles = angles_[i];
        int nangles = angles.size();
        shear.resize(nangles);
        for (int j = 0; j < nangles; j++) {
          DALI_ENFORCE(angles[j] >= -90.0f && angles[j] <= 90.0f,
            make_string("Angle is expected to be in the range [-90, 90]. Got: ", angles[j]));
          shear[j] = std::tan(deg2rad(angles[j]));
        }
      }
    }
    if (center_.IsDefined()) {
      center_.Read(spec, ws, repeat);
      DALI_ENFORCE(ndim_ == static_cast<int>(center_[0].size()),
        make_string("Unexpected number of dimensions for ``center`` argument. Got: ",
                    center_[0].size(), " but ``scale`` argument suggested ", ndim_,
                    " dimensions."));
    }
  }

  bool IsConstantTransform() const {
    return !shear_.IsArgInput() && !angles_.IsArgInput() && !center_.IsArgInput();
  }

 private:
  int InferNumDims(const Argument<std::vector<float>> &arg) {
    DALI_ENFORCE(arg[0].size() == 2 || arg[0].size() == 6,
      make_string("Unexpected number of elements in ``", arg.name(), "`` argument. "
                  "Expected 2 or 6 arguments. Got: ", arg[0].size()));
    return arg[0].size() == 6 ? 3 : 2;
  }

  Argument<std::vector<float>> shear_;
  Argument<std::vector<float>> angles_;
  Argument<std::vector<float>> center_;
};

DALI_REGISTER_OPERATOR(transforms__Shear, TransformShearCPU, CPU);

}  // namespace dali
