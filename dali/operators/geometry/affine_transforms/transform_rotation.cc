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

DALI_SCHEMA(TransformRotation)
  .DocStr(R"code(Produces a rotation affine transform matrix.

If another transform matrix is passed as an input, the operator applies rotation to the matrix provided.

The number of dimensions is assumed to be 3 if a rotation axis is provided or 2 otherwise.

.. note::
    The output of this operator can be fed directly to the ``MT`` argument of ``CoordTransform`` operator.
)code")
  .AddArg(
    "angle",
    R"code(Angle, in degrees.)code",
    DALI_FLOAT, true)
  .AddOptionalArg<std::vector<float>>(
    "axis",
    R"code(Axis of rotation (applies **only** to 3D transforms).

The vector does not need to be normalized, but it must have a non-zero length.

Reversing the vector is equivalent to changing the sign of ``angle``.)code",
    nullptr, true)
  .AddOptionalArg<std::vector<float>>(
    "center",
    R"code(The center of the rotation.

If provided, the number of elements should match the dimensionality of the transform.)code",
    nullptr, true)
  .NumInput(0, 1)
  .NumOutput(1)
  .AddParent("TransformAttr");

/**
 * @brief Rotate transformation.
 */
class TransformRotationCPU
    : public TransformBaseOp<CPUBackend, TransformRotationCPU> {
 public:
  using SupportedDims = dims<2, 3>;

  explicit TransformRotationCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, TransformRotationCPU>(spec),
      angle_("angle", spec),
      axis_("axis", spec),
      center_("center", spec) {
    assert(angle_.IsDefined());
  }

  /**
   * @brief 2D rotation
   */
  template <typename T>
  void DefineTransforms(span<affine_mat_t<T, 3>> matrices) {
    constexpr int ndim = 2;
    assert(matrices.size() == static_cast<int>(angle_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      auto angle = angle_[i];
      mat = rotation2D(deg2rad(angle));

      if (center_.IsDefined()) {
        const vec2 &center = *reinterpret_cast<const vec2*>(center_[i].data());
        mat.set_col(ndim, cat(sub<ndim, ndim>(mat) * -center + center, 1.0f));
      }
    }
  }

  /**
   * @brief 3D rotation
   */
  template <typename T>
  void DefineTransforms(span<affine_mat_t<T, 4>> matrices) {
    constexpr int ndim = 3;
    assert(matrices.size() == static_cast<int>(angle_.size()));
    assert(matrices.size() == static_cast<int>(axis_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      auto angle = angle_[i];
      const vec3 &axis = *reinterpret_cast<const vec3*>(axis_[i].data());
      mat = rotation3D(axis, deg2rad(angle));

      if (center_.IsDefined()) {
        const vec3 &center = *reinterpret_cast<const vec3*>(center_[i].data());
        mat.set_col(ndim, cat(sub<ndim, ndim>(mat) * -center + center, 1.0f));
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    int repeat = IsConstantTransform() ? 1 : nsamples_;
    angle_.Read(spec, ws, repeat);
    ndim_ = axis_.IsDefined() ? 3 : 2;
    if (axis_.IsDefined()) {
      axis_.Read(spec, ws, repeat);
      DALI_ENFORCE(ndim_ == static_cast<int>(axis_[0].size()),
        make_string("Unexpected number of dimensions for ``axis`` argument. Got: ",
                    axis_[0].size(), " but expected ", ndim_,
                    " dimensions."));
    }
    if (center_.IsDefined()) {
      center_.Read(spec, ws, repeat);
      DALI_ENFORCE(ndim_ == static_cast<int>(center_[0].size()),
        make_string("Unexpected number of dimensions for ``center`` argument. Got: ",
                    center_[0].size(), " but ``axis`` argument suggested ", ndim_,
                    " dimensions."));
    }
  }

  bool IsConstantTransform() const {
    return !angle_.IsArgInput() && !axis_.IsArgInput() && !center_.IsArgInput();
  }

 private:
  Argument<float> angle_;
  Argument<std::vector<float>> axis_;
  Argument<std::vector<float>> center_;
};

DALI_REGISTER_OPERATOR(TransformRotation, TransformRotationCPU, CPU);

}  // namespace dali
