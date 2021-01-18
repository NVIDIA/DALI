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
    assert((shear_.IsDefined() && matrices.size() <= static_cast<int>(shear_.size())) ||
           (angles_.IsDefined() && matrices.size() <= static_cast<int>(angles_.size())));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      if (shear_.IsDefined()) {
        bool is_vec = shear_.get().sample_dim() == 1;
        vec2 shear_factors = is_vec ? as_vec<2>(shear_[i])
                                    : as_mat<2, 1>(shear_[i]).col(0);
        mat = shear(shear_factors);
      } else {
        assert(angles_.IsDefined());
        bool is_vec = angles_.get().sample_dim() == 1;
        vec2 angles = is_vec ? as_vec<2>(angles_[i])
                             : as_mat<2, 1>(angles_[i]).col(0);
        vec2 shear_factors;
        shear_factors[0] = std::tan(deg2rad(angles[0]));
        shear_factors[1] = std::tan(deg2rad(angles[1]));
        mat = shear(shear_factors);
      }

      if (center_.IsDefined()) {
        vec2 center = as_vec<2>(center_[i]);
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
    assert((shear_.IsDefined() && matrices.size() <= static_cast<int>(shear_.size())) ||
           (angles_.IsDefined() && matrices.size() <= static_cast<int>(angles_.size())));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      if (shear_.IsDefined()) {
        const mat3x2 &shear_factors = as_mat<3, 2>(shear_[i]);
        mat = shear(shear_factors);
      } else {
        assert(angles_.IsDefined());
        vec<6> shear_factors;
        for (int j = 0; j < 6; j++)
          shear_factors[j] = std::tan(deg2rad(angles_[i].data[j]));
        mat = shear(*reinterpret_cast<mat3x2*>(&shear_factors));
      }
      if (center_.IsDefined()) {
        const vec3 &center = as_vec<3>(center_[i]);
        mat.set_col(ndim, cat(sub<ndim, ndim>(mat) * -center + center, 1.0f));
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    auto shape_from_size =
      [](int64_t size) {
        int ndim = sqrt(size) + 1;
        DALI_ENFORCE(size == ndim * (ndim - 1),
            make_string("Shear matrix must have D*(D-1) elements where D is the number "
                        "of dimensions. Got ", size, " elements."));
        if (ndim == 2) {
          return TensorShape<>{ndim};
        }
        return TensorShape<>{ndim, ndim - 1};
      };
    auto analyze_shape = [](const TensorShape<> &shape)->int {
      if (shape.size() == 1) {
        DALI_ENFORCE(shape[0] == 2, make_string("Shear coefficient must be a D x (D-1) "
            "matrix or a 2-element vector. Got: ", shape));
        return 2;
      } else {
        DALI_ENFORCE(shape.size() == 2 && shape[1] == shape[0] - 1,
            make_string("Shear coefficient must be a D x (D-1) "
                "matrix or a 2-element vector. Got: ", shape));
        return shape[0];
      }
    };
    if (shear_.IsDefined()) {
      shear_.Acquire(spec, ws, nsamples_, true, shape_from_size);
      ndim_ = analyze_shape(shear_.get().tensor_shape(0));
    } else {
      assert(angles_.IsDefined());
      angles_.Acquire(spec, ws, nsamples_, true, shape_from_size);
      ndim_ = analyze_shape(angles_.get().tensor_shape(0));
      for (int i = 0; i < angles_.size(); i++) {
        const auto& angles = angles_[i];
        for (int j = 0; j < angles.num_elements(); j++) {
          DALI_ENFORCE(angles.data[j] >= -90.0f && angles.data[j] <= 90.0f,
            make_string("Angle is expected to be in the range [-90, 90]. Got: ", angles.data[j]));
        }
      }
    }
    if (center_.IsDefined()) {
      center_.Acquire(spec, ws, nsamples_, TensorShape<1>{ndim_});
    }
  }

  bool IsConstantTransform() const {
    return !shear_.IsArgInput() && !angles_.IsArgInput() && !center_.IsArgInput();
  }

 private:
  ArgValue<float, DynamicDimensions> shear_;   // can either be a vec2 (ndim=2) or mat3x2 (ndim=3)
  ArgValue<float, DynamicDimensions> angles_;  // same as shear_
  ArgValue<float, 1> center_;
};

DALI_REGISTER_OPERATOR(transforms__Shear, TransformShearCPU, CPU);

}  // namespace dali
