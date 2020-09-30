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

DALI_SCHEMA(RotateTransform)
  .DocStr(R"code(Produces a rotation affine transform matrix.

If another transform matrix is passed as an input, the operator applies rotation to the matrix provided.

The number of dimensions will be 2 or 3 depending on whether the argument ``axis`` is provided or not,
respectively.

.. note::
    The output of this operator can be fed directly to the ``MT`` argument of ``CoordTransform`` operator.
)code")
  .AddArg(
    "angle",
    R"code(Angle, in degrees.

For two-dimensional data, the rotation is counter-clockwise, assuming the top-left corner is
at ``(0,0)``. For three-dimensional data, the ``angle`` is a positive rotation around the provided
axis.)code",
    DALI_FLOAT, true)
  .AddOptionalArg<std::vector<float>>(
    "axis",
    R"code(Applies **only** to three-dimension and is the axis around which to rotate the image.

The vector does not need to be normalized, but it must have a non-zero length.

Reversing the vector is equivalent to changing the sign of ``angle``.)code",
    nullptr, true)
  .AddOptionalArg<std::vector<float>>(
    "center",
    R"code(The center of the rotation operation.

If provided, the number of elements should match the number of dimensions.)code",
    nullptr, true)
  .NumInput(0, 1)
  .NumOutput(1)
  .AddParent("TransformAttr");

/**
 * @brief Rotate transformation.
 */
class RotateTransformCPU
    : public TransformBaseOp<CPUBackend, RotateTransformCPU> {
 public:
  using SupportedDims = dims<2, 3>;

  explicit RotateTransformCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, RotateTransformCPU>(spec),
      has_angle_input_(spec.HasTensorArgument("angle")),
      has_axis_(spec.HasArgument("axis")),
      has_axis_input_(spec.HasTensorArgument("axis")),
      has_center_(spec.HasArgument("center")),
      has_center_input_(spec.HasTensorArgument("center")) {

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

      if (has_center_ || has_center_input_) {
        vec2 &center = *reinterpret_cast<vec2*>(center_[i].data());
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
      vec3 &axis = *reinterpret_cast<vec3*>(axis_[i].data());
      mat = rotation3D(axis, deg2rad(angle));

      if (has_center_ || has_center_input_) {
        vec3 &center = *reinterpret_cast<vec3*>(center_[i].data());
        mat.set_col(ndim, cat(sub<ndim, ndim>(mat) * -center + center, 1.0f));
      }
    }
  }

  void ProcessAngleConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    angle_.resize(1);
    angle_[0] = spec.GetArgument<float>("angle");
  }

  void ProcessAxisConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    axis_.resize(1);
    axis_[0] = spec.GetArgument<std::vector<float>>("axis");
  }

  void ProcessCenterConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    center_.resize(1);
    center_[0] = spec.GetArgument<std::vector<float>>("center");
  }

  void ProcessAngleArgInput(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    const auto& angle = ws.ArgumentInput("angle");
    auto angle_view = view<const float>(angle);
    DALI_ENFORCE(is_uniform(angle_view.shape),
      "All samples in argument ``angle`` should have the same shape");
    DALI_ENFORCE(angle_view.shape[0][1] == 1,
      "``angle`` must be a single number");

    angle_.resize(nsamples_);
    for (int i = 0; i < nsamples_; i++) {
      angle_[i] = angle_view[i].data[0];
    }
  }

  void ProcessAxisArgInput(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    const auto& axis = ws.ArgumentInput("axis");
    auto axis_view = view<const float>(axis);
    DALI_ENFORCE(is_uniform(axis_view.shape),
      "All samples in argument ``axis`` should have the same shape");
    DALI_ENFORCE(axis_view.shape.sample_dim() == 1,
      "``axis`` must be a 1D tensor");

    axis_.resize(nsamples_);
    for (int i = 0; i < nsamples_; i++) {
      axis_[i].resize(ndim_);
      for (int d = 0; d < ndim_; d++) {
        axis_[i][d] = axis_view[i].data[d];
      }
    }
  }

  void ProcessCenterArgInput(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    const auto& center = ws.ArgumentInput("center");
    auto center_view = view<const float>(center);
    DALI_ENFORCE(is_uniform(center_view.shape),
      "All samples in argument ``center`` should have the same shape");
    DALI_ENFORCE(center_view.shape.sample_dim() == 1,
      "``center`` must be a 1D tensor");

    center_.resize(nsamples_);
    for (int i = 0; i < nsamples_; i++) {
      center_[i].resize(ndim_);
      for (int d = 0; d < ndim_; d++) {
        center_[i][d] = center_view[i].data[d];
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    if (has_angle_input_) {
      ProcessAngleArgInput(spec, ws);
    } else {
      ProcessAngleConstant(spec, ws);
    }

    ndim_ = has_axis_ || has_axis_input_ ? 3 : 2;

    if (has_axis_input_ || has_axis_) {
      if (has_center_input_) {
        ProcessAxisArgInput(spec, ws);
      } else {
        ProcessAxisConstant(spec, ws);
      }

      DALI_ENFORCE(ndim_ == static_cast<int>(axis_[0].size()),
        make_string("Unexpected number of dimensions for ``axis`` argument. Got: ",
                    axis_[0].size(), " but expected ", ndim_,
                    " dimensions."));
    }

    if (has_center_input_ || has_center_) {
      if (has_center_input_) {
        ProcessCenterArgInput(spec, ws);
      } else {
        ProcessCenterConstant(spec, ws);
      }

      DALI_ENFORCE(ndim_ == static_cast<int>(center_[0].size()),
        make_string("Unexpected number of dimensions for ``center`` argument. Got: ",
                    center_[0].size(), " but ``axis`` argument suggested ", ndim_,
                    " dimensions."));
    }

    if (center_.size() > 1 && axis_.size() == 1) {
      axis_.resize(nsamples_, axis_[0]);
    } else if (center_.size() == 1 && axis_.size() > 1) {
      center_.resize(nsamples_, center_[0]);
    }
  }

  bool IsConstantTransform() const {
    return !has_angle_input_ && !has_axis_input_ && !has_center_input_;
  }

 private:
  bool has_angle_input_ = false;
  bool has_axis_ = false;
  bool has_axis_input_ = false;
  bool has_center_ = false;
  bool has_center_input_ = false;
  std::vector<float> angle_;
  std::vector<std::vector<float>> axis_;
  std::vector<std::vector<float>> center_;
};

DALI_REGISTER_OPERATOR(RotateTransform, RotateTransformCPU, CPU);

}  // namespace dali
