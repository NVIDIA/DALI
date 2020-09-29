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

DALI_SCHEMA(ScaleTransform)
  .DocStr(R"code(Produces a scale affine transform matrix.

If another transform matrix is passed as an input, the operator applies scaling to the matrix provided.

.. note::
    The output of this operator can be fed directly to the ``MT`` argument of ``CoordTransform`` operator.
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
class ScaleTransformCPU
    : public TransformBaseOp<CPUBackend, ScaleTransformCPU> {
 public:
  using SupportedDims = dims<1, 2, 3, 4, 5, 6>;

  explicit ScaleTransformCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, ScaleTransformCPU>(spec),
      has_scale_input_(spec.HasTensorArgument("scale")),
      has_center_(spec.HasArgument("center")),
      has_center_input_(spec.HasTensorArgument("center")) {
  }

  template <typename T, int mat_dim>
  void DefineTransforms(span<affine_mat_t<T, mat_dim>> matrices) {
    constexpr int ndim = mat_dim - 1;
    assert(matrices.size() == static_cast<int>(scale_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      auto *scale = scale_[i].data();
      mat = affine_mat_t<T, mat_dim>::identity();
      for (int d = 0; d < ndim; d++) {
        mat(d, d) = scale[d];
      }

      if (has_center_ || has_center_input_) {
        auto *center = center_[i].data();
        for (int d = 0; d < ndim; d++) {
          mat(d, ndim) = center[d] * (T(1) - scale[d]);
        }
      }
    }
  }

  void ProcessScaleConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    scale_.resize(1);
    scale_[0] = spec.GetArgument<std::vector<float>>("scale");
    ndim_ = scale_[0].size();
  }

  void ProcessCenterConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    center_.resize(1);
    center_[0] = spec.GetArgument<std::vector<float>>("center");
  }

  void ProcessScaleArgInput(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    const auto& scale = ws.ArgumentInput("scale");
    auto scale_view = view<const float>(scale);
    DALI_ENFORCE(is_uniform(scale_view.shape),
      "All samples in argument ``scale`` should have the same shape");
    DALI_ENFORCE(scale_view.shape.sample_dim() == 1,
      "``scale`` must be a 1D tensor");

    scale_.resize(nsamples_);
    for (int i = 0; i < nsamples_; i++) {
      scale_[i].resize(ndim_);
      for (int d = 0; d < ndim_; d++) {
        scale_[i][d] = scale_view[i].data[d];
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
    if (has_scale_input_) {
      ProcessScaleArgInput(spec, ws);
    } else {
      ProcessScaleConstant(spec, ws);
    }

    if (has_center_input_ || has_center_) {
      if (has_center_input_) {
        ProcessCenterArgInput(spec, ws);
      } else {
        ProcessCenterConstant(spec, ws);
      }

      DALI_ENFORCE(ndim_ == static_cast<int>(scale_[0].size()),
        make_string("Unexpected number of dimensions for ``center`` argument. Got: ",
                    center_[0].size(), " but ``scale`` argument suggested ", ndim_,
                    " dimensions."));
    }

    if (center_.size() > 1 && scale_.size() == 1) {
      scale_.resize(nsamples_, scale_[0]);
    } else if (center_.size() == 1 && scale_.size() > 1) {
      center_.resize(nsamples_, center_[0]);
    }
  }

  bool IsConstantTransform() const {
    return !has_scale_input_ && !has_center_input_;
  }

 private:
  bool has_scale_input_ = false;
  bool has_center_ = false;
  bool has_center_input_ = false;
  std::vector<std::vector<float>> scale_;
  std::vector<std::vector<float>> center_;
};

DALI_REGISTER_OPERATOR(ScaleTransform, ScaleTransformCPU, CPU);

}  // namespace dali
