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

DALI_SCHEMA(TranslateTransform)
  .DocStr(R"code(Produces a translation affine transform matrix.

If another transform matrix is passed as an input, the operator apply translation to the matrix provided.

.. note::
    The output of this operator can be fed directly to the ``MT`` argument of ``CoordTransform`` operator.
)code")
  .AddArg(
    "offset",
    R"code(The translation vector.

The number of dimensions of the transform is inferred from this argument.)code",
    DALI_FLOAT_VEC, true)
  .NumInput(0, 1)
  .NumOutput(1);

/**
 * @brief Translation transformation.
 */
class TranslateTransformCPU
    : public TransformBaseOp<CPUBackend, TranslateTransformCPU> {
 public:
  using SupportedDims = dims<1, 2, 3, 4, 5, 6>;

  explicit TranslateTransformCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, TranslateTransformCPU>(spec),
      has_offset_input_(spec.HasTensorArgument("offset")) {
  }

  template <typename T, int ndim>
  void DefineTransforms(span<affine_mat_t<T, ndim>> matrices) {
    assert(matrices.size() == static_cast<int>(offset_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      auto *offset = offset_[i].data();
      mat = affine_mat_t<T, ndim>::identity();
      for (int d = 0; d < ndim; d++) {
        mat(d, ndim) = offset[d];
      }
    }
  }

  void ProcessOffsetConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    offset_.resize(1);
    offset_[0] = spec.GetArgument<std::vector<float>>("offset");
    ndim_ = offset_[0].size();
  }

  void ProcessOffsetArgInput(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    const auto& offset = ws.ArgumentInput("offset");
    auto offset_view = view<const float>(offset);
    DALI_ENFORCE(is_uniform(offset_view.shape),
      "All samples in argument ``offset`` should have the same shape");
    DALI_ENFORCE(offset_view.shape.sample_dim() == 1,
      "``offset`` must be a 1D tensor");
    ndim_ = offset_view[0].shape[0];

    offset_.resize(nsamples_);
    for (int i = 0; i < nsamples_; i++) {
      offset_[i].resize(ndim_);
      for (int d = 0; d < ndim_; d++) {
        offset_[i][d] = offset_view[i].data[d];
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    if (has_offset_input_) {
      ProcessOffsetArgInput(spec, ws);
    } else {
      ProcessOffsetConstant(spec, ws);
    }
  }

  bool IsConstantTransform() const {
    return !has_offset_input_;
  }

 private:
  bool has_offset_input_ = false;
  std::vector<std::vector<float>> offset_;
};

DALI_REGISTER_OPERATOR(TranslateTransform, TranslateTransformCPU, CPU);

}  // namespace dali
