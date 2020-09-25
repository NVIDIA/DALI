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

#include "dali/operators/geometry/transform_base_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(TranslateTransform)
  .DocStr(R"(TranslateTransform)")
  .AddArg("offset", "Translation vector", DALI_FLOAT_VEC, true)
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

  template <typename T>
  void DefineAffineMatrixTyped(int, const float*, dims<>) {
    DALI_FAIL(make_string("Unsupported number of dimensions ", ndim_));
  }

  template <typename T, int ndim, int... ndims>
  void DefineAffineMatrixTyped(int sample_idx, const float* offset, dims<ndim, ndims...>) {
    if (ndim_ != ndim) {
      DefineAffineMatrixTyped<T>(sample_idx, offset, dims<ndims...>());
      return;
    }
    auto mat_ptr = reinterpret_cast<affine_mat_t<T, ndim> *>(
      matrices_data_.data() + sample_idx * sizeof(affine_mat_t<T, ndim>));
    *mat_ptr = affine_mat_t<T, ndim>::identity();  // identity
    for (int d = 0; d < ndim; d++) {
      (*mat_ptr)(d, ndim) = offset[d];
    }
  }

  void ProcessOffsetConstant(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    auto offset = spec.GetArgument<std::vector<float>>("offset");
    ndim_ = offset.size();
    TYPE_SWITCH(dtype_, type2id, T, TRANSFORM_INPUT_TYPES, (
      matrices_data_.resize(nsamples_ * (ndim_+1) * (ndim_+1) * sizeof(T));
      for (int i = 0; i < nsamples_; i++) {
        DefineAffineMatrixTyped<T>(i, offset.data(), SupportedDims());
      }
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

  void ProcessOffsetArgInput(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    const auto& offset = ws.ArgumentInput("offset");
    auto offset_view = view<const float>(offset);
    DALI_ENFORCE(is_uniform(offset_view.shape),
      "All samples in argument ``offset`` should have the same shape");
    DALI_ENFORCE(offset_view.shape.sample_dim() == 1,
      "``offset`` must be a 1D tensor");
    ndim_ = offset_view[0].shape[0];
    TYPE_SWITCH(dtype_, type2id, T, TRANSFORM_INPUT_TYPES, (
      matrices_data_.resize(nsamples_ * (ndim_+1) * (ndim_+1) * sizeof(T));
      for (int i = 0; i < nsamples_; i++) {
        DefineAffineMatrixTyped<T>(i, offset_view[i].data, SupportedDims());
      }
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    if (!has_offset_input_) {
      ProcessOffsetConstant(spec, ws);
    } else {
      ProcessOffsetArgInput(spec, ws);
    }
  }

 private:
  bool has_offset_input_ = false;
};

DALI_REGISTER_OPERATOR(TranslateTransform, TranslateTransformCPU, CPU);

}  // namespace dali
