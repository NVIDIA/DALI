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

/**
 * @brief Identity transformation. Just an example of a transform implementation
 */
class IdentityTransformCPU
    : public TransformBaseOp<CPUBackend, IdentityTransformCPU> {
 public:
  explicit IdentityTransformCPU(const OpSpec &spec)
      : TransformBaseOp<CPUBackend, IdentityTransformCPU>(spec) {}
  using SupportedDims = dims<1, 2, 3, 4, 5, 6>;

  template <typename T, int ndim>
  void DefineTransform(span<affine_mat_t<T, ndim>> matrices, workspace_t<CPUBackend> &ws) {
    (void) ws;
    for (auto &m : matrices) {
      m = affine_mat_t<T, ndim>::identity();
    }
  }
};

DALI_SCHEMA(IdentityTransform)
  .DocStr(R"(IdentityTransform)")
  .NumInput(0, 1)
  .NumOutput(1);

DALI_REGISTER_OPERATOR(IdentityTransform, IdentityTransformCPU, CPU);


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
    if (!has_offset_input_) {
      offset_ = spec.GetArgument<std::vector<float>>("offset");
    }
  }

  int ndim(const workspace_t<CPUBackend> &ws) const {
    int ndim = -1;
    int expected_ndim = offset_.size();
    if (has_input_) {
      ndim = input_transform_ndim(ws);
      DALI_ENFORCE(ndim == expected_ndim,
        make_string(
          "Unexpected number of dimensions: ``offset`` provided , ", expected_ndim, 
          " dimensions but the input transport has ", ndim, " dimensions"));
    } else {
      ndim = expected_ndim;
    }
    return ndim;
  }

  template <typename T, int ndim>
  void DefineTransform(span<affine_mat_t<T, ndim>> matrices, workspace_t<CPUBackend> &ws) {
    if (matrices.empty())
      return;

    if (has_offset_input_) {
      const auto& offset = ws.ArgumentInput("offset");
      auto offset_view = view<const T>(offset);
      assert(offset_view.shape.sample_dim() == 1);
      for (int i = 0; i < matrices.size(); i++) {
        assert(ndim == offset_view.shape[i][0]);
        auto &matrix = matrices[i];
        matrix = affine_mat_t<T, ndim>::identity();
        for (int d = 0; d < ndim; d++) {
          matrix(d, ndim) = offset_view.data[i][d];
        }
      }
    } else {
      assert(ndim == offset_.size());
      auto &matrix = matrices[0];
      matrix = affine_mat_t<T, ndim>::identity();
      for (int d = 0; d < ndim; d++) {
        matrix(d, ndim) = offset_[d];
      }
      for (int i = 1; i < matrices.size(); i++) {
        matrices[i] = matrix;
      }
    }
  }

 private:
  std::vector<float> offset_;
  bool has_offset_input_ = false;
};

DALI_SCHEMA(TranslateTransform)
  .DocStr(R"(TranslateTransform)")
  .AddArg("offset", "Translation vector", DALI_FLOAT_VEC, true)
  .NumInput(0, 1)
  .NumOutput(1);

DALI_REGISTER_OPERATOR(TranslateTransform, TranslateTransformCPU, CPU);

}  // namespace dali
