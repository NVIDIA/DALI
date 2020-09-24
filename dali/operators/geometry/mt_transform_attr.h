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

#ifndef DALI_OPERATORS_GEOMETRY_MT_TRANSFORM_ATTR_H_
#define DALI_OPERATORS_GEOMETRY_MT_TRANSFORM_ATTR_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/mat.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

class DLL_PUBLIC MTTransformAttr {
 public:
  explicit MTTransformAttr(const OpSpec &spec) {
    has_mt_ = spec.HasArgument("MT");  // check if there's a regular argument MT (not input)
    has_mt_input_ = spec.HasTensorArgument("MT");  // check if there's an Argument Input MT
    has_matrix_ = spec.HasArgument("M");  // check if there's a regular argument M (not input)
    has_matrix_input_ = spec.HasTensorArgument("M");  // check if there's an Argument Input M
    has_translation_ = spec.HasArgument("T");  // ...and similarly, for T
    has_translation_input_ = spec.HasTensorArgument("T");

    bool has_fused_arg = has_mt_ || has_mt_input_;
    bool has_separate_args = has_matrix_ || has_matrix_input_ ||
                             has_translation_ || has_translation_input_;
    DALI_ENFORCE(!(has_fused_arg && has_separate_args), "The combined ``MT`` argument cannot be "
      "used together with separate ``M``, ``T`` arguments.");
  }

  template <int out_dim, int in_dim>
  span<const mat<out_dim, in_dim>> GetMatrices() const {
    assert(out_dim == output_pt_dim_);
    assert(in_dim == input_pt_dim_);
    const auto *M = reinterpret_cast<const mat<out_dim, in_dim> *>(per_sample_mtx_.data());
    int N = per_sample_mtx_.size() / (out_dim * in_dim);
    return { M, N };
  }

  template <int out_dim>
  span<const vec<out_dim>> GetTranslations() const {
    assert(out_dim == output_pt_dim_);
    const auto *T = reinterpret_cast<const vec<out_dim> *>(per_sample_translation_.data());
    int N = per_sample_translation_.size() / out_dim;
    return { T, N };
  }

  void SetTransformDims(int input_pt_dim, int output_pt_dim = -1) {
      input_pt_dim_  = input_pt_dim;
      output_pt_dim_ = output_pt_dim;
  }

  bool HasFusedMT() const {
    return has_mt_ || has_mt_input_;
  }

  bool HasMatrixRegularArg() const {
    return has_matrix_ || has_mt_;
  }

  bool HasMatrixInputArg() const {
    return has_matrix_input_ || has_mt_input_;
  }


  vector<float> mtx_;
  vector<float> translation_;
  vector<float> per_sample_mtx_;
  vector<float> per_sample_translation_;

  int input_pt_dim_ = -1, output_pt_dim_ = -1;

  void ProcessTransformArgs(const OpSpec &spec, const ArgumentWorkspace &ws, int N) {
    ProcessMatrixArg(spec, ws, N);
    if (!HasFusedMT())
      ProcessTranslationArg(spec, ws, N);
  }

 private:
  void SplitFusedMT();

  void ProcessMatrixArg(const OpSpec &spec, const ArgumentWorkspace &ws, int N);
  void ProcessTranslationArg(const OpSpec &spec, const ArgumentWorkspace &ws, int N);

  void SetOutputPtDim(int d) {
    if (output_pt_dim_ > 0) {
      DALI_ENFORCE(d == output_pt_dim_, make_string("The arguments suggest ", d,
        " output components, but it was previously set to ", output_pt_dim_));
    } else {
      output_pt_dim_ = d;
    }
  }

  /** @brief Fill the diagonal with a scalar value, put zeros elsewhere */
  template <typename Container>
  void MakeDiagonalMatrix(Container &&mtx, float value) {
    assert(static_cast<int>(size(mtx)) == input_pt_dim_ * output_pt_dim_);
    for (int i = 0, k = 0; i < output_pt_dim_; i++)
      for (int j = 0; j < input_pt_dim_; j++, k++)
          mtx[k] = (i == j ? value : 0);
  }

  template <typename OutRange, typename InRange>
  void Repeat(OutRange &&out, InRange &&in, int times) {
    ssize_t n = size(in);
    resize_if_possible(out, n * times);
    for (ssize_t i = 0, k = 0; i < times; i++)
      for (ssize_t j = 0; j < n; j++, k++)
        out[k] = in[j];
  }

  bool has_mt_                = false;
  bool has_mt_input_          = false;
  bool has_matrix_            = false;
  bool has_matrix_input_      = false;
  bool has_translation_       = false;
  bool has_translation_input_ = false;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GEOMETRY_MT_TRANSFORM_ATTR_H_
