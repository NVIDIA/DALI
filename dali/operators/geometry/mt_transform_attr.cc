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

#include <vector>
#include "dali/core/tensor_shape_print.h"
#include "dali/operators/geometry/mt_transform_attr.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

DALI_SCHEMA(MTTransformAttr)
  .DocStr(R"(A base schema for operators that require affine-like transform parameters as matrix
M and translation T or one fused argument MT.)")
  .AddOptionalArg<vector<float>>("M", R"(The matrix used for transforming the input vectors.

If left unspecified, identity matrix is used.

The matrix ``M`` does not need to be square - if it's not, the output vectors will have a
number of components equal to the number of rows in ``M``.

If a scalar value is provided, ``M`` is assumed to be a square matrix with that value on the
diagonal. The size of the matrix is then assumed to match the number of components in the
input vectors.)",
    nullptr,  // no default value
    true)
  .AddOptionalArg<vector<float>>("T", R"(The translation vector.

If left unspecified, no translation is applied unless MT argument is used.

The number of components of this vector must match the number of rows in matrix ``M``.
If a scalar value is provided, that value is broadcast to all components of ``T`` and the number
of components is chosen to match the number of rows in ``M``.)", nullptr, true)
  .AddOptionalArg<vector<float>>("MT", R"(A block matrix [M T] which combines the arguments
``M`` and ``T``.

Providing a scalar value for this argument is equivalent to providing the same scalar for
M and leaving T unspecified.

The number of columns must be one more than the number of components in the input.
This argument is mutually exclusive with ``M`` and ``T``.)",
    nullptr,
    true);

void MTTransformAttr::ProcessMatrixArg(const OpSpec &spec, const ArgumentWorkspace &ws, int N) {
  bool is_fused = HasFusedMT();
  const char *name = is_fused ? "MT" : "M";

  int cols = input_pt_dim_ + (is_fused ? 1 : 0);
  if (is_fused)
    translation_.clear();

  if (HasMatrixRegularArg()) {
    mtx_ = spec.template GetRepeatedArgument<float>(name);
    if (mtx_.size() == 1) {
      SetOutputPtDim(input_pt_dim_);
      mtx_.resize(output_pt_dim_ * input_pt_dim_);
      MakeDiagonalMatrix(mtx_, mtx_[0]);
      if (is_fused) {
        translation_.resize(output_pt_dim_, 0);
      }
    } else {
      DALI_ENFORCE(mtx_.size() > 0, make_string("The matrix argument ``", name,
        "`` must not be an empty list."));
      DALI_ENFORCE(mtx_.size() % cols == 0, make_string("Cannot form a matrix with ",
        mtx_.size(), " elements and ", cols, "columns"));
      SetOutputPtDim(mtx_.size() / cols);
      if (is_fused)
        SplitFusedMT();
    }
    Repeat(per_sample_mtx_, mtx_, N);
    if (is_fused)
      Repeat(per_sample_translation_, translation_, N);
  } else if (HasMatrixInputArg()) {
    const auto &M_inp = ws.ArgumentInput(name);
    auto M = view<const float>(M_inp);
    DALI_ENFORCE(is_uniform(M.shape), "Matrices for all samples in the batch must have "
      "the same shape.");
    DALI_ENFORCE(M.shape.sample_dim() == 0 || M.shape.sample_dim() == 2, make_string(
      "The parameter M must be a list of 2D matrices of same size or a list of scalars. Got: ",
      M.shape));
    if (M.shape.sample_dim() == 0) {  // we have a list of scalars - put them on diagonals
      SetOutputPtDim(cols);
      int mat_size = output_pt_dim_ * cols;
      per_sample_mtx_.resize(mat_size * N);
      for (int i = 0; i < N; i++) {
        MakeDiagonalMatrix(make_span(&per_sample_mtx_[i * mat_size], mat_size), M.data[i][0]);
      }
      if (is_fused) {
        translation_.resize(output_pt_dim_, 0);
        Repeat(per_sample_translation_, translation_, N);
      }
    } else {
      DALI_ENFORCE(M.shape[0][0] > 0, make_string("The matrix arugment ``", name,
        "`` with 0 rows is illegal."));
      DALI_ENFORCE(M.shape[0][1] == cols, make_string("The shape of the argument ``", name,
          "`` does not match the input shape. Got ", M.shape[0], " matrices and the input "
          "requires matrices with ", cols, " columns"));
      SetOutputPtDim(M.shape[0][0]);
      int mat_size = output_pt_dim_ * input_pt_dim_;
      per_sample_mtx_.resize(mat_size * N);
      if (is_fused)
        per_sample_translation_.resize(output_pt_dim_ * N);
      for (int s = 0; s < N; s++) {
        for (int i = 0; i < output_pt_dim_; i++) {
          for (int j = 0; j < input_pt_dim_; j++)
              per_sample_mtx_[s * mat_size + i * input_pt_dim_ + j] = M.data[s][i * cols + j];
          if (is_fused)
            per_sample_translation_[s * output_pt_dim_ + i] = M.data[s][i * cols + cols - 1];
        }
      }
    }
  } else {
    SetOutputPtDim(input_pt_dim_);
    if (static_cast<int>(mtx_.size()) != output_pt_dim_ * input_pt_dim_) {
      mtx_.resize(output_pt_dim_ * input_pt_dim_, 0);
      MakeDiagonalMatrix(mtx_, 1);
      Repeat(per_sample_mtx_, mtx_, N);
      if (is_fused) {
        translation_.resize(output_pt_dim_, 0);
        Repeat(per_sample_translation_, translation_, N);
      }
    }
  }
}

void MTTransformAttr::SplitFusedMT() {
  int cols = input_pt_dim_ + 1;
  translation_.resize(output_pt_dim_);
  for (int i = 0; i < output_pt_dim_; i++) {
    // store the last column in the translation vector
    translation_[i] = mtx_[i * cols + input_pt_dim_];

    // compact the matrix
    if (i > 0) {
      for (int j = 0; j < input_pt_dim_; j++)
        mtx_[i * input_pt_dim_ + j] = mtx_[i * cols + j];
    }  // else: the first row is already where it should be
  }
  mtx_.resize(output_pt_dim_ * input_pt_dim_);
}

void MTTransformAttr::ProcessTranslationArg(const OpSpec &spec,
                                            const ArgumentWorkspace &ws,
                                            int N) {
  const char *name = "T";
  if (has_translation_) {
    GetSingleOrRepeatedArg(spec, translation_, name, output_pt_dim_);
    Repeat(per_sample_translation_, translation_, N);
  } else if (has_translation_input_) {
    const auto &T_inp = ws.ArgumentInput(name);
    auto T = view<const float>(T_inp);
    DALI_ENFORCE(is_uniform(T.shape), "Translation vectors for all samples must have "
      "the same shape.");
    DALI_ENFORCE(T.shape.sample_dim() == 0 || T.shape.sample_dim() == 1, "The translation "
      "argument input must be a list of 1D tensors or a list of scalars.");
    translation_.clear();
    per_sample_translation_.resize(N * output_pt_dim_);
    if (T.shape.sample_dim() == 0) {
      for (int i = 0, k = 0; i < N; i++)
        for (int j = 0; j < output_pt_dim_; j++, k++)
          per_sample_translation_[k] = T.data[i][0];
    } else {
      DALI_ENFORCE(T.shape[0][0] == output_pt_dim_, make_string("Expected ",
        output_pt_dim_, "-D translation vectors, got ", T.shape[0][0], "-D"));
      for (int i = 0, k = 0; i < N; i++)
        for (int j = 0; j < output_pt_dim_; j++, k++)
          per_sample_translation_[k] = T.data[i][j];
    }
  } else {
    translation_.resize(output_pt_dim_, 0.0f);
    per_sample_translation_.resize(N * output_pt_dim_, 0.0f);
  }
}


}  // namespace dali
