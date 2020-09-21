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

namespace dali {

class MTTransformAttr {
 public:
  explicit MTTransformAttr(const OpSpec &spec) {
    has_mt_ = spec_.HasArgument("MT");  // check if there's a regular argument MT (not input)
    has_mt_input_ = spec_.HasTensorArgument("MT");  // check if there's an Argument Input MT
    has_matrix_ = spec_.HasArgument("M");  // check if there's a regular argument M (not input)
    has_matrix_input_ = spec_.HasTensorArgument("M");  // check if there's an Argument Input M
    has_translation_ = spec_.HasArgument("T");  // ...and similarly, for T
    has_translation_input_ = spec.HasTensorArgument("T");

    bool has_fused_arg = has_mt_ || has_mt_input_;
    bool has_separate_args = has_matrix_ || has_matrix_input_ ||
                             has_translation_ || has_translation_input_;
    DALI_ENFORCE(!(has_fused_arg && has_separate_args), "The combined ``MT`` argument cannot be "
      "used together with separate ``M``, ``T`` arguments.");
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

 private:


  void ProcessMatrixArg(const workspace_t<Backend> &ws, const char *name, int N);

  void SplitFusedMT();

  void SetOutputPtDim(int d) {
    if (output_pt_dim_ > 0) {
      DALI_ENFORCE(d == output_pt_dim_, "The arguments suggest ", d, " output components, "
         "but it was previously set to ", output_pt_dim_);
    } else {
      output_pt_dim_ = d;
    }
  }

  void ProcessTranslationArg(const workspace_t<Backend> &ws, const char *name, int N);
  /** @brief Fill the diagonal with a scalar value, put zeros elsewhere */
  template <typename Container>
  void FillDiag(Container &&mtx, float value) {
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
