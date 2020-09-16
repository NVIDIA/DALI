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

#ifndef DALI_OPERATORS_COORD_COORD_TRANSFORM_H_
#define DALI_OPERATORS_COORD_COORD_TRANSFORM_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/mat.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

#define COORD_TRANSFORM_INPUT_TYPES (uint8_t, int16_t, uint16_t, int32_t, float)
#define COORD_TRANSFORM_DIMS (1, 2, 3, 4, 5, 6)

template <typename Backend>
class CoordTransform : public Operator<Backend> {
 public:
  explicit CoordTransform(const OpSpec &spec) : Operator<Backend>(spec) {
    dtype_ = spec_.template GetArgument<DALIDataType>("dtype");
    has_matrix_ = spec_.HasArgument("M");  // checks if there's a regular argument M (not input)
    has_matrix_input_ = spec_.HasTensorArgument("M");  // cheks if there's an Argument Input M
    has_translation_ = spec_.HasArgument("T");  // ...and similarly, for T
    has_translation_input_ = spec.HasTensorArgument("T");
  }

  bool CanInferOutputs() const override { return true; }

 protected:
  using Operator<Backend>::spec_;
  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) override {
    auto &input = ws.template InputRef<Backend>(0);  // get a reference to the input tensor list
    const auto &input_shape = input.shape();         // get a shape - use const-ref to avoid copying
    output_descs.resize(1);                          // only one output
    output_descs[0].type = TypeTable::GetTypeInfo(dtype_);

    PrepareTransformArguments(ws, input_shape);      // this is where the magic happens
    // Now we know the matrix size and therefore number of output vector components.
    // This allows us to set the output shape.

    const int N = input_shape.num_samples();
    output_descs[0].shape = input_shape;             // copy the input shape...
    for (int i = 0; i < N; i++) {
      // ...and replace the number of vector components
      output_descs[0].shape.tensor_shape_span(i).back() = output_pt_dim_;
    }
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    auto &in = ws.template InputRef<Backend>(0);
    DALIDataType in_type = in.type().id();
    VALUE_SWITCH(output_pt_dim_, static_out_dim, COORD_TRANSFORM_DIMS, (
        VALUE_SWITCH(input_pt_dim_, static_in_dim, COORD_TRANSFORM_DIMS, (
            TYPE_SWITCH(in_type, type2id, InputType, COORD_TRANSFORM_INPUT_TYPES, (
                if (dtype_ == in_type) {
                  RunTyped<InputType, InputType, static_out_dim, static_in_dim>(ws);
                } else {
                  DALI_ENFORCE(dtype_ == DALI_FLOAT, make_string("CoordTransform output type "
                    "must be the same as input type or `float`. Got: ", dtype_));
                  RunTyped<float, InputType, static_out_dim, static_in_dim>(ws);
                }
              ), (  // NOLINT
                DALI_FAIL(make_string("Unsupported input type: ", in_type));
              )  // NOLINT
            )
          ), (  // NOLINT
            DALI_FAIL(make_string("Unsupported input point dimensionality: ", input_pt_dim_));
          )   // NOLINT
        )  // NOLINT
      ), (  // NOLINT
        DALI_FAIL(make_string("Unsupported input point dimensionality: ", input_pt_dim_));
      )  // NOLINT
    )  // NOLINT
  }

  void PrepareTransformArguments(const workspace_t<Backend> &ws,
                                 const TensorListShape<> &input_shape) {
    input_pt_dim_ = 0;
    output_pt_dim_ = 0;

    DALI_ENFORCE(input_shape.sample_dim() >= 2,
      "CoordTransform expects an input with at least 2 dimensions.");

    const int N = input_shape.num_samples();
    input_pt_dim_ = 0;
    for (int i = 0; i < N; i++) {
      auto sample_shape = input_shape.tensor_shape_span(i);
      if (volume(sample_shape) == 0)
        continue;
      int pt_dim = input_shape.tensor_shape_span(i).back();
      if (input_pt_dim_ == 0) {
        input_pt_dim_ = pt_dim;
      } else {
        DALI_ENFORCE(pt_dim == input_pt_dim_, make_string("The point dimensions must be the same "
        "for all input samples. Got: ", input_shape, "."));
      }
    }
    if (input_pt_dim_ == 0)
      return;  // data is degenerate - empty batch or a batch of empty tensors

    ProcessMatrixArg(ws, "M", N);
    ProcessTranslationArg(ws, "T", N);
  }

  void ProcessMatrixArg(const workspace_t<Backend> &ws, const char *name, int N) {
    if (has_matrix_) {
      mtx_ = spec_.template GetRepeatedArgument<float>(name);
      if (mtx_.size() == 1) {
        output_pt_dim_ = input_pt_dim_;
        mtx_.resize(output_pt_dim_ * input_pt_dim_);
        FillDiag(mtx_, mtx_[0]);
      } else {
        DALI_ENFORCE(mtx_.size() % input_pt_dim_ == 0, make_string("Cannot form a matrix ",
          mtx_.size(), " elements and ", input_pt_dim_, "columns"));
        output_pt_dim_ = mtx_.size() / input_pt_dim_;
      }
      Repeat(per_sample_mtx_, mtx_, N);
    } else if (has_matrix_input_) {
      const auto &M_inp = ws.ArgumentInput(name);
      auto M = view<const float>(M_inp);
      DALI_ENFORCE(is_uniform(M.shape), "Matrices for all samples int the batch must have "
        "the same shape.");
      DALI_ENFORCE(M.shape.sample_dim() == 0 || M.shape.sample_dim() == 2, make_string(
        "The parameter M must be a list of 2D matrices of same size or a list of scalars. Got: ",
        M.shape));
      if (M.shape.sample_dim() == 0) {  // we have a list of scalars - put them on diagonals
        output_pt_dim_ = input_pt_dim_;
        int mat_size = output_pt_dim_ * input_pt_dim_;
        per_sample_mtx_.resize(mat_size * N);
        for (int i = 0; i < N; i++) {
          FillDiag(make_span(&per_sample_mtx_[i * mat_size], mat_size), M.data[i][0]);
        }
      } else {
        DALI_ENFORCE(M.shape[0][1] == input_pt_dim_, make_string("The shape of the matrix argument "
           "does not match the input shape. Got ", M.shape[0], " matrices and the input requires "
           "matrices with ", input_pt_dim_, " columns"));
        output_pt_dim_ = M.shape[0][0];
        int mat_size = output_pt_dim_ * input_pt_dim_;
        per_sample_mtx_.resize(mat_size * N);
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < mat_size; j++)
              per_sample_mtx_[i * mat_size + j] = M.data[i][j];
        }
      }
    } else {
      output_pt_dim_ = input_pt_dim_;
      if (static_cast<int>(mtx_.size()) != output_pt_dim_ * input_pt_dim_) {
        mtx_.resize(output_pt_dim_ * input_pt_dim_, 0);
        FillDiag(mtx_, 1);
        Repeat(per_sample_mtx_, mtx_, N);
      }
    }
  }

  /** @brief Fill the diagonal with a scalar value, put zeros elsewhere */
  template <typename Container>
  void FillDiag(Container &&mtx, float value) {
    assert(static_cast<int>(size(mtx)) == input_pt_dim_ * output_pt_dim_);
    for (int i = 0, k = 0; i < output_pt_dim_; i++)
      for (int j = 0; j < input_pt_dim_; j++, k++)
          mtx[k] = (i == j ? value : 0);
  }

  void ProcessTranslationArg(const workspace_t<Backend> &ws, const char *name, int N) {
    if (has_translation_) {
      GetSingleOrRepeatedArg(spec_, translation_, name, output_pt_dim_);
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

  template <typename OutRange, typename InRange>
  void Repeat(OutRange &&out, InRange &&in, int times) {
    ssize_t n = size(in);
    resize_if_possible(out, n * times);
    for (ssize_t i = 0, k = 0; i < times; i++)
      for (ssize_t j = 0; j < n; j++, k++)
        out[k] = in[j];
  }

 private:
  template <typename OutputType, typename InputType, int out_dim, int in_dim>
  void RunTyped(workspace_t<Backend> &ws);


  vector<float> mtx_;
  vector<float> translation_;
  vector<float> per_sample_mtx_;
  vector<float> per_sample_translation_;
  int input_pt_dim_ = 0, output_pt_dim_ = 0;

  bool has_matrix_            = false;
  bool has_matrix_input_      = false;
  bool has_translation_       = false;
  bool has_translation_input_ = false;

  DALIDataType dtype_;

  kernels::KernelManager kmgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_COORD_COORD_TRANSFORM_H_
