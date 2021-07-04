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

#ifndef DALI_OPERATORS_GEOMETRY_COORD_TRANSFORM_H_
#define DALI_OPERATORS_GEOMETRY_COORD_TRANSFORM_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/mat.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/geometry/mt_transform_attr.h"

namespace dali {

#define COORD_TRANSFORM_INPUT_TYPES (uint8_t, int16_t, uint16_t, int32_t, float)
#define COORD_TRANSFORM_DIMS (1, 2, 3, 4, 5, 6)

template <typename Backend>
class CoordTransform : public Operator<Backend>, private MTTransformAttr {
 public:
  explicit CoordTransform(const OpSpec &spec) : Operator<Backend>(spec), MTTransformAttr(spec) {
    dtype_ = spec_.template GetArgument<DALIDataType>("dtype");
  }

  bool CanInferOutputs() const override { return true; }

 protected:
  using Operator<Backend>::spec_;
  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) override {
    auto &input = ws.template InputRef<Backend>(0);  // get a reference to the input tensor list
    const auto &input_shape = input.shape();         // get a shape - use const-ref to avoid copying
    output_descs.resize(1);                          // only one output
    output_descs[0].type = TypeTable::GetTypeInfo(dtype_);

    CheckType(input.type().id());

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

  void CheckType(DALIDataType input_type) {
    DALI_ENFORCE(dtype_ == input_type || dtype_ == DALI_FLOAT,
      make_string("CoordTransform output type must be the same as input type, which is ",
                  input_type, ", or `float`. Got: ", dtype_));
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    auto &in = ws.template InputRef<Backend>(0);
    auto &out = ws.template OutputRef<Backend>(0);
    out.SetLayout(in.GetLayout());

    if (out.shape().num_elements() == 0)
      return;

    DALIDataType in_type = in.type().id();
    VALUE_SWITCH(output_pt_dim_, static_out_dim, COORD_TRANSFORM_DIMS, (
        VALUE_SWITCH(input_pt_dim_, static_in_dim, COORD_TRANSFORM_DIMS, (
            TYPE_SWITCH(in_type, type2id, InputType, COORD_TRANSFORM_INPUT_TYPES, (
                if (dtype_ == in_type) {
                  RunTyped<InputType, InputType, static_out_dim, static_in_dim>(ws);
                } else {
                  CheckType(in_type);
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
        DALI_FAIL(make_string("Unsupported output point dimensionality: ", input_pt_dim_));
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

    SetTransformDims(input_pt_dim_);
    ProcessTransformArgs(spec_, ws, N);
  }

 private:
  template <typename OutputType, typename InputType, int out_dim, int in_dim>
  void RunTyped(workspace_t<Backend> &ws);

  DALIDataType dtype_;

  kernels::KernelManager kmgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GEOMETRY_COORD_TRANSFORM_H_
