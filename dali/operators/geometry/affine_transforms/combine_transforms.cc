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

#include <string>
#include <utility>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/mat.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/operator/operator.h"

#define TRANSFORM_INPUT_TYPES (float)

namespace dali {

template <int... values>
using dims = std::integer_sequence<int, values...>;

template <typename T, int mat_dim>
using affine_mat_t = mat<mat_dim, mat_dim, T>;

DALI_SCHEMA(transforms__Combine)
  .DocStr(R"code(Combines two or more affine transforms.)code")
  .NumInput(2, 99)
  .NumOutput(1)
  .AddParent("TransformAttr");

class CombineTransformsCPU : public Operator<CPUBackend> {
 public:
  explicit CombineTransformsCPU(const OpSpec &spec) :
      Operator<CPUBackend>(spec),
      reverse_order_(spec.GetArgument<bool>("reverse_order")) {
  }

  bool CanInferOutputs() const override { return true; }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_descs,
                 const workspace_t<CPUBackend> &ws) override {
    assert(ws.NumInput() > 1);
    TensorListShape<> in0_shape = ws.template InputRef<CPUBackend>(0).shape();
    ndim_ = in0_shape[0][0];
    nsamples_ = in0_shape.size();

    DALI_ENFORCE(in0_shape.sample_dim() == 2 &&
                 in0_shape.size() > 0 &&
                 in0_shape[0][1] == (in0_shape[0][0] + 1),
      make_string(
        "The input, if provided, is expected to be a 2D tensor with dimensions "
        "(ndim, ndim+1) representing an affine transform. Got: ", in0_shape));

    for (int i = 0; i < ws.NumInput(); i++) {
      const auto &shape = ws.template InputRef<CPUBackend>(i).shape();
      DALI_ENFORCE(shape == in0_shape,
        make_string("All input transforms are expected to have the same shape. Got: ",
                    in0_shape, " and ", shape, " for the ", i, "-th input."));
    }

    output_descs.resize(1);  // only one output
    output_descs[0].type = TypeTable::GetTypeInfo(dtype_);
    output_descs[0].shape = uniform_list_shape(nsamples_, {ndim_, ndim_+1});
    return true;
  }

  template <typename T>
  void RunImplTyped(workspace_t<CPUBackend> &ws, dims<>) {
    DALI_FAIL(make_string("Unsupported number of dimensions ", ndim_));
  }

  template <typename T, int ndim, int... ndims>
  void RunImplTyped(workspace_t<CPUBackend> &ws, dims<ndim, ndims...>) {
    if (ndim_ != ndim) {
      RunImplTyped<T>(ws, dims<ndims...>());
      return;
    }

    constexpr int mat_dim = ndim + 1;
    auto &out = ws.template OutputRef<CPUBackend>(0);
    out.SetLayout({});  // no layout
    auto out_view = view<T>(out);

    for (int sample_idx = 0; sample_idx < nsamples_; sample_idx++) {
      auto mat = affine_mat_t<T, mat_dim>::identity();
      for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
        auto &in = ws.template InputRef<CPUBackend>(input_idx);
        auto in_view = view<T>(in);
        auto next_mat = affine_mat_t<T, mat_dim>::identity();
        for (int i = 0, k = 0; i < ndim; i++)
          for (int j = 0; j < ndim + 1; j++, k++)
            next_mat(i, j) = in_view[sample_idx].data[k];
        mat = reverse_order_ ? mat * next_mat : next_mat * mat;  // mat mul
      }

      for (int i = 0, k = 0; i < ndim; i++) {
        for (int j = 0; j < ndim + 1; j++, k++) {
          out_view[sample_idx].data[k] = mat(i, j);
        }
      }
    }
  }

  void RunImpl(workspace_t<CPUBackend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, TRANSFORM_INPUT_TYPES, (
      RunImplTyped<T>(ws, SupportedDims());
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

 private:
  using SupportedDims = dims<1, 2, 3, 4, 5, 6>;
  DALIDataType dtype_ = DALI_FLOAT;
  int ndim_ = -1;  // will be inferred from the arguments or the input
  int nsamples_ = -1;
  bool reverse_order_ = false;
};

DALI_REGISTER_OPERATOR(transforms__Combine, CombineTransformsCPU, CPU);

}  // namespace dali
