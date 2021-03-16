// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/math_util.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/operators/generic/squeeze.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Squeeze)
  .DocStr(R"code(Removes the dimensions given as ``axes`` or ``axis_names``.

It's an error to remove a dimension that would cause the total volume to change.)code")
  .NumInput(1)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be squeezed")
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<int>("axes", R"code(Indices of dimensions which should be removed.

All squeezed dimensions should have size 1, unless the total volume of the tensor is 0 before and after squeeze.
All indices must be in the range of valid dimensions of the input)code", std::vector<int>(), true)
  .AddOptionalArg("axis_names", R"code(Layout columns which should be removed.
  
All squeezed dimensions should have size 1, unless the total volume of the tensor is 0 before and after squeeze.
All layout names should be present in data layout.)code", TensorLayout(""));

template <typename Backend>
Squeeze<Backend>::Squeeze(const OpSpec &spec)
    : Reshape<Backend>(spec, typename Reshape<Backend>::BypassInit()) {
  axes_ = spec.GetRepeatedArgument<int>("axes");
  axis_names_ = spec.GetArgument<TensorLayout>("axis_names");

    DALI_ENFORCE(spec.HasArgument("axes") + spec.HasArgument("axis_names") == 1,
      spec.HasArgument("axes") ? "Provided both ``axes`` and ``axis_names`` arguments"
                               : "Missing argument ``axes`` or ``axis_names``.");

  this->use_src_dims_ = true;
}

template <typename Backend>
bool Squeeze<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);
  this->SetOutputType(ws);

  GenerateSrcDims(ws);
  this->CalculateOutputShape(ws);

  output_desc[0].type = *(this->output_type_);
  output_desc[0].shape = this->output_shape_;
  // return false, because we don't want the executor to allocate anything
  // - this operator returns pointer to input memory
  return false;
}

template <typename Backend>
void Squeeze<Backend>::GenerateSrcDims(const Workspace &ws) {
  auto &in = ws.template InputRef<Backend>(0);
  const auto &input_shape = in.shape();
  const int ndim = input_shape.sample_dim();
  auto in_layout = in.GetLayout();

  this->src_dims_.clear();
  auto axes = axis_names_.empty() ? axes_ : GetDimIndices(in_layout, axis_names_);
  std::sort(axes.begin(), axes.end());
  DALI_ENFORCE(std::adjacent_find(axes.begin(), axes.end()) == axes.end(),
    make_string("Specified at least twice same dimension to remove."));
  TensorLayout out_layout;
  size_t axis_ind = 0;
  for (int d = 0; d < ndim; d++) {
    if (axis_ind < axes.size() && axes[axis_ind] == d) {
      axis_ind++;
      continue;
    }

    this->src_dims_.push_back(d);
    if (!in_layout.empty()) {
      out_layout += in_layout[d];
    }
  }
  this->layout_ = out_layout;
}

DALI_REGISTER_OPERATOR(Squeeze, Squeeze<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Squeeze, Squeeze<GPUBackend>, GPU);

}  // namespace namespace dali
