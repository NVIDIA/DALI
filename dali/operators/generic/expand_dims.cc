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
#include <set>

#include "dali/core/math_util.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/operators/generic/expand_dims.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(ExpandDims)
    .DocStr(R"code(Insert new dimension(s) with extent 1 to the data shape.
The new dimensions are inserted at the positions specified by ``axes``.
  
If ``new_axis_names`` is provided, the new dimension names will be inserted in the data layout, 
at the positions specified by ``axes``. If ``new_axis_names`` is not provided, the output data 
layout will be empty.")code")
  .NumInput(1)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be expanded")
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<int>("axes", R"code(Indices at which the new dimensions are inserted.)code",
    std::vector<int>(), true)
  .AddOptionalArg("new_axis_names", R"code(Names of the new dimensions in the data layout.
  
The length of ``new_axis_names`` must match the length of ``axes``.
If argument won't be provided layout will be cleared.)code", TensorLayout(""));

template <typename Backend>
ExpandDims<Backend>::ExpandDims(const OpSpec &spec)
    : Reshape<Backend>(spec, typename Reshape<Backend>::BypassInit()) {
  axes_ = spec.GetRepeatedArgument<int>("axes");
  DALI_ENFORCE(spec.HasArgument("axes"), make_string("``axes`` argument should be provided."));
  for (auto axis : axes_) {
    DALI_ENFORCE(0 <= axis, make_string("Axis value can't be negative"));
  }
  std::sort(axes_.begin(), axes_.end());
  DALI_ENFORCE(std::adjacent_find(axes_.begin(), axes_.end()) == axes_.end(),
    make_string("Specified at least twice same index to add new dimension."));

  use_new_axis_names_arg_ = spec.HasArgument("new_axis_names");
  new_axis_names_ = spec.GetArgument<TensorLayout>("new_axis_names");
  if (!new_axis_names_.empty()) {
    DALI_ENFORCE(new_axis_names_.size() == axes_.size(), make_string("Specified ", axes_.size(),
      " new dimensions, but layout contains only ",
      new_axis_names_.size(), " new dimension names"));
  }
  this->use_src_dims_ = true;
}

template <typename Backend>
bool ExpandDims<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
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
void ExpandDims<Backend>::GenerateSrcDims(const Workspace &ws) {
  auto &in = ws.template InputRef<Backend>(0);
  const auto &input_shape = in.shape();
  int ndim = input_shape.sample_dim();
  auto in_layout = in.GetLayout();
  if (in_layout.empty() && ndim) {
    DALI_ENFORCE(!use_new_axis_names_arg_,
      make_string("Specifying ``new_axis_names`` requires an input with a proper layuout."));
  }
  DALI_ENFORCE(in_layout.size() == ndim || in_layout.empty(),
    make_string("Layout for data has size ",
    in_layout.size(), " but data has ", ndim, " dimensions."));

  this->src_dims_.clear();
  TensorLayout out_layout;
  size_t axes_ind = 0;
  int out_ndim = ndim + axes_.size();
  for (int i = 0, d = 0; i < out_ndim; i++) {
    if (axes_ind < axes_.size() && axes_[axes_ind] == i) {
      this->src_dims_.push_back(-1);
      out_layout += use_new_axis_names_arg_ ? new_axis_names_[axes_ind] : 0;
      axes_ind++;
      continue;
    }

    DALI_ENFORCE(d < ndim,
      make_string("Data has not enough dimensions to add new axes at specified indices."));
    out_layout += in_layout.empty() ? 0 : in_layout[d];
    this->src_dims_.push_back(d++);
  }
  if (!in_layout.empty()) {
    this->layout_ = use_new_axis_names_arg_ ? out_layout : TensorLayout();
  }
}

DALI_REGISTER_OPERATOR(ExpandDims, ExpandDims<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ExpandDims, ExpandDims<GPUBackend>, GPU);

}  // namespace namespace dali
