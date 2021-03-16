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
  .DocStr(R"code(Insert new dimension[s] of extent 1 and inserts new entries in "
    "the layout (new_axis_names) at these indices in the layout.)code")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be expanded")
  .InputDox(1, "shape_input", "1D TensorList of integers", "Same as ``shape`` keyword argument")
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<int>("axes", R"code(Indices where to put new dimensions of size 1.)code",
    std::vector<int>(), true)
  .AddOptionalArg("new_axis_names", R"code(Names of new dimensions in data layout.
  
Size of ``new_axis_names`` should be equal to ``axe``s size. 
If argument won't be provided new dimensions will have layout '?')code", TensorLayout(""));

template <typename Backend>
ExpandDims<Backend>::ExpandDims(const OpSpec &spec)
    : Reshape<Backend>(spec) {
  axes_ = spec.GetRepeatedArgument<int>("axes");
  for (auto axis : axes_) {
    DALI_ENFORCE(0 <= axis, make_string("axis number can't be negative"));
  }

  std::sort(axes_.begin(), axes_.end());
  axes_.erase(std::unique(axes_.begin(), axes_.end()), axes_.end());
  DALI_ENFORCE(!axes_.empty(), make_string("Axes can't be empty"));

  new_axis_names_ = spec.GetArgument<TensorLayout>("new_axis_names");
  if (!new_axis_names_.empty()) {
    DALI_ENFORCE(new_axis_names_.size() == axes_.size(), make_string("Specified ", axes_.size(),
      " new dimensions, but layout specify ", new_axis_names_.size(), " new names"));
  }
}

template <typename Backend>
bool ExpandDims<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);
  this->SetOutputType(ws);

  GenerateSrcDims(ws);
  Reshape<Backend>::CalculateOutputShape(ws);

  output_desc[0].type = *(this->output_type_);
  output_desc[0].shape = this->output_shape_;
  // return false, because we don't want the executor to allocate anything
  // - this operator returns pointer to input memory
  return false;
}

template <typename Backend>
void ExpandDims<Backend>::GenerateSrcDims(const Workspace &ws) {
  this->use_src_dims_ = true;
  auto &in = ws.template InputRef<Backend>(0);
  const auto &input_shape = in.shape();
  int ndim = input_shape.sample_dim();
  auto in_layout = in.GetLayout();
  DALI_ENFORCE(in_layout.size() == ndim || in_layout.empty(),
    make_string("Layout for data has size ",
    in_layout.size(), " but data has ", ndim, " dimensions."));

  TensorLayout out_layout;
  size_t axes_ind = 0;
  this->src_dims_.clear();
  for (int d = 0; d < ndim; d++) {
    if (axes_[axes_ind] == d) {
      this->src_dims_.push_back(-1);
      out_layout += new_axis_names_.empty() ? '?' : new_axis_names_[axes_ind];
      axes_ind++;
    }
    out_layout += in_layout.empty() ? '?' : in_layout[d];
    this->src_dims_.push_back(d);
  }

  while (axes_ind < axes_.size()) {
    DALI_ENFORCE(axes_[axes_ind] <= ndim, make_string("ERROR"));
    this->src_dims_.push_back(-1);
    out_layout += new_axis_names_.empty() ? '?' : new_axis_names_[axes_ind];
    ndim++;
    axes_ind++;
  }
  this->layout_ = out_layout;
}

DALI_REGISTER_OPERATOR(ExpandDims, ExpandDims<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ExpandDims, ExpandDims<GPUBackend>, GPU);

}  // namespace namespace dali
