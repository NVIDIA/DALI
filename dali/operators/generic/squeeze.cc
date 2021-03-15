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
#include "dali/operators/generic/squeeze.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Squeeze)
  .DocStr("Collapses the dimensions given as axes or axis_names. "
    "It's an error to collapse dims that would cause the volume to change "
    "(we can collapse a non-unit dim if a non-collapsed dim is 0).")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be squeezed")
  .InputDox(1, "shape_input", "1D TensorList of integers", "Same as `shape` keyword argument")
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<int>("axes", "", std::vector<int>(), true)
  .AddOptionalArg("axis_names", "", TensorLayout(""));

template <typename Backend>
Squeeze<Backend>::Squeeze(const OpSpec &spec)
    : Reshape<Backend>(spec) {
  axes_ = spec.GetRepeatedArgument<int>("axes");
  std::sort(axes_.begin(), axes_.end());
  axes_.erase(std::unique(axes_.begin(), axes_.end()), axes_.end());

  axis_names_ = spec.GetArgument<TensorLayout>("axis_names");
}

template <typename Backend>
bool Squeeze<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
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
void Squeeze<Backend>::GenerateSrcDims(const Workspace &ws) {
  this->use_src_dims_ = true;
  auto &in = ws.template InputRef<Backend>(0);
  const auto &input_shape = in.shape();
  const int ndim = input_shape.sample_dim();
  auto in_layout = in.GetLayout();
  DALI_ENFORCE(in_layout.size() == ndim || in_layout.empty(),
      make_string("Layout for data has size ",
      in_layout.size(), " but data has ", ndim, " dimensions."));

  this->src_dims_.clear();
  std::string out_layout;
  if (axes_.empty()) {
    std::set<char> axis_names_set(axis_names_.begin(), axis_names_.end());
    std::set<char> in_layout_set(in_layout.begin(), in_layout.end());
    for (auto axes : axis_names_set) {
      DALI_ENFORCE(in_layout_set.count(axes),
        make_string("Provided ", axes, " in ``axis_names``, but it's not present in data layout"));
    }

    for (size_t i = 0; i < in_layout.size(); i++) {
      auto layout_let = in_layout[i];
      if (axis_names_set.count(layout_let)) {
        continue;
      }

      out_layout.push_back(layout_let);
      this->src_dims_.push_back(i);
    }
  } else {
    for (auto axis : axes_) {
      DALI_ENFORCE(axis < ndim,
        make_string("Dimension passed in axes is out of the bounds"));
    }

    size_t axis_ind = 0;
    for (int d = 0; d < ndim; d++) {
      if (axis_ind < axes_.size() && axes_[axis_ind] == d) {
        axis_ind++;
        continue;
      }

      this->src_dims_.push_back(d);
      if (!in_layout.empty()) {
        out_layout.push_back(in_layout[d]);
      }
    }
  }
  this->layout_ = out_layout;
}

DALI_REGISTER_OPERATOR(Squeeze, Squeeze<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Squeeze, Squeeze<GPUBackend>, GPU);

}  // namespace namespace dali
