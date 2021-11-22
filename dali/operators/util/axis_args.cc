// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/util/axis_args.h"
#include <vector>

namespace dali {

AxisArgs::AxisArgs(const OpSpec &spec, const char *axes_arg, const char *axis_names_arg,
                   unsigned int flags)
    : flags_(flags) {
  bool has_axes_arg = axes_arg && spec.ArgumentDefined(axes_arg);
  bool has_axis_names_arg = axis_names_arg && spec.HasArgument(axis_names_arg);
  if (has_axes_arg && has_axis_names_arg) {
    DALI_FAIL(
        make_string("\"", axes_arg, "\" and \"", axis_names_arg, "\" are mutually exclusive"));
  }

  if (axes_arg)  // unique_ptr serves as optional
    axes_ = std::make_unique<ArgValue<int, 1>>(axes_arg, spec);

  per_sample_axes_ = axes_arg && spec.HasTensorArgument(axes_arg);

  bool allow_empty = flags_ & AllowEmpty;
  if (!per_sample_axes_) {
    if ((has_axis_names_arg || !has_axes_arg) && axis_names_arg) {
      use_axis_names_ = spec.TryGetArgument(axis_names_, axis_names_arg);
      if (use_axis_names_ && !allow_empty && axis_names_.empty())
        DALI_FAIL(make_string("Can't have empty axes. Check argument name: ", axis_names_arg));
    }
    if (!use_axis_names_ && axes_arg) {
      std::vector<int> tmp;  // TODO(janton): support SmallVector in TryGetRepeatedArgument
      if (spec.TryGetRepeatedArgument(tmp, axes_arg))
        const_axes_ = {tmp.begin(), tmp.end()};

      if (!allow_empty && const_axes_.empty())
        DALI_FAIL(make_string("Can't have empty axes. Check argument name: ", axes_arg));
    }
  }
}

void AxisArgs::Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples) {
  if (per_sample_axes_) {
    assert(axes_);
    axes_->Acquire(spec, ws, nsamples);
    shape_ = axes_->get().shape;
  } else if (use_axis_names_) {
    shape_ = uniform_list_shape(nsamples, TensorShape<1>(axis_names_.size()));
  } else {
    shape_ = uniform_list_shape(nsamples, TensorShape<1>(const_axes_.size()));
  }
}

TensorListShape<1> AxisArgs::AxesShape() {
  return shape_;
}

SmallVector<int, 6> AxisArgs::Get(int data_idx, int ndim, const TensorLayout &layout) {
  SmallVector<int, 6> axes;
  if (per_sample_axes_) {
    assert(axes_);
    auto view = (*axes_)[data_idx];
    int n = view.shape.num_elements();
    axes.resize(n);
    for (int i = 0; i < n; i++)
      axes[i] = view.data[i];
  } else if (use_axis_names_) {
    axes = GetDimIndices(layout, axis_names_);
  } else {
    axes = const_axes_;
  }
  Process(ndim, axes);
  return axes;
}

void AxisArgs::Process(int ndim, SmallVector<int, 6> &axes) {
  if (axes.empty()) {
    if (!(flags_ & AllowEmpty))
      DALI_FAIL("Need to specify at least one axis");
    if (flags_ & AllIfEmpty) {
      axes.resize(ndim);
      std::iota(axes.begin(), axes.end(), 0);
    }
  }

  if (flags_ & AllowNegative) {
    for (auto &axis : axes) {
      DALI_ENFORCE(axis >= -ndim && axis < ndim,
                   make_string("Axis ", axis, " out of range. Expected range is [", -ndim, ", ",
                               ndim - 1, "] for a ", ndim, "D input"));
      if (axis < 0)
        axis += ndim;
    }
  } else {
    for (auto &axis : axes) {
      DALI_ENFORCE(axis >= 0 && axis < ndim,
                   make_string("Axis ", axis, " out of range. Expected range is [0, ", ndim - 1,
                               "] for a ", ndim, "D input"));
    }
  }

  if (!(flags_ & AllowEmpty) && axes.empty())
    DALI_FAIL("Need to specify at least one axis");


  SmallVector<bool, 6> axes_check;
  for (auto axis : axes) {
    if (axes_check[axis])
      DALI_FAIL(make_string("Axis index ", axis,
                            " occurs more than once in ``axes`` "
                            "(might include negative indices referring to the same axis"));
    axes_check[axis] = true;
  }
}

}  // namespace dali
