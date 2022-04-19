// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

AxisArgs::AxisArgs(const OpSpec &spec, const char *axis_index_arg, const char *axis_name_arg,
                   unsigned int flags)
    : flags_(flags) {
  bool has_axis_index_arg = axis_index_arg && spec.ArgumentDefined(axis_index_arg);
  bool has_axis_name_arg = axis_name_arg && spec.HasArgument(axis_name_arg);
  if (has_axis_index_arg && has_axis_name_arg) {
    DALI_FAIL(
        make_string("\"", axis_index_arg, "\" and \"", axis_name_arg, "\" are mutually exclusive"));
  }

  if (axis_index_arg)  // unique_ptr serves as optional
    axes_ = std::make_unique<ArgValue<int, 1>>(axis_index_arg, spec);

  per_sample_axes_ = axis_index_arg && spec.HasTensorArgument(axis_index_arg);

  bool allow_empty = flags_ & AllowEmpty;
  if (!allow_empty && !axis_index_arg && !axis_name_arg) {
    DALI_FAIL("At least one argument name must be provided if allow_empty is false.");
  }

  if (!per_sample_axes_) {
    if ((has_axis_name_arg || !has_axis_index_arg) && axis_name_arg) {
      use_axis_names_ = spec.TryGetArgument(axis_names_, axis_name_arg);
      if (use_axis_names_ && !allow_empty && axis_names_.empty())
        DALI_FAIL(make_string("Can't have empty axes. Check argument name: ", axis_name_arg));
    }
    if (!use_axis_names_ && axis_index_arg) {
      spec.TryGetRepeatedArgument(const_axes_, axis_index_arg);

      if (!allow_empty && const_axes_.empty())
        DALI_FAIL(make_string("Can't have empty axes. Check argument name: ", axis_index_arg));
    }
  }
}

void AxisArgs::Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples, int ndim) {
  if (per_sample_axes_) {
    assert(axes_);
    axes_->Acquire(spec, ws, nsamples);
    shape_ = axes_->get().shape;
  } else if (use_axis_names_) {
    shape_ = uniform_list_shape(nsamples, TensorShape<1>(axis_names_.size()));
  } else {
    shape_ = uniform_list_shape(nsamples, TensorShape<1>(const_axes_.size()));
  }

  if (flags_ & AllIfEmpty) {
    TensorShape<1> sh(ndim);
    if (shape_.num_elements() == 0) {
      shape_ = uniform_list_shape(nsamples, sh);
    } else {
      for (int i = 0; i < shape_.size(); i++) {
        if (volume(shape_.tensor_shape_span(i)) == 0)
          shape_.set_tensor_shape(i, sh);
      }
    }
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
