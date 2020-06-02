// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_ERASE_ERASE_UTILS_H_
#define DALI_OPERATORS_GENERIC_ERASE_ERASE_UTILS_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/erase/erase_args.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace detail {

static SmallVector<int, 6> GetAxes(const OpSpec &spec, TensorLayout layout) {
  SmallVector<int, 6> axes;
  if (spec.HasArgument("axis_names")) {
    axes = GetDimIndices(layout, spec.GetArgument<TensorLayout>("axis_names"));
  } else if (spec.HasArgument("axes")) {
    axes = spec.GetRepeatedArgument<int>("axes");
  } else {
    // no axes info, expecting all dimensions except 'C'
    for (int d = 0; d < layout.size(); d++) {
      if (layout[d] == 'C')
        continue;
      axes.push_back(d);
    }
  }
  return axes;
}

template <typename T, int Dims>
std::vector<kernels::EraseArgs<T, Dims>> GetEraseArgs(const OpSpec &spec,
                                                      const ArgumentWorkspace &ws,
                                                      TensorListShape<> in_shape,
                                                      TensorLayout in_layout) {
  int nsamples = in_shape.num_samples();

  std::vector<float> roi_anchor;
  bool has_tensor_roi_anchor = spec.HasTensorArgument("anchor");
  if (!has_tensor_roi_anchor)
    roi_anchor = spec.template GetArgument<std::vector<float>>("anchor");
  auto norm_anchor = spec.template GetArgument<bool>("normalized_anchor");

  std::vector<float> roi_shape;
  bool has_tensor_roi_shape = spec.HasTensorArgument("shape");
  if (!has_tensor_roi_shape)
    roi_shape = spec.template GetArgument<std::vector<float>>("shape");
  auto norm_shape = spec.template GetArgument<bool>("normalized_shape");

  if (spec.HasArgument("normalized")) {
    DALI_ENFORCE(!spec.HasArgument("normalized_anchor") && !spec.HasArgument("normalized_shape"),
      "`normalized` argument is incompatible with providing a separate value for "
      "`normalized_anchor` and `normalized_shape`");
    norm_anchor = spec.GetArgument<bool>("normalized");
    norm_shape = norm_anchor;
  }

  if (roi_anchor.empty() && !roi_shape.empty()) {
    roi_anchor.resize(roi_shape.size(), 0);
  }

  bool centered_anchor = spec.GetArgument<bool>("centered_anchor");

  auto fill_value = spec.template GetRepeatedArgument<float>("fill_value");
  auto channels_dim = in_layout.find('C');
  DALI_ENFORCE(channels_dim >= 0 || fill_value.size() <= 1,
    "If a multi channel fill value is provided, the input layout must have a 'C' dimension");

  auto axes = detail::GetAxes(spec, in_layout);
  int naxes = axes.size();
  assert(naxes > 0);

  std::vector<kernels::EraseArgs<T, Dims>> out;
  out.resize(nsamples);

  for (int i = 0; i < nsamples; i++) {
    if (has_tensor_roi_anchor) {
      const auto& anchor = ws.ArgumentInput("anchor")[i];
      assert(anchor.size() > 0);
      roi_anchor.resize(anchor.size());
      std::memcpy(roi_anchor.data(), anchor.data<float>(), sizeof(float) * roi_anchor.size());
    }

    if (has_tensor_roi_shape) {
      const auto& shape = ws.ArgumentInput("shape")[i];
      assert(shape.size() > 0);
      roi_shape.resize(shape.size());
      std::memcpy(roi_shape.data(), shape.data<float>(), sizeof(float) * roi_shape.size());
    }

    DALI_ENFORCE(roi_anchor.size() == roi_shape.size());
    int args_len = roi_shape.size();
    DALI_ENFORCE(args_len % naxes == 0);
    int nregions = args_len / naxes;

    auto &args = out[i];
    auto sample_shape = in_shape.tensor_shape(i);
    int k = 0;
    args.rois.reserve(nregions);
    for (int roi_idx = 0; roi_idx < nregions; roi_idx++) {
      typename kernels::EraseArgs<T, Dims>::ROI roi;
      roi.fill_values = fill_value;
      roi.channels_dim = channels_dim;
      for (int d = 0; d < Dims; d++) {
        roi.anchor[d] = 0;
        roi.shape[d] = sample_shape[d];
      }

      for (int j=0; j < naxes; j++, k++) {
        int axis = axes[j];
        auto anchor_val = norm_anchor ? roi_anchor[k] * sample_shape[axis] : roi_anchor[k];
        auto shape_val = norm_shape ? roi_shape[k] * sample_shape[axis] : roi_shape[k];
        if (centered_anchor) {
          anchor_val -= shape_val / 2;
        }
        roi.anchor[axis] = static_cast<int64_t>(anchor_val);

        auto end_val = static_cast<int64_t>(anchor_val + shape_val);
        roi.shape[axis] = end_val - roi.anchor[axis];
      }
      args.rois.push_back(roi);
    }
  }
  return out;
}

}  // namespace detail
}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_ERASE_ERASE_UTILS_H_
