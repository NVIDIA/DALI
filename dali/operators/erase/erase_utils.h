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

#ifndef DALI_OPERATORS_ERASE_ERASE_UTILS_H_
#define DALI_OPERATORS_ERASE_ERASE_UTILS_H_

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

SmallVector<int, 3> GetAxes(const OpSpec &spec, TensorLayout layout) {
  SmallVector<int, 3> axes;
  if (spec.HasArgument("axis_names")) {
    for (auto axis_name : spec.GetArgument<TensorLayout>("axis_names")) {
      int d = layout.find(axis_name);
      DALI_ENFORCE(d >= 0);
      axes.push_back(d);
    }
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
                                                      TensorListShape<> shape,
                                                      TensorLayout layout) {
  int nsamples = shape.num_samples();
  auto roi_anchor = spec.template GetArgument<std::vector<float>>("anchor");
  auto roi_shape = spec.template GetArgument<std::vector<float>>("shape");
  if (roi_anchor.empty()) {
    roi_anchor.resize(roi_shape.size(), 0);
  }
  DALI_ENFORCE(roi_anchor.size() == roi_shape.size());
  int args_ndim = roi_shape.size();

  auto axes = detail::GetAxes(spec, layout);
  int naxes = axes.size();
  assert(naxes > 0);
  DALI_ENFORCE(args_ndim % naxes == 0);
  int nregions = args_ndim / naxes;

  std::vector<kernels::EraseArgs<T, Dims>> out;
  out.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    auto &args = out[i];
    int k = 0;
    args.rois.resize(nregions);
    for (auto &roi : args.rois) {
      auto sample_shape = shape.tensor_shape(i);
      for (int d = 0; d < Dims; d++) {
        roi.anchor[d] = 0;
        roi.shape[d] = sample_shape[d];
      }

      for (int j=0; j < naxes; j++, k++) {
        int axis = axes[j];
        roi.anchor[axis] = roi_anchor[k];
        roi.shape[axis] = roi_shape[k];
      }
    }
  }
  return out;
}

}  // namespace detail
}  // namespace dali

#endif  // DALI_OPERATORS_ERASE_ERASE_UTILS_H_
