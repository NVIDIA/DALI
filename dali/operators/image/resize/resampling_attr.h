// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESAMPLING_ATTR_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESAMPLING_ATTR_H_

#include "dali/core/common.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/operators/image/resize/resize_attr.h"

namespace dali {

class DLL_PUBLIC ResamplingFilterAttr {
 public:
  /**
   * @brief Initializes min/mag filters
   *
   * @param spec OpSpec for the operator
   * @param ws   ArgumentWorkspace, used when filtering methods are set per-sample
   * @param num_samples number of samples; if not specified, value of "batch_size" argument is used
   */
  DLL_PUBLIC void Setup(const OpSpec &spec, const ArgumentWorkspace &ws, int num_samples = -1);

  /**
   * Filter used when downscaling
   */
  std::vector<kernels::ResamplingFilterType> min_filter_;
  /**
   * Filter used when upscaling
   */
  std::vector<kernels::ResamplingFilterType> mag_filter_;
  /**
   * Initial size, in bytes, of a temporary buffer for resampling.
   */
  size_t temp_buffer_hint_ = 0;

  void GetResamplingParams(span<kernels::ResamplingFilterParams> resample_params,
                           span<const ResizeParams> resize_params);

  template <size_t N>
  void GetResamplingParams(span<kernels::ResamplingFilterParamsND<N>> resample_params,
                           span<const ResizeParams> resize_params) {
    GetResamplingParams(flatten(resample_params), resize_params);
  }

 private:
  std::vector<DALIInterpType> interp_type_arg_, min_arg_, mag_arg_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESAMPLING_ATTR_H_
