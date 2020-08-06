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

#include <vector>
#include "dali/core/common.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/operators/image/resize/resize_attr.h"

namespace dali {

/**
 * @brief Handles operator arguments shared by operators using separable resampling kernel.
 */
class DLL_PUBLIC ResamplingFilterAttr {
 public:
  /**
   * @brief Initializes min/mag filters
   *
   * @param spec OpSpec for the operator
   * @param ws   ArgumentWorkspace, used when filtering methods are set per-sample
   * @param num_samples number of samples in the batch
   */
  void PrepareFilterParams(const OpSpec &spec, const ArgumentWorkspace &ws, int num_samples);

  /**
   * @brief Returns the value of `dtype` argument, if present, or input_type otherwise.
   */
  DALIDataType GetOutputType(DALIDataType input_type) const {
    return dtype_arg_ != DALI_NO_TYPE ? dtype_arg_ : input_type;
  }

  /**
   * Filter used when downscaling
   */
  std::vector<kernels::ResamplingFilterType> min_filter_;
  /**
   * Filter used when upscaling
   */
  std::vector<kernels::ResamplingFilterType> mag_filter_;

  /**
   * @brief Patches existing ResamplingParams by adding filter properties.
   *
   * There's no need to call this function on parameters obtained from GetResamplingParams.
   * It's intended to work with externally constructed ResamplingParams, which have the
   * geometric information in place, but are missing the resampling filter info.
   */
  void ApplyFilterParams(span<kernels::ResamplingParams> resample_params, int ndim) const;

  /**
   * @brief Patches existing ResamplingParams by adding filter properties.
   *
   * There's no need to call this function on parameters obtained from GetResamplingParams.
   * It's intended to work with externally constructed ResamplingParams, which have the
   * geometric information in place, but are missing the resampling filter info.
   */
  template <size_t N>
  void ApplyFilterParams(span<kernels::ResamplingParamsND<N>> resample_params) const {
    ApplyFilterParams(flatten(resample_params), N);
  }

  /**
   * @brief Constructs ResamplingParams from ResizeParams
   */
  void GetResamplingParams(span<kernels::ResamplingParams> resample_params,
                           span<const ResizeParams> resize_params) const;

  /**
   * @brief Constructs ResamplingParams from ResizeParams
   */
  template <size_t N>
  void GetResamplingParams(span<kernels::ResamplingParamsND<N>> resample_params,
                           span<const ResizeParams> resize_params) const {
    GetResamplingParams(flatten(resample_params), resize_params);
  }

 private:
  std::vector<DALIInterpType> interp_type_arg_, min_arg_, mag_arg_;
  DALIDataType dtype_arg_ = DALI_NO_TYPE;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESAMPLING_ATTR_H_
