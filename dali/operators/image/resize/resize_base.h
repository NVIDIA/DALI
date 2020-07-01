// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_BASE_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_BASE_H_

#include <memory>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/span.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

class DLL_PUBLIC ResamplingFilterAttr {
 public:
  DLL_PUBLIC ResamplingFilterAttr(const OpSpec &spec);

  /**
   * Filter used when downscaling
   */
  kernels::FilterDesc min_filter_{ kernels::ResamplingFilterType::Triangular, 0 };
  /**
   * Filter used when upscaling
   */
  kernels::FilterDesc mag_filter_{ kernels::ResamplingFilterType::Linear, 0 };
  /**
   * Initial size, in bytes, of a temporary buffer for resampling.
   */
  size_t temp_buffer_hint_ = 0;
};

template <typename Backend>
class DLL_PUBLIC ResizeBase : public ResamplingFilterAttr {
 public:
  explicit ResizeBase(const OpSpec &spec);
  ~ResizeBase();

  void InitializeCPU(int num_threads);
  void InitializeGPU(int minibatch_size);

  using Workspace = workspace_t<Backend>;

  using input_type =  typename Workspace::template input_t<Backend>::element_type;
  using output_type = typename Workspace::template output_t<Backend>::element_type;

  void RunResize(Workspace &ws, output_type &output, const input_type &input);

  /**
   * @param ws                workspace object
   * @param out_shape         output shape, determined by params
   * @param input             input; data is not accessed, only shape and metadata are relevant
   * @param params            resampling parameters; this is a flattened array of size
   *                          `spatial_ndim*num_samples`, each sample is described by spatial_ndim
   *                          ResamplingParams, starting from outermost spatial dimension
   *                          (i.e. [depthwise,] vertical, horizontal)
   * @param out_type          desired output type
   * @param first_spatial_dim index of the first resized dim
   * @param spatial_ndim      number of resized dimensions - these need to form a
   *                          contiguous block in th layout
   *
   * @return number of spatial dimensions
   */
  int SetupResize(const Workspace &ws,
                  TensorListShape<> &out_shape,
                  const input_type &input,
                  span<const kernels::ResamplingParams> params,
                  DALIDataType out_type,
                  int first_spatial_dim,
                  int spatial_ndim);

 private:
  DALIDataType output_type_;
  int minibatch_size_ = 32;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};


extern template class ResizeBase<CPUBackend>;
extern template class ResizeBase<GPUBackend>;

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_BASE_H_
