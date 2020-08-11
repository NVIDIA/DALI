// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/image/resize/resampling_attr.h"

namespace dali {

template <typename Backend>
class DLL_PUBLIC ResizeBase {
 public:
  explicit ResizeBase(const OpSpec &spec);
  ~ResizeBase();

  void InitializeCPU(int num_threads);
  void InitializeGPU(int minibatch_size, size_t temp_buffer_hint = 0);

  using Workspace = workspace_t<Backend>;

  using InputBufferType =  typename Workspace::template input_t<Backend>::element_type;
  using OutputBufferType = typename Workspace::template output_t<Backend>::element_type;

  void RunResize(Workspace &ws, OutputBufferType &output, const InputBufferType &input);

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
   */
  void SetupResize(TensorListShape<> &out_shape,
                   DALIDataType out_type,
                   const TensorListShape<> &in_shape,
                   DALIDataType in_type,
                   span<const kernels::ResamplingParams> params,
                   int spatial_ndim,
                   int first_spatial_dim = 0);

  template <size_t spatial_ndim>
  void SetupResize(TensorListShape<> &out_shape,
                   DALIDataType out_type,
                   const TensorListShape<> &in_shape,
                   DALIDataType in_type,
                   span<const kernels::ResamplingParamsND<spatial_ndim>> params,
                   int first_spatial_dim = 0) {
    SetupResize(out_shape, out_type, in_shape, in_type, flatten(params),
                spatial_ndim, first_spatial_dim);
  }

  struct Impl;  // this needs to be public, because implementations inherit from it
 private:
  template <typename OutType, typename InType, int spatial_ndim>
  void SetupResizeStatic(TensorListShape<> &out_shape,
                         const TensorListShape<> &in_shape,
                         span<const kernels::ResamplingParams> params,
                         int first_spatial_dim = 0);

  template <typename OutType, typename InType>
  void SetupResizeTyped(TensorListShape<> &out_shape,
                        const TensorListShape<> &in_shape,
                        span<const kernels::ResamplingParams> params,
                        int spatial_ndim,
                        int first_spatial_dim = 0);

  int num_threads_ = 1;
  int minibatch_size_ = 32;
  std::unique_ptr<Impl> impl_;
  kernels::KernelManager kmgr_;
};

template <>
void ResizeBase<CPUBackend>::InitializeCPU(int num_threads);

template <>
void ResizeBase<GPUBackend>::InitializeGPU(int minibatch_size, size_t temp_buffer_hint);

extern template class ResizeBase<CPUBackend>;
extern template class ResizeBase<GPUBackend>;

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_BASE_H_
