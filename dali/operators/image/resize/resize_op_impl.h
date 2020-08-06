// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_H_

#ifndef DALI_RESIZE_BASE_CC
#error This file is a part of resize base implementation and should not be included elsewhere
#endif

#include <cassert>
#include <vector>
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/operators/image/resize/resize_base.h"

namespace dali {

using kernels::ResamplingParams;
using kernels::ResamplingParamsND;

template <typename Backend>
struct ResizeBase<Backend>::Impl {
  using InputBufferType  = typename Workspace::template input_t<Backend>::element_type;
  using OutputBufferType = typename Workspace::template output_t<Backend>::element_type;

  virtual void RunResize(workspace_t<Backend> &ws,
                         OutputBufferType &output,
                         const InputBufferType &input) = 0;

  virtual void Setup(TensorListShape<> &out_shape,
                     const TensorListShape<> &in_shape,
                     int first_spatial_dim,
                     span<const kernels::ResamplingParams> paramss) = 0;

  virtual ~Impl() = default;
};

template <typename Backend>
ResizeBase<Backend>::~ResizeBase() = default;

template <int spatial_ndim, int out_ndim, int in_ndim>
void GetFrameShapesAndParams(
      TensorListShape<out_ndim> &frame_shapes,
      std::vector<ResamplingParamsND<spatial_ndim>> &frame_params,
      const TensorListShape<in_ndim> &in_shape,
      const span<const ResamplingParams> &in_params,
      int first_spatial_dim) {
  assert(first_spatial_dim + spatial_ndim <= in_shape.sample_dim());
  const int frame_ndim = spatial_ndim + 1;
  static_assert(out_ndim == frame_ndim || out_ndim < 0, "Invalid frame tensor rank.");

  int N = in_shape.num_samples();
  int total_frames = 0;

  for (int i = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    total_frames += volume(&in_sample_shape[0], &in_sample_shape[first_spatial_dim]);
  }

  frame_params.resize(total_frames);
  frame_shapes.resize(total_frames, frame_ndim);

  int ndim = in_shape.sample_dim();
  for (int i = 0, flat_frame_idx = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    // Collapse leading dimensions, if any, as frame dim. This handles channel-first.
    int seq_len = volume(&in_sample_shape[0], &in_sample_shape[first_spatial_dim]);
    if (seq_len == 0)
      continue;  // skip empty sequences
    TensorShape<out_ndim> frame_shape;
    frame_shape.resize(frame_ndim);

    for (int d = first_spatial_dim, od = 0; od < spatial_ndim; d++, od++) {
      frame_shape[od] = in_sample_shape[d];
    }
    // Collapse trailing dimensions, if any, as channel dim.
    int num_channels = volume(&in_sample_shape[first_spatial_dim + spatial_ndim],
                              &in_sample_shape[ndim]);
    frame_shape[frame_ndim-1] = num_channels;

    // Replicate parameters and frame shape.
    for (int f = 0; f < seq_len; f++, flat_frame_idx++) {
      for (int d = 0; d < spatial_ndim; d++)
        frame_params[flat_frame_idx][d] = in_params[i * spatial_ndim + d];

      frame_shapes.set_tensor_shape(flat_frame_idx, frame_shape);
    }
  }
}

template <int out_ndim, int in_ndim>
void GetResizedShape(
      TensorListShape<out_ndim> &out_shape, const TensorListShape<in_ndim> &in_shape,
      span<const ResamplingParams> params, int spatial_ndim, int first_spatial_dim) {
  assert(params.size() == spatial_ndim * in_shape.num_samples());
  assert(first_spatial_dim >= 0 && first_spatial_dim + spatial_ndim <= in_shape.sample_dim());

  out_shape = in_shape;
  int N = out_shape.num_samples();
  for (int i = 0; i < N; i++) {
    auto out_sample_shape = out_shape.tensor_shape_span(i);
    for (int d = 0; d < spatial_ndim; d++) {
      auto out_extent = params[i * spatial_ndim + d].output_size;
      if (out_extent != dali::kernels::KeepOriginalSize)
        out_sample_shape[d + first_spatial_dim] = out_extent;
    }
  }
}

template <size_t spatial_ndim, int out_ndim, int in_ndim>
void GetResizedShape(
      TensorListShape<out_ndim> &out_shape, const TensorListShape<in_ndim> &in_shape,
      span<const ResamplingParamsND<spatial_ndim>> params, int first_spatial_dim) {
  GetResizedShape(out_shape, in_shape, flatten(params), spatial_ndim, first_spatial_dim);
}

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_H_
