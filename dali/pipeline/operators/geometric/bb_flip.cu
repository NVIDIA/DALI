// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <dali/pipeline/operators/geometric/bb_flip.cuh>
#include <dali/pipeline/operators/arg_helper.h>
#include <vector>

namespace dali {

/// @param output                - output bounding boxes
/// @param input                 - input bounding boxes
/// @param num_boxes             - number of bounding boxes in the input
/// @param sample_indices        - when using per-sample flip, contains sample indices for each
///                                bounding box in the input tensor list
/// @param per_sample_horizontal - per-sample flag indicating whether bounding boxes from
//                                 a given sample should be flipped horizontally; may by NULL
/// @param per_sample_vertical   - per-sample flag indicating whether bounding boxes from
//                                 a given sample should be flipped vertically; may be NULL
/// @param global_horizontal     - whether to flip horizontally; overriden by
///                                per_sample_horizontal, if specified
/// @param global_vertical       - whether to flip vertically; overriden by
///                                per_sample_vertical, if specified
template <bool ltrb>
__global__ void BbFlipKernel(float *output, const float *input, size_t num_boxes,
                             bool global_horizontal, const int *per_sample_horizontal,
                             bool global_vertical, const int *per_sample_vertical,
                             const int *sample_indices) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_boxes)
    return;

  bool h = per_sample_horizontal
         ? per_sample_horizontal[sample_indices[idx]]
         : global_horizontal;
  bool v = per_sample_vertical
         ? per_sample_vertical[sample_indices[idx]]
         : global_vertical;

  const auto *in = &input[4 * idx];
  auto *out = &output[4 * idx];
  if (ltrb) {
    out[0] = h ? 1.0f - in[2] : in[0];
    out[1] = v ? 1.0f - in[3] : in[1];
    out[2] = h ? 1.0f - in[0] : in[2];
    out[3] = v ? 1.0f - in[1] : in[3];
  } else {
    // No range checking required if the parenthesis is respected in the two lines below.
    // If the original bounding box satisfies the condition that x + w <= 1.0f, then the expression
    // 1.0f - (x + w) is guaranteed to yield a non-negative result. QED.
    out[0] = h ? 1.0f - (in[0] + in[2]) : in[0];
    out[1] = v ? 1.0f - (in[1] + in[3]) : in[1];
    out[2] = in[2];  // width and
    out[3] = in[3];  // height remain unaffected
  }
}


void BbFlip<GPUBackend>::RunImpl(Workspace<GPUBackend> *ws, int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto&output = ws->Output<GPUBackend>(idx);

  DALI_ENFORCE(IsType<float>(input.type()), "Expected input data as float;"
               " got " + input.type().name());
  DALI_ENFORCE(input.size() % 4 == 0,
               "Input data size must be a multiple of 4 if it contains bounding boxes;"
               " got " + std::to_string(input.size()));

  ArgValue<int> horz("horizontal", spec_, ws);
  ArgValue<int> vert("vertical", spec_, ws);
  bool ltrb = spec_.GetArgument<bool>("ltrb");

  auto stream = ws->stream();

  const auto num_boxes = input.size() / 4;

  const int *sample_idx = nullptr;
  const int *per_sample_horz = nullptr;
  const int *per_sample_vert = nullptr;

  // contains a map from box index to sample index - used
  // for accessing per-sample horz/vert arguments.
  Tensor<GPUBackend> sample_idx_tensor;

  if (horz.IsTensor() || vert.IsTensor()) {
    std::vector<int> indices;
    indices.reserve(num_boxes);

    // populate the index map
    auto shape = input.shape();
    for (size_t sample = 0; sample < shape.size(); sample++) {
      auto dim = shape[sample].size();

      DALI_ENFORCE(dim < 2 || shape[sample][dim-1] == 4,
                   "If bounding box tensor is >= 2D, innermost dimension must be 4");
      DALI_ENFORCE(dim > 1 || shape[sample][0] % 4 == 0,
                   "Flat representation of bouding boxes must have size divisible by 4");

      size_t sample_boxes = dim == 2 ? shape[sample][0] : shape[sample][0] / 4;
      for (size_t i = 0; i < sample_boxes ; i++) {
        indices.push_back(sample);
      }
    }
    sample_idx_tensor.Copy(indices, stream);

    if (horz.IsTensor())
      per_sample_horz = horz.AsGPU(stream)->data<int>();

    if (vert.IsTensor())
      per_sample_vert = vert.AsGPU(stream)->data<int>();

    sample_idx = sample_idx_tensor.data<int>();
  }

  output.ResizeLike(input);

  if (num_boxes == 0) {
    return;
  }

  const unsigned block = num_boxes < 1024 ? num_boxes : 1024;
  const unsigned grid = (num_boxes + block - 1) / block;

  if (ltrb) {
    BbFlipKernel<true><<<grid, block, 0, stream>>>(
      output.mutable_data<float>(), input.data<float>(), num_boxes,
      !per_sample_horz && horz[0], per_sample_horz,
      !per_sample_vert && vert[0], per_sample_vert,
      sample_idx);
  } else {
    BbFlipKernel<false><<<grid, block, 0, stream>>>(
      output.mutable_data<float>(), input.data<float>(), num_boxes,
      !per_sample_horz && horz[0], per_sample_horz,
      !per_sample_vert && vert[0], per_sample_vert,
      sample_idx);
  }
}

DALI_REGISTER_OPERATOR(BbFlip, BbFlip<GPUBackend>, GPU);

}  // namespace dali
