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

#include <vector>
#include "dali/image/transform.h"
#include "dali/pipeline/operators/crop/crop.h"

namespace dali {

namespace {

template <typename Out>
__global__ void BatchedCropKernel(const int C, const int *height,
                                  const int *width,
                                  const uint8 *const *img_ptrs,
                                  const int *in_strides,
                                  DALITensorLayout layout, Out *out,
                                  const int *output_offsets) {
  const int n = blockIdx.x;
  const int W = width[n];
  const int H = height[n];
  const int in_stride = in_strides[n];
  const uint8 *input_ptr = img_ptrs[n];

  if (layout == DALI_NCHW) {
    for (int c = 0; c < C; ++c) {
      for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
          // From HWC
          int in_idx = h * in_stride + w * C + c;
          // To CHW
          int out_idx = (c * H + h) * W + w;
          out[output_offsets[n] + out_idx] =
              static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  } else {  // Layout == DALI_NHWC
    for (int c = 0; c < C; ++c) {
      for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
          // From HWC
          int in_idx = h * in_stride + w * C + c;
          // To HWC
          int out_idx = (h * W + w) * C + c;
          out[output_offsets[n] + out_idx] =
              static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  }
}

template <typename Out>
DALIError_t BatchedCrop(const uint8 *const *in_batch, const int *in_strides,
                        int N, const int *H, const int *W, int C,
                        DALITensorLayout L, Out *out_batch,
                        const int *output_offsets, cudaStream_t stream) {
  BatchedCropKernel<Out><<<N, dim3(32, 32), 0, stream>>>(
      C, H, W, in_batch, in_strides, L, out_batch, output_offsets);
  return DALISuccess;
}

}  // namespace

template <>
template <typename Out>
void Crop<GPUBackend>::RunHelper(Workspace<GPUBackend> *ws, const int idx) {
  auto &output = ws->Output<GPUBackend>(idx);
  DALI_CALL((BatchedCrop<Out>(
      input_ptrs_gpu_.data<const uint8 *>(),
      input_strides_gpu_.data<int>(), batch_size_ * SequenceSize(idx),
      crop_height_gpu_.data<int>(),
      crop_width_gpu_.data<int>(), C_, output_layout_,
      output.mutable_data<Out>(),
      output_offsets_gpu_.data<int>(), ws->stream())));
}

template <>
void Crop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  CropAttr::ProcessArguments(ws);
  const auto &input = ws->Input<GPUBackend>(0);
  const int sequence_size = SequenceSize(0);
  const int total_batch_size = batch_size_ * sequence_size;
  if (sequence_size > 1) {
    per_sample_crop_.resize(total_batch_size);
    per_sample_dimensions_.resize(total_batch_size);
    std::vector<int> new_crop_height(total_batch_size);
    std::vector<int> new_crop_width(total_batch_size);
    std::vector<float> new_crop_y_norm(total_batch_size);
    std::vector<float> new_crop_x_norm(total_batch_size);
    for (int i = 0; i < total_batch_size; ++i) {
      new_crop_height[i] = crop_height_[i / sequence_size];
      new_crop_width[i] = crop_width_[i / sequence_size];
      new_crop_y_norm[i] = crop_y_norm_[i / sequence_size];
      new_crop_x_norm[i] = crop_x_norm_[i / sequence_size];
    }
    crop_height_ = new_crop_height;
    crop_width_ = new_crop_width;
    crop_y_norm_ = new_crop_y_norm;
    crop_x_norm_ = new_crop_x_norm;
  }

  if (output_type_ == DALI_NO_TYPE) output_type_ = input.type().id();

  for (int i = 0; i < total_batch_size; ++i)
    SetupSharedSampleParams(ws, input.tensor_shape(i), i, i);
}

template <>
void Crop<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);

  const DALITensorLayout in_layout = input.GetLayout();
  DALI_ENFORCE(in_layout == DALI_NHWC);

  auto &output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  DALITensorLayout out_layout = DALI_UNKNOWN;

  int total_batch_size = batch_size_ * SequenceSize(idx);
  if (SequenceSize(idx) > 1) {
    crop_offsets_.resize(total_batch_size);
    input_ptrs_.Resize({total_batch_size});
    input_strides_.Resize({total_batch_size});
    output_offsets_.Resize({total_batch_size});
  }

  std::vector<Dims> output_shape(total_batch_size);
  for (int i = 0; i < total_batch_size; ++i) {
    const auto input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    const auto H = static_cast<int>(input_shape[0]);
    const auto W = static_cast<int>(input_shape[1]);

    DALI_ENFORCE(H == per_sample_dimensions_[i].first &&
                 W == per_sample_dimensions_[i].second,
                 "Corresponding images in different input sets need to have "
                 "the same height and width");
    const auto C = static_cast<int>(input_shape[2]);

    DALI_ENFORCE(C == C_,
                 "Input channel dimension does not match "
                 "the output image type. Expected input with " +
                 to_string(C_) + " channels, got " + to_string(C) + ".");

    const int crop_y = per_sample_crop_[i].first;
    const int crop_x = per_sample_crop_[i].second;

    input_strides_.mutable_data<int>()[i] = W * C;
    crop_offsets_[i] = (crop_y * W + crop_x) * C;
    output_shape[i] = GetOutShape(in_layout, &out_layout, i);

    if (i == 0) {
      output_offsets_.mutable_data<int>()[i] = 0;
    } else {
      auto cumulative_offset =
          (crop_height_[i - 1] * crop_width_[i - 1] * C_) +
          output_offsets_.mutable_data<int>()[i - 1];
      output_offsets_.mutable_data<int>()[i] = cumulative_offset;
    }
  }

  output.Resize(output_shape);
  output.SetLayout(out_layout);

  // Calculate input pointers and copy to gpu
  for (int i = 0; i < total_batch_size; ++i) {
    input_ptrs_.mutable_data<const uint8 *>()[i] =
        input.tensor<uint8>(i) + crop_offsets_[i];
  }
  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());
  input_strides_gpu_.Copy(input_strides_, ws->stream());
  output_offsets_gpu_.Copy(output_offsets_, ws->stream());

  crop_width_gpu_.Copy(crop_width_, ws->stream());
  crop_height_gpu_.Copy(crop_height_, ws->stream());
}

template <>
void Crop<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  DataDependentSetup(ws, idx);
  DALI_TYPE_SWITCH_WITH_FP16_GPU(output_type_, OutputType,
    RunHelper<OutputType>(ws, idx););
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<GPUBackend>, GPU);

}  // namespace dali
