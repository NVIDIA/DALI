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
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  BatchedCropKernel<Out><<<N, dim3(32, 32), 0, stream>>>(
      C, H, W, in_batch, in_strides, L, out_batch, output_offsets);
  return DALISuccess;
}

template <typename Out>
DALIError_t ValidateBatchedCrop(const uint8 *const *in_batch,
                                const int *in_strides, int N, int *H, int *W,
                                int C, const Out *out_batch,
                                const int *output_offsets) {
  DALI_ASSERT(N > 0);
  DALI_ASSERT(C == 1 || C == 3);
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(in_strides != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  for (int i = 0; i < N; ++i) {
    DALI_ASSERT(in_batch[i] != nullptr);
    DALI_ASSERT(H[i] > 0);
    DALI_ASSERT(W[i] > 0);
    DALI_ASSERT(in_strides[i] >= C * W[i]);

    if (i == 0) {
      DALI_ASSERT(output_offsets[i] == 0);
    } else {
      DALI_ASSERT(output_offsets[i] ==
                  output_offsets[i - 1] + (W[i - 1] * H[i - 1] * C));
    }
  }

  return DALISuccess;
}

template <>
template <typename Out>
void Crop<GPUBackend>::RunHelper(Workspace<GPUBackend> *ws, const int idx) {
  const auto output = ws->Output<GPUBackend>(idx);
  ValidateHelper<Out>(output);

  DALI_CALL((BatchedCrop<Out>(
      input_ptrs_gpu_.template data<const uint8 *>(),
      input_strides_gpu_.template data<int>(), batch_size_,
      crop_height_gpu_.template data<int>(),
      crop_width_gpu_.template data<int>(), C_, output_layout_,
      output->template mutable_data<Out>(),
      output_offsets_gpu_.template data<int>(), ws->stream())));
}

template <>
template <typename Out>
void Crop<GPUBackend>::ValidateHelper(TensorList<GPUBackend> *output) {
  // Validate parameters
  DALI_CALL(ValidateBatchedCrop(
      input_ptrs_.template mutable_data<const uint8 *>(),
      input_strides_.template data<int>(), batch_size_, crop_height_.data(),
      crop_width_.data(), C_, output->template mutable_data<Out>(),
      output_offsets_.template data<int>()));
}

template <>
void Crop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  const auto &input = ws->Input<GPUBackend>(0);

  if (output_type_ == DALI_NO_TYPE) output_type_ = input.type().id();

  for (int i = 0; i < batch_size_; ++i)
    SetupSharedSampleParams(ws, input.tensor_shape(i), i, i);
}

template <>
void Crop<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  DALITensorLayout outLayout = DALI_UNKNOWN;

  std::vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
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

    input_strides_.template mutable_data<int>()[i] = W * C;
    crop_offsets_[i] = (crop_y * W + crop_x) * C;
    output_shape[i] = GetOutShape(input.GetLayout(), &outLayout, i);

    if (i == 0) {
      output_offsets_.template mutable_data<int>()[i] = 0;
    } else {
      auto cumulative_offset =
          (crop_height_[i - 1] * crop_width_[i - 1] * C_) +
          output_offsets_.template mutable_data<int>()[i - 1];
      output_offsets_.template mutable_data<int>()[i] = cumulative_offset;
    }
  }

  output->Resize(output_shape);
  output->SetLayout(outLayout);

  // Calculate input pointers and copy to gpu
  for (int i = 0; i < batch_size_; ++i) {
    input_ptrs_.template mutable_data<const uint8 *>()[i] =
        input.template tensor<uint8>(i) + crop_offsets_[i];
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
  if (output_type_ == DALI_FLOAT16)
    RunHelper<float16>(ws, idx);
  else
    CallRunHelper(ws, idx);
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<GPUBackend>, GPU);

}  // namespace dali
