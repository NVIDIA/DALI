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

#include "dali/pipeline/operators/crop/crop.h"
#include <vector>
#include "dali/image/transform.h"

namespace dali {

template <typename Out>
__global__ void BatchedCropKernel(
  const int C,
  const int H,
  const int W,
  const uint8* const * img_ptrs,
  const int* in_strides,
  DALITensorLayout layout,
  Out* out) {
  const int n = blockIdx.x;
  const int in_stride = in_strides[n];
  const uint8* input_ptr = img_ptrs[n];
  Out* output_ptr = out + (n * C * H * W);

  if (layout == DALI_NCHW) {
    for (int c = 0; c < C; ++c) {
      for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
          // From HWC
          int in_idx = h*in_stride + w*C + c;
          // To CHW
          int out_idx = (c * H + h) * W + w;
          output_ptr[out_idx] = static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  } else {  // Layout == DALI_NHWC
    for (int c = 0; c < C; ++c) {
      for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
          // From HWC
          int in_idx = h*in_stride + w*C + c;
          // To HWC
          int out_idx = (h * W + w) * C + c;
          output_ptr[out_idx] = static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  }
}

template <typename Out>
DALIError_t BatchedCrop(const uint8 * const *in_batch, const int *in_strides,
                        int N, int H, int W, int C, DALITensorLayout L,
                        Out *out_batch, cudaStream_t stream) {
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  BatchedCropKernel<Out><<<N, dim3(32, 32), 0, stream>>>(
    C, H, W, in_batch, in_strides, L, out_batch);
  return DALISuccess;
}

template <typename Out>
DALIError_t ValidateBatchedCrop(const uint8 * const *in_batch, const int *in_strides,
                                int N, int H, int W, int C, const Out *out_batch) {
  DALI_ASSERT(N > 0);
  DALI_ASSERT(H > 0);
  DALI_ASSERT(W > 0);
  DALI_ASSERT(C == 1 || C == 3);
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(in_strides != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  for (int i = 0; i < N; ++i) {
    DALI_ASSERT(in_batch[i] != nullptr);
    DALI_ASSERT(in_strides[i] >= C*W);
  }

  return DALISuccess;
}

template<>
template <typename Out, class null>
void Crop<GPUBackend>::RunHelper(Workspace<GPUBackend> *ws, const int idx) {
  const auto output = ws->Output<GPUBackend>(idx);
  ValidateHelper<Out>(output);

  DALI_CALL((BatchedCrop<Out>(
      InputImgsBatch(),
      InputStridesBatch(),
      batch_size_, crop_[0], crop_[1], C_, output_layout_,
      output->template mutable_data<Out>(),
      ws->stream())));
}

template<>
template <typename Out>
void Crop<GPUBackend>::ValidateHelper(TensorList<GPUBackend> *output) {
  // Validate parameters
  DALI_CALL(ValidateBatchedCrop(
    input_ptrs_.template mutable_data<const uint8*>(),
    input_strides_.template data<int>(),
    batch_size_, crop_[0], crop_[1], C_,
    output->template mutable_data<Out>()));
}

template<>
void Crop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  const auto & input = ws->Input<GPUBackend>(0);
  CastPermuteAttr::SetupSharedSampleParams(ws);

  for (int i = 0; i < batch_size_; ++i)
    SetupSharedSampleParams(ws, input.tensor_shape(i), i, i);
}

template<>
void Crop<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()),
               "Expected input data as uint8.");

  DALITensorLayout outLayout;
  const Dims out_shape = GetOutShape(input.GetLayout(), &outLayout);

  std::vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    const auto input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
                 "Expects 3-dimensional image input.");

    const int H = input_shape[0];
    const int W = input_shape[1];

    DALI_ENFORCE(H == per_sample_dimensions_[i].first &&
                 W == per_sample_dimensions_[i].second,
         "Corresponding images in different input sets need to have the same height and width");
    const int C = input_shape[2];

    DALI_ENFORCE(C == C_,
                 "Input channel dimension does not match "
                 "the output image type. Expected input with "
                 + to_string(C_) + " channels, got " + to_string(C) + ".");

    const int crop_y = per_sample_crop_[i].first;
    const int crop_x = per_sample_crop_[i].second;

    input_strides_.template mutable_data<int>()[i] = W*C;
    crop_offsets_[i] = (crop_y * W + crop_x) * C;
    output_shape[i] = out_shape;
  }

  output->Resize(output_shape);
  output->SetLayout(outLayout);

  // Calculate input pointers and copy to gpu
  for (int i = 0; i < batch_size_; ++i) {
    input_ptrs_.template mutable_data<const uint8*>()[i] =
      input.template tensor<uint8>(i) + crop_offsets_[i];
  }
  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());
  input_strides_gpu_.Copy(input_strides_, ws->stream());
}

template <>
void Crop<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  RUN_IMPL(ws, idx);
}

// Register operator
DALI_REGISTER_OPERATOR(Crop, Crop<GPUBackend>, GPU);

}  // namespace dali
