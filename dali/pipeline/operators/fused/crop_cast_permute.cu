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

#include "dali/pipeline/operators/fused/crop_cast_permute.h"

#include <utility>
#include <vector>

namespace dali {

namespace {

template <DALITensorLayout Layout, typename Out>
__global__ void BatchedCropCastPermuteKernel(
    const int N,
    const int C,
    const int H,
    const int W,
    const uint8* const * img_ptrs,
    const int* in_strides,
    Out* out) {
  const int n = blockIdx.x;
  const int nStride = C * H * W;
  int in_stride = in_strides[n];
  const uint8* input_ptr = img_ptrs[n];
  Out* output_ptr = out + (n * nStride);

  if (Layout == DALI_NCHW) {
    for (int c = 0; c < C; ++c) {
      for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
          // From HWC
          int in_idx = h*in_stride + w*C + c;
          // To CHW
          int out_idx = c*H*W + h*W + w;
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
          int out_idx = h*W*C + w*C + c;
          output_ptr[out_idx] = static_cast<Out>(input_ptr[in_idx]);
        }
      }
    }
  }
}

template <DALITensorLayout L, typename Out>
DALIError_t BatchedCropCastPermute(const uint8 * const *in_batch,
    const int *in_strides,
    int N, int H, int W, int C,
    Out *out_batch, cudaStream_t stream) {
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  BatchedCropCastPermuteKernel<L, Out><<<N, dim3(32, 32), 0, stream>>>(
      N, C, H, W, in_batch, in_strides, out_batch);
  return DALISuccess;
}

template <typename Out>
DALIError_t ValidateBatchedCropCastPermute(const uint8 * const *in_batch,
    const int *in_strides,
    int N, int H, int W, int C, Out *out_batch) {
  DALI_ASSERT(N > 0);
  DALI_ASSERT(H > 0);
  DALI_ASSERT(W > 0);
  DALI_ASSERT(C == 1 || C == 3);
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(in_strides != nullptr);
  for (int i = 0; i < N; ++i) {
    DALI_ASSERT(in_batch[i] != nullptr);
    DALI_ASSERT(in_strides[i] >= C*W);
  }
  return DALISuccess;
}

}  // namespace

template<>
template <typename Out>
void CropCastPermute<GPUBackend>::RunHelper(Workspace<GPUBackend> *ws, const int idx) {
  auto output = ws->Output<GPUBackend>(idx);
  if (output_layout_ == DALI_NCHW) {
    DALI_CALL((BatchedCropCastPermute<DALI_NCHW, Out>(
            input_ptrs_gpu_.template data<const uint8*>(),
            input_strides_gpu_.template data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            output->template mutable_data<Out>(),
            ws->stream())));
  } else {
    DALI_CALL((BatchedCropCastPermute<DALI_NHWC, Out>(
            input_ptrs_gpu_.template data<const uint8*>(),
            input_strides_gpu_.template data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            output->template mutable_data<Out>(),
            ws->stream())));
  }
}

template<>
template <typename Out>
void CropCastPermute<GPUBackend>::ValidateHelper(TensorList<GPUBackend> *output) {
  // Validate parameters
  DALI_CALL(ValidateBatchedCropCastPermute(
          input_ptrs_.template mutable_data<const uint8*>(),
          input_strides_.template data<int>(),
          batch_size_, crop_h_, crop_w_, C_,
          output->template mutable_data<Out>()));
}

template<>
void CropCastPermute<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);

  if (output_type_ == DALI_NO_TYPE) {
    const TypeInfo& in_type = input.type();
    output_type_ = in_type.id();
  }

  for (int i = 0; i < batch_size_; ++i) {
    std::vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];

    per_sample_dimensions_[i] = std::make_pair(H, W);

    int C = input_shape[2];
    DALI_ENFORCE(C == C_,
        "Input channel dimension does not match "
        "the output image type. Expected input with "
        + to_string(C_) + " channels, got " + to_string(C) + ".");

    DALI_ENFORCE(H >= crop_h_);
    DALI_ENFORCE(W >= crop_w_);

    float crop_x_normalized = spec_.GetArgument<float>("crop_pos_x", ws, i);
    float crop_y_normalized = spec_.GetArgument<float>("crop_pos_y", ws, i);

    DALI_ENFORCE(crop_y_normalized >= 0.f &&  crop_y_normalized <= 1.f,
        "CropCastPermute coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_x_normalized >= 0.f &&  crop_x_normalized <= 1.f,
        "CropCastPermute coordinates need to be in range [0.0, 1.0]");

    int crop_y = crop_y_normalized * (H - crop_h_);
    int crop_x = crop_x_normalized * (W - crop_w_);
    per_sample_crop_[i] = std::make_pair(crop_y, crop_x);
  }
}

template<>
void CropCastPermute<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  std::vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    std::vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
         "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];

    DALI_ENFORCE(H == per_sample_dimensions_[i].first &&
        W == per_sample_dimensions_[i].second,
        "Corresponding images in different input sets need to have the same height and width");
    int C = input_shape[2];

    DALI_ENFORCE(C == C_,
            "Input channel dimension does not match "
            "the output image type. Expected input with "
            + to_string(C_) + " channels, got " + to_string(C) + ".");

    int crop_y = per_sample_crop_[i].first;
    int crop_x = per_sample_crop_[i].second;

    input_strides_.template mutable_data<int>()[i] = W*C_;
    crop_offsets_[i] = crop_y * C * W + crop_x * C;

    if (output_layout_ == DALI_SAME) {
      output_layout_ = input.GetLayout();
    }
    if (output_layout_ == DALI_NCHW) {
      output_shape[i] = {C, crop_h_, crop_w_};
    } else {
      output_shape[i] = {crop_h_, crop_w_, C};
    }
  }

  output->Resize(output_shape);

  // Calculate input pointers and copy to gpu
  for (int i = 0; i < batch_size_; ++i) {
    input_ptrs_.template mutable_data<const uint8*>()[i] =
      input.template tensor<uint8>(i) + crop_offsets_[i];
  }
  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());
  input_strides_gpu_.Copy(input_strides_, ws->stream());

  // Validate
  if (output_type_ == DALI_FLOAT) {
    ValidateHelper<float>(output);
  } else if (output_type_ == DALI_FLOAT16) {
    ValidateHelper<float16>(output);
  } else if (output_type_ == DALI_UINT8) {
    ValidateHelper<unsigned char>(output);
  } else if (output_type_ == DALI_INT16) {
    ValidateHelper<int16>(output);
  } else if (output_type_ == DALI_INT32) {
    ValidateHelper<int>(output);
  } else if (output_type_ == DALI_INT64) {
    ValidateHelper<int64>(output);
  } else {
    DALI_FAIL("Unsupported output type.");
  }
}


template <>
void CropCastPermute<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  DataDependentSetup(ws, idx);
  if (output_type_ == DALI_FLOAT) {
    RunHelper<float>(ws, idx);
  } else if (output_type_ == DALI_FLOAT16) {
    RunHelper<float16>(ws, idx);
  } else if (output_type_ == DALI_UINT8) {
    RunHelper<unsigned char>(ws, idx);
  } else if (output_type_ == DALI_INT16) {
    RunHelper<int16>(ws, idx);
  } else if (output_type_ == DALI_INT32) {
    RunHelper<int>(ws, idx);
  } else if (output_type_ == DALI_INT64) {
    RunHelper<int64>(ws, idx);
  } else {
    DALI_FAIL("Unsupported output type.");
  }
}


// Register operator
DALI_REGISTER_OPERATOR(CropCastPermute, CropCastPermute<GPUBackend>, GPU);

}  // namespace dali
