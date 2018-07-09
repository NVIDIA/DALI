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

#include "dali/pipeline/operators/fused/crop_mirror_normalize.h"

#include <utility>
#include <vector>

namespace dali {

namespace {

// Crop, mirror, mean sub, stddev div, NHWC->NCHW, Npp8u->fp32
template <DALITensorLayout Layout, typename Out, bool pad>
__global__ void BatchedCropMirrorNormalizePermuteKernel(
    const int N,
    const int C,
    const int H,
    const int W,
    const int *mirror,
    const float* mean,
    const float* inv_std,
    const uint8* const * img_ptrs,
    const int *input_steps,
    Out* out) {
  const int n = blockIdx.x;

  const int pad_C = pad ? 4 : C;
  const int nStride = pad_C*H*W;

  // pointers to data for this image
  const uint8* input_ptr = img_ptrs[n];
  int in_step = input_steps[n];
  Out* output_ptr = &out[n*nStride];
  bool mirror_image = mirror[n];

  if (Layout == DALI_NCHW) {
    if (mirror_image) {
      // Mirror the image - coalesced writes
      for (int c=0; c < C; ++c) {
        for (int h=threadIdx.y; h < H; h += blockDim.y) {
          for (int w=threadIdx.x; w < W; w += blockDim.x) {
            int mirrored_width = (W - 1) - w;
            int in_idx = c + C*mirrored_width + in_step*h;  // HWC, mirrored
            int out_idx = c*H*W + h*W + w;  // CHW

            output_ptr[out_idx] = StaticCastGpu<Out>(
                (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c]);
          }
        }
      }
    } else {
      // Copy normally - coalesced writes
      for (int c=0; c < C; ++c) {
        for (int h=threadIdx.y; h < H; h += blockDim.y) {
          for (int w=threadIdx.x; w < W; w += blockDim.x) {
            int in_idx = c + C*w + in_step*h;  // HWC
            int out_idx = c*H*W + h*W + w;  // CHW

            output_ptr[out_idx] = StaticCastGpu<Out>(
                (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c]);
          }
        }
      }
    }
    // Pad to 4 channels with 0s
    if (pad) {
      for (int c=C; c < 4; ++c) {
        for (int h=threadIdx.y; h < H; h += blockDim.y) {
          for (int w=threadIdx.x; w < W; w += blockDim.x) {
            int out_idx = c*H*W + h*W + w;  // CHW

            output_ptr[out_idx] = StaticCastGpu<Out>(0);
          }
        }
      }
    }
  } else {
    for (int tid = threadIdx.x + threadIdx.y * blockDim.x;
         tid < pad_C * H * W;
         tid += blockDim.x * blockDim.y) {
      const int c = tid % pad_C;
      const int w = (tid / pad_C) % W;
      const int h = tid / (pad_C * W);

      const int in_c = c;
      const int in_w = mirror_image ? (W - 1) - w : w;
      const int in_h = h;

      const int in_idx = in_c + in_w * C + in_step * in_h;
      const int out_idx = c + w*pad_C + h*W*pad_C;

      float input;
      if (pad && c == 3) {
        input = 0;
      } else {
        input = (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c];
      }

      output_ptr[out_idx] = StaticCastGpu<Out>(input);
    }
  }
}

/**
 * @brief Takes in a jagged batch of images and crops, (optional) mirrors,
 * performs mean subtraction & stddev division per channel, cast to output
 * type, and NHWC->NCHW permutation
 *
 * The crop is performed by offsetting the ptrs in 'in_batch' to the beginning
 * of the crop region, and then passing in the stride of each image so that
 * the kernel can correctly process the crop region.
 *
 * @param in_batch device pointer to pointer to the beginning of the crop
 * region for each image
 * @param in_strides device pointer to `N` ints whose value is the stride
 * of each input image
 * @param mirror device pointer to `N` bools whose values indicate whether
 * the image should be mirrored or not
 * @param N number of elements in the batch
 * @param H output height for all images in the batch
 * @param W output width for all images in the batch
 * @param C number of channels of images in the batch
 * @param mean device pointer of length `C` to the mean to subtract for
 * each image channel
 * @param std device pointer of length `C` to the inverse std dev. to multiply by
 * for each image channel
 * @param out_batch pointer of size `N*C*H*W` to store the dense, cropped,
 * NCHW output batch
 * @param stream cuda stream to operate in
 */
template <DALITensorLayout L, typename OUT>
DALIError_t BatchedCropMirrorNormalizePermute(const uint8 * const *in_batch,
    const int *in_strides, int N, int H, int W, int C, bool pad, const int *mirror,
    const float *mean, const float *inv_std, OUT *out_batch, cudaStream_t stream) {
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(in_strides != nullptr);
  DALI_ASSERT(mirror != nullptr);
  DALI_ASSERT(mean != nullptr);
  DALI_ASSERT(inv_std != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  if (pad) {
    BatchedCropMirrorNormalizePermuteKernel<L, OUT, true><<<N, dim3(32, 32), 0, stream>>>(
        N, C, H, W, mirror, mean, inv_std, in_batch, in_strides, out_batch);
  } else {
    BatchedCropMirrorNormalizePermuteKernel<L, OUT, false><<<N, dim3(32, 32), 0, stream>>>(
        N, C, H, W, mirror, mean, inv_std, in_batch, in_strides, out_batch);
  }
  return DALISuccess;
}

/**
 * @brief Validates the parameters for 'BatchedCropMirrorNormalizePermute'
 * on host
 *
 * All parameters are host-side versions of the arguments to
 * 'BatchedCropMirrorNormalizePermute'. This method exists so that
 * the user can efficiently manage memory copies to the GPU, but stil
 * have a method for validating input arguments before calling the
 * batched function.
 *
 * Checks that...
 * - in_batch device pointers are not nullptr
 * - in_strides values are >= W*C
 * - N > 0, H > 0, W > 0, C == 1 || C == 3
 */
template <typename OUT>
DALIError_t ValidateBatchedCropMirrorNormalizePermute(const uint8 * const *in_batch,
    const int *in_strides, int N, int H, int W, int C,
    const float *mean, const float *inv_std, OUT *out_batch) {
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
template <typename OUT>
void CropMirrorNormalize<GPUBackend>::RunHelper(Workspace<GPUBackend> *ws, const int idx) {
  auto output = ws->Output<GPUBackend>(idx);
  if (output_layout_ == DALI_NCHW) {
    DALI_CALL((BatchedCropMirrorNormalizePermute<DALI_NCHW, OUT>(
            input_ptrs_gpu_.template data<const uint8*>(),
            input_strides_gpu_.template data<int>(),
            batch_size_, crop_h_, crop_w_, C_, pad_,
            mirror_gpu_.template data<int>(),
            mean_.template data<float>(),
            inv_std_.template data<float>(),
            output->template mutable_data<OUT>(),
            ws->stream())));
  } else {
    DALI_CALL((BatchedCropMirrorNormalizePermute<DALI_NHWC, OUT>(
            input_ptrs_gpu_.template data<const uint8*>(),
            input_strides_gpu_.template data<int>(),
            batch_size_, crop_h_, crop_w_, C_, pad_,
            mirror_gpu_.template data<int>(),
            mean_.template data<float>(),
            inv_std_.template data<float>(),
            output->template mutable_data<OUT>(),
            ws->stream())));
  }
}

template<>
template <typename OUT>
void CropMirrorNormalize<GPUBackend>::ValidateHelper(TensorList<GPUBackend> *output) {
  // Validate parameters
  DALI_CALL(ValidateBatchedCropMirrorNormalizePermute(
          input_ptrs_.template mutable_data<const uint8*>(),
          input_strides_.template mutable_data<int>(),
          batch_size_, crop_h_, crop_w_, C_,
          mean_vec_.data(), inv_std_vec_.data(),
          output->template mutable_data<OUT>()));
}

template<>
void CropMirrorNormalize<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
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

    // retrieve already determined crop parameters
    int crop_y = per_sample_crop_[i].first;
    int crop_x = per_sample_crop_[i].second;

    // Save image stride & crop offset
    input_strides_.template mutable_data<int>()[i] = W*C_;
    crop_offsets_[i] = crop_y*W*C_ + crop_x*C_;

    // Pad to 4 channels
    int pad_C = pad_ ? 4 : C_;

    // Save the output shape of this image
    if (output_layout_ == DALI_NCHW) {
      output_shape[i] = {pad_C, crop_h_, crop_w_};
    } else {
      output_shape[i] = {crop_h_, crop_w_, pad_C};
    }
  }

  // Resize the output data
  output->Resize(output_shape);

  // Set the layout of the output data
  output->SetLayout(output_layout_);

  // Copy strides to gpu
  input_strides_gpu_.Copy(input_strides_, ws->stream());

  // Calculate input pointers and copy to gpu
  for (int i = 0; i < batch_size_; ++i) {
    input_ptrs_.template mutable_data<const uint8*>()[i] =
      input.template tensor<uint8>(i) + crop_offsets_[i];
  }
  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());

  // Validate
  if (output_type_ == DALI_FLOAT) {
    ValidateHelper<float>(output);
  } else if (output_type_ == DALI_FLOAT16) {
    ValidateHelper<float16>(output);
  } else {
    DALI_FAIL("Unsupported output type.");
  }
}

template<>
void CropMirrorNormalize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  // Before we start working on the next input set, we need
  // to wait until the last one is finished. Otherwise we risk
  // overwriting data used by the kernel called for previous image
  if (idx != 0)
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));
  DataDependentSetup(ws, idx);
  if (output_type_ == DALI_FLOAT) {
    RunHelper<float>(ws, idx);
  } else if (output_type_ == DALI_FLOAT16) {
    RunHelper<float16>(ws, idx);
  } else {
    DALI_FAIL("Unsupported output type.");
  }
}

template<>
void CropMirrorNormalize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");
  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
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


    // Crop
    DALI_ENFORCE(H >= crop_h_);
    DALI_ENFORCE(W >= crop_w_);

    float crop_x_image_coord = spec_.GetArgument<float>("crop_pos_x", ws, i);
    float crop_y_image_coord = spec_.GetArgument<float>("crop_pos_y", ws, i);

    DALI_ENFORCE(crop_x_image_coord >= 0.f && crop_x_image_coord <= 1.f,
        "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_y_image_coord >= 0.f && crop_y_image_coord <= 1.f,
        "Crop coordinates need to be in range [0.0, 1.0]");

    int crop_y = crop_y_image_coord * (H - crop_h_);
    int crop_x = crop_x_image_coord * (W - crop_w_);
    per_sample_crop_[i] = std::make_pair(crop_y, crop_x);
  }
  if (has_mirror_) {
    const Tensor<CPUBackend> &mirror = ws->ArgumentInput("mirror");
    mirror_gpu_.Copy(mirror, ws->stream());
  } else {
    mirror_gpu_.Copy(mirror_, ws->stream());
  }
}

// Register operator
DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<GPUBackend>, GPU);

}  // namespace dali
