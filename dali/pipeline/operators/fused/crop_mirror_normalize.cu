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

namespace dali {

// Crop, mirror, mean sub, stddev div, NHWC->NCHW, Npp8u->fp32
template <typename Out>
__global__ void BatchedCropMirrorNormalizePermuteKernel(
    const int C,
    const int H,
    const int W,
    const bool pad,
    const int *mirror,
    const float* mean,
    const float* inv_std,
    const uint8* const * img_ptrs,
    const int *input_steps,
    DALITensorLayout layout,
    Out* out) {
  const int n = blockIdx.x;

  const int pad_C = pad ? 4 : C;
  const int nStride = pad_C*H*W;

  // pointers to data for this image
  const uint8* input_ptr = img_ptrs[n];
  const int in_step = input_steps[n];
  Out* output_ptr = &out[n*nStride];
  const int mirror_image = mirror[n];

  const int a = mirror_image? (W - 1) * C : 0;
  const int b = mirror_image? -C : C;
  if (layout == DALI_NCHW) {
    // Coalesced writes
    for (int c=0; c < C; ++c) {
      for (int h=threadIdx.y; h < H; h += blockDim.y) {
        for (int w=threadIdx.x; w < W; w += blockDim.x) {
          const int in_idx = a + c + b * w + in_step*h;   // HWC
          const int out_idx = (c*H + h)*W + w;            // CHW

          output_ptr[out_idx] = StaticCastGpu<Out>(
                (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c]);
        }
      }
    }

    // Pad to 4 channels with 0s
    if (pad) {
      const Out out = StaticCastGpu<Out>(0);
      for (int c=C; c < 4; ++c) {
        for (int h=threadIdx.y; h < H; h += blockDim.y) {
          for (int w=threadIdx.x; w < W; w += blockDim.x) {
            const int out_idx = (c*H + h)*W + w;  // CHW
            output_ptr[out_idx] = out;
          }
        }
      }
    }
  } else {
    for (int tid = threadIdx.x + threadIdx.y * blockDim.x;
         tid < nStride;
         tid += blockDim.x * blockDim.y) {
      const int c = tid % pad_C;
      const int w = (tid / pad_C) % W;
      const int h = tid / (pad_C * W);

      float input;
      if (pad && c == 3) {
        input = 0;
      } else {
        const int in_idx =  a + c + b * w + in_step * h;
        input = (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c];
      }

      const int out_idx = c + (w + h*W) * pad_C;
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
template <typename OUT>
DALIError_t BatchedCropMirrorNormalizePermute(const uint8 * const *in_batch, const int *in_strides,
    int N, int H, int W, int C, bool pad, const int *mirror, const float *mean,
    const float *inv_std, DALITensorLayout layout, OUT *out_batch, cudaStream_t stream) {
  DALI_ASSERT(mirror != nullptr);
  DALI_ASSERT(mean != nullptr);
  DALI_ASSERT(inv_std != nullptr);

  BatchedCropMirrorNormalizePermuteKernel<OUT><<<N, dim3(32, 32), 0, stream>>>(
        C, H, W, pad, mirror, mean, inv_std, in_batch, in_strides, layout, out_batch);

  return DALISuccess;
}

template<>
template <typename Out, class null>
void CropMirrorNormalize<GPUBackend>::RunHelper(Workspace<GPUBackend> *ws, const int idx) {
  const auto output = ws->Output<GPUBackend>(idx);
  ValidateHelper<Out>(output);

  DALI_CALL(BatchedCropMirrorNormalizePermute<Out>(
            InputImgsBatch(),
            InputStridesBatch(),
            batch_size_, crop_[0], crop_[1], C_, pad_,
            mirror_gpu_.template data<int>(),
            mean_.template data<float>(),
            inv_std_.template data<float>(),
            output_layout_,
            output->template mutable_data<Out>(),
            ws->stream()));
}

template <>
void CropMirrorNormalize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  RUN_IMPL_GPU(ws, idx);
}

template<>
void CropMirrorNormalize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  Crop<GPUBackend>::SetupSharedSampleParams(ws);
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
