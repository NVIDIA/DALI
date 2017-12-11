// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/image/transform.h"

#include "ndll/util/npp.h"

namespace ndll {

namespace {
// Source: Based off kernel from caffe2/image/transform_gpu.cu. This was
// written by slayton I believe and the file has an Nvidia license at the top
template <typename OUT>
__global__ void BatchedNormalizePermuteKernel(const uint8 *in_batch,
    int N, int H, int W, int C,  float *mean, float *inv_std, OUT *out_batch) {
  // We process one image per thread block
  const int n = blockIdx.x;
  const int stride = H*W*C;

  // Get pointer to my blocks image
  const uint8 *in = in_batch + n * stride;
  OUT *out = out_batch + n * stride;

  for (int c = 0; c < C; ++c) {
    for (int h = threadIdx.y; h < H; h += blockDim.y) {
      for (int w = threadIdx.x; w < W; w += blockDim.x) {
        out[c*H*W + h*W + w] = static_cast<OUT>(
            (static_cast<float>(in[h*W*C + w*C + c]) - mean[c]) * inv_std[c]);
      }
    }
  }
}

// Crop, mirror, mean sub, stddev div, NHWC->NCHW, Npp8u->fp32
template <typename Out>
__global__ void BatchedCropMirrorNormalizePermuteKernel(
    const int N,
    const int C,
    const int H,
    const int W,
    const bool *mirror,
    const float* mean,
    const float* inv_std,
    const uint8* const * img_ptrs,
    const int *input_steps,
    Out* out) {
  const int n = blockIdx.x;

  const int nStride = C*H*W;

  // pointers to data for this image
  const uint8* input_ptr = img_ptrs[n];
  int in_step = input_steps[n];
  Out* output_ptr = &out[n*nStride];
  bool mirror_image = mirror[n];

  if (mirror_image) {
    // Mirror the image - coalesced writes
    for (int c=0; c < C; ++c) {
      for (int h=threadIdx.y; h < H; h += blockDim.y) {
        for (int w=threadIdx.x; w < W; w += blockDim.x) {
          int mirrored_width = (W - 1) - w;
          int in_idx = c + C*mirrored_width + in_step*h;  // HWC, mirrored
          int out_idx = c*H*W + h*W + w;  // CHW

          output_ptr[out_idx] = static_cast<Out>(
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

          output_ptr[out_idx] = static_cast<Out>(
              (static_cast<float>(input_ptr[in_idx])-mean[c]) * inv_std[c]);
        }
      }
    }
  }
}

}  // namespace

template <typename OUT>
NDLLError_t BatchedNormalizePermute(const uint8 *in_batch,
    int N, int H, int W, int C,  float *mean, float *inv_std,
    OUT *out_batch, cudaStream_t stream) {
  // TODO(tgale): Do we really want to verify all of this stuff
  // or should we leave some of the more obvious ones up to the user?
  NDLL_ASSERT(in_batch != nullptr);
  NDLL_ASSERT(mean != nullptr);
  NDLL_ASSERT(inv_std != nullptr);
  NDLL_ASSERT(out_batch != nullptr);
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT((C == 1) || (C == 3));
  NDLL_ASSERT(W > 0);
  NDLL_ASSERT(H > 0);

  BatchedNormalizePermuteKernel<<<N, dim3(32, 32), 0, stream>>>(
      in_batch, N, H, W, C, mean, inv_std, out_batch);
  return NDLLSuccess;
}

template NDLLError_t BatchedNormalizePermute<float16>(const uint8 *in_batch,
    int N, int H, int W, int C, float *mean, float *inv_std, float16 *out_batch,
    cudaStream_t stream);

template NDLLError_t BatchedNormalizePermute<float>(const uint8 *in_batch,
    int N, int H, int W, int C, float *mean, float *inv_std, float *out_batch,
    cudaStream_t stream);

template NDLLError_t BatchedNormalizePermute<double>(const uint8 *in_batch,
    int N, int H, int W, int C, float *mean, float *inv_std, double *out_batch,
    cudaStream_t stream);

template <typename OUT>
NDLLError_t BatchedCropMirrorNormalizePermute(const uint8 * const *in_batch,
    const int *in_strides, int N, int H, int W, int C, const bool *mirror,
    const float *mean, const float *inv_std, OUT *out_batch, cudaStream_t stream) {
  NDLL_ASSERT(in_batch != nullptr);
  NDLL_ASSERT(in_strides != nullptr);
  NDLL_ASSERT(mirror != nullptr);
  NDLL_ASSERT(mean != nullptr);
  NDLL_ASSERT(inv_std != nullptr);
  NDLL_ASSERT(out_batch != nullptr);
  BatchedCropMirrorNormalizePermuteKernel<<<N, dim3(32, 32), 0, stream>>>(
      N, C, H, W, mirror, mean, inv_std, in_batch, in_strides, out_batch);
  return NDLLSuccess;
}

template NDLLError_t BatchedCropMirrorNormalizePermute<float16>(
    const uint8 * const *in_batch, const int *in_strides, int N, int H, int W, int C,
    const bool *mirror, const float *mean, const float *inv_std, float16 *out_batch,
    cudaStream_t stream);

template NDLLError_t BatchedCropMirrorNormalizePermute<float>(
    const uint8 * const *in_batch, const int *in_strides, int N, int H, int W, int C,
    const bool *mirror, const float *mean, const float *inv_std, float *out_batch,
    cudaStream_t stream);

template NDLLError_t BatchedCropMirrorNormalizePermute<double>(
    const uint8 * const *in_batch, const int *in_strides, int N, int H, int W, int C,
    const bool *mirror, const float *mean, const float *inv_std, double *out_batch,
    cudaStream_t stream);

template <typename OUT>
NDLLError_t ValidateBatchedCropMirrorNormalizePermute(const uint8 * const *in_batch,
    const int *in_strides, int N, int H, int W, int C, const bool *mirror,
    const float *mean, const float *inv_std, OUT *out_batch) {
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT(H > 0);
  NDLL_ASSERT(W > 0);
  NDLL_ASSERT(C == 1 || C == 3);
  NDLL_ASSERT(in_batch != nullptr);
  NDLL_ASSERT(in_strides != nullptr);
  for (int i = 0; i < N; ++i) {
    NDLL_ASSERT(in_batch[i] != nullptr);
    NDLL_ASSERT(in_strides[i] >= C*W);
  }
  return NDLLSuccess;
}

template NDLLError_t ValidateBatchedCropMirrorNormalizePermute<float16>(
    const uint8 * const *in_batch, const int *in_strides, int N, int H, int W, int C,
    const bool *mirror, const float *mean, const float *inv_std, float16 *out_batch);

template NDLLError_t ValidateBatchedCropMirrorNormalizePermute<float>(
    const uint8 * const *in_batch, const int *in_strides, int N, int H, int W, int C,
    const bool *mirror, const float *mean, const float *inv_std, float *out_batch);

template NDLLError_t ValidateBatchedCropMirrorNormalizePermute<double>(
    const uint8 * const *in_batch, const int *in_strides, int N, int H, int W, int C,
    const bool *mirror, const float *mean, const float *inv_std, double *out_batch);

NDLLError_t BatchedResize(const uint8 **in_batch, int N, int C, const NDLLSize *in_sizes,
    uint8 **out_batch, const NDLLSize *out_sizes, NDLLInterpType type) {
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT(C == 1 || C == 3);
  NDLL_ASSERT(in_sizes != nullptr);
  NDLL_ASSERT(out_sizes != nullptr);

  NppiInterpolationMode npp_type;
  NDLL_FORWARD_ERROR(NPPInterpForNDLLInterp(type, &npp_type));

  for (int i = 0; i < N; ++i) {
    NDLL_ASSERT(in_batch[i] != nullptr);
    NDLL_ASSERT(out_batch[i] != nullptr);

    // Setup region of interests to whole image
    NppiRect in_roi, out_roi;
    in_roi.x = 0; in_roi.y = 0;
    in_roi.width = in_sizes[i].width;
    in_roi.height = in_sizes[i].height;
    out_roi.x = 0; out_roi.y = 0;
    out_roi.width = out_sizes[i].width;
    out_roi.height = out_sizes[i].height;

    // TODO(tgale): Can move condition out w/ function ptr or std::function obj
    if (C == 3) {
      NDLL_CHECK_NPP(nppiResize_8u_C3R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    } else {
      NDLL_CHECK_NPP(nppiResize_8u_C1R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    }
  }
  return NDLLSuccess;
}

}  // namespace ndll
