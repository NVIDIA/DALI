#include "ndll/image/transform.h"

#include "ndll/util/npp.h"

namespace ndll {

namespace {
// Source: Based off kernel from caffe2/image/transform_gpu.cu. This was
// written by slayton I believe and the file has an Nvidia license at the top
template <typename OUT>
__global__ void BatchedNormalizePermuteKernel(const uint8 *in_batch,
    int N, int H, int W, int C,  float *mean, float *std, OUT *out_batch) {
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
            (static_cast<float>(in[h*W*C + w*C + c]) - mean[c]) / std[c]);
      }
    }
  }
}
} // namespace

template <typename OUT>
NDLLError_t BatchedNormalizePermute(const uint8 *in_batch,
    int N, int H, int W, int C,  float *mean, float *std,
    OUT *out_batch, cudaStream_t stream) {
  // TODO(tgale): Do we really want to verify all of this stuff
  // or should we leave some of the more obvious ones up to the user?
  NDLL_ASSERT(in_batch != nullptr);
  NDLL_ASSERT(mean != nullptr);
  NDLL_ASSERT(std != nullptr);
  NDLL_ASSERT(out_batch != nullptr);
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT((C == 1) || (C == 3));
  NDLL_ASSERT(W > 0);
  NDLL_ASSERT(H > 0);

  BatchedNormalizePermuteKernel<<<N, dim3(16, 16), 0, stream>>>(
      in_batch, N, H, W, C, mean, std, out_batch);
  return NDLLSuccess;
}

template NDLLError_t BatchedNormalizePermute<float16>(const uint8 *in_batch,
    int N, int H, int W, int C, float *mean, float *std, float16 *out_batch,
    cudaStream_t stream);

template NDLLError_t BatchedNormalizePermute<float>(const uint8 *in_batch,
    int N, int H, int W, int C, float *mean, float *std, float *out_batch,
    cudaStream_t stream);

template NDLLError_t BatchedNormalizePermute<double>(const uint8 *in_batch,
    int N, int H, int W, int C, float *mean, float *std, double *out_batch,
    cudaStream_t stream);

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
    
    // TODO: Can move condition out w/ function ptr or std::function obj
    if (C == 3) {
      NDLL_CHECK_NPP(nppiResize_8u_C3R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    } else {
      NDLL_CHECK_NPP(nppiResize_8u_C1R(in_batch[i], in_sizes[i].width*C, in_sizes[i],
              in_roi, out_batch[i], out_sizes[i].width*C, out_sizes[i], out_roi, npp_type));
    }
    CUDA_CALL(cudaDeviceSynchronize());
  }
  return NDLLSuccess;
}

} // namespace ndll
