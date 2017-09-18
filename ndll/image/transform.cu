#include "ndll/image/transform.h"

namespace ndll {

namespace {
// Source: Based off kernel from caffe2/image/transform_gpu.cu. This was
// written by slayton I believe and the file has an Nvidia license at the top
//
// TODO(tgale): We'll need to do some funky-ness to add fp16 support to
// this and other similar functions.
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

} // namespace ndll
