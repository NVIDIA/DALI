// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

namespace {

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

/**
 * @brief Performs mean subtraction & stddev division per channel, cast
 * to output type, and NHWC->NCHW permutation.
 *
 * 'mean' and 'inv_std' are assumed to point to device memory of size `c`.
 * Input data is assumed to be stored in NHWC layout in memory. Output
 * data will be stored in NCHW.
 */
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

}  // namespace

template<>
template <typename OUT>
void NormalizePermute<GPUBackend>::GPURunHelper(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  // Validate input shape and type
  NDLL_ENFORCE(IsType<uint8>(input.type()));
  NDLL_ENFORCE(input.ntensor() == batch_size_,
      "Input does not have batch_size samples ("
      + std::to_string(input.ntensor()) + " v. " +
      std::to_string(batch_size_) + ")");

  for (int i = 0; i < batch_size_; ++i) {
    NDLL_ENFORCE(input.tensor_shape(i).size() == 3,
        "Expects 3-dim image input (v. " +
        std::to_string(input.tensor_shape(i).size()) + ")");
    NDLL_ENFORCE(input.tensor_shape(i)[0] == H_,
        "Input image height does not match output height.");
    NDLL_ENFORCE(input.tensor_shape(i)[1] == W_,
        "Input image width does not match output width.");
    NDLL_ENFORCE(input.tensor_shape(i)[2] == C_,
        "Input image channels does not match output channels.");
  }

  // Resize the output & run
  output->Resize(output_shape_);
  NDLL_CALL(BatchedNormalizePermute(
          input.template data<uint8>(),
          batch_size_, H_, W_, C_,
          mean_.template mutable_data<float>(),
          inv_std_.template mutable_data<float>(),
          output->template mutable_data<OUT>(),
          ws->stream()));
}

template<>
void NormalizePermute<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  if (output_type_ == NDLL_FLOAT) {
    GPURunHelper<float>(ws, idx);
  } else if (output_type_ == NDLL_FLOAT16) {
    GPURunHelper<float16>(ws, idx);
  } else {
    NDLL_FAIL("Unsupported output type.");
  }
}

NDLL_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>, GPU);

}  // namespace ndll

