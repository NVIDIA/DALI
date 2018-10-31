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

#include "dali/pipeline/operators/fused/normalize_permute.h"

namespace dali {

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
        out[c*H*W + h*W + w] = StaticCastGpu<OUT>(
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
DALIError_t BatchedNormalizePermute(const uint8 *in_batch,
    int N, int H, int W, int C,  float *mean, float *inv_std,
    OUT *out_batch, cudaStream_t stream) {
  DALI_ASSERT(in_batch != nullptr);
  DALI_ASSERT(mean != nullptr);
  DALI_ASSERT(inv_std != nullptr);
  DALI_ASSERT(out_batch != nullptr);
  DALI_ASSERT(N > 0);
  DALI_ASSERT((C == 1) || (C == 3));
  DALI_ASSERT(W > 0);
  DALI_ASSERT(H > 0);

  BatchedNormalizePermuteKernel<<<N, dim3(32, 32), 0, stream>>>(
      in_batch, N, H, W, C, mean, inv_std, out_batch);
  return DALISuccess;
}

}  // namespace

template<>
template <typename OUT>
void NormalizePermute<GPUBackend>::GPURunHelper(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  // Validate input shape and type
  DALI_ENFORCE(IsType<uint8>(input.type()));
  DALI_ENFORCE(input.ntensor() == static_cast<size_t>(batch_size_),
      "Input does not have batch_size samples ("
      + std::to_string(input.ntensor()) + " v. " +
      std::to_string(batch_size_) + ")");

  for (int i = 0; i < batch_size_; ++i) {
    DALI_ENFORCE(input.tensor_shape(i).size() == 3,
        "Expects 3-dim image input (v. " +
        std::to_string(input.tensor_shape(i).size()) + ")");
    DALI_ENFORCE(input.tensor_shape(i)[0] == H_,
        "Input image height does not match output height.");
    DALI_ENFORCE(input.tensor_shape(i)[1] == W_,
        "Input image width does not match output width.");
    DALI_ENFORCE(input.tensor_shape(i)[2] == C_,
        "Input image channels does not match output channels.");
  }

  // Resize the output & run
  output->Resize(output_shape_);
  DALI_CALL(BatchedNormalizePermute(
          input.template data<uint8>(),
          batch_size_, H_, W_, C_,
          mean_.template mutable_data<float>(),
          inv_std_.template mutable_data<float>(),
          output->template mutable_data<OUT>(),
          ws->stream()));
}

template<>
void NormalizePermute<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  if (output_type_ == DALI_FLOAT) {
    GPURunHelper<float>(ws, idx);
  } else if (output_type_ == DALI_FLOAT16) {
    GPURunHelper<float16>(ws, idx);
  } else {
    DALI_FAIL("Unsupported output type.");
  }
}

DALI_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>, GPU);

}  // namespace dali

