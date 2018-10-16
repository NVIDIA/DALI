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

template <typename OUT>
__global__ void BatchedNormalizePermuteKernel(const uint8 *in_batch,
    int H, int W, int C, const float *mean, const float *inv_std, OUT *out_batch) {
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
            (in[h*W*C + w*C + c] - mean[c]) * inv_std[c]);
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
    int N, int H, int W, int C,  const float *mean, const float *inv_std,
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
      in_batch, H, W, C, mean, inv_std, out_batch);
  return DALISuccess;
}

template<>
void NormalizePermute<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  // Validate input shape and type
  DALI_ENFORCE(IsType<uint8>(input.type()));
  DALI_ENFORCE(input.ntensor() == batch_size_,
               "Input does not have batch_size samples ("
               + std::to_string(input.ntensor()) + " v. " +
               std::to_string(batch_size_) + ")");

  for (int i = 0; i < batch_size_; ++i)
    CheckShape(input.tensor_shape(i));

  // Initiate shapes, if it was not done yet
  if (!output_shape_.size()) {
    output_shape_.resize(batch_size_);
    for (auto &shape : output_shape_)
      shape = {C_, H_, W_};
  }

  // Resize the output & run
  output->Resize(output_shape_);
  output->SetLayout(DALI_NCHW);
}

template<>
template<typename Out, class null>
void NormalizePermute<GPUBackend>::RunHelper(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  DALI_CALL(BatchedNormalizePermute(
          input.template data<uint8>(),
          batch_size_, H_, W_, C_,
          mean_.template mutable_data<float>(),
          inv_std_.template mutable_data<float>(),
          output->template mutable_data<Out>(),
          ws->stream()));
}

template<>
void NormalizePermute<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  RUN_IMPL(ws, idx);
}

DALI_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>, GPU);

}  // namespace dali

