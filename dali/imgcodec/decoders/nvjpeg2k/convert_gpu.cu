// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/imgcodec/decoders/nvjpeg2k/convert_gpu.h"
#include "dali/kernels/imgproc/pointwise/multiply_add_gpu.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace imgcodec {

template<class DataType>
void AdjustBitdepthImpl(DataType *in, DataType *out, int in_bpp, cudaStream_t stream,
                        const TensorShape<3> &shape) {
  // values are in range [0, 2^in_bpp - 1], we want them in range [0, 2^type_bpp - 1]
  size_t type_bpp = 8 * sizeof(DataType);
  float current_max_value = (1 << in_bpp) - 1;
  float expected_max_value = (1 << type_bpp) - 1;

  static const std::vector<float> zero = {0.0};
  std::vector<float> multiplier = {expected_max_value / current_max_value};

  TensorListView<StorageGPU, DataType, 3> view_in(in, {shape});
  TensorListView<StorageGPU, DataType, 3> view_out(out, {shape});

  assert(zero.size() == 1);
  assert(multiplier.size() == 1);
  assert(view_in.size() == 1);
  assert(view_out.size() == 1);

  dali::kernels::DynamicScratchpad scratchpad({}, AccessOrder(stream));
  dali::kernels::KernelContext ctx;
  ctx.gpu.stream = stream;
  ctx.scratchpad = &scratchpad;
  dali::kernels::MultiplyAddGpu<DataType, DataType, 3> kernel;
  kernel.Setup(ctx, view_in, zero, multiplier);
  kernel.Run(ctx, view_out, view_in, zero, multiplier);
}

template void AdjustBitdepthImpl<uint8_t>(uint8_t *, uint8_t *, int, cudaStream_t,
                                          const TensorShape<3> &);
template void AdjustBitdepthImpl<uint16_t>(uint16_t *, uint16_t *, int, cudaStream_t,
                                           const TensorShape<3> &);

}  // namespace imgcodec
}  // namespace dali
