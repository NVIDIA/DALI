// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <utility>
#include <vector>

#include "dali/pipeline/operators/sequence/sequence_rearrange.h"

namespace dali {

// __global__ void
// SequenceRearrangeKernel(const char *input, float *output, size_t pitch, size_t width_px,
//                           size_t height) {
  // size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  // size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  // if (x >= width_px || y >= height) return;
  // auto value_in = pitch_xy(input, x, y, pitch);
  // size_t outidx = x + width_px * y;
  // output[outidx] = decode_flow_component(value_in);
// }





template <>
void SequenceRearrange<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);

  std::vector<std::vector<Index>> new_list_shape;
  std::vector<Index> list_elements_sizes;

  for (size_t tensor_idx = 0; tensor_idx < input.ntensor(); tensor_idx++) {
    // TODO(klecki) not sure if we should allow for different length of sequence
    // and different element sizes in one batch
    auto ith_shape = input.tensor_shape(tensor_idx);
    DALI_ENFORCE(ith_shape.size() > 1, "Sequence elements must have at least 1 dim");

    std::vector<Index> new_sample_shape;
    Index element_size;
    std::tie(new_sample_shape, element_size) = GetNewShapeAndElementSize(ith_shape, new_order_);

    new_list_shape.push_back(std::move(new_sample_shape));
    list_elements_sizes.push_back(element_size);
  }

  output.Resize(new_list_shape);
  output.set_type(input.type());
  output.SetLayout(input.GetLayout());

  for (size_t tensor_idx = 0; tensor_idx < input.ntensor(); tensor_idx++) {
    TypeInfo type = input.type();
    for (int elem = 0; elem < GetSeqLength(new_list_shape[tensor_idx]); elem++) {
      Index element_size = list_elements_sizes[tensor_idx];
      Index element_bytes = element_size * type.size();
      type.Copy<GPUBackend, GPUBackend>(
          static_cast<uint8 *>(output.raw_mutable_tensor(tensor_idx)) + elem * element_bytes,
          static_cast<const uint8 *>(input.raw_tensor(tensor_idx)) +
              new_order_[elem] * element_bytes,
          element_size, ws->stream());
    }
  }
}

DALI_REGISTER_OPERATOR(SequenceRearrange, SequenceRearrange<GPUBackend>, GPU);

}  // namespace dali
