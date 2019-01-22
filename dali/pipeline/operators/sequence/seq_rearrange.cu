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

#include "dali/pipeline/operators/sequence/seq_rearrange.h"

namespace dali {

template <>
void SequenceRearrange<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);
  std::vector<std::vector<Index>> new_list_shape;
  std::vector<Index> list_elements_bytes;
  for (decltype(input.ntensor()) tensor_idx = 0; tensor_idx < input.ntensor(); tensor_idx++) {
    // TODO(klecki) not sure if we should allow for different length of sequence
    // and different element sizes in one batch
    auto ith_shape = input.tensor_shape(tensor_idx);
    DALI_ENFORCE(ith_shape.size() > 1, "Sequence elements must have at least 1 dim");
    std::vector<Index> new_sample_shape;

    Index element_size;
    std::tie(new_sample_shape, element_size) = GetNewShapeAndElementSize(input.tensor_shape(tensor_idx), new_order_);
    Index element_bytes = element_size * input.type().size();
    new_list_shape.push_back(new_sample_shape);
    list_elements_bytes.push_back(element_bytes);
  }

  output.set_type(input.type());
  output.Resize(new_list_shape);
  for (decltype(input.ntensor()) tensor_idx = 0; tensor_idx < input.ntensor(); tensor_idx++) {
    TypeInfo type = input.type();
    for (int elem = 0; elem < GetSeqLength(new_list_shape[tensor_idx]); elem++) {
      Index element_bytes = list_elements_bytes[tensor_idx];
      type.Copy<GPUBackend, GPUBackend>(
          output.mutable_tensor<char>(tensor_idx) + elem * element_bytes,
          input.tensor<char>(tensor_idx) + new_order_[elem] * element_bytes,
          element_bytes, 0);
    }
  }
}

DALI_REGISTER_OPERATOR(SequenceRearrange, SequenceRearrange<GPUBackend>, GPU);

}  // namespace dali