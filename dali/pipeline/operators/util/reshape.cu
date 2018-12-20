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

#include "dali/pipeline/operators/util/reshape.h"
#include <cuda_runtime_api.h>

namespace dali {

template <>
std::vector<std::vector<Index>> Reshape<GPUBackend>::GetNewShapesForSamples(DeviceWorkspace *ws) {
  auto num_samples = ws->Input<CPUBackend>(kInTensorIdx).ntensor();
  std::vector<std::vector<Index>> result;
  if (use_input_for_shapes_) {
    // TODO(klecki) is there a way that we could handle GPU shapes here?
    auto &new_shapes = ws->Input<CPUBackend>(kInShapeIdx);
    DALI_ENFORCE(new_shapes.IsDenseTensor() && new_shapes.tensor_shape(0).size() == 1,
                 "New shape should be represented as 1-dim tensor with uniform dimension");
    int dim = new_shapes.tensor_shape(0)[0];
    for (decltype(num_samples) i = 0; i < num_samples; i++) {
      std::vector<Index> shape = {new_shapes.tensor<Index>(i), new_shapes.tensor<Index>(i) + dim};
      result.push_back(shape);
    }
  } else {
    for (decltype(num_samples) i = 0; i < num_samples; i++) {
      result.push_back(new_shape_);
    }
  }
  return result;
}

template <>
void Reshape<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(kInTensorIdx);
  auto *output = ws->Output<GPUBackend>(kOutTensorIdx);
  auto new_shapes = GetNewShapesForSamples(ws);
  output->set_type(input.type());
  output->Resize(new_shapes);
  for (size_t i = 0; i < new_shapes.size(); i++) {
    DALI_ENFORCE(Product(new_shapes[i]) == Product(input.tensor_shape(i)),
                 "New shape must represent exactly the same number of elements as old shape.");
  }
  CUDA_CALL(cudaMemcpyAsync(output->raw_mutable_data(), input.raw_data(), input.nbytes(),
                            cudaMemcpyDeviceToDevice, ws->stream()));
}

DALI_REGISTER_OPERATOR(Reshape, Reshape<GPUBackend>, GPU);

}  // namespace dali
