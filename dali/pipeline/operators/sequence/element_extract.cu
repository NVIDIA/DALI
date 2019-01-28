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
#include "dali/pipeline/operators/sequence/element_extract.h"

namespace dali {

namespace detail {

template <typename T>
void ElementExtractImpl(const TensorList<GPUBackend> &input,
                        TensorList<GPUBackend> &output,
                        int index,
                        cudaStream_t cuda_stream) {
    for (unsigned int i = 0; i < input.ntensor(); i++) {
        auto* output_data = output.mutable_tensor<T>(i);
        const auto* input_data = input.tensor<T>(i);
        const auto& tensor_shape = input.tensor_shape(i);
        const auto element_size = Product(tensor_shape) / tensor_shape[0];

        const auto input_offset = index * element_size;

        CUDA_CALL(cudaMemcpyAsync(
            output_data,
            &input_data[input_offset],
            element_size * sizeof(T),
            cudaMemcpyDeviceToDevice,
            cuda_stream));
    }
}

std::vector<Dims> GetOutputShape(const TensorList<GPUBackend> &input,
                                 const std::vector<int>& element_map) {
    std::vector<Dims> output_shape;
    for (unsigned int i = 0; i < input.ntensor(); ++i) {
        auto shape = input.tensor_shape(i);
        CheckInputShape(shape, element_map);
        Dims element_shape(shape.begin() + 1, shape.end());
        output_shape.push_back(element_shape);
    }
    return output_shape;
}


}  // namespace detail

template <>
void ElementExtract<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
    auto &input = ws->Input<GPUBackend>(idx);
    auto output_shape = detail::GetOutputShape(input, element_map_);
    auto element_layout = detail::GetElementLayout(input.GetLayout());
    int elements_per_sample = element_map_.size();
    int output_offset = idx * elements_per_sample;
    auto data_type = input.type().id();
    DALI_TYPE_SWITCH(data_type, Type,
        for (int k = 0; k < elements_per_sample; k++) {
            int element = element_map_[k];
            auto &output = ws->Output<GPUBackend>(output_offset + k);
            output.set_type(input.type());
            output.SetLayout(element_layout);
            output.Resize(output_shape);
            detail::ElementExtractImpl<Type>(
                input, output, element, ws->stream());
        }
    )
}

DALI_REGISTER_OPERATOR(ElementExtract, ElementExtract<GPUBackend>, GPU);

}  // namespace dali
