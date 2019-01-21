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
                        const std::vector<int> &indexes,
                        cudaStream_t cuda_stream) {
    for (unsigned int i = 0; i < input.ntensor(); i++) {
        auto* output_data = output.mutable_tensor<T>(i);
        const auto* input_data = input.tensor<T>(i);
        const auto& tensor_shape = input.tensor_shape(i);
        const auto element_size = tensor_shape[1] * tensor_shape[2] * tensor_shape[3];

        for (unsigned int k = 0; k < indexes.size(); k++) {
            const auto output_offset = k * element_size;
            const auto input_offset = indexes[k] * element_size;

            CUDA_CALL(cudaMemcpyAsync(
                &output_data[output_offset],
                &input_data[input_offset],
                element_size * sizeof(T),
                cudaMemcpyDeviceToDevice,
                cuda_stream));
        }
    }
}

}  // namespace detail

template <>
void ElementExtract<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
    auto &input = ws->Input<GPUBackend>(idx);
    auto &output = ws->Output<GPUBackend>(idx);
    output.set_type(input.type());
    output.SetLayout(input.GetLayout());

    std::vector<Dims> output_shape;
    for (unsigned int i = 0; i < input.ntensor(); ++i) {
        auto shape = input.tensor_shape(i);
        CheckInputShape(shape);
        int N_output = element_map_.size();
        shape[0] = N_output;
        output_shape.push_back(shape);
    }
    output.Resize(output_shape);

    auto data_type = input.type().id();
    DALI_TYPE_SWITCH(data_type, Type,
        detail::ElementExtractImpl<Type>(
            input, output, element_map_, ws->stream());
    )
}

DALI_REGISTER_OPERATOR(ElementExtract, ElementExtract<GPUBackend>, GPU);

}  // namespace dali
