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

#include "dali/pipeline/operators/sequence/element_extract.h"

namespace dali {

DALI_SCHEMA(ElementExtract)
    .DocStr(R"code(Extracts one or more elements from input)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg("element_map",
        R"code(Indices of extracted elements)code",
        DALI_INT_VEC);

namespace detail {

template <typename T>
void ElementExtractImpl(const Tensor<CPUBackend> &input,
                        Tensor<CPUBackend> &output,
                        const std::vector<int> &indexes) {
    auto* output_data = output.mutable_data<T>();
    const auto* input_data = input.data<T>();
    const auto& tensor_shape = input.shape();
    const auto element_size = Product(tensor_shape) / tensor_shape[0];

    for (unsigned int k = 0; k < indexes.size(); k++) {
        const auto output_offset = k * element_size;
        const auto input_offset = indexes[k] * element_size;
        memcpy(
            &output_data[output_offset],
            &input_data[input_offset],
            element_size * sizeof(T));
    }
}

}  // namespace detail

template <>
void ElementExtract<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
    const auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);
    output.set_type(input.type());
    output.SetLayout(input.GetLayout());

    std::vector<Dims> output_shape;
    auto shape = input.shape();
    CheckInputShape(shape);
    int N_output = element_map_.size();
    shape[0] = N_output;
    output.Resize(shape);

    auto data_type = input.type().id();
    DALI_TYPE_SWITCH(data_type, Type,
        detail::ElementExtractImpl<Type>(
            input, output, element_map_);
    )
}

DALI_REGISTER_OPERATOR(ElementExtract, ElementExtract<CPUBackend>, CPU);

}  // namespace dali
