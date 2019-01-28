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
#include "dali/error_handling.h"

namespace dali {

DALI_SCHEMA(ElementExtract)
    .DocStr(R"code(Extracts one or more elements from input)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg("element_map",
        R"code(Indices of extracted elements)code",
        DALI_INT_VEC)
    .AdditionalOutputsFn(
        [](const OpSpec& spec) {
            auto element_map = spec.GetRepeatedArgument<int>("element_map");
            DALI_ENFORCE(element_map.size() >= 1);
            auto additional_outputs = element_map.size() - 1;
            return additional_outputs;
        });

namespace detail {

template <typename T>
void ElementExtractImpl(const Tensor<CPUBackend> &input,
                        Tensor<CPUBackend> &output,
                        int input_offset,
                        int element_size) {
    const auto* input_data = input.data<T>();
    memcpy(
        output.mutable_data<T>(),
        &input_data[input_offset],
        element_size * sizeof(T));
}

}  // namespace detail

template <>
void ElementExtract<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
    const auto &input = ws->Input<CPUBackend>(idx);
    auto data_type = input.type().id();
    auto element_layout = detail::GetElementLayout(input.GetLayout());

    auto shape = input.shape();
    detail::CheckInputShape(shape, element_map_);
    Dims output_shape(shape.begin()+1, shape.end());
    auto element_size = Product(shape) / shape[0];

    auto elements_per_sample = element_map_.size();
    auto output_offset = idx * elements_per_sample;

    DALI_TYPE_SWITCH(data_type, Type,
        for (std::size_t k = 0; k < elements_per_sample; k++) {
            auto output_idx = output_offset + k;
            auto &output = ws->Output<CPUBackend>(output_idx);
            output.set_type(input.type());
            output.SetLayout(element_layout);
            output.Resize(output_shape);

            auto element_idx = element_map_[k];
            auto element_offset = element_idx * element_size;

            detail::ElementExtractImpl<Type>(
                input, output, element_offset, element_size);
        }
    )
}

DALI_REGISTER_OPERATOR(ElementExtract, ElementExtract<CPUBackend>, CPU);

}  // namespace dali
