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

#include "dali/operators/generic/lookup_table.h"
#include <limits>
#include "dali/core/span.h"
#include "dali/core/convert.h"

namespace dali {

namespace detail {

constexpr auto kMaxKey = LookupTable<GPUBackend>::kMaxKey;

template <typename OutputType, typename InputType>
__global__ void
LookupValuesImpl(OutputType *output,
                 const InputType *input,
                 size_t data_size,
                 const OutputType* lookup_table,
                 const OutputType default_value) {
  // We do not check the key range when the type range is smaller than the supported range
  constexpr bool check_range =
       !std::is_same<InputType, uint8_t>::value
    && !std::is_same<InputType, uint16_t>::value;
  constexpr auto max_key = ConvertSat<InputType>(kMaxKey);

  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= data_size)
    return;

  const auto key = input[tid];
  if (check_range) {
    output[tid] = (std::is_unsigned<InputType>::value || key >= 0) && key <= max_key ?
      lookup_table[key] : default_value;
  } else {
    output[tid] = lookup_table[key];
  }
}

}  // namespace detail

template<>
void LookupTable<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  auto data_size = input.size();
  constexpr int kThreads = 512;
  const int blocks = (data_size + kThreads - 1) / kThreads;
  const auto stream = ws.stream();
  Tensor<GPUBackend> lookup_table_gpu;

  TYPE_SWITCH(input.type().id(), dali::type2id, InputType,
              (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t), (
    DALI_TYPE_SWITCH_WITH_FP16(output_type_, OutputType,
      auto *out_data = output.mutable_data<OutputType>();
      const auto *in_data = input.data<InputType>();

      OutputType *tensor_lut_cpu = static_cast<OutputType*>(value_mem_.get());
      lookup_table_gpu.Copy(make_span(tensor_lut_cpu, kLookupTableSize), stream);
      const OutputType *lookup_table = lookup_table_gpu.data<OutputType>();
      OutputType default_value = ConvertSat<OutputType>(default_value_f_);

      detail::LookupValuesImpl<OutputType, InputType><<<blocks, kThreads, 0, stream>>>(
        out_data, in_data, data_size,
        lookup_table, default_value);
    )
  ), DALI_FAIL("Unsupported input type"); );     // NOLINT
}

DALI_REGISTER_OPERATOR(LookupTable, LookupTable<GPUBackend>, GPU);

}  // namespace dali
