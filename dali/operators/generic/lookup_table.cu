// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <limits>
#include "dali/core/convert.h"
#include "dali/core/span.h"
#include "dali/operators/generic/lookup_table.h"

namespace dali {

namespace detail {

constexpr auto kMaxKey = LookupTable<GPUBackend>::kMaxKey;

template <typename OutputType, typename InputType>
__global__ void LookupValuesImpl(const LutSampleDesc *samples, const kernels::BlockDesc<1> *blocks,
                                 const OutputType *lookup_table, const OutputType default_value) {
  // We do not check the key range when the type range is smaller than the supported range
  constexpr bool check_range =
      !std::is_same<InputType, uint8_t>::value && !std::is_same<InputType, uint16_t>::value;
  constexpr auto max_key = ConvertSat<InputType>(kMaxKey);

  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  auto *output = reinterpret_cast<OutputType *>(sample.output);
  const auto *input = reinterpret_cast<const InputType *>(sample.input);
  for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
    const auto key = input[x];
    if (check_range) {
      output[x] = (std::is_unsigned<InputType>::value || key >= 0) && key <= max_key ?
                      lookup_table[key] :
                      default_value;
    } else {
      output[x] = lookup_table[key];
    }
  }
}

}  // namespace detail

template<>
void LookupTable<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  const auto &shape = input.shape();
  auto &output = ws.OutputRef<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  const auto stream = ws.stream();

  auto num_samples = shape.num_samples();
  samples_.resize(num_samples);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    samples_[sample_id].output = output.raw_mutable_tensor(sample_id);
    samples_[sample_id].input = input.raw_tensor(sample_id);
  }
  samples_dev_.from_host(samples_, stream);

  std::array<std::pair<int, int>, 1> collapse_groups = {{{0, shape.sample_dim()}}};
  auto collapsed_shape = collapse_dims<1>(shape, collapse_groups);

  block_setup_.SetupBlocks(collapsed_shape, true);
  blocks_dev_.from_host(block_setup_.Blocks(), stream);

  TYPE_SWITCH(input.type().id(), dali::type2id, InputType, LUT_IN_TYPES, (
    TYPE_SWITCH(output_type_, dali::type2id, OutputType, LUT_OUT_TYPES, (

      const OutputType *lookup_table = lut_.data<OutputType>();
      OutputType default_value = ConvertSat<OutputType>(default_value_f_);

      dim3 grid_dim = block_setup_.GridDim();
      dim3 block_dim = block_setup_.BlockDim();

      detail::LookupValuesImpl<OutputType, InputType><<<grid_dim, block_dim, 0, stream>>>(
          samples_dev_.data(), blocks_dev_.data(), lookup_table, default_value);

    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)); );       // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())); );     // NOLINT
}

DALI_REGISTER_OPERATOR(LookupTable, LookupTable<GPUBackend>, GPU);

}  // namespace dali
