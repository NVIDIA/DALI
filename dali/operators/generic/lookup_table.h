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

#ifndef DALI_OPERATORS_GENERIC_LOOKUP_TABLE_H_
#define DALI_OPERATORS_GENERIC_LOOKUP_TABLE_H_

#define LUT_IN_TYPES (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t)
#define LUT_OUT_TYPES                                                        \
  (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, \
  float16, float, double, bool)

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/host_dev.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/type_tag.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace detail {

template <typename T>
void value_mem_deleter(void *ptr) {
  delete [] static_cast<T*>(ptr);
}

}  // namespace detail

struct LutSampleDesc {
  void *output;
  const void *input;
};

template <typename Backend>
class LookupTable : public Operator<Backend> {
 public:
  static constexpr size_t kLookupTableSize = 0x10000;
  static constexpr size_t kMaxKey = kLookupTableSize - 1;

  explicit inline LookupTable(const OpSpec &spec)
      : Operator<Backend>(spec),
        input_type_(DALI_NO_TYPE),
        output_type_(spec.GetArgument<DALIDataType>("dtype")),
        default_value_f_(spec.GetArgument<float>("default_value")) {
    lut_.Reset();
    std::vector<int> keys;
    int min_key = -1, max_key = -1;
    std::vector<float> values_f;

    if (spec.HasArgument("keys")) {
      keys = spec.GetRepeatedArgument<int>("keys");
      min_key = *std::min_element(keys.begin(), keys.end());
      max_key = *std::max_element(keys.begin(), keys.end());
      DALI_ENFORCE(min_key >= 0 && max_key <= static_cast<int>(kMaxKey),
        "`keys` should be in the range [0, " + std::to_string(kMaxKey) + "]");
    }
    if (spec.HasArgument("values")) {
      values_f = spec.GetRepeatedArgument<float>("values");
    }
    DALI_ENFORCE(keys.size() == values_f.size(),
      "`keys` size should match `values` size");

    TYPE_SWITCH(output_type_, dali::type2id, OutputType, LUT_OUT_TYPES, (
        value_mem_ = {new OutputType[kLookupTableSize], detail::value_mem_deleter<OutputType>};
        OutputType *values = static_cast<OutputType*>(value_mem_.get());
        for (size_t i = 0; i < kLookupTableSize; i++) {
          values[i] = ConvertSat<OutputType>(default_value_f_);
        }
        auto keys_size = keys.size();
        for (size_t i = 0; i < keys_size; i++) {
          values[keys[i]] = ConvertSat<OutputType>(values_f[i]);
        }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_));  // NOLINT
    );  // NOLINT
  }

  ~LookupTable() override = default;
  DISABLE_COPY_MOVE_ASSIGN(LookupTable);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    if (std::is_same<Backend, GPUBackend>::value && !lut_.shape().num_elements()) {
      TYPE_SWITCH(output_type_, dali::type2id, OutputType, LUT_OUT_TYPES, (
          auto data = span<OutputType>(static_cast<OutputType *>(value_mem_.get()),
                                       kLookupTableSize);
          lut_.Copy(data, ws.stream());
        ), DALI_FAIL(make_string("Unsupported output type: ", output_type_));  // NOLINT
      );  // NOLINT
    }
    output_desc.resize(1);
    output_desc[0].type = TypeTable::GetTypeInfo(output_type_);
    const auto &input = ws.template InputRef<Backend>(0);
    output_desc[0].shape = input.shape();
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override;

 private:
  DALIDataType input_type_, output_type_;
  float default_value_f_ = 0.0f;
  std::unique_ptr<void, void(*)(void*)> value_mem_ = {nullptr, free};
  Tensor<GPUBackend> lut_;

  using GpuBlockSetup = kernels::BlockSetup<1, -1>;

  GpuBlockSetup block_setup_;
  std::vector<LutSampleDesc> samples_;
  DeviceBuffer<GpuBlockSetup::BlockDesc> blocks_dev_;
  DeviceBuffer<LutSampleDesc> samples_dev_;

  USE_OPERATOR_MEMBERS();
};

template <typename Backend, typename Output, typename Input>
DALI_HOST_DEV
void DoLookup(Output &output, Input key, const Output *lookup_table, Output default_value) {
  // We do not check the key range when the type range is smaller than the supported range
  constexpr bool check_range =
      !std::is_same<Input, uint8_t>::value && !std::is_same<Input, uint16_t>::value;
  constexpr auto max_key = ConvertSat<Input>(LookupTable<Backend>::kMaxKey);

  if (check_range) {
    output = (std::is_unsigned<Input>::value || key >= 0) && key <= max_key ? lookup_table[key] :
                                                                              default_value;
  } else {
    output = lookup_table[key];
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_LOOKUP_TABLE_H_
