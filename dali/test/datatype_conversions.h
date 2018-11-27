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

#ifndef DALI_TEST_DATATYPE_CONVERSIONS_H_
#define DALI_TEST_DATATYPE_CONVERSIONS_H_

#include <memory>
#include <vector>
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

namespace detail {

/// Return type here is std::unique_ptr, since TensorList doesn't have move constructor (DALI-385)
template<typename Backend, typename DataType>
std::unique_ptr<TensorList<Backend>> ToTensorList(const std::vector<DataType> &input_batch,
                                                  const std::vector<int64_t> &shape) {
  DALI_FAIL(
          "Converting provided InputType to TensorList is not supported."
          " You may want to write your own specialization for it.");
}


/// Specialization for std::vector<float>
// TODO(mszolucha) support other backends
template<>
inline std::unique_ptr<TensorList<CPUBackend>>
ToTensorList<CPUBackend, std::vector<float>>
(const std::vector<std::vector<float>> &input_batch, const std::vector<int64_t> &shape) {
  std::unique_ptr<TensorList<CPUBackend>> tensor_list(new TensorList<CPUBackend>);

  std::vector<std::vector<int64_t>> new_shape(input_batch.size(), shape);
  tensor_list->Resize(new_shape);

  auto ptr = tensor_list->template mutable_tensor<float>(0);
  for (const auto &input : input_batch) {
    for (const auto &val : input) {
      *ptr++ = val;
    }
  }
  return tensor_list;
}

}  // namespace detail


/**
 * Function, that converts given batch of inputs to the TensorList
 *
 * Return type here is std::unique_ptr, since TensorList doesn't have move constructor (DALI-385)
 *
 * @tparam InputType type of a single sample in batch
 * @param input_batch
 * @param shape what the output TensorList shape shall be
 */
template<typename Backend, typename InputType>
std::unique_ptr<TensorList<Backend>> ToTensorList(const std::vector<InputType> &input_batch,
                                                  const std::vector<size_t>& shape) {
  std::vector<int64_t> converted_shape{shape.begin(), shape.end()};
  return detail::ToTensorList<Backend>(input_batch, converted_shape);
}

/**
 * Function, that extracts a data batch from given TensorList.
 * @tparam OutputType type of a single sample in batch
 */
template<typename Backend, typename OutputType>
std::vector<OutputType> FromTensorList(const TensorList<Backend> &tensor_list) {
  DALI_FAIL(
          "Converting TensorList to provided OutputType is not supported. "
          "You may want to write your own specialization for it.");
}


/// Specialization for std::vector<float>
template<typename Backend>
std::vector<std::vector<float>> FromTensorList(const TensorList<Backend> &tensor_list) {
  auto single_output_size = tensor_list.size() / tensor_list.ntensor();
  std::vector<std::vector<float>> ret;
  for (size_t i = 0; i < tensor_list.ntensor(); i++) {
    auto begin =
            tensor_list.template data<float>() + tensor_list.tensor_offset(static_cast<int>(i));
    auto end = begin + single_output_size;
    ret.emplace_back(std::vector<float>{begin, end});
  }
  return ret;
}

}  // namespace dali

#endif  // DALI_TEST_DATATYPE_CONVERSIONS_H_
