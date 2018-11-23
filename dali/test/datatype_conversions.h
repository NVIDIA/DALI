#ifndef DALI_DATATYPE_CONVERSIONS_H
#define DALI_DATATYPE_CONVERSIONS_H

#include "dali/pipeline/data/tensor_list.h"
#include <memory>

namespace dali {

namespace detail {

/// Return type here is std::unique_ptr, since TensorList doesn't have move constructor (DALI-385)
template<typename Backend, typename Input>
std::unique_ptr<TensorList<Backend>> ToTensorList(const Input &input, std::vector<int64_t> shape) {
  DALI_FAIL(
          "Converting provided input type to TensorList is not supported. You may want to write your own specialization for it.");
};


/// Return type here is std::unique_ptr, since TensorList doesn't have move constructor (DALI-385)
template<>
inline std::unique_ptr<TensorList<CPUBackend>>
ToTensorList<CPUBackend, std::array<float, 4ul>>
(const std::array<float, 4ul> &input, std::vector<int64_t> shape) {
  std::unique_ptr<TensorList<CPUBackend>> tensor_list(new TensorList<CPUBackend>);
  tensor_list->Resize({shape});
  auto ptr = tensor_list->template mutable_tensor<float>(0);
  for (size_t i = 0; i < input.size(); i++) {
    ptr[i] = input[i];
  }
  return tensor_list;
}


/// Return type here is std::unique_ptr, since TensorList doesn't have move constructor (DALI-385)
template<>
inline std::unique_ptr<TensorList<CPUBackend>>
ToTensorList<CPUBackend, std::vector<float>>
(const std::vector<float> &input, std::vector<int64_t> shape) {
  std::unique_ptr<TensorList<CPUBackend>> tensor_list(new TensorList<CPUBackend>);
  tensor_list->Resize({shape});
  auto ptr = tensor_list->template mutable_tensor<float>(0);
  for (size_t i = 0; i < input.size(); i++) {
    ptr[i] = input[i];
  }
  return tensor_list;
}

} // namespace detail

/**
 * Return type here is std::unique_ptr, since TensorList doesn't have move constructor (DALI-385)
 * @tparam Backend
 * @tparam Input
 * @param input
 * @param shape
 * @return
 */
template<typename Backend, typename Input>
std::unique_ptr<TensorList<Backend>> ToTensorList(const Input &input, std::vector<size_t> shape) {
  std::vector<int64_t> converted_shape{shape.begin(), shape.end()};
  return detail::ToTensorList<Backend>(input, converted_shape);
};

}  // namespace dali

#endif //DALI_DATATYPE_CONVERSIONS_H
