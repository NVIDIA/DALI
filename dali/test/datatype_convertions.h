#ifndef DALI_DATATYPE_CONVERTIONS_H
#define DALI_DATATYPE_CONVERTIONS_H

namespace dali {

namespace detail {
template<typename Backend, typename Input>
TensorList <Backend> ToTensorList(const Input &input, std::vector<int64_t> shape) {
};


template<>
TensorList <CPUBackend>
ToTensorList<CPUBackend, std::array<float, 4>>(const std::array<float, 4> &input, std::vector<int64_t> shape) {
  TensorList<CPUBackend> tensor_list;
  tensor_list.Resize({shape});
  auto ptr = tensor_list.template mutable_tensor<float>(0);
  for (size_t i = 0; i < input.size(); i++) {
    ptr[i] = input[i];
  }
  return tensor_list;

}
} // namespace detail

template<typename Backend, typename Input>
TensorList <Backend> ToTensorList(const Input &input, std::vector<size_t> shape) {
  std::vector<int64_t> converted_shape{shape.begin(), shape.end()};
  return detail::ToTensorList<Backend>(input, converted_shape);
};
}

#endif //DALI_DATATYPE_CONVERTIONS_H
