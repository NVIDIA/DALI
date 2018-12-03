#ifndef DALI_TENSOR_BATCH_ADAPTER_H
#define DALI_TENSOR_BATCH_ADAPTER_H

#include "../pipeline/data/tensor_list.h"
#include <vector>

namespace dali {

namespace testing {

template<typename UnderlyingDataType>
class TensorAdapter {
  static_assert(std::is_fundamental<UnderlyingDataType>::value, "Underlying data type for tensor has to be fundamental");
public:
  TensorAdapter(std::vector<UnderlyingDataType> data, std::vector<size_t> shape) noexcept {}

  template<typename Backend, typename T>
  static dali::TensorList<Backend> ToTensorList(std::vector<TensorAdapter<T>> tensors) noexcept {
  }

private:
  std::vector<UnderlyingDataType> data;
  std::vector<size_t> shape;
};



template<typename UnderlyingDataType, typename Backend>
std::vector<TensorAdapter<UnderlyingDataType>> CreateTensorAdapters(dali::TensorList<Backend> tensor_list) noexcept {}

}  // namespace testing

}  // namespace dali


#endif //DALI_TENSOR_BATCH_ADAPTER_H
