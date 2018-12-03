#ifndef DALI_TENSOR_BATCH_ADAPTER_H
#define DALI_TENSOR_BATCH_ADAPTER_H

#include "../pipeline/data/tensor_list.h"
#include <vector>

namespace dali {

namespace testing {

template<typename DataType>
class TensorAdapter {
  static_assert(std::is_fundamental<DataType>::value, "Underlying data type for tensor has to be fundamental");
public:
  TensorAdapter(std::vector<DataType> data, std::vector<size_t> shape) noexcept {}

private:
  std::vector<DataType> data;
  std::vector<size_t> shape;
};

template<typename Backend, typename DataType>
dali::TensorList<Backend> ToTensorList(TensorAdapter<DataType> tensors) noexcept {}

template<typename DataType, typename Backend>
std::vector<TensorAdapter<DataType>> CreateTensorAdapters(dali::TensorList<Backend> tensor_list) noexcept {}

}  // namespace testing

}  // namespace dali


#endif //DALI_TENSOR_BATCH_ADAPTER_H
