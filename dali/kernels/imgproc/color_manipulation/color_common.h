#ifndef DALI_COLOR_COMMON_H
#define DALI_COLOR_COMMON_H

#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {
namespace color_manipulation {


/**
 * TODO
 * @tparam ndims
 * @param shape
 * @return
 */
template <size_t ndims>
TensorShape<ndims - 1> Flatten(const TensorShape<ndims> &shape) {
  TensorShape<ndims - 1> ret;
  for (int i = 0; i < shape.size() - 1; i++) {
    ret[i] = shape[i];
  }
  ret[shape.size() - 2] *= shape[shape.size() - 1];
  return ret;
}


template <size_t ndims>
TensorListShape<ndims - 1> Flatten(const TensorListShape<ndims> &shape) {
  TensorListShape<ndims - 1> ret(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    ret.set_tensor_shape(i, Flatten<ndims>(shape[i]));
  }
  return ret;
}
}
}
}
#endif //DALI_COLOR_COMMON_H
