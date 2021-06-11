// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TF_PLUGIN_DALI_SHAPE_HELPER_H_
#define DALI_TF_PLUGIN_DALI_SHAPE_HELPER_H_

#include <memory>

#include "dali/c_api.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace dali_tf_impl {

struct CDeleter {
  void operator()(void* p) {
    free(p);
  }
};

template <typename T>
using AutoCPtr = std::unique_ptr<T, CDeleter>;

static tensorflow::TensorShape DaliToShape(const AutoCPtr<int64_t>& ns) {
  tensorflow::TensorShape ts;
  for (int i = 0; ns.get()[i] != 0; ++i)
    ts.InsertDim(i, ns.get()[i]);
  return ts;
}

constexpr tensorflow::DataType DaliToTfType(dali_data_type_t dali_type) {
  switch (dali_type) {
    case DALI_UINT8:
      return tensorflow::DT_UINT8;
    case DALI_UINT16:
      return tensorflow::DT_UINT16;
    case DALI_UINT32:
      return tensorflow::DT_UINT32;
    case DALI_UINT64:
      return tensorflow::DT_UINT64;
    case DALI_INT8:
      return tensorflow::DT_INT8;
    case DALI_INT16:
      return tensorflow::DT_INT16;
    case DALI_INT32:
      return tensorflow::DT_INT32;
    case DALI_INT64:
      return tensorflow::DT_INT64;
    case DALI_FLOAT16:
      return tensorflow::DT_HALF;
    case DALI_FLOAT:
      return tensorflow::DT_FLOAT;
    case DALI_FLOAT64:
      return tensorflow::DT_DOUBLE;
    case DALI_BOOL:
      return tensorflow::DT_BOOL;
    default:
      return tensorflow::DT_INVALID;
  }
}

constexpr dali_data_type_t TfToDaliType(tensorflow::DataType tf_type) {
  switch (tf_type) {
    case tensorflow::DT_UINT8:
      return DALI_UINT8;
    case tensorflow::DT_UINT16:
      return DALI_UINT16;
    case tensorflow::DT_UINT32:
      return DALI_UINT32;
    case tensorflow::DT_UINT64:
      return DALI_UINT64;
    case tensorflow::DT_INT8:
      return DALI_INT8;
    case tensorflow::DT_INT16:
      return DALI_INT16;
    case tensorflow::DT_INT32:
      return DALI_INT32;
    case tensorflow::DT_INT64:
      return DALI_INT64;
    case tensorflow::DT_HALF:
      return DALI_FLOAT16;
    case tensorflow::DT_FLOAT:
      return DALI_FLOAT;
    case tensorflow::DT_DOUBLE:
      return DALI_FLOAT64;
    case tensorflow::DT_BOOL:
      return DALI_BOOL;
    default:
      return DALI_NO_TYPE;
  }
}


}  // namespace dali_tf_impl


#endif  // DALI_TF_PLUGIN_DALI_SHAPE_HELPER_H_
