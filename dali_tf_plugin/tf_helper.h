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

#ifndef DALI_TENSORFLOW_TF_HELPER_H_
#define DALI_TENSORFLOW_TF_HELPER_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"



#define _CTX_ERROR(error)                                                      \
    context->SetStatus(error);                                                 \
    return;

#define _RET_ERROR(error)                                                      \
    return error;

#define _DALI_CALL_IMPL(FUNC, ERROR_EXP)                                           \
    do {                                                                           \
      try {                                                                        \
        FUNC;                                                                      \
      } catch (std::runtime_error& e) {                                            \
        std::string error = "DALI " + std::string(#FUNC)                           \
                            + " failed: " + std::string(e.what());                 \
        std::cout << error << std::endl;                                           \
        ERROR_EXP(tensorflow::errors::Internal(error))                             \
      }                                                                            \
    } while (0)


struct CDeleter {
  void operator()(void *p) {
    free(p);
  }
};

template <typename T>
using AutoCPtr = std::unique_ptr<T, CDeleter>;

inline tensorflow::TensorShape DaliToShape(const AutoCPtr<int64_t>& ns) {
  tensorflow::TensorShape ts;
  for (int i = 0; ns.get()[i] != 0; ++i)
    ts.InsertDim(i, ns.get()[i]);
  return ts;
}

#endif  // DALI_TENSORFLOW_TF_HELPER_H_