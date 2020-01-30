// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorflow/core/framework/tensor_shape.h"

// namespace dali-tf-impl {

struct CDeleter {
  void operator()(void *p) {
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

// }  // namespace dali-tf-impl
