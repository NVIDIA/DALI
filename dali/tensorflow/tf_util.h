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

#ifndef DALI_TF_UTIL_H
#define DALI_TF_UTIL_H

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

inline TensorShape DaliToShape(int64_t* ns) {
  TensorShape ts;
  for (int i = 0; ns[i] != 0; ++i) ts.InsertDim(i, ns[i]);
  delete[] ns;
  return ts;
}

const auto dali_shape_fn =
    [](tensorflow::shape_inference::InferenceContext* c) {
      std::vector<tensorflow::PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (unsigned int i = 0; i < shapes.size(); ++i) {
        if (shapes[i].dims() > 0) {
          tensorflow::shape_inference::ShapeHandle passed_shape;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromPartialTensorShape(shapes[i], &passed_shape));
          TF_RETURN_IF_ERROR(
              c->WithRank(passed_shape, shapes[i].dims(), &passed_shape));
          c->set_output(i, passed_shape);
        }
      }
      return tensorflow::Status::OK();
    };

}  // namespace tensorflow

#endif  // DALI_TF_UTIL_H
