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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop

class DaliDatasetOp : public tensorflow::DatasetOpKernel {
public:
  DaliDatasetOp(tensorflow::OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::DatasetBase** output) override {}
};

REGISTER_OP("DALIDataset")
    .Attr("serialized_pipeline: string")
    .Attr("shapes: list(shape) >= 1")
    .Attr("num_threads: int = -1")
    .Attr("prefetch_queue_depth: int = 2")
    .Attr("dtypes: list({float, int32, int64, half}) >= 1")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
DALI Dataset plugin

Creates a Dali dataset compatible with tf.data.Dataset from a serialized DALI pipeline
(given in `serialized_pipeline` parameter).

`shapes` must match the shape of the coresponding DALI Pipeline output tensor shape.
`dtypes` must match the type of the coresponding DALI Pipeline output tensors type.
 )doc");

REGISTER_KERNEL_BUILDER(Name("DALIDataset").Device(tensorflow::DEVICE_GPU),
                        DaliDatasetOp)
