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


#include "dali/operators/reader/caffe2_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(Caffe2Reader, Caffe2Reader, CPU);

DALI_SCHEMA(Caffe2Reader)
  .DocStr("Read sample data from a Caffe2 Lightning Memory-Mapped Database (LMDB).")
  .NumInput(0)
  .OutputFn([](const OpSpec& spec) {
      int img_idx = spec.GetArgument<bool>("image_available") ? 1 : 0;
      auto label_type = static_cast<LabelType>(spec.GetArgument<int>("label_type"));

      int num_label_outputs = (label_type == NO_LABEL) ? 0 : 1;
      num_label_outputs += (label_type == MULTI_LABEL_SPARSE ||
                            label_type == MULTI_LABEL_WEIGHTED_SPARSE) ? 1 : 0;
      int additional_inputs = spec.GetArgument<int>("additional_inputs");
      int has_bbox = static_cast<int>(spec.GetArgument<bool>("bbox"));
    return img_idx + num_label_outputs + additional_inputs + has_bbox;
  })
  .AddArg("path",
      R"code(List of paths to Caffe2 LMDB directories.)code",
      DALI_STRING_VEC)
  .AddOptionalArg("num_labels",
      R"code(Number of classes in dataset. Required when sparse labels are used.)code", 1)
  .AddOptionalArg("label_type",
      R"code(Type of label stored in dataset.

* 0 = SINGLE_LABEL : single integer label for multi-class classification
* 1 = MULTI_LABEL_SPARSE : sparse active label indices for multi-label classification
* 2 = MULTI_LABEL_DENSE : dense label embedding vector for label embedding regression
* 3 = MULTI_LABEL_WEIGHTED_SPARSE : sparse active label indices with per-label weights for multi-label classification.
* 4 = NO_LABEL : no label is available.
)code", 0)
  .AddOptionalArg("image_available",
      R"code(If image is available at all in this LMDB.)code", true)
  .AddOptionalArg("additional_inputs",
      R"code(Additional auxiliary data tensors provided for each sample.)code", 0)
  .AddOptionalArg("bbox",
      R"code(Denotes if bounding-box information is present.)code", false)
  .AddParent("LoaderBase");

}  // namespace dali
