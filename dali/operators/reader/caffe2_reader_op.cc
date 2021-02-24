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

namespace {

int Caffe2ReaderOutputFn(const OpSpec &spec) {
  int img_idx = spec.GetArgument<bool>("image_available") ? 1 : 0;
  auto label_type = static_cast<LabelType>(spec.GetArgument<int>("label_type"));

  int num_label_outputs = (label_type == NO_LABEL) ? 0 : 1;
  num_label_outputs +=
      (label_type == MULTI_LABEL_SPARSE || label_type == MULTI_LABEL_WEIGHTED_SPARSE) ? 1 : 0;
  int additional_inputs = spec.GetArgument<int>("additional_inputs");
  int has_bbox = static_cast<int>(spec.GetArgument<bool>("bbox"));
  return img_idx + num_label_outputs + additional_inputs + has_bbox;
}

}  // namespace

DALI_REGISTER_OPERATOR(readers__Caffe2, Caffe2Reader, CPU);

DALI_SCHEMA(readers__Caffe2)
  .DocStr("Reads sample data from a Caffe2 Lightning Memory-Mapped Database (LMDB).")
  .NumInput(0)
  .OutputFn(Caffe2ReaderOutputFn)
  .AddArg("path",
      R"code(List of paths to the Caffe2 LMDB directories.)code",
      DALI_STRING_VEC)
  .AddOptionalArg("num_labels",
      R"code(Number of classes in the dataset.

Required when sparse labels are used.)code", 1)
  .AddOptionalArg("label_type",
      R"code(Type of label stored in dataset.

Here is a list of the available values:

* 0 = SINGLE_LABEL: which is the integer label for the multi-class classification.
* 1 = MULTI_LABEL_SPARSE: which is the sparse active label indices for multi-label classification.
* 2 = MULTI_LABEL_DENSE: which is the dense label embedding vector for label embedding regression.
* 3 = MULTI_LABEL_WEIGHTED_SPARSE: which is the sparse active label indices with per-label weights for multi-label classification.
* 4 = NO_LABEL: where no label is available.
)code", 0)
  .AddOptionalArg("image_available",
      R"code(Determines whether an image is available in this LMDB.)code", true)
  .AddOptionalArg("additional_inputs",
      R"code(Additional auxiliary data tensors that are provided for each sample.)code", 0)
  .AddOptionalArg("bbox",
      R"code(Denotes whether the bounding-box information is present.)code", false)
  .AddParent("LoaderBase");

// Deprecated alias
DALI_REGISTER_OPERATOR(Caffe2Reader, Caffe2Reader, CPU);

DALI_SCHEMA(Caffe2Reader)
    .DocStr("Legacy alias for :meth:`readers.caffe2`.")
    .NumInput(0)
    .OutputFn(Caffe2ReaderOutputFn)
    .AddParent("readers__Caffe2")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__Caffe2",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali
