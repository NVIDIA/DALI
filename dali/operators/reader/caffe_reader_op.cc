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

#include "dali/operators/reader/caffe_reader_op.h"

namespace dali {

namespace {

int CaffeReaderOutputFn(const OpSpec &spec) {
  auto image_available = spec.GetArgument<bool>("image_available");
  auto label_available = spec.GetArgument<bool>("label_available");
  return image_available + label_available;
}

}  // namespace

DALI_REGISTER_OPERATOR(readers__Caffe, CaffeReader, CPU);

DALI_SCHEMA(readers__Caffe)
  .DocStr("Reads (Image, label) pairs from a Caffe LMDB.")
  .NumInput(0)
  .OutputFn(CaffeReaderOutputFn)
  .AddArg("path",
      R"code(List of paths to the Caffe LMDB directories.)code",
      DALI_STRING_VEC)
  .AddOptionalArg("image_available",
      R"code(Determines whether an image is available in this LMDB.)code", true)
  .AddOptionalArg("label_available",
      R"code(Determines whether a label is available.)code", true)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(CaffeReader, CaffeReader, CPU);

DALI_SCHEMA(CaffeReader)
    .DocStr("Legacy alias for :meth:`readers.caffe`.")
    .NumInput(0)
    .OutputFn(CaffeReaderOutputFn)
    .AddParent("readers__Caffe")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__Caffe",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali
