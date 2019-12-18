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

DALI_REGISTER_OPERATOR(CaffeReader, CaffeReader, CPU);

DALI_SCHEMA(CaffeReader)
  .DocStr("Read (Image, label) pairs from a Caffe LMDB.")
  .NumInput(0)
  .OutputFn([](const OpSpec& spec) {
    auto image_available = spec.GetArgument<bool>("image_available");
    auto label_available = spec.GetArgument<bool>("label_available");
    return image_available + label_available;
  })
  .AddArg("path",
      R"code(List of paths to Caffe LMDB directories.)code",
      DALI_STRING_VEC)
  .AddOptionalArg("image_available",
      R"code(If image is available at all in this LMDB.)code", true)
  .AddOptionalArg("label_available",
      R"code(If label is available at all.)code", true)
  .AddParent("LoaderBase");

}  // namespace dali
