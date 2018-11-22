// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_TYPED_LAYOUT_H_
#define DALI_PIPELINE_UTIL_TYPED_LAYOUT_H_

#include "dali/pipeline/util/typed_layout.h"
#include "dali/pipeline/basic/coords.h"
#include "dali/pipeline/data/types.h"

namespace dali {

DALIDataType layoutToTypeId(DALITensorLayout layout) {
  if (layout == DALITensorLayout::DALI_NHWC) {
    return TypeInfo::Create<dali_index_sequence<0, 1, 2>>().id();
  } else if (layout == DALITensorLayout::DALI_NCHW) {
    return TypeInfo::Create<dali_index_sequence<2, 0, 1>>().id();
  }
  return DALIDataType::DALI_NO_TYPE;
}

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_TYPED_LAYOUT_H_