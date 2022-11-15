// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// limitations under the License.

#include <string>

#include "dali/core/tensor_shape.h"
#include "dali/imgcodec/image_source.h"
#include "dali/imgcodec/image_decoder.h"

#ifndef DALI_OPERATORS_IMGCODEC_OPERATOR_UTILS_H_
#define DALI_OPERATORS_IMGCODEC_OPERATOR_UTILS_H_

namespace dali {
namespace imgcodec {

inline ImageSource SampleAsImageSource(const ConstSampleView<CPUBackend>& encoded,
                                       const std::string &source_info) {
  return ImageSource::FromHostMem(encoded.raw_data(), volume(encoded.shape()), source_info);
}

}  // namespace imgcodec
}  // namespace dali
#endif  // DALI_OPERATORS_IMGCODEC_OPERATOR_UTILS_H_
