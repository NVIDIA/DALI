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
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_UTIL_FITS_H_
#define DALI_UTIL_FITS_H_

#include <fitsio.h>
#include <string>
#include "dali/core/common.h"
#include "dali/core/static_switch.h"
#include "dali/core/stream.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"

#define NUMPY_ALLOWED_TYPES                                                                        \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
   double)

namespace dali {
namespace fits {

enum HDUType {
  IMAGE_HDU = 0,
  ASCII_TBL = 1,
  BINARY_TBL = 2,
  ANY_HDU = -1,
}

class DLL_PUBLIC HeaderData {
 public:
  TensorShape<> shape;
  HDUType hdu_type;
  const TypeInfo *type_info = nullptr;
  bool compressed = false;
  int64_t data_offset = 0;

  DALIDataType type() const;

  size_t size() const;

  size_t nbytes() const;
};

DLL_PUBLIC void ParseHeader(HeaderData &parsed_header, fitsfile *src);

}  // namespace fits
}  // namespace dali

#endif  // DALI_UTIL_FITS_H_
